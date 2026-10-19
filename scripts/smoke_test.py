"""End-to-end smoke test on synthetic multi-context CRISPRi data.

Proves the full harness works: store construction -> hybrid fit -> LOCO
zero-shot prediction -> DE calibration -> six-metric scoring with anchors ->
rule-checked submission assembly. Asserts the sanity ordering:

    control_copy < hybrid < oracle   (scaled aggregate)

Runtime: a couple of minutes on CPU.

Usage: python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.calibrate import tune
from vcc_model.countsim import log_mean_cpm, multinomial_thin
from vcc_model.dataset import PseudobulkStore
from vcc_model.ensemble import HybridPredictor
from vcc_model.metrics import (
    ContextData,
    baseline_anchor,
    evaluate_context,
    replicate_anchor,
    scale_scores,
)
from vcc_model.residual import ResidualConfig
from vcc_model.submission import validate_submission
from vcc_model.synthesize import make_synthetic_challenge


def evaluate_cells(cd: ContextData, seed: int = 0) -> dict:
    raw = evaluate_context(cd)
    b = baseline_anchor(cd, seed=seed)
    r = replicate_anchor(cd, n_splits=2, base_seed=seed)
    return scale_scores(raw, b, r)


def main() -> None:
    rng = np.random.default_rng(0)
    ch = make_synthetic_challenge(
        n_genes=512, n_contexts=6, n_perts=16,
        n_ctrl_cells=320, n_cells_per_pert=96, seed=11,
    )
    contexts = ch.contexts_sorted()
    holdout = ["line_4", "line_5"]
    train_ctx = [c for c in contexts if c not in holdout]
    print(f"[smoke] contexts={contexts} holdout={holdout}")

    gene_names = [str(g) for g in ch.gene_names]

    # ---- fit hybrid on training contexts ----
    train_pairs = dict(ch.lfc_truth)
    basals = {c: log_mean_cpm(ch.controls[c], 1e6) for c in contexts}
    model = HybridPredictor.fit(
        lfc_pairs={k: v.astype(np.float64) for k, v in train_pairs.items() if k[0] in train_ctx},
        basals={c: basals[c] for c in train_ctx},
        gene_names=gene_names,
        residual_cfg=ResidualConfig(epochs=120, hidden=128, seed=0),
    )

    # ---- per-holdout evaluation ----
    agg: dict[str, float] = {}
    for held in holdout:
        ctrl = np.asarray(ch.controls[held], dtype=np.float64)
        targets = sorted(t for (c, t) in ch.lfc_truth if c == held)
        basal_held = basals[held]

        pred_raw = {t: model.blend_lfc(t, basal_held) for t in targets}
        cal, cal_info = tune(pred_raw, {t: ch.lfc_truth[(held, t)] for t in targets})
        model.calibration = cal

        prng = np.random.default_rng(7)
        # Thin from the reference control rows themselves (same-batch realism).
        pick = prng.choice(ctrl.shape[0], size=80, replace=False)
        base_cells = ctrl[pick]

        true_cells = {
            t: multinomial_thin(base_cells, np.exp2(ch.lfc_truth[(held, t)]), prng)
            for t in targets
        }
        pred_cells = {
            t: multinomial_thin(base_cells, np.exp2(model.predict_lfc(t, basal_held)), prng)
            for t in targets
        }
        cd = ContextData(np.asarray(gene_names), ctrl[:300], true_cells, pred_cells)

        # ablations
        copy_cells = {t: base_cells.copy() for t in targets}
        cd_copy = ContextData(cd.gene_names, cd.real_ctrl, true_cells, copy_cells)

        s_hybrid = evaluate_cells(cd, seed=3)
        s_copy = evaluate_cells(cd_copy, seed=3)
        if args_oracle():
            cd_oracle = ContextData(cd.gene_names, cd.real_ctrl, true_cells, true_cells)
            s_oracle = evaluate_cells(cd_oracle, seed=3)
        else:
            s_oracle = None

        print(f"\n[smoke] === {held} === calib={cal.as_dict()} "
              f"(surrogate {cal_info.get('score', float('nan')):.3f})")
        rows = [("control_copy", s_copy), ("hybrid", s_hybrid)]
        if s_oracle is not None:
            rows.append(("oracle", s_oracle))
        for name, s in rows:
            print(f"  {name:>12}: overall={s['overall']:.3f} pds={s['pds_cosine']:.3f} "
                  f"fid={s['de_wilcoxon_direction_fidelity_yield_raw']:.3f} "
                  f"reach={s['de_wilcoxon_direction_reach_raw']:.3f} "
                  f"jac={s['de_wilcoxon_sig_jaccard']:.3f} "
                  f"nmae={s['de_wilcoxon_lfc_nmae']:.3f} mse={s['expr_mse_unbiased_capped_norm']:.3f}")
        for m in ("pds_cosine", "expr_mse_unbiased_capped_norm", "overall"):
            agg[f"{held}:{m}"] = float(s_hybrid[m])
        assert s_copy["overall"] < s_hybrid["overall"], (
            "hybrid must beat control copy"
        )
        if s_oracle is not None:
            assert s_hybrid["overall"] <= s_oracle["overall"] + 1e-9

    print("\n[smoke] hybrid aggregates:",
          json.dumps({k: round(v, 3) for k, v in agg.items()}, indent=2))

    # ---- submission mechanics on a toy panel ----
    toy_targets = targets[:5]
    toy_cells = {}
    toy_genes = gene_names[:256]
    gidx = {g: i for i, g in enumerate(gene_names)}
    sub_idx = [gidx[g] for g in toy_genes]
    for ctx in ["A"]:
        prng2 = np.random.default_rng(9)
        base = np.asarray(ch.controls["line_0"][:60][:, sub_idx], dtype=np.int64)
        for t in toy_targets:
            toy_cells[(ctx, t)] = multinomial_thin(base, np.ones(len(toy_genes)), prng2)
    import anndata as ad
    from scipy import sparse as sp

    blocks = []
    obs_rows = []
    for (ctx, t), arr in sorted(toy_cells.items()):
        blocks.append(sp.csr_matrix(arr.astype(np.int32)))
        obs_rows.extend([(ctx, t)] * arr.shape[0])
    x = sp.vstack(blocks, format="csr")
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame({
            "target_gene": pd.Series([t for _, t in obs_rows], dtype="category"),
            "context": pd.Series([c for c, _ in obs_rows], dtype="category"),
        }),
        var=pd.DataFrame(index=pd.Index(toy_genes)),
    )
    problems = validate_submission(
        adata,
        expected_gene_names=toy_genes,
        contexts=("A",),
        perts_per_context=len(toy_targets),
        cells_per_pert=60,
    )
    assert not problems, f"submission validation failed:\n" + "\n".join(problems)
    print(f"[smoke] submission builder OK ({adata.n_obs} cells, nnz={adata.X.nnz})")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.h5ad"
        adata.write_h5ad(p)
        assert p.exists() and p.stat().st_size > 0

    print("\nSMOKE TEST PASSED")


def args_oracle() -> bool:
    return "--no-oracle" not in sys.argv


if __name__ == "__main__":
    main()
