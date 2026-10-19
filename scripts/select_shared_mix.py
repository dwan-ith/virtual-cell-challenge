"""Select the shared-program mixing fraction by honest LOCO on source contexts.

The full LOCO weight selection (7 folds x sub-residual refits) was run twice
end-to-end and deterministically chose blend weights (stat=1, resid=0, stack=0)
both times (identical per-fold best-so-far scores). Re-deriving that costs ~2.3h
per attempt in this environment; this script instead fixes that measured verdict
and scans only `shared_mix` -- which needs just cheap prior rebuilds per fold,
no residual nets:

  for each source context: refit LFCTransferPrior without it (no residual),
  predict every held-fold target under each candidate mix, score with
  (pair proxy + PDS rank proxy) exactly like ensemble._select_weights_loco.

Writes artifacts/shared_mix.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.ensemble import lfc_proxy_aggregate, pds_proxy_fast
from vcc_model.prior import LFCTransferPrior
from vcc_model.dataset import PseudobulkStore

STORES = [
    "replogle_gw_k562", "replogle_ess_k562", "replogle_gw_rpe1",
    "tahoe_hepg2", "tahoe_jurkat", "tahoe_k562", "tahoe_rpe1", "vcc2025_h1",
]
MIX_GRID = (0.0, 0.15, 0.25, 0.40)


def main() -> None:
    arrays: list[np.ndarray] = []
    merged: PseudobulkStore | None = None
    for name in STORES:
        s = PseudobulkStore.load(f"artifacts/stores/{name}.npz")
        if merged is None:
            merged = s
            continue
        assert np.array_equal(merged.gene_names, s.gene_names)
        merged.pairs = merged.pairs + s.pairs
        arrays.append(s.lfc)
        for c, b in s.basal.items():
            merged.basal.setdefault(c, b)
    if arrays:
        merged.lfc = np.vstack([merged.lfc] + arrays)  # one allocation, not 7
        del arrays
    gene_names = [str(g) for g in merged.gene_names]
    by_ctx: dict[str, list[int]] = {}
    for i, (c, _t) in enumerate(merged.pairs):
        by_ctx.setdefault(c, []).append(i)
    contexts = sorted(by_ctx)
    print(f"[mix] {len(contexts)} folds: {contexts}", flush=True)

    rng_cap = np.random.default_rng(0)
    scores: dict[float, list[float]] = {m: [] for m in MIX_GRID}

    for fi, held in enumerate(contexts):
        t0 = time.time()
        rows = [i for i in by_ctx[held] if merged.pairs[i][0] == held]
        # float32 views -- no 1.5 GB float64 duplicate of the GW fold
        truth = {merged.pairs[i][1]: merged.lfc[i] for i in rows}
        fold_genes = list(truth)
        proxy_genes = fold_genes
        if len(proxy_genes) > 400:
            proxy_genes = sorted(rng_cap.choice(proxy_genes, size=400, replace=False).tolist())
        pds_names = fold_genes
        if len(pds_names) > 120:
            pds_names = sorted(rng_cap.choice(pds_names, size=120, replace=False).tolist())

        train_lfc: dict[str, dict[str, np.ndarray]] = {}
        basals: dict[str, np.ndarray] = {}
        for c in contexts:
            if c == held:
                continue
            ids = by_ctx[c]
            train_lfc[c] = {merged.pairs[i][1]: merged.lfc[i] for i in ids}
            basals[c] = merged.basal[c]
        prior = LFCTransferPrior().fit(train_lfc, basals, gene_names)

        orig = prior.shared_mix
        # Stream per mix: never hold every mix's full prediction set at once
        # (all-mixes x fold-genes x G float64 peaked at ~6 GB on the GW fold).
        for m in MIX_GRID:
            prior.shared_mix = m
            vals = []
            for g in proxy_genes:
                v = prior.predict_lfc(g, merged.basal[held])
                vals.append(lfc_proxy_aggregate(v, truth[g]))
                del v
            pair_s = float(np.mean(vals))
            pv_pds = prior.predict_many(pds_names, merged.basal[held])
            pds_s = pds_proxy_fast(
                pv_pds,
                {g: truth[g] for g in pds_names},
                gene_names,
            )
            del pv_pds
            combined = 0.5 * pair_s + 0.5 * pds_s if np.isfinite(pds_s) else pair_s
            scores[m].append(float(combined))
        prior.shared_mix = orig
        del prior, train_lfc
        import gc

        gc.collect()

        detail = {str(m): round(float(np.mean(v)), 4) for m, v in scores.items() if v}
        print(f"[mix] fold {fi + 1}/{len(contexts)} ({held}) in {time.time() - t0:.0f}s "
              f"-> {detail}", flush=True)

    means = {m: float(np.mean(v)) for m, v in scores.items() if v}
    best_mix = max(means, key=means.get)
    out = {"chosen": best_mix, "means": means, "grid": [float(m) for m in MIX_GRID]}
    Path("artifacts/shared_mix.json").write_text(json.dumps(out, indent=2))
    print(f"[mix] chosen shared_mix={best_mix} :: {means}", flush=True)


if __name__ == "__main__":
    main()
