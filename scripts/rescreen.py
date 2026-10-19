"""Re-screen DE-calibration candidates on the H1 holdout using cached preds.

Loads artifacts/loco_v8.preds.npz (RAW blended LFCs for held-out H1 targets)
plus the H1 control cells, then for each candidate transform:
  1. generates Poisson-thinned truth + prediction cells,
  2. scores all six members raw,
  3. projects through OFFICIAL anchors -> projected leaderboard overall.
No model refit -- a full screen runs in minutes, so many candidates can be
compared offline before spending a daily submission.

Usage:
  python scripts/rescreen.py --preds artifacts/loco_v8.preds.npz \
      --store artifacts/stores/vcc2025_h1.npz [--out artifacts/rescreen.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.calibrate import DECalibration
from vcc_model.countsim import poisson_thin
from vcc_model.dataset import PseudobulkStore

OFFICIAL_ANCHORS = {
    "pds_cosine": (0.500, 0.956),
    "expr_mse_unbiased_capped_norm": (0.989, 0.037),
    "de_wilcoxon_direction_fidelity_yield_raw": (0.514, 0.814),
    "de_wilcoxon_direction_reach_raw": (0.072, 0.968),
    "de_wilcoxon_sig_jaccard": (0.029, 0.400),
    "de_wilcoxon_lfc_nmae": (1.001, 0.400),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=Path("artifacts/loco_v8.preds.npz"))
    ap.add_argument("--store", type=Path, default=Path("artifacts/stores/vcc2025_h1.npz"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/rescreen.json"))
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.preds, allow_pickle=True)
    ctxs = sorted(set(map(str, z["context"])))
    assert len(ctxs) == 1, f"expected single holdout context, got {ctxs}"
    held = ctxs[0]
    pred_raw = {str(t): np.asarray(v, dtype=np.float64)
                for c, t, v in zip(z["context"], z["target"], z["pred"])}

    store = PseudobulkStore.load(args.store)
    ctrl_sparse = store.controls[held]
    rng0 = np.random.default_rng(args.seed)
    n_ctrl = min(ctrl_sparse.shape[0], 2200)
    sel = rng0.choice(ctrl_sparse.shape[0], size=n_ctrl, replace=False)
    ctrl = np.asarray(ctrl_sparse[sel].todense(), dtype=np.float64)

    # truth cells from the store's observed LFCs for the same targets
    truth_lfc = {}
    for i, (c, t) in enumerate(store.pairs):
        if c == held and str(t) in pred_raw:
            truth_lfc[str(t)] = store.lfc[i].astype(np.float64)
    keys = sorted(set(truth_lfc) & set(pred_raw))[: args.n_pairs]
    gene_names = [str(g) for g in store.gene_names]

    rng_t = np.random.default_rng(args.seed + 4242)
    pick_t = rng0.choice(n_ctrl, size=args.n_cells, replace=args.n_cells > n_ctrl)
    base_truth = ctrl[pick_t]
    truth_cells = {
        g: poisson_thin(base_truth, np.exp2(np.clip(truth_lfc[g], -6, 6)), rng_t)
        for g in keys
    }

    # truth DE tables once
    from vcc_model.metrics import (
        ContextData, attach_pred_table, build_real_pairs, direction_fidelity,
        direction_reach, expr_mse_unbiased, lfc_nmae, pds_cosine, sig_jaccard,
        baseline_anchor, replicate_anchor, scale_scores,
    )
    from vcc_model.countsim import normalize_cpm as _ncpm

    cd_real = ContextData(
        gene_names=np.asarray(gene_names, dtype=str),
        real_ctrl=ctrl, real_pert=truth_cells, pred_pert={},
    )
    rpairs, ctrl_used, gate = build_real_pairs(cd_real, None, args.seed)
    ctrl_cpm = _ncpm(ctrl_used, 1_000_000.0)

    candidates = []
    for gain in (1.8, 2.2, 2.6, 3.1, 3.7):
        for q in (0.0, 0.45, 0.7):
            candidates.append(DECalibration(0.0, gain, 1.0, q))
    # two power variants on the current best guess operating point
    candidates.append(DECalibration(0.0, 2.6, 0.75, 0.0))
    candidates.append(DECalibration(0.0, 3.1, 0.75, 0.45))

    results = []
    t0 = time.time()
    for ci, cal in enumerate(candidates):
        rng = np.random.default_rng(args.seed + 1000 * ci + 7)
        pick = rng.choice(n_ctrl, size=args.n_cells, replace=False)
        base_pred = ctrl[pick]
        pred_map = {}
        for g in keys:
            lfc = np.clip(cal.transform(pred_raw[g]), -6, 6)
            pred_map[g] = poisson_thin(base_pred, np.exp2(lfc), rng)
        for name, pair in rpairs.items():
            attach_pred_table(pair, pred_map[name], gate, ctrl_cpm)
        vals = {
            "fid": direction_fidelity(rpairs),
            "reach": direction_reach(rpairs),
            "jac": sig_jaccard(rpairs),
            "nmae": lfc_nmae(rpairs),
        }
        cd_eval = ContextData(
            gene_names=np.asarray(gene_names, dtype=str),
            real_ctrl=ctrl_used, real_pert=truth_cells, pred_pert=pred_map,
        )
        vals["pds"] = pds_cosine(cd_eval)
        vals["mse"] = expr_mse_unbiased(cd_eval)

        proj = {}
        short_of = {
            "pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse",
            "de_wilcoxon_direction_fidelity_yield_raw": "fid",
            "de_wilcoxon_direction_reach_raw": "reach",
            "de_wilcoxon_sig_jaccard": "jac",
            "de_wilcoxon_lfc_nmae": "nmae"}
        for m, (b, r) in OFFICIAL_ANCHORS.items():
            u = vals.get(short_of[m])
            if not isinstance(u, (int, float)) or not np.isfinite(u):
                continue
            if m == "expr_mse_unbiased_capped_norm":
                u = min(max(u, 0.0), 1.0)
            proj[short_of[m]] = (u - b) / (r - b)
        finite = [v for v in proj.values()]
        overall = float(np.mean(finite)) if len(finite) >= 5 else float("nan")
        results.append({"calib": cal.as_dict(), "proj_overall": overall,
                        "proj": {k: round(v, 4) for k, v in proj.items()},
                        "raw": {k: (round(v, 4) if np.isfinite(v) else None) for k, v in vals.items()}})
        print("[%4d/%d] gain=%.2f q=%.2f p=%.2f -> proj %+.4f %s" % (
            ci + 1, len(candidates), cal.gain, cal.q_threshold, cal.power,
            overall, {k: round(v, 3) for k, v in proj.items()}), flush=True)

    results.sort(key=lambda r: -(r["proj_overall"] if np.isfinite(r["proj_overall"]) else -9))
    args.out.write_text(json.dumps(results[:20], indent=2))
    print("\nTOP 5:")
    for r in results[:5]:
        print("  proj %+.4f  %s" % (r["proj_overall"], r["calib"]))
    print("screen done in %.0fs -> %s" % (time.time() - t0, args.out))


if __name__ == "__main__":
    main()
