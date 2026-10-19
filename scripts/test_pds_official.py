"""Official-space PDS evaluator on the H1 fold (thinned truth).

Replicates the scored computation exactly: pseudobulk per group ->
log1p(BULK_TARGET_SUM * fraction) -> remove ALL panel-target coordinates ->
cosine-rank each predicted effect among all measured effects. Runs on
Poisson-thinned cells so count-level effects (normalization weighting) are
present, unlike the earlier LFC-space proxy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.countsim import poisson_thin
from vcc_model.dataset import PseudobulkStore


def log_profile(group_counts, ts=5e4):
    total = group_counts.sum()
    if total <= 0:
        return np.zeros(group_counts.shape[1])
    return np.log1p(ts * group_counts / total)


def official_pds_from_cells(pred_cells, truth_cells, ctrl_counts, gene_names, panel):
    ctrl_prof = log_profile(np.asarray(ctrl_counts).sum(axis=0))
    pos_of = {g: i for i, g in enumerate(gene_names)}
    mask = np.ones(len(gene_names), dtype=bool)
    for g in panel:
        gi = pos_of.get(g)
        if gi is not None:
            mask[gi] = False
    names = sorted(pred_cells)
    n = len(names)
    R = np.stack([(log_profile(np.asarray(truth_cells[g]).sum(axis=0)) - ctrl_prof)[mask] for g in names])
    P = np.stack([(log_profile(np.asarray(pred_cells[g]).sum(axis=0)) - ctrl_prof)[mask] for g in names])
    nr = np.linalg.norm(R, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    unit = R / nr
    ph = np.linalg.norm(P, axis=1, keepdims=True)
    zp = ph[:, 0] == 0
    ph[ph == 0] = 1.0
    dists = 1.0 - (P / ph) @ unit.T
    d_self = dists[np.arange(n), np.arange(n)]
    k = (dists < d_self[:, None]).sum(axis=1) + 0.5 * ((dists == d_self[:, None]).sum(axis=1) - 1)
    scores = np.where(zp, 0.5, 1.0 - k / max(n - 1, 1))
    return float(scores.mean()), scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=Path("artifacts/loco_v8.preds.npz"))
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pds_official.json"))
    args = ap.parse_args()

    z = np.load(args.preds, allow_pickle=True)
    held = sorted(set(map(str, z["context"])))[0]
    pred_raw = {str(t): np.asarray(v, dtype=np.float64)
                for c, t, v in zip(z["context"], z["target"], z["pred"])}

    store = PseudobulkStore.load("artifacts/stores/vcc2025_h1.npz")
    gene_names = [str(g) for g in store.gene_names]
    truths, pairs_idx = {}, {}
    for i, (c, t) in enumerate(store.pairs):
        if c == held and str(t) in pred_raw:
            truths[str(t)] = store.lfc[i].astype(np.float64)
            pairs_idx[str(t)] = i
    keys = sorted(truths)

    rng0 = np.random.default_rng(args.seed)
    ctrl_sparse = store.controls[held]
    n_ctrl = min(ctrl_sparse.shape[0], 2200)
    sel = rng0.choice(ctrl_sparse.shape[0], size=n_ctrl, replace=False)
    ctrl = np.asarray(ctrl_sparse[sel].todense(), dtype=np.float64)

    rng_t = np.random.default_rng(args.seed + 4242)
    pick = rng0.choice(n_ctrl, size=args.n_cells, replace=False)
    base_truth = ctrl[pick]
    truth_cells = {g: poisson_thin(base_truth, np.exp2(np.clip(truths[g], -6, 6)), rng_t)
                   for g in keys}
    rng_p = np.random.default_rng(args.seed + 777)
    pick_p = rng_p.choice(n_ctrl, size=args.n_cells, replace=False)
    base_pred = ctrl[pick_p]
    pred_cells = {g: poisson_thin(base_pred, np.exp2(np.clip(pred_raw[g], -6, 6)), rng_p)
                  for g in keys}

    pds_all, scores_all = official_pds_from_cells(
        pred_cells, truth_cells, ctrl, gene_names, panel=set(keys))

    # ablation: how does pds look restricted to different |truth lfc| strata?
    # (which genes carry the discrimination?)
    out = {"pds_official_thinned": pds_all, "n": len(keys)}
    print(f"official-space pds (thinned H1): {pds_all:.4f} over {len(keys)} targets")

    # where does ranking power live? correlation between self-cosine and
    # truth-program norm; plus per-decile purity
    order = np.argsort(scores_all)
    print("worst 10 targets:", [keys[i] for i in order[:10]])
    out["worst"] = [keys[i] for i in order[:10]]
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
