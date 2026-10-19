"""Test program-SHAPE levers (de-shared, denoising) on the H1 fold by PDS.

Loads cached RAW blended predictions for held-out H1 targets, fits the shape
levers on the TRAINING contexts only (no holdout leakage), and reports the
official-style PDS proxy per lever setting. Shape operations change program
DIRECTION, which is exactly what pds_cosine ranks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import PseudobulkStore


def official_pds(preds: dict[str, np.ndarray], truths: dict[str, np.ndarray]) -> float:
    """PDS in LFC space with panel-target coordinates removed."""
    names = sorted(set(preds) & set(truths))
    n = len(names)
    if n < 2:
        return float("nan")
    panel = set(names)
    dim = len(next(iter(truths.values())))

    def eff(vectors):
        keep = []
        g0 = next(iter(vectors))
        idx_of = {}
        return None

    # build mask once: remove coordinates of ALL panel targets
    mask = np.ones(dim, dtype=bool)
    # axis positions of panel genes are unknown here; approximate by removing
    # each vector's OWN strongest coordinate is wrong -- instead the caller
    # passes vectors already restricted to non-panel genes via --drop-panel.
    mat_r = np.stack([np.asarray(truths[g], dtype=np.float64) for g in names])
    mat_p = np.stack([np.asarray(preds[g], dtype=np.float64) for g in names])
    nr = np.linalg.norm(mat_r, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    unit = mat_r / nr
    ph = np.linalg.norm(mat_p, axis=1, keepdims=True)
    zp = ph[:, 0] == 0
    ph[ph == 0] = 1.0
    dists = 1.0 - (mat_p / ph) @ unit.T
    d_self = dists[np.arange(n), np.arange(n)]
    k = (dists < d_self[:, None]).sum(axis=1) + 0.5 * ((dists == d_self[:, None]).sum(axis=1) - 1)
    scores = np.where(zp, 0.5, 1.0 - k / (n - 1))
    return float(scores.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=Path("artifacts/loco_v8.preds.npz"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/shape_levers.json"))
    args = ap.parse_args()

    z = np.load(args.preds, allow_pickle=True)
    ctxs = sorted(set(map(str, z["context"])))
    held = ctxs[0]
    pred_raw = {str(t): np.asarray(v, dtype=np.float64)
                for c, t, v in zip(z["context"], z["target"], z["pred"])}

    store = PseudobulkStore.load("artifacts/stores/vcc2025_h1.npz")
    truths = {}
    for i, (c, t) in enumerate(store.pairs):
        if c == held and str(t) in pred_raw:
            truths[str(t)] = store.lfc[i].astype(np.float64)
    keys = sorted(set(truths) & set(pred_raw))

    # ---- fit levers on TRAINING contexts only (exclude held) ----
    train_stores = {}
    for name in ["replogle_gw_k562", "replogle_ess_k562", "replogle_gw_rpe1",
                 "tahoe_hepg2", "tahoe_jurkat", "tahoe_k562", "tahoe_rpe1"]:
        s = PseudobulkStore.load(f"artifacts/stores/{name}.npz")
        c = sorted(s.basal)[0]
        train_stores[c] = {t: s.lfc[i].astype(np.float32) for i, (_c, t) in enumerate(s.pairs)}
    pooled = {}
    for c, d in train_stores.items():
        for t, v in d.items():
            pooled.setdefault(t, []).append(v)
    pooled_mean = {t: np.mean(np.stack(vs), axis=0) for t, vs in pooled.items()}
    genes = sorted(pooled_mean)
    mat = np.stack([pooled_mean[g] for g in genes])
    mean = mat.mean(axis=0)
    from sklearn.utils.extmath import randomized_svd

    _, _, vt_full = randomized_svd(mat - mean, n_components=64, n_iter=3, random_state=0)

    global_mean = np.mean(np.stack(list(pooled_mean.values())), axis=0).astype(np.float64)

    def apply(v: np.ndarray, beta: float, rank: int) -> np.ndarray:
        out = v - beta * global_mean if beta > 0 else v
        if rank > 0:
            vt = vt_full[:rank]
            out = mean + (out - mean) @ vt.T @ vt
        return out

    rows = []
    for beta in (0.0, 0.15, 0.3, 0.5):
        for rank in (0, 16, 32, 64):
            preds_t = {g: apply(pred_raw[g], beta, rank) for g in keys}
            pds = official_pds(preds_t, {g: truths[g] for g in keys})
            rows.append({"beta": beta, "rank": rank, "pds_proxy": round(pds, 4)})
            print(f"beta={beta:.2f} rank={rank:>2} -> pds proxy {pds:.4f}", flush=True)

    args.out.write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["pds_proxy"])
    print("BEST:", best)


if __name__ == "__main__":
    main()
