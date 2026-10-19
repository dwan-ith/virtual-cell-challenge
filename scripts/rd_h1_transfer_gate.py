"""H1-holdout transfer gate: the only honest offline signal we have.

Why this exists: v40/v41/v42 all moved the EMITTER and all failed to beat v38
(0.09233). The synthetic gate was misleading because it built "truth" from the
same control pool and the same effect vector it predicted, so the answer was
trivially recoverable. This gate instead holds out a REAL context (H1, the
Arc 2025 Flex release) and predicts it from the other seven stores, so transfer
error is real.

Reports official-style PDS ranking plus cosine/nmae proxies, and tests the
v43 candidates:
  * min_cells abstention (drop sources with too few cells behind a target)
  * coverage weighting (a source that saw a target once should not vote as
    hard as one that saw it 300 times)
  * source chemistry weights (H1/CD4-like contexts may transfer better)
  * amplification (sources measured partial knockdown)

Nothing here is a leaderboard prediction; it is a directional screen. Gate
requires a candidate to beat the baseline on BOTH PDS and cosine, since past
submissions moved one and lost the other.
"""
import sys
sys.path.insert(0, ".")
import itertools
import numpy as np

from vcc_model.prior import LFCTransferPrior
from vcc_model.protein import build_per_gene_gain_map, load_esm_embeddings
from vcc_model.dataset import load_gene_names_csv

HOLDOUT = "vcc2025/H1"

pairs = {}
basals = {}
pair_n = {}
genes = None
from pathlib import Path
for p in sorted(Path("artifacts/stores").glob("*.npz")):
    if "__controls" in p.name:
        continue
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z["gene_names"].astype(str))
    for c, g, v, n in zip(z["context"], z["target"], z["lfc"], z["n_cells"]):
        pairs[(str(c), str(g))] = np.asarray(v, dtype=np.float32)
        pair_n[(str(c), str(g))] = int(n)
    for i, c in enumerate(z["basal_ctx"]):
        basals[str(c)] = z["basal"][i]

train = {k: v for k, v in pairs.items() if k[0] != HOLDOUT}
test = {k: v for k, v in pairs.items() if k[0] == HOLDOUT}
nested = {}
for (c, g), v in train.items():
    nested.setdefault(c, {})[g] = v
train_n = {k: n for k, n in pair_n.items() if k[0] != HOLDOUT}

truth = {g: v for (_, g), v in test.items()}
print(f"H1 holdout: {len(truth)} targets, {len(nested)} train contexts", flush=True)

esm = load_esm_embeddings("data/protein/se_full_18533.npz")


def panel_pds(preds):
    names = sorted(set(preds) & set(truth))
    n = len(names)
    panel = {names.index(g) if False else None}
    idx = {g: i for i, g in enumerate(genes)}
    mask = np.ones(len(genes), dtype=bool)
    for g in names:
        if g in idx:
            mask[idx[g]] = False
    keep = np.flatnonzero(mask)
    R = np.stack([truth[g][keep] for g in names]).astype(np.float64)
    P = np.stack([preds[g][keep] for g in names]).astype(np.float64)
    nr = np.linalg.norm(R, axis=1, keepdims=True); nr[nr == 0] = 1
    Ru = R / nr
    ph = np.linalg.norm(P, axis=1, keepdims=True)
    zero = ph[:, 0] == 0
    ph[ph == 0] = 1
    d = 1.0 - (P / ph) @ Ru.T
    ds = d[np.arange(n), np.arange(n)]
    k = (d < ds[:, None]).sum(1) + 0.5 * ((d == ds[:, None]).sum(1) - 1)
    sc = np.full(n, 0.5)
    sc[~zero] = 1.0 - k[~zero] / (n - 1)
    return float(sc.mean())


def cos(preds):
    names = sorted(set(preds) & set(truth))
    vals = []
    for g in names:
        p, t = preds[g], truth[g]
        np_, nt = np.linalg.norm(p), np.linalg.norm(t)
        if np_ > 0 and nt > 0:
            vals.append(float(p @ t) / (np_ * nt))
    return float(np.mean(vals))


def nmse(preds):
    names = sorted(set(preds) & set(truth))
    vals = []
    for g in names:
        gate = np.abs(truth[g]) > 0.3
        if gate.sum() >= 10:
            den = float(np.abs(truth[g][gate]).sum())
            if den > 0:
                vals.append(float(np.abs(preds[g][gate] - truth[g][gate]).sum() / den))
    return float(np.mean(vals)) if vals else float("nan")


def run(tag, min_cells=0, coverage=False, src_w=None, gain=1.0, q_thr=0.25):
    prior = LFCTransferPrior(
        similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
        gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
        zero_unseen=True, default_conservation=0.45,
    ).fit(nested, basals, genes, pair_n=train_n, source_weights=src_w, min_cells=min_cells)
    if coverage:
        cov = {}
        for (c, g), n in train_n.items():
            cov.setdefault(c, {})[g] = n
        prior.coverage = cov
    pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
    from vcc_model.calibrate import DECalibration
    cal = DECalibration(0.0, gain, 1.0, q_thr, pg)
    tgt = basals[HOLDOUT]
    preds = {}
    for g in truth:
        raw = prior.predict_lfc(g, tgt)
        preds[g] = np.asarray(cal.transform(raw, gene=g), dtype=np.float64)
    pds, c, n_ = panel_pds(preds), cos(preds), nmse(preds)
    print(f"{tag:34s} pds={pds:.4f} cos={c:+.4f} nmse={n_:.3f}", flush=True)
    return {"pds": pds, "cos": c, "nmse": n_}


base = run("baseline(v42 cfg)")
cands = {
    "min_cells=50": dict(min_cells=50),
    "min_cells=200": dict(min_cells=200),
    "min_cells=500": dict(min_cells=500),
    "gain=1.4": dict(gain=1.4),
    "gain=2.2": dict(gain=2.2),
    "gain=1.4,min_cells=200": dict(gain=1.4, min_cells=200),
}
res = {}
for tag, kw in cands.items():
    res[tag] = run(tag, **kw)

print("\n--- gate ---", flush=True)
passed = [
    t for t, r in res.items()
    if r["pds"] > base["pds"] and r["cos"] > base["cos"]
]
print("candidates beating baseline on BOTH pds and cos:", passed or "none", flush=True)