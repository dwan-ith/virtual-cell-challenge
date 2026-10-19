"""Where does the transferable signal actually live? Restrict to tested genes.

Finding that motivated this: measured on real H1 data, only ~2% of the truth
|LFC| mass lies above the 5 CPM gate, i.e. 98% of the norm of a truth vector is
numerical noise from near-zero mean CPM. An all-gene cosine is therefore
dominated by noise in BOTH truth and prediction, so cos ~= 0.003 across all
genes does not prove "no signal" -- it proves the all-gene cosine is the wrong
measurement.

The scored DE metrics test only genes above the gate, and pds compares
group-summed log profiles (which are dominated by expressed genes). So measure:
  * cosine restricted to control-expressed genes (>=5 CPM in holdout controls)
  * cosine on the top-K truth genes (signal-focused)
  * Spearman on gated genes (rank agreement, scale-free)

If gated cosine is meaningfully above zero, real transfer signal exists and the
lever is magnitude/gating, not the emitter.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import build_per_gene_gain_map, load_esm_embeddings
from vcc_model.dataset import load_gene_names_csv

HOLDOUT = "vcc2025/H1"
genes = load_gene_names_csv("data/vcc2026/gene_names.csv")

nested, basals, truth, pair_n = {}, {}, {}, {}
for p in sorted(Path("artifacts/stores").glob("*.npz")):
    if "__controls" in p.name:
        continue
    z = np.load(p, allow_pickle=True)
    for c, g, v, n in zip(z["context"], z["target"], z["lfc"], z["n_cells"]):
        c, g = str(c), str(g)
        if c == HOLDOUT:
            truth[g] = np.asarray(v, dtype=np.float64)
        else:
            nested.setdefault(c, {})[g] = np.asarray(v, dtype=np.float64)
            pair_n[(c, g)] = int(n)
    for i, c in enumerate(z["basal_ctx"]):
        basals[str(c)] = z["basal"][i].astype(np.float64)

tg = sorted(truth)
esm = load_esm_embeddings("data/protein/se_full_18533.npz")
tgt = basals[HOLDOUT]
cpm = np.exp(tgt) - 1e-9
expressed = cpm >= 5.0
expressed5 = cpm >= 1.0
expressed1 = cpm >= 0.1
print(f"H1 holdout {len(tg)} targets | genes: expr>=5CPM {int(expressed.sum())}, "
      f">=1 {int(expressed5.sum())}, >=0.1 {int(expressed1.sum())}", flush=True)

prior = LFCTransferPrior(
    similarity_temperature=0.1, basal_gate_cpm=0.0, use_alpha=False,
    gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
    zero_unseen=True, default_conservation=0.45,
).fit(nested, basals, genes, pair_n=pair_n)
raw = {g: np.asarray(prior.predict_lfc(g, tgt), dtype=np.float64) for g in tg}


def gated_cos(preds, mask, label):
    vals = []
    for g in tg:
        p, t = preds[g], truth[g]
        pm, tm = p[mask], t[mask]
        npn, tnn = np.linalg.norm(pm), np.linalg.norm(tm)
        if npn > 1e-9 and tnn > 1e-9:
            vals.append(float(pm @ tm) / (npn * tnn))
    print(f"{label:44s} n={len(vals):3d} cos={np.mean(vals):+.4f}", flush=True)
    return float(np.mean(vals))


def topk_cos(preds, k, label):
    vals = []
    for g in tg:
        t = truth[g]
        idx = np.argsort(-np.abs(t))[:k]
        p, t = preds[g][idx], t[idx]
        npn, tnn = np.linalg.norm(p), np.linalg.norm(t)
        if npn > 1e-9 and tnn > 1e-9:
            vals.append(float(p @ t) / (npn * tnn))
    print(f"{label:44s} n={len(vals):3d} cos={np.mean(vals):+.4f}", flush=True)
    return float(np.mean(vals))


def spear(preds, mask, label):
    vals = []
    for g in tg:
        p, t = preds[g][mask], truth[g][mask]
        if np.std(p) > 1e-9 and np.std(t) > 1e-9:
            r = spearmanr(p, t).statistic
            if np.isfinite(r):
                vals.append(float(r))
    print(f"{label:44s} n={len(vals):3d} rho={np.mean(vals):+.4f}", flush=True)
    return float(np.mean(vals))


pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
cal = DECalibration(0.0, 1.8, 1.0, 0.25, pg)
calout = {g: np.asarray(cal.transform(raw[g], gene=g), dtype=np.float64) for g in tg}

print("\n--- all genes (noise-dominated) ---", flush=True)
gated_cos(calout, np.ones(len(genes), bool), "calibrated, all genes")
print("\n--- restricted to control-expressed genes ---", flush=True)
for m, nm in [(expressed, ">=5 CPM"), (expressed5, ">=1 CPM"), (expressed1, ">=0.1 CPM")]:
    gated_cos(raw, m, f"raw prior, expr {nm}")
    gated_cos(calout, m, f"calibrated, expr {nm}")
    spear(calout, m, f"spearman, expr {nm}")
print("\n--- top truth genes ---", flush=True)
for k in (50, 200, 1000):
    topk_cos(calout, k, f"top-{k} truth genes")
print("\n--- oracle sanity (same measurement code) ---", flush=True)
gated_cos(truth, expressed, "ORACLE truth, expr >=5 CPM")
topk_cos(truth, 200, "ORACLE truth, top-200")