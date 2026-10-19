"""Is our transfer actually producing signal? Diagnostic, not a submission gate.

The H1 transfer gate reported cos ~= 0.002 and NMAE exactly 1.000, which are the
fingerprints of an all-zero prediction rather than a weak one. This script
isolates where the signal dies:

  1. Is the raw prior output nonzero for held-out targets?
  2. Does the calibration (gain/threshold/per-gene gain) zero it out?
  3. Does the basal gate (5 CPM) wipe responsive genes in the target context?
  4. What cosine do we get before vs after each stage?

Uses real H1 measurements as holdout. No synthetic truth -- synthetic gates
already misled us once (see validate_v42_gates.py).
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import build_per_gene_gain_map, load_esm_embeddings
from vcc_model.dataset import load_gene_names_csv

HOLDOUT = "vcc2025/H1"
genes = load_gene_names_csv("data/vcc2026/gene_names.csv")
G = len(genes)

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
print(f"H1 holdout targets: {len(tg)}", flush=True)

esm = load_esm_embeddings("data/protein/se_full_18533.npz")


def cos_stats(name, preds):
    vals, nz, big = [], 0, 0
    for g in tg:
        p, t = preds.get(g), truth[g]
        if p is None:
            continue
        nrm = float(np.linalg.norm(p))
        if nrm > 1e-9:
            nz += 1
            tn = float(np.linalg.norm(t))
            if tn > 0:
                vals.append(float(p @ t) / (nrm * tn))
        if np.abs(p).max() > 0.25:
            big += 1
    return name, float(np.mean(vals)) if vals else float("nan"), nz, big


prior = LFCTransferPrior(
    similarity_temperature=0.1, basal_gate_cpm=0.0, use_alpha=False,
    gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
    zero_unseen=True, default_conservation=0.45,
).fit(nested, basals, genes, pair_n=pair_n)
tgt = basals[HOLDOUT]

raw = {g: np.asarray(prior.predict_lfc(g, tgt), dtype=np.float64) for g in tg}
print("%-34s %8s %5s %5s" % ("stage", "cos", "nz", "big"), flush=True)
print("%-34s %8s %5s %5s" % cos_stats("raw prior (no basal gate)", raw), flush=True)

# Where does truth mass live relative to the basal gate?
cpm = np.exp(tgt) - 1e-9
for thr in (0.0, 5.0):
    keep = cpm >= thr
    mass = np.mean([np.abs(truth[g][keep]).sum() / max(np.abs(truth[g]).sum(), 1e-9) for g in tg])
    print(f"truth |lfc| mass above {thr:>4.1f} CPM: {mass:.3f}  (genes kept {int(keep.sum())})", flush=True)

prior.basal_gate_cpm = 5.0
gated = {g: np.asarray(prior.predict_lfc(g, tgt), dtype=np.float64) for g in tg}
print("%-34s %8s %5s %5s" % cos_stats("after basal gate 5 CPM", gated), flush=True)

pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
for thr, gain, q in [(0.0, 1.0, 0.0), (0.0, 1.0, 0.25), (0.0, 1.8, 0.25), (0.0, 1.8, 0.0), (0.1, 1.8, 0.25)]:
    cal = DECalibration(thr, gain, 1.0, q, pg)
    out = {g: np.asarray(cal.transform(raw[g], gene=g), dtype=np.float64) for g in tg}
    print("%-34s %8s %5s %5s" % cos_stats(
        f"cal t={thr} g={gain} q={q}", out), flush=True)

# Honest upper bound: transfer using the H1 vector itself (sanity == should be 1.0)
print("%-34s %8s %5s %5s" % cos_stats("ORACLE (truth vs truth)", truth), flush=True)