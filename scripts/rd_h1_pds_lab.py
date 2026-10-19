"""Offline PDS lab on the H1 context -- the honest lever selector.

Emitting cells and scoring against a self-made truth misled us twice (v42). This
lab avoids emission entirely: it works in the metric's own profile space.

For a context with control basal mean-CPM `b` (from the H1 store) and a per-gene
LFC vector `e`, the group-mean CPM is `b * 2**e`, and the scorer's log profile is
    prof(e) = log1p(TS * (b*2**e) / sum(b*2**e)),  TS = 5e4.
PDS (perturbation discrimination) = for each held-out target i, rank every
target j by how close prof(pred_j) is to prof(true_i); score = 1 when j=i ranks
first. This isolates transfer quality from the emitter.

Reported per variant: PDS (panel genes excluded, like the official metric),
cosine on control-expressed genes, and cosine on the top-50 true genes. Oracle
row must be 1.000 or the code is wrong.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path

from vcc_model.prior import LFCTransferPrior
from vcc_model.protein import build_per_gene_gain_map, load_esm_embeddings
from vcc_model.dataset import load_gene_names_csv

HOLDOUT = "vcc2025/H1"
PANEL = "data/vcc2026/pert_counts.csv"
TS = 5e4

genes = load_gene_names_csv("data/vcc2026/gene_names.csv")
G = len(genes)
gidx = {g: i for i, g in enumerate(genes)}


def load_panel():
    try:
        import pandas as pd
        df = pd.read_csv(PANEL)
        col = "target_gene" if "target_gene" in df.columns else df.columns[0]
        return {str(x) for x in df[col].dropna()}
    except Exception:
        return set()


nested, basals, truth, pair_n = {}, {}, {}, {}
for p in sorted(Path("artifacts/stores").glob("*.npz")):
    z = np.load(p, allow_pickle=True)
    for c, g, v, n in zip(z["context"], z["target"], z["lfc"], z["n_cells"]):
        c, g = str(c), str(g)
        if c == HOLDOUT:
            truth[g] = np.asarray(v, dtype=np.float64)
        else:
            nested.setdefault(c, {})[g] = np.asarray(v, dtype=np.float64)
            pair_n[(c, g)] = int(n)
    for i, c in enumerate(z["basal_ctx"]):
        basals.setdefault(str(c), z["basal"][i].astype(np.float64))

tg = [g for g in sorted(truth) if g in gidx]
n = len(tg)
basal = basals[HOLDOUT]
cpm = np.exp(basal) - 1e-9
expressed = cpm >= 5.0

panel = load_panel()
keep_pds = np.ones(G, bool)
for t in panel:
    if t in gidx:
        keep_pds[gidx[t]] = False
print(f"H1 holdout targets usable: {n} | panel-excluded genes: {int((~keep_pds).sum())} "
      f"(panel size {len(panel)}) | expressed genes: {int(expressed.sum())}", flush=True)

esm = load_esm_embeddings("data/protein/se_full_18533.npz")


def profiles(lfc_mat, mask):
    """lfc_mat (n, G) -> row-normalized log profiles on `mask` genes."""
    cpm_m = np.clip(basal, -30, 30)[None, :] * np.exp2(np.clip(lfc_mat, -8, 8))
    cpm_m = np.clip(cpm_m, 1e-9, None)
    tot = cpm_m.sum(axis=1, keepdims=True)
    prof = np.log1p(TS * cpm_m / tot)
    p = prof[:, mask]
    nrm = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.maximum(nrm, 1e-12)


def evaluate(name, lfc_mat, return_mat=False):
    P = profiles(lfc_mat, keep_pds)
    T = profiles(np.stack([truth[g] for g in tg]), keep_pds)
    d = 1.0 - P @ T.T                      # d[i,j] = dist(pred_j, true_i)
    order = np.argsort(d, axis=1)
    rank = np.empty(n)
    for i in range(n):
        rank[i] = int(np.where(order[i] == i)[0][0]) + 1
    pds = float(np.mean((n - rank) / (n - 1)))
    # gated + top-50 cosine
    Ph = profiles(lfc_mat, expressed)
    Th = profiles(np.stack([truth[g] for g in tg]), expressed)
    gc = float(np.mean(np.sum(Ph * Th, axis=1)))
    top = np.zeros((n, 50))
    tk = np.zeros((n, 50))
    for i, g in enumerate(tg):
        idx = np.argsort(-np.abs(truth[g]))[:50]
        pv, tv = lfc_mat[i][idx], truth[g][idx]
        top[i], tk[i] = pv, tv
    tc = float(np.mean([float(top[i] @ tk[i]) / (np.linalg.norm(top[i]) * np.linalg.norm(tk[i]) + 1e-12) for i in range(n)]))
    print(f"{name:40s} PDS={pds:.4f} cosExpr={gc:+.4f} cosTop50={tc:+.4f}", flush=True)
    return pds, (lfc_mat if return_mat else None)


prior = LFCTransferPrior(
    similarity_temperature=0.1, basal_gate_cpm=0.0, use_alpha=False,
    gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
    zero_unseen=True, default_conservation=0.45,
).fit(nested, basals, genes, pair_n=pair_n)


def predict_all(p):
    return np.stack([np.asarray(p.predict_lfc(g, basal), dtype=np.float64) for g in tg])


def soft_shrink(v, tau):
    s = np.median(np.abs(v)) + 1e-9
    return np.sign(v) * np.maximum(np.abs(v) - tau * s, 0.0)


print("\n== transfer-stage variants (no emission) ==", flush=True)
evaluate("ORACLE (truth)", np.stack([truth[g] for g in tg]))

cons_backup = dict(prior.consistency)
prior.consistency = {}
raw = predict_all(prior)
evaluate("raw prior (no cons shrink)", raw, return_mat=True)
prior.consistency = cons_backup
consh = predict_all(prior)
evaluate("prior + cons shrink", consh)

for tau in (0.25, 0.5, 1.0, 2.0):
    evaluate(f"cons + soft-shrink tau={tau}",
             np.stack([soft_shrink(consh[i], tau) for i in range(n)]))

pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
gain = np.array([float(pg.get(g, 1.0)) for g in genes])
evaluate("cons x per-gene gain", consh * gain[None, :])

print("\n== amplification sweep (Sidechain lever 'a') ==", flush=True)
for a in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
    evaluate(f"raw x amplify {a}", raw * a)
for a in (1.5, 2.0, 3.0, 4.0, 6.0):
    evaluate(f"cons x amplify {a}", consh * a)
for a in (2.0, 3.0, 4.0, 6.0):
    evaluate(f"cons x gain x amplify {a}", consh * gain[None, :] * a)