"""v11 development: hierarchical precision-weighted prior with basal gating.

Tests three improvements over v6's simple mean pooling:
1. Precision weighting: weight source contexts by 1/var (inverse variance) per gene
   plus basal similarity (so noisy source genes contribute less)
2. Empirical Bayes shrinkage: global James-Stein shrinkage per program toward 0,
   factor = max(0, 1 - (G-2)*sigma2/||pooled||^2)
3. Basal gating: zero predicted LFC for genes with low basal expression in
   target context (mean CPM < threshold), since those genes cannot respond

All evaluated via LOCO PDS proxy on H1 fold (the honest held-out).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import PseudobulkStore
from vcc_model.ensemble import pds_proxy_fast

# Load stores to get per-context LFC variance
stores = {}
for name in ["replogle_gw_k562", "replogle_ess_k562", "replogle_gw_rpe1",
             "tahoe_hepg2", "tahoe_jurkat", "tahoe_k562", "tahoe_rpe1", "vcc2025_h1"]:
    s = PseudobulkStore.load(f"artifacts/stores/{name}.npz")
    c = sorted(s.basal)[0]
    stores[c] = s

gene_names = [str(g) for g in s.gene_names]
held = "vcc2025/H1"
train_ctxs = [c for c in stores if c != held]

# Build per-gene variance across sources (for precision weighting)
print("Building per-gene variance estimates...")
# For each target gene g, collect its program vectors across contexts where observed
from collections import defaultdict
gene_programs = defaultdict(list)
for c in train_ctxs:
    s = stores[c]
    for i, (ctx, t) in enumerate(s.pairs):
        gene_programs[t].append(s.lfc[i])

# Test on H1 fold: predict each H1 target using train contexts
held_store = stores[held]
held_targets = [t for _, t in held_store.pairs]
print(f"Held-out {held}: {len(held_targets)} targets")

# Basal for held context
held_basal = held_store.basal[held]
# Compute control mean CPM for gating (from basal which is log mean CPM)
# basal = log(mean CPM + 1e-9) -> mean CPM = exp(basal) - 1e-9
held_mean_cpm = np.exp(held_basal) - 1e-9

# Baseline: simple mean pooling (v6 behavior)
from vcc_model.prior import LFCTransferPrior
import copy

def evaluate_prior(prior, label):
    preds = {}
    truths = {}
    for t in held_targets:
        preds[t] = prior.predict_lfc(t, held_basal)
        # Find truth LFC for this target in held store
        for i, (c, tt) in enumerate(held_store.pairs):
            if tt == t:
                truths[t] = held_store.lfc[i].astype(np.float64)
                break
    pds = pds_proxy_fast(preds, truths, gene_names)
    print(f"{label}: pds_proxy = {pds:.4f}")
    return pds

# Baseline v6 prior
train_lfc = {c: {t: stores[c].lfc[i] for i, (_, t) in enumerate(stores[c].pairs) if c in train_ctxs} for c in train_ctxs}
train_basal = {c: stores[c].basal[c] for c in train_ctxs}
prior_v6 = LFCTransferPrior().fit(train_lfc, train_basal, gene_names)
prior_v6.shared_mix = 0.0
prior_v6.use_alpha = False
evaluate_prior(prior_v6, "v6 baseline (mean pooling)")

# v11a: precision-weighted pooling
# For each held target, compute weighted mean where weight = similarity * precision
# Precision per gene per source = 1 / (1 + var_gene_in_source_context)
# Approximate var via per-context LFC variance across genes? Simpler: use n_cells as proxy
# n_cells per (context,target) is stored; higher n_cells -> lower variance
print("\nv11a: precision-weighted pooling")
# Build n_cells lookup
n_cells_lookup = {}
for c in train_ctxs:
    s = stores[c]
    for (ctx, t), n in s.n_cells.items():
        n_cells_lookup[(ctx, t)] = n

# For v11a, we modify the prior to use precision weighting
# Quick test: for each target, compute weighted mean with w = similarity * sqrt(n_cells)
prior_v11a = copy.deepcopy(prior_v6)
# We'll test by manually computing predictions with modified weights
# For now, just test the concept on a subset
print("Testing precision weighting concept...")
# For held targets, compare simple mean vs precision-weighted mean
for t in held_targets[:3]:
    # Collect source programs for t
    progs = []
    weights_simple = []
    weights_prec = []
    for c in train_ctxs:
        if t in train_lfc[c]:
            progs.append(train_lfc[c][t])
            # Simple weight = 1
            weights_simple.append(1.0)
            # Precision weight = sqrt(n_cells)
            n = n_cells_lookup.get((c, t), 50)
            weights_prec.append(np.sqrt(n))
    if len(progs) >= 2:
        progs = np.stack(progs)
        simple = progs.mean(axis=0)
        prec = np.average(progs, axis=0, weights=weights_prec)
        # Compare cosine to truth
        truth = None
        for i, (c, tt) in enumerate(held_store.pairs):
            if tt == t:
                truth = held_store.lfc[i].astype(np.float64)
                break
        cos_simple = np.dot(simple, truth) / (np.linalg.norm(simple)*np.linalg.norm(truth) + 1e-9)
        cos_prec = np.dot(prec, truth) / (np.linalg.norm(prec)*np.linalg.norm(truth) + 1e-9)
        print(f"  {t}: cos_simple={cos_simple:.3f} cos_prec={cos_prec:.3f} (n_progs={len(progs)})")

# v11b: basal gating
print("\nv11b: basal gating test")
# Count how many predicted LFC genes would be gated (low basal)
low_genes = np.where(held_mean_cpm < 5.0)[0]
print(f"Genes with mean CPM <5 in held context: {len(low_genes)}/{len(gene_names)} ({len(low_genes)/len(gene_names)*100:.1f}%)")
# For a sample prediction, see what fraction of top predicted genes are in low basal set
sample_pred = prior_v6.predict_lfc(held_targets[0], held_basal)
top_genes = np.argsort(np.abs(sample_pred))[-100:]
n_top_low = sum(1 for g in top_genes if g in set(low_genes))
print(f"Top 100 predicted genes that are low-basal: {n_top_low} (these would be zeroed by gating)")

# v11c: James-Stein shrinkage
print("\nv11c: James-Stein shrinkage test")
G = len(gene_names)
for t in held_targets[:3]:
    progs = [train_lfc[c][t] for c in train_ctxs if t in train_lfc[c]]
    if len(progs) < 2:
        continue
    pooled = np.mean(np.stack(progs), axis=0)
    # Estimate sigma2 as mean variance across sources per gene, averaged
    mat = np.stack(progs)
    # Per-gene variance across sources
    per_gene_var = mat.var(axis=0).mean()
    norm_sq = np.dot(pooled, pooled)
    # James-Stein factor
    factor = max(0, 1 - (G - 2) * per_gene_var / (norm_sq + 1e-9))
    shrunk = pooled * factor
    print(f"  {t}: ||pooled||={np.linalg.norm(pooled):.1f} sigma2={per_gene_var:.4f} factor={factor:.3f}")

print("\nDone - see above for v11 lever estimates")
