"""Diagnose why PDS < 0.50 (worse than random)."""

import numpy as np
from pathlib import Path
from vcc_model.dataset import PseudobulkStore
from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.countsim import log_mean_cpm, normalize_cpm
from vcc_model.ensemble import pds_proxy_fast, _panel_effect_matrix
from scipy import sparse

STORE_DIR = Path("artifacts/stores")
GENE_NAMES_FILE = Path("data/vcc2026/gene_names.csv")

def load_gene_names():
    import pandas as pd
    df = pd.read_csv(GENE_NAMES_FILE, header=None, dtype=str)
    col = df.iloc[:, 0].astype(str).str.strip()
    first = col.iloc[0].lower()
    if first in {"gene", "gene_name", "gene_names", "gene_symbol", "x", "index", "symbol"}:
        col = col.iloc[1:]
    return [g for g in col if g and g.lower() != "nan"]

def cosine_sim_matrix(mat):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    return unit @ unit.T

def main():
    print("=" * 70)
    print("PDS DIAGNOSIS: Why are predictions worse than random (PDS < 0.50)?")
    print("=" * 70)

    # 1. Load data
    gene_names = load_gene_names()
    print(f"\nGene axis: {len(gene_names)} genes")

    # Load VCC2025/H1 as target context
    h1_path = STORE_DIR / "vcc2025_h1.npz"
    h1_store = PseudobulkStore.load(h1_path)
    print(f"H1 store: {len(h1_store.pairs)} pairs, contexts={h1_store.contexts()}")
    print(f"H1 basal keys: {list(h1_store.basal.keys())}")

    # Get target genes in H1
    h1_targets = [t for _, t in h1_store.pairs if _ == "vcc2025/H1"]
    print(f"H1 targets: {len(h1_targets)} (expected 300)")

    # Load source stores
    source_stores = {}
    for f in sorted(STORE_DIR.glob("*.npz")):
        if "controls" in f.stem or "calibration" in f.stem or "vcc2025_h1" in f.stem:
            continue
        try:
            s = PseudobulkStore.load(f)
            source_stores[f.stem] = s
            print(f"  Loaded {f.stem}: {len(s.pairs)} pairs, contexts={s.contexts()}")
        except Exception as e:
            print(f"  SKIP {f.stem}: {e}")

    # 2. Build training data for the prior (nested dict format: {ctx: {gene: lfc}})
    nested = {}
    all_basals = {}
    for name, store in source_stores.items():
        for (ctx, gene), lfc in store.lfc_pairs().items():
            nested.setdefault(ctx, {})[gene] = lfc
        all_basals.update(store.basal)

    # Add H1 as a source (it IS in the training data)
    for (ctx, gene), lfc in h1_store.lfc_pairs().items():
        nested.setdefault(ctx, {})[gene] = lfc
    all_basals.update(h1_store.basal)

    total_pairs = sum(len(v) for v in nested.values())
    print(f"\nTotal pairs for prior: {total_pairs}")
    print(f"Total basals: {len(all_basals)}")

    # 3. Fit the prior
    prior = LFCTransferPrior()
    prior.fit(nested, all_basals, gene_names)
    print(f"Prior fitted: {len(prior.contexts)} contexts, global_alpha={prior.global_alpha:.4f}")

    # 4. Get H1 basal and predict for all 300 targets
    h1_basal = h1_store.basal["vcc2025/H1"]
    pred_map = prior.predict_many(h1_targets, h1_basal)

    # Also get the true LFC vectors for H1
    true_map = {}
    for p, lfc in h1_store.lfc_pairs().items():
        if p[0] == "vcc2025/H1":
            true_map[p[1]] = lfc

    pred_mat = np.stack([pred_map[g] for g in h1_targets])
    true_mat = np.stack([true_map[g] for g in h1_targets])

    print(f"\nPredicted LFC matrix shape: {pred_mat.shape}")
    print(f"True LFC matrix shape: {true_mat.shape}")

    # ================================================================
    # ANALYSIS A: Cosine similarity between predicted LFC vectors
    # ================================================================
    print("\n" + "=" * 70)
    print("A) PREDICTED INTER-TARGET COSINE SIMILARITY")
    print("=" * 70)
    pred_cos = cosine_sim_matrix(pred_mat)
    # Get upper triangle (exclude diagonal)
    idx = np.triu_indices(len(h1_targets), k=1)
    pred_cos_vals = pred_cos[idx]
    print(f"  Mean cosine between predicted LFCs: {pred_cos_vals.mean():.4f}")
    print(f"  Median: {np.median(pred_cos_vals):.4f}")
    print(f"  Std: {pred_cos_vals.std():.4f}")
    print(f"  Min: {pred_cos_vals.min():.4f}, Max: {pred_cos_vals.max():.4f}")
    print(f"  Fraction with cos > 0.9: {(pred_cos_vals > 0.9).mean():.3f}")
    print(f"  Fraction with cos > 0.5: {(pred_cos_vals > 0.5).mean():.3f}")
    print(f"  Fraction with cos < 0.0: {(pred_cos_vals < 0.0).mean():.3f}")

    # ================================================================
    # ANALYSIS B: Cosine similarity between TRUE LFC vectors
    # ================================================================
    print("\n" + "=" * 70)
    print("B) TRUE INTER-TARGET COSINE SIMILARITY")
    print("=" * 70)
    true_cos = cosine_sim_matrix(true_mat)
    true_cos_vals = true_cos[idx]
    print(f"  Mean cosine between true LFCs: {true_cos_vals.mean():.4f}")
    print(f"  Median: {np.median(true_cos_vals):.4f}")
    print(f"  Std: {true_cos_vals.std():.4f}")
    print(f"  Min: {true_cos_vals.min():.4f}, Max: {true_cos_vals.max():.4f}")
    print(f"  Fraction with cos > 0.9: {(true_cos_vals > 0.9).mean():.3f}")
    print(f"  Fraction with cos > 0.5: {(true_cos_vals > 0.5).mean():.3f}")
    print(f"  Fraction with cos < 0.0: {(true_cos_vals < 0.0).mean():.3f}")

    # ================================================================
    # ANALYSIS C: Cross (pred vs true) cosine similarity
    # ================================================================
    print("\n" + "=" * 70)
    print("C) CROSS COSINE SIMILARITY (pred_i vs true_i)")
    print("=" * 70)
    cross_cos = []
    for i, g in enumerate(h1_targets):
        p = pred_mat[i]
        t = true_mat[i]
        pn = np.linalg.norm(p)
        tn = np.linalg.norm(t)
        if pn > 1e-10 and tn > 1e-10:
            cross_cos.append(float(p @ t / (pn * tn)))
    cross_cos = np.array(cross_cos)
    print(f"  Mean pred-vs-true cosine: {cross_cos.mean():.4f}")
    print(f"  Median: {np.median(cross_cos):.4f}")
    print(f"  Fraction with cos > 0.3: {(cross_cos > 0.3).mean():.3f}")
    print(f"  Fraction with cos > 0.0: {(cross_cos > 0.0).mean():.3f}")
    print(f"  Fraction with cos < 0.0: {(cross_cos < 0.0).mean():.3f}")

    # ================================================================
    # ANALYSIS D: Magnitude comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("D) LFC MAGNITUDE COMPARISON")
    print("=" * 70)
    pred_norms = np.linalg.norm(pred_mat, axis=1)
    true_norms = np.linalg.norm(true_mat, axis=1)
    print(f"  Predicted L2 norms:  mean={pred_norms.mean():.4f}, median={np.median(pred_norms):.4f}, std={pred_norms.std():.4f}")
    print(f"  True L2 norms:       mean={true_norms.mean():.4f}, median={np.median(true_norms):.4f}, std={true_norms.std():.4f}")
    print(f"  Ratio (pred/true):   {pred_norms.mean() / max(true_norms.mean(), 1e-10):.4f}")

    pred_abs = np.abs(pred_mat)
    true_abs = np.abs(true_mat)
    print(f"\n  Per-gene |LFC|:")
    print(f"    Predicted: mean={pred_abs.mean():.6f}, median={np.median(pred_abs):.6f}, max={pred_abs.max():.6f}")
    print(f"    True:      mean={true_abs.mean():.6f}, median={np.median(true_abs):.6f}, max={true_abs.max():.6f}")

    # Fraction of near-zero predictions
    print(f"\n  Fraction of predicted |LFC| < 0.01: {(pred_abs < 0.01).mean():.4f}")
    print(f"  Fraction of true |LFC| < 0.01:      {(true_abs < 0.01).mean():.4f}")
    print(f"  Fraction of predicted |LFC| < 0.001: {(pred_abs < 0.001).mean():.4f}")
    print(f"  Fraction of true |LFC| < 0.001:      {(true_abs < 0.001).mean():.4f}")

    # ================================================================
    # ANALYSIS E: Compute PDS proxy
    # ================================================================
    print("\n" + "=" * 70)
    print("E) PDS PROXY ON PREDICTED vs TRUE")
    print("=" * 70)
    pds_val = pds_proxy_fast(pred_map, true_map, gene_names)
    print(f"  PDS proxy: {pds_val:.4f}")

    # Also compute PDS on TRUE vs TRUE (should be ~1.0 or at least > 0.5)
    pds_true = pds_proxy_fast(true_map, true_map, gene_names)
    print(f"  PDS on true vs true (sanity): {pds_true:.4f}")

    # ================================================================
    # ANALYSIS F: Consistency shrinkage analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("F) CONSISTENCY SHRINKAGE ANALYSIS")
    print("=" * 70)
    cons_vals = []
    for g in h1_targets:
        c = prior.consistency.get(g)
        if c is not None:
            cons_vals.append(c)
    cons_vals = np.array(cons_vals)
    print(f"  Consistency vector shape: {cons_vals.shape}")
    print(f"  Mean consistency: {cons_vals.mean():.4f}")
    print(f"  Median consistency: {np.median(cons_vals):.4f}")
    print(f"  Min: {cons_vals.min():.4f}, Max: {cons_vals.max():.4f}")
    print(f"  Fraction with cons < 0.5: {(cons_vals < 0.5).mean():.4f}")
    print(f"  Fraction with cons < 0.3: {(cons_vals < 0.3).mean():.4f}")

    # The shrinkage factor is (0.25 + 0.75 * cons)
    shrink_factors = 0.25 + 0.75 * cons_vals
    print(f"\n  Shrinkage factor (0.25 + 0.75*cons):")
    print(f"    Mean: {shrink_factors.mean():.4f}")
    print(f"    Median: {np.median(shrink_factors):.4f}")
    print(f"    Min: {shrink_factors.min():.4f}, Max: {shrink_factors.max():.4f}")

    # ================================================================
    # ANALYSIS G: shared_mix analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("G) SHARED PROGRAM BLEND ANALYSIS")
    print("=" * 70)
    print(f"  shared_mix value: {prior.shared_mix}")
    shared = prior._shared_program_from(prior.similarity_weights(h1_basal))
    shared_norm = np.linalg.norm(shared)
    print(f"  Shared program L2 norm: {shared_norm:.4f}")

    # Compute the shared program's contribution to each prediction
    shared_contributions = []
    for g in h1_targets:
        p = pred_map[g]
        # The shared component is shared_mix * shared_program
        # But it's blended as: (1 - shared_mix) * specific + shared_mix * shared
        # So the shared contribution is shared_mix * shared / (norm of pred)
        p_norm = np.linalg.norm(p)
        if p_norm > 1e-10:
            # Project prediction onto shared direction
            cos_with_shared = float(p @ shared / (p_norm * shared_norm)) if shared_norm > 1e-10 else 0.0
            shared_contributions.append(cos_with_shared)
    shared_contributions = np.array(shared_contributions)
    print(f"  Mean cosine of predictions with shared program: {shared_contributions.mean():.4f}")
    print(f"  Median: {np.median(shared_contributions):.4f}")
    print(f"  Fraction with cos > 0.9: {(shared_contributions > 0.9).mean():.3f}")
    print(f"  Fraction with cos > 0.5: {(shared_contributions > 0.5).mean():.3f}")

    # ================================================================
    # ANALYSIS H: Panel effect matrix (PDS space)
    # ================================================================
    print("\n" + "=" * 70)
    print("H) PANEL EFFECT MATRIX (PDS SPACE - no target genes)")
    print("=" * 70)
    pred_panel = _panel_effect_matrix(pred_map, h1_targets)
    true_panel = _panel_effect_matrix(true_map, h1_targets)
    print(f"  Pred panel shape: {pred_panel.shape}")
    print(f"  True panel shape: {true_panel.shape}")

    pred_panel_cos = cosine_sim_matrix(pred_panel)
    pred_panel_cos_vals = pred_panel_cos[idx]
    print(f"  Pred inter-target cosine in PDS space: mean={pred_panel_cos_vals.mean():.4f}, median={np.median(pred_panel_cos_vals):.4f}")

    true_panel_cos = cosine_sim_matrix(true_panel)
    true_panel_cos_vals = true_panel_cos[idx]
    print(f"  True inter-target cosine in PDS space: mean={true_panel_cos_vals.mean():.4f}, median={np.median(true_panel_cos_vals):.4f}")

    # ================================================================
    # ANALYSIS I: The CORE problem - why PDS < 0.5
    # ================================================================
    print("\n" + "=" * 70)
    print("I) ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # For PDS to be < 0.5, on average each target's prediction must be CLOSER
    # to other targets' truths than to its own truth.
    # This means the predictions are all similar to each other (collapse).

    # Compute distance matrix in PDS space
    pred_dists = 1.0 - pred_panel_cos  # pred_i vs pred_j
    true_dists = 1.0 - true_panel_cos  # true_i vs true_j

    # For each target, compute: distance to own truth vs distance to other truths
    pred_panel_unit = pred_panel / (np.linalg.norm(pred_panel, axis=1, keepdims=True) + 1e-10)
    true_panel_unit = true_panel / (np.linalg.norm(true_panel, axis=1, keepdims=True) + 1e-10)
    cross_dist = 1.0 - pred_panel_unit @ true_panel_unit.T  # pred_i vs true_j

    n = len(h1_targets)
    d_self = cross_dist[np.arange(n), np.arange(n)]
    d_other_min = np.full(n, np.inf)
    d_other_mean = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        d_other_min[i] = cross_dist[i, mask].min()
        d_other_mean[i] = cross_dist[i, mask].mean()

    print(f"  Distance pred_i -> true_i (self):  mean={d_self.mean():.4f}, median={np.median(d_self):.4f}")
    print(f"  Distance pred_i -> true_j (min):   mean={d_other_min.mean():.4f}, median={np.median(d_other_min):.4f}")
    print(f"  Distance pred_i -> true_j (mean):  mean={d_other_mean.mean():.4f}, median={np.median(d_other_mean):.4f}")

    # Fraction where self-distance < min other distance (correct ranking)
    correct_rank = (d_self < d_other_min).mean()
    print(f"\n  Fraction where self-distance < min other: {correct_rank:.4f}")
    print(f"  (PDS = correct_rank + 0.5 * ties; PDS < 0.5 means MORE wrong than right)")

    # ================================================================
    # ANALYSIS J: What does the raw prior produce before calibration?
    # ================================================================
    print("\n" + "=" * 70)
    print("J) RAW PRIOR (no calibration) vs CALIBRATED")
    print("=" * 70)

    # The calibration from fit_info
    cal = DECalibration()  # default identity
    # Check what calibration was used
    if hasattr(prior, 'shared_mix'):
        print(f"  Prior shared_mix: {prior.shared_mix}")
    print(f"  Prior use_alpha: {prior.use_alpha}")
    print(f"  Prior global_alpha: {prior.global_alpha:.4f}")

    # Check alpha per context
    w_map = prior.similarity_weights(h1_basal)
    alpha = prior.calibrated_alpha(h1_basal, w=w_map)
    print(f"  Alpha for H1: {alpha:.4f}")
    print(f"  Similarity weights: { {k: f'{v:.4f}' for k, v in w_map.items()} }")

    # ================================================================
    # ANALYSIS K: Predict with different shared_mix values
    # ================================================================
    print("\n" + "=" * 70)
    print("K) EFFECT OF shared_mix ON PREDICTION DIVERSITY")
    print("=" * 70)
    orig_mix = prior.shared_mix
    for sm in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        prior.shared_mix = sm
        pm = prior.predict_many(h1_targets, h1_basal)
        mat = np.stack([pm[g] for g in h1_targets])
        cos = cosine_sim_matrix(mat)
        cos_vals = cos[idx]
        norms = np.linalg.norm(mat, axis=1)
        pds = pds_proxy_fast(pm, true_map, gene_names)
        print(f"  shared_mix={sm:.2f}: mean_cos={cos_vals.mean():.4f}, "
              f"mean_norm={norms.mean():.4f}, PDS={pds:.4f}")
    prior.shared_mix = orig_mix

    # ================================================================
    # ANALYSIS L: Consistency shrinkage breakdown
    # ================================================================
    print("\n" + "=" * 70)
    print("L) WHAT HAPPENS WITHOUT CONSISTENCY SHRINKAGE?")
    print("=" * 70)
    # Manually predict without consistency shrinkage
    w_map = prior.similarity_weights(h1_basal)
    alpha = prior.calibrated_alpha(h1_basal, w=w_map)
    raw_noshrink = {}
    for g in h1_targets:
        stack, weights = [], []
        for c, w in w_map.items():
            if g in prior.lfc.get(c, {}):
                stack.append(prior.lfc[c][g])
                weights.append(w)
        if stack:
            w_arr = np.array(weights)
            w_arr = w_arr / w_arr.sum()
            specific = np.average(np.stack(stack), axis=0, weights=w_arr)
            # NO shrinkage: just specific + shared
            shared = prior._shared_program_from(w_map)
            raw = (1.0 - prior.shared_mix) * specific + prior.shared_mix * shared
            raw_noshrink[g] = (alpha * raw).astype(np.float64)
        else:
            # fallback
            raw_noshrink[g] = pm[g]  # use whatever the prior gave

    mat_ns = np.stack([raw_noshrink[g] for g in h1_targets])
    cos_ns = cosine_sim_matrix(mat_ns)
    cos_ns_vals = cos_ns[idx]
    norms_ns = np.linalg.norm(mat_ns, axis=1)
    pds_ns = pds_proxy_fast(raw_noshrink, true_map, gene_names)
    print(f"  No shrinkage: mean_cos={cos_ns_vals.mean():.4f}, "
          f"mean_norm={norms_ns.mean():.4f}, PDS={pds_ns:.4f}")

    # With shrinkage (current)
    pds_curr = pds_proxy_fast(pred_map, true_map, gene_names)
    print(f"  Current:      mean_cos={pred_cos_vals.mean():.4f}, "
          f"mean_norm={pred_norms.mean():.4f}, PDS={pds_curr:.4f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    print(f"""
1. INTER-TARGET SIMILARITY:
   - Predicted LFCs have mean cosine = {pred_cos_vals.mean():.4f}
   - True LFCs have mean cosine = {true_cos_vals.mean():.4f}
   - {'PROBLEM: Predictions are MORE similar to each other than truth!' if pred_cos_vals.mean() > true_cos_vals.mean() + 0.1 else 'Predictions and truth have similar diversity'}

2. MAGNITUDES:
   - Predicted mean L2 norm = {pred_norms.mean():.4f}
   - True mean L2 norm = {true_norms.mean():.4f}
   - Ratio = {pred_norms.mean() / max(true_norms.mean(), 1e-10):.4f}

3. PDS:
   - PDS proxy = {pds_val:.4f}
   - PDS < 0.5 means predictions are on average CLOSER to OTHER targets' truths

4. ROOT CAUSE:
   - All predictions project heavily onto shared program (mean cos = {shared_contributions.mean():.4f})
   - Consistency shrinkage is aggressive: mean factor = {shrink_factors.mean():.4f}
   - The prior's shared_mix = {prior.shared_mix} adds a common direction to all 300 vectors
   - Result: 300 predicted LFCs are a narrow fan pointing ~same direction,
     while the 300 true LFCs span diverse directions in gene space
""")

if __name__ == "__main__":
    main()
