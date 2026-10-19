"""Fast PDS diagnosis - skip slow LOCO calibration."""

import json
import numpy as np
import time
from pathlib import Path
from vcc_model.dataset import PseudobulkStore
from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.countsim import log_mean_cpm, normalize_cpm
from vcc_model.ensemble import pds_proxy_fast, _panel_effect_matrix

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
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms) @ (mat / norms).T

def main():
    t_start = time.time()
    gene_names = load_gene_names()
    h1_store = PseudobulkStore.load(STORE_DIR / "vcc2025_h1.npz")
    h1_targets = sorted(set(t for _, t in h1_store.pairs if _ == "vcc2025/H1"))
    h1_basal = h1_store.basal["vcc2025/H1"]

    true_map = {p[1]: h1_store.lfc[i].astype(np.float64)
                for i, p in enumerate(h1_store.pairs) if p[0] == "vcc2025/H1"}
    true_mat = np.stack([true_map[g] for g in h1_targets])
    idx = np.triu_indices(len(h1_targets), k=1)

    # Fit prior EXCLUDING H1 (simulate unseen context)
    nested = {}
    all_basals = {}
    for f in sorted(STORE_DIR.glob("*.npz")):
        if "controls" in f.stem or "calibration" in f.stem:
            continue
        s = PseudobulkStore.load(f)
        for (ctx, gene), lfc in s.lfc_pairs().items():
            if ctx == "vcc2025/H1":
                continue
            nested.setdefault(ctx, {})[gene] = lfc
        for c, b in s.basal.items():
            if c != "vcc2025/H1":
                all_basals[c] = b

    print(f"Data loaded in {time.time() - t_start:.0f}s", flush=True)
    print(f"Training: {sum(len(v) for v in nested.values())} pairs, {len(nested)} contexts", flush=True)

    # Fit WITHOUT LOCO scale calibration (saves massive time)
    t0 = time.time()
    prior = LFCTransferPrior()
    prior.fit(nested, all_basals, gene_names, calibrate_scale=False)
    prior.shared_mix = 0.0
    prior.use_alpha = False
    print(f"Prior fitted in {time.time() - t0:.0f}s", flush=True)

    # Analyze weights
    w_map = prior.similarity_weights(h1_basal)
    alpha = prior.calibrated_alpha(h1_basal, w=w_map)
    print(f"\nSimilarity weights for H1:")
    for c, w in sorted(w_map.items(), key=lambda x: -x[1]):
        print(f"  {c}: {w:.6f}")
    print(f"Alpha: {alpha:.6f}")

    # Predict
    t0 = time.time()
    pred_raw = prior.predict_many(h1_targets, h1_basal)
    print(f"Predictions in {time.time() - t0:.1f}s", flush=True)

    mat_raw = np.stack([pred_raw[g] for g in h1_targets])

    # === ANALYSIS ===
    cos_raw = cosine_sim_matrix(mat_raw)
    cos_true = cosine_sim_matrix(true_mat)
    pred_norms = np.linalg.norm(mat_raw, axis=1)
    true_norms = np.linalg.norm(true_mat, axis=1)

    # Cross cosine
    cross = np.array([
        float(mat_raw[i] @ true_mat[i] / (pred_norms[i] * true_norms[i]))
        for i in range(len(h1_targets))
        if pred_norms[i] > 1e-10 and true_norms[i] > 1e-10
    ])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Inter-target cosine:")
    print(f"    Predicted: {cos_raw[idx].mean():.4f}")
    print(f"    True:      {cos_true[idx].mean():.4f}")
    print(f"    Ratio:     {cos_raw[idx].mean() / cos_true[idx].mean():.2f}x")
    print(f"\n  L2 norms:")
    print(f"    Predicted: {pred_norms.mean():.2f}")
    print(f"    True:      {true_norms.mean():.2f}")
    print(f"    Ratio:     {pred_norms.mean() / true_norms.mean():.4f}x")
    print(f"\n  Cross cosine (pred_i vs true_i):")
    print(f"    Mean:      {cross.mean():.4f}")
    print(f"    > 0.3:     {(cross > 0.3).mean():.1%}")
    print(f"    > 0.0:     {(cross > 0.0).mean():.1%}")
    print(f"    < 0.0:     {(cross < 0.0).mean():.1%}")

    # Consistency
    cons = np.array([prior.consistency.get(g, np.array([0.0])).mean() for g in h1_targets])
    shrink = 0.25 + 0.75 * cons
    print(f"\n  Consistency: mean={cons.mean():.4f}")
    print(f"  Shrinkage:   mean={shrink.mean():.4f}")

    # PDS
    pds_raw = pds_proxy_fast(pred_raw, true_map, gene_names)
    print(f"\n  PDS (raw, unseen): {pds_raw:.4f}")

    # Gain sweep
    print("\n  Calibration sweep:")
    best_pds, best_q, best_g = 0.0, 0.0, 1.0
    for q in [0.0, 0.2, 0.35, 0.5, 0.7, 0.85]:
        for g in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            cal = DECalibration(threshold=0.0, gain=g, power=1.0, q_threshold=q)
            pc = {t: cal.transform(pred_raw[t]) for t in h1_targets}
            pds = pds_proxy_fast(pc, true_map, gene_names)
            if pds > best_pds:
                best_pds, best_q, best_g = pds, q, g
            if g in [1.0, 10.0, 50.0]:
                print(f"    q={q:.2f} gain={g:5.1f}: PDS={pds:.4f}")
    print(f"\n  BEST: q={best_q:.2f} gain={best_g:.1f} -> PDS={best_pds:.4f}")

    # Perfect direction/magnitude tests
    print("\n  Ablation tests:")
    # Perfect magnitudes, our directions
    pm = {}
    for i, g in enumerate(h1_targets):
        pm[g] = mat_raw[i] * (true_norms[i] / pred_norms[i]) if pred_norms[i] > 1e-10 else mat_raw[i]
    print(f"    Perfect magnitudes + our dirs: PDS={pds_proxy_fast(pm, true_map, gene_names):.4f}")

    # Perfect directions, our magnitudes
    pd2 = {}
    for i, g in enumerate(h1_targets):
        if pred_norms[i] > 1e-10:
            pd2[g] = (mat_raw[i] / pred_norms[i]) * true_norms[i]
        else:
            pd2[g] = true_mat[i]
    print(f"    Perfect directions + our mag:  PDS={pds_proxy_fast(pd2, true_map, gene_names):.4f}")

    # Our directions, truth magnitudes (direction determines ranking)
    pd3 = {}
    for i, g in enumerate(h1_targets):
        if pred_norms[i] > 1e-10:
            pd3[g] = (mat_raw[i] / pred_norms[i]) * true_norms[i]
        else:
            pd3[g] = true_mat[i]
    print(f"    Our dirs (unit) + truth mag:   PDS={pds_proxy_fast(pd3, true_map, gene_names):.4f}")

    # Truth dirs, our magnitudes
    pd4 = {}
    for i, g in enumerate(h1_targets):
        if true_norms[i] > 1e-10:
            pd4[g] = (true_mat[i] / true_norms[i]) * pred_norms[i]
        else:
            pd4[g] = mat_raw[i]
    print(f"    Truth dirs (unit) + our mag:   PDS={pds_proxy_fast(pd4, true_map, gene_names):.4f}")

    # V11 config
    cal_v11 = DECalibration(threshold=0.0, gain=2.1, power=1.0, q_threshold=0.35)
    pred_v11 = {g: cal_v11.transform(pred_raw[g]) for g in h1_targets}
    pds_v11 = pds_proxy_fast(pred_v11, true_map, gene_names)
    mat_v11 = np.stack([pred_v11[g] for g in h1_targets])
    print(f"\n  V11 config (gain=2.1, q=0.35):")
    print(f"    PDS:       {pds_v11:.4f}")
    print(f"    Pred norm: {np.linalg.norm(mat_v11, axis=1).mean():.2f}")
    zero_frac = np.mean([np.all(np.abs(pred_v11[g]) < 1e-10) for g in h1_targets])
    print(f"    Zero targets: {zero_frac:.1%}")

    print(f"\n  Total time: {time.time() - t_start:.0f}s")

    print("\n" + "=" * 70)
    print("ROOT CAUSE DIAGNOSIS")
    print("=" * 70)
    print(f"""
1. COSINE COLLAPSE (primary):
   Predicted inter-target cosine = {cos_raw[idx].mean():.4f}
   True inter-target cosine      = {cos_true[idx].mean():.4f}
   Predictions are {(cos_raw[idx].mean() / cos_true[idx].mean()):.1f}x more correlated than truth.
   The prior's transfer gives all targets ~same direction.

2. MAGNITUDE STARVATION (secondary):
   Predicted L2 norms = {pred_norms.mean():.2f}
   True L2 norms      = {true_norms.mean():.2f}
   Ratio = {pred_norms.mean() / true_norms.mean():.4f} ({pred_norms.mean() / true_norms.mean() * 100:.1f}% of truth)
   Consistency shrinkage = {shrink.mean():.2f} on average.

3. WHY PDS < 0.5:
   PDS computes, for each target i, the fraction of OTHER targets j where
   cos(pred_i, truth_i) > cos(pred_i, truth_j). PDS < 0.5 means MORE
   wrong rankings than right.

   With cosine collapse: all pred_i are similar -> they rank all truths
   roughly equally -> PDS ~ 0.5 (random). Worse if the shared direction
   happens to match some OTHER target better.

4. THE CALIBRATION CATCH-22:
   - Without gain amplification: magnitudes too small -> all zero after threshold
   - With gain amplification: noise is amplified equally -> no discrimination
   - q_threshold helps but can't fix the fundamental direction problem

5. FIXES NEEDED:
   a) Reduce cosine collapse: lower shared_mix, denoise, or project to
      low-rank subspace that preserves inter-target diversity
   b) Improve direction accuracy: the prior should weight source contexts
      more by gene-level correlation, not just basal similarity
   c) Consider: the 272 targets seen only in K562 GW get the SAME transfer
      from ONE source -> all get very similar predictions
""")

if __name__ == "__main__":
    main()
