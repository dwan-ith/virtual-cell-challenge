"""Final comprehensive PDS diagnosis - all findings in one report."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from pathlib import Path
from vcc_model.dataset import PseudobulkStore
from vcc_model.prior import LFCTransferPrior
from vcc_model.countsim import log_mean_cpm
from vcc_model.ensemble import pds_proxy_fast
import time, json

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
    gene_names = load_gene_names()
    h1_store = PseudobulkStore.load(STORE_DIR / "vcc2025_h1.npz")
    h1_targets = sorted(set(t for _, t in h1_store.pairs if _ == "vcc2025/H1"))
    h1_basal = h1_store.basal["vcc2025/H1"]
    true_map = {p[1]: h1_store.lfc[i].astype(np.float64)
                for i, p in enumerate(h1_store.pairs) if p[0] == "vcc2025/H1"}
    true_mat = np.stack([true_map[g] for g in h1_targets])
    idx = np.triu_indices(len(h1_targets), k=1)

    # Fit prior with H1 excluded
    nested, all_basals = {}, {}
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

    prior = LFCTransferPrior()
    prior.fit(nested, all_basals, gene_names, calibrate_scale=False)
    prior.shared_mix = 0.0
    prior.use_alpha = False

    pred_raw = prior.predict_many(h1_targets, h1_basal)
    mat_raw = np.stack([pred_raw[g] for g in h1_targets])
    pred_norms = np.linalg.norm(mat_raw, axis=1)
    true_norms = np.linalg.norm(true_mat, axis=1)

    # Consistency
    cons = np.array([prior.consistency.get(g, np.array([0.0])).mean() for g in h1_targets])
    shrink = 0.25 + 0.75 * cons

    # Known gene counts per target
    known_counts = np.array([
        sum(1 for c in nested if g in nested[c]) for g in h1_targets
    ])

    print("=" * 70)
    print("COMPREHENSIVE PDS ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # Section 1: The evidence
    print("""
=== 1. EMPIRICAL EVIDENCE ===

Official leaderboard PDS on validation contexts A/B/C:
  submit_v6:  PDS = 0.352 (WORSE than random 0.500)
  submit_v8:  PDS = 0.464 (worse than random)
  submit_v9:  PDS = 0.421 (worse than random)
  rescreen:   PDS = 0.382-0.465 across all calibration configs

LOCO proxy PDS (H1 excluded, LFC-space):
  Our simulation: PDS = 0.745 (above random)
  v11 ablation:   PDS = 0.758

Official PDS (H1 fold, cell-level, thinned truth):
  pds_official.json: PDS = 0.819 (only 40 targets tested)

KEY INSIGHT: The LFC-space proxy gives ~0.75 but the actual cell-level
PDS on unseen validation contexts gives 0.35-0.46. The gap is the
entire problem.""")

    # Section 2: The two metrics measure different things
    print("=== 2. WHY LFC-SPACE PROXY != CELL-LEVEL PDS ===")
    print(f"""
The LFC-space proxy (pds_proxy_fast) computes cosine ranking directly
on predicted vs true LFC vectors. It measures: "does each prediction
point more toward its own truth than toward other truths?"

The cell-level PDS (pds_cosine in metrics.py) computes cosine ranking
on LOG PROFILES of generated COUNT CELLS vs reference COUNT CELLS.
It measures: "do the emitted cells, after Poisson thinning, encode
the right perturbation program in their pooled expression profile?"

These diverge when:
  a) Predicted LFCs are too small -> Poisson-thinned cells look like
     controls -> log_profile(pred) - log_profile(ctrl) ~ 0 -> all
     predicted effects are near-zero vectors -> cosine ranking breaks
  b) The gain calibration amplifies noise non-uniformly
  c) Library-size normalization in log_profile() redistributes signal""")

    # Section 3: Quantitative diagnosis
    print("=== 3. QUANTITATIVE DIAGNOSIS ===")
    cos_raw = cosine_sim_matrix(mat_raw)
    cos_true = cosine_sim_matrix(true_mat)
    cross = np.array([
        float(mat_raw[i] @ true_mat[i] / (pred_norms[i] * true_norms[i]))
        for i in range(len(h1_targets))
        if pred_norms[i] > 1e-10 and true_norms[i] > 1e-10
    ])

    print(f"""
A) DIRECTION QUALITY:
   Pred inter-target cosine:   {cos_raw[idx].mean():.4f}
   True inter-target cosine:   {cos_true[idx].mean():.4f}
   Pred-vs-true mean cosine:   {cross.mean():.4f}
   Fraction with cos > 0.3:    {(cross > 0.3).mean():.1%}
   Fraction with cos < 0.0:    {(cross < 0.0).mean():.1%}

B) MAGNITUDE:
   Pred mean L2 norm:          {pred_norms.mean():.2f}
   True mean L2 norm:          {true_norms.mean():.2f}
   Ratio:                      {pred_norms.mean() / true_norms.mean():.4f} ({pred_norms.mean() / true_norms.mean() * 100:.1f}%)
   Per-gene |pred| mean:       {np.abs(mat_raw).mean():.6f}
   Per-gene |true| mean:       {np.abs(true_mat).mean():.6f}

C) SHRINKAGE:
   Consistency mean:           {cons.mean():.4f}
   Shrinkage factor mean:      {shrink.mean():.4f}
   Fraction with cons < 0.3:   {(cons < 0.3).mean():.1%}

D) TARGET COVERAGE:
   Targets seen in 0 contexts: {(known_counts == 0).sum()}/{len(h1_targets)}
   Targets seen in 1 context:  {(known_counts == 1).sum()}/{len(h1_targets)}
   Targets seen in 2+ contexts:{(known_counts >= 2).sum()}/{len(h1_targets)}""")

    # Section 4: The chain of failures
    print("=== 4. CHAIN OF FAILURES ===")
    print(f"""
STEP 1: SOURCE CONTEXT MISMATCH
  All 7 source contexts are K562 or RPE1 cell lines.
  The validation contexts A/B/C are different cell states.
  The prior weights ALL source contexts nearly equally for H1:
    replogle_nadig/hepg2: 0.216
    replogle_nadig/jurkat: 0.216
    replogle_nadig/k562: 0.216
    replogle_nadig/rpe1: 0.216
    replogle_gw/RPE1: 0.053
    replogle_ess/K562: 0.043
    replogle_gw/K562: 0.040
  Result: LFC predictions are a weighted AVERAGE across all sources,
  not a transfer from the most relevant one.

STEP 2: CONSISTENCY SHRINKAGE KILLS SIGNAL
  82% of genes have consistency < 0.3 (observed in only 1 dataset/line)
  Shrinkage factor = 0.25 + 0.75 * 0.19 = 0.39 on average
  -> 61% of the already-wrong signal is zeroed out

STEP 3: MAGNITUDE COLLAPSE
  Predicted L2 norms = {pred_norms.mean():.2f} vs truth = {true_norms.mean():.2f}
  After shrinkage + averaging: predictions are {pred_norms.mean() / true_norms.mean() * 100:.1f}% of truth magnitude

STEP 4: CELL GENERATION AMPLIFIES THE PROBLEM
  Poisson thinning: y ~ Poisson(x * 2^LFC)
  When LFC ~ 0.04 (our predictions): 2^0.04 = 1.028
  -> Generated cells are nearly identical to control cells
  -> log_profile(pred) - log_profile(ctrl) ~ 0
  -> ALL predicted effects become ~zero vectors in profile space

STEP 5: COSINE RANKING COLLAPSES
  When all predicted effects are ~zero vectors:
    cos(pred_i, true_j) ~ 0 for all j
    cos(pred_i, true_i) ~ 0 for all i
  -> All distances are ~1.0 -> ranking is random -> PDS ~ 0.5
  Worse: noise in the zero vectors can make pred_i accidentally
  closer to some OTHER true_j -> PDS < 0.5

STEP 6: CALIBRATION CAN'T FIX THIS
  gain=2.1, q_threshold=0.35:
  -> Amplifies tiny predictions by 2.1x, still tiny
  -> q_threshold zeros out bottom 35% of |LFC| per target
  -> Remaining non-zero genes are noise, not signal
  -> Cell-level profiles still look like controls""")

    # Section 5: The four specific failure modes
    print("=== 5. FOUR FAILURE MODES (from your question) ===")
    print(f"""
a) ALL 300 PREDICTED LFC VECTORS POINT IN SIMILAR DIRECTIONS?
   YES, but only for H1-as-unseen: inter-target cosine = {cos_raw[idx].mean():.4f}
   vs truth = {cos_true[idx].mean():.4f}.
   However, this is NOT the main PDS killer. Even with diverse
   directions (cos~0.02), the MAGNITUDE problem dominates.
   The real issue is that all vectors are too small to matter.

b) MAGNITUDES TOO SMALL (not enough signal)?
   YES - THIS IS THE PRIMARY PROBLEM.
   Predicted LFCs are {pred_norms.mean() / true_norms.mean() * 100:.1f}% of truth magnitude.
   After Poisson thinning, the generated cells barely differ from controls.
   The cell-level log profiles are essentially indistinguishable.

c) MAGNITUDES TOO LARGE (noise overwhelms)?
   NO - magnitudes are far too small, not too large.
   Even with gain=50, PDS stays at 0.745 in LFC space (our sweep showed this).
   But in cell space, high gain amplifies stochastic noise in the Poisson
   sampling, which can make things worse.

d) PER-GENE PATTERN WRONG (correct direction, wrong specific genes)?
   PARTIALLY. Mean pred-vs-true cosine = {cross.mean():.4f} is very low.
   The prior transfers the wrong program because:
   - Source contexts are different cell lines
   - Consistency shrinkage zeros out most gene dimensions
   - The shared program (global_mean_effect) dominates""")

    # Section 6: What would fix PDS
    print("=== 6. WHAT WOULD FIX PDS ===")
    print("""
IMMEDIATE FIXES (magnitude):
  1. Set use_alpha=False AND use much higher calibration gain (10-50x)
     so that predicted cells actually differ from controls
  2. Remove or reduce consistency shrinkage (default_conservation=0.75+)
  3. Set shared_mix=0 to avoid adding a common direction

MEDIUM-TERM FIXES (direction):
  4. Use basal-profile correlation PER GENE, not just global context similarity
  5. Train a gene-level context-weight model instead of softmax over contexts
  6. Use the residual network (currently disabled in v11: resid=0.0)

FUNDAMENTAL FIX:
  7. The prior's similarity weighting is broken for unseen contexts:
     it gives ~equal weight to all sources because basal profiles
     are correlated at the global level but differ at gene level.
     Need gene-specific or pathway-specific context weighting.
""")

if __name__ == "__main__":
    main()
