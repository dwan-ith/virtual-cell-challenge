"""Official VCC 2026 constants, transcribed from primary sources.

Sources (verified 2026-08-22):
  - virtualcellchallenge.org/evaluation page bundle (six metric names, submission rules)
  - cell-eval2 configs/vcc2026.yaml + docs/vcc2026_metrics/vcc2026-metrics-brief.md
  - vcc-cli-wiki.virtualcellchallenge.org (CLI guide: counts, cells, contexts, caps)
"""

from __future__ import annotations

# --- Panel / submission shape -------------------------------------------------
N_GENES = 18_533            # gene_names.csv axis length (--expected-gene-dim)
N_PERTS_PER_CONTEXT = 300   # perturbations per context
CELLS_PER_PERT = 400        # exact cells per perturbation, every context
VALIDATION_CONTEXTS = ("A", "B", "C")
FINAL_CONTEXTS = ("D", "E", "F")
CONTROL_LABEL = "non-targeting"
TARGET_COL = "target_gene"
CONTEXT_COL = "context"

MAX_CELLS_TOTAL = 400_000
PANEL_TOTAL_CELLS = N_PERTS_PER_CONTEXT * CELLS_PER_PERT * 3  # 360,000
MAX_COUNTS_PER_CELL = 1_000_000
DENSITY_CAP_ENTRIES = 4_750_000_000  # stored entries (nnz), not file size

# --- Metric parameters (cell-eval2 vcc2026.yaml) ------------------------------
BULK_TARGET_SUM = 50_000.0     # pseudobulk normalization for PDS + expr MSE
DE_TARGET_SUM = 1_000_000.0    # per-cell CPM total for DE tests
DE_EPSILON = 1e-9              # fold-change epsilon
DE_PADJ_THRESHOLD = 0.05       # Benjamini-Hochberg, within perturbation
DE_MIN_CPM_CONTROL = 5.0       # low-expression gate, read on reference control
REACH_PURITY_FLOOR = 0.9       # direction-reach prefix purity P0
LFC_NMAE_MIN_GATE = 10         # minimum reference-significant genes per pert
EXPR_MSE_CLAMP = (0.0, 1.0)
LFC_NMAE_FLOOR = -6.0
N_REPLICATE_SPLITS = 5         # split-half replicate anchor seeds
REPLICATE_BASE_SEED = 0

# The six scored members of the vcc2026 profile (leaderboard short names):
#   pds                                        -> pds_cosine                          [0,1] up
#   mse                                        -> expr_mse_unbiased_capped_norm      [0,1] down
#   nmae                                       -> de_wilcoxon_lfc_nmae               floor -6, down
#   fid                                        -> de_wilcoxon_direction_fidelity_yield_raw   up
#   reach                                      -> de_wilcoxon_direction_reach_raw    up
#   jac                                        -> de_wilcoxon_sig_jaccard            up
SCORED_MEMBERS = (
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)

HIGHER_IS_BETTER = {
    "pds_cosine": True,
    "expr_mse_unbiased_capped_norm": False,
    "de_wilcoxon_direction_fidelity_yield_raw": True,
    "de_wilcoxon_direction_reach_raw": True,
    "de_wilcoxon_sig_jaccard": True,
    "de_wilcoxon_lfc_nmae": False,
}

LEADERBOARD_SHORTNAMES = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
    "de_wilcoxon_lfc_nmae": "nmae",
}
