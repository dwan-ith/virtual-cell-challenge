"""Differential-expression table per the vcc2026 spec (section 3 of the brief).

For one perturbation group vs the reference control cells:
  - rank values are counts normalized per cell to DE_TARGET_SUM CPM
  - two-sided Wilcoxon rank-sum, normal-approximation p-value (asymptotic)
  - genes tested only if mean expression > 5 CPM in the CONTROL cells
  - BH correction within the perturbation over surviving genes
  - lfc = log2((mean_pert_cpm + eps)/(mean_ctrl_cpm + eps)), arithmetic means

The gate reads the reference control alone, so it admits the same gene set for
every submission in a context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu

from . import spec
from .countsim import benjamini_hochberg, normalize_cpm


@dataclass
class DETable:
    """Per-gene arrays for one perturbation (all restricted to tested genes)."""

    genes_idx: np.ndarray      # indices into the gene axis that were tested
    pval: np.ndarray
    padj: np.ndarray
    lfc: np.ndarray

    def significant(self, alpha: float = spec.DE_PADJ_THRESHOLD) -> np.ndarray:
        m = self.padj < alpha
        m &= ~np.isnan(self.padj)
        return self.genes_idx[m]


def control_gate(ctrl_cpm: np.ndarray) -> np.ndarray:
    """Genes with mean CPM strictly above the filter in control cells."""
    return ctrl_cpm.mean(axis=0) > spec.DE_MIN_CPM_CONTROL


def de_table_for_group(
    pert_counts: np.ndarray,
    ctrl_counts: np.ndarray | None,
    gate: np.ndarray | None = None,
    ctrl_cpm: np.ndarray | None = None,
) -> DETable:
    """Full DE table for one (perturbation vs control) comparison.

    Pass precomputed `ctrl_cpm` (normalize_cpm of the control matrix) instead
    of `ctrl_counts` to avoid re-normalizing a large control matrix on every
    call -- hot path for exact calibration tuning.
    """
    pert_cpm = normalize_cpm(pert_counts, spec.DE_TARGET_SUM)
    if ctrl_cpm is None:
        if ctrl_counts is None:
            raise ValueError("need either ctrl_counts or ctrl_cpm")
        ctrl_cpm = normalize_cpm(ctrl_counts, spec.DE_TARGET_SUM)
    if gate is None:
        gate = control_gate(ctrl_cpm)
    idx = np.flatnonzero(gate)
    if idx.size == 0:
        return DETable(idx, np.array([]), np.array([]), np.array([]))

    x = pert_cpm[:, idx]
    y = ctrl_cpm[:, idx]
    # Two-sided asymptotic Mann-Whitney per column.
    res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", axis=0)
    pval = np.asarray(res.pvalue, dtype=np.float64)

    mp = x.mean(axis=0)
    mc = y.mean(axis=0)
    lfc = np.log2((mp + spec.DE_EPSILON) / (mc + spec.DE_EPSILON))

    padj = benjamini_hochberg(pval)
    return DETable(idx, pval, padj, lfc)


def full_de_tables(
    pred_pert: dict[str, np.ndarray],
    real_ctrl: np.ndarray,
    n_control_subsample: int | None = None,
    seed: int = 0,
) -> dict[str, DETable]:
    """DE tables for every predicted perturbation against reference controls.

    n_control_subsample: cap on control cells used in the tests (power drops but
    ranking signal is preserved); None uses all controls exactly as scoring does.
    """
    ctrl = real_ctrl
    if n_control_subsample is not None and ctrl.shape[0] > n_control_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(ctrl.shape[0], size=n_control_subsample, replace=False)
        ctrl = ctrl[idx]

    gate = control_gate(normalize_cpm(ctrl, spec.DE_TARGET_SUM))
    out: dict[str, DETable] = {}
    for name, cells in pred_pert.items():
        out[name] = de_table_for_group(cells, ctrl, gate)
    return out
