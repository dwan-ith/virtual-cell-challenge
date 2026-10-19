"""AnnData helpers and synthetic multi-context Perturb-seq for zero-shot splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CONTROL_LABEL = "non-targeting"


@dataclass
class ZeroShotSplit:
    """Train contexts see perturbations; holdout contexts only see basal cells."""

    train: ad.AnnData
    holdout_controls: ad.AnnData
    holdout_truth: ad.AnnData
    holdout_genes: list[str]
    holdout_contexts: list[str]


def _to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def mean_expression(adata: ad.AnnData) -> np.ndarray:
    return _to_dense(adata.X).mean(axis=0)


def log1p_normalize(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    lib = counts.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p(counts * (target_sum / lib))


def make_synthetic_perturb_seq(
    n_genes: int = 64,
    n_contexts: int = 6,
    n_pert_genes: int = 12,
    cells_per_condition: int = 40,
    seed: int = 0,
) -> ad.AnnData:
    """Simulate multi-line CRISPRi responses with shared + context-specific effects.

    Shared Δ captures transferable biology; context scaling / residual captures
    the zero-shot transfer problem Arc is testing in 2026.
    """
    rng = np.random.default_rng(seed)
    gene_names = [f"G{i:03d}" for i in range(n_genes)]
    contexts = [f"line_{c}" for c in range(n_contexts)]
    pert_genes = gene_names[:n_pert_genes]

    # Shared regulatory effects: knocking down gene j shifts a sparse program.
    shared_delta = np.zeros((n_pert_genes, n_genes), dtype=np.float32)
    for j, g in enumerate(pert_genes):
        targets = rng.choice(n_genes, size=max(4, n_genes // 8), replace=False)
        shared_delta[j, targets] = rng.normal(0, 1.2, size=len(targets)).astype(np.float32)
        # Knockdown of the target itself.
        shared_delta[j, gene_names.index(g)] -= abs(rng.normal(2.0, 0.3))

    # Per-context basal programs and response scales.
    # Residual effects are a function of basal state so a context-conditioned
    # network can learn transferrable corrections the mean prior misses.
    basal = rng.normal(2.0, 0.5, size=(n_contexts, n_genes)).astype(np.float32)
    basal = np.clip(basal, 0.05, None)
    context_scale = rng.uniform(0.6, 1.4, size=(n_contexts, 1)).astype(np.float32)
    # Low-rank basal→residual map shared across contexts (the learnable piece).
    w_resid = rng.normal(0, 0.15, size=(n_genes, n_genes)).astype(np.float32)
    context_residual = np.zeros((n_contexts, n_pert_genes, n_genes), dtype=np.float32)
    for c_idx in range(n_contexts):
        activity = np.tanh(basal[c_idx] - basal[c_idx].mean())
        for j in range(n_pert_genes):
            context_residual[c_idx, j] = (w_resid @ activity) * (0.5 + 0.1 * j)

    rows: list[np.ndarray] = []
    obs_context: list[str] = []
    obs_pert: list[str] = []

    for c_idx, context in enumerate(contexts):
        # Controls
        noise = rng.normal(0, 0.25, size=(cells_per_condition, n_genes)).astype(np.float32)
        ctrl = np.clip(basal[c_idx] + noise, 0, None)
        rows.append(ctrl)
        obs_context.extend([context] * cells_per_condition)
        obs_pert.extend([CONTROL_LABEL] * cells_per_condition)

        for j, g in enumerate(pert_genes):
            delta = context_scale[c_idx] * shared_delta[j] + context_residual[c_idx, j]
            noise = rng.normal(0, 0.25, size=(cells_per_condition, n_genes)).astype(np.float32)
            expr = np.clip(basal[c_idx] + delta + noise, 0, None)
            rows.append(expr)
            obs_context.extend([context] * cells_per_condition)
            obs_pert.extend([g] * cells_per_condition)

    x = np.vstack(rows)
    # Convert soft expression to count-like integers then log1p-normalize.
    counts = rng.poisson(np.expm1(x) * 5.0).astype(np.float32)
    x_norm = log1p_normalize(counts)

    adata = ad.AnnData(
        X=x_norm,
        obs=pd.DataFrame(
            {
                "context": obs_context,
                "perturbation": obs_pert,
            }
        ),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.uns["shared_delta"] = shared_delta
    adata.uns["pert_genes"] = pert_genes
    return adata


def make_zero_shot_split(
    adata: ad.AnnData,
    holdout_contexts: Iterable[str],
    control_key: str = "perturbation",
    context_key: str = "context",
    control_label: str = CONTROL_LABEL,
) -> ZeroShotSplit:
    """Hide all perturbation responses from holdout contexts (2026 task shape)."""
    holdout = list(holdout_contexts)
    train_mask = ~adata.obs[context_key].isin(holdout).to_numpy()
    hold_mask = ~train_mask

    train = adata[train_mask].copy()
    hold = adata[hold_mask].copy()

    hold_ctrl = hold[hold.obs[control_key] == control_label].copy()
    hold_truth = hold[hold.obs[control_key] != control_label].copy()
    genes = sorted(g for g in hold_truth.obs[control_key].unique() if g != control_label)

    return ZeroShotSplit(
        train=train,
        holdout_controls=hold_ctrl,
        holdout_truth=hold_truth,
        holdout_genes=genes,
        holdout_contexts=holdout,
    )


def align_genes(adata: ad.AnnData, gene_order: list[str]) -> ad.AnnData:
    """Reindex to a shared gene vocabulary; missing genes filled with 0."""
    present = [g for g in gene_order if g in adata.var_names]
    missing = [g for g in gene_order if g not in adata.var_names]
    out = adata[:, present].copy()
    if missing:
        fill = np.zeros((out.n_obs, len(missing)), dtype=np.float32)
        fill_ad = ad.AnnData(
            X=fill,
            obs=out.obs.copy(),
            var=pd.DataFrame(index=missing),
        )
        out = ad.concat([out, fill_ad], axis=1, merge="same")
    return out[:, gene_order].copy()
