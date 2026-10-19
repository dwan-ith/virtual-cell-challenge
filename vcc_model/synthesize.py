"""Synthetic multi-context CRISPRi count data for honest zero-shot validation.

Simulates the structure the 2026 challenge tests:
  (a) a sparse shared program per perturbation (transferable directions),
  (b) a per-context magnitude scale (what the transfer prior must calibrate),
  (c) context-private extra hits and occasional sign flips on the shared
      program (what only context-conditioned models / consistency shrinkage
      can capture).

Sparsity matters: in real CRISPRi screens only ~2-10% of tested genes are
significant per perturbation. The generator enforces exact-zero LFC outside
each program's support so the Wilcoxon null behaves realistically and the
anchored six-metric scores have meaningful spans (b_jac << r_jac etc.).

Counts come from Poisson sampling of per-cell rates with lognormal depth —
the same generative direction as multinomial thinning, so our decoder's
output distribution matches how reference "truth" cells are produced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticChallenge:
    gene_names: np.ndarray                       # (G,)
    contexts: list[str]
    pert_genes: list[str]                        # panel
    controls: dict[str, np.ndarray]              # ctx -> (n_ctrl, G) counts
    truth: dict[tuple[str, str], np.ndarray]     # (ctx, pert) -> (n, G) counts
    lfc_truth: dict[tuple[str, str], np.ndarray] # (ctx, pert) -> (G,) true LFC

    def contexts_sorted(self) -> list[str]:
        return sorted(self.contexts)


def make_synthetic_challenge(
    n_genes: int = 512,
    n_contexts: int = 6,
    n_perts: int = 16,
    n_ctrl_cells: int = 400,
    n_cells_per_pert: int = 96,
    median_depth: int = 20_000,
    effect_scale: float = 1.0,
    support_div: int = 12,          # shared-program size = n_genes // this
    effect_sigma: float = 1.0,
    own_knockdown: float = 2.2,
    flip_prob: float = 0.35,        # P(a given (ctx,pert) flips some signs)
    flip_frac: float = 0.3,
    extra_div: int = 24,            # context-private extra-hit count
    seed: int = 0,
) -> SyntheticChallenge:
    rng = np.random.default_rng(seed)
    gene_names = np.array([f"G{i:05d}" for i in range(n_genes)])
    contexts = [f"line_{c}" for c in range(n_contexts)]
    pert_genes = [gene_names[i] for i in rng.choice(n_genes, size=n_perts, replace=False)]

    support_size = max(6, n_genes // support_div)
    extra_size = max(4, n_genes // extra_div)

    own_idx = [int(np.flatnonzero(gene_names == g)[0]) for g in pert_genes]

    # ---- shared sparse programs + supports ------------------------------
    shared_lfc = np.zeros((n_perts, n_genes))
    supports: list[np.ndarray] = []
    for j in range(n_perts):
        supp = rng.choice(n_genes, size=support_size, replace=False)
        if own_idx[j] in supp:  # keep own-gene effect clean of the draw
            supp = supp[supp != own_idx[j]]
        supports.append(supp)
        shared_lfc[j, supp] = rng.normal(0.0, effect_sigma * effect_scale, size=supp.size)
        shared_lfc[j, own_idx[j]] -= abs(rng.normal(own_knockdown, 0.4)) * effect_scale

    # ---- per-context behavior ------------------------------------------
    basal_log = rng.normal(2.5, 1.1, size=(n_contexts, n_genes))  # log-rate space
    context_scale = rng.uniform(0.55, 1.45, size=n_contexts)

    controls: dict[str, np.ndarray] = {}
    truth: dict[tuple[str, str], np.ndarray] = {}
    lfc_truth: dict[tuple[str, str], np.ndarray] = {}

    def sample_counts(mean_rate: np.ndarray, n: int, depth_med: float) -> np.ndarray:
        depths = rng.lognormal(np.log(depth_med), 0.35, size=n)
        probs = mean_rate / max(mean_rate.sum(), 1e-12)
        return rng.poisson(np.outer(depths, probs))

    for c, ctx in enumerate(contexts):
        ctrl_rate = np.exp(basal_log[c])
        controls[ctx] = sample_counts(ctrl_rate, n_ctrl_cells, median_depth)
        for j, g in enumerate(pert_genes):
            lfc = context_scale[c] * shared_lfc[j].copy()

            # Occasional sign flips on part of the shared support: models must
            # learn WHEN directions transfer (consistency shrinkage target).
            if rng.random() < flip_prob and supports[j].size >= 4:
                k = max(2, int(flip_frac * supports[j].size))
                flip = rng.choice(supports[j], size=k, replace=False)
                lfc[flip] *= -rng.uniform(0.5, 1.0)

            # Context-private extra hits (unseeable from other contexts alone;
            # the residual net may learn them only via basal-state features).
            pool = np.setdiff1d(np.arange(n_genes), np.union1d(supports[j], own_idx[j]))
            extra = rng.choice(pool, size=min(extra_size, pool.size), replace=False)
            lfc[extra] += rng.normal(0.0, 0.8 * effect_scale, size=extra.size)

            pert_rate = ctrl_rate * np.exp2(lfc)
            truth[(ctx, g)] = sample_counts(pert_rate, n_cells_per_pert, median_depth)
            lfc_truth[(ctx, g)] = lfc.astype(np.float32)

    return SyntheticChallenge(
        gene_names=gene_names,
        contexts=contexts,
        pert_genes=pert_genes,
        controls=controls,
        truth=truth,
        lfc_truth=lfc_truth,
    )
