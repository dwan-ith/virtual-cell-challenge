"""Count-space primitives: CPM, pseudobulk, log profiles, fold changes.

All scoring-relevant quantities in VCC 2026 are defined on raw integer counts
(see spec.py). Everything here operates on 2D arrays shaped (n_cells, n_genes),
dense float32/float64 or scipy sparse; outputs are dense.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.todense(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def library_sizes(counts: np.ndarray) -> np.ndarray:
    return counts.sum(axis=1)


def normalize_cpm(counts: np.ndarray, target_sum: float = 1e6) -> np.ndarray:
    """Per-cell normalize to a common total (spec DE_TARGET_SUM for DE)."""
    totals = library_sizes(counts)
    safe = np.where(totals > 0, totals, 1.0)
    return counts * (target_sum / safe[:, None])


def pseudobulk_sum(counts: np.ndarray) -> np.ndarray:
    return counts.sum(axis=0)


def log_profile(group_counts: np.ndarray, target_sum: float = 5e4) -> np.ndarray:
    """b_{p,g} = log(1 + TS * P_pg / sum_g' P_pg') — the PDS / expr-MSE space."""
    total = group_counts.sum()
    if total <= 0:
        return np.zeros(group_counts.shape[1], dtype=np.float64)
    return np.log1p(target_sum * group_counts / total)


def log2_fold_change(
    mean_pert_cpm: np.ndarray,
    mean_ctrl_cpm: np.ndarray,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """lfc = log2((m_pert + eps) / (m_ctrl + eps)) on CPM-normalized means."""
    return np.log2((mean_pert_cpm + epsilon) / (mean_ctrl_cpm + epsilon))


def mean_cpm_profile(counts: np.ndarray, target_sum: float = 1e6) -> np.ndarray:
    """Arithmetic mean of per-cell CPM — the space the DE LFC is defined in."""
    return normalize_cpm(counts, target_sum).mean(axis=0)


def log_mean_cpm(counts: np.ndarray, target_sum: float = 1e6) -> np.ndarray:
    """Log of the mean-CPM profile; used as a context's basal-state vector."""
    return np.log(mean_cpm_profile(counts, target_sum) + 1e-9)


def sparse_mean_cpm(x, target_sum: float = 1e6) -> np.ndarray:
    """Mean per-cell CPM profile for a sparse (cells x genes) count matrix."""
    import numpy as _np
    from scipy import sparse as _sp

    if not _sp.issparse(x):
        return mean_cpm_profile(np.asarray(x), target_sum)
    x = x.tocsr()
    totals = _np.asarray(x.sum(axis=1)).ravel()
    safe = _np.where(totals > 0, totals, 1.0)
    scaled = x.multiply((target_sum / safe)[:, None]).tocsr()
    return _np.asarray(scaled.sum(axis=0)).ravel() / max(x.shape[0], 1)


def sparse_log_mean_cpm(x, target_sum: float = 1e6) -> np.ndarray:
    """log(mean CPM) without densifying a large control matrix."""
    return np.log(sparse_mean_cpm(x, target_sum) + 1e-9)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH FDR adjustment over a 1-D array of p-values (preserves NaN input)."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)
    adjusted = np.empty_like(ranked)
    adjusted[order] = ranked
    out[valid] = adjusted
    return out


def multinomial_thin(
    base_cells: np.ndarray,
    multipliers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate integer-count cells by resampling each base cell's molecules.

    For cell i with counts x_i and per-gene multipliers m_g > 0, draws
        y_i ~ Multinomial(sum(x_i), q) with q_g proportional to x_ig * m_g.
    Preserves library-size distribution exactly; support stays within the base
    cell's support (density cap friendly); expected value scales by m_g.
    """
    x = to_dense(base_cells).astype(np.int64)
    m = np.asarray(multipliers, dtype=np.float64)
    m = np.clip(m, 1e-4, None)
    out = np.zeros_like(x)
    weighted = x * m[None, :]
    row_tot = weighted.sum(axis=1)
    for i in range(x.shape[0]):
        if row_tot[i] <= 0 or x[i].sum() == 0:
            continue
        probs = weighted[i] / row_tot[i]
        out[i] = rng.multinomial(int(x[i].sum()), probs)
    return out.astype(np.int64)


def poisson_thin(
    base_cells: np.ndarray,
    multipliers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized count resampler: y_ig ~ Poisson(x_ig * m_g), one draw per entry.

    Same expectation as multinomial_thin but independent per gene, which adds a
    realistic amount of sampling noise (library sizes fluctuate like real
    sequencing instead of being frozen to the base cells'). Fully vectorized --
    orders of magnitude faster than the per-cell multinomial loop at submission
    scale. Support only shrinks (a nonzero x may draw 0; zero stays zero), so
    the density cap stays safe.
    """
    x = to_dense(base_cells)
    m = np.asarray(multipliers, dtype=np.float64)
    m = np.clip(m, 1e-6, None)
    if m.ndim == 1:
        m = m[None, :]
    lam = np.clip(x, 0.0, None) * m
    return rng.poisson(lam).astype(np.int64)


def bulk_probability_from_lfc(
    control_probability: np.ndarray,
    lfc: np.ndarray,
    bulk_gain: float = 1.0,
    clip: float = 6.0,
) -> np.ndarray:
    """Target pooled bulk probabilities for one perturbation.

    Uses the scorer's pooled-control basis: the desired bulk is proportional
    to ``control_probability * 2 ** (bulk_gain * lfc)``. Keeping this separate
    from the per-cell multiplier allows the summed profile to carry a slightly
    different strength than individual cells.
    """
    q = np.asarray(control_probability, dtype=np.float64)
    e = np.asarray(lfc, dtype=np.float64)
    if q.shape != e.shape:
        raise ValueError("control_probability and lfc must have the same shape")
    if not np.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
        raise ValueError("control_probability must be finite, nonnegative, and nonzero")
    q = q / q.sum()
    desired = q * np.exp2(np.clip(bulk_gain * e, -clip, clip))
    total = float(desired.sum())
    if not np.isfinite(total) or total <= 0:
        return q
    return desired / total


def load_promoter_pairs(path, gene_names) -> dict[str, list[tuple[int, float]]]:
    """Load AtlasShift-style promoter pairs: target, neighbor, distance."""
    import pandas as pd

    pairs = pd.read_csv(path, usecols=["target", "neighbor", "distance"])
    idx = {str(g): i for i, g in enumerate(gene_names)}
    out: dict[str, list[tuple[int, float]]] = {}
    for row in pairs.itertuples(index=False):
        target, neighbor = str(row.target), str(row.neighbor)
        if target not in idx or neighbor not in idx or target == neighbor:
            continue
        try:
            distance = float(row.distance)
        except (TypeError, ValueError):
            continue
        if not 0 <= distance <= 5000:
            continue
        out.setdefault(target, []).append((idx[neighbor], distance))
    return out


def apply_promoter_ceiling(
    probability: np.ndarray,
    control_probability: np.ndarray,
    target: str,
    pairs: dict[str, list[tuple[int, float]]],
    fraction: float = 0.15,
) -> np.ndarray:
    """Cap promoter neighbors of a CRISPRi target at a distance-ramped ceiling."""
    p = np.asarray(probability, dtype=np.float64).copy()
    c = np.asarray(control_probability, dtype=np.float64)
    if p.shape != c.shape:
        raise ValueError("probability and control_probability must have the same shape")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("probability must sum to a positive finite value")
    for j, distance in pairs.get(str(target), []):
        ramp = np.clip(np.log(max(float(distance), 500.0) / 500.0) / np.log(10.0), 0.0, 1.0)
        ceiling = c[j] * (fraction + (1.0 - fraction) * ramp)
        if p[j] > ceiling:
            p[j] = ceiling
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("promoter ceiling removed all probability mass")
    return p / total


def repair_bulk_totals(
    integer_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> np.ndarray:
    """Adjust integer cells to exact bulk totals while preserving row depths.

    This keeps per-cell library sizes and most Poisson/multinomial sampling
    noise, while forcing the column sums to the desired bulk totals. It may
    create support in deficit genes; that is preferable to leaving a bulk
    error, but callers should monitor density and DE behavior.

    Raises ValueError when the requested bulk is infeasible from the supplied
    nonzero support/depths.
    """
    y = np.asarray(integer_counts, dtype=np.int64).copy()
    e = np.asarray(expected_counts, dtype=np.float64)
    if y.ndim != 2 or e.shape != y.shape:
        raise ValueError("integer and expected count matrices must have the same 2D shape")
    if (y < 0).any() or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("counts must be finite; integer counts must be nonnegative")
    depths = y.sum(axis=1, dtype=np.int64)
    if (depths <= 0).any():
        raise ValueError("every repaired cell must have positive depth")
    grand = int(depths.sum())
    col_expected = e.sum(axis=0)
    if abs(float(col_expected.sum()) - grand) > 1e-6 * max(grand, 1):
        raise ValueError("expected and integer grand totals differ")

    target = np.floor(col_expected).astype(np.int64)
    remainder = int(grand - int(target.sum()))
    if remainder < 0 or remainder > target.size:
        raise ValueError("cannot apportion bulk remainder")
    if remainder:
        fractional = col_expected - np.floor(col_expected)
        target[np.argsort(fractional, kind="stable")[-remainder:]] += 1

    diff = y.sum(axis=0, dtype=np.int64) - target
    if not diff.any():
        return y

    floor_e = np.floor(e).astype(np.int64)
    capacity = np.zeros(y.shape[0], dtype=np.int64)

    for g in np.flatnonzero(diff > 0):
        need = int(diff[g])
        surplus = y[:, g] - floor_e[:, g]
        donors = np.flatnonzero(surplus > 0)
        if donors.size == 0:
            raise ValueError("bulk excess cannot be removed without going below floor")
        order = donors[np.argsort(-surplus[donors], kind="stable")]
        pos = 0
        while need > 0:
            if pos >= order.size:
                # All donors are at floor; take additional counts from the
                # currently largest entries rather than failing silently.
                order = np.flatnonzero(y[:, g] > 0)
                order = order[np.argsort(-y[order, g], kind="stable")]
                if order.size == 0:
                    raise ValueError("bulk excess has no removable counts")
                pos = 0
            i = int(order[pos])
            y[i, g] -= 1
            capacity[i] += 1
            need -= 1
            pos += 1

    deficit_genes = np.flatnonzero(diff < 0)
    deficit_genes = deficit_genes[np.argsort(diff[deficit_genes], kind="stable")]
    for g in deficit_genes:
        need = int(-diff[g])
        while need > 0:
            rows = np.flatnonzero(capacity > 0)
            if rows.size == 0:
                raise ValueError("bulk deficit has no row capacity")
            score = e[rows, g] - y[rows, g] + 0.25 * (y[rows, g] > 0)
            take = min(need, rows.size)
            selected = rows[np.argsort(-score, kind="stable")[:take]]
            y[selected, g] += 1
            capacity[selected] -= 1
            need -= take

    if capacity.any() or (y < 0).any():
        raise ValueError("bulk repair lost counts")
    if not np.array_equal(y.sum(axis=1, dtype=np.int64), depths):
        raise ValueError("bulk repair changed a cell depth")
    if not np.array_equal(y.sum(axis=0, dtype=np.int64), target):
        raise ValueError("bulk repair failed its bulk totals")
    return y
