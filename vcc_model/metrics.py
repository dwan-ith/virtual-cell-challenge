"""Faithful local implementation of the six vcc2026 metrics (cell-eval2 0.15.0,
rule_version 3 semantics) plus the two anchors used to scale them.

Definitions follow docs/vcc2026_metrics/vcc2026-metrics-brief.md exactly:
  s = (u - b) / (r - b)   per member, per context
with b the mean-response baseline anchor and r the split-half replicate anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import spec
from .countsim import log_profile, normalize_cpm
from .de_table import DETable, de_table_for_group


@dataclass
class ContextData:
    """One scored context: raw integer counts on a shared gene axis."""

    gene_names: np.ndarray                 # (G,) gene symbols
    real_ctrl: np.ndarray                  # (n_ctrl, G) reference control cells
    real_pert: dict[str, np.ndarray]       # pert name -> (n, G) reference cells
    pred_pert: dict[str, np.ndarray]       # pert name -> (n, G) predicted cells

    def perts(self) -> list[str]:
        return sorted(self.real_pert)

    def gene_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.gene_names)}


# --------------------------------------------------------------------------
# PDS (brief section 1)
# --------------------------------------------------------------------------

def pds_cosine(
    cd: ContextData,
    target_sum: float = spec.BULK_TARGET_SUM,
) -> float:
    """Cosine-rank discrimination; tied block shares midrank; zero-norm -> 0.5."""
    names = cd.perts()
    n = len(names)
    if n < 2:
        return float("nan")

    ctrl_log = log_profile(cd.real_ctrl.sum(axis=0), target_sum)
    idx = cd.gene_index()
    panel = {idx[g] for g in names if g in idx}
    keep = np.array([i for i in range(len(cd.gene_names)) if i not in panel], dtype=int)

    def effects(pert_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name in names:
            prof = log_profile(pert_map[name].sum(axis=0), target_sum)
            out[name] = (prof - ctrl_log)[keep]
        return out

    real_eff = effects(cd.real_pert)
    pred_eff = effects(cd.pred_pert)

    real_mat = np.stack([real_eff[nm] for nm in names])
    nr = np.linalg.norm(real_mat, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    real_unit = real_mat / nr

    pos_of = {nm: i for i, nm in enumerate(names)}
    scores = []
    for name in names:
        dhat = pred_eff[name]
        nh = np.linalg.norm(dhat)
        if nh == 0:
            scores.append(0.5)  # every distance ties at 1 -> midrank 0.5
            continue
        dists = 1.0 - real_unit @ (dhat / nh)
        d_self = dists[pos_of[name]]
        less = int(np.count_nonzero(dists < d_self))
        equal = int(np.count_nonzero(dists == d_self))
        k = less + 0.5 * (equal - 1)
        scores.append(1.0 - k / (n - 1))
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# Expression error (brief section 2)
# --------------------------------------------------------------------------

def _jackknife_c(group_counts: np.ndarray, target_sum: float) -> float:
    """C_p: delete-one jackknife variance of the group log profile."""
    x = np.asarray(group_counts, dtype=np.float64)
    n_cells = x.shape[0]
    if n_cells < 2:
        return 0.0
    totals = x.sum(axis=1)
    group_total = float(totals.sum())
    col_sums = x.sum(axis=0)
    v_sum = np.zeros(x.shape[1])
    v_sq = np.zeros(x.shape[1])
    counted = 0
    for i in range(n_cells):
        den = group_total - totals[i]
        if den <= 0:
            continue
        v = np.log1p(target_sum * (col_sums - x[i]) / den)
        v_sum += v
        v_sq += v * v
        counted += 1
    if counted < 2:
        return 0.0
    vbar = v_sum / counted
    ss = float(v_sq.sum() - counted * float(vbar @ vbar))
    return (counted - 1) / counted * max(ss, 0.0)


def expr_mse_unbiased(
    cd: ContextData,
    target_sum: float = spec.BULK_TARGET_SUM,
) -> float:
    """expr_mse_unbiased_capped_norm: panel ratio of unbiased squared errors."""
    idx = cd.gene_index()
    ctrl_log = log_profile(cd.real_ctrl.sum(axis=0), target_sum)
    c_ctrl = _jackknife_c(cd.real_ctrl, target_sum)

    names = cd.perts()
    b_hats, b_reals, gps, c_hats, c_reals = [], [], [], [], []
    for name in names:
        own = idx.get(name)
        keep = np.ones(len(cd.gene_names), dtype=bool)
        if own is not None:
            keep[own] = False
        g_p = int(keep.sum())
        b_hats.append(log_profile(cd.pred_pert[name].sum(axis=0), target_sum)[keep])
        b_reals.append(log_profile(cd.real_pert[name].sum(axis=0), target_sum)[keep])
        gps.append(1.0 / g_p)
        c_hats.append(_jackknife_c(cd.pred_pert[name], target_sum))
        c_reals.append(_jackknife_c(cd.real_pert[name], target_sum))

    credit_pool = sum(w * min(ch, cr) for w, ch, cr in zip(gps, c_hats, c_reals))

    if credit_pool > 0:
        wsum = sum(gps)
        bbar_w = sum(w * b for w, b in zip(gps, b_hats)) / wsum
        big_b = sum(
            w * float((b - bbar_w) @ (b - bbar_w)) for w, b in zip(gps, b_hats)
        )
        rho = min(1.0, big_b / credit_pool)
    else:
        rho = 1.0

    numer = sum(
        w * (float(d @ d) - rho * min(ch, cr) - cr)
        for w, d, ch, cr in zip(gps, (bh - br for bh, br in zip(b_hats, b_reals)), c_hats, c_reals)
    )
    denom = 0.0
    for name, br, cr, w in zip(names, b_reals, c_reals, gps):
        own = idx.get(name)
        keep = np.ones(len(cd.gene_names), dtype=bool)
        if own is not None:
            keep[own] = False
        d = br - ctrl_log[keep]
        denom += w * (float(d @ d) - cr - c_ctrl)

    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.clip(numer / denom, *spec.EXPR_MSE_CLAMP))


# --------------------------------------------------------------------------
# DE-based members (brief sections 3-7)
# --------------------------------------------------------------------------

@dataclass
class _DEPair:
    real: DETable
    pred: DETable | None
    own_idx: int | None                    # this perturbation's target gene axis id
    sig_positions_real: np.ndarray         # positions in real table (sig, own-target-free)

    def adjudicable(self) -> np.ndarray:
        """Reference LFC defined, non-null, non-zero (infinite carries direction)."""
        lfc = self.real.lfc
        return ~np.isnan(lfc) & (lfc != 0)


def _build_pairs(
    cd: ContextData,
    n_control_subsample: int | None,
    seed: int,
) -> tuple[dict[str, _DEPair], np.ndarray]:
    """DE pairs + the control matrix used for them (subsampled if requested)."""
    from .de_table import control_gate

    ctrl = cd.real_ctrl
    if n_control_subsample is not None and ctrl.shape[0] > n_control_subsample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(ctrl.shape[0], size=n_control_subsample, replace=False)
        ctrl = ctrl[sel]
    gate = control_gate(normalize_cpm(ctrl, spec.DE_TARGET_SUM))

    idx = cd.gene_index()
    pairs: dict[str, _DEPair] = {}
    for name in cd.perts():
        real_t = de_table_for_group(cd.real_pert[name], ctrl, gate)
        pred_t = de_table_for_group(cd.pred_pert[name], ctrl, gate) if name in cd.pred_pert else None

        own = idx.get(name)
        is_sig = (real_t.padj < spec.DE_PADJ_THRESHOLD) & ~np.isnan(real_t.padj)
        if own is not None:
            is_sig &= real_t.genes_idx != own
        pairs[name] = _DEPair(
            real=real_t,
            pred=pred_t,
            own_idx=own,
            sig_positions_real=np.flatnonzero(is_sig),
        )
    return pairs, ctrl


def build_real_pairs(
    cd: ContextData,
    n_control_subsample: int | None,
    seed: int,
    ctrl_cpm: np.ndarray | None = None,
) -> tuple[dict[str, _DEPair], np.ndarray, np.ndarray]:
    """Candidate-independent half of _build_pairs: real DE tables only.

    Returns (pairs-with-pred-None, ctrl-used, gate). Used by exact calibration
    tuning so the expensive reference tables are computed once across the
    whole (threshold, gain, power) grid instead of once per candidate.
    Pass `ctrl_cpm` (precomputed normalize_cpm of the returned ctrl) to skip
    re-normalizing a large control matrix per call.
    """
    from .de_table import control_gate

    ctrl = cd.real_ctrl
    if n_control_subsample is not None and ctrl.shape[0] > n_control_subsample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(ctrl.shape[0], size=n_control_subsample, replace=False)
        ctrl = ctrl[sel]
    if ctrl_cpm is None:
        ctrl_cpm = normalize_cpm(ctrl, spec.DE_TARGET_SUM)
    gate = control_gate(ctrl_cpm)

    idx = cd.gene_index()
    pairs: dict[str, _DEPair] = {}
    for name in cd.perts():
        real_t = de_table_for_group(cd.real_pert[name], None, gate, ctrl_cpm=ctrl_cpm)
        own = idx.get(name)
        is_sig = (real_t.padj < spec.DE_PADJ_THRESHOLD) & ~np.isnan(real_t.padj)
        if own is not None:
            is_sig &= real_t.genes_idx != own
        pairs[name] = _DEPair(
            real=real_t,
            pred=None,
            own_idx=own,
            sig_positions_real=np.flatnonzero(is_sig),
        )
    return pairs, ctrl, gate


def attach_pred_table(
    pair: _DEPair,
    pred_counts: np.ndarray,
    gate: np.ndarray,
    ctrl_cpm: np.ndarray,
) -> None:
    """Compute and attach one candidate's predicted DE table to a pair."""
    from .de_table import de_table_for_group

    pair.pred = de_table_for_group(pred_counts, None, gate, ctrl_cpm=ctrl_cpm)


def direction_fidelity(pairs: dict[str, _DEPair]) -> float:
    """F_p = k / max(n_pred, n_real) over adjudicable calls."""
    vals = []
    for name, pair in pairs.items():
        adj = pair.adjudicable()
        real_pos_of = {g: p for p, g in enumerate(pair.real.genes_idx)}

        if pair.pred is None:
            if pair.sig_positions_real.size == 0:
                continue  # 0/0 dropped
            vals.append(0.0)
            continue

        psig = (pair.pred.padj < spec.DE_PADJ_THRESHOLD) & ~np.isnan(pair.pred.padj)
        n_pred = 0
        k = 0
        n_real = pair.sig_positions_real.size
        for p in np.flatnonzero(psig):
            g = int(pair.pred.genes_idx[p])
            rp = real_pos_of.get(g)
            if rp is None or g == pair.own_idx:
                continue
            if not adj[rp]:
                continue
            n_pred += 1
            if np.sign(pair.pred.lfc[p]) == np.sign(pair.real.lfc[rp]):
                k += 1
        if n_pred == 0 and n_real == 0:
            continue
        vals.append(k / max(n_pred, n_real))
    return float(np.mean(vals)) if vals else float("nan")


def direction_reach(pairs: dict[str, _DEPair]) -> float:
    """Deepest prefix of submission-ordered pool genes with purity >= P0."""
    vals = []
    for name, pair in pairs.items():
        pool = pair.sig_positions_real
        if pool.size == 0:
            continue
        adj = pair.adjudicable()

        if pair.pred is None:
            ordered = pool
        else:
            pred_pos_of = {g: p for p, g in enumerate(pair.pred.genes_idx)}
            keys = []
            for rp in pool:
                g = int(pair.real.genes_idx[rp])
                pp = pred_pos_of.get(g)
                if pp is None or np.isnan(pair.pred.padj[pp]):
                    keys.append((2, np.inf, np.inf, 0.0, str(g)))
                else:
                    psig = (
                        pair.pred.padj[pp] < spec.DE_PADJ_THRESHOLD
                        and not np.isnan(pair.pred.padj[pp])
                    )
                    keys.append(
                        (
                            0 if psig else 1,
                            float(pair.pred.padj[pp]),
                            float(pair.pred.pval[pp]),
                            -abs(float(pair.pred.lfc[pp])),
                            str(g),
                        )
                    )
            order_positions = sorted(range(pool.size), key=lambda j: keys[j])
            ordered = [pool[i] for i in order_positions]

        pred_pos_of = {g: p for p, g in enumerate(pair.pred.genes_idx)} if pair.pred is not None else {}
        depth = correct = best_k = 0
        for rp in ordered:
            if not adj[rp]:
                continue
            g = int(pair.real.genes_idx[rp])
            pp = pred_pos_of.get(g)
            pred_sign = np.sign(pair.pred.lfc[pp]) if pp is not None else 0.0
            if pred_sign == np.sign(pair.real.lfc[rp]):
                correct += 1
            depth += 1
            if correct / depth >= spec.REACH_PURITY_FLOOR:
                best_k = depth
        vals.append(best_k / pool.size)
    return float(np.mean(vals)) if vals else float("nan")


def sig_jaccard(pairs: dict[str, _DEPair]) -> float:
    vals = []
    for name, pair in pairs.items():
        real_ids = set(int(g) for g in pair.real.genes_idx[pair.sig_positions_real])
        if pair.pred is None:
            pred_ids: set[int] = set()
        else:
            psig = (pair.pred.padj < spec.DE_PADJ_THRESHOLD) & ~np.isnan(pair.pred.padj)
            pred_ids = {
                int(g)
                for g in pair.pred.genes_idx[psig]
                if int(g) != pair.own_idx
            }
        union = real_ids | pred_ids
        vals.append(1.0 if not union else len(real_ids & pred_ids) / len(union))
    return float(np.mean(vals)) if vals else float("nan")


def lfc_nmae(pairs: dict[str, _DEPair]) -> float:
    vals = []
    for name, pair in pairs.items():
        finite = ~np.isnan(pair.real.lfc) & np.isfinite(pair.real.lfc)
        gate = [p for p in pair.sig_positions_real if finite[p]]
        if len(gate) < spec.LFC_NMAE_MIN_GATE:
            continue
        denom = float(np.abs(pair.real.lfc[gate]).sum())
        if denom <= 0 or not np.isfinite(denom):
            continue
        if pair.pred is None:
            numer = denom  # absent predictions read as zero LFC
        else:
            pred_pos_of = {g: p for p, g in enumerate(pair.pred.genes_idx)}
            acc = 0.0
            for rp in gate:
                pp = pred_pos_of.get(int(pair.real.genes_idx[rp]))
                pv = pair.pred.lfc[pp] if pp is not None else 0.0
                if pv is None or np.isnan(pv):
                    pv = 0.0
                acc += abs(float(pv) - float(pair.real.lfc[rp]))
            numer = acc
        vals.append(numer / denom)
    result = float(np.mean(vals)) if vals else float("nan")
    return max(result, spec.LFC_NMAE_FLOOR) if np.isfinite(result) else result


# --------------------------------------------------------------------------
# Driver + anchors
# --------------------------------------------------------------------------

def evaluate_context(
    cd: ContextData,
    n_control_subsample: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Raw (unscaled) values of the six scored members for one context."""
    pairs, _ = _build_pairs(cd, n_control_subsample, seed)
    return {
        "pds_cosine": pds_cosine(cd),
        "expr_mse_unbiased_capped_norm": expr_mse_unbiased(cd),
        "de_wilcoxon_direction_fidelity_yield_raw": direction_fidelity(pairs),
        "de_wilcoxon_direction_reach_raw": direction_reach(pairs),
        "de_wilcoxon_sig_jaccard": sig_jaccard(pairs),
        "de_wilcoxon_lfc_nmae": lfc_nmae(pairs),
    }


def scale_scores(
    model: dict[str, float],
    baseline: dict[str, float],
    replicate: dict[str, float],
) -> dict[str, float]:
    """s=(u-b)/(r-b) per member plus 'overall', the unweighted mean."""
    out: dict[str, float] = {}
    for m in spec.SCORED_MEMBERS:
        span = replicate[m] - baseline[m]
        out[m] = (model[m] - baseline[m]) / span if abs(span) > 1e-12 else float("nan")
    finite = [out[m] for m in spec.SCORED_MEMBERS if np.isfinite(out[m])]
    out["overall"] = float(np.mean(finite)) if finite else float("nan")
    return out


def baseline_counts(cd: ContextData, seed: int = 0) -> dict[str, np.ndarray]:
    """Mean-response baseline through the count generator: one shared profile."""
    from .countsim import multinomial_thin

    rng = np.random.default_rng(seed + 991)
    ctrl_mean_cpm = normalize_cpm(cd.real_ctrl, spec.DE_TARGET_SUM).mean(axis=0)
    deltas = [
        normalize_cpm(cd.real_pert[nm], spec.DE_TARGET_SUM).mean(axis=0) - ctrl_mean_cpm
        for nm in cd.perts()
    ]
    mean_resp = (
        ctrl_mean_cpm + np.mean(np.stack(deltas), axis=0)
        if deltas else ctrl_mean_cpm.copy()
    )
    mult = np.divide(
        np.maximum(mean_resp, 0.0), ctrl_mean_cpm,
        out=np.ones_like(ctrl_mean_cpm), where=ctrl_mean_cpm > 0,
    )
    mult = np.clip(mult, 1e-3, 64.0)

    n = next(iter(cd.real_pert.values())).shape[0] if cd.real_pert else spec.CELLS_PER_PERT
    pick = rng.choice(cd.real_ctrl.shape[0], size=n, replace=True)
    gen = multinomial_thin(cd.real_ctrl[pick], mult, rng)
    return {name: gen for name in cd.perts()}


def baseline_anchor(
    cd: ContextData,
    n_control_subsample: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    cd_b = ContextData(
        gene_names=cd.gene_names,
        real_ctrl=cd.real_ctrl,
        real_pert=cd.real_pert,
        pred_pert=baseline_counts(cd, seed=seed),
    )
    return evaluate_context(cd_b, n_control_subsample, seed)


def replicate_anchor(
    cd: ContextData,
    n_splits: int = spec.N_REPLICATE_SPLITS,
    base_seed: int = spec.REPLICATE_BASE_SEED,
    n_control_subsample: int | None = None,
) -> dict[str, float]:
    """Split-half replicate anchor r, averaged over seeded disjoint splits."""
    per_member: dict[str, list[float]] = {m: [] for m in spec.SCORED_MEMBERS}
    for s in range(n_splits):
        rng = np.random.default_rng(base_seed + s)
        pred: dict[str, np.ndarray] = {}
        real: dict[str, np.ndarray] = {}
        for name, cells in cd.real_pert.items():
            perm = rng.permutation(cells.shape[0])
            half = cells.shape[0] // 2
            pred[name] = cells[perm[:half]]
            real[name] = cells[perm[half:]]
        perm = rng.permutation(cd.real_ctrl.shape[0])
        chalf = cd.real_ctrl.shape[0] // 2
        cd_h = ContextData(cd.gene_names, cd.real_ctrl[perm[chalf:]], real, pred)
        raw = evaluate_context(cd_h, n_control_subsample, seed=base_seed + s)
        for m in spec.SCORED_MEMBERS:
            if np.isfinite(raw.get(m, np.nan)):
                per_member[m].append(raw[m])
    return {m: float(np.mean(v)) if v else float("nan") for m, v in per_member.items()}
