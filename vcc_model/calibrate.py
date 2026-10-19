"""Metric-aligned post-processing of predicted LFC vectors.

Why this exists (vcc2026 scoring mechanics):
    Four of the six scored members are functions of a Wilcoxon rank-sum test
    between OUR emitted cells and the REFERENCE control cells:
        fid   = correct-sign calls / max(n_called_adjudicable, n_real)
        reach = deepest prefix of ref-significant genes whose predicted
                ranking keeps >= 0.9 sign purity
        jac   = |sig_real ^ sig_pred| / |union|
        nmae  = |lfc_hat - lfc| / |lfc| over ref-significant genes
    With 400 predicted cells vs ~18k control cells, statistical power is high,
    so ANY consistent nonzero shift in our emitted counts becomes "significant".
    False positives therefore come almost entirely from NOISE IN THE PREDICTED
    LFC VECTOR itself -- genes where the transfer prior/residual net emits a
    small spurious shift. Those calls cost jaccard (invented genes) and reach
    (wrong-sign prefixes), while shrinking magnitudes costs nmae/fidelity.

    The fix is classic shrinkage applied where it matters:
        lfc' = sign(lfc) * gain * max(|lfc| - tau, 0) ** power
    which zeroes sub-threshold noise, preserves large movers, and can sharpen
    (power > 1) or flatten (power < 1) the response profile.

Tuning:
    `tune()` grid-searches (tau, gain, power) to maximize a fast surrogate of
    the six-member aggregate computed on held-out source folds where truth is
    known. `tune_exact()` instead runs the full count-generation + local
    six-metric evaluation for each candidate -- slower but identical to what
    the leaderboard measures (up to anchor scaling).

Defaults reflect observed VCC-scale biology: true CRISPRi LFCs concentrate
below 0.25 in absolute value; significance at 400-vs-18k cells triggers around
|lfc| >~ 0.1; strong movers sit above ~0.5.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np


# Genes whose |true LFC| exceeds this count as "truly responding" in surrogates.
STRONG_TRUE_LFC = 0.30
# A predicted call must exceed this (post-transform) to be counted as called.
CALL_THRESHOLD = 0.10
# Surrogate weights: (fidelity, jaccard, nmae-inverted). PDS/reach correlate
# with these through shared geometry; exact tuning covers the rest.
_W_FID, _W_JAC, _W_NMAE = 0.35, 0.40, 0.25

# Official vcc2026 anchors (cell-eval2 0.15.0 reference bundles, rule_version 3;
# midpoints over contexts A/B/C from the metrics brief section 8). Tuning
# candidates by the OFFICIAL scaled mean s=(u-b)/(r-b) aligns the search with
# the leaderboard aggregation -- raw-value means weight members arbitrarily
# (reach has an official span of ~0.90 while nmae's is ~-0.60; treating them
# equally optimizes a different objective than the one scored).
OFFICIAL_ANCHORS: dict[str, tuple[float, float]] = {
    # member: (baseline b, replicate r)
    "pds": (0.500, 0.956),
    "mse": (0.989, 0.037),
    "fid": (0.514, 0.814),
    "reach": (0.072, 0.968),
    "jac": (0.029, 0.400),
    "nmae": (1.001, 0.400),
}


@dataclass(frozen=True)
class DECalibration:
    threshold: float = 0.0
    gain: float = 1.0
    power: float = 1.0
    # Optional per-target adaptive floor: effective threshold is
    # max(threshold, q-quantile of |lfc| within the vector). Programs vary hugely
    # in overall magnitude (essential vs non-essential targets); a global absolute
    # tau alone over-calls strong programs and under-calls weak ones.
    q_threshold: float = 0.0
    # Per-gene gain multipliers (e.g. from consistency or ESM). When set,
    # effective gain for gene j is gain * per_gene_gain[j]. None = uniform.
    per_gene_gain: dict[str, float] | None = None

    def transform(self, lfc: np.ndarray, gene: str | None = None) -> np.ndarray:
        mag = np.abs(np.asarray(lfc, dtype=np.float64))
        if self.q_threshold > 0.0 and mag.size:
            tau_adaptive = float(np.quantile(mag, self.q_threshold))
        else:
            tau_adaptive = 0.0
        shifted = mag - max(self.threshold, tau_adaptive)
        eff_gain = self.gain
        if gene is not None and self.per_gene_gain is not None:
            eff_gain = self.gain * float(self.per_gene_gain.get(gene, 1.0))
        mag2 = np.clip(shifted, 0.0, None) ** self.power * eff_gain
        return (np.sign(lfc) * mag2).astype(np.float32)

    def as_dict(self) -> dict:
        d: dict = {
            "threshold": float(self.threshold),
            "gain": float(self.gain),
            "power": float(self.power),
            "q_threshold": float(self.q_threshold),
        }
        if self.per_gene_gain is not None:
            d["per_gene_gain"] = {k: float(v) for k, v in self.per_gene_gain.items()}
        return d

    @classmethod
    def from_per_gene_consistency(
        cls,
        base_gain: float,
        threshold: float = 0.0,
        q_threshold: float = 0.0,
        power: float = 1.0,
        consistency: dict[str, float] | None = None,
        default_cons: float = 0.45,
        scale: float = 0.5,
    ) -> "DECalibration":
        """Build a calibration where gain is modulated by per-gene consistency.

        gain_g = base_gain * (1 - scale + scale * cons_g)
        High-consistency genes (conserved programs) get full gain; low-consistency
        get reduced gain, breaking the PDS-nmae tradeoff that a single scalar incurs.
        """
        pg = None
        if consistency is not None:
            pg = {g: float(1.0 - scale + scale * max(0.0, min(1.0, v))) for g, v in consistency.items()}
            # Unseen genes fall back to default conservation
            # (handled by caller via .get default 1.0; we store default explicitly for transparency)
            pg["_default"] = float(1.0 - scale + scale * default_cons)
        return cls(threshold=threshold, gain=base_gain, power=power, q_threshold=q_threshold, per_gene_gain=pg)


def _surrogate(pred: np.ndarray, truth: np.ndarray) -> float:
    """Aggregate proxy for the four DE members on one (context, gene) pair."""
    called = np.abs(pred) > CALL_THRESHOLD
    strong = np.abs(truth) > STRONG_TRUE_LFC

    n_strong = int(strong.sum())
    if n_strong == 0:
        # Nothing truly responds: reward empty call sets (jaccard=1 case).
        return 1.0 if not called.any() else _W_JAC * 0.0 + (1 - _W_JAC)

    # Fidelity proxy: directional precision x coverage, capped like F_p.
    both = called & strong
    k = float((np.sign(pred[both]) == np.sign(truth[both])).sum())
    n_pred = int((called & strong).sum()) + int((called & ~strong).sum()) * 0
    # Called-but-weak genes still enter n_pred in the real metric via padj;
    # approximate with all called genes that are adjudicable (truth != 0).
    adj = truth != 0
    n_pred = int((called & adj).sum())
    fid = (k / n_pred) * min(1.0, n_pred / n_strong) if n_pred else 0.0

    # Jaccard proxy between call set and strong-truth set.
    union = int((called | strong).sum())
    inter = int(both.sum())
    jac = inter / union if union else 1.0

    # NMAE proxy restricted to strong-truth genes (gate-like).
    gate = strong
    denom = float(np.abs(truth[gate]).sum())
    nmae_err = (
        float(np.abs(pred[gate] - truth[gate]).sum() / denom) if denom > 1e-9 else 1.0
    )

    return _W_FID * fid + _W_JAC * jac + _W_NMAE * max(0.0, 1.0 - min(nmae_err, 2.0) / 2.0)


def tune(
    pred_lfcs: dict[tuple[str, str], np.ndarray],
    true_lfcs: dict[tuple[str, str], np.ndarray],
    thresholds: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
    gains: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5),
    powers: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> tuple[DECalibration, dict]:
    """Grid-search calibration maximizing mean surrogate across pairs."""
    keys = [k for k in pred_lfcs if k in true_lfcs]
    if not keys:
        return DECalibration(), {"skipped": "no eval pairs"}

    truths = {k: np.asarray(true_lfcs[k], dtype=np.float64) for k in keys}
    preds_raw = {k: np.asarray(pred_lfcs[k], dtype=np.float64) for k in keys}

    best, best_cfg, curve = -np.inf, None, []
    for tau, gain, power in itertools.product(thresholds, gains, powers):
        cal = DECalibration(threshold=tau, gain=gain, power=power)
        score = float(
            np.mean([_surrogate(cal.transform(preds_raw[k]), truths[k]) for k in keys])
        )
        curve.append({**cal.as_dict(), "score": score})
        if score > best:
            best, best_cfg = score, cal
    return best_cfg, {"score": best, "n_pairs": len(keys), "grid": curve}


def _candidate_grid(
    thresholds: tuple[float, ...],
    gains: tuple[float, ...],
    powers: tuple[float, ...],
    q_thresholds: tuple[float, ...],
) -> list[DECalibration]:
    return [
        DECalibration(threshold=t, gain=g, power=p, q_threshold=q)
        for t, g, p, q in itertools.product(thresholds, gains, powers, q_thresholds)
    ]


def _normalized_surrogate_screen(
    pred_lfcs: dict,
    true_lfcs: dict,
    cands: list[DECalibration],
) -> list[tuple[float, DECalibration]]:
    """Scale-normalized LFC-space ranking of candidates (cheap round 1).

    Raw surrogates are blind to global scale: a candidate that shrinks all
    predictions below the absolute call threshold looks identical to one that
    destroys signal quality. Normalizing both prediction and truth by their own
    strong-effect magnitude makes the fixed CALL_THRESHOLD comparable across
    candidates, which is enough to pick a shortlist; the exact count-level
    round then decides.
    """
    keys = sorted(set(pred_lfcs) & set(true_lfcs))
    out = []
    for cal in cands:
        acc = []
        for k in keys:
            t = np.asarray(true_lfcs[k], dtype=np.float64)
            strong = np.abs(t) > STRONG_TRUE_LFC
            scale = float(np.median(np.abs(t[strong]))) if strong.any() else float(np.median(np.abs(t))) or 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            p = cal.transform(pred_lfcs[k]) / scale
            acc.append(_surrogate(p, t / scale))
        out.append((float(np.mean(acc)) if acc else -np.inf, cal))
    return out


def tune_exact(
    pred_lfcs: dict[tuple[str, str], np.ndarray],
    true_lfcs: dict[tuple[str, str], np.ndarray],
    ctrl_counts_by_context: dict[str, np.ndarray],
    gene_names: list[str],
    thresholds: tuple[float, ...] = (0.0, 0.05, 0.10),
    gains: tuple[float, ...] = (0.25, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.1, 3.7, 4.4, 5.2),
    powers: tuple[float, ...] = (1.0,),
    q_thresholds: tuple[float, ...] = (0.0, 0.45, 0.7, 0.88),
    seed: int = 0,
    final_pairs: int = 24,
    final_cells: int = 260,
    shortlist: int = 5,
    official_span_weighting: bool = True,
    verbose: bool = True,
) -> tuple[DECalibration, dict]:
    """Screen + exact-refine tuning against the true six-metric aggregate.

    Round 1 ranks the full grid instantly with a scale-normalized LFC
    surrogate (raw surrogates cannot see global scale, which is exactly the
    degree of freedom being tuned). Round 2 evaluates the top `shortlist`
    through the real pipeline -- emit Poisson-thinned counts per candidate and
    score all six members (raw values) on synthetic truth built by thinning
    control cells with the TRUE LFCs. Reference DE tables are built once per
    candidate batch, so the round costs ~shortlist x pairs Wilcoxon tables.

    The grid spans LOW gains on purpose: cross-context transfer slopes measured
    on real panels are small (~0.07-0.3), so the optimal end-to-end scale can
    be far below unity -- a decision only a DE-member-aware objective can make.
    """
    from .countsim import poisson_thin as _pthin
    from .countsim import normalize_cpm as _ncpm
    from .metrics import (
        ContextData,
        attach_pred_table,
        build_real_pairs,
        direction_fidelity,
        direction_reach,
        expr_mse_unbiased,
        lfc_nmae,
        pds_cosine,
        sig_jaccard,
    )

    keys = sorted(k for k in pred_lfcs if k in true_lfcs)
    if not keys or not ctrl_counts_by_context:
        return DECalibration(), {"skipped": "no eval pairs"}
    if not all(isinstance(k, tuple) and len(k) == 2 for k in keys):
        raise ValueError(
            "tune_exact expects (context, target) tuple keys; got "
            f"{type(keys[0]).__name__} — wrap plain gene keys as (ctx, gene)"
        )

    def pick_balanced(n_pairs: int) -> list[tuple[str, str]]:
        by_ctx: dict[str, list[tuple[str, str]]] = {}
        for k in keys:
            by_ctx.setdefault(k[0], []).append(k)
        per_ctx = max(1, n_pairs // max(len(by_ctx), 1))
        chosen: list[tuple[str, str]] = []
        for ctx in sorted(by_ctx):
            chosen.extend(by_ctx[ctx][:per_ctx])
        return chosen

    def run_round(cands: list[DECalibration], n_pairs: int, n_cells: int):
        chosen = pick_balanced(n_pairs)
        ctx_packs = {}
        for ctx in sorted({c for c, _ in chosen}):
            ctrl_full = ctrl_counts_by_context[ctx]
            n_ctrl = min(ctrl_full.shape[0], 2200)
            sel = np.random.default_rng(seed).choice(ctrl_full.shape[0], size=n_ctrl, replace=False)
            ctrl = np.asarray(
                ctrl_full[sel].todense() if hasattr(ctrl_full, "todense") else ctrl_full[sel],
                dtype=np.float64,
            )
            pairs_here = [(c, g) for c, g in chosen if c == ctx]
            rng_t = np.random.default_rng(seed + 4242)
            pick_t = rng_t.choice(n_ctrl, size=n_cells, replace=n_cells > n_ctrl)
            base_truth = ctrl[pick_t]
            real_map = {}
            for _, gene in pairs_here:
                tl = np.clip(true_lfcs[(ctx, gene)], -6, 6)
                real_map[gene] = _pthin(base_truth, np.exp2(tl), rng_t)
            cd_real = ContextData(
                gene_names=np.asarray(gene_names, dtype=str),
                real_ctrl=ctrl,
                real_pert=real_map,
                pred_pert={},
            )
            rpairs, ctrl_used, gate = build_real_pairs(cd_real, None, seed)
            ctx_packs[ctx] = {
                "pairs": pairs_here,
                "rpairs": rpairs,
                "gate": gate,
                "ctrl": ctrl_used,
                "ctrl_cpm": _ncpm(ctrl_used, 1_000_000.0),
                "truth_cells": real_map,
                "gene_names_arr": np.asarray(gene_names, dtype=str),
                "base_pool": ctrl,
            }

        results = []
        for ci, cal in enumerate(cands):
            member_scores = []
            for ctx, pack in ctx_packs.items():
                rng = np.random.default_rng(seed + 1000 * ci + 7)
                pick = rng.choice(pack["base_pool"].shape[0], size=n_cells,
                                  replace=n_cells > pack["base_pool"].shape[0])
                base_pred = pack["base_pool"][pick]
                pred_map = {}
                for _, gene in pack["pairs"]:
                    lfc = np.clip(cal.transform(pred_lfcs[(ctx, gene)]), -6, 6)
                    pred_map[gene] = _pthin(base_pred, np.exp2(lfc), rng)
                for name, pair in pack["rpairs"].items():
                    attach_pred_table(pair, pred_map[name], pack["gate"], pack["ctrl_cpm"])
                vals = {
                    "fid": direction_fidelity(pack["rpairs"]),
                    "reach": direction_reach(pack["rpairs"]),
                    "jac": sig_jaccard(pack["rpairs"]),
                    "nmae": lfc_nmae(pack["rpairs"]),
                }
                cd_eval = ContextData(
                    gene_names=pack["gene_names_arr"],
                    real_ctrl=pack["ctrl"],
                    real_pert=pack["truth_cells"],
                    pred_pert=pred_map,
                )
                vals["pds"] = pds_cosine(cd_eval)
                vals["mse"] = expr_mse_unbiased(cd_eval)
                finite = [v for v in vals.values() if np.isfinite(v)]
                if official_span_weighting:
                    scaled_vals = []
                    for mname, v in vals.items():
                        if not np.isfinite(v):
                            continue
                        b_off, r_off = OFFICIAL_ANCHORS[mname]
                        span = r_off - b_off
                        if abs(span) < 1e-9:
                            continue
                        sv = (float(v) - b_off) / span
                        if mname == "mse":
                            sv = float(np.clip(sv, 0.0, 1.0))  # scored clamp
                        scaled_vals.append(sv)
                    if len(scaled_vals) >= 5:
                        member_scores.append(float(np.mean(scaled_vals)))
                elif len(finite) >= 5:
                    member_scores.append(float(np.mean(finite)))
            score = float(np.mean(member_scores)) if member_scores else -np.inf
            results.append((score, cal))
        return results

    grid = _candidate_grid(thresholds, gains, powers, q_thresholds)
    t0 = time.time()
    screened = _normalized_surrogate_screen(pred_lfcs, true_lfcs, grid)
    screened.sort(key=lambda t: -t[0])
    cands2 = [c for _, c in screened[:shortlist]]
    round2 = run_round(cands2, final_pairs, final_cells)
    round2.sort(key=lambda t: -t[0])
    best_score, best_cal = round2[0]
    if verbose:
        print(f"  [calibrate-exact] screened {len(grid)} via normalized surrogate "
              f"({time.time() - t0:.0f}s); exact top-{shortlist}; "
              f"best={best_cal.as_dict()} score={best_score:.4f}", flush=True)
    info = {
        "score": float(best_score),
        "grid_round1": [{**c.as_dict(), "score": s} for s, c in screened[:12]],
        "grid_round2": [{**c.as_dict(), "score": s} for s, c in round2],
        "n_screen": len(grid),
    }
    return best_cal, info
