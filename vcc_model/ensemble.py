"""Hybrid zero-shot predictor: statistical transfer prior + neural residual
[+ optional Stack ICL delta], decoded into integer counts by multinomial
thinning of real control cells, then metric-aligned DE-calibration.

Pipeline per (context, perturbation):
    1. basal vector  = log mean-CPM profile of the context's control cells
    2. LFC prediction = w_stat * prior_lfc + w_resid * residual_net(...)
                       + w_stack * stack_icl_delta(...)
       (weights chosen by leave-one-context-out proxy validation)
    3. calibration   = soft-threshold + rescale (DECalibration) tuned against
       held-out source contexts for the four Wilcoxon-based members
    4. counts: sample control cells, resample each cell's molecules with
       per-gene multipliers 2^LFC. This preserves library sizes, sparsity
       (density-cap safe), and cell-to-cell covariance while shifting the
       mean profile exactly where the metrics look.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import spec
from .calibrate import DECalibration
from .countsim import log_mean_cpm, normalize_cpm, poisson_thin
from .prior import LFCTransferPrior
from .residual import ResidualConfig, ResidualLFCModel

# (w_stat, w_resid, w_stack) candidates for the blend:
#   pred = w_stat * prior + w_resid * net + w_stack * stack
WEIGHT_GRID: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.00, 0.00),
    (1.00, 0.35, 0.00),
    (1.00, 0.70, 0.00),
    (1.00, 1.00, 0.00),
    (0.85, 0.70, 0.00),
    (0.70, 1.00, 0.00),
    (0.50, 1.00, 0.00),
    # Stack-enabled candidates are appended at fit time when a stack_fn exists.
)
STACK_WEIGHT_GRID: tuple[float, ...] = (0.25, 0.5, 1.0)
# Shared-program mixing fractions scanned during weight selection: the shared
# component adds a common vector to every prediction, which can flatten PDS
# ranking toward the 0.5 baseline; the PDS-aware proxy arbitrates.
SHARED_MIX_GRID: tuple[float, ...] = (0.0, 0.25)

LFC_CLIP = 6.0


def lfc_proxy_aggregate(pred: np.ndarray, truth: np.ndarray) -> float:
    """Fast LFC-space proxy correlated with the six scored members.

    Combines cosine similarity, top-gene sign accuracy, normalized MAE on the
    strongest genes, and a magnitude-ratio penalty. Used only for cheap weight
    selection; honest numbers come from metrics.evaluate_context.
    """
    pn = float(np.linalg.norm(pred))
    tn = float(np.linalg.norm(truth))
    cos = float(pred @ truth) / (pn * tn) if pn > 0 and tn > 0 else 0.0

    k = int(np.clip((np.abs(truth) > 0.25).sum(), 10, 50))
    top = np.argsort(np.abs(truth))[-k:]
    sign_acc = float((np.sign(pred[top]) == np.sign(truth[top])).mean())
    denom = float(np.abs(truth[top]).sum())
    nmae = float(np.abs(pred[top] - truth[top]).sum() / denom) if denom > 0 else 1.0
    mag_pen = abs(np.log((pn + 1e-9) / (tn + 1e-9)))

    return (
        0.40 * max(cos, 0.0)
        + 0.30 * sign_acc
        + 0.20 * max(0.0, 1.0 - nmae)
        + 0.10 * float(np.exp(-mag_pen))
    )


PANEL_IDX: dict[str, int] = {}


def _panel_effect_matrix(vectors: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Stack LFC vectors with ALL panel-target coordinates removed (PDS space)."""
    g0 = next(iter(vectors))
    dim = len(vectors[g0])
    mask = np.ones(dim, dtype=bool)
    for g in names:
        gi = PANEL_IDX.get(g)
        if gi is not None:
            mask[gi] = False
    keep = np.flatnonzero(mask)
    return np.stack([np.asarray(vectors[nm], dtype=np.float64)[keep] for nm in names])


def pds_proxy_fast(preds: dict[str, np.ndarray], truths: dict[str, np.ndarray],
                   gene_names: list[str]) -> float:
    """Vectorized official-style PDS computed on LFC vectors.

    Ranks each target's predicted effect among all measured effects of the
    context by cosine distance with panel-target coordinates removed, exactly
    like the scored metric but in LFC space. Directly optimizes what
    `pds_cosine` measures, which the per-pair proxy is blind to: a blend can
    have great per-pair cosine yet emit vectors so shared that every row ties
    at rank-median (= the 0.5 baseline).
    """
    global PANEL_IDX
    if not PANEL_IDX or len(PANEL_IDX) != len(gene_names):
        PANEL_IDX = {g: i for i, g in enumerate(gene_names)}
    names = sorted(set(preds) & set(truths))
    n = len(names)
    if n < 2:
        return float("nan")
    real_mat = _panel_effect_matrix(truths, names)
    pred_mat = _panel_effect_matrix(preds, names)
    nr = np.linalg.norm(real_mat, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    real_unit = real_mat / nr
    ph = np.linalg.norm(pred_mat, axis=1, keepdims=True)
    zero_pred = (ph[:, 0] == 0)
    ph[ph == 0] = 1.0
    sims = pred_mat / ph @ real_unit.T          # cosine similarity pred_i vs real_j
    dists = 1.0 - sims
    scores = np.full(n, 0.5)
    d_self = dists[np.arange(n), np.arange(n)]
    k = (dists < d_self[:, None]).sum(axis=1) + 0.5 * ((dists == d_self[:, None]).sum(axis=1) - 1)
    ok = ~zero_pred
    scores[ok] = 1.0 - k[ok] / (n - 1)
    return float(scores.mean())


@dataclass
class HybridPredictor:
    gene_names: list[str]
    prior: LFCTransferPrior | None = None
    residual: ResidualLFCModel | None = None
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0)  # (stat, resid, stack)
    stack_fn: Callable[[str, np.ndarray], np.ndarray] | None = None
    calibration: DECalibration | None = field(default=None)
    fit_info: dict | None = None

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls,
        lfc_pairs: dict[tuple[str, str], np.ndarray],
        basals: dict[str, np.ndarray],
        gene_names: list[str],
        residual_cfg: ResidualConfig | None = None,
        fit_residual: bool = True,
        select_weights: bool = True,
        weight_grid: tuple = WEIGHT_GRID,
        stack_fn: Callable[[str, np.ndarray], np.ndarray] | None = None,
        prior_kwargs: dict | None = None,
        pair_n: dict[tuple[str, str], int] | None = None,
        source_weights: dict[str, float] | None = None,
        min_cells: int = 0,
    ) -> "HybridPredictor":
        """Fit prior (+residual if enough data) on source contexts.

        Stores sanitize their LFC matrix on load (finite, clipped to +-8),
        so pairs are passed by reference — no per-vector copies.
        """
        nested: dict[str, dict[str, np.ndarray]] = {}
        for (c, g), v in lfc_pairs.items():
            nested.setdefault(c, {})[g] = np.asarray(v)
        t0 = time.time()
        pk = prior_kwargs or {}
        # Pre-load ESM if requested so prior.fit can compute centroids
        esm_pre = None
        esm_path = pk.get("esm_path") or pk.get("esm_embeddings_path")
        if esm_path:
            try:
                from .protein import load_esm_embeddings
                esm_pre = load_esm_embeddings(esm_path)
            except Exception:
                esm_pre = None
        prior = LFCTransferPrior(
            similarity_temperature=pk.get("similarity_temperature", 0.1),
            default_conservation=pk.get("default_conservation", 0.45),
            james_stein=pk.get("james_stein", False),
            james_stein_dim=pk.get("james_stein_dim", 32),
            james_stein_strength=pk.get("james_stein_strength", 1.0),
            basal_gate_cpm=pk.get("basal_gate_cpm", 0.0),
            shared_mix=pk.get("shared_mix", 0.25),
            use_alpha=pk.get("use_alpha", True),
            beta_deshared=pk.get("beta_deshared", 0.0),
            denoise_rank=pk.get("denoise_rank", 0),
            per_gene_gain=pk.get("per_gene_gain"),
            gene_conditioned=pk.get("gene_conditioned", False),
            zero_unseen=pk.get("zero_unseen", False),
            use_consistency=pk.get("use_consistency", True),
            esm_embeddings=esm_pre,
        ).fit(nested, basals, gene_names, pair_n=pair_n, source_weights=source_weights, min_cells=min_cells)
        # Wire ESM protein embeddings if requested (post-fit fallback for centroid recompute)
        if esm_path and esm_pre is None:
            try:
                from .protein import load_esm_embeddings
                esm = load_esm_embeddings(esm_path)
                if esm:
                    prior.esm_embeddings = esm
                    prior.gene_conditioned = bool(pk.get("gene_conditioned", True))
                    # Recompute centroids now that esm is set
                    prior._esm_context_centroids = {}
                    for c in prior.contexts:
                        genes_here = [g for g in prior.lfc.get(c, {}) if g in esm and np.linalg.norm(esm[g]) > 1e-6]
                        if len(genes_here) >= 3:
                            mat = np.stack([esm[g]/max(np.linalg.norm(esm[g]),1e-9) for g in genes_here])
                            centroid = mat.mean(axis=0)
                            centroid /= max(np.linalg.norm(centroid),1e-9)
                            prior._esm_context_centroids[c] = centroid.astype(np.float32)
                    print(f"[fit] ESM wired: {len(esm)} genes, centroids for {len(prior._esm_context_centroids)} contexts")
            except Exception as e:
                print(f"[fit][warn] ESM wiring failed: {e}")
        # Fit denoiser if requested via prior_kwargs (kept separate for explicit logging)
        if pk.get("denoise_rank", 0):
            try:
                prior.fit_denoiser(rank=int(pk["denoise_rank"]))
            except Exception:
                pass
        print(f"[fit] transfer prior built in {time.time() - t0:.0f}s "
              f"({len(prior.contexts)} contexts, global_alpha={prior.global_alpha:.3f})",
              flush=True)
        info: dict = {"n_pairs": len(lfc_pairs), "n_source_contexts": len(prior.contexts)}

        residual: ResidualLFCModel | None = None
        if fit_residual and len(lfc_pairs) >= 12:
            cfg = residual_cfg or ResidualConfig()
            # Wire ESM into residual if prior has ESM and residual requested
            if esm_pre is not None and cfg.esm_dim == 0:
                try:
                    cfg.esm_dim = int(next(iter(esm_pre.values())).shape[0])
                    cfg.esm_path = str(esm_path)
                    print(f"[fit] residual ESM enabled dim {cfg.esm_dim}")
                except Exception:
                    pass
            residual = ResidualLFCModel(cfg)
            info["residual"] = residual.fit(prior, lfc_pairs, basals, gene_names)
            print(f"[fit] residual net trained: {info['residual']}", flush=True)
        else:
            info["residual"] = {"skipped": "disabled or too few pairs"}

        grid = weight_grid
        if stack_fn is not None:
            grid = grid + tuple(
                (s * 0.5 + 0.15, r * 0.75, k)
                for (s, r, _) in weight_grid
                for k in STACK_WEIGHT_GRID
                if s > 0 or r > 0
            )

        model = cls(
            gene_names=list(gene_names),
            prior=prior,
            residual=residual,
            stack_fn=stack_fn,
            fit_info=info,
        )
        if select_weights and residual is not None and residual.net is not None:
            model.weights = model._select_weights_loco(basals, grid)
        elif select_weights and stack_fn is not None:
            model.weights = model._select_weights_loco(basals, grid)
        elif residual is not None and residual.net is not None and select_weights:
            model.weights = (1.0, 0.7, 0.0)
        else:
            model.weights = (1.0, 0.0, 0.0)
        info["weights"] = {"stat": model.weights[0], "resid": model.weights[1], "stack": model.weights[2]}
        return model

    def _fit_submodel(self, lfc_pairs, basals, cfg: ResidualConfig | None) -> "HybridPredictor":
        return HybridPredictor.fit(
            lfc_pairs,
            basals,
            self.gene_names,
            residual_cfg=cfg,
            select_weights=False,
            stack_fn=self.stack_fn,
        )

    def _select_weights_loco(
        self,
        basals: dict[str, np.ndarray],
        weight_grid: tuple,
        pds_weight: float = 0.5,
    ) -> tuple[float, float, float]:
        """Leave-one-context-out selection of the blend weights.

        For every source context: refit prior+residual without it, then score
        each candidate's blended LFCs against that context's observed LFCs with
        (1 - pds_weight) * per-pair proxy + pds_weight * panel PDS proxy.
        The PDS term matters because a blend can score well per pair while its
        vectors are too correlated with each other to rank self first -- which
        is exactly what the scored pds_cosine punishes to 0.5.
        """
        all_pairs: dict[tuple[str, str], np.ndarray] = {
            (c, g): v for c in self.prior.contexts for g, v in self.prior.lfc[c].items()
        }
        contexts = sorted(basals)
        base_cfg = self.residual.cfg if self.residual is not None else ResidualConfig()
        cfg = ResidualConfig(
            seed=base_cfg.seed + 17,
            epochs=min(base_cfg.epochs, 80),
            max_pairs=min(base_cfg.max_pairs, 8000),
        )
        print(f"[weights] LOCO selection over {len(contexts)} folds "
              f"(grid={len(weight_grid)} x mixes={len(SHARED_MIX_GRID)})", flush=True)
        combos: list[tuple[tuple[float, float, float], float]] = [
            (w, sm) for w in weight_grid for sm in SHARED_MIX_GRID
        ]
        scores: dict[tuple, list[float]] = {combo: [] for combo in combos}

        for fi, held in enumerate(contexts):
            t_fold = time.time()
            sub_pairs = {k: v for k, v in all_pairs.items() if k[0] != held}
            if len(sub_pairs) < 12:
                continue
            sub_basals = {c: b for c, b in basals.items() if c != held}
            try:
                sub = self._fit_submodel(sub_pairs, sub_basals, cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[weights] fold {held}: submodel failed ({exc})", flush=True)
                continue
            truths = [(g, v) for (c, g), v in all_pairs.items() if c == held]
            truth_map = {g: v for g, v in truths}
            fold_genes = [g for g, _ in truths]
            # Stable caps: the O(n^2) PDS proxy and the per-pair proxy only
            # need a sample to rank blend candidates reliably (the GW fold has
            # ~10k truths; scoring all of them x every combo is pure waste).
            rng_cap = np.random.default_rng(0)
            pds_names = list(fold_genes)
            if len(pds_names) > 120:
                pds_names = sorted(rng_cap.choice(pds_names, size=120, replace=False).tolist())
            proxy_genes = list(fold_genes)
            if len(proxy_genes) > 400:
                proxy_genes = sorted(rng_cap.choice(proxy_genes, size=400, replace=False).tolist())

            # Component cache: the prior output depends on shared_mix only;
            # the residual input depends on that prior output. Neither depends
            # on (w_stat, w_resid), so compute each ONCE per (gene, mix) and
            # blend arithmetically across the weight grid.
            orig_mix = sub.prior.shared_mix
            comp: dict[tuple[float, str], tuple[np.ndarray, np.ndarray | None]] = {}
            for sm in SHARED_MIX_GRID:
                sub.prior.shared_mix = sm
                pv_map = sub.prior.predict_many(fold_genes, basals[held])
                rv_map = (
                    sub.residual.predict_many(pv_map, basals[held])
                    if sub.residual is not None and sub.residual.net is not None
                    else {g: None for g in fold_genes}
                )
                for g in fold_genes:
                    comp[(sm, g)] = (pv_map[g], rv_map[g])
            sub.prior.shared_mix = orig_mix

            def _blend(w3: tuple[float, float, float], sm: float, g: str) -> np.ndarray:
                s, r, k = w3
                pv, rv = comp[(sm, g)]
                out = s * pv
                if r > 0.0 and rv is not None:
                    out = out + r * rv
                if k > 0.0 and sub.stack_fn is not None:
                    out = out + k * np.asarray(sub.stack_fn(g, basals[held]), dtype=np.float64)
                return out

            for (w, sm) in combos:
                vals = [
                    lfc_proxy_aggregate(_blend(w, sm, g), truth_map[g])
                    for g in proxy_genes
                ]
                pair_score = float(np.mean(vals))
                pred_map = {g: _blend(w, sm, g) for g in pds_names}
                pds_s = pds_proxy_fast(pred_map, {g: truth_map[g] for g in pds_names},
                                       self.gene_names)
                combined = (
                    (1.0 - pds_weight) * pair_score + pds_weight * pds_s
                    if np.isfinite(pds_s) else pair_score
                )
                scores[(w, sm)].append(float(combined))
            del comp, sub
            best_so_far = max(
                (float(np.mean(v)) for v in scores.values() if v), default=float("nan")
            )
            print(f"[weights] fold {fi + 1}/{len(contexts)} ({held}) done "
                  f"in {time.time() - t_fold:.0f}s; best-so-far={best_so_far:.4f}",
                  flush=True)

        best_combo: tuple = ((1.0, 0.0, 0.0), 0.25)
        best_s = -np.inf
        detail = {}
        for combo, vals in scores.items():
            if not vals:
                continue
            s = float(np.mean(vals))
            detail[f"{combo[0]}|mix={combo[1]}"] = s
            if s > best_s + 1e-9:
                best_s, best_combo = s, combo
        (best_w, best_sm) = best_combo
        self.prior.shared_mix = best_sm
        if getattr(self, "fit_info", None) is not None:
            self.fit_info["weight_selection"] = {
                "chosen": list(best_w), "shared_mix": best_sm, "proxy": best_s,
                "grid": detail, "pds_weight": pds_weight,
            }
        return best_w

    # ------------------------------------------------------------ predict
    def predict_lfc(self, gene: str, basal_vec: np.ndarray) -> np.ndarray:
        """Calibrated final LFC prediction (what gets decoded into counts)."""
        raw = self.blend_lfc(gene, basal_vec, weights=self.weights)
        if self.calibration is not None:
            raw = self.calibration.transform(raw, gene=gene)
        return raw

    def blend_lfc(
        self,
        gene: str,
        basal_vec: np.ndarray,
        weights: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        s, r, k = weights if weights is not None else self.weights
        p = self.prior.predict_lfc(gene, basal_vec)
        out = s * p
        if r > 0.0 and self.residual is not None:
            out = out + r * self.residual.predict(p, basal_vec, gene)
        if k > 0.0 and self.stack_fn is not None:
            out = out + k * np.asarray(self.stack_fn(gene, basal_vec), dtype=np.float64)
        return np.asarray(out)

    # ---------------------------------------------------------- generation
    def generate_cells(
        self,
        ctrl_counts: np.ndarray,
        lfc: np.ndarray,
        n_cells: int,
        rng: np.random.Generator,
        method: str = "poisson",
    ) -> np.ndarray:
        pick = rng.choice(ctrl_counts.shape[0], size=n_cells, replace=n_cells > ctrl_counts.shape[0])
        multipliers = np.exp2(np.clip(lfc, -LFC_CLIP, LFC_CLIP))
        base = ctrl_counts[pick]
        if method == "multinomial":
            from .countsim import multinomial_thin

            return multinomial_thin(base, multipliers, rng)
        return poisson_thin(base, multipliers, rng)

    def predict_context_cells(
        self,
        ctrl_counts: np.ndarray,
        targets: list[str],
        n_cells: int = spec.CELLS_PER_PERT,
        seed: int = 0,
        target_sum: float = spec.DE_TARGET_SUM,
        weights: tuple[float, float, float] | None = None,
        calibrated: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate integer-count cells for every target in one context."""
        basal = log_mean_cpm(ctrl_counts, target_sum)
        rng = np.random.default_rng(seed)
        out: dict[str, np.ndarray] = {}
        for t in targets:
            lfc = self.blend_lfc(t, basal, weights=weights)
            if calibrated and self.calibration is not None:
                lfc = self.calibration.transform(lfc, gene=t)
            out[t] = self.generate_cells(ctrl_counts, lfc, n_cells, rng)
        return out

    # -------------------------------------------------------- persistence
    def save(self, path) -> None:
        import torch

        payload = {
            "gene_names": self.gene_names,
            "prior": self.prior,
            "residual": None if self.residual is None else self.residual.state_dict(),
            "weights": tuple(self.weights),
            "has_stack_fn": self.stack_fn is not None,
            "calibration": None if self.calibration is None else self.calibration.as_dict(),
            "fit_info": self.fit_info,
        }
        torch.save(payload, path, _use_new_zipfile_serialization=True)

    @classmethod
    def load(cls, path) -> "HybridPredictor":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        weights = tuple(payload["weights"])
        if len(weights) == 2:  # v1 checkpoints
            weights = (weights[0], weights[1], 0.0)
        model = cls(
            gene_names=payload["gene_names"],
            prior=payload["prior"],
            residual=None,
            weights=weights,
            fit_info=payload.get("fit_info"),
        )
        cal = payload.get("calibration")
        if cal is not None:
            model.calibration = DECalibration(**{
                k: v for k, v in cal.items() if k in DECalibration.__dataclass_fields__
            })
        if payload["residual"] is not None:
            model.residual = ResidualLFCModel()
            model.residual.load_state_dict(payload["residual"])
        # Backfill fields added after a checkpoint was written so older
        # pickles keep working (dataclass defaults apply only at __init__).
        _defaults = LFCTransferPrior()
        for fname, fval in _defaults.__dict__.items():
            if not hasattr(model.prior, fname):
                setattr(model.prior, fname, fval)
        return model


# ------------------------------------------------------------------ helpers


def observed_lfc_table(
    pert_counts: dict[str, np.ndarray],
    ctrl_counts: np.ndarray,
    target_sum: float = spec.DE_TARGET_SUM,
    epsilon: float = spec.DE_EPSILON,
) -> dict[str, np.ndarray]:
    """Observed LFC vectors for each perturbation group vs its context controls."""
    ctrl_mean = normalize_cpm(ctrl_counts, target_sum).mean(axis=0)
    from .countsim import log2_fold_change

    return {
        g: log2_fold_change(normalize_cpm(x, target_sum).mean(axis=0), ctrl_mean, epsilon)
        for g, x in pert_counts.items()
    }
