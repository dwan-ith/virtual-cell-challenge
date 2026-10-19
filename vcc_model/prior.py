"""Cross-context statistical transfer prior over log2FC responses.

For each (source context, perturbation) pair we hold an observed LFC vector.
To predict perturbation g in an unseen context we pool source observations
weighted by basal-profile similarity, shrink unreliable/inconsistent genes
toward zero, and rescale by a magnitude factor calibrated leave-one-context-out
on the source panel. Unseen genes fall back to a global-response +
target-knockdown signature blended with basal-correlation neighbors.

Consistency is two-tier (v2):
  * within-line agreement  -- sign agreement across different DATASETS of the
    same cell line (e.g. K562 genome-wide vs K562 essentialome vs Tahoe K562).
    Measures measurement reliability of the observed program.
  * cross-line agreement   -- sign agreement across different CELL LINES.
    Measures how conserved the program is across cellular states.
A program observed in a single (dataset, line) gets the calibrated default
conservation prior instead of implicit full trust -- the single most common
situation for the 2026 panel (272/300 targets seen only in K562 GW).

This is the TransPert-class component that stayed competitive in 2025 while
being nearly free to compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _softmax(x: np.ndarray, temperature: float = 0.01) -> np.ndarray:
    # Uniform fallback when all sims are clipped to 0 (all correlations ≤0)
    # is intentional — no source is more similar, so weight equally.
    z = (x - x.max()) / max(temperature, 1e-6)
    e = np.exp(z)
    return e / max(e.sum(), 1e-12)


def line_key(context: str) -> str:
    """Cell-line key of a namespaced context: 'src/LINE' -> 'LINE', else 'context'."""
    return context.split("/", 1)[1] if "/" in context else context


@dataclass
class LFCTransferPrior:
    gene_names: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    # lfc[c][g] = observed LFC vector in context c for perturbation g
    lfc: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    basal: dict[str, np.ndarray] = field(default_factory=dict)   # log mean ctrl CPM
    consistency: dict[str, np.ndarray] = field(default_factory=dict)  # per-gene [0,1]
    pooled: dict[str, np.ndarray] = field(default_factory=dict)  # unweighted mean LFC
    scale_loo: dict[str, float] = field(default_factory=dict)    # per-source alpha
    global_alpha: float = 1.0
    # Softmax temperature for context similarity weighting (tunable)
    similarity_temperature: float = 0.1
    # Two-tier consistency hyperparameters (tunable via fit kwargs):
    default_conservation: float = 0.45   # assumed sign-reliability for singly-observed programs
    within_weight: float = 0.5           # weight of within-line agreement in the mix
    single_obs_trust: float = 0.75       # trust for programs observed in >=2 datasets, 1 line
    # Per-program consistency shrinkage: multiply the transferred vector by
    # (0.25 + 0.75*consistency). Sidechain's 'n' lever (no shrinkage) gained
    # +0.0034 on the real board and +0.02 PDS on the H1 proxy; set False to test.
    use_consistency: bool = True
    # Fraction of the shared (panel-mean) program blended into direct transfers.
    # Large values inflate prediction-pairwise similarity, which collapses PDS
    # ranking toward 0.5; selected jointly with blend weights on the PDS proxy.
    shared_mix: float = 0.25
    # Apply the LOCO magnitude slope (alpha) inside predict. Default on for
    # backward compatibility; the calibrated pipeline prefers use_alpha=False
    # so the DE-calibration search owns ALL scale decisions (the measured LOCO
    # slopes onto unseen lines are ~0.07 and compound with the calibration gain,
    # which starves the Wilcoxon members of callable signal).
    use_alpha: bool = True
    # Optional de-shared correction: subtract beta * global_mean_effect from
    # every transferred program before output. Rationale: cross-line transfer
    # carries a systematic component (line-specific response bias); removing a
    # fraction of the panel-mean direction can improve program SHAPE, which is
    # what PDS ranking sees. beta=0 disables (v6/v8 behavior).
    beta_deshared: float = 0.0
    # Optional factor denoising: project every emitted program onto the top-k
    # principal components of the training response matrix. Single-replicate
    # pseudobulk LFCs carry substantial isotropic-ish noise; low-rank
    # reconstruction removes a share of it and sharpens program shape.
    denoise_rank: int = 0
    # --- v11 additions: hierarchical precision-weighted prior ---
    # Basal gating: set predicted LFC to 0 for genes with mean CPM below
    # this threshold in the target context. Low-expression genes cannot
    # respond; including them adds false-positive noise to jac/pds.
    # 0.0 = disabled; 5.0 = recommended (Arc wiki DE gate).
    basal_gate_cpm: float = 0.0
    # James-Stein shrinkage: shrink pooled programs toward 0 by a factor
    # estimated from cross-context variance of each gene. Reduces noise
    # from low-coverage genes while preserving strong signals.
    james_stein: bool = False
    # Effective dimension for JS: G=18533 forces factor 0; use rank-scale instead
    james_stein_dim: int = 32
    james_stein_strength: float = 1.0
    # Per-gene variance cache (populated during fit when james_stein=True)
    _gene_var: dict[str, float] = field(default_factory=dict)
    # Per-gene adaptive gain: when set, scales each gene's prediction by
    # a learned factor (e.g. from ESM/consistency). None = use global gain only.
    per_gene_gain: dict[str, float] | None = None
    # Zero unseen targets: when True, genes never observed as a perturbation
    # target in ANY training context return all-zero LFC instead of the
    # neighbor/shared fallback. LOCO-validated (H1): fallback guesses rank
    # below median and inflate nmae; zeros score exactly median PDS with
    # smaller amplitude error. Applies to the 28 val targets missing from
    # training (~9% of predictions).
    zero_unseen: bool = False
    # Protein-conditioned weighting: when set, per-gene context weights use ESM
    esm_embeddings: dict[str, np.ndarray] | None = None
    gene_conditioned: bool = False
    # Coverage-aware source fusion (default off for backward compatibility).
    # pair_n[(context, gene)] = number of cells behind that observed LFC.
    # source_weights maps a source context (or line key) to a reliability multiplier.
    # min_cells drops thin observations entirely; a source that saw a perturbation
    # only a handful of times abstains rather than voting with noise.
    pair_n: dict[tuple[str, str], int] = field(default_factory=dict)
    source_weights: dict[str, float] | None = None
    min_cells: int = 0
    # Gene-name-to-index lookup (built once in fit, used by basal gating)
    _gene_idx: dict[str, int] = field(default_factory=dict)
    _denoise_mean: np.ndarray | None = None
    _denoise_components: np.ndarray | None = None
    global_mean_effect: np.ndarray | None = None
    per_ctx_mean: dict[str, np.ndarray] = field(default_factory=dict)  # ctx -> mean program
    _basal_z: np.ndarray | None = None   # (n_ctx, G) standardized basal rows (lazy corr)
    _known_cols: list[str] = field(default_factory=list)  # genes eligible as neighbors
    _esm_context_centroids: dict[str, np.ndarray] | None = None  # ctx -> ESM centroid of observed genes

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        lfc: dict[str, dict[str, np.ndarray]],
        basal: dict[str, np.ndarray],
        gene_names: list[str],
        calibrate_scale: bool = True,
        pair_n: dict[tuple[str, str], int] | None = None,
        source_weights: dict[str, float] | None = None,
        min_cells: int = 0,
    ) -> "LFCTransferPrior":
        self.lfc = {c: dict(v) for c, v in lfc.items() if v}
        self.basal = {c: np.asarray(b, dtype=np.float64) for c, b in basal.items()}
        self.gene_names = list(gene_names)
        self.contexts = sorted(self.lfc)
        self.pair_n = dict(pair_n) if pair_n else {}
        self.source_weights = dict(source_weights) if source_weights else None
        self.min_cells = int(min_cells)

        perts = sorted({g for c in self.contexts for g in self.lfc[c]})
        idx_of = {g: i for i, g in enumerate(self.gene_names)}

        # Pooled mean + two-tier sign consistency (float32 keeps RAM bounded
        # with ~10k targets x 18.5k genes).
        self.pooled = {}
        self.consistency = {}
        ctx_lines = {c: line_key(c) for c in self.contexts}
        for g in perts:
            obs_ctx = [c for c in self.contexts if g in self.lfc[c]]
            mat = np.stack([self.lfc[c][g] for c in obs_ctx])
            mean = mat.mean(axis=0)
            agree = (np.sign(mat) == np.sign(mean)).mean(axis=0)

            lines = [ctx_lines[c] for c in obs_ctx]
            n_lines = len(set(lines))
            if len(obs_ctx) == 1:
                cons = np.full(mat.shape[1], self.default_conservation, dtype=np.float32)
            elif n_lines == 1:
                cons = np.full(mat.shape[1], self.single_obs_trust, dtype=np.float32)
                # Same line, several datasets: upgrade where they agree on sign
                # among STRONG entries (measurement reliability evidence).
                strong = np.abs(mean) > 0.25
                wcons = np.clip(2.0 * agree - 1.0, 0.0, 1.0)
                cons[strong] = (
                    self.single_obs_trust * (1 - self.within_weight)
                    + self.within_weight * wcons[strong]
                )
            else:
                cross = np.clip(2.0 * agree - 1.0, 0.0, 1.0)
                cons = cross.astype(np.float32)
            self.pooled[g] = mean.astype(np.float32)
            self.consistency[g] = cons

        if perts:
            self.global_mean_effect = (
                np.mean(np.stack([np.asarray(v, dtype=np.float32) for v in self.pooled.values()]), axis=0)
                .astype(np.float32)
            )
        # Per-context mean program, cached once (used by the shared-program
        # blend; stacking a whole context's LFC dict per call would be O(GxB)).
        self.per_ctx_mean = {}
        for c in self.contexts:
            if self.lfc[c]:
                self.per_ctx_mean[c] = np.mean(
                    np.stack([np.asarray(v, dtype=np.float32) for v in self.lfc[c].values()]),
                    axis=0,
                ).astype(np.float32)

        # Magnitude calibration: slope of observed ~ prior per held-out context
        self.scale_loo = {}
        if calibrate_scale:
            for held in self.contexts:
                slopes = []
                others = [c for c in self.contexts if c != held]
                for g in perts:
                    stack = [
                        self.lfc[c][g] for c in others if g in self.lfc.get(c, {})
                    ]
                    if not stack or g not in self.lfc.get(held, {}):
                        continue
                    prior = np.mean(np.stack(stack), axis=0)
                    obs = self.lfc[held][g]
                    den = float(prior @ prior)
                    if den > 1e-10:
                        slopes.append(float(obs @ prior) / den)
                if slopes:
                    self.scale_loo[held] = float(np.median(slopes))
            if self.scale_loo:
                self.global_alpha = float(np.median(list(self.scale_loo.values())))

        # James-Stein per-gene variance cache (computed once during fit)
        if self.james_stein and perts:
            self._gene_var = {}
            for g in perts:
                obs_ctx = [c for c in self.contexts if g in self.lfc[c]]
                if len(obs_ctx) >= 2:
                    mat = np.stack([self.lfc[c][g] for c in obs_ctx])
                    self._gene_var[g] = float(mat.var(axis=0).mean())
                else:
                    self._gene_var[g] = 0.0

        # Neighbor-transfer bookkeeping (lazy): standardize basal rows once;
        # correlations for an unseen gene are computed on demand.
        known_genes = sorted(g for g in perts if g in idx_of)
        if len(known_genes) >= 5 and len(self.contexts) >= 3:
            B = np.stack([self.basal[c] for c in self.contexts])   # ctx x G
            std = B.std(axis=0, keepdims=True)
            self._basal_z = ((B - B.mean(axis=0, keepdims=True)) / (std + 1e-6)).astype(np.float32)
            self._known_cols = known_genes
        else:
            self._basal_z = None
            self._known_cols = []
        self._gene_idx = {g: i for i, g in enumerate(self.gene_names)}
        # ESM context centroids for gene-conditioned weighting
        if self.esm_embeddings is not None:
            self._esm_context_centroids = {}
            for c in self.contexts:
                genes_here = [g for g in self.lfc.get(c, {}) if g in self.esm_embeddings and np.linalg.norm(self.esm_embeddings[g]) > 1e-6 and np.isfinite(self.esm_embeddings[g]).all()]
                if len(genes_here) >= 3:
                    mat = np.stack([self.esm_embeddings[g] / max(np.linalg.norm(self.esm_embeddings[g]),1e-9) for g in genes_here])
                    centroid = mat.mean(axis=0)
                    centroid /= max(np.linalg.norm(centroid), 1e-9)
                    self._esm_context_centroids[c] = centroid.astype(np.float32)
        return self

    def _esm_centroid(self) -> np.ndarray | None:
        if self.esm_embeddings is None or not self.esm_embeddings:
            return None
        vals = list(self.esm_embeddings.values())
        if not vals:
            return None
        mat = np.stack([v/np.linalg.norm(v) for v in vals])
        c = mat.mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-9)
        return c

    # -------------------------------------------------------------- weights
    def similarity_weights(self, target_basal: np.ndarray, exclude: set[str] | None = None) -> dict[str, float]:
        exclude = exclude or set()
        ctxs = [c for c in self.contexts if c not in exclude and c in self.basal]
        if not ctxs:
            return {}
        sims = np.array([
            float(np.corrcoef(target_basal, self.basal[c])[0, 1]) if (
                np.std(target_basal) > 1e-9 and np.std(self.basal[c]) > 1e-9
            ) else 0.0
            for c in ctxs
        ])
        sims = np.clip(sims, 0.0, None)
        w = _softmax(sims, temperature=self.similarity_temperature)
        return {c: float(wi) for c, wi in zip(ctxs, w)}

    def gene_conditioned_weights(
        self,
        target_basal: np.ndarray,
        gene: str,
        exclude: set[str] | None = None,
        protein_sim: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Per-gene context weights: basal similarity modulated by protein similarity.

        protein_sim maps source context -> protein-level similarity for this gene
        (e.g. from ESM-2 embeddings). When None, falls back to basal weights.
        This provides gene-level adaptivity without requiring a full foundation model.
        """
        base = self.similarity_weights(target_basal, exclude)
        if protein_sim is None or not base:
            return base
        # Geometric blend: w_g(c) ∝ w_basal(c) * (0.5 + 0.5*protein_sim(c))
        blended = {}
        for c, w in base.items():
            p = float(protein_sim.get(c, 0.5))
            blended[c] = w * (0.5 + 0.5 * max(0.0, min(1.0, p)))
        s = sum(blended.values())
        if s < 1e-12:
            return base
        return {c: v / s for c, v in blended.items()}

    def calibrated_alpha(
        self,
        target_basal: np.ndarray,
        exclude: set[str] | None = None,
        w: dict[str, float] | None = None,
    ) -> float:
        """Magnitude slope for the target state; pass precomputed weights `w`."""
        if w is None:
            w = self.similarity_weights(target_basal, exclude)
        if not w:
            return self.global_alpha
        alphas = [self.scale_loo.get(c, self.global_alpha) for c in w]
        weights = np.array([w[c] for c in w])
        return float(np.average(alphas, weights=weights))

    # -------------------------------------------------------------- predict
    def transfer_params(
        self,
        target_basal: np.ndarray,
        exclude_contexts: set[str] | None = None,
    ) -> tuple[dict[str, float], float]:
        """Context-level quantities (similarity weights + magnitude alpha).

        These depend only on the target basal state, not on the gene -- compute
        once per context and reuse across every target (the hot path in both
        residual training and submission generation).
        """
        exclude = exclude_contexts or set()
        w_map = self.similarity_weights(target_basal, exclude)
        alpha = self.calibrated_alpha(target_basal, exclude, w=w_map)
        return w_map, alpha

    def _is_unseen(self, gene: str) -> bool:
        """True when gene was never observed as a target in any training context."""
        for c in self.contexts:
            if gene in self.lfc.get(c, {}):
                return False
        return True

    def predict_lfc(
        self,
        gene: str,
        target_basal: np.ndarray,
        exclude_contexts: set[str] | None = None,
    ) -> np.ndarray:
        """Predicted LFC vector for `gene` in a context with this basal state."""
        if self.zero_unseen and self._is_unseen(gene):
            return np.zeros(len(self.gene_names), dtype=np.float64)
        if self.gene_conditioned and self.esm_embeddings is not None and self._esm_context_centroids is not None and gene in self.esm_embeddings:
            w_map, alpha = self.transfer_params(target_basal, exclude_contexts)
            # Gene-conditioned: per-context protein similarity via ESM centroid cosine
            v = self.esm_embeddings[gene] / max(np.linalg.norm(self.esm_embeddings[gene]), 1e-9)
            protein_sim: dict[str, float] = {}
            for c, centroid in self._esm_context_centroids.items():
                if c in w_map:
                    cos = float(np.dot(v, centroid))
                    protein_sim[c] = max(0.0, (cos + 1) / 2)  # [0,1]
            if protein_sim:
                w_map = self.gene_conditioned_weights(target_basal, gene, exclude=exclude_contexts, protein_sim=protein_sim)
        else:
            w_map, alpha = self.transfer_params(target_basal, exclude_contexts)
        v = self._predict_with(gene, w_map, alpha)
        if v is None:
            v = ((alpha if self.use_alpha else 1.0) * self._neighbor_fallback(gene, target_basal)).astype(np.float64)
        # Basal gate: zero out individual gene dimensions where the target
        # context has low expression (those genes cannot respond, so predicted
        # LFC for those dimensions is noise). Applied AFTER prediction on the
        # response vector, not on the perturbation target itself.
        if self.basal_gate_cpm > 0.0:
            mean_cpm = np.exp(target_basal) - 1e-9
            low_mask = mean_cpm < self.basal_gate_cpm
            v = v.copy()
            v[low_mask] = 0.0
        return v

    def _predict_with(
        self,
        gene: str,
        w_map: dict[str, float],
        alpha: float,
    ) -> np.ndarray | None:
        """Direct-transfer prediction; None when the gene was never observed."""
        stack, weights = [], []
        for c, w in w_map.items():
            if gene in self.lfc.get(c, {}):
                n = self.pair_n.get((c, gene), 0) if self.pair_n else 0
                if self.min_cells > 0 and n < self.min_cells:
                    continue
                vote = float(w)
                if self.source_weights:
                    vote *= float(self.source_weights.get(c, self.source_weights.get(line_key(c), 1.0)))
                if self.pair_n:
                    vote *= min(1.0, float(n) / 100.0) if n > 0 else 0.0
                if vote <= 0:
                    continue
                stack.append(self.lfc[c][gene])
                weights.append(vote)

        if not stack:
            return None
        w_arr = np.asarray(weights, dtype=np.float64)
        w_arr = w_arr / max(w_arr.sum(), 1e-12)
        specific = np.average(np.stack(stack), axis=0, weights=w_arr)
        # Apply per-gene adaptive gain if set on prior (alternative to calibrate-level)
        if self.per_gene_gain is not None and gene in self.per_gene_gain:
            specific = specific * float(self.per_gene_gain[gene])
        cons = self.consistency.get(gene)
        if cons is not None and self.use_consistency:
            specific = specific * (0.25 + 0.75 * cons)
        shared = self._shared_program_from(w_map)
        raw = (1.0 - self.shared_mix) * specific + self.shared_mix * shared
        raw = self._postprocess_shape(raw)
        # James-Stein shrinkage: shrink toward 0 based on cross-context variance
        # Uses effective dim (default 32 = lfc_rank) not G=18533, else factor collapses to 0
        if self.james_stein and gene in self._gene_var and len(stack) >= 2:
            p_eff = max(int(self.james_stein_dim), 3)
            var_gene = self._gene_var[gene]
            norm_sq = float(np.dot(raw, raw))
            if norm_sq > 1e-10 and var_gene > 0:
                factor = max(0.0, 1.0 - self.james_stein_strength * (p_eff - 2) * var_gene / norm_sq)
                raw = raw * factor
        return (alpha * raw).astype(np.float64) if self.use_alpha else raw.astype(np.float64)

    def predict_many(
        self,
        genes: list[str],
        target_basal: np.ndarray,
        exclude_contexts: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict a batch of targets sharing one basal state efficiently."""
        w_map, alpha = self.transfer_params(target_basal, exclude_contexts)
        out: dict[str, np.ndarray] = {}
        fallback_genes = [
            g for g in genes
            if not any(g in self.lfc.get(c, {}) for c in w_map)
        ]
        shared_hint = self._shared_program_from(w_map) if fallback_genes else None
        scale_alpha = alpha if self.use_alpha else 1.0
        # Basal gate mask: precompute once for all targets
        gate_mask = None
        if self.basal_gate_cpm > 0.0:
            mean_cpm = np.exp(target_basal) - 1e-9
            gate_mask = mean_cpm < self.basal_gate_cpm
        for g in genes:
            if self.zero_unseen and self._is_unseen(g):
                out[g] = np.zeros(len(self.gene_names), dtype=np.float64)
                continue
            v = self._predict_with(g, w_map, alpha)
            if v is None:
                v = (scale_alpha * self._neighbor_fallback(g, target_basal, shared_hint=shared_hint)).astype(np.float64)
            if gate_mask is not None:
                v = v.copy()
                v[gate_mask] = 0.0
            out[g] = v
        return out

    def predict_many_gc(
        self,
        genes: list[str],
        target_basal: np.ndarray,
        exclude_contexts: set[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Batched predict_lfc WITH gene-conditioned weights (v29 submission path).

        Identical outputs to calling predict_lfc per gene; hoists the
        context-level transfer_params (basal correlations) once per context
        instead of once per gene. predict_many intentionally skips
        gene-conditioning; this method does not.
        """
        import numpy as _np
        w_map, alpha = self.transfer_params(target_basal, exclude_contexts)
        out: dict[str, np.ndarray] = {}
        use_gc = (
            self.gene_conditioned and self.esm_embeddings is not None
            and self._esm_context_centroids is not None
        )
        gate_mask = None
        if self.basal_gate_cpm > 0.0:
            mean_cpm = _np.exp(target_basal) - 1e-9
            gate_mask = mean_cpm < self.basal_gate_cpm
        for g in genes:
            if self.zero_unseen and self._is_unseen(g):
                out[g] = _np.zeros(len(self.gene_names), dtype=_np.float64)
                continue
            w_here = w_map
            if use_gc and g in self.esm_embeddings:
                v = self.esm_embeddings[g] / max(_np.linalg.norm(self.esm_embeddings[g]), 1e-9)
                protein_sim: dict[str, float] = {}
                for c, centroid in self._esm_context_centroids.items():
                    if c in w_map:
                        cos = float(_np.dot(v, centroid))
                        protein_sim[c] = max(0.0, (cos + 1) / 2)
                if protein_sim:
                    w_here = self.gene_conditioned_weights(
                        target_basal, g, exclude=exclude_contexts, protein_sim=protein_sim)
            v = self._predict_with(g, w_here, alpha)
            if v is None:
                v = ((alpha if self.use_alpha else 1.0) * self._neighbor_fallback(g, target_basal)).astype(_np.float64)
            if gate_mask is not None:
                v = v.copy()
                v[gate_mask] = 0.0
            out[g] = v
        return out

    def fit_denoiser(self, rank: int = 32, max_targets: int = 6000) -> "LFCTransferPrior":
        """Fit a low-rank denoiser on the pooled training programs.

        Uses randomized SVD over up to `max_targets` pooled program vectors
        (the largest contexts first for coverage). Call after fit().
        """
        if rank <= 0 or not self.pooled:
            self.denoise_rank = 0
            return self
        rng = np.random.default_rng(0)
        genes = sorted(self.pooled)
        if len(genes) > max_targets:
            keep = rng.choice(len(genes), size=max_targets, replace=False)
            genes = [genes[i] for i in keep]
        mat = np.stack([np.asarray(self.pooled[g], dtype=np.float32) for g in genes])
        mean = mat.mean(axis=0)
        from sklearn.utils.extmath import randomized_svd

        _, _, vt = randomized_svd(mat - mean, n_components=min(rank, min(mat.shape)), n_iter=3, random_state=0)
        self._denoise_mean = mean.astype(np.float32)
        self._denoise_components = vt.astype(np.float32)
        self.denoise_rank = int(vt.shape[0])
        return self

    def _postprocess_shape(self, vec: np.ndarray) -> np.ndarray:
        """De-shared correction + low-rank denoising (shape-only operations)."""
        v = vec
        if self.beta_deshared > 0.0 and self.global_mean_effect is not None:
            v = v - self.beta_deshared * np.asarray(self.global_mean_effect, dtype=np.float64)
        if self.denoise_rank > 0 and self._denoise_components is not None:
            c = (v - self._denoise_mean) @ self._denoise_components.T
            v = self._denoise_mean + c @ self._denoise_components
        return v

    def _shared_program_from(self, w_map: dict[str, float]) -> np.ndarray:
        assert self.global_mean_effect is not None
        if w_map and self.lfc:
            acc, ws = [], []
            for c, wi in w_map.items():
                pm = self.per_ctx_mean.get(c)
                if pm is not None:
                    acc.append(pm)
                    ws.append(wi)
            if acc:
                ws = np.asarray(ws)
                ws = ws / ws.sum()
                return np.average(np.stack(acc), axis=0, weights=ws)
        return self.global_mean_effect

    def _neighbor_fallback(
        self,
        gene: str,
        target_basal: np.ndarray,
        shared_hint: np.ndarray | None = None,
    ) -> np.ndarray:
        """Program for a never-perturbed gene: global response + own knockdown,
        blended with the pooled programs of the most basal-similar targets."""
        out_dim = len(self.gene_names)
        out = np.zeros(out_dim)
        base = (shared_hint if shared_hint is not None else self.global_mean_effect)
        if base is None:
            base = out
        base = np.asarray(base, dtype=np.float64).copy()
        gidx = {g: i for i, g in enumerate(self.gene_names)}
        gi = gidx.get(gene)
        if gi is not None:
            base[gi] -= 1.4
        if self._basal_z is None or gi is None or not self.pooled or not self._known_cols:
            return base
        # Correlate the unseen gene's basal trajectory across source contexts
        # with every known target gene's trajectory (lazy, O(ctx x n_known)).
        gz = self._basal_z[:, gi]
        kz_idx = [gidx[k] for k in self._known_cols if k in gidx]
        kz = self._basal_z[:, kz_idx]                      # ctx x n_known
        denom = max(len(self.contexts) - 1, 1)
        scores = (kz * gz[:, None]).sum(axis=0) / denom    # Pearson (standardized)
        scores = np.clip(scores, 0.0, None)
        if scores.size > 5:
            top = np.argpartition(scores, -5)[-5:]
        else:
            top = np.argsort(scores)
        wt = scores[top]
        if wt.sum() < 1e-8:
            return base
        wt = wt / wt.sum()
        blend = np.zeros_like(base)
        known_cols = [k for k in self._known_cols if k in gidx]
        for wi, ti in zip(wt, top):
            tg = known_cols[int(ti)]
            pv = self.pooled.get(tg)
            if pv is not None:
                blend += wi * pv
        return 0.6 * base + 0.4 * blend
