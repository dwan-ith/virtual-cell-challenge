"""Protein / pathway similarity for per-gene transfer weighting.

Provides gene-conditioned context weights without requiring a 650M ESM-2 download.
Two tiers:
  1. If ESM-2 embeddings are available (data/protein/esm2_*.npy), use cosine.
  2. Fallback: gene co-expression similarity from training LFC matrix
     (genes with correlated perturbation responses get higher transfer weights).

This breaks the single-correlation bottleneck: different genes can weight
source contexts differently, which is exactly what BM_xTVC/XLearning did
with ESM-2 and what TransPert did with per-gene DE-probability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_esm_embeddings(path: Path | None = None) -> dict[str, np.ndarray] | None:
    """Load precomputed ESM-2 / SE protein embeddings if present."""
    candidates = [
        Path("data/protein/se_full_18533.npz"),
        Path("data/protein/esm2_t6_8m_panel.npz"),
        Path("data/protein/esm2_650m.npy"),
        Path("data/protein/esm2_embeddings.npz"),
        Path("artifacts/protein_embeddings.npz"),
        Path("data/state/se600m/protein_embeddings.pt"),
    ]
    if path is not None:
        candidates = [Path(path)] + candidates
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix == ".npy":
                arr = np.load(p, allow_pickle=True)
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    return {str(k): np.asarray(v, dtype=np.float32) for k, v in arr.item().items()}
            elif p.suffix == ".npz":
                data = np.load(p, allow_pickle=True)
                if "gene_names" in data and "embeddings" in data:
                    genes = [str(g) for g in data["gene_names"]]
                    emb = data["embeddings"]
                    return {g: emb[i] for i, g in enumerate(genes)}
                return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
        except Exception:
            continue
    return None


def build_coexpression_similarity(
    lfc_dict: dict[str, np.ndarray],
    gene_names: list[str],
) -> dict[str, float]:
    """Per-gene consistency proxy from LFC effect size — used when ESM unavailable.

    Returns gene -> [0,1] where high mean|lfc| and low variance → high score.
    Computed from pooled LFC vectors: score = sigmoid(mean|lfc| - 0.1) * (1 - cv)
    """
    out: dict[str, float] = {}
    for g in gene_names:
        v = lfc_dict.get(g)
        if v is None or v.size == 0:
            out[g] = 0.5
            continue
        v = np.asarray(v, dtype=np.float64)
        m_abs = float(np.mean(np.abs(v)))
        # Scale m_abs 0.05-0.5 to 0-1 via logistic
        s = 1.0 / (1.0 + np.exp(-10 * (m_abs - 0.15)))
        # Low variance bonus
        sd = float(np.std(v))
        mean = float(np.mean(v))
        cv = sd / max(abs(mean), 0.1)
        cv_factor = max(0.0, 1.0 - min(cv, 1.0) * 0.5)
        out[g] = float(np.clip(s * (0.7 + 0.3 * cv_factor), 0.0, 1.0))
    return out


def compute_gene_conditioned_weights(
    basal_weights: dict[str, float],
    gene: str,
    esm_embeddings: dict[str, np.ndarray] | None = None,
    coexpression_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Blend basal weights with protein/coexpression similarity for one gene."""
    if not basal_weights:
        return basal_weights
    if esm_embeddings is None and coexpression_scores is None:
        return basal_weights
    # Modulate via coexpression: high score -> sharpen toward best basal context
    if coexpression_scores is not None:
        cons = float(coexpression_scores.get(gene, 0.5))
        # cons 0→1 maps to blend factor 0.7→1.3 on top-weighted contexts
        if cons > 0.6:
            # Sharpen: amplify max weight
            max_ctx = max(basal_weights, key=basal_weights.get)
            out = {c: w * (1.4 if c == max_ctx else 0.9) for c, w in basal_weights.items()}
            s = sum(out.values())
            return {c: v / s for c, v in out.items()}
        elif cons < 0.3:
            # Flatten: more uniform
            n = len(basal_weights)
            uniform = 1.0 / n
            return {c: 0.7 * w + 0.3 * uniform for c, w in basal_weights.items()}
    return basal_weights


def build_per_gene_gain_map(
    prior_consistency: dict[str, np.ndarray],
    gene_names: list[str],
    default_cons: float = 0.45,
    scale: float = 0.5,
    esm_embeddings: dict[str, np.ndarray] | None = None,
    esm_boost: float = 0.15,
) -> dict[str, float]:
    """Construct per-gene gain multipliers from consistency + optional ESM.

    gain_g = 1 - scale + scale * cons_g  (cons in [0,1])
    Genes with high cross-context agreement get full gain; low-agreement
    get reduced gain, breaking the single-scalar PDS-nmae tradeoff.
    When ESM embeddings are supplied, genes whose protein is similar to
    highly-consistently-perturbed genes get a boost: cons' = (1-b)*cons + b*cos.
    """
    pg: dict[str, float] = {}
    # Precompute ESM centroid for boost — mean of all available panel embeddings
    # (previous high-cons threshold yielded 0 genes for 92% singleton panel)
    esm_centroid = None
    if esm_embeddings is not None:
        avail = [g for g in gene_names if g in esm_embeddings and np.linalg.norm(esm_embeddings[g]) > 1e-6 and np.isfinite(esm_embeddings[g]).all()]
        if len(avail) >= 5:
            mat = np.stack([esm_embeddings[g] / max(np.linalg.norm(esm_embeddings[g]),1e-9) for g in avail])
            esm_centroid = mat.mean(axis=0)
            esm_centroid /= max(np.linalg.norm(esm_centroid), 1e-9)
    for g in gene_names:
        cons_arr = prior_consistency.get(g)
        if cons_arr is not None and cons_arr.size:
            cons = float(np.clip(cons_arr.mean(), 0.0, 1.0))
        else:
            cons = float(default_cons)
        # ESM boost: nudge cons toward protein similarity to high-cons centroid
        if esm_embeddings is not None and esm_centroid is not None and g in esm_embeddings:
            v = esm_embeddings[g] / max(np.linalg.norm(esm_embeddings[g]), 1e-9)
            cos = float(np.dot(v, esm_centroid))
            cos = max(0.0, min(1.0, (cos + 1) / 2))  # map [-1,1] -> [0,1]
            cons = (1 - esm_boost) * cons + esm_boost * cos
        pg[g] = float(1.0 - scale + scale * cons)
    pg["_default"] = float(1.0 - scale + scale * default_cons)
    return pg
