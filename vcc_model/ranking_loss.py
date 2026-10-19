"""PDS-aligned ranking loss + cosine + magnitude, for CellDynamics.

Ranking loss directly optimizes what pds_cosine scores: for each predicted
effect in a batch, its cosine distance to its OWN true effect should be the
smallest among all true effects in the batch (panel-target coordinates removed
when gene_names supplied, else on full vectors).

Uses vcc_model.ensemble.pds_proxy_fast semantics but differentiable via
cross-entropy over distances with temperature.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _panel_mask(gene_names: list[str] | None, batch_genes: list[str] | None, G: int, device) -> torch.Tensor | None:
    if gene_names is None or batch_genes is None:
        return None
    idx = {g: i for i, g in enumerate(gene_names)}
    panel = {idx[g] for g in batch_genes if g in idx}
    if not panel:
        return None
    mask = torch.ones(G, dtype=torch.bool, device=device)
    for gi in panel:
        mask[gi] = False
    keep = torch.where(mask)[0]
    return keep


def ranking_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    temperature: float = 0.1,
    gene_names: list[str] | None = None,
    batch_genes: list[str] | None = None,
) -> torch.Tensor:
    """Contrastive ranking: pred[i] should be closest to truth[i].

    pred, truth: (B,G) LFC vectors.
    Returns scalar cross-entropy over (B,B) distance matrix rows.
    B<2 → 0.
    """
    B = pred.shape[0]
    if B < 2:
        return pred.sum() * 0.0
    device = pred.device
    G = pred.shape[1]
    keep = _panel_mask(gene_names, batch_genes, G, device)
    if keep is not None and keep.numel() >= 64:
        pred_m = pred[:, keep]
        truth_m = truth[:, keep]
    else:
        pred_m = pred
        truth_m = truth
    # unit-normalize for cosine
    pred_n = F.normalize(pred_m, dim=1, eps=1e-8)
    truth_n = F.normalize(truth_m, dim=1, eps=1e-8)
    # guard zero-norm rows (all-zero LFC): keep them as zero, dist=1 to everything
    pred_norm = pred_m.norm(dim=1)
    truth_norm = truth_m.norm(dim=1)
    pred_n = torch.where((pred_norm > 1e-8).unsqueeze(1), pred_n, torch.zeros_like(pred_n))
    truth_n = torch.where((truth_norm > 1e-8).unsqueeze(1), truth_n, torch.zeros_like(truth_n))
    # cosine distance matrix (B,B): 1 - cos
    sims = pred_n @ truth_n.t()  # (B,B)
    dists = 1.0 - sims
    # target: diagonal should be minimal → smallest distance
    # cross-entropy over -dists/τ (larger logit = smaller distance)
    logits = -dists / max(temperature, 1e-6)
    labels = torch.arange(B, device=device)
    return F.cross_entropy(logits, labels)


def cosine_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity (direction alignment)."""
    pred_n = F.normalize(pred, dim=1, eps=1e-8)
    truth_n = F.normalize(truth, dim=1, eps=1e-8)
    # mask zero-norm
    pn = pred.norm(dim=1)
    tn = truth.norm(dim=1)
    valid = (pn > 1e-8) & (tn > 1e-8)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    cos = (pred_n[valid] * truth_n[valid]).sum(dim=1)
    return (1.0 - cos).mean()


def l1_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Mean L1 on LFC magnitudes (regularizer, not primary signal)."""
    return F.l1_loss(pred, truth)


def combined_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    w_rank: float = 1.0,
    w_cos: float = 0.3,
    w_l1: float = 0.1,
    temperature: float = 0.1,
    gene_names: list[str] | None = None,
    batch_genes: list[str] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Weighted sum; returns (loss, parts dict)."""
    parts: dict = {}
    loss = pred.sum() * 0.0
    if w_rank > 0:
        r = ranking_loss(pred, truth, temperature=temperature, gene_names=gene_names, batch_genes=batch_genes)
        parts["rank"] = float(r.detach().item()) if r.requires_grad else float(r.item())
        loss = loss + w_rank * r
    else:
        parts["rank"] = 0.0
    if w_cos > 0:
        c = cosine_loss(pred, truth)
        parts["cos"] = float(c.detach().item())
        loss = loss + w_cos * c
    else:
        parts["cos"] = 0.0
    if w_l1 > 0:
        l = l1_loss(pred, truth)
        parts["l1"] = float(l.detach().item())
        loss = loss + w_l1 * l
    else:
        parts["l1"] = 0.0
    return loss, parts
