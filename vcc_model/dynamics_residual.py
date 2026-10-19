"""v35 ResidualCellDynamics: predict CORRECTION to prior, not absolute LFC.

Fixes v32 failures (all measured):
  1. Absolute prediction ignored prior's 0.68-0.89 signal -> predict delta,
     zero-init decoder so model starts exactly at prior (bitsandatomss Model B).
  2. Random-batch ranking != PDS -> panel batches: each step contrasts one
     context's full gene panel (the actual PDS task).
  3. LFC-space proxy != count-space metric -> targets/effects in count-effect
     space: effect = log1p(TS*pert/sum) - log1p(TS*ctrl/sum), same as
     metrics.pds_cosine effects() but from mean-CPM profiles.
  4. Strict gate lives in rd_v35_probe.py (both holdouts + count proxy).

Input per gene: concat[prior_lfc(G), basal_log(G), esm_proj(256)] -> latent
  -> gated residual transition w/ quadratic pert skip -> correction(G).
Output: prior_lfc + correction (clipped at predict time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ config
@dataclass
class ResidualDynamicsConfig:
    gene_dim: int = 18533
    esm_dim: int = 5120
    esm_proj_dim: int = 256
    latent_dim: int = 64
    in_proj_dim: int = 512
    enc_h1: int = 512
    enc_h2: int = 256
    trans_h1: int = 512
    trans_h2: int = 256
    dec_hidden: int = 512
    dropout: float = 0.10
    epochs: int = 5
    steps_per_epoch: int = 50
    panel_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_rank: float = 0.5
    w_cos: float = 1.0
    w_l1: float = 0.3
    rank_temperature: float = 0.1
    effect_target_sum: float = 50000.0
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    max_pairs: int = 3000


# ------------------------------------------------------------------ model
class ResidualCellDynamics(nn.Module):
    def __init__(self, cfg: ResidualDynamicsConfig | None = None):
        super().__init__()
        self.cfg = cfg or ResidualDynamicsConfig()
        cfg = self.cfg
        self.esm_proj = nn.Linear(cfg.esm_dim, cfg.esm_proj_dim)
        in_dim = 2 * cfg.gene_dim + cfg.esm_proj_dim
        self.in_proj = nn.Linear(in_dim, cfg.in_proj_dim)
        self.in_ln = nn.LayerNorm(cfg.in_proj_dim)
        self.enc1 = nn.Linear(cfg.in_proj_dim, cfg.enc_h1)
        self.enc_ln1 = nn.LayerNorm(cfg.enc_h1)
        self.enc2 = nn.Linear(cfg.enc_h1, cfg.enc_h2)
        self.enc_ln2 = nn.LayerNorm(cfg.enc_h2)
        self.enc3 = nn.Linear(cfg.enc_h2, cfg.latent_dim)
        # transition on concat[z, p_quad]
        tin = cfg.latent_dim + 2 * cfg.esm_proj_dim
        self.tr1 = nn.Linear(tin, cfg.trans_h1)
        self.tr_ln1 = nn.LayerNorm(cfg.trans_h1)
        self.tr2 = nn.Linear(cfg.trans_h1, cfg.trans_h2)
        self.tr_ln2 = nn.LayerNorm(cfg.trans_h2)
        self.gate = nn.Linear(cfg.trans_h2, cfg.latent_dim)
        self.dz = nn.Linear(cfg.trans_h2, cfg.latent_dim)
        # decoder -> correction
        self.dec1 = nn.Linear(cfg.latent_dim, cfg.dec_hidden)
        self.dec_ln1 = nn.LayerNorm(cfg.dec_hidden)
        self.dec2 = nn.Linear(cfg.dec_hidden, cfg.gene_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()
        # zero-init decoder output: model starts exactly at prior
        nn.init.zeros_(self.dec2.weight)
        nn.init.zeros_(self.dec2.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _esm_p(self, esm_vec: torch.Tensor) -> torch.Tensor:
        p = self.esm_proj(esm_vec)
        return F.normalize(p, dim=1, eps=1e-8) * math.sqrt(p.shape[1]) * 0.1

    def forward(self, prior_lfc: torch.Tensor, basal_log: torch.Tensor, esm_vec: torch.Tensor) -> torch.Tensor:
        """Return CORRECTION (B,G). Final = prior_lfc + correction."""
        p = self._esm_p(esm_vec)
        x = torch.cat([prior_lfc, basal_log, p], dim=1)
        h = self.drop(F.gelu(self.in_ln(self.in_proj(x))))
        h = self.drop(F.gelu(self.enc_ln1(self.enc1(h))))
        h = self.drop(F.gelu(self.enc_ln2(self.enc2(h))))
        z = torch.tanh(self.enc3(h))
        p_quad = torch.cat([p, p * p], dim=1)
        t = torch.cat([z, p_quad], dim=1)
        t = self.drop(F.gelu(self.tr_ln1(self.tr1(t))))
        t = self.drop(F.gelu(self.tr_ln2(self.tr2(t))))
        gate = torch.sigmoid(self.gate(t))
        dz = self.dz(t)
        z_next = torch.tanh(z + gate * torch.tanh(dz))
        d = self.drop(F.gelu(self.dec_ln1(self.dec1(z_next))))
        correction = self.dec2(d)
        return correction

    @torch.no_grad()
    def predict_lfc(self, prior_lfc: np.ndarray, basal_log: np.ndarray, esm_vec: np.ndarray | None) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        G = self.cfg.gene_dim
        pr = torch.from_numpy(np.asarray(prior_lfc, dtype=np.float32).reshape(1, G)).to(device)
        b = torch.from_numpy(np.asarray(basal_log, dtype=np.float32).reshape(1, G)).to(device)
        if esm_vec is None:
            esm_vec = np.zeros(self.cfg.esm_dim, dtype=np.float32)
        e = torch.from_numpy(np.asarray(esm_vec, dtype=np.float32).reshape(1, -1)).to(device)
        corr = self.forward(pr, b, e)
        out = (pr + corr).squeeze(0).cpu().numpy().astype(np.float64)
        return np.clip(out, -8.0, 8.0)

    def save(self, path: str):
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "ResidualCellDynamics":
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        cfg = ckpt["cfg"] if isinstance(ckpt, dict) and "cfg" in ckpt else ResidualDynamicsConfig()
        m = cls(cfg)
        state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
        m.load_state_dict(state)
        return m


# ------------------------------------------------------------------ count-effect space
def count_effect_matrix(lfc_mat: np.ndarray, basal_log: np.ndarray, target_sum: float = 50000.0) -> np.ndarray:
    """LFC vectors -> PDS-space effects via mean-CPM profiles.

    m_ctrl = exp(basal); m_pert = m_ctrl * 2^lfc (per row).
    effect = log1p(TS*m_pert/sum) - log1p(TS*m_ctrl/sum).
    Mirrors metrics.pds_cosine effects() (which uses summed counts).
    """
    basal_log = np.asarray(basal_log, dtype=np.float32)
    m_c = np.exp(np.clip(basal_log, -30, 30)) - 1e-9
    m_c = np.clip(m_c, 1e-12, None)
    lfc = np.clip(np.asarray(lfc_mat, dtype=np.float32), -6.0, 6.0)
    pert = m_c[None, :] * np.exp2(lfc)
    sp = pert.sum(axis=1, keepdims=True)
    sp = np.where(sp > 0, sp, 1.0)
    sc = m_c.sum()
    prof_p = np.log1p(target_sum * pert / sp)
    prof_c = np.log1p(target_sum * m_c / sc)
    return prof_p - prof_c[None, :]


def pds_proxy_count(
    preds: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    basal_log: np.ndarray,
    gene_names: list[str],
    target_sum: float = 50000.0,
) -> float:
    """Official-style PDS computed on count-effects (not raw LFC).

    Same ranking code as ensemble.pds_proxy_fast, but effects pass through
    count_effect_matrix first so magnitude errors are penalized the way
    cell-eval2 penalizes them.
    """
    names = sorted(set(preds) & set(truths))
    n = len(names)
    if n < 2:
        return float("nan")
    idx = {g: i for i, g in enumerate(gene_names)}
    panel = {idx[g] for g in names if g in idx}
    G = len(gene_names)
    keep = np.array([i for i in range(G) if i not in panel], dtype=int)
    real_eff = count_effect_matrix(np.stack([truths[nm] for nm in names]), basal_log, target_sum)[:, keep]
    pred_eff = count_effect_matrix(np.stack([preds[nm] for nm in names]), basal_log, target_sum)[:, keep]
    nr = np.linalg.norm(real_eff, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    real_unit = real_eff / nr
    ph = np.linalg.norm(pred_eff, axis=1, keepdims=True)
    zero_pred = (ph[:, 0] == 0)
    ph[ph == 0] = 1.0
    sims = pred_eff / ph @ real_unit.T
    dists = 1.0 - sims
    d_self = dists[np.arange(n), np.arange(n)]
    k = (dists < d_self[:, None]).sum(axis=1) + 0.5 * ((dists == d_self[:, None]).sum(axis=1) - 1)
    scores = np.full(n, 0.5)
    ok = ~zero_pred
    scores[ok] = 1.0 - k[ok] / (n - 1)
    return float(scores.mean())


# ------------------------------------------------------------------ panel-batch training
def _stratified_keys(
    lfc_pairs: dict[tuple[str, str], np.ndarray],
    max_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    keys = sorted(lfc_pairs)
    if len(keys) <= max_pairs:
        return keys
    rng = np.random.default_rng(seed)
    by_ctx: dict[str, list] = {}
    for k in keys:
        by_ctx.setdefault(k[0], []).append(k)
    per = max(1, max_pairs // max(len(by_ctx), 1))
    picked = []
    for c in sorted(by_ctx):
        ks = sorted(by_ctx[c])
        if len(ks) > per:
            sel = rng.choice(len(ks), size=per, replace=False)
            ks = [ks[i] for i in sel]
        picked.extend(ks)
    return sorted(picked)


def train_residual_dynamics(
    prior,
    lfc_pairs: dict[tuple[str, str], np.ndarray],
    basals: dict[str, np.ndarray],
    gene_names: list[str],
    esm_embeddings: dict[str, np.ndarray] | None = None,
    cfg: ResidualDynamicsConfig | None = None,
) -> tuple[ResidualCellDynamics, dict]:
    """Train with LOCO prior inputs, panel batches, effect-space ranking loss.

    prior: fitted LFCTransferPrior on train contexts. For each training context
    c, inputs use prior.predict_many(genes, basal_c, exclude={c}) (true zero-shot).
    """
    from .ranking_loss import combined_loss
    cfg = cfg or ResidualDynamicsConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    keys = _stratified_keys(lfc_pairs, cfg.max_pairs, cfg.seed)
    print(f"[res-dyn] {len(keys)} pairs, panel {cfg.panel_size}, epochs {cfg.epochs}x{cfg.steps_per_epoch}", flush=True)

    # Group by context; build LOCO prior inputs once per context
    by_ctx: dict[str, list[str]] = {}
    for (c, g) in keys:
        by_ctx.setdefault(c, []).append(g)
    panels: dict[str, dict] = {}
    for ctx, genes_here in sorted(by_ctx.items()):
        basal = np.asarray(basals[ctx], dtype=np.float32)
        truth = np.stack([np.asarray(lfc_pairs[(ctx, g)], dtype=np.float32) for g in genes_here])
        truth = np.nan_to_num(truth, nan=0.0, posinf=8.0, neginf=-8.0)
        np.clip(truth, -8.0, 8.0, out=truth)
        pv_map = prior.predict_many(genes_here, basals[ctx], exclude_contexts={ctx})
        prior_mat = np.stack([np.asarray(pv_map[g], dtype=np.float32) for g in genes_here])
        if esm_embeddings is not None:
            esm_mat = np.stack([
                np.asarray(esm_embeddings.get(g, np.zeros(cfg.esm_dim, dtype=np.float32)), dtype=np.float32)
                for g in genes_here
            ])
        else:
            esm_mat = np.zeros((len(genes_here), cfg.esm_dim), dtype=np.float32)
        panels[ctx] = {"genes": genes_here, "truth": truth, "prior": prior_mat, "esm": esm_mat, "basal": basal}
    print(f"[res-dyn] {len(panels)} context panels", flush=True)

    import gc as _gc
    # Move panels to GPU ONCE (kills per-step CPU allocs that OOM system RAM).
    # truth/prior/y_eff/esm per context + basal m_c. y_eff precomputed (no grad).
    gpu_panels: dict[str, dict] = {}
    with torch.no_grad():
        for ctx, P in panels.items():
            truth_t = torch.from_numpy(P["truth"]).to(device)
            prior_t = torch.from_numpy(P["prior"]).to(device)
            esm_t = torch.from_numpy(P["esm"]).to(device)
            basal_t = torch.from_numpy(P["basal"]).to(device)
            m_c = torch.clamp(torch.exp(torch.clamp(basal_t, -30, 30)) - 1e-9, min=1e-12)
            pert = m_c[None, :] * torch.exp2(torch.clamp(truth_t, -6.0, 6.0))
            sp = pert.sum(dim=1, keepdim=True).clamp_min(1.0)
            sc = m_c.sum()
            y_eff = (torch.log1p(cfg.effect_target_sum * pert / sp)
                     - torch.log1p(cfg.effect_target_sum * m_c / sc)[None, :])
            gpu_panels[ctx] = {"genes": P["genes"], "truth": truth_t, "prior": prior_t,
                               "esm": esm_t, "basal": basal_t, "m_c": m_c, "y_eff": y_eff,
                               "sc": sc}
            del P["truth"], P["prior"], P["esm"]
    del panels
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = ResidualCellDynamics(cfg).to(device)
    print(f"[res-dyn] params {model.count_params()/1e6:.1f}M device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * cfg.steps_per_epoch)
    rng = np.random.default_rng(cfg.seed)
    ctx_list = sorted(gpu_panels)
    ctx_w = np.array([len(gpu_panels[c]["genes"]) for c in ctx_list], dtype=np.float64)
    ctx_w /= ctx_w.sum()
    G = len(gene_names)
    last = float("nan")
    for epoch in range(1, cfg.epochs + 1):
        tot = 0.0
        for _ in range(cfg.steps_per_epoch):
            c = ctx_list[rng.choice(len(ctx_list), p=ctx_w)]
            P = gpu_panels[c]
            K = min(cfg.panel_size, len(P["genes"]))
            sel = torch.from_numpy(rng.choice(len(P["genes"]), size=K, replace=False)).to(device)
            genes_b = [P["genes"][i] for i in sel.cpu().tolist()]
            pr_b = P["prior"][sel]
            b_b = P["basal"][None, :].expand(K, -1)
            e_b = P["esm"][sel]
            y_b = P["truth"][sel]
            y_eff = P["y_eff"][sel]
            opt.zero_grad()
            corr = model(pr_b, b_b, e_b)
            pred = pr_b + corr
            # effect-space ranking + cosine on effects; l1 on LFC
            m_c = P["m_c"]
            pert = m_c[None, :] * torch.exp2(torch.clamp(pred, -6.0, 6.0))
            sp = pert.sum(dim=1, keepdim=True).clamp_min(1.0)
            prof_p = torch.log1p(cfg.effect_target_sum * pert / sp)
            prof_c = torch.log1p(cfg.effect_target_sum * m_c / P["sc"])
            p_eff = prof_p - prof_c[None, :]
            loss, parts = combined_loss(
                p_eff, y_eff, w_rank=cfg.w_rank, w_cos=cfg.w_cos, w_l1=0.0,
                temperature=cfg.rank_temperature, gene_names=gene_names, batch_genes=genes_b,
            )
            # l1 on LFC (keeps magnitudes honest)
            if cfg.w_l1 > 0:
                l1 = F.l1_loss(pred, y_b)
                parts["l1"] = float(l1.item())
                loss = loss + cfg.w_l1 * l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.item())
        last = tot / cfg.steps_per_epoch
        if epoch == 1 or epoch % 2 == 0 or epoch == cfg.epochs:
            print(f" epoch {epoch}/{cfg.epochs} loss={last:.4f} (rank {parts.get('rank',0):.3f} cos {parts.get('cos',0):.3f} l1 {parts.get('l1',0):.3f})", flush=True)
    model.eval()
    # free GPU panels before returning (caller may train next member)
    del gpu_panels
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, {"final_loss": last, "n_params": model.count_params(), "n_panels": len(ctx_list)}
