"""CellDynamics: latent E->F->D surrogate for Virtual Cell (bitsandatomss pattern).

Architecture (100M params, GPU):
  E: basal_log[18533] + proj(ESM5120->256) -> latent[512]  (4-layer MLP, LayerNorm+GELU, tanh)
  F: (latent[512], quad_pert[512]) -> latent'[512]  gated residual z' = z + gate*tanh(dz)
  D: latent'[512] -> ΔLFC[18533]  (full-rank, 2-layer)

Training data: (basal, gene, observed_LFC) from PseudobulkStore pairs.
Prediction: basal_log + ESM(gene) -> ΔLFC; counts via countsim (2^LFC multipliers).

Single source of ESM projection (Linear 5120->256) shared by E and F quadratics.
No PCA compression — decoder emits full 18533 gene vector.
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
class DynamicsConfig:
    gene_dim: int = 18533
    esm_dim: int = 5120
    esm_proj_dim: int = 256
    latent_dim: int = 512
    # Encoder widths: input -> h1 -> h2 -> h3 -> latent
    enc_h1: int = 2048
    enc_h2: int = 2048
    enc_h3: int = 1024
    # Transition widths
    trans_h1: int = 2048
    trans_h2: int = 2048
    trans_h3: int = 1024
    # Decoder widths
    dec_hidden: int = 2048
    dropout: float = 0.10
    # Training
    epochs: int = 80
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Loss weights (see ranking_loss.py)
    w_rank: float = 0.5
    w_cos: float = 1.0
    w_l1: float = 0.3
    rank_temperature: float = 0.1
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    # Data caps
    max_pairs: int = 20000


# ------------------------------------------------------------------ blocks
class _Encoder(nn.Module):
    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        in_dim = cfg.gene_dim + cfg.esm_proj_dim
        self.fc1 = nn.Linear(in_dim, cfg.enc_h1)
        self.ln1 = nn.LayerNorm(cfg.enc_h1)
        self.fc2 = nn.Linear(cfg.enc_h1, cfg.enc_h2)
        self.ln2 = nn.LayerNorm(cfg.enc_h2)
        self.fc3 = nn.Linear(cfg.enc_h2, cfg.enc_h3)
        self.ln3 = nn.LayerNorm(cfg.enc_h3)
        self.fc4 = nn.Linear(cfg.enc_h3, cfg.latent_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.drop(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop(F.gelu(self.ln2(self.fc2(x))))
        x = self.drop(F.gelu(self.ln3(self.fc3(x))))
        z = torch.tanh(self.fc4(x))
        return z


class _Transition(nn.Module):
    """Gated residual in latent space: z' = z + gate * tanh(dz).
    Input: concat[z(ldim), p_quad(2*esm_proj)].
    Mirrors Kernel dynamics.py: residual + bounded latent, plus quadratic action skip.
    """
    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        in_dim = cfg.latent_dim + 2 * cfg.esm_proj_dim
        self.fc1 = nn.Linear(in_dim, cfg.trans_h1)
        self.ln1 = nn.LayerNorm(cfg.trans_h1)
        self.fc2 = nn.Linear(cfg.trans_h1, cfg.trans_h2)
        self.ln2 = nn.LayerNorm(cfg.trans_h2)
        self.fc3 = nn.Linear(cfg.trans_h2, cfg.trans_h3)
        self.ln3 = nn.LayerNorm(cfg.trans_h3)
        self.gate = nn.Linear(cfg.trans_h3, cfg.latent_dim)
        self.dz = nn.Linear(cfg.trans_h3, cfg.latent_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, z, p_quad):
        x = torch.cat([z, p_quad], dim=1)
        h = self.drop(F.gelu(self.ln1(self.fc1(x))))
        h = self.drop(F.gelu(self.ln2(self.fc2(h))))
        h = self.drop(F.gelu(self.ln3(self.fc3(h))))
        gate = torch.sigmoid(self.gate(h))
        dz = self.dz(h)
        z_next = z + gate * torch.tanh(dz)
        # keep latent bounded in [-1,1] like Kernel's tanh(z+0.5*dz)
        z_next = torch.tanh(z_next)
        return z_next


class _Decoder(nn.Module):
    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.latent_dim, cfg.dec_hidden)
        self.ln1 = nn.LayerNorm(cfg.dec_hidden)
        self.fc2 = nn.Linear(cfg.dec_hidden, cfg.gene_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, z):
        h = self.drop(F.gelu(self.ln1(self.fc1(z))))
        delta = self.fc2(h)
        return delta


# ------------------------------------------------------------------ model
class CellDynamics(nn.Module):
    """E->F->D dynamics: basal + ESM(gene) -> ΔLFC."""

    def __init__(self, cfg: DynamicsConfig | None = None):
        super().__init__()
        self.cfg = cfg or DynamicsConfig()
        cfg = self.cfg
        self.esm_proj = nn.Linear(cfg.esm_dim, cfg.esm_proj_dim)
        self.encoder = _Encoder(cfg)
        self.transition = _Transition(cfg)
        self.decoder = _Decoder(cfg)
        self._init_weights()

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

    # -- forward pieces (for analysis) --
    def encode(self, basal_log: torch.Tensor, esm_vec: torch.Tensor) -> torch.Tensor:
        """basal_log (B,G), esm_vec (B,esm_dim) -> latent (B,ldim)."""
        p = self.esm_proj(esm_vec)  # (B,256)
        # normalize ESM proj for stability
        p = F.normalize(p, dim=1) * math.sqrt(p.shape[1]) * 0.1
        x = torch.cat([basal_log, p], dim=1)
        return self.encoder(x)

    def forward(self, basal_log: torch.Tensor, esm_vec: torch.Tensor) -> torch.Tensor:
        """Full E->F->D: (B,G)+(B,esm_dim) -> ΔLFC (B,G)."""
        p = self.esm_proj(esm_vec)
        p = F.normalize(p, dim=1) * math.sqrt(p.shape[1]) * 0.1
        z = self.encoder(torch.cat([basal_log, p], dim=1))
        # quadratic action skip V@[p,p²] like Kernel
        p_quad = torch.cat([p, p * p], dim=1)
        z_next = self.transition(z, p_quad)
        delta = self.decoder(z_next)
        return delta

    # -- numpy convenience (single prediction) --
    @torch.no_grad()
    def predict_lfc(self, basal_log: np.ndarray, esm_vec: np.ndarray | None) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        b = torch.from_numpy(np.asarray(basal_log, dtype=np.float32)).unsqueeze(0).to(device)
        if esm_vec is None:
            esm_vec = np.zeros(self.cfg.esm_dim, dtype=np.float32)
        e = torch.from_numpy(np.asarray(esm_vec, dtype=np.float32)).unsqueeze(0).to(device)
        delta = self.forward(b, e)
        return delta.squeeze(0).cpu().numpy().astype(np.float64)

    # -- persistence --
    def save(self, path: str):
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "CellDynamics":
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        cfg = ckpt["cfg"] if isinstance(ckpt, dict) and "cfg" in ckpt else DynamicsConfig()
        m = cls(cfg)
        state = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
        m.load_state_dict(state)
        return m


# ------------------------------------------------------------------ training helpers
def _build_training_arrays(
    lfc_pairs: dict[tuple[str, str], np.ndarray],
    basals: dict[str, np.ndarray],
    gene_names: list[str],
    esm_embeddings: dict[str, np.ndarray] | None,
    cfg: DynamicsConfig,
    max_pairs: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (basal_mat (N,G), esm_mat (N,esm_dim), lfc_mat (N,G), genes)."""
    keys = sorted(lfc_pairs)
    if max_pairs is not None and len(keys) > max_pairs:
        rng = np.random.default_rng(seed)
        # stratified by context
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
        keys = sorted(picked)
    N = len(keys)
    G = len(gene_names)
    basal_mat = np.zeros((N, G), dtype=np.float32)
    esm_mat = np.zeros((N, cfg.esm_dim), dtype=np.float32)
    lfc_mat = np.zeros((N, G), dtype=np.float32)
    genes: list[str] = []
    for i, (ctx, gene) in enumerate(keys):
        basal_mat[i] = np.asarray(basals[ctx], dtype=np.float32)
        if esm_embeddings is not None and gene in esm_embeddings:
            v = np.asarray(esm_embeddings[gene], dtype=np.float32)
            # raw ESM, model will project+normalize
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                # keep raw, let linear learn scale; just avoid huge norms
                v = v / max(n, 1.0) * 3.0
            esm_mat[i] = v
        lfc_mat[i] = np.asarray(lfc_pairs[(ctx, gene)], dtype=np.float32)
        genes.append(gene)
    # sanitize
    np.nan_to_num(basal_mat, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.nan_to_num(lfc_mat, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
    np.clip(lfc_mat, -8.0, 8.0, out=lfc_mat)
    return basal_mat, esm_mat, lfc_mat, genes


def train_cell_dynamics(
    lfc_pairs: dict[tuple[str, str], np.ndarray],
    basals: dict[str, np.ndarray],
    gene_names: list[str],
    esm_embeddings: dict[str, np.ndarray] | None = None,
    cfg: DynamicsConfig | None = None,
    val_pairs: dict[tuple[str, str], np.ndarray] | None = None,
    val_basals: dict[str, np.ndarray] | None = None,
) -> tuple[CellDynamics, dict]:
    """Train one CellDynamics; returns (model, info)."""
    from .ranking_loss import combined_loss
    cfg = cfg or DynamicsConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    # Build arrays
    basal_mat, esm_mat, lfc_mat, genes = _build_training_arrays(
        lfc_pairs, basals, gene_names, esm_embeddings, cfg, max_pairs=cfg.max_pairs, seed=cfg.seed
    )
    N = len(genes)
    if N < 8:
        raise ValueError(f"too few pairs {N}")
    print(f"[dynamics] train arrays N={N} G={len(gene_names)} esm={cfg.esm_dim}->{cfg.esm_proj_dim} latent={cfg.latent_dim}", flush=True)
    model = CellDynamics(cfg).to(device)
    n_params = model.count_params()
    print(f"[dynamics] params {n_params/1e6:.1f}M device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    # tensors on device
    b_t = torch.from_numpy(basal_mat).to(device)
    e_t = torch.from_numpy(esm_mat).to(device)
    y_t = torch.from_numpy(lfc_mat).to(device)
    # val tensors if any
    val_b = val_e = val_y = None
    if val_pairs is not None and val_basals is not None and len(val_pairs) >= 4:
        vb, ve, vy, _ = _build_training_arrays(val_pairs, val_basals, gene_names, esm_embeddings, cfg, max_pairs=None, seed=cfg.seed+1)
        val_b = torch.from_numpy(vb).to(device)
        val_e = torch.from_numpy(ve).to(device)
        val_y = torch.from_numpy(vy).to(device)
        print(f"[dynamics] val N={len(vb)}", flush=True)
    bs = min(cfg.batch_size, N)
    steps_per = max(1, N // bs)
    rng = np.random.default_rng(cfg.seed)
    best_loss = float("inf")
    info: dict = {"n_train": N, "n_params": n_params}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = rng.permutation(N)
        tot = 0.0
        for s0 in range(0, N - bs + 1, bs):
            idx = perm[s0:s0+bs]
            opt.zero_grad()
            pred = model(b_t[idx], e_t[idx])
            loss, parts = combined_loss(pred, y_t[idx], w_rank=cfg.w_rank, w_cos=cfg.w_cos, w_l1=cfg.w_l1, temperature=cfg.rank_temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
        sched.step()
        avg = tot / steps_per
        if epoch % 10 == 0 or epoch == 1:
            msg = f" epoch {epoch}/{cfg.epochs} loss={avg:.4f}"
            if val_b is not None:
                model.eval()
                with torch.no_grad():
                    # cap val batch for ranking loss stability
                    vp = model(val_b, val_e)
                    vl, vparts = combined_loss(vp, val_y, w_rank=cfg.w_rank, w_cos=cfg.w_cos, w_l1=cfg.w_l1, temperature=cfg.rank_temperature)
                    msg += f" val={float(vl):.4f} (rank {vparts['rank']:.3f} cos {vparts['cos']:.3f} l1 {vparts['l1']:.3f})"
                    if float(vl) < best_loss:
                        best_loss = float(vl)
            else:
                msg += f" (rank {parts['rank']:.3f} cos {parts['cos']:.3f} l1 {parts['l1']:.3f})"
            print(msg, flush=True)
    model.eval()
    info["final_loss"] = avg
    info["best_val"] = best_loss
    return model, info
