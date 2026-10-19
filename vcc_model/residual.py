"""Low-rank context-conditioned residual network over LFC space.

The statistical prior (prior.LFCTransferPrior) supplies the zero-shot
prediction; this module learns, from source (context, perturbation) pairs, the
part of the response the prior misses. Everything lives in gene-LFC space:

  1. An LFC basis (truncated SVD of stacked observed LFC vectors) captures the
     manifold of perturbation responses. Predictions are decoded through it,
     which regularizes outputs to realistic response directions.
  2. Features per (context, perturbation):
       - basal PCs of the context's control profile   (context identity)
       - basis coordinates of the prior's LFC guess   (perturbation identity)
       - a learned embedding of the target gene       (gene-ID effects; unseen
         genes share an <unk> row and lean on the prior features)
  3. A small MLP maps features -> basis coefficients. Training minimizes
     effect-weighted gene-space MSE plus a soft sign hinge that rewards
     getting directions right on strongly-changing genes (what fidelity /
     reach / jaccard reward), plus coefficient-space MSE for stability.

The prior inputs are computed leave-one-context-out during training so the
network never sees a prior that was allowed to peek at the pair it must
explain — matching true zero-shot inference on unseen contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

UNK_GENE = "<unk>"


# ------------------------------------------------------------------ basis


@dataclass
class TruncatedProjector:
    """Mean + top components of a data matrix; used for both LFC and basal."""

    n_components: int = 32
    mean_: np.ndarray | None = None          # (D,)
    components_: np.ndarray | None = None    # (R, D)

    def fit(self, x: np.ndarray) -> "TruncatedProjector":
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        r = min(self.n_components, min(xc.shape))
        if r <= 0:
            self.components_ = np.zeros((0, xc.shape[1]))
            return self
        # Randomized SVD: avoids the O(n^2) left-vector matrix a full
        # np.linalg.svd materializes for large pair counts.
        from sklearn.utils.extmath import randomized_svd

        _, _, vt = randomized_svd(xc, n_components=r, n_iter=3, random_state=0)
        self.components_ = vt
        return self

    @property
    def n_actual(self) -> int:
        return 0 if self.components_ is None else int(self.components_.shape[0])

    def transform(self, vec: np.ndarray) -> np.ndarray:
        return (np.asarray(vec, dtype=np.float64) - self.mean_) @ self.components_.T

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        return self.mean_ + np.asarray(coords) @ self.components_


# ------------------------------------------------------------------ config


@dataclass
class ResidualConfig:
    lfc_rank: int = 32
    basal_pcs: int = 16
    embed_dim: int = 16
    hidden: int = 256
    dropout: float = 0.10
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    # ESM protein embeddings: if set, adds esm_proj to feature stack
    esm_dim: int = 0  # 0 = disabled, 320 = t6_8M, 1280 = t33_650M
    esm_path: str | None = None  # path to .npz with gene_names, embeddings
    # Cap on training pairs kept resident as one dense (N x G) float32 block.
    # The full store can exceed RAM on 16 GB machines; a stratified random
    # subset preserves context/gene coverage while bounding peak memory.
    max_pairs: int = 12000
    # Gene-space MSE weights genes proportionally to (1 + |y| / effect_scale),
    # focusing capacity on responsive genes, which drive 5 of 6 scored metrics.
    effect_scale: float = 0.5
    coef_loss_weight: float = 0.25
    dir_hinge_weight: float = 2.0
    dir_margin: float = 0.10
    dir_gene_threshold: float = 0.20
    seed: int = 0
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


class _CoefMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ model


class ResidualLFCModel:
    """Fits and applies the context-conditioned LFC corrector."""

    def __init__(self, cfg: ResidualConfig | None = None):
        self.cfg = cfg or ResidualConfig()
        self.gene_names: list[str] = []
        self.lfc_basis = TruncatedProjector(self.cfg.lfc_rank)
        self.basal_basis = TruncatedProjector(self.cfg.basal_pcs)
        self.gene_vocab: list[str] = []
        self.net: _CoefMLP | None = None
        self.embed: nn.Embedding | None = None
        self.emb_proj: nn.Linear | None = None
        self._vocab_of: dict[str, int] = {}
        # ESM protein conditioning
        self.esm: dict[str, np.ndarray] | None = None
        self.esm_proj: nn.Linear | None = None
        self._esm_dim: int = 0

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        prior,
        lfc_pairs: dict[tuple[str, str], np.ndarray],
        basals: dict[str, np.ndarray],
        gene_names: list[str],
    ) -> dict:
        """prior: a fitted LFCTransferPrior (used only to build inputs)."""
        self.gene_names = list(gene_names)
        keys = sorted(lfc_pairs)
        n_pairs = len(keys)
        info: dict = {"n_pairs": n_pairs}
        if n_pairs < 8 or len({c for c, _ in keys}) < 2:
            info["skipped"] = "too few pairs"
            return info

        # Bound resident memory: keep at most cfg.max_pairs rows as one dense
        # (N x G) float32 block (~0.9 GB at 12k x 18.5k). Subsample evenly
        # across contexts so every source stays represented.
        rng_fit = np.random.default_rng(self.cfg.seed)
        if n_pairs > self.cfg.max_pairs:
            by_ctx: dict[str, list] = {}
            for k in keys:
                by_ctx.setdefault(k[0], []).append(k)
            per = max(1, self.cfg.max_pairs // max(len(by_ctx), 1))
            picked: list = []
            for c in sorted(by_ctx):
                ks = sorted(by_ctx[c])
                if len(ks) > per:
                    sel = rng_fit.choice(len(ks), size=per, replace=False)
                    ks = [ks[i] for i in sel]
                picked.extend(ks)
            keys = sorted(picked)
            n_pairs = len(keys)
            info["n_pairs_used"] = n_pairs

        y_mat = np.stack([np.asarray(lfc_pairs[k], dtype=np.float32) for k in keys])
        y_mat = np.nan_to_num(y_mat, copy=False, nan=0.0, posinf=np.float32(8.0), neginf=np.float32(-8.0))
        np.clip(y_mat, -8.0, 8.0, out=y_mat)
        self.lfc_basis.fit(y_mat)
        contexts = sorted({c for c, _ in keys})
        self.basal_basis.fit(
            np.nan_to_num(np.stack([basals[c] for c in contexts]), nan=0.0)
        )

        out_dim = self.lfc_basis.n_actual
        # Try to load ESM protein embeddings if configured
        esm_raw: dict[str, np.ndarray] | None = None
        esm_proj_dim = 8  # compressed ESM contribution
        if self.cfg.esm_dim and self.cfg.esm_path:
            try:
                from pathlib import Path
                p = Path(self.cfg.esm_path)
                if p.exists():
                    data = np.load(p, allow_pickle=True)
                    if "gene_names" in data:
                        genes = [str(g) for g in data["gene_names"]]
                        mat = data["embeddings"]
                        # Normalize for cosine stability
                        norms = np.linalg.norm(mat, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-9, 1.0, norms)
                        mat = mat / norms
                        esm_raw = {g: mat[i].astype(np.float32) for i, g in enumerate(genes)}
                        self.esm = esm_raw
                        self._esm_dim = int(mat.shape[1])
                    else:
                        esm_raw = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
                        self.esm = esm_raw
                if self.esm is not None:
                    print(f"[residual] ESM loaded: {len(self.esm)} genes, dim {self._esm_dim}")
            except Exception as e:
                print(f"[residual][warn] ESM load failed {self.cfg.esm_path}: {e}")
                self.esm = None

        # Gene identity enters via nn.Embedding inside the forward pass, so
        # feature width is basal PCs + prior-LFC basis coordinates (+ ESM if enabled).
        base_in_dim = self.basal_basis.n_actual + out_dim
        esm_extra = esm_proj_dim if self.esm is not None else 0
        in_dim = base_in_dim + esm_extra

        vocab = sorted({g for _, g in keys})
        self.gene_vocab = vocab + [UNK_GENE]
        self._vocab_of = {g: i for i, g in enumerate(self.gene_vocab)}
        unk = self._vocab_of[UNK_GENE]

        torch.manual_seed(self.cfg.seed)
        device = self.cfg.device
        self.net = _CoefMLP(in_dim, self.cfg.hidden, out_dim, self.cfg.dropout).to(device)
        self.embed = nn.Embedding(len(self.gene_vocab), self.cfg.embed_dim).to(device)
        self.emb_proj = nn.Linear(self.cfg.embed_dim, out_dim).to(device)
        nn.init.normal_(self.embed.weight, std=0.05)
        if self.esm is not None:
            self.esm_proj = nn.Linear(self._esm_dim, esm_proj_dim).to(device)
            nn.init.kaiming_uniform_(self.esm_proj.weight, a=0.1)
            nn.init.zeros_(self.esm_proj.bias)

        # Training tensors, built with leave-one-context-out prior inputs.
        # Grouped by context so the prior's transfer params are computed once
        # per context (predict_many) instead of once per pair.
        feats = np.zeros((n_pairs, in_dim), dtype=np.float32)
        z_true = np.zeros((n_pairs, out_dim), dtype=np.float32)
        gene_ids = np.zeros(n_pairs, dtype=np.int64)
        by_ctx: dict[str, list[int]] = {}
        for i, (ctx, gene) in enumerate(keys):
            by_ctx.setdefault(ctx, []).append(i)
        for ctx, row_ids in by_ctx.items():
            genes_here = [keys[i][1] for i in row_ids]
            preds_map = prior.predict_many(genes_here, basals[ctx], exclude_contexts={ctx})
            for i in row_ids:
                prior_guess = preds_map[keys[i][1]]
                base_feats = np.concatenate(
                    [self.basal_basis.transform(basals[ctx]), self.lfc_basis.transform(prior_guess)]
                ).astype(np.float32)
                if self.esm is not None and self.esm_proj is not None:
                    gene = keys[i][1]
                    ev = self.esm.get(gene)
                    if ev is not None:
                        # Project ESM through linear on CPU then concat (avoid device matmul per row)
                        ev_t = torch.from_numpy(ev).to(device).float()
                        with torch.no_grad():
                            esm_f = self.esm_proj(ev_t).cpu().numpy()
                        base_feats = np.concatenate([base_feats, esm_f.astype(np.float32)])
                    else:
                        base_feats = np.concatenate([base_feats, np.zeros(esm_proj_dim, dtype=np.float32)])
                feats[i] = base_feats
                gene_ids[i] = self._vocab_of.get(keys[i][1], unk)
        for i in range(n_pairs):
            z_true[i] = self.lfc_basis.transform(y_mat[i]).astype(np.float32)

        # Small tensors live on device; y_mat itself stays on CPU and is sliced
        # per mini-batch (no second full-size copy), so RAM use is bounded by
        # the single y_mat allocation plus device tensors.
        f_t = torch.tensor(feats, device=device)
        g_t = torch.tensor(gene_ids, device=device)
        comp_t = torch.tensor(
            self.lfc_basis.components_, dtype=torch.float32, device=device
        )
        mean_t = torch.tensor(self.lfc_basis.mean_, dtype=torch.float32, device=device)

        params = (
            list(self.net.parameters())
            + list(self.embed.parameters())
            + list(self.emb_proj.parameters())
            + (list(self.esm_proj.parameters()) if self.esm_proj is not None else [])
        )
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)

        rng_np = np.random.default_rng(self.cfg.seed)
        batch_size = int(min(self.cfg.batch_size, max(n_pairs, 1)))
        steps_per_epoch = max(1, n_pairs // batch_size)

        self.net.train()
        self.embed.train()
        self.emb_proj.train()
        last = float("nan")
        for _ in range(self.cfg.epochs):
            perm = rng_np.permutation(n_pairs)
            epoch_loss = 0.0
            for s0 in range(0, n_pairs - batch_size + 1, batch_size):
                idx = perm[s0 : s0 + batch_size]
                opt.zero_grad()
                f_b = f_t[idx]
                g_b = g_t[idx]
                z_hat = self.net(f_b) + self.emb_proj(self.embed(g_b))
                pred_y = z_hat @ comp_t + mean_t
                y_b = torch.from_numpy(y_mat[idx].astype(np.float32, copy=False)).to(device)
                resid = pred_y - y_b
                w_gene = 1.0 + y_b.abs() / self.cfg.effect_scale
                mse = (w_gene * resid * resid).sum() / w_gene.sum().clamp_min(1e-6)
                z_b = torch.from_numpy(z_true[idx]).to(device)
                coef = ((z_hat - z_b) ** 2).mean()
                strong = y_b.abs() > self.cfg.dir_gene_threshold
                dir_w = torch.where(strong, y_b.abs(), torch.zeros_like(y_b))
                sign_y = torch.where(strong, torch.sign(y_b), torch.zeros_like(y_b))
                miss = torch.relu(self.cfg.dir_margin - sign_y * pred_y)
                dloss = (dir_w * miss * miss).sum() / dir_w.sum().clamp_min(1e-6)
                loss = mse + self.cfg.coef_loss_weight * coef + self.cfg.dir_hinge_weight * dloss
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
            sched.step()
            last = epoch_loss / steps_per_epoch

        del y_mat
        info.update({"train_loss": last, "lfc_rank": self.lfc_basis.n_actual})
        return info

    # ------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, prior_lfc: np.ndarray, basal_vec: np.ndarray, gene: str) -> np.ndarray:
        if self.net is None:
            return np.zeros(len(self.gene_names))
        self.net.eval()
        b_pc = self.basal_basis.transform(basal_vec)
        p_pc = self.lfc_basis.transform(prior_lfc)
        base_feats = np.concatenate([b_pc, p_pc]).astype(np.float32)
        if self.esm is not None and self.esm_proj is not None:
            ev = self.esm.get(gene)
            if ev is not None:
                ev_t = torch.from_numpy(ev).to(self.cfg.device).float()
                with torch.no_grad():
                    esm_f = self.esm_proj(ev_t).cpu().numpy()
                base_feats = np.concatenate([base_feats, esm_f.astype(np.float32)])
            else:
                base_feats = np.concatenate([base_feats, np.zeros(self.esm_proj.out_features, dtype=np.float32)])
        feats = base_feats[None, :]
        f_t = torch.tensor(feats, device=self.cfg.device)
        gidx = self._vocab_of.get(gene, self._vocab_of[UNK_GENE])
        g_t = torch.tensor([gidx], device=self.cfg.device)
        z_hat = self.net(f_t) + self.emb_proj(self.embed(g_t))
        return self.lfc_basis.inverse(z_hat.cpu().numpy()[0])

    @torch.no_grad()
    def predict_many(
        self,
        prior_lfcs: dict[str, np.ndarray],
        basal_vec: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Batched prediction for many targets sharing one context basal state."""
        out: dict[str, np.ndarray] = {}
        if self.net is None:
            for g in prior_lfcs:
                out[g] = np.zeros(len(self.gene_names))
            return out
        self.net.eval()
        genes = list(prior_lfcs)
        b_pc = self.basal_basis.transform(basal_vec)
        # Build feats with optional ESM
        base_feats_list = []
        for g in genes:
            bf = np.concatenate([b_pc, self.lfc_basis.transform(prior_lfcs[g])]).astype(np.float32)
            if self.esm is not None and self.esm_proj is not None:
                ev = self.esm.get(g)
                if ev is not None:
                    ev_t = torch.from_numpy(ev).to(self.cfg.device).float()
                    esm_f = self.esm_proj(ev_t).cpu().numpy()
                    bf = np.concatenate([bf, esm_f.astype(np.float32)])
                else:
                    bf = np.concatenate([bf, np.zeros(self.esm_proj.out_features, dtype=np.float32)])
            base_feats_list.append(bf)
        feats = np.stack(base_feats_list).astype(np.float32)
        gids = [self._vocab_of.get(g, self._vocab_of[UNK_GENE]) for g in genes]
        f_t = torch.tensor(feats, device=self.cfg.device)
        g_t = torch.tensor(gids, dtype=torch.int64, device=self.cfg.device)
        z_hat = self.net(f_t) + self.emb_proj(self.embed(g_t))
        decoded = z_hat.cpu().numpy() @ self.lfc_basis.components_ + self.lfc_basis.mean_
        for i, g in enumerate(genes):
            out[g] = decoded[i]
        return out

    # ---------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg,
            "gene_names": self.gene_names,
            "gene_vocab": self.gene_vocab,
            "lfc_basis": (self.lfc_basis.mean_, self.lfc_basis.components_),
            "basal_basis": (self.basal_basis.mean_, self.basal_basis.components_),
            "net": None if self.net is None else self.net.state_dict(),
            "embed": None if self.embed is None else self.embed.state_dict(),
            "emb_proj": None if self.emb_proj is None else self.emb_proj.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.cfg = state["cfg"]
        self.gene_names = state["gene_names"]
        self.gene_vocab = state["gene_vocab"]
        self._vocab_of = {g: i for i, g in enumerate(self.gene_vocab)}
        self.lfc_basis = TruncatedProjector(self.cfg.lfc_rank)
        self.lfc_basis.mean_, self.lfc_basis.components_ = state["lfc_basis"]
        self.basal_basis = TruncatedProjector(self.cfg.basal_pcs)
        self.basal_basis.mean_, self.basal_basis.components_ = state["basal_basis"]
        if state["net"] is None:
            self.net = self.embed = self.emb_proj = None
            return
        out_dim = self.lfc_basis.n_actual
        in_dim = self.basal_basis.n_actual + out_dim
        self.net = _CoefMLP(in_dim, self.cfg.hidden, out_dim, self.cfg.dropout)
        self.net.load_state_dict(state["net"])
        self.embed = nn.Embedding(len(self.gene_vocab), self.cfg.embed_dim)
        self.embed.load_state_dict(state["embed"])
        self.emb_proj = nn.Linear(self.cfg.embed_dim, out_dim)
        self.emb_proj.load_state_dict(state["emb_proj"])
