"""Surrogate DIRECTION corrector: angular, not scalar.

Learns Delta = unit(true - prior) in PCA-32 space where PDS lives, with cosine loss.
At inference: pred' = prior + gamma * ||prior|| * proj^{-1}(Delta_hat) gated by orthogonality.
Scalar gain is PDS-invariant (cos(alpha*a,b)=cos(a,b)); only angular moves matter.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

class _DirNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        v = self.net(x)
        # L2-normalize to unit direction (PDS cares about angle)
        n = v.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return v / n

class _Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]

class SurrogateCorrector:
    """Direction surrogate. Falsifiable gate: LOCO DeltaPDS >0.015 before submitting."""
    def __init__(self, esm_proj_dim: int = 32, hidden: int = 64, lr: float = 1e-3, epochs: int = 60, batch_size: int = 256, seed: int = 0, gamma: float = 0.35, orth_thresh: float = 0.2):
        self.esm_proj_dim = esm_proj_dim
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.gamma = gamma
        self.orth_thresh = orth_thresh
        self.net = None
        self.pca = None  # PCA 18533 -> 32 for Delta target
        self._esm_proj_weight = None
        self._esm_proj_mean = None
        self._gene_idx = {}
        self._gene_ctx_var = None
        self._gene_mean_abs = None

    def _featurize(self, gene, basal, sim, gene_emb):
        gi = self._gene_idx.get(gene, -1)
        basal_gene = np.array([basal[gi]] if gi>=0 else [0.0], dtype=np.float32)
        sim_l1 = np.array([np.mean(np.abs(sim))], dtype=np.float32)
        sim_frac = np.array([float(np.mean(sim!=0))], dtype=np.float32)
        ctx_var = np.array([self._gene_ctx_var[gi] if gi>=0 and self._gene_ctx_var is not None else 0.0], dtype=np.float32)
        mean_abs = np.array([self._gene_mean_abs[gi] if gi>=0 and self._gene_mean_abs is not None else 0.0], dtype=np.float32)
        # ESM proj 32 + sim PCA 32 = 70d total
        esm_vec = np.zeros(self.esm_proj_dim, dtype=np.float32)
        if gene_emb is not None and self._esm_proj_weight is not None:
            esm_vec = (gene_emb - self._esm_proj_mean) @ self._esm_proj_weight.T
        # sim PCA coords 32
        sim_pca = np.zeros(32, dtype=np.float32)
        if hasattr(self, '_sim_pca') and self._sim_pca is not None:
            # quick 32-d random proj of sim for feature (not target PCA)
            # use first 32 of sim downsampled
            sim_pca = sim[::578][:32].astype(np.float32)  # 18533/32~579
            if sim_pca.shape[0] < 32:
                sim_pca = np.pad(sim_pca, (0, 32-sim_pca.shape[0]))
        return np.concatenate([basal_gene, sim_l1, sim_frac, ctx_var, mean_abs, esm_vec, sim_pca])

    def fit(self, train_pairs, basals, gene_names, prior, esm_embeddings=None):
        rng = np.random.RandomState(self.seed)
        self._gene_idx = {g:i for i,g in enumerate(gene_names)}
        G=len(gene_names)
        # gene stats
        gene_lfcs={}
        for (ctx,gene),lfc in train_pairs.items():
            gene_lfcs.setdefault(gene, []).append(lfc)
        self._gene_ctx_var=np.zeros(G,dtype=np.float32)
        self._gene_mean_abs=np.zeros(G,dtype=np.float32)
        for gene,lfcs in gene_lfcs.items():
            gi=self._gene_idx.get(gene,-1)
            if gi<0: continue
            mat=np.stack(lfcs)
            self._gene_ctx_var[gi]=float(np.mean(np.var(mat,axis=0)))
            self._gene_mean_abs[gi]=float(np.mean(np.abs(mat)))
        print(f"[dir-surrogate] gene stats {len(gene_lfcs)}", flush=True)
        # ESM proj
        if esm_embeddings:
            vecs=[esm_embeddings[g] for g in gene_names if g in esm_embeddings]
            if vecs:
                m=np.stack(vecs).astype(np.float32)
                self._esm_proj_mean=m.mean(axis=0)
                mc=m-self._esm_proj_mean
                _,_,Vt=randomized_svd(mc, n_components=self.esm_proj_dim, random_state=self.seed)
                self._esm_proj_weight=Vt.astype(np.float32)
                print(f"[dir-surrogate] ESM {m.shape[1]}->{self.esm_proj_dim}", flush=True)
        # Build X, Y (Y = unit(true-prior) in PCA32)
        all_keys=list(train_pairs.keys())
        if len(all_keys)>2000:
            idx=rng.choice(len(all_keys),2000,replace=False)
            all_keys=[all_keys[i] for i in idx]
        by_ctx={}
        for ctx,gene in all_keys:
            by_ctx.setdefault(ctx, []).append((gene, train_pairs[(ctx,gene)]))
        Xs=[]; Deltas=[]
        for ctx, gene_truths in by_ctx.items():
            basal=basals.get(ctx)
            if basal is None: continue
            gene_list=[g for g,_ in gene_truths]
            sims=prior.predict_many(gene_list, basal)
            for gene,truth in gene_truths:
                sim=sims.get(gene)
                if sim is None: continue
                delta=truth - sim
                if np.linalg.norm(delta)<1e-6: continue
                # skip near-collinear (already good)
                # keep all for now
                gene_emb=esm_embeddings.get(gene) if esm_embeddings else None
                feat=self._featurize(gene, basal, sim, gene_emb)
                Xs.append(feat)
                Deltas.append(delta)
        Deltas=np.stack(Deltas).astype(np.float32)  # (N,18533)
        print(f"[dir-surrogate] {len(Xs)} samples, Deltas {Deltas.shape}", flush=True)
        # PCA 18533->32 on Deltas
        self.pca=PCA(n_components=32, random_state=self.seed)
        Yp=self.pca.fit_transform(Deltas)
        # normalize to unit per row for cosine loss target
        Yn=Yp / np.linalg.norm(Yp,axis=1,keepdims=True).clip(min=1e-6)
        print(f"[dir-surrogate] PCA var {self.pca.explained_variance_ratio_.sum():.3f}", flush=True)
        X_all=np.stack(Xs).astype(np.float32)
        print(f" X {X_all.shape} Y {Yn.shape} feat {X_all.shape[1]}", flush=True)
        self.net=_DirNet(X_all.shape[1], 32, self.hidden)
        ds=_Dataset(X_all, Yn.astype(np.float32))
        dl=DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt=torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for epoch in range(self.epochs):
            tot=0; nb=0
            for xb,yb in dl:
                pred=self.net(xb)
                # cosine loss: 1 - cos
                cos=(pred*yb).sum(dim=1).mean()
                loss=1 - cos
                opt.zero_grad(); loss.backward(); opt.step()
                tot+=float(loss); nb+=1
            if (epoch+1)%20==0:
                print(f" epoch {epoch+1}/{self.epochs} loss={tot/max(nb,1):.4f} cos={1-tot/max(nb,1):.3f}", flush=True)
        self.net.eval()
        with torch.no_grad():
            # quick train cos
            xb=torch.as_tensor(X_all[:512], dtype=torch.float32)
            yb=torch.as_tensor(Yn[:512], dtype=torch.float32)
            pred=self.net(xb)
            cos=(pred*yb).sum(dim=1).mean().item()
            print(f"[dir-surrogate] train cos {cos:.3f}", flush=True)
        return self

    def predict_correction(self, gene, basal, sim_pred, gene_emb=None):
        if self.net is None or self.pca is None:
            return np.zeros_like(sim_pred)
        feat=self._featurize(gene, basal, sim_pred, gene_emb)
        x=torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y=self.net(x).numpy()[0]  # 32 unit
        scale=self.gamma * float(np.linalg.norm(sim_pred))
        delta=self.pca.inverse_transform((y*scale).reshape(1,-1))[0]
        # gate by orthogonality: if delta strongly collinear with sim, suppress (would be scalar)
        # cos between delta and sim
        den=np.linalg.norm(delta)*np.linalg.norm(sim_pred)
        if den>1e-9:
            c=float(np.dot(delta, sim_pred)/den)
            if abs(c) > (1-self.orth_thresh):  # near parallel -> likely scalar, suppress
                return np.zeros_like(sim_pred)
        return delta

    # legacy compat
    def predict_gain(self, gene, basal, sim_pred, gene_emb=None):
        return 1.0
