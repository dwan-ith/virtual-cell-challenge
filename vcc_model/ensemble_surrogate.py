"""Deep ensemble of CellDynamics + OOD detection (bitsandatomss ensemble.py / ood.py).

- 5 members, different seeds + data shuffle
- Mean = prediction, std = uncertainty
- OOD novelty on basal embeddings (per-dim z-score + radial), 99th percentile threshold

Persistence: ensemble saved as directory of .pt files + meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .dynamics import CellDynamics, DynamicsConfig, train_cell_dynamics


class CellOODDetector:
    """Novelty on training basal distribution. Mirrors Kernel virtual_kernel/ood.py."""

    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.threshold_: float = float("inf")

    def fit(self, train_basals: np.ndarray) -> "CellOODDetector":
        X = np.asarray(train_basals, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1e-9
        self.threshold_ = float(np.quantile(self.novelty(X), self.quantile))
        return self

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "fit() first"
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def novelty(self, X) -> np.ndarray:
        Z = self._standardize(np.atleast_2d(np.asarray(X, dtype=np.float64)))
        per_dim = np.abs(Z).max(axis=1)
        radial = np.sqrt((Z ** 2).sum(axis=1))
        return np.maximum(per_dim, radial / np.sqrt(Z.shape[1]))

    def flagged(self, X) -> np.ndarray:
        return self.novelty(X) > self.threshold_

    def report(self, train: np.ndarray, test: np.ndarray) -> dict:
        n_tr = self.novelty(train)
        n_te = self.novelty(test)
        sep = float((n_te[:, None] > n_tr[None, :]).mean()) if len(n_tr) and len(n_te) else 0.5
        return {
            "threshold": self.threshold_,
            "train_flag_rate": float((n_tr > self.threshold_).mean()),
            "test_flag_rate": float((n_te > self.threshold_).mean()),
            "train_novelty_median": float(np.median(n_tr)),
            "test_novelty_median": float(np.median(n_te)),
            "separation": sep,
            "n_train": len(n_tr), "n_test": len(n_te),
        }


class EnsembleDynamics:
    """5-member CellDynamics ensemble."""

    def __init__(self, members: list[CellDynamics]):
        assert members, "need >=1 member"
        self.members = members
        self.ood: CellOODDetector | None = None
        self.gene_names: list[str] | None = None
        self._esm: dict[str, np.ndarray] | None = None

    @classmethod
    def train(
        cls,
        lfc_pairs: dict[tuple[str, str], np.ndarray],
        basals: dict[str, np.ndarray],
        gene_names: list[str],
        esm_embeddings: dict[str, np.ndarray] | None = None,
        n_members: int = 5,
        base_cfg: DynamicsConfig | None = None,
        verbose: bool = False,
    ) -> "EnsembleDynamics":
        base_cfg = base_cfg or DynamicsConfig()
        members: list[CellDynamics] = []
        for i in range(n_members):
            cfg = DynamicsConfig(**{**base_cfg.__dict__, "seed": base_cfg.seed + 1000 + i})
            # small data-shuffle variation already via seed in _build_training_arrays
            m, info = train_cell_dynamics(lfc_pairs, basals, gene_names, esm_embeddings, cfg)
            if verbose:
                print(f"[ensemble] member {i} done {info}", flush=True)
            members.append(m)
        ens = cls(members)
        ens.gene_names = list(gene_names)
        ens._esm = esm_embeddings
        # fit OOD on training basals
        try:
            train_mat = np.stack([basals[c] for c in sorted(basals)])
            ens.ood = CellOODDetector().fit(train_mat)
            print(f"[ensemble] OOD threshold {ens.ood.threshold_:.3f}", flush=True)
        except Exception:
            ens.ood = None
        return ens

    def __len__(self):
        return len(self.members)

    # -- prediction --
    @torch.no_grad()
    def predict_lfc(self, gene: str, basal_log: np.ndarray) -> np.ndarray:
        """Mean over members."""
        esm_vec = None
        if self._esm is not None and gene in self._esm:
            esm_vec = self._esm[gene]
        outs = []
        for m in self.members:
            m.eval()
            outs.append(m.predict_lfc(basal_log, esm_vec))
        return np.mean(np.stack(outs), axis=0)

    @torch.no_grad()
    def predict_with_uncertainty(self, gene: str, basal_log: np.ndarray) -> tuple[np.ndarray, float]:
        esm_vec = None
        if self._esm is not None and gene in self._esm:
            esm_vec = self._esm[gene]
        outs = []
        for m in self.members:
            m.eval()
            outs.append(m.predict_lfc(basal_log, esm_vec))
        arr = np.stack(outs)
        mean = arr.mean(axis=0)
        # mean per-gene std across members
        unc = float(arr.std(axis=0).mean())
        return mean, unc

    def predict_many(self, genes: list[str], basal_log: np.ndarray) -> dict[str, np.ndarray]:
        return {g: self.predict_lfc(g, basal_log) for g in genes}

    def uncertainty(self, basal_log: np.ndarray, gene: str) -> float:
        _, u = self.predict_with_uncertainty(gene, basal_log)
        return u

    def ood_flag(self, basal_log: np.ndarray) -> bool:
        if self.ood is None:
            return False
        return bool(self.ood.flagged(basal_log.reshape(1, -1))[0])

    # -- persistence (directory) --
    def save(self, dir_path: str | Path):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_members": len(self.members),
            "gene_names": self.gene_names,
            "ood": None if self.ood is None else {
                "quantile": self.ood.quantile,
                "mean": self.ood.mean_.tolist() if self.ood.mean_ is not None else None,
                "std": self.ood.std_.tolist() if self.ood.std_ is not None else None,
                "threshold": self.ood.threshold_,
            },
        }
        (p / "meta.json").write_text(json.dumps(meta))
        for i, m in enumerate(self.members):
            m.save(str(p / f"member_{i}.pt"))

    @classmethod
    def load(cls, dir_path: str | Path, map_location=None) -> "EnsembleDynamics":
        p = Path(dir_path)
        meta = json.loads((p / "meta.json").read_text())
        members = []
        for i in range(meta["n_members"]):
            m = CellDynamics.load(str(p / f"member_{i}.pt"), map_location=map_location)
            members.append(m)
        ens = cls(members)
        ens.gene_names = meta.get("gene_names")
        ood_meta = meta.get("ood")
        if ood_meta is not None and ood_meta.get("mean") is not None:
            det = CellOODDetector(quantile=ood_meta["quantile"])
            det.mean_ = np.array(ood_meta["mean"], dtype=np.float64)
            det.std_ = np.array(ood_meta["std"], dtype=np.float64)
            det.threshold_ = float(ood_meta["threshold"])
            ens.ood = det
        # ESM not persisted (large); caller can reattach: ens._esm = load_esm(...)
        return ens
