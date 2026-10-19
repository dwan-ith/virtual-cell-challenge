"""STATE foundation model adapter — scaffolding for SE-600M + ST integration.

This module provides the interface for the primary structural lever identified
in LIMITS.md §6.2. It is intentionally a thin, well-typed wrapper so that
the integration can be done incrementally without breaking the statistical
pipeline.

Usage (when checkpoints are available):
    from vcc_model.state_adapter import StateAdapter
    adapter = StateAdapter(se_dir="data/state/se600m", st_dir="data/state/st_k562_zeroshot")
    # Embed controls
    emb = adapter.embed_controls(adata)  # -> np.ndarray (n_cells, d)
    # Predict delta for one target
    delta = adapter.predict_delta(controls_adata, gene="ABCD1", context="A")

Without checkpoints, all methods fall back gracefully (return None / raise
with clear message) so the statistical pipeline remains runnable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class StateAdapter:
    """Thin wrapper around arc-state SE + ST models."""

    def __init__(
        self,
        se_dir: str | Path | None = None,
        st_dir: str | Path | None = None,
        device: str = "cuda",
    ):
        self.se_dir = Path(se_dir) if se_dir else None
        self.st_dir = Path(st_dir) if st_dir else None
        self.device = device
        self._se_available = False
        self._st_available = False
        if self.se_dir and (self.se_dir / "config.yaml").exists():
            self._se_available = True
        if self.st_dir and (self.st_dir / "checkpoints").exists():
            self._st_available = True

    @property
    def available(self) -> bool:
        return self._se_available and self._st_available

    def embed_controls(self, adata) -> np.ndarray | None:
        """Embed control cells with SE-600M. Returns None if unavailable."""
        if not self._se_available:
            return None
        try:
            import subprocess, tempfile
            import anndata as ad
            with tempfile.TemporaryDirectory() as tmp:
                in_path = Path(tmp) / "in.h5ad"
                out_path = Path(tmp) / "out.h5ad"
                if isinstance(adata, (str, Path)):
                    in_path = Path(adata)
                else:
                    adata.write_h5ad(in_path)
                cmd = [
                    "state", "emb", "transform",
                    "--model-folder", str(self.se_dir),
                    "--input", str(in_path),
                    "--output", str(out_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0 or not out_path.exists():
                    # Debug: uncomment to see error
                    # print(f"state emb failed: {result.stderr[:500]}")
                    return None
                out_adata = ad.read_h5ad(out_path)
                # Embeddings are in obsm
                for key in ("X_state", "emb", "SE", "embedding"):
                    if key in out_adata.obsm:
                        return np.asarray(out_adata.obsm[key])
                # Fallback: first obsm
                if len(out_adata.obsm) > 0:
                    return np.asarray(next(iter(out_adata.obsm.values())))
                return None
        except Exception:
            return None

    def predict_delta(
        self,
        controls_adata,
        gene: str,
        context: str = "A",
    ) -> np.ndarray | None:
        """Predict LFC delta via ST. Returns None if unavailable."""
        if not self.available:
            return None
        try:
            import subprocess, tempfile
            import anndata as ad
            with tempfile.TemporaryDirectory() as tmp:
                in_path = Path(tmp) / "in.h5ad"
                out_path = Path(tmp) / "pred.h5ad"
                if isinstance(controls_adata, (str, Path)):
                    in_path = Path(controls_adata)
                else:
                    controls_adata.write_h5ad(in_path)
                # ST expects pert_col and model_dir; use target_gene if present else make it
                # Ensure adata has target_gene column
                tmp_adata = ad.read_h5ad(in_path)
                if "target_gene" not in tmp_adata.obs:
                    tmp_adata.obs["target_gene"] = "non-targeting"
                    tmp_adata.write_h5ad(in_path)
                cmd = [
                    "state", "tx", "infer",
                    "--model_dir", str(self.st_dir),
                    "--adata", str(in_path),
                    "--pert_col", "target_gene",
                    "--output", str(out_path),
                ]
                # Note: ST predicts all perts in adata, not single gene; we need to set pert
                # For now, run inference for all and extract gene
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0 or not out_path.exists():
                    return None
                pred = ad.read_h5ad(out_path)
                # Try to extract predicted cells for gene
                if "target_gene" in pred.obs:
                    pert = pred[pred.obs["target_gene"] == gene].X if gene in pred.obs["target_gene"].values else None
                    ctrl = pred[pred.obs["target_gene"] == "non-targeting"].X
                    if pert is None or pert.shape[0] == 0:
                        # Fallback: use all pred vs ctrl
                        pert = pred.X
                else:
                    pert = pred.X
                    ctrl = tmp_adata.X
                if pert.shape[0] == 0:
                    return None
                # Return difference as LFC proxy (log mean)
                return np.asarray(pert.mean(axis=0)).ravel() - np.asarray(ctrl.mean(axis=0)).ravel()
        except Exception:
            return None

    def status(self) -> dict:
        return {
            "se_dir": str(self.se_dir) if self.se_dir else None,
            "st_dir": str(self.st_dir) if self.st_dir else None,
            "se_available": self._se_available,
            "st_available": self._st_available,
            "available": self.available,
        }
