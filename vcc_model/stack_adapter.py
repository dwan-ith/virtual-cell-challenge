r"""Stack ICL adapter: cached cross-context deltas as a third ensemble member.

Stack (ArcInstitute, Jan 2026) performs in-context learning over cell sets:
prompt it with cells experiencing a condition (here: CRISPRi of gene g in
public source lines) and query it with unlabeled cells from the target
context; it predicts how the query population responds to the prompt
condition without fine-tuning. That is exactly the VCC 2026 task shape.

Two ways to produce cached deltas:

1. Batch CLI runner (scripts/run_stack_generation.py):
       stack-generation --checkpoint <ckpt> \
           --base-adata prompts/GENE.h5ad \        # perturbed cells, obs.target_gene col
           --test-adata queries/context_A.h5ad \   # challenge control subsample
           --genelist <ckpt_dir>/basecount_1000per_15000max.pkl \
           --output-dir generations/A/GENE --split-column target_gene
   then run scripts/import_stack_cache.py to map outputs onto the challenge
   gene axis as {context}__{gene}.npy files inside --cache-dir.

2. Python API (if your arc-stack install exposes stack.generation):
       from vcc_model.stack_adapter import StackAdapter
       adapter = StackAdapter(gene_names, checkpoint=..., cache_dir=...)
       adapter.generate_delta_online(ctrl_adata, gene, prompt_builder=...)

Either way, at ensemble time only `delta()` matters: it returns the cached
log-space delta (predicted-perturbed minus observed-control, CPM-log space)
mapped onto the challenge axis, or raises RuntimeError when absent so the
caller can keep weight 0 for this component.

Cache layout (one file per pair, float32[G] in log1p-CPM units):
    {cache_dir}/{context}__{gene}.npy
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import anndata as ad
import numpy as np


class StackAdapter:
    def __init__(
        self,
        gene_names: list[str],
        checkpoint: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        device: str | None = None,
    ):
        self.gene_names = list(gene_names)
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- cached
    def delta(
        self,
        control_adata: ad.AnnData | None,
        gene: str,
        context: str | None = None,
    ) -> np.ndarray:
        """Cached LFC-space delta for (context, gene); raises if unavailable."""
        if self.cache_dir is not None:
            path = self.cache_dir / f"{context or 'ctx'}__{gene}.npy"
            if path.exists():
                d = np.load(path).astype(np.float32)
                if d.shape[0] != len(self.gene_names):
                    raise ValueError(f"cached delta shape mismatch for {path.name}")
                return d

        if self.checkpoint is not None:
            assert control_adata is not None
            return self.generate_delta_online(control_adata, gene, context=context)

        raise RuntimeError(
            f"Stack delta unavailable for ({context}, {gene}). Either:\n"
            "  1) generate caches with scripts/run_stack_generation.py and pass "
            "--stack-cache, or\n"
            "  2) construct StackAdapter(checkpoint=...) and call "
            "generate_delta_online() directly."
        )

    def has_cache(self, gene: str, context: str | None = None) -> bool:
        if self.cache_dir is None:
            return False
        return (self.cache_dir / f"{context or 'ctx'}__{gene}.npy").exists()

    def write_delta(self, delta: np.ndarray, gene: str, context: str | None = None) -> Path:
        assert self.cache_dir is not None, "no cache_dir configured"
        path = self.cache_dir / f"{context or 'ctx'}__{gene}.npy"
        np.save(path, np.asarray(delta, dtype=np.float32))
        return path

    # ------------------------------------------------------------ online
    def generate_delta_online(
        self,
        control_adata: ad.AnnData,
        gene: str,
        context: str | None = None,
        prompt_builder: Optional[Callable[[str], ad.AnnData]] = None,
        n_query_cells: int = 512,
    ) -> np.ndarray:
        """Run Stack ICL in-process and cache the result.

        Requires `arc-stack` installed and a local checkpoint. prompt_builder
        supplies the perturbed-cell prompt AnnData for a target gene (see
        scripts/build_stack_prompts.py); without it we cannot build a prompt
        and fall back to raising.
        """
        try:
            import torch
            from stack import model_loading  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("arc-stack not installed; pip install arc-stack") from exc

        assert prompt_builder is not None, (
            "provide prompt_builder(AnnData for gene) or precompute caches"
        )
        prompt = prompt_builder(gene)
        query = control_adata[: min(n_query_cells, control_adata.n_obs)].copy()

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _model = model_loading.load_model(self.checkpoint, device=device)  # type: ignore[attr-defined]

        # The exact generation entry point varies by arc-stack version; this
        # mirrors notebooks/tutorial-predict.ipynb (masking schedule +
        # prompt/query concatenation). Adapt names to the installed release.
        try:
            from stack.generation import run_in_context_generation  # type: ignore

            pred = run_in_context_generation(_model, prompt, query)
        except ImportError:
            raise RuntimeError(
                "stack.generation.run_in_context_generation not found; use the "
                "CLI cache path (scripts/run_stack_generation.py) instead."
            )

        from .countsim import sparse_log_mean_cpm

        ctrl_prof = sparse_log_mean_cpm(query.X, 1e6)
        pred_prof = sparse_log_mean_cpm(pred.X, 1e6)
        delta = self._to_axis(pred_prof - ctrl_prof, list(query.var_names))
        if self.cache_dir is not None:
            self.write_delta(delta, gene, context)
        return delta.astype(np.float32)

    def _to_axis(self, vec: np.ndarray, source_genes: list[str]) -> np.ndarray:
        """Map a prediction vector from Stack's gene axis onto the challenge axis."""
        pos = {g: i for i, g in enumerate(source_genes)}
        out = np.zeros(len(self.gene_names), dtype=np.float64)
        hit = [pos.get(g, -1) for g in self.gene_names]
        mask = np.array([i >= 0 for i in hit])
        out[mask] = vec[np.array([i for i in hit if i >= 0])]
        return out
