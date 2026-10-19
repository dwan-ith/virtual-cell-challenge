"""Build a VCC-compatible prediction AnnData (prep with cell-eval before upload)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from .ensemble import HybridVCCModel


def write_submission(
    pred: ad.AnnData,
    path: str | Path,
    perturbation_col: str = "perturbation",
) -> Path:
    """Write predictions. Run `cell-eval prep -i <path>` before uploading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pred.copy()
    if perturbation_col != "target_gene" and perturbation_col in out.obs:
        out.obs["target_gene"] = out.obs[perturbation_col].astype(str)
    out.write_h5ad(path)
    return path


def predict_from_vcc_inputs(
    model: HybridVCCModel,
    control_adata: ad.AnnData,
    gene_list_csv: str | Path,
    context_name: str = "val_line",
    cells_per_pert: int = 80,
    control_cells: int = 2000,
    context_col: str | None = None,
) -> ad.AnnData:
    """Predict for one or more contexts given basal cells + target gene CSV.

    gene_list_csv: column of gene symbols (or a single-column CSV).
    If control_adata has multiple contexts, set context_col.
    """
    genes = pd.read_csv(gene_list_csv)
    gene_col = genes.columns[0]
    target_genes = genes[gene_col].astype(str).tolist()

    if context_col and context_col in control_adata.obs:
        contexts = sorted(control_adata.obs[context_col].unique())
        controls = {
            c: control_adata[control_adata.obs[context_col] == c].copy() for c in contexts
        }
        genes_by_context = {c: target_genes for c in contexts}
    else:
        controls = {context_name: control_adata}
        genes_by_context = {context_name: target_genes}

    return model.predict_anndata(
        controls_by_context=controls,
        genes_by_context=genes_by_context,
        cells_per_pert=cells_per_pert,
        include_controls=True,
        control_cells=control_cells,
    )
