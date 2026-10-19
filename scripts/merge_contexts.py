"""Merge per-context submission h5ads into one panel h5ad.

Usage: python scripts/merge_contexts.py <out.h5ad> <in1.h5ad> <in2.h5ad> ...
Concatenates CSR X and obs rows; var axis must match. Validates lightly.
"""
import sys
from pathlib import Path
import anndata as ad
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcc_model.submission import validate_h5ad_file_light
from vcc_model.dataset import load_gene_names_csv
from vcc_model import spec


def main() -> None:
    out = Path(sys.argv[1])
    ins = [Path(p) for p in sys.argv[2:]]
    adatas = [ad.read_h5ad(p) for p in ins]
    base_vars = list(adatas[0].var_names)
    for a in adatas[1:]:
        assert list(a.var_names) == base_vars, "var axis mismatch"
    x = sparse.vstack([a.X.tocsr() for a in adatas], format="csr")
    obs = pd.concat([a.obs for a in adatas], ignore_index=True)
    merged = ad.AnnData(X=x, obs=obs, var=adatas[0].var.copy())
    merged.obs_names = [f"pred_{i}" for i in range(merged.n_obs)]
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(out, compression="gzip")
    print(f"merged {len(ins)} files -> {out} shape {merged.shape} nnz {x.nnz}")
    gene_names = load_gene_names_csv(Path("data/vcc2026/gene_names.csv"))
    panel = load_gene_names_csv(Path("data/vcc2026/pert_counts.csv")) if False else None
    from vcc_model.dataset import load_panel_csv
    panel_df = load_panel_csv(Path("data/vcc2026/pert_counts.csv"))
    contexts = tuple(sorted(panel_df[spec.CONTEXT_COL].unique()))
    # panel csv has only target_gene; contexts come from merged obs
    contexts = tuple(sorted(merged.obs[spec.CONTEXT_COL].astype(str).unique()))
    problems = validate_h5ad_file_light(out, gene_names, contexts=contexts,
                                        perts_per_context=spec.N_PERTS_PER_CONTEXT,
                                        cells_per_pert=spec.CELLS_PER_PERT)
    if problems:
        raise SystemExit("merged failed validation:\n" + "\n".join(problems))
    print("merged file validated OK")


if __name__ == "__main__":
    main()
