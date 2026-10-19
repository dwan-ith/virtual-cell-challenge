"""Salvage: extract a compact control-cell subsample for the H1 store.

The main store (artifacts/stores/vcc2025_h1.npz) already contains LFC pairs +
basal; only its __controls.npz companion crashed on memory. This script
samples N control rows deterministically, harmonizes them to the official
axis in small slices, and writes the companion file directly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad

from vcc_model.dataset import harmonize_to_axis, load_gene_names_csv

TEST_H5AD = Path("data/hf/adata_Test.h5ad")
GENE_CSV = Path("data/vcc2026/gene_names.csv")
STORE = Path("artifacts/stores/vcc2025_h1.npz")
N_CTRL = 3000
SLICE_ROWS = 250
CTX = "replogle_nadig/H1" if False else "vcc2025"  # single-context store


def main() -> None:
    gene_axis = load_gene_names_csv(GENE_CSV)
    print(f"[h1c] axis {len(gene_axis)} | sampling {N_CTRL} control rows", flush=True)

    # Deterministic sample of control row positions.
    obs_cols = ["target_gene"]
    import pyarrow  # noqa: F401  (h5ad uses hdf5, not arrow; kept minimal)

    a = ad.read_h5ad(TEST_H5AD, backed="r")
    labels = a.obs["target_gene"].astype(str)
    ctrl_positions = np.flatnonzero((labels == "non-targeting").to_numpy())
    rng = np.random.default_rng(7)
    sel = np.sort(rng.choice(ctrl_positions, size=min(N_CTRL, len(ctrl_positions)), replace=False))
    print(f"[h1c] {len(sel)} of {len(ctrl_positions)} control rows selected", flush=True)

    blocks = []
    done = 0
    while done < len(sel):
        chunk = sel[done : done + SLICE_ROWS]
        m = harmonize_to_axis(a[chunk.tolist()].X, a.var_names, gene_axis)
        blocks.append(m.astype(np.int32))
        done += len(chunk)
        if done % 1000 < SLICE_ROWS:
            print(f"[h1c] {done}/{len(sel)} rows", flush=True)
    a.file.close()

    mat = sparse.vstack(blocks, format="csr")
    del blocks
    out = STORE.with_name(STORE.stem + "__controls.npz")
    np.savez_compressed(
        out,
        context=np.array([CTX], dtype=object),
        indptr=mat.indptr,
        indices=mat.indices,
        data=mat.data,
        shape=np.array(mat.shape),
        row_starts=np.array([mat.indptr[-1]]),
    )
    print(f"[h1c] wrote {out} shape={mat.shape} nnz={mat.nnz}", flush=True)


if __name__ == "__main__":
    main()
