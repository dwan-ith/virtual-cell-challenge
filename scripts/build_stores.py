"""Build PseudobulkStore files from raw public Perturb-seq h5ads.

Config is a JSON file mapping friendly names -> source specs:

{
  "replogle_k562": {"path": "data/raw/replogle_k562_gw.h5ad"},
  "vcc2025_h1":    {"path": "data/raw/vcc2025_train.h5ad",
                    "context_col": null,
                    "target_col": "target_gene",
                    "control_label": "non-targeting"}
}

Contexts are namespaced `{store}/{context}` so multiple cell lines inside one
file stay distinct sources for leave-one-context-out validation.

Usage:
    python scripts/build_stores.py --config configs/stores.json --out-dir artifacts/stores
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import build_pseudobulk_store, load_gene_names_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stores"))
    ap.add_argument("--gene-axis", type=Path, default=None,
                    help="gene_names.csv to project onto (challenge axis). "
                         "If omitted, uses the union axis of the first source.")
    ap.add_argument("--min-cells", type=int, default=10)
    args = ap.parse_args()

    spec = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gene_axis = (
        load_gene_names_csv(args.gene_axis) if args.gene_axis else None
    )
    if gene_axis is None:
        import anndata as ad

        first = ad.read_h5ad(Path(next(iter(spec.values()))["path"]), backed="r")
        gene_axis = [str(v) for v in first.var_names]
        first.file.close()
        print(f"[stores] no --gene-axis given; using {len(gene_axis)} genes from first source")

    for name, src in spec.items():
        t0 = time.time()
        print(f"[stores] building '{name}' from {src['path']} ...")
        store = build_pseudobulk_store(
            sources=[{**src, "name": name}],
            gene_names=gene_axis,
            min_cells=args.min_cells,
        )
        out = args.out_dir / f"{name}.npz"
        store.save(out)
        n_ctx = len(store.basal)
        print(
            f"[stores] {name}: {len(store.pairs)} pairs over {n_ctx} contexts "
            f"-> {out} ({time.time() - t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
