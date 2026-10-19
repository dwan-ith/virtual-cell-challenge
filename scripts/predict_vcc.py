"""Predict on real VCC validation inputs once you download them.

Expected layout (adjust paths as needed):
  data/vcc2026/controls.h5ad          # unperturbed cells (obs has context/line)
  data/vcc2026/target_genes.csv       # genes to knock down
  data/vcc2025/train.h5ad             # public 2025 Perturb-seq for fitting

Example:
  python scripts/predict_vcc.py \\
    --train data/vcc2025/train.h5ad \\
    --controls data/vcc2026/controls.h5ad \\
    --genes data/vcc2026/target_genes.csv \\
    --context-col cell_line \\
    --pert-col gene \\
    --out artifacts/vcc2026_pred.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from vcc_model.ensemble import HybridVCCModel
from vcc_model.residual import ResidualTrainerConfig
from vcc_model.submit import predict_from_vcc_inputs, write_submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True, help="Public Perturb-seq h5ad")
    p.add_argument("--controls", type=Path, required=True, help="2026 basal-state h5ad")
    p.add_argument("--genes", type=Path, required=True, help="Target gene CSV")
    p.add_argument("--out", type=Path, default=Path("artifacts/vcc_pred.h5ad"))
    p.add_argument("--context-col", default="context")
    p.add_argument("--pert-col", default="perturbation")
    p.add_argument("--control-label", default="non-targeting")
    p.add_argument("--cells-per-pert", type=int, default=80)
    p.add_argument("--control-cells", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/hybrid_vcc.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = ad.read_h5ad(args.train)
    controls = ad.read_h5ad(args.controls)

    # Normalize column names expected by the model.
    if args.context_col != "context" and args.context_col in train.obs:
        train.obs["context"] = train.obs[args.context_col].astype(str)
    if args.pert_col != "perturbation" and args.pert_col in train.obs:
        train.obs["perturbation"] = train.obs[args.pert_col].astype(str)
    if args.context_col != "context" and args.context_col in controls.obs:
        controls.obs["context"] = controls.obs[args.context_col].astype(str)

    model = HybridVCCModel(control_label=args.control_label)
    info = model.fit(
        train,
        residual_cfg=ResidualTrainerConfig(epochs=args.epochs),
    )
    print("fit:", info)
    model.save(args.checkpoint)

    pred = predict_from_vcc_inputs(
        model,
        controls,
        args.genes,
        context_col="context" if "context" in controls.obs else None,
        cells_per_pert=args.cells_per_pert,
        control_cells=args.control_cells,
    )
    path = write_submission(pred, args.out)
    print(f"wrote {path} with {pred.n_obs} cells")
    print("Next: cell-eval prep -i", path)


if __name__ == "__main__":
    main()
