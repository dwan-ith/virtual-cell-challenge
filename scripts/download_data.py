"""Download challenge + training data.

Two distinct sources:

1. CHALLENGE CONTROLS (required for submission)
   Authenticated via the official CLI:
       uv tool install vcc-cli
       vcc login --token-stdin        # paste API key from virtualcellchallenge.org
       vcc datasets list
       vcc datasets download controls -d data/vcc2026
   -> context_A/B/C.h5ad, gene_names.csv, pert_counts.csv

   This script shells out to `vcc` when available; otherwise prints the steps.

2. PUBLIC TRAINING DATA (Arc Virtual Cell Atlas on Google Cloud Marketplace;
   requester-pays bucket, free tier ~2TB/month):
       gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/...
   Browse with:
       gcloud storage ls gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/**
   Download (billing project required even on free tier):
       gcloud storage cp -r \
         "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train" data/raw/vcc2025_train

   Other useful prefixes to explore the same way: scBaseCount perturbation
   subsets (Replogle K562/RPE1 CRISPRi live there with `perturbation` and
   `cell_line` obs columns), Tahoe-100M.

Usage:
    python scripts/download_data.py --controls-dir data/vcc2026 [--raw-dir data/raw]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def try_vcc_cli(controls_dir: Path) -> bool:
    exe = shutil.which("vcc")
    if exe is None:
        print(
            "[data] `vcc` CLI not found. Install + authenticate:\n"
            "       uv tool install vcc-cli\n"
            "       vcc login --token-stdin      # API key from the member portal\n"
            f"       vcc datasets download controls -d {controls_dir}"
        )
        return False
    controls_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading controls bundle -> {controls_dir} (resumable)")
    rc = subprocess.call([exe, "datasets", "download", "controls", "-d", str(controls_dir)])
    if rc != 0:
        print("[data] download failed — check `vcc whoami` (auth/approval) and retry;")
        print("       interrupted downloads resume automatically on re-run.")
        return False
    return True


def print_atlas_guide(raw_dir: Path) -> None:
    print(
        f"\n[data] Training-data fetch (manual, requester-pays GCS):\n"
        f"  gcloud storage ls \"gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/**\"\n"
        f"  gcloud storage cp -r \\\n"
        f"    \"gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train\" {raw_dir}/vcc2025\n\n"
        f"  Expected file: adata_Training.h5ad (~300k cells, H1 hESC, 300 CRISPRi targets)\n"
        f"  Then build a store:\n"
        f"    python scripts/build_stores.py --config configs/stores.json --out-dir artifacts/stores\n"
        f"  with configs/stores.json:\n"
        f'    {{"vcc2025_h1": {{"path": "{raw_dir.as_posix()}/vcc2025/train/adata_Training.h5ad"}}}}'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controls-dir", type=Path, default=Path("data/vcc2026"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()

    ok = True
    if not args.skip_controls:
        marker = args.controls_dir / "gene_names.csv"
        if marker.exists():
            print(f"[data] controls already present at {args.controls_dir}")
        else:
            ok = try_vcc_cli(args.controls_dir)

    print_atlas_guide(args.raw_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
