"""Build atlas-transfer source statistics from our local stores.

The public reference pipeline (method reproduced under MIT; see
NOTES-attribution.md) consumes one *_full_statistics.npz per source screen,
holding per-target pseudobulk count sums, mean CPM, and the matched control
profile. Those are derivable from the pseudobulk LFC stores we already have:

    mean_cpm_target = mean_cpm_control * 2 ** lfc
    probability      = mean_cpm / sum(mean_cpm)
    count_sums       = mean_cpm * n_cells        (a CPM-consistent mass proxy)

Substituting real sources for the two we cannot download (X-Atlas HCT116 and
HEK293T) keeps the reference's four-source, weights-[2,1,1,2] structure while
using tissue-diverse screens we actually hold.
"""
import sys
sys.path.insert(0, ".")
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vcc_model.dataset import load_gene_names_csv

# Atlas pipeline source slots. Reference method uses K562/HCT116/HEK293T/H1 at
# weights [2,1,1,2] plus CD4 at 0.5.
#
# FEASIBILITY (measured 2026-10-01, do not retry without new disk):
#   - X-Atlas-Orion is Lance v2 on HF, NOT parquet. lance requires the WHOLE
#     directory locally. HCT116 = 47.07 GiB, HEK293T = 77.27 GiB
#     (expression.lance is 99.7% of it; metadata alone is only ~130 MiB).
#   - CD4 GWCD4i.DE_stats.h5ad = 16.8 GB public S3, byte-range readable, but
#     measured 0.15 MB/s single-stream and 0.4 MB/s with 8 parallel readers
#     => 12-30 h and 16.8 GB of disk we do not have (7.8 GB free).
#   => HCT116/HEK293T/CD4 cannot be obtained on this machine. Slots below fall
#      back to real local screens; RPE1/HepG2 have zero panel overlap so they
#      are skipped at build time. Effective fusion is K562(2) + H1(2).
SLOTS = [
    ("k562", "replogle_gw/K562", 2),
    ("hct116", "replogle_gw/RPE1", 1),
    ("hek293t", "replogle_nadig/hepg2", 1),
    ("h1", "vcc2025/H1", 2),
]

PRIOR_CELLS = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", default=["artifacts/stores"])
    ap.add_argument("--out-dir", type=Path, default=Path("data/atlas_sources"))
    ap.add_argument("--panel", type=Path, default=Path("data/vcc2026/pert_counts.csv"))
    ap.add_argument("--min-cells", type=int, default=PRIOR_CELLS)
    args = ap.parse_args()

    genes = load_gene_names_csv("data/vcc2026/gene_names.csv")
    G = len(genes)
    idx_of = {str(g): i for i, g in enumerate(genes)}
    panel = pd.read_csv(args.panel).iloc[:, 0].astype(str).tolist()

    nested: dict[str, dict[str, np.ndarray]] = {}
    n_cells: dict[tuple[str, str], int] = {}
    basals: dict[str, np.ndarray] = {}
    store_dir = Path(args.stores[0])
    files = sorted(store_dir.glob("*.npz")) if store_dir.is_dir() else [store_dir]
    for p in files:
        if "__controls" in p.name:
            continue
        z = np.load(p, allow_pickle=True)
        assert np.array_equal([str(g) for g in z["gene_names"]], genes), f"axis mismatch {p}"
        for c, g, v, n in zip(z["context"], z["target"], z["lfc"], z["n_cells"]):
            nested.setdefault(str(c), {})[str(g)] = np.asarray(v, dtype=np.float64)
            n_cells[(str(c), str(g))] = int(n)
        for i, c in enumerate(z["basal_ctx"]):
            basals.setdefault(str(c), z["basal"][i].astype(np.float64))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, ctx, weight in SLOTS:
        if ctx not in nested:
            print(f"[atlas] SKIP {name}: no store context {ctx}", flush=True)
            continue
        ctrl_cpm = np.exp(basals[ctx]) - 1e-9
        ctrl_cpm = np.clip(ctrl_cpm, 0.0, None)
        total = float(ctrl_cpm.sum())
        if total <= 0:
            print(f"[atlas] SKIP {name}: empty control profile", flush=True)
            continue
        ctrl_prob = ctrl_cpm / total

        targets, rows, ncells = [], [], []
        for g in panel:
            v = nested[ctx].get(g)
            n = n_cells.get((ctx, g), 0)
            if v is None or n < args.min_cells:
                continue
            mean_cpm = np.clip(ctrl_cpm * np.exp2(np.clip(v, -8, 8)), 0.0, None)
            s = float(mean_cpm.sum())
            if s <= 0:
                continue
            targets.append(g)
            rows.append(mean_cpm / s * 1e6)
            ncells.append(n)
        if not targets:
            print(f"[atlas] SKIP {name}: no panel targets pass min_cells", flush=True)
            continue

        mean_cpm = np.stack(rows).astype(np.float32)
        counts = np.stack([r * n for r, n in zip(rows, ncells)]).astype(np.float32)
        out = args.out_dir / f"{name}_full_statistics.npz"
        np.savez_compressed(
            out,
            source=np.asarray(name),
            targets=np.asarray(targets),
            genes=np.asarray(genes),
            n_cells=np.asarray(ncells, dtype=np.int32),
            target_count_sums=counts,
            target_mean_cpm=mean_cpm,
            matched_control_probability=ctrl_prob.astype(np.float32),
            matched_control_mean_cpm=ctrl_cpm.astype(np.float32),
            global_control_probability=ctrl_prob.astype(np.float32),
            global_control_mean_cpm=ctrl_cpm.astype(np.float32),
            measured_genes=np.ones(G, dtype=bool),
        )
        manifest[name] = {
            "context": ctx, "weight": weight, "targets": len(targets),
            "median_cells": int(np.median(ncells)), "path": str(out),
        }
        print(f"[atlas] {name} <- {ctx}: {len(targets)}/{len(panel)} targets, "
              f"median {int(np.median(ncells))} cells -> {out.name}", flush=True)

    # The reference pipeline always loads a CD4 T-cell family file. Our stores
    # hold no CD4 measurement, so we emit a fully-unavailable stub: the fusion
    # step then contributes exactly 0/0 and leaves effects untouched.
    cd4 = args.out_dir / "CD4_DE_statistics.npz"
    np.savez_compressed(
        cd4,
        targets=np.asarray(panel),
        genes=np.asarray(genes),
        conditions=np.asarray(["none"]),
        log2fc=np.zeros((1, len(panel), G), dtype=np.float32),
        adjusted_p=np.ones((1, len(panel), G), dtype=np.float32),
        lfcSE=np.zeros((1, len(panel), G), dtype=np.float32),
        available=np.zeros((1, len(panel)), dtype=bool),
        quality_pass=np.zeros((1, len(panel)), dtype=bool),
        n_cells=np.zeros((1, len(panel)), dtype=np.int64),
    )
    print(f"[atlas] CD4 stub written (all-unavailable, fusion is a no-op) -> {cd4.name}", flush=True)

    covered = set()
    for m in manifest.values():
        pass
    union = set()
    for name in manifest:
        z = np.load(manifest[name]["path"], allow_pickle=True)
        union |= set(z["targets"].astype(str).tolist())
    print(f"[atlas] panel coverage across all sources: {len(union)}/{len(panel)}", flush=True)
    import json
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()