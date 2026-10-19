"""Build exact stores from Replogle bulk (pseudobulk) h5ads.

K562 genome-wide: 11,258 target rows x 8,248 expressed genes (symbols in
var['gene_name']); 514 core_control rows give the true K562 basal profile.
Rows are per-group mean-UMI vectors -> CPM-normalize per row, average over
control rows for basal, LFC per target row. Same semantics as the DE metric.

Outputs artifacts/stores/replogle_{gw_k562,ess_k562,gw_rpe1}.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad

from vcc_model.countsim import log2_fold_change
from vcc_model.dataset import PseudobulkStore, load_gene_names_csv
from vcc_model import spec

FILES = [
    ("data/replogle/K562_gwps_raw_bulk_01.h5ad", "replogle_gw/K562", "replogle_gw_k562"),
    ("data/replogle/K562_essential_raw_bulk_01.h5ad", "replogle_ess/K562", "replogle_ess_k562"),
    ("data/replogle/rpe1_raw_bulk_01.h5ad", "replogle_gw/RPE1", "replogle_gw_rpe1"),
]
OUT_DIR = Path("artifacts/stores")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    axis = load_gene_names_csv("data/vcc2026/gene_names.csv")
    pos_of = {g: i for i, g in enumerate(axis)}

    for path, ctx, out_name in FILES:
        t0 = time.time()
        a = ad.read_h5ad(path)
        symbols = a.var["gene_name"].astype(str).to_numpy()
        keep_src = np.array([s in pos_of for s in symbols])
        X = np.asarray(a.X, dtype=np.float64)
        if sparse_ok := hasattr(a.X, "todense"):
            pass
        totals = X.sum(axis=1)
        cpm = X / np.maximum(totals[:, None], 1e-9) * spec.DE_TARGET_SUM

        ctrl_mask = a.obs["core_control"].astype(bool).to_numpy()
        basal_cpm = cpm[ctrl_mask].mean(axis=0)

        # Target symbol parsed from obs_names "{n}_{SYMBOL}_{guides}_{ENSG}"
        names = list(map(str, a.obs_names))
        tgt_of_row = []
        for nm in names:
            parts = nm.split("_")
            tgt_of_row.append(parts[1] if len(parts) >= 4 else nm)

        pairs, lfc_rows = [], []
        seen: set[str] = set()
        axis_keep_positions = np.array([pos_of.get(s, -1) for s in symbols])
        src_ok = axis_keep_positions >= 0
        for r in np.flatnonzero(~ctrl_mask):
            t = tgt_of_row[r]
            if t == "non-targeting" or t in seen:
                continue
            seen.add(t)
            vec = np.zeros(len(axis))
            lfc_src = log2_fold_change(
                cpm[r][src_ok], basal_cpm[src_ok], spec.DE_EPSILON
            )
            vec[axis_keep_positions[src_ok]] = lfc_src
            pairs.append((ctx, t))
            lfc_rows.append(np.clip(vec, -8, 8).astype(np.float32))

        full_basal = np.full(len(axis), np.log(1e-9), dtype=np.float64)
        idx_axis = np.array([pos_of.get(s, -1) for s in symbols])
        m = src_ok
        full_basal[idx_axis[m]] = np.log(basal_cpm[src_ok].clip(min=0) + 1e-9)

        n_cells_raw = pd.to_numeric(a.obs["num_cells_filtered"], errors="coerce").to_numpy()
        n_cells_raw = np.nan_to_num(n_cells_raw[~ctrl_mask][: len(pairs)], nan=0.0)
        store = PseudobulkStore(
            gene_names=np.array(axis),
            pairs=pairs,
            lfc=np.stack(lfc_rows),
            basal={ctx: full_basal},
            n_cells=dict(zip(pairs, n_cells_raw.astype(int).tolist())),
            source=Path(path).name,
        )
        out = OUT_DIR / f"{out_name}.npz"
        store.save(out)
        n_panel = len({t for _, t in pairs} & set(
            l.strip() for l in open("data/vcc2026/pert_counts.csv").read().splitlines()[1:] if l.strip()
        ))
        print(f"[bulk] {ctx}: {len(pairs)} targets ({n_panel} on 2026 panel) -> {out} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del a, X, cpm


if __name__ == "__main__":
    main()
