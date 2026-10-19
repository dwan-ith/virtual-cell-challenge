"""Convert Tahoe replogle-nadig cell_eval deltas into PseudobulkStores.

Input : data/hf/cell_eval/all_delta.parquet
        rows = (cell_line, gem_group, target gene), ~6,545 feature columns
        holding LINEAR pseudobulk deltas mean(perturbed) - mean(control).
Output: artifacts/stores/tahoe_{hepg2,jurkat,k562,rpe1}.npz on the official
        axis, one pooled (gem_group-averaged) LFC vector per target.

LFC approximation (documented): lfc = log2(1 + delta / m(gene)), where
m(gene) is a generic basal-CPM curve taken as the per-gene MEDIAN across the
challenge control contexts A/B/C (plus optional H1 store when present).
Replogle/Nadig processed values are count-normalized like CPM, so this ratio
is approximately the scored log2 fold change; residual scale error per line
is corrected by matching the median |lfc| of nonzero-delta entries to the H1
store's empirical median once available (--calibrate-to h1store.npz).
"""
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.countsim import sparse_mean_cpm
from vcc_model.dataset import PseudobulkStore, load_gene_names_csv

PARQUET = Path("data/hf/cell_eval/all_delta.parquet")
FEAT_JSON = Path("data/hf/definition/feature_names.json")
CTRL_DIR = Path("data/vcc2026")
OUT_DIR = Path("artifacts/stores")
BLOCK = 300


def generic_basal_curve(axis: list[str]) -> np.ndarray:
    """Median CPM profile across challenge control contexts."""
    profiles = []
    for ctx in ["A", "B", "C"]:
        a = ad.read_h5ad(CTRL_DIR / f"context_{ctx}.h5ad")
        prof = sparse_mean_cpm(a.X, 1e6)
        profiles.append(prof)
        del a
    return np.median(np.stack(profiles), axis=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    axis = load_gene_names_csv(CTRL_DIR / "gene_names.csv")
    pos_of = {g: i for i, g in enumerate(axis)}

    feats = json.loads(FEAT_JSON.read_text())
    feats = [str(f) for f in feats]
    print(f"[tahoe] {len(feats)} features, {sum(f in pos_of for f in feats)} map to official axis", flush=True)

    t0 = time.time()
    print("[tahoe] generic basal curve from A/B/C controls ...", flush=True)
    m = generic_basal_curve(axis)
    m_floor = float(np.percentile(m[m > 0], 10)) if (m > 0).any() else 1.0
    m_safe = np.maximum(m, m_floor)

    pf = pq.ParquetFile(PARQUET)
    ids = pf.read_row_group(0, columns=["cell_line", "gene"]).to_pandas()
    key = ids["cell_line"].astype(str) + "\x1f" + ids["gene"].astype(str)
    codes, uniques = pd.factorize(key)
    pair_keys = [tuple(u.split("\x1f")) for u in uniques]
    n_pairs = len(pair_keys)
    print(f"[tahoe] {n_pairs} unique (line, target) pairs over {pf.metadata.num_rows} batch rows", flush=True)

    feat_axis_pos = np.array([pos_of.get(f, -1) for f in feats])
    mapped = feat_axis_pos >= 0

    # Pass 1: accumulate pooled deltas per pair for all feature columns,
    # then keep the mapped subset.
    schema_names = pf.schema_arrow.names
    feat_cols = list(schema_names[3:])
    pooled = np.zeros((n_pairs, len(feat_cols)), dtype=np.float64)

    for start in range(0, len(feat_cols), BLOCK):
        block_names = feat_cols[start : start + BLOCK]
        tbl = pf.read_row_group(0, columns=block_names)
        vals = np.asarray(tbl.to_pandas(), dtype=np.float64)
        vals[~np.isfinite(vals)] = 0.0
        for j in range(vals.shape[1]):
            pooled[:, start + j] = np.bincount(codes, weights=vals[:, j], minlength=n_pairs)
    pooled /= np.bincount(codes, minlength=n_pairs)[:, None]
    print(f"[tahoe] pooled deltas done ({time.time()-t0:.0f}s)", flush=True)

    # Convert to pseudo-LFC on the official axis.
    feat_pos_mapped = feat_axis_pos[mapped]
    denom = m_safe[feat_pos_mapped]
    ratio = 1.0 + pooled[:, mapped] / denom[None, :]
    # Deltas can exceed -m at raw scale (scale mismatch) -> clamp ratio > 0.
    np.clip(ratio, 1e-12, None, out=ratio)
    lfc_block = np.log2(ratio)

    # Optional global calibration against an exact store.
    cal_path = Path("artifacts/stores/calibration_target.json")
    if cal_path.exists():
        target_med = float(json.loads(cal_path.read_text())["median_abs_lfc"])
        for li, line in enumerate(sorted({k[0] for k in pair_keys})):
            mask = np.array([k[0] == line for k in pair_keys])
            nz = np.abs(lfc_block[mask])
            nz = nz[nz > 1e-6]
            med = float(np.median(nz)) if nz.size else 0.0
            gain = target_med / med if med > 0 else 1.0
            lfc_block[mask] = np.clip(lfc_block[mask] * gain, -6.0, 6.0)
            print(f"[tahoe] {line}: median|lfc|={med:.3f} -> x{gain:.2f} (clipped to +-6)", flush=True)
    else:
        np.clip(lfc_block, -6.0, 6.0, out=lfc_block)
        print("[tahoe] no calibration target found; identity scale (clipped)", flush=True)

    # One store per cell line.
    lines = sorted({k[0] for k in pair_keys})
    for line in lines:
        idx = [i for i, k in enumerate(pair_keys) if k[0] == line]
        pairs = [(f"replogle_nadig/{line}", str(k[1])) for k in (pair_keys[i] for i in idx)]
        lfc_mat = np.zeros((len(idx), len(axis)), dtype=np.float32)
        lfc_mat[:, feat_pos_mapped] = lfc_block[idx].astype(np.float32)
        # Basal proxy for the source line: not measured here; reuse the generic
        # curve so similarity weighting falls back to uniform pooling.
        store = PseudobulkStore(
            gene_names=np.array(axis),
            pairs=pairs,
            lfc=lfc_mat,
            basal={"replogle_nadig/" + line: np.log(m_safe + 1e-9)},
            n_cells={},
            source=f"tahoebio/replogle-nadig-de-rhaister:{line}",
        )
        out = OUT_DIR / f"tahoe_{line}.npz"
        store.save(out)
        print(f"[tahoe] {line}: {len(idx)} targets -> {out}", flush=True)

    print(f"[tahoe] ALL DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
