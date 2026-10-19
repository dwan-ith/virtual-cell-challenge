"""Analyze the 2026 validation controls bundle: schema, count properties,
cross-context basal similarity, and panel composition."""
import sys
import json
from pathlib import Path

import numpy as np
import anndata as ad
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcc_model.countsim import sparse_mean_cpm

CTRL = Path(r"C:\Users\aacer\Downloads\controls")

gene_names = pd.read_csv(CTRL / "gene_names.csv", header=None)[0].astype(str).tolist()
panel = pd.read_csv(CTRL / "pert_counts.csv")
print(f"gene axis: {len(gene_names)} (dup {len(gene_names) - len(set(gene_names))}) | panel: {len(panel)} targets")

basal_profiles = {}
for ctx in ["A", "B", "C"]:
    a = ad.read_h5ad(CTRL / f"context_{ctx}.h5ad")
    x = a.X
    print(f"\n=== context {ctx} ===")
    print(f"shape={a.shape} X={type(x).__name__}/{x.dtype}")
    obs_cols = list(a.obs.columns)
    var_cols = list(a.var.columns)
    print(f"obs cols: {obs_cols}")
    if obs_cols:
        for c in obs_cols[:4]:
            vals = a.obs[c].astype(str).value_counts()
            print(f"  {c}: nunique={len(vals)} top={dict(vals.head(2))}")
    print(f"var cols: {var_cols}; var_names==axis: {list(map(str, a.var_names)) == gene_names}")

    data = np.asarray(x.data)
    totals = np.asarray(x.sum(axis=1)).ravel()
    print(f"counts integral: {bool(np.all(data == np.round(data)))} | min={data.min():.0f} max={data.max():.0f}")
    print(f"lib size: median={np.median(totals):,.0f} mean={totals.mean():,.0f} "
          f"p10={np.percentile(totals,10):,.0f} p90={np.percentile(totals,90):,.0f}")
    nnz_per_cell = np.diff(x.indptr)
    print(f"genes/cell: median={np.median(nnz_per_cell):.0f} p90={np.percentile(nnz_per_cell,90):.0f}")
    prof = sparse_mean_cpm(x, 1e6)
    basal_profiles[ctx] = np.log2(prof + 1)
    det = (x > 0).sum(axis=0).A1 / x.shape[0]
    print(f"detection rate: median gene={np.median(det):.2f} | genes>5CPM-ish: {(prof>5).sum()}")
    del a, x

print("\n=== cross-context basal correlation (log2 CPM) ===")
keys = list(basal_profiles)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        c = np.corrcoef(basal_profiles[keys[i]], basal_profiles[keys[j]])[0, 1]
        print(f"  {keys[i]} vs {keys[j]}: r={c:.4f}")

print("\npanel head:", panel.columns.tolist(), panel.head(3).to_dict("records"))
