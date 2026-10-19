"""Inspect Replogle bulk h5ads: shape, obs schema, X scale, control rows."""
import sys
from pathlib import Path

import numpy as np
import anndata as ad

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for name in ["K562_gwps_raw_bulk_01", "K562_essential_raw_bulk_01", "rpe1_raw_bulk_01"]:
    p = Path("data/replogle") / f"{name}.h5ad"
    a = ad.read_h5ad(p, backed="r")
    print(f"=== {name}: {a.shape} ===")
    print("obs cols:", list(a.obs.columns)[:10])
    for c in list(a.obs.columns)[:4]:
        vc = a.obs[c].astype(str).value_counts()
        print(f"  {c}: nunique={len(vc)} head={dict(vc.head(3))}")
    x0 = a.X[: min(5, a.shape[0])]
    d = np.asarray(x0.todense() if hasattr(x0, "todense") else x0)
    print(f"X dtype={a.X.dtype if hasattr(a.X,'dtype') else '?'} "
          f"row sums sample: {np.asarray(d.sum(axis=1)).ravel()[:5]}")
    print("var head:", list(map(str, a.var_names[:4])))
    a.file.close()
