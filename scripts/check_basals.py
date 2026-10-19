import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import anndata as ad
from pathlib import Path

# Load A, B, C basals
ctrl_dir = Path("data/vcc2026")
basals = {}
for ctx in ["A", "B", "C"]:
    a = ad.read_h5ad(ctrl_dir / f"context_{ctx}.h5ad")
    prof = np.asarray(a.X.mean(axis=0)).ravel()
    prof = prof / prof.sum() * 1e6
    basals[ctx] = np.log2(prof + 1)

# Check stores
stores = sorted(Path("artifacts/stores").glob("*.npz"))
print("Comparing against stores:")
for s in stores:
    z = np.load(s, allow_pickle=True)
    if "basal" in z and "basal_ctx" in z:
        for c, b in zip(z["basal_ctx"], z["basal"]):
            r_A = np.corrcoef(basals["A"], b)[0, 1]
            r_B = np.corrcoef(basals["B"], b)[0, 1]
            r_C = np.corrcoef(basals["C"], b)[0, 1]
            print(f"Store {s.stem:20s} [{str(c):20s}]: A={r_A:.4f}  B={r_B:.4f}  C={r_C:.4f}")
