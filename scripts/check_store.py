"""Sanity-check a PseudobulkStore: load, LFC magnitude stats, panel overlap."""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import PseudobulkStore, load_gene_names_csv

store = PseudobulkStore.load("artifacts/stores/vcc2025_h1.npz")
axis = set(load_gene_names_csv("data/vcc2026/gene_names.csv"))
print(f"contexts: {sorted(store.basal)} | pairs: {len(store.pairs)} | controls: "
      f"{ {c: m.shape for c, m in store.controls.items()} }")

lfc = store.lfc
mag = np.abs(lfc)
nz = mag[mag > 1e-6]
print(f"lfc nnz/vec median: {np.median((mag > 1e-6).sum(axis=1)):.0f} / {lfc.shape[1]}")
print(f"|lfc| nonzero: median={np.median(nz):.3f} p90={np.percentile(nz, 90):.3f} max={nz.max():.2f}")

targets = [t for _, t in store.pairs]
panel = [l.strip() for l in open("data/vcc2026/pert_counts.csv").read().splitlines()[1:] if l.strip()]
overlap = len(set(targets) & set(panel))
print(f"H1 targets overlapping 2026 panel: {overlap}/{len(panel)}")

# basal sanity
basal = store.basal[sorted(store.basal)[0]]
print(f"basal log-mean-CPM: median={np.median(basal):.2f} max={basal.max():.2f}")

for name in ["hepg2", "jurkat", "k562", "rpe1"]:
    s = PseudobulkStore.load(f"artifacts/stores/tahoe_{name}.npz")
    m2 = np.abs(s.lfc); nz2 = m2[m2 > 1e-6]
    print(f"{name}: pairs={len(s.pairs)} med|lfc|={np.median(nz2):.3f} p90={np.percentile(nz2,90):.2f}")
