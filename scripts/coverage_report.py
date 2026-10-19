"""Coverage diagnostics: how much of the 2026 task can current sources see?

1. Panel overlap: how many of the 300 targets were perturbed in >=1 source
   context (direct transfer possible) vs unseen-gene fallback territory.
2. Gene-axis gate coverage: fraction of genes passing the official >5 CPM
   control gate (context A) that Tahoe features measured. Genes outside the
   feature set get predicted LFC=0 -> invisible to DE metrics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad

from vcc_model.countsim import sparse_mean_cpm
from vcc_model.dataset import PseudobulkStore, load_gene_names_csv

panel = [l.strip() for l in open("data/vcc2026/pert_counts.csv").read().splitlines()[1:] if l.strip()]
print(f"panel: {len(panel)} targets")

STORE_NAMES = [
    "replogle_gw_k562", "replogle_ess_k562", "replogle_gw_rpe1",
    "tahoe_hepg2", "tahoe_jurkat", "tahoe_k562", "tahoe_rpe1",
]
seen: set[str] = set()
for name in STORE_NAMES:
    s = PseudobulkStore.load(f"artifacts/stores/{name}.npz")
    n = len(set(panel) & {t for _, t in s.pairs})
    print(f"  {name}: panel overlap {n}")
    seen |= {t for _, t in s.pairs}
h1 = PseudobulkStore.load("artifacts/stores/vcc2025_h1.npz")
seen_h1 = {t for _, t in h1.pairs}

n_panel_seen = len(set(panel) & seen)
n_any = len(set(panel) & (seen | seen_h1))
print(f"panel targets with direct source observations: {n_panel_seen}/300 "
      f"(+{n_any - n_panel_seen} via H1) -> fallback for {300 - n_any}")

axis = load_gene_names_csv("data/vcc2026/gene_names.csv")
a = ad.read_h5ad("data/vcc2026/context_A.h5ad")
prof_a = sparse_mean_cpm(a.X, 1e6)
gate = prof_a > 5.0
print(f"context A gate (>5 CPM): {int(gate.sum())} genes")

feat_axis_pos = set()
for name in ["hepg2", "jurkat", "k562", "rpe1"]:
    s = PseudobulkStore.load(f"artifacts/stores/tahoe_{name}.npz")
    for gi, g in enumerate(s.gene_names):
        if s.lfc[0][gi] != 0 or True:  # features stored on full axis; zeros are real values too
            pass
# Feature identity: rebuild from feature_names.json mapping instead.
import json
feats = json.loads(Path("data/hf/definition/feature_names.json").read_text())
pos_of = {g: i for i, g in enumerate(axis)}
covered = {g for g in map(str, feats) if g in pos_of}
inter = sum(1 for i, g in enumerate(axis) if gate[i] and g in covered)
print(f"gate∩features: {inter}/{int(gate.sum())} = {inter/max(int(gate.sum()),1):.1%} of scorable genes covered")

# same for H1 truth axis coverage
h1_covered = sum(1 for i, g in enumerate(h1.gene_names) if h1.lfc[0][i] != 0 or True)
print(f"H1 store rows span full axis ({h1.gene_names.size}); nonzero-lfc fraction="
      f"{(np.abs(h1.lfc) > 1e-6).mean():.2f}")
