import collections
import numpy as np
from pathlib import Path

pairs = {}
genes = None
for p in sorted(Path("artifacts/stores").glob("*.npz")):
    if "__controls" in p.name:
        continue
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z["gene_names"].astype(str))
    for c, g in zip(z["context"], z["target"]):
        pairs[(str(c), str(g))] = 1

ctxs = sorted({c for c, _ in pairs})
h1 = sorted({g for c, g in pairs if c == "vcc2025/H1"})
h1set = set(h1)
covered = []
per = collections.Counter()
for g in h1:
    hits = [c for c in ctxs if c != "vcc2025/H1" and (c, g) in pairs]
    if hits:
        covered.append(g)
        for c in hits:
            per[c] += 1

print(f"H1 targets: {len(h1)}")
print(f"H1 targets covered by >=1 other store: {len(covered)}")
print(f"uncovered (would be zeroed by zero_unseen): {len(h1) - len(covered)}")
print(f"per-context coverage of H1 targets: {dict(per)}")

panel = [l.strip() for l in open("data/vcc2026/pert_counts.csv").read().splitlines()[1:] if l.strip()]
panel = [p.split(",")[0] for p in panel]
print(f"\n2026 panel: {len(panel)}")
cov_panel = []
per_panel = collections.Counter()
for g in panel:
    hits = [c for c in ctxs if (c, g) in pairs]
    if hits:
        cov_panel.append(g)
        for c in hits:
            per_panel[c] += 1
print(f"panel targets covered by >=1 store: {len(cov_panel)}")
print(f"panel targets NOT covered: {len(panel) - len(cov_panel)}")
print(f"per-store panel coverage: {dict(per_panel)}")