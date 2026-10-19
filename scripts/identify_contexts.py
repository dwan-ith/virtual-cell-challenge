"""Identify A/B/C by basal correlation against real-basals sources + H1."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from vcc_model.dataset import load_controls_bundle
from vcc_model.countsim import sparse_log_mean_cpm

stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
basals, genes = {}, None
for p in stores:
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z['gene_names'].astype(str))
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]

controls = load_controls_bundle(Path('data/vcc2026'), genes)
targ = {}
for ctx in ['A', 'B', 'C']:
    targ[ctx] = sparse_log_mean_cpm(controls[ctx][:4000], 1e6)

print(f"{'source':28s} {'A':>7s} {'B':>7s} {'C':>7s}  note", flush=True)
for s in sorted(basals):
    tag = 'TAHOE-FAKE' if 'nadig' in s else 'real'
    cs = [float(np.corrcoef(targ[c], basals[s])[0, 1]) for c in ['A', 'B', 'C']]
    print(f"{s:28s} {cs[0]:7.4f} {cs[1]:7.4f} {cs[2]:7.4f}  {tag}", flush=True)
# A/B/C mutual correlations
print('A-B:', float(np.corrcoef(targ['A'], targ['B'])[0, 1]), flush=True)
print('A-C:', float(np.corrcoef(targ['A'], targ['C'])[0, 1]), flush=True)
print('B-C:', float(np.corrcoef(targ['B'], targ['C'])[0, 1]), flush=True)
