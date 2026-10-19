import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.dataset import load_controls_bundle
from vcc_model.countsim import sparse_log_mean_cpm

stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs, basals, genes = {}, {}, None
for p in stores:
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z['gene_names'].astype(str))
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c), str(g))] = v
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]
esm = load_esm_embeddings(Path("data/protein/se_full_18533.npz"))
nested = {}
for (c, g), v in pairs.items():
    nested.setdefault(c, {})[g] = v
prior = LFCTransferPrior(
    similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
    gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
    zero_unseen=True, default_conservation=0.45).fit(nested, basals, genes)
pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
cal = DECalibration(threshold=0.0, gain=1.8, power=1.0, q_threshold=0.25, per_gene_gain=pg)

import pandas as pd
panel = pd.read_csv("data/vcc2026/pert_counts.csv")
panel_genes = panel[panel.columns[0]].astype(str).tolist()[:300]
controls = load_controls_bundle(Path('data/vcc2026'), genes)

print('=== per-target |lfc| distribution after v29 calibration (ctx A) ===', flush=True)
ctrl = controls['A'][:4000]
basal = sparse_log_mean_cpm(ctrl, 1e6)
raw_map = prior.predict_many_gc(panel_genes, basal)
nzs, n_zero_unseen, tops = [], 0, []
for g in panel_genes:
    lfc = np.asarray(cal.transform(raw_map[g], gene=g), dtype=np.float64)
    nz = int((np.abs(lfc) > 0).sum())
    nzs.append(nz)
    if nz == 0:
        n_zero_unseen += 1
    if nz:
        tops.append(np.sort(np.abs(lfc))[::-1][:10])
nzs = np.array(nzs)
print('nonzero-lfc genes per target: median', int(np.median(nzs)), 'p10', int(np.percentile(nzs, 10)), 'p90', int(np.percentile(nzs, 90)))
print('targets with ZERO nonzero lfc:', n_zero_unseen)
print('frac of genome predicted nonzero (median):', round(float(np.median(nzs)) / len(genes), 4))
if tops:
    T = np.stack(tops)
    print('mean top-10 |lfc| per target:', np.round(T.mean(axis=0), 3))
print()
print('=== what v36 masked away (genes 151..nz) ===', flush=True)
# how much magnitude mass is in the discarded tail?
for g in panel_genes[:5]:
    lfc = np.asarray(cal.transform(raw_map[g], gene=g), dtype=np.float64)
    mag = np.abs(lfc)
    order = np.argsort(mag)[::-1]
    top150 = order[:150]
    tail = order[150:]
    print(f'{g}: nz={int((mag>0).sum())} top150 mass={mag[top150].sum():.2f} tail mass={mag[tail].sum():.2f} ratio={mag[tail].sum()/max(mag[top150].sum(),1e-9):.3f}')
