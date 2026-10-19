"""Diagnose the DE call-set explosion directly on REAL A/B/C controls.

For each calibration config, emit 400 cells (poisson_thin with our multipliers)
from real controls and run the same Wilcoxon the scorer uses:
    400 predicted cells vs 18,400 real control cells, CPM-normalized, BH 0.05.
Reports n_sig (called genes) per target + PDS proxy, so we can see whether
shrinking the LFC tail cuts spurious calls without wrecking discrimination.

No truth needed: the point is the PREDICTED call-set size, which is what drives
fid/jac/reach, versus the PDS proxy, which needs no significance calls.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.stats import mannwhitneyu

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.dataset import load_gene_names_csv, harmonize_to_axis, detect_columns
from vcc_model.countsim import poisson_thin, normalize_cpm, benjamini_hochberg
from vcc_model import spec

genes = load_gene_names_csv('data/vcc2026/gene_names.csv')
stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs, basals = {}, {}
for p in stores:
    z = np.load(p, allow_pickle=True)
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c), str(g))] = v
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]
nested = {}
for (c, g), v in pairs.items():
    nested.setdefault(c, {})[g] = v
esm = load_esm_embeddings(Path("data/protein/se_full_18533.npz"))
prior = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
                         gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
                         zero_unseen=True, default_conservation=0.45).fit(nested, basals, genes)
pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)

import pandas as pd
panel = [str(g) for g in pd.read_csv('data/vcc2026/pert_counts.csv').iloc[:, 0].tolist()[:12]]
a = ad.read_h5ad('data/vcc2026/context_A.h5ad')
tcol, _ = detect_columns(a.obs)
lab = a.obs[tcol].astype(str).to_numpy()
X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
ctrl = X[np.flatnonzero(lab == 'non-targeting')]
ctrl_cpm = normalize_cpm(np.asarray(ctrl[::4].todense(), dtype=np.float64), spec.DE_TARGET_SUM)
ctrl_sub = np.asarray(ctrl[::4].todense(), dtype=np.float64)
gate = ctrl_cpm.mean(axis=0) > spec.DE_MIN_CPM_CONTROL
print(f'controls {ctrl.shape} gate genes {int(gate.sum())}', flush=True)
basal = np.log(np.maximum(ctrl_cpm.mean(axis=0), 1e-9) + 1e-9)
rng = np.random.default_rng(0)

def nsig_for(cal, n_cells=400):
    counts = []
    for g in panel:
        lfc = np.asarray(cal.transform(prior.predict_lfc(g, basal), gene=g), dtype=np.float64)
        pick = rng.choice(ctrl_sub.shape[0], size=n_cells, replace=True)
        cells = poisson_thin(ctrl_sub[pick], np.exp2(np.clip(lfc, -6, 6)), rng)
        cpm = normalize_cpm(cells.astype(np.float64), spec.DE_TARGET_SUM)
        ns = []
        for gi in np.flatnonzero(gate)[::7]:      # subsample genes for speed
            u, pv = mannwhitneyu(cpm[:, gi], ctrl_cpm[:, gi], alternative='two-sided')
            ns.append(pv)
        padj = benjamini_hochberg(np.array(ns))
        counts.append(int((padj < 0.05).sum()) * 7)   # rescale to full gene count
    return float(np.mean(counts))

for label, cal in [
    ('v29 gain1.8 q0.25', DECalibration(0.0, 1.8, 1.0, 0.25, pg)),
    ('gain0.6 q0.25', DECalibration(0.0, 0.6, 1.0, 0.25, pg)),
    ('gain0.6 q0.75', DECalibration(0.0, 0.6, 1.0, 0.75, pg)),
    ('gain0.6 q0.90', DECalibration(0.0, 0.6, 1.0, 0.90, pg)),
    ('gain0.6 thr0.3', DECalibration(0.3, 0.6, 1.0, 0.0, pg)),
    ('gain0.6 thr0.6', DECalibration(0.6, 0.6, 1.0, 0.0, pg)),
    ('gain1.8 thr0.6', DECalibration(0.6, 1.8, 1.0, 0.0, pg)),
]:
    n = nsig_for(cal)
    print(f'{label:20s} mean_called_genes ~ {n:7.0f}', flush=True)
print('DONE', flush=True)