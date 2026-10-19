"""Decisive test: which decoder preserves real-cell dispersion, and what does it
do to the DE call set + expression error?

We know poisson_thin calls ~6223 genes on PURE CONTROL cells (no biology).
Test multinomial_thin (covariance-preserving) on the same arm, plus both on a
real biological arm, and compare against the scorer-like quantities.

Arms:
  A control + poisson_thin        (current pipeline)
  B control + multinomial_thin
  C control + no thinning (copy)
  D prior-LFC + poisson_thin
  E prior-LFC + multinomial_thin
Report n_called genes and the expression distance to the control bulk profile
(a local stand-in for the mse member: squared log-profile distance).
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import anndata as ad
from scipy.stats import mannwhitneyu

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.dataset import load_gene_names_csv, harmonize_to_axis, detect_columns
from vcc_model.countsim import poisson_thin, multinomial_thin, normalize_cpm, benjamini_hochberg, log_profile
from vcc_model import spec
import pandas as pd

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
cal = DECalibration(0.0, 1.8, 1.0, 0.25, pg)

a = ad.read_h5ad('data/vcc2026/context_A.h5ad')
tcol, _ = detect_columns(a.obs)
lab = a.obs[tcol].astype(str).to_numpy()
X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
ctrl = np.asarray(X[np.flatnonzero(lab == 'non-targeting')].todense(), dtype=np.float32)
rng = np.random.default_rng(0)
cpm_all = normalize_cpm(ctrl, spec.DE_TARGET_SUM)
gate = cpm_all.mean(axis=0) > spec.DE_MIN_CPM_CONTROL
gidx = np.flatnonzero(gate)[::7]
ctrl_bulk = log_profile(ctrl.sum(axis=0), spec.BULK_TARGET_SUM)
panel = [str(g) for g in pd.read_csv('data/vcc2026/pert_counts.csv').iloc[:, 0].tolist()[:6]]
basal = np.log(np.maximum(cpm_all.mean(axis=0), 1e-9) + 1e-9)
pool = ctrl[rng.choice(ctrl.shape[0], size=1500, replace=False)]

def stats(cells):
    cpm = normalize_cpm(cells.astype(np.float64), spec.DE_TARGET_SUM)
    ps = []
    for gi in gidx:
        _, pv = mannwhitneyu(cpm[:, gi], cpm_all[:, gi], alternative='two-sided')
        ps.append(pv)
    n = int((benjamini_hochberg(np.array(ps)) < 0.05).sum()) * 7
    bulk = log_profile(cells.sum(axis=0), spec.BULK_TARGET_SUM)
    dist = float(np.mean((bulk - ctrl_bulk) ** 2))
    return n, dist

arms = {}
arms['ctrl + poisson'] = [poisson_thin(pool[rng.choice(pool.shape[0], 400, True)], np.ones(len(genes)), rng) for _ in range(3)]
arms['ctrl + multinomial'] = [multinomial_thin(pool[rng.choice(pool.shape[0], 400, True)], np.ones(len(genes)), rng) for _ in range(3)]
arms['ctrl + copy'] = [pool[rng.choice(pool.shape[0], 400, False)] for _ in range(3)]
for g in panel[:3]:
    lfc = np.asarray(cal.transform(prior.predict_lfc(g, basal), gene=g), dtype=np.float64)
    m = np.exp2(np.clip(lfc, -6, 6))
    arms[f'LFC({g}) + poisson'] = [poisson_thin(pool[rng.choice(pool.shape[0], 400, True)], m, rng) for _ in range(3)]
    arms[f'LFC({g}) + multinomial'] = [multinomial_thin(pool[rng.choice(pool.shape[0], 400, True)], m, rng) for _ in range(3)]

for k, v in arms.items():
    ns, ds = [], []
    for cells in v:
        n, d = stats(cells)
        ns.append(n); ds.append(d)
    print(f'{k:28s} called ~ {np.mean(ns):7.0f}   expr_dist_to_ctrl {np.mean(ds):.4f}', flush=True)
print('DONE', flush=True)