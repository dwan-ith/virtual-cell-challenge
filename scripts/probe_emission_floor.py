"""Is the over-calling caused by our LFC, or by the emission procedure itself?

Arms (all on real context_A controls, same 400-vs-18400 Wilcoxon):
  1. v29 emission (poisson_thin with v29 multipliers)   <- what we ship
  2. gain 0.6
  3. pure control COPY (multipliers = 1 everywhere)
  4. control copy but cells drawn WITHOUT replacement from held-out controls
If (3) also calls thousands of genes, the floor is the test's power against
resampled controls, not our magnitudes -> no LFC tuning can fix fid/jac/reach,
and the only lever left is emission that is not a control resample.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import anndata as ad
from scipy.stats import mannwhitneyu

from vcc_model.dataset import load_gene_names_csv, harmonize_to_axis, detect_columns
from vcc_model.countsim import poisson_thin, normalize_cpm, benjamini_hochberg
from vcc_model import spec

genes = load_gene_names_csv('data/vcc2026/gene_names.csv')
a = ad.read_h5ad('data/vcc2026/context_A.h5ad')
tcol, _ = detect_columns(a.obs)
lab = a.obs[tcol].astype(str).to_numpy()
X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
ctrl = np.asarray(X[np.flatnonzero(lab == 'non-targeting')].todense(), dtype=np.float64)
rng = np.random.default_rng(0)

# reference pool = first 80% of controls; emissions compared against ALL controls
ref = ctrl[:int(0.8 * ctrl.shape[0])]
pool = ctrl[int(0.8 * ctrl.shape[0]):]
cpm_all = normalize_cpm(ctrl, spec.DE_TARGET_SUM)
gate = cpm_all.mean(axis=0) > spec.DE_MIN_CPM_CONTROL
gidx = np.flatnonzero(gate)[::7]
print(f'ref {ref.shape} pool {pool.shape} gate {int(gate.sum())}', flush=True)

def count_sig(cells):
    cpm = normalize_cpm(cells.astype(np.float64), spec.DE_TARGET_SUM)
    ps = []
    for gi in gidx:
        _, pv = mannwhitneyu(cpm[:, gi], cpm_all[:, gi], alternative='two-sided')
        ps.append(pv)
    padj = benjamini_hochberg(np.array(ps))
    return int((padj < 0.05).sum()) * 7

# 1) exact control copy: 400 cells taken from INSIDE the reference set
copy_in = ref[rng.choice(ref.shape[0], size=400, replace=False)]
print(f'copy-from-reference      called ~ {count_sig(copy_in):7d}', flush=True)
# 2) resample WITH replacement from reference (what we ship)
copy_rep = ref[rng.choice(ref.shape[0], size=400, replace=True)]
print(f'resample-from-reference  called ~ {count_sig(copy_rep):7d}', flush=True)
# 3) 400 cells from a DISJOINT pool
disj = pool[rng.choice(pool.shape[0], size=400, replace=False)]
print(f'copy-from-disjoint-pool  called ~ {count_sig(disj):7d}', flush=True)
# 4) disjoint pool, poisson_thin with multiplier 1 (no biological shift)
thin1 = poisson_thin(disj, np.ones(len(genes)), rng)
print(f'disjoint + thinning(1x) called ~ {count_sig(thin1):7d}', flush=True)
print('DONE', flush=True)