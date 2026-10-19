import sys
sys.path.insert(0, '.')
import numpy as np, anndata as ad, scipy.sparse as sp, pandas as pd
from vcc_model.metrics import _jackknife_c
from vcc_model.atlas_core import desired_mean
from scripts.atlas_predict import control_template
from vcc_model.atlas_core import dual_moment_counts

TS = 5e4
genes = [str(x) for x in pd.read_csv('data/vcc2026/gene_names.csv').iloc[:, 0]]

for ctx in 'A':
    a = ad.read_h5ad(f'data/vcc2026/context_{ctx}.h5ad')
    X = sp.csr_matrix(a.X, dtype=np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    print(f'=== context {ctx}: {X.shape} ===')

    # Real 400-cell group dispersion (proxy for a real perturbed group)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lib), 400, replace=False)
    real = np.asarray(X[idx].todense())
    C_real = _jackknife_c(real, TS)
    # and a group-matched-by-depth subsample
    print(f'real400: C={C_real:.3f}  genes/cell={(real>0).sum(1).mean():.0f}  depth={real.sum(1).mean():.0f}')

    for pool_k in [1, 4]:
        template, depths, mean, bulk = control_template(
            f'data/vcc2026/context_{ctx}.h5ad', genes, 400, pool_k, 20260910)
        p = mean / mean.sum()
        emit = dual_moment_counts(template, p, bulk, depths=depths, seed=7)
        C_hat = _jackknife_c(emit.astype(np.float64), TS)
        print(f'pool_k={pool_k}: emit C={C_hat:.3f}  C/C_real={C_hat/C_real:.3f} '
              f'genes/cell={(emit>0).sum(1).mean():.0f} depth={emit.sum(1).mean():.0f}')
    del a, X
