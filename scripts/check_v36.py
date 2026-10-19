import sys
sys.path.insert(0, '.')
import anndata as ad
import numpy as np
from vcc_model.submission import validate_h5ad_file_light
from vcc_model.dataset import load_gene_names_csv

genes = load_gene_names_csv('data/vcc2026/gene_names.csv')
p = 'artifacts/submission/pred_v36.h5ad'
problems = validate_h5ad_file_light(p, genes, contexts=("A", "B", "C"),
                                    perts_per_context=300, cells_per_pert=400)
print('validator problems:', problems, flush=True)
a = ad.read_h5ad(p, backed='r')
print('n_obs', a.n_obs, 'n_vars', a.n_vars, flush=True)
print('ctx', sorted(a.obs['context'].cat.categories.tolist()), flush=True)
print('n_perts', len(a.obs['target_gene'].cat.categories), flush=True)
blk = a.X[:2000]
d = blk
if not hasattr(d, 'toarray'):
    from scipy import sparse as _sp
    d = _sp.csr_matrix(d)
mx = np.asarray(d.max(axis=1).todense()).ravel()
print('max cell counts (2000 rows):', int(mx.max()), 'cap', 1000000, flush=True)
a.file.close()
print('OK', flush=True)
