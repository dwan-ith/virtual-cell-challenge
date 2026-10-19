import anndata as ad, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
a = ad.read_h5ad('artifacts/submission/prediction_v13.h5ad')
print(f'Shape: {a.shape}')
print(f'Obs: {list(a.obs.columns)}')
print(f'X type: {type(a.X).__name__}')
if hasattr(a.X, 'nnz'): print(f'nnz: {a.X.nnz:,}')
tg = 'target_gene' in a.obs
ct = 'context' in a.obs
print(f'Target genes: {a.obs["target_gene"].nunique() if tg else "N/A"}')
print(f'Contexts: {a.obs["context"].unique().tolist() if ct else "N/A"}')
print(f'Cells per pert: {a.shape[0] // (a.obs["target_gene"].nunique() * a.obs["context"].nunique()) if tg and ct else "N/A"}')
