import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad, numpy as np, scipy.sparse as sp
from vcc_model.dataset import PseudobulkStore

# Load H1 controls
ctrl_path=Path("artifacts/stores/vcc2025_h1__controls.npz")
z=np.load(ctrl_path, allow_pickle=True)
mat=sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
# Need gene_names for var
store=PseudobulkStore.load("artifacts/stores/vcc2025_h1.npz")
gene_names=list(store.gene_names)
# Create AnnData for controls
adata=ad.AnnData(X=mat, var=dict(var_names=gene_names))
adata.obs["target_gene"]="non-targeting"
# Save temp
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    in_path=Path(tmp)/"h1_ctrl.h5ad"
    adata.write_h5ad(in_path)
    print(f"wrote {in_path} {adata.shape}")
    # Try state emb
    from vcc_model.state_adapter import StateAdapter
    sa=StateAdapter(se_dir="data/state/se600m", st_dir="data/state/st_k562_zeroshot")
    print(f"status {sa.status()}")
    # Try embed
    emb=sa.embed_controls(in_path)
    print(f"emb {emb.shape if emb is not None else None}")
    if emb is not None:
        print(f"emb mean {emb.mean():.4f} std {emb.std():.4f}")
    # Try predict for one gene
    delta=sa.predict_delta(in_path, gene="ADNP", context="H1")
    print(f"delta {delta.shape if delta is not None else None} mean {delta.mean() if delta is not None else None}")
