"""Generate v33 with OOD-gated blend, reusing trained ensemble."""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import gc, sys as _sys
from vcc_model.dataset import load_controls_bundle
from vcc_model.ensemble_surrogate import EnsembleDynamics
from vcc_model.prior import LFCTransferPrior
from vcc_model.countsim import sparse_log_mean_cpm, log_mean_cpm
from vcc_model.submission import StreamingSubmissionWriter

# load stores for prior
stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs, basals, genes = {}, {}, None
for p in stores:
    z=np.load(p, allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']):
        basals[str(c)]=z['basal'][i]
nested={}
for (c,g),v in pairs.items():
    nested.setdefault(c,{})[g]=v
prior=LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0, use_alpha=False, zero_unseen=True).fit(nested, basals, genes)
print(f"[prior] {len(prior.contexts)}", flush=True)

# load ensemble
ens=EnsembleDynamics.load("artifacts/ensemble_v32_3members")
# reattach ESM for predict
import pathlib
d=np.load(Path('data/protein/se_full_18533.npz'), allow_pickle=True)
gn=[str(x) for x in d['gene_names']]; mat=d['embeddings']
n=np.linalg.norm(mat, axis=1, keepdims=True); n=np.where(n<1e-9,1.0,n); mat=mat/n
esm={g:mat[i].astype(np.float32) for i,g in enumerate(gn)}
ens._esm=esm
ens.gene_names=genes
print(f"[ens] {len(ens)} members OOD thr {ens.ood.threshold_ if ens.ood else 'none'}", flush=True)

controls=load_controls_bundle(Path("data/vcc2026"), genes)
panel=pd.read_csv("data/vcc2026/pert_counts.csv")
panel_genes=panel[panel.columns[0]].astype(str).tolist()[:300]
print(f"[panel] {len(panel_genes)} {panel_genes[:2]}", flush=True)

rng=np.random.default_rng(0)
out=Path("artifacts/submission/pred_v34.h5ad")
out.parent.mkdir(parents=True, exist_ok=True)

# OOD-gated blend: w=0.2 if flagged else 0.05
def blend_for(ctx_basal):
    if ens.ood is None:
        return 0.15
    flagged=ens.ood.flagged(ctx_basal.reshape(1,-1))[0]
    # also use uncertainty
    return 0.20 if flagged else 0.05

with StreamingSubmissionWriter(out, genes) as writer:
    for ctx in sorted(controls.keys())[:3]:
        ctrl=controls[ctx]
        # basal sample 4000
        sel=rng.choice(ctrl.shape[0], size=min(4000, ctrl.shape[0]), replace=False)
        basal=sparse_log_mean_cpm(ctrl[sel], 1e6)
        w=blend_for(basal)
        print(f"[ctx {ctx}] w={w:.2f} flagged={ens.ood.flagged(basal.reshape(1,-1))[0] if ens.ood else 'no'}", flush=True)
        for gene in panel_genes:
            p=prior.predict_lfc(gene, basal)
            d=ens.predict_lfc(gene, basal)
            lfc=(1-w)*p + w*d
            lfc=np.clip(lfc,-6,6)
            mult=np.exp2(lfc)
            pick=rng.choice(ctrl.shape[0], size=400, replace=True)
            base=np.asarray(ctrl[pick].todense(), dtype=np.float32)
            from vcc_model.countsim import poisson_thin
            cells=poisson_thin(base, mult, rng)
            writer.add_block(cells, [gene]*400, [ctx]*400)
            del base, cells
        print(f"[ctx {ctx}] done", flush=True)
        gc.collect()
    print(f"[writer] {writer._nnz:,} {writer._n_obs:,}", flush=True)

print(f"[save] {out} {out.stat().st_size/1e6:.1f} MB", flush=True)
# pack
import subprocess
vcc=out.with_suffix(".vcc")
subprocess.run([_sys.executable, "scripts/pack_vcc.py", str(out), str(vcc)], check=True)
print(f"[pack] {vcc}", flush=True)
