import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import pds_proxy_fast
from vcc_model.protein import load_esm_embeddings
from vcc_model.surrogate import SurrogateCorrector
stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]
esm=load_esm_embeddings('data/protein/se_full_18533.npz')
holdout='vcc2025/H1'
train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
prior=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, genes)
print("prior done", flush=True)
surr=SurrogateCorrector(epochs=30, hidden=64, gamma=0.35, orth_thresh=0.2)
surr.fit(train, {k:v for k,v in basals.items() if k!=holdout}, genes, prior, esm_embeddings=esm)
print("surr fit done", flush=True)
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:200]
truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
tgt=basals[holdout]
pred0=prior.predict_many(list(truth.keys()), tgt)
pds0=pds_proxy_fast(pred0, truth, genes)
print(f"baseline {pds0:.3f}", flush=True)
pred1={}
for g in truth:
    base=pred0[g]
    emb=esm.get(g) if esm else None
    delta=surr.predict_correction(g, tgt, base, emb)
    pred1[g]=base + delta
pds1=pds_proxy_fast(pred1, truth, genes)
print(f"H1 baseline {pds0:.3f} corrected {pds1:.3f} delta {pds1-pds0:+.3f}", flush=True)
# try gamma 0.15
surr.gamma=0.15
pred2={}
for g in truth:
    base=pred0[g]
    emb=esm.get(g) if esm else None
    delta=surr.predict_correction(g, tgt, base, emb)
    pred2[g]=base + delta
print(f"H1 gamma 0.15 {pds_proxy_fast(pred2, truth, genes):.3f}", flush=True)
