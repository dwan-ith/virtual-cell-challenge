import sys; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import pds_proxy_fast
from vcc_model.protein import load_esm_embeddings
stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]
esm=load_esm_embeddings('data/protein/se_full_18533.npz')
configs={
    'v29': dict(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True, beta_deshared=0.0),
    'temp03': dict(similarity_temperature=0.3, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True, beta_deshared=0.0),
    'desh02': dict(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True, beta_deshared=0.2),
    'temp03_desh015': dict(similarity_temperature=0.3, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True, beta_deshared=0.15),
}
holdout='vcc2025/H1'
train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:200]
truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
tgt=basals[holdout]
for name,cfg in configs.items():
    pri=LFCTransferPrior(**cfg).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, genes)
    preds=pri.predict_many(list(truth.keys()), tgt)
    pds=pds_proxy_fast(preds, truth, genes)
    nmae=np.mean([np.abs(preds[g]-truth[g]).mean() for g in truth])
    print(f"{name:15s} pds={pds:.3f} nmae={nmae:.3f}", flush=True)
print("H1 sweep done", flush=True)
holdout='replogle_gw/RPE1'
train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:200]
truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
tgt=basals[holdout]
for name,cfg in configs.items():
    pri=LFCTransferPrior(**cfg).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, genes)
    preds=pri.predict_many(list(truth.keys()), tgt)
    pds=pds_proxy_fast(preds, truth, genes)
    print(f"RPE1 {name:15s} pds={pds:.3f}", flush=True)
