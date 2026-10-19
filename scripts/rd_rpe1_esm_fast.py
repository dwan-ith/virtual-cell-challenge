import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vcc_model.dataset import PseudobulkStore
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import pds_proxy_fast
stores=[p for p in Path("artifacts/stores").glob("*.npz") if "__controls" not in p.name]
pairs={}; basals={}; genes=None
for p in stores:
  z=np.load(p,allow_pickle=True)
  if genes is None: genes=list(z['gene_names'].astype(str))
  for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
  for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]
holdout="replogle_gw/RPE1"
train={k:v for k,v in pairs.items() if k[0]!=holdout}
test={k:v for k,v in pairs.items() if k[0]==holdout}
print(f"train {len(train)} test {len(test)}",flush=True)
# subsample test to 200 for speed
import random
random.seed(0)
test_genes=sorted({g for _,g in test})[:200]
test2={k:v for k,v in test.items() if k[1] in test_genes}
print(f"subsample {len(test2)}",flush=True)
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
tb={k:v for k,v in basals.items() if k!=holdout}
# baseline
prior=LFCTransferPrior(similarity_temperature=0.1,basal_gate_cpm=5.0,use_alpha=False).fit(nested,tb,genes)
print("fit base done",flush=True)
tgt=basals[holdout]
preds={g:prior.predict_lfc(g,tgt) for _,g in test2}
truths={g:v for (_,g),v in test2.items()}
preds2={}; truths2={}
for g in preds: preds2[g]=preds[g]; truths2[g]=truths[g]
p=pds_proxy_fast(preds2,truths2,genes)
cos=float(np.mean([float(np.dot(preds2[g],truths2[g])/(np.linalg.norm(preds2[g])*np.linalg.norm(truths2[g])+1e-9)) for g in preds2]))
print(f"RESULT base pds={p:.3f} cos={cos:.4f}",flush=True)
# ESM
from vcc_model.protein import load_esm_embeddings
esm=load_esm_embeddings("data/protein/se_full_18533.npz")
print(f"esm {len(esm) if esm else 0}",flush=True)
prior2=LFCTransferPrior(similarity_temperature=0.1,basal_gate_cpm=5.0,use_alpha=False,gene_conditioned=True,esm_embeddings=esm).fit(nested,tb,genes)
print(f"fit esm done centroids={len(prior2._esm_context_centroids) if prior2._esm_context_centroids else 0}",flush=True)
preds_e={g:prior2.predict_lfc(g,tgt) for g in preds2}
pe=pds_proxy_fast(preds_e,truths2,genes)
cose=float(np.mean([float(np.dot(preds_e[g],truths2[g])/(np.linalg.norm(preds_e[g])*np.linalg.norm(truths2[g])+1e-9)) for g in preds_e]))
print(f"RESULT esm pds={pe:.3f} cos={cose:.4f} dPDS={pe-p:+.3f} dCOS={cose-cos:+.4f}",flush=True)
print("Done",flush=True)
