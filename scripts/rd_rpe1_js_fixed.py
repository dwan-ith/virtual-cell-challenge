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
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:200]
test2={k:v for k,v in test.items() if k[1] in tgenes}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
tb={k:v for k,v in basals.items() if k!=holdout}
for js_dim,js_str in [(None,None),(32,1.0),(32,0.3),(16,1.0)]:
  if js_dim is None:
    prior=LFCTransferPrior(similarity_temperature=0.1,basal_gate_cpm=5.0,use_alpha=False,james_stein=False).fit(nested,tb,genes)
    tag="js=False"
  else:
    prior=LFCTransferPrior(similarity_temperature=0.1,basal_gate_cpm=5.0,use_alpha=False,james_stein=True,james_stein_dim=js_dim,james_stein_strength=js_str).fit(nested,tb,genes)
    tag=f"js_dim={js_dim}_str={js_str}"
  tgt=basals[holdout]
  preds={g:prior.predict_lfc(g,tgt) for _,g in test2}
  truths={g:v for (_,g),v in test2.items()}
  p=pds_proxy_fast(preds,truths,genes)
  cos=float(np.mean([float(np.dot(preds[g],truths[g])/(np.linalg.norm(preds[g])*np.linalg.norm(truths[g])+1e-9)) for g in preds]))
  # factor stats: mean shrink?
  print(f"RESULT {tag} pds={p:.3f} cos={cos:.4f}",flush=True)
print("Done",flush=True)
