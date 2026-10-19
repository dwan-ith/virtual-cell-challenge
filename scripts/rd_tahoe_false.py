import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import pds_proxy_fast
stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]
holdout='replogle_gw/RPE1'
train={(c,g):v for (c,g),v in pairs.items() if c!=holdout and 'tahoe' not in c and 'nadig' not in c}
test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:100]
truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
tgt=basals[holdout]
prior=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, shared_mix=0.0, zero_unseen=True).fit(nested, {k:v for k,v in basals.items() if k!=holdout and 'tahoe' not in k and 'nadig' not in k}, genes)
pred=prior.predict_many(list(truth.keys()), tgt)
print(f"tahoe False pds {pds_proxy_fast(pred, truth, genes):.3f}", flush=True)
