import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from vcc_model.ensemble import pds_proxy_fast
from vcc_model.protein import load_esm_embeddings

stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v

esm=load_esm_embeddings('data/protein/se_full_18533.npz')
# prepare X = embeddings, Y = mean LFC across contexts (pooled)
from collections import defaultdict
pooled={}
counts=defaultdict(int)
sums={}
for (c,g),v in pairs.items():
    if g not in esm: continue
    if g not in sums:
        sums[g]=np.zeros(len(genes), dtype=np.float64)
    sums[g]+=v
    counts[g]+=1
for g in sums:
    pooled[g]=sums[g]/counts[g]

# filter genes with both embedding and pooled
genes_with=[g for g in pooled if g in esm]
print(f"pooled {len(pooled)} with emb {len(genes_with)}", flush=True)
X=np.stack([esm[g] for g in genes_with])
Y=np.stack([pooled[g] for g in genes_with])
print(f"X {X.shape} Y {Y.shape}", flush=True)
# ridge: Y ~ X
# use small alpha
from sklearn.decomposition import PCA
# try PCA to 32 then ridge to Y PCA?
# direct ridge 5120 -> 18533 is heavy (95M params). Use low-rank: PCA Y to 32, learn X->32, then decode
pca=PCA(n_components=32, random_state=0)
Yp=pca.fit_transform(Y)  # (n,32)
print(f"pca explained {pca.explained_variance_ratio_.sum():.3f}", flush=True)
reg=Ridge(alpha=1.0)
reg.fit(X, Yp)
# test on holdout genes: use RPE1 holdout genes
holdout='replogle_gw/RPE1'
test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:200]
truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
pred={}
for g in truth:
    if g in esm:
        yp=reg.predict(esm[g].reshape(1,-1))
        y=pca.inverse_transform(yp)[0]
        pred[g]=y
    else:
        pred[g]=np.zeros(len(genes))
pds=pds_proxy_fast(pred, truth, genes)
print(f"protein direct pds {pds:.3f}", flush=True)
# compare to prior baseline
from vcc_model.prior import LFCTransferPrior
# quick prior baseline
import pathlib
basals={}
for p in Path('artifacts/stores').glob('*.npz'):
    z=np.load(p,allow_pickle=True)
    for i,c in enumerate(z['basal_ctx']):
        basals[str(c)]=z['basal'][i]
train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
prior=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, shared_mix=0.0, zero_unseen=True).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, genes)
tgt=basals[holdout]
pred2=prior.predict_many(list(truth.keys()), tgt)
print(f"prior pds {pds_proxy_fast(pred2, truth, genes):.3f}", flush=True)
