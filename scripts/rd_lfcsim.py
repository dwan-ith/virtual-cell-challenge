import sys; sys.path.insert(0,'.')
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

def lfc_sim_weights(nested, target_ctx, holdout, temp=0.1):
    # compute context similarity as mean cosine of shared genes' LFC
    contexts=[c for c in nested if c!=holdout]
    sims={}
    for c in contexts:
        shared=[g for g in nested[c] if any(g in nested[other] for other in contexts if other!=c)]
        # use 200 shared genes sample
        import random; random.seed(1)
        if len(shared)>200:
            shared=random.sample(shared,200)
        cos=[]
        for g in shared:
            # compare c vs average of other contexts?
            # simple: cosine between c's LFC and mean of others
            others=[nested[other][g] for other in contexts if other!=c and g in nested[other]]
            if not others: continue
            mean=np.mean(np.stack(others),axis=0)
            a=nested[c][g]; b=mean
            den=np.linalg.norm(a)*np.linalg.norm(b)
            if den>1e-9:
                cos.append(float(np.dot(a,b)/den))
        sims[c]=float(np.mean(cos)) if cos else 0.0
    # softmax
    vals=np.array([max(0,sims[c]) for c in contexts])
    # use same softmax as prior
    def softmax(x,T):
        z=(x - x.max())/max(T,1e-6)
        e=np.exp(z)
        return e/e.sum()
    w=softmax(vals, temp)
    return {c: float(wi) for c,wi in zip(contexts,w)}, sims

for holdout in ['replogle_gw/RPE1','vcc2025/H1']:
    train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
    test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
    nested={}
    for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
    import random; random.seed(0)
    tgenes=sorted({g for _,g in test})[:200]
    truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
    tgt=basals[holdout]
    # baseline prior basal
    esm=load_esm_embeddings('data/protein/se_full_18533.npz')
    prior=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, genes)
    pred=prior.predict_many(list(truth.keys()), tgt)
    pds_baseline=pds_proxy_fast(pred, truth, genes)
    # lfc sim
    w_lfc, sims = lfc_sim_weights(nested, holdout, holdout, temp=0.1)
    print(f"\n{holdout} baseline pds {pds_baseline:.3f} sims { {k: round(v,3) for k,v in sims.items()} } w { {k: round(v,3) for k,v in w_lfc.items()} }")
    # also try using lfc sim to reweight prior manually: create new prior with uniform basal but override w_map?
    # quick approximate: predict with custom w_map via _predict_with
    # we can monkey-patch similarity_weights to return lfc weights
    orig_sw=prior.similarity_weights
    prior.similarity_weights=lambda tgt, exclude=None: w_lfc if not exclude or holdout not in exclude else {k:v for k,v in w_lfc.items() if k not in exclude}
    pred2=prior.predict_many(list(truth.keys()), tgt)
    pds2=pds_proxy_fast(pred2, truth, genes)
    print(f" lfc-weighted pds {pds2:.3f} delta {pds2-pds_baseline:+.3f}")
    prior.similarity_weights=orig_sw
