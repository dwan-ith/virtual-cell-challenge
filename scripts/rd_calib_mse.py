import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import HybridPredictor
from vcc_model.calibrate import DECalibration
from vcc_model.metrics import ContextData, evaluate_context, baseline_anchor, replicate_anchor, scale_scores
from vcc_model.countsim import poisson_thin, log_mean_cpm
from vcc_model.protein import load_esm_embeddings

stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]

holdout='replogle_gw/RPE1'
train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
nested={}
for (c,g),v in train.items(): nested.setdefault(c,{})[g]=v
import random; random.seed(0)
tgenes=sorted({g for _,g in test})[:50]
truth_lfc={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
tgt=basals[holdout]
esm=load_esm_embeddings('data/protein/se_full_18533.npz')
prior=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0, zero_unseen=True).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, genes)
# get prior predictions
pred_raw={g: prior.predict_lfc(g, tgt) for g in truth_lfc}
# simulate controls for official scoring
rng=np.random.default_rng(0)
mean_cpm=np.exp(tgt)
probs=mean_cpm/mean_cpm.sum()
n_ctrl=600
lib=5000
ctrl_counts=np.zeros((n_ctrl,len(genes)),dtype=int)
for i in range(n_ctrl):
    tot=rng.poisson(lib)
    ctrl_counts[i]=rng.multinomial(tot, probs)
for gain in [0.7,1.0,1.4,1.8,2.2,2.6,3.0]:
    cal=DECalibration(gain=gain, threshold=0.0, q_threshold=0.25)
    # generate cells
    true_cells={}
    pred_cells={}
    for g in truth_lfc:
        tl=np.clip(truth_lfc[g],-6,6)
        pl=np.clip(cal.transform(pred_raw[g]),-6,6)
        base=np.random.default_rng(hash(g)%10000).choice(ctrl_counts.shape[0], size=80, replace=False)
        base_cells=ctrl_counts[base]
        true_cells[g]=poisson_thin(base_cells, np.exp2(tl), np.random.default_rng(1))
        pred_cells[g]=poisson_thin(base_cells, np.exp2(pl), np.random.default_rng(2))
    cd=ContextData(np.array(genes), ctrl_counts, true_cells, pred_cells)
    raw=evaluate_context(cd)
    b=baseline_anchor(cd, seed=0)
    r=replicate_anchor(cd, n_splits=2, base_seed=0)
    scaled=scale_scores(raw,b,r)
    print(f"gain {gain:.1f} raw pds {raw['pds_cosine']:.3f} mse {raw['expr_mse_unbiased_capped_norm']:.3f} scaled pds {scaled['pds_cosine']:.3f} mse {scaled['expr_mse_unbiased_capped_norm']:.3f} overall {scaled['overall']:.3f}", flush=True)
