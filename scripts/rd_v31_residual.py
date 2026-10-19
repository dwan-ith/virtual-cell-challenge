import sys; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from vcc_model.ensemble import HybridPredictor, pds_proxy_fast
from vcc_model.residual import ResidualConfig
from vcc_model.protein import load_esm_embeddings

stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]

esm_path="data/protein/se_full_18533.npz"

def run_holdout(holdout, cfg):
    train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
    test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
    import random; random.seed(0)
    tgenes=sorted({g for _,g in test})[:200]
    truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
    print(f"\n[{holdout}] train {len(train)} test {len(truth)} cfg epochs={cfg.epochs} hidden={cfg.hidden}", flush=True)
    # prior_kwargs = v29 best
    pk=dict(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_path=esm_path, shared_mix=0.0, zero_unseen=True)
    model=HybridPredictor.fit(lfc_pairs=train, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=genes, residual_cfg=cfg, fit_residual=True, select_weights=False, prior_kwargs=pk)
    print(f" weights {model.weights} fit_info {model.fit_info.get('residual',{})}", flush=True)
    tgt=basals[holdout]
    # blend without residual vs with
    preds={}
    for g in truth:
        # blend_lfc uses residual if weights resid>0 and model has residual
        # force stat-only vs blend
        preds[g]=model.blend_lfc(g, tgt, weights=(1.0,0.0,0.0)) if False else model.blend_lfc(g, tgt)
        # actually test both: we want model.weights vs prior-only
    # compute both
    # prior-only
    prior_only={g: model.prior.predict_lfc(g, tgt) for g in truth}
    # blended (uses model.weights)
    blended={g: model.blend_lfc(g, tgt) for g in truth}
    pds_prior=pds_proxy_fast(prior_only, truth, genes)
    pds_blend=pds_proxy_fast(blended, truth, genes)
    print(f"  pds_prior {pds_prior:.3f} pds_blend {pds_blend:.3f} delta {pds_blend-pds_prior:+.3f}", flush=True)
    # also calibrated? but PDS is scale invariant so calibration no effect
    return pds_prior, pds_blend

for holdout in ['replogle_gw/RPE1','vcc2025/H1']:
    for cfg in [ResidualConfig(epochs=30, hidden=128, lfc_rank=32, basal_pcs=16, seed=0), ResidualConfig(epochs=60, hidden=128, lfc_rank=32, basal_pcs=16, seed=0)]:
        run_holdout(holdout, cfg)
print("done", flush=True)
