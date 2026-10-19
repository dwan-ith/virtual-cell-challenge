import sys; sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from vcc_model.ensemble import HybridPredictor, pds_proxy_fast
from vcc_model.residual import ResidualConfig
stores=[p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs,basals,genes={},{},None
for p in stores:
    z=np.load(p,allow_pickle=True)
    if genes is None: genes=list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'],z['target'],z['lfc']): pairs[(str(c),str(g))]=v
    for i,c in enumerate(z['basal_ctx']): basals[str(c)]=z['basal'][i]
for holdout in ['replogle_gw/RPE1','vcc2025/H1']:
    train={(c,g):v for (c,g),v in pairs.items() if c!=holdout}
    test={(c,g):v for (c,g),v in pairs.items() if c==holdout}
    import random; random.seed(0)
    tgenes=sorted({g for _,g in test})[:200]
    truth={g: test[(holdout,g)] for g in tgenes if (holdout,g) in test}
    print(f"\n{holdout} train {len(train)}", flush=True)
    pk=dict(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, shared_mix=0.0, zero_unseen=True)
    cfg=ResidualConfig(epochs=15, hidden=128, lfc_rank=32, basal_pcs=16, seed=0, batch_size=256, max_pairs=4000)
    model=HybridPredictor.fit(lfc_pairs=train, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=genes, residual_cfg=cfg, fit_residual=True, select_weights=False, prior_kwargs=pk)
    print(f" resid {model.fit_info.get('residual')}", flush=True)
    tgt=basals[holdout]
    prior_only={g: model.prior.predict_lfc(g, tgt) for g in truth}
    # test multiple blend weights
    for w in [0.0, 0.35, 0.7, 1.0]:
        blended={g: (1.0*prior_only[g] + w*model.residual.predict(prior_only[g], tgt, g) if model.residual else prior_only[g]) for g in truth}
        pds=pds_proxy_fast(blended, truth, genes)
        print(f" w_resid {w:.2f} pds {pds:.3f}", flush=True)
