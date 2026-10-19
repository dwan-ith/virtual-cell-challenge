import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from vcc_model.prior import LFCTransferPrior
from vcc_model.dataset import load_controls_bundle

stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs, basals, genes = {}, {}, None
for p in stores:
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z['gene_names'].astype(str))
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c), str(g))] = v
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]

nested = {}
for (c, g), v in pairs.items():
    nested.setdefault(c, {})[g] = v
prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0,
                         use_alpha=False, zero_unseen=True).fit(nested, basals, genes)
print('contexts:', prior.contexts)
print('n pairs per ctx:', {c: len(v) for c, v in nested.items()})

controls = load_controls_bundle(Path('data/vcc2026'), genes)
from vcc_model.countsim import sparse_log_mean_cpm
for ctx in ['A', 'B', 'C']:
    basal = sparse_log_mean_cpm(controls[ctx][:4000], 1e6)
    w = prior.similarity_weights(basal)
    tot_tahoe = sum(v for k, v in w.items() if 'nadig' in k or 'tahoe' in k)
    print(f'--- target {ctx} ---')
    for k in sorted(w, key=lambda k: -w[k]):
        print(f'  {k:28s} {w[k]:.4f}')
    print(f'  TAHOE BLOC TOTAL: {tot_tahoe:.4f}')
