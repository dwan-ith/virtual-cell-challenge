"""Ablate Tahoe (replogle_nadig/*) stores: LOCO PDS with vs without.

Usage: python scripts/rd_tahoe_ablation.py
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np

from vcc_model.ensemble import pds_proxy_fast
from vcc_model.prior import LFCTransferPrior
from vcc_model.dynamics_residual import pds_proxy_count

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

TAHOE = [c for c in basals if 'nadig' in c]
print('tahoe contexts:', TAHOE, flush=True)

for holdout in ['replogle_gw/RPE1', 'vcc2025/H1']:
    test = {(c, g): v for (c, g), v in pairs.items() if c == holdout}
    tgenes = sorted({g for _, g in test})[:150]
    truth = {g: test[(holdout, g)] for g in tgenes if (holdout, g) in test}
    held_basal = basals[holdout]
    for label, drop in [('WITH tahoe', set()), ('WITHOUT tahoe', set(TAHOE))]:
        train = {(c, g): v for (c, g), v in pairs.items()
                 if c != holdout and c not in drop}
        train_basals = {k: v for k, v in basals.items() if k != holdout and k not in drop}
        nested = {}
        for (c, g), v in train.items():
            nested.setdefault(c, {})[g] = v
        prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0,
                                 use_alpha=False, zero_unseen=True).fit(nested, train_basals, genes)
        preds = {g: prior.predict_lfc(g, held_basal) for g in truth}
        p_lfc = pds_proxy_fast(preds, truth, genes)
        p_cnt = pds_proxy_count(preds, truth, held_basal, genes)
        print(f'{holdout} | {label:14s} | train {len(train):5d} | pds_lfc {p_lfc:.4f} | pds_cnt {p_cnt:.4f}', flush=True)
