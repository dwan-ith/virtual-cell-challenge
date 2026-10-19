"""Part 2 of local sweep: q sweep + tahoe mode (part A done: gain 0.6 wins)."""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np

from vcc_model.ensemble import pds_proxy_fast
from vcc_model.dynamics_residual import pds_proxy_count
from vcc_model.prior import LFCTransferPrior
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.calibrate import DECalibration

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

TAHOE = sorted(c for c in basals if 'nadig' in c)
REAL = {
    'replogle_nadig/k562': 'replogle_gw/K562',
    'replogle_nadig/rpe1': 'replogle_gw/RPE1',
    'replogle_nadig/hepg2': None,
    'replogle_nadig/jurkat': None,
}
esm = load_esm_embeddings(Path("data/protein/se_full_18533.npz"))
print('tahoe ctx', TAHOE, flush=True)

def build_basals(mode: str) -> dict:
    b = dict(basals)
    if mode == 'fake':
        return b
    if mode == 'fixed':
        for t, src in REAL.items():
            if src is not None and src in basals:
                b[t] = basals[src]
            else:
                b.pop(t, None)
        return b
    if mode == 'drop':
        for t in TAHOE:
            b.pop(t, None)
        return b
    raise ValueError(mode)

def run(holdout: str, mode: str, gain: float, q: float) -> tuple[float, float]:
    b_all = build_basals(mode)
    train = {(c, g): v for (c, g), v in pairs.items()
             if c != holdout and c in b_all}
    tb = {k: v for k, v in b_all.items() if k != holdout}
    test = {(c, g): v for (c, g), v in pairs.items() if c == holdout}
    tgenes = sorted({g for _, g in test})[:150]
    truth = {g: test[(holdout, g)] for g in tgenes if (holdout, g) in test}
    if holdout not in b_all:
        return float('nan'), float('nan')
    held = b_all[holdout]
    nested = {}
    for (c, g), v in train.items():
        nested.setdefault(c, {})[g] = v
    prior = LFCTransferPrior(
        similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
        gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
        zero_unseen=True, default_conservation=0.45).fit(nested, tb, genes)
    pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
    cal = DECalibration(threshold=0.0, gain=gain, power=1.0, q_threshold=q, per_gene_gain=pg)
    preds = {g: np.asarray(cal.transform(prior.predict_lfc(g, held), gene=g), dtype=np.float64)
             for g in truth}
    return (pds_proxy_fast(preds, truth, genes),
            pds_proxy_count(preds, truth, held, genes))

print('=== B) gain=1.8, q sweep (fake) ===', flush=True)
for q in [0.0, 0.25, 0.5, 0.75]:
    r1 = run('replogle_gw/RPE1', 'fake', 1.8, q)
    r2 = run('vcc2025/H1', 'fake', 1.8, q)
    print(f' q {q:.2f} | RPE1 lfc {r1[0]:.4f} cnt {r1[1]:.4f} | H1 lfc {r2[0]:.4f} cnt {r2[1]:.4f}', flush=True)

print('=== B2) gain=0.6, q sweep (fake) ===', flush=True)
for q in [0.0, 0.25, 0.5, 0.75]:
    r1 = run('replogle_gw/RPE1', 'fake', 0.6, q)
    r2 = run('vcc2025/H1', 'fake', 0.6, q)
    print(f' q {q:.2f} | RPE1 lfc {r1[0]:.4f} cnt {r1[1]:.4f} | H1 lfc {r2[0]:.4f} cnt {r2[1]:.4f}', flush=True)

print('=== C) tahoe mode @ gain=0.6 q=0.25 ===', flush=True)
for mode in ['fake', 'fixed', 'drop']:
    r1 = run('replogle_gw/RPE1', mode, 0.6, 0.25)
    r2 = run('vcc2025/H1', mode, 0.6, 0.25)
    print(f' {mode:5s} | RPE1 lfc {r1[0]:.4f} cnt {r1[1]:.4f} | H1 lfc {r2[0]:.4f} cnt {r2[1]:.4f}', flush=True)
print('DONE', flush=True)
