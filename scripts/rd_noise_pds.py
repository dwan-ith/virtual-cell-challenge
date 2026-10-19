"""Noise-inclusive local PDS gate, computed on BULK profiles (what the scorer reads).

metrics.pds_cosine only ever touches pert_map[name].sum(axis=0) -- the bulk.
So emitting individual cells is wasted work. This simulates the emitted bulk
directly with correct multinomial noise, which is precisely the term that makes
official PDS non-scale-invariant (SNR-limited) and that every earlier LFC-space
proxy missed.

Emission model per target (matches what a submission's cells sum to):
    T        = total counts of the 400 emitted control cells
    q_g      = base_profile_g * mult_g / sum(base_profile * mult)
    bulk_g   ~ Multinomial(T, q)      <- adds the sampling noise PDS actually sees
PDS then compares log_profile(bulk) between predictions and truths.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import anndata as ad

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.dataset import load_gene_names_csv, harmonize_to_axis, detect_columns
from vcc_model.countsim import log_profile
from vcc_model.metrics import pds_cosine, ContextData

store = np.load('artifacts/stores/replogle_gw_k562.npz', allow_pickle=True)
genes = [str(g) for g in store['gene_names']]
lfc = store['lfc']
tgt = [str(t) for t in store['target']]
ctxs = [str(c) for c in store['context']]
k562 = [(i, t) for i, t in enumerate(tgt)
        if ctxs[i] == 'replogle_gw/K562'
        and np.isfinite(lfc[i]).all() and np.abs(lfc[i]).max() > 0.2]
print(f'usable K562 targets: {len(k562)}', flush=True)

a = ad.read_h5ad('data/vcc2026/context_A.h5ad')
tcol, _ = detect_columns(a.obs)
lab = a.obs[tcol].astype(str).to_numpy()
X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
ctrl = np.asarray(X[np.flatnonzero(lab == 'non-targeting')].todense(), dtype=np.float32)
rng = np.random.default_rng(0)
pool_idx = rng.choice(ctrl.shape[0], size=400, replace=False)
pool = ctrl[pool_idx].astype(np.float64)
base_profile = pool.sum(axis=0)
T = int(base_profile.sum())
ctrl_bulk = base_profile.copy()
print(f'scaffold {ctrl.shape} emitted-total {T}', flush=True)

basals, pairs = {}, {}
HOLD = 'replogle_gw/K562'
for p in Path('artifacts/stores').glob('*.npz'):
    if '__controls' in p.name:
        continue
    z = np.load(p, allow_pickle=True)
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c), str(g))] = v
# LOCO: drop the held-out context from training so K562 is genuinely unseen
pairs = {k: v for k, v in pairs.items() if k[0] != HOLD}
basals.pop(HOLD, None)
nested = {}
for (c, g), v in pairs.items():
    nested.setdefault(c, {})[g] = v
print(f'prior train contexts (K562 held out): {sorted(nested)}', flush=True)
esm = load_esm_embeddings(Path("data/protein/se_full_18533.npz"))
prior = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
                         gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
                         zero_unseen=True, default_conservation=0.45).fit(nested, basals, genes)
pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
from vcc_model.countsim import sparse_log_mean_cpm
basal = sparse_log_mean_cpm(pool[:200], 1e6)


def emit_bulk(mult: np.ndarray) -> np.ndarray:
    w = base_profile * mult
    tot = w.sum()
    q = w / tot if tot > 0 else np.full_like(w, 1.0 / len(w))
    return rng.multinomial(T, q)


SEL = k562[:120]
truth_bulk = {}
for i, t in SEL:
    tv = np.clip(np.asarray(lfc[i], dtype=np.float64), -6, 6)
    # 2-D (1, G): metrics.pds_cosine does pert_map[name].sum(axis=0)
    truth_bulk[t] = emit_bulk(np.exp2(tv)).reshape(1, -1)
print('truth bulks built', flush=True)

print('\n=== official pds_cosine on emitted BULKS, K562 HELD OUT (noise-inclusive) ===', flush=True)
res = {}
for gain in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0, 4.0]:
    cal = DECalibration(0.0, gain, 1.0, 0.25, pg)
    pred_bulk = {}
    for i, t in SEL:
        pl = np.asarray(cal.transform(prior.predict_lfc(t, basal), gene=t), dtype=np.float64)
        pred_bulk[t] = emit_bulk(np.exp2(np.clip(pl, -6, 6))).reshape(1, -1)
    cd = ContextData(np.asarray(genes), ctrl_bulk.reshape(1, -1), truth_bulk, pred_bulk)
    p = pds_cosine(cd)
    res[gain] = p
    print(f'  gain {gain:4.1f}  pds {p:.4f}', flush=True)

best = max(res, key=res.get)
print(f'\nBEST gain: {best} (pds {res[best]:.4f})', flush=True)
print(f'v29 shipped gain 1.8 -> local pds {res.get(1.8, float("nan")):.4f}', flush=True)
print('DONE', flush=True)