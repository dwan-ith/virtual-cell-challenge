"""v11 A/B/C comparison: gate-only vs JS+gate vs baseline on RPE1 holdout."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import PseudobulkStore
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import pds_proxy_fast

t_start = time.time()
s1 = PseudobulkStore.load('artifacts/stores/replogle_gw_k562.npz')
s2 = PseudobulkStore.load('artifacts/stores/replogle_gw_rpe1.npz')
gene_names = [str(g) for g in s1.gene_names]
print(f'Loaded in {time.time()-t_start:.0f}s: K562 {len(s1.pairs)}p, RPE1 {len(s2.pairs)}p', flush=True)

train_lfc = {'replogle_gw/K562': {t: s1.lfc[i] for i, (_, t) in enumerate(s1.pairs)}}
train_basal = {'replogle_gw/K562': s1.basal['replogle_gw/K562']}

held_basal = s2.basal['replogle_gw/RPE1']
held_targets = sorted(set(t for _, t in s2.pairs))
truths = {}
for t in held_targets:
    for i, (c, tt) in enumerate(s2.pairs):
        if tt == t:
            truths[t] = s2.lfc[i].astype(np.float64)
            break

# --- v6 baseline ---
t0 = time.time()
p6 = LFCTransferPrior().fit(train_lfc, train_basal, gene_names)
p6.shared_mix = 0.0; p6.use_alpha = False
preds_v6 = {t: p6.predict_lfc(t, held_basal) for t in held_targets}
pds_v6 = pds_proxy_fast(preds_v6, truths, gene_names)
print(f'v6 baseline: pds={pds_v6:.4f} ({time.time()-t0:.0f}s)', flush=True)

# --- gate-only (reuse v6 predictions, zero gated) ---
for cp, label in [(1.0, 'gate1'), (2.0, 'gate2'), (5.0, 'gate5')]:
    preds_g = {}
    for t in held_targets:
        gi = p6._gene_idx.get(t)
        if gi is not None and float(np.exp(held_basal[gi]) - 1e-9) < cp:
            preds_g[t] = np.zeros(len(gene_names), dtype=np.float64)
        else:
            preds_g[t] = preds_v6[t]
    pds_g = pds_proxy_fast(preds_g, truths, gene_names)
    gated = sum(1 for t in held_targets if np.all(preds_g[t] == 0))
    print(f'{label} only: pds={pds_g:.4f} gated={gated}/{len(held_targets)}', flush=True)

# --- JS only ---
t0 = time.time()
p_js = LFCTransferPrior(james_stein=True).fit(train_lfc, train_basal, gene_names)
p_js.shared_mix = 0.0; p_js.use_alpha = False
preds_js = {t: p_js.predict_lfc(t, held_basal) for t in held_targets}
pds_js = pds_proxy_fast(preds_js, truths, gene_names)
print(f'JS only: pds={pds_js:.4f} ({time.time()-t0:.0f}s)', flush=True)

# --- v11 (JS + gate 5.0) ---
t0 = time.time()
p11 = LFCTransferPrior(basal_gate_cpm=5.0, james_stein=True).fit(train_lfc, train_basal, gene_names)
p11.shared_mix = 0.0; p11.use_alpha = False
preds_v11 = {t: p11.predict_lfc(t, held_basal) for t in held_targets}
pds_v11 = pds_proxy_fast(preds_v11, truths, gene_names)
gated = sum(1 for t in held_targets if np.all(preds_v11[t] == 0))
print(f'JS+gate5 (v11): pds={pds_v11:.4f} gated={gated} ({time.time()-t0:.0f}s)', flush=True)

print(f'\nTotal: {time.time()-t_start:.0f}s', flush=True)
print(f'SUMMARY:', flush=True)
print(f'  v6 baseline:   {pds_v6:.4f}', flush=True)
print(f'  gate-only:     see above', flush=True)
print(f'  JS-only:       {pds_js:.4f}', flush=True)
print(f'  v11 (JS+gate): {pds_v11:.4f}', flush=True)
