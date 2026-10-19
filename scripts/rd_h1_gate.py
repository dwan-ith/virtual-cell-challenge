"""H1 LOCO: hard-context test of quality gating.
Signals: max transfer weight (confident match?) and weight entropy (diffuse?).
Sweep gates on both; measure proxy-PDS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vcc_model.ensemble import pds_proxy_fast
from vcc_model.prior import LFCTransferPrior

stores = [p for p in Path("artifacts/stores").glob("*.npz") if "__controls" not in p.name]
pairs, basals, genes = {}, {}, None
for p in stores:
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z["gene_names"].astype(str))
    for c, g, v in zip(z["context"], z["target"], z["lfc"]):
        pairs[(str(c), str(g))] = v
    for i, c in enumerate(z["basal_ctx"]):
        basals[str(c)] = z["basal"][i]

holdout = "vcc2025/H1"
train = {k: v for k, v in pairs.items() if k[0] != holdout}
test = {k: v for k, v in pairs.items() if k[0] == holdout}
import random; random.seed(0)
tgenes = sorted({g for _, g in test})[:200]
test2 = {k: v for k, v in test.items() if k[1] in tgenes}

nested = {}
for (c, g), v in train.items():
    nested.setdefault(c, {})[g] = v

prior = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False)
prior.fit(nested, basals, genes)

tgt = basals[holdout]
truths = {g: v for (_, g), v in test2.items()}

# Per-gene: prediction + w_map diagnostics
w_map, alpha = prior.transfer_params(tgt)
print(f"w_map: { {k: round(float(v),3) for k,v in w_map.items()} }", flush=True)
ws = np.array(sorted(w_map.values(), reverse=True))
print(f"max_w={ws[0]:.3f} entropy={float(-(ws/ws.sum() * np.log(ws/ws.sum()+1e-12)).sum()):.3f}", flush=True)

preds = {}
for g in truths:
    raw = prior.predict_lfc(g, tgt)
    if raw is not None:
        preds[g] = raw
p_base = pds_proxy_fast(preds, truths, genes)
print(f"BASELINE H1 pds={p_base:.3f}", flush=True)

# Per-gene rank
from vcc_model.ensemble import _panel_effect_matrix
import vcc_model.ensemble as E
if not E.PANEL_IDX or len(E.PANEL_IDX) != len(genes):
    E.PANEL_IDX = {g: i for i, g in enumerate(genes)}
names = sorted(set(preds) & set(truths))
real_mat = _panel_effect_matrix(truths, names)
pred_mat = _panel_effect_matrix(preds, names)
nr = np.linalg.norm(real_mat, axis=1, keepdims=True); nr[nr == 0] = 1.0
real_unit = real_mat / nr
ph = np.linalg.norm(pred_mat, axis=1, keepdims=True)
ph[ph == 0] = 1.0
sims = pred_mat / ph @ real_unit.T
dists = 1.0 - sims
n = len(names)
rk = {}
for i, g in enumerate(names):
    d_self = dists[i, i]
    k = (dists[i] < d_self).sum() + 0.5 * ((dists[i] == d_self).sum() - 1)
    rk[g] = 1.0 - k / (n - 1)
rkv = np.array([rk[g] for g in names])
print(f"rank<0.5 fraction: {float((rkv<0.5).mean()):.2f}  mean rank of bottom decile: {np.sort(rkv)[:20].mean():.3f}", flush=True)

# Gate on max weight is context-global (same for all genes) — instead gate on
# per-gene n_obs (contexts where gene was seen)
from collections import Counter
nobs = {g: sum(1 for c in nested if g in nested[c]) for g in names}
nov = np.array([nobs[g] for g in names])
print(f"nobs-rank corr: {np.corrcoef(nov, rkv)[0,1]:+.3f}", flush=True)
for lo, hi in [(1, 2), (2, 4), (4, 8)]:
    m = (nov >= lo) & (nov < hi)
    if m.sum():
        print(f"  nobs [{lo},{hi}): n={m.sum()} mean_rank={rkv[m].mean():.3f}", flush=True)

for th in [1, 2, 3]:
    gated = {g: (np.zeros_like(preds[g]) if nobs[g] < th else preds[g]) for g in preds}
    p = pds_proxy_fast(gated, truths, genes)
    nz = sum(1 for g in preds if nobs[g] < th)
    print(f"RESULT nobs_gate={th} zeroed={nz} pds={p:.3f}({p-p_base:+.3f})", flush=True)
print("Done", flush=True)
