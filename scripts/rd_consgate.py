"""RPE1 LOCO: per-gene PDS rank vs prior consistency; sweep hard consistency gate.
Hypothesis: low-consistency genes rank < 0.5 (worse than median); zeroing them
lifts PDS. Zero-predictions score exactly 0.5 in pds_proxy_fast.
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

holdout = "replogle_gw/RPE1"
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
print(f"consistency stats: n={len(prior.consistency)}", flush=True)

tgt = basals[holdout]
truths = {g: v for (_, g), v in test2.items()}
preds = {}
for g in truths:
    raw = prior.predict_lfc(g, tgt)
    if raw is not None:
        preds[g] = raw

# Per-gene rank: replicate pds_proxy_fast internals per gene
from vcc_model.ensemble import _panel_effect_matrix, PANEL_IDX
import vcc_model.ensemble as E
if not E.PANEL_IDX or len(E.PANEL_IDX) != len(genes):
    E.PANEL_IDX = {g: i for i, g in enumerate(genes)}
names = sorted(set(preds) & set(truths))
real_mat = _panel_effect_matrix(truths, names)
pred_mat = _panel_effect_matrix(preds, names)
nr = np.linalg.norm(real_mat, axis=1, keepdims=True); nr[nr == 0] = 1.0
real_unit = real_mat / nr
ph = np.linalg.norm(pred_mat, axis=1, keepdims=True)
zero_pred = (ph[:, 0] == 0); ph[ph == 0] = 1.0
sims = pred_mat / ph @ real_unit.T
dists = 1.0 - sims
n = len(names)
ranks = {}
for i, g in enumerate(names):
    d_self = dists[i, i]
    k = (dists[i] < d_self).sum() + 0.5 * ((dists[i] == d_self).sum() - 1)
    ranks[g] = 1.0 - k / (n - 1)

def _cons_scalar(g: str) -> float:
    c = prior.consistency.get(g)
    if c is None:
        return 0.0
    return float(np.mean(c))

cons = np.array([_cons_scalar(g) for g in names])
rk = np.array([ranks[g] for g in names])
print(f"rank-consistency corr: {np.corrcoef(cons, rk)[0,1]:+.3f}", flush=True)
for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
    m = (cons >= lo) & (cons < hi)
    if m.sum():
        print(f"  cons [{lo:.1f},{hi:.1f}): n={m.sum()} mean_rank={rk[m].mean():.3f}", flush=True)

p_base = pds_proxy_fast(preds, truths, genes)
print(f"BASELINE pds={p_base:.3f}", flush=True)

# Sweep hard gate: zero predictions with consistency < threshold
for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    gated = {}
    nz = 0
    for g in preds:
        if _cons_scalar(g) < th:
            gated[g] = np.zeros_like(preds[g]); nz += 1
        else:
            gated[g] = preds[g]
    p = pds_proxy_fast(gated, truths, genes)
    print(f"RESULT cons_gate={th:.1f} zeroed={nz} pds={p:.3f}({p-p_base:+.3f})", flush=True)
print("Done", flush=True)
