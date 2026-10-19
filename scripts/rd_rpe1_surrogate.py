"""LOCO validation: surrogate as confidence gate — zero out bad predictions.
Train surrogate to predict gain; at test time, gate = 0 if gain < threshold,
else pass through. This CAN move PDS by demoting bad predictions to median.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vcc_model.ensemble import pds_proxy_fast
from vcc_model.prior import LFCTransferPrior
from vcc_model.surrogate import SurrogateCorrector
from vcc_model.protein import load_esm_embeddings

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
print(f"train {len(train)} test {len(test2)}", flush=True)

prior = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False)
prior.fit(nested, basals, genes)
esm = load_esm_embeddings("data/protein/se_full_18533.npz")
if esm:
    esm = {g: v for g, v in esm.items() if g in set(genes)}

tgt = basals[holdout]
truths = {g: v for (_, g), v in test2.items()}

# Baseline
preds_base = {}
for g in truths:
    raw = prior.predict_lfc(g, tgt)
    if raw is not None:
        preds_base[g] = raw
p_base = pds_proxy_fast(preds_base, truths, genes)
cos_base = float(np.mean([float(np.dot(preds_base[g], truths[g]) /
              (np.linalg.norm(preds_base[g]) * np.linalg.norm(truths[g]) + 1e-9))
              for g in truths]))
print(f"BASELINE: pds={p_base:.3f} cos={cos_base:.4f}", flush=True)

# Train surrogate
sc = SurrogateCorrector(esm_proj_dim=32, hidden=64, lr=1e-3, epochs=100, batch_size=256)
sc.fit(train, basals, genes, prior, esm)

# Test: confidence gating — precompute all predictions and gains once
print("\n--- Confidence gating (zero out low-confidence) ---")
all_test_genes = sorted(truths.keys())
all_raw = {}
all_gains = {}
import time as _time
_t0 = _time.time()
for i, g in enumerate(all_test_genes):
    raw = prior.predict_lfc(g, tgt)
    if raw is not None:
        all_raw[g] = raw
        all_gains[g] = sc.predict_gain(g, tgt, raw, esm.get(g) if esm else None)
    if (i+1) % 50 == 0:
        print(f"  precomputed {i+1}/{len(all_test_genes)} in {_time.time()-_t0:.0f}s", flush=True)
print(f"precomputed {len(all_raw)} predictions in {_time.time()-_t0:.0f}s", flush=True)

for threshold in [0.0, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]:
    preds = {}
    n_kept = 0
    for g in all_raw:
        gain = all_gains[g]
        if gain >= threshold:
            preds[g] = all_raw[g]
            n_kept += 1
        else:
            preds[g] = np.zeros_like(all_raw[g])
    p = pds_proxy_fast(preds, truths, genes)
    cos_parts = []
    for g in preds:
        pn = np.linalg.norm(preds[g])
        tn = np.linalg.norm(truths[g])
        if pn > 1e-9 and tn > 1e-9:
            cos_parts.append(float(np.dot(preds[g], truths[g]) / (pn * tn)))
    cos = float(np.mean(cos_parts)) if cos_parts else 0.0
    print(f"thresh={threshold:.1f} kept={n_kept}/{len(all_raw)} pds={p:.3f}({p-p_base:+.3f}) cos={cos:.4f}", flush=True)

# Also test: soft gate using precomputed data
print("\n--- Soft gating: blend toward zero based on gain ---")
for beta in [0.0, 0.3, 0.5, 0.7, 1.0]:
    preds = {}
    for g in all_raw:
        gain = all_gains[g]
        gate = np.clip((gain - 0.5) / 1.0, 0.0, 1.0)
        gate = gate ** (1.0 + beta * 2)
        preds[g] = all_raw[g] * gate
    p = pds_proxy_fast(preds, truths, genes)
    cos_parts = []
    for g in preds:
        pn = np.linalg.norm(preds[g])
        tn = np.linalg.norm(truths[g])
        if pn > 1e-9 and tn > 1e-9:
            cos_parts.append(float(np.dot(preds[g], truths[g]) / (pn * tn)))
    cos = float(np.mean(cos_parts)) if cos_parts else 0.0
    print(f"beta={beta:.1f} pds={p:.3f}({p-p_base:+.3f}) cos={cos:.4f}", flush=True)

print("Done", flush=True)
