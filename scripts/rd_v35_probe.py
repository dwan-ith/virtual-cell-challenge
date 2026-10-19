"""v35 probe: residual-from-prior + panel batches + count-effect loss.

STRICT GATE (both must hold, else do not scale up / do not submit):
  1. count-space PDS proxy (pds_proxy_count) improves over prior on BOTH holdouts
  2. LFC-space PDS proxy (pds_proxy_fast) does not collapse (dyn >= 0.55)

Usage: python scripts/rd_v35_probe.py [--epochs 5] [--max-pairs 3000]
"""
import sys; sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--max-pairs", type=int, default=3000)
parser.add_argument("--panel", type=int, default=128)
args = parser.parse_args()

from vcc_model.ensemble import pds_proxy_fast
from vcc_model.prior import LFCTransferPrior
from vcc_model.dynamics_residual import (
    ResidualDynamicsConfig, train_residual_dynamics, pds_proxy_count,
)

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
print(f"[probe] {len(pairs)} pairs {len(basals)} ctx", flush=True)

esm = None
for cand in [Path("data/protein/se_full_18533.npz")]:
    if cand.exists():
        d = np.load(cand, allow_pickle=True)
        gn = [str(x) for x in d['gene_names']]; mat = d['embeddings']
        n = np.linalg.norm(mat, axis=1, keepdims=True); n = np.where(n < 1e-9, 1.0, n); mat = mat / n
        esm = {g: mat[i].astype(np.float32) for i, g in enumerate(gn)}
        print(f"[esm] {len(esm)}", flush=True)
        break

results = {}
for holdout in ['replogle_gw/RPE1', 'vcc2025/H1']:
    train = {(c, g): v for (c, g), v in pairs.items() if c != holdout}
    test = {(c, g): v for (c, g), v in pairs.items() if c == holdout}
    train_basals = {k: v for k, v in basals.items() if k != holdout}
    held_basal = basals[holdout]
    tgenes = sorted({g for _, g in test})[:150]
    truth = {g: test[(holdout, g)] for g in tgenes if (holdout, g) in test}
    print(f"\n[LOCO] {holdout} train {len(train)} truth {len(truth)}", flush=True)

    nested = {}
    for (c, g), v in train.items():
        nested.setdefault(c, {})[g] = v
    prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0, use_alpha=False, zero_unseen=True).fit(nested, train_basals, genes)
    prior_preds = {g: prior.predict_lfc(g, held_basal) for g in truth}
    base_lfc = pds_proxy_fast(prior_preds, truth, genes)
    base_cnt = pds_proxy_count(prior_preds, truth, held_basal, genes)
    print(f" prior: lfc-proxy {base_lfc:.4f} count-proxy {base_cnt:.4f}", flush=True)

    cfg = ResidualDynamicsConfig(
        epochs=args.epochs, steps_per_epoch=50, panel_size=args.panel,
        max_pairs=args.max_pairs, w_rank=0.5, w_cos=1.0, w_l1=0.3,
    )
    model, info = train_residual_dynamics(prior, train, train_basals, genes, esm_embeddings=esm, cfg=cfg)
    print(f" model: {info}", flush=True)
    dyn_preds = {}
    for g in truth:
        pv = prior_preds[g]
        ev = esm.get(g) if esm else None
        dyn_preds[g] = model.predict_lfc(pv, held_basal, ev)
    dyn_lfc = pds_proxy_fast(dyn_preds, truth, genes)
    dyn_cnt = pds_proxy_count(dyn_preds, truth, held_basal, genes)
    print(f" dyn:   lfc-proxy {dyn_lfc:.4f} (d {dyn_lfc-base_lfc:+.4f}) count-proxy {dyn_cnt:.4f} (d {dyn_cnt-base_cnt:+.4f})", flush=True)
    results[holdout] = {"base_lfc": base_lfc, "dyn_lfc": dyn_lfc, "base_cnt": base_cnt, "dyn_cnt": dyn_cnt,
                        "d_lfc": dyn_lfc - base_lfc, "d_cnt": dyn_cnt - base_cnt}

print("\nSTRICT GATE:", flush=True)
ok = True
for k, v in results.items():
    c1 = v["d_cnt"] > 0.0
    c2 = v["dyn_lfc"] >= 0.55
    print(f" {k}: d_count {v['d_cnt']:+.4f} ({'PASS' if c1 else 'FAIL'}>0) dyn_lfc {v['dyn_lfc']:.4f} ({'PASS' if c2 else 'FAIL'}>=0.55)", flush=True)
    ok = ok and c1 and c2
print(f"OVERALL: {'PASS - scale up' if ok else 'FAIL - do not scale up, iterate'}", flush=True)
