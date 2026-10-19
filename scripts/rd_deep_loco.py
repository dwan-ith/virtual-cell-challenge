"""Deep LOCO grid sweep: prior-level params for RPE1 and H1 holdouts.

Sweeps: scale, esm_boost, shared_mix, zero_unseen, q_threshold
Fixed: gain=1.8, power=1.0, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True

Heavily optimized: all per-gene computation precomputed once per holdout;
gain maps vectorized; calibration vectorized; PDS proxy cached.
"""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import itertools
import time
import numpy as np
from collections import defaultdict

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Load data ───────────────────────────────────────────────────────────────
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

P(f"Loaded {len(pairs)} pairs, {len(basals)} contexts, {len(genes)} genes")
P(f"Contexts: {sorted(basals.keys())}")

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.ensemble import pds_proxy_fast
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map

esm = load_esm_embeddings()
P(f"ESM: {len(esm) if esm else 0} genes")

# ── Holdouts ────────────────────────────────────────────────────────────────
holdouts = {'RPE1': 'replogle_gw/RPE1', 'H1': 'vcc2025/H1'}

# ── Parameter grid ──────────────────────────────────────────────────────────
scales       = [0.3, 0.5, 0.7, 0.9, 1.1]
esm_boosts   = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
shared_mixes = [0.0, 0.1, 0.25]
zero_unseens = [True, False]
q_thresholds = [0.0, 0.2, 0.3, 0.5]

BASE_GAIN  = 1.8
BASE_POWER = 1.0
N_TEST     = 200
SEED       = 0

configs = list(itertools.product(scales, esm_boosts, shared_mixes,
                                zero_unseens, q_thresholds))
P(f"Grid: {len(scales)}x{len(esm_boosts)}x{len(shared_mixes)}"
  f"x{len(zero_unseens)}x{len(q_thresholds)} = {len(configs)} configs")

# ── Vectorized PDS proxy (cached matrix ops) ────────────────────────────────
Gene_idx = {}

def _pds_precomputed(pred_mat, truth_mat, n_panel):
    """PDS proxy from pre-built (n, G-n_panel) float64 matrices."""
    n = pred_mat.shape[0]
    if n < 2:
        return float("nan")
    nr = np.linalg.norm(truth_mat, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    real_unit = truth_mat / nr
    ph = np.linalg.norm(pred_mat, axis=1, keepdims=True)
    zero_pred = (ph[:, 0] == 0)
    ph[ph == 0] = 1.0
    sims = pred_mat / ph @ real_unit.T
    dists = 1.0 - sims
    scores = np.full(n, 0.5)
    d_self = dists[np.arange(n), np.arange(n)]
    k = (dists < d_self[:, None]).sum(axis=1) + 0.5 * ((dists == d_self[:, None]).sum(axis=1) - 1)
    ok = ~zero_pred
    scores[ok] = 1.0 - k[ok] / (n - 1)
    return float(scores.mean())


# ── Sweep ───────────────────────────────────────────────────────────────────
results = []

for holdout_name, holdout_ctx in holdouts.items():
    P(f"\n{'='*60}")
    P(f"  {holdout_name} ({holdout_ctx})")
    P(f"{'='*60}")

    train_pairs = {k: v for k, v in pairs.items() if k[0] != holdout_ctx}
    test_pairs_all = {k: v for k, v in pairs.items() if k[0] == holdout_ctx}
    if not test_pairs_all:
        P(f"  No test pairs for {holdout_ctx}, skipping")
        continue

    rng = np.random.default_rng(SEED)
    all_test_genes = sorted({g for _, g in test_pairs_all})
    n_take = min(N_TEST, len(all_test_genes))
    if n_take < len(all_test_genes):
        test_gene_set = set(rng.choice(all_test_genes, size=n_take, replace=False).tolist())
    else:
        test_gene_set = set(all_test_genes)
    test_genes = sorted(test_gene_set)
    P(f"  Train: {len(train_pairs)} pairs | Test: {len(test_genes)} genes "
      f"(from {len(all_test_genes)} available)")

    nested = {}
    for (c, g), v in train_pairs.items():
        nested.setdefault(c, {})[g] = v
    train_basals = {k: v for k, v in basals.items() if k != holdout_ctx}
    target_basal = basals.get(holdout_ctx)
    if target_basal is None:
        target_basal = np.mean(list(train_basals.values()), axis=0)

    # Fit prior ONCE
    t0 = time.time()
    prior = LFCTransferPrior(
        shared_mix=0.0, use_alpha=False, basal_gate_cpm=5.0,
        similarity_temperature=0.1, gene_conditioned=True,
        esm_embeddings=esm, zero_unseen=False,
    ).fit(nested, train_basals, genes)
    P(f"  Prior fit: {time.time()-t0:.1f}s ({len(prior.contexts)} contexts, "
      f"alpha={prior.global_alpha:.3f})")

    # ── Precompute per-gene predictions ONCE ────────────────────────────
    t1 = time.time()
    w_map, alpha = prior.transfer_params(target_basal)

    gate_mask = None
    if prior.basal_gate_cpm > 0.0:
        mean_cpm = np.exp(target_basal) - 1e-9
        gate_mask = mean_cpm < prior.basal_gate_cpm

    shared_vec = prior._shared_program_from(w_map).astype(np.float64)
    if gate_mask is not None:
        shared_vec[gate_mask] = 0.0

    unseen_set = {g for g in test_genes if prior._is_unseen(g)}
    P(f"  Unseen genes: {len(unseen_set)}/{len(test_genes)}")

    # Compute specific for each test gene
    gene_specific = {}
    for g in test_genes:
        if g in unseen_set:
            continue
        stack, weights = [], []
        for c, w in w_map.items():
            if g in prior.lfc.get(c, {}):
                stack.append(prior.lfc[c][g])
                weights.append(w)
        if stack:
            w_arr = np.asarray(weights, dtype=np.float64)
            w_arr = w_arr / max(w_arr.sum(), 1e-12)
            spec = np.average(np.stack(stack), axis=0, weights=w_arr)
        else:
            spec = prior._neighbor_fallback(g, target_basal, shared_hint=shared_vec)
        cons = prior.consistency.get(g)
        if cons is not None:
            spec = spec * (0.25 + 0.75 * cons)
        if gate_mask is not None:
            spec = spec.copy()
            spec[gate_mask] = 0.0
        gene_specific[g] = spec

    P(f"  Precompute: {time.time()-t1:.1f}s")

    # Truths + eval genes
    truths = {g: pairs[(holdout_ctx, g)] for g in test_genes if (holdout_ctx, g) in pairs}
    eval_genes = sorted(set(test_genes) & set(truths.keys()))
    n_genes = len(genes)
    P(f"  Eval genes: {len(eval_genes)}")

    # ── Build eval matrices for vectorized PDS ──────────────────────────
    t_pds = time.time()
    G = len(genes)
    gene_idx_map = {g: i for i, g in enumerate(genes)}
    panel_targets = set()
    for (c, g) in pairs:
        panel_targets.add(g)
    panel_mask = np.ones(G, dtype=bool)
    for g in panel_targets:
        gi = gene_idx_map.get(g)
        if gi is not None:
            panel_mask[gi] = False
    keep_cols = np.flatnonzero(panel_mask)
    n_panel = G - len(keep_cols)

    truth_mat = np.stack([np.asarray(truths[g], dtype=np.float64)[keep_cols] for g in eval_genes])
    truth_mat_full = np.stack([np.asarray(truths[g], dtype=np.float64) for g in eval_genes])
    P(f"  PDS precompute: {time.time()-t_pds:.1f}s (panel={n_panel}, eval={len(eval_genes)})")

    # ── Precompute gain maps (vectorized for eval genes only) ───────────
    t2 = time.time()
    eval_gene_set = set(eval_genes)
    gain_maps = {}
    for scale, esm_boost in itertools.product(scales, esm_boosts):
        gm = build_per_gene_gain_map(
            prior.consistency, genes, scale=scale,
            esm_embeddings=esm, esm_boost=esm_boost,
        )
        # Extract only eval gene gains as array
        gain_arr = np.array([gm.get(g, gm.get('_default', 1.0)) for g in eval_genes])
        gain_maps[(scale, esm_boost)] = gain_arr
    P(f"  Gain maps: {time.time()-t2:.1f}s ({len(gain_maps)} maps)")

    # ── Build eval-specific arrays ──────────────────────────────────────
    # specific matrix: (n_eval, G)
    specific_mat = np.stack([gene_specific.get(g, shared_vec) for g in eval_genes])
    unseen_mask = np.array([g in unseen_set for g in eval_genes])

    # ── Sweep configs ───────────────────────────────────────────────────
    t_h = time.time()

    for ci, (scale, esm_boost, sm, zu, qt) in enumerate(configs):
        gain_arr = gain_maps[(scale, esm_boost)]

        # Vectorized blend: (n_eval, G)
        pred_mat_full = specific_mat * (1.0 - sm) + shared_vec[np.newaxis, :] * sm
        if zu:
            pred_mat_full[unseen_mask] = 0.0

        # Vectorized calibration
        mag = np.abs(pred_mat_full)
        if qt > 0.0:
            tau = np.quantile(mag, qt, axis=1, keepdims=True)
            tau = np.maximum(tau, 0.0)
        else:
            tau = 0.0
        shifted = np.clip(mag - tau, 0.0, None)
        # gain_arr shape: (n_eval,) -> (n_eval, 1) for broadcasting
        eff_gain = BASE_GAIN * gain_arr[:, np.newaxis]
        mag2 = shifted ** BASE_POWER * eff_gain
        pred_cal = np.sign(pred_mat_full) * mag2

        # PDS proxy using precomputed matrices
        pred_keep = pred_cal[:, keep_cols].astype(np.float64)
        pds = _pds_precomputed(pred_keep, truth_mat, n_panel)

        # NMAE (vectorized)
        nmae = float(np.mean(np.abs(pred_cal - truth_mat_full)))

        results.append({
            'holdout': holdout_name,
            'scale': scale, 'esm_boost': esm_boost,
            'shared_mix': sm, 'zero_unseen': zu,
            'q_threshold': qt,
            'pds': pds, 'nmae': nmae, 'n_genes': len(eval_genes),
        })

        if (ci + 1) % 100 == 0:
            elapsed = time.time() - t_h
            rate = (ci + 1) / max(elapsed, 0.001)
            eta = (len(configs) - ci - 1) / max(rate, 0.001)
            P(f"  [{ci+1}/{len(configs)}] {elapsed:.0f}s elapsed, "
              f"~{eta:.0f}s remaining | PDS={pds:.3f} nmae={nmae:.3f}")

    P(f"  Done: {time.time()-t_h:.0f}s for {len(configs)} configs")

# ── Results ─────────────────────────────────────────────────────────────────
P(f"\n{'='*80}")
P(f"  RESULTS ({len(results)} evaluations)")
P(f"{'='*80}")

config_agg = defaultdict(lambda: {'pds': {}, 'nmae': {}})
for r in results:
    key = (r['scale'], r['esm_boost'], r['shared_mix'],
           r['zero_unseen'], r['q_threshold'])
    config_agg[key]['pds'][r['holdout']] = r['pds']
    config_agg[key]['nmae'][r['holdout']] = r['nmae']

combos = []
for key, vals in config_agg.items():
    pds_vals = list(vals['pds'].values())
    nmae_vals = list(vals['nmae'].values())
    combos.append({
        'scale': key[0], 'esm_boost': key[1], 'shared_mix': key[2],
        'zero_unseen': key[3], 'q_threshold': key[4],
        'mean_pds': np.mean(pds_vals),
        'pds_rpe1': vals['pds'].get('RPE1', float('nan')),
        'pds_h1': vals['pds'].get('H1', float('nan')),
        'mean_nmae': np.mean(nmae_vals),
        'nmae_rpe1': vals['nmae'].get('RPE1', float('nan')),
        'nmae_h1': vals['nmae'].get('H1', float('nan')),
    })

HDR = (f"{'scale':>6} {'esm_b':>6} {'mix':>5} {'zero':>5} {'qt':>5} | "
       f"{'PDS':>6} {'RPE1':>6} {'H1':>6} | "
       f"{'NMAE':>6} {'RPE1':>6} {'H1':>6}")
SEP = '-' * 80

def _row(c):
    def _f(v):
        return f"{v:6.3f}" if np.isfinite(v) else "    nan"
    return (f"{c['scale']:6.1f} {c['esm_boost']:6.2f} {c['shared_mix']:5.2f} "
            f"{str(c['zero_unseen']):>5} {c['q_threshold']:5.1f} | "
            f"{_f(c['mean_pds'])} {_f(c['pds_rpe1'])} {_f(c['pds_h1'])} | "
            f"{_f(c['mean_nmae'])} {_f(c['nmae_rpe1'])} {_f(c['nmae_h1'])}")

combos.sort(key=lambda x: -x['mean_pds'])
P(f"\nTOP 10 BY MEAN PDS:")
P(HDR); P(SEP)
for c in combos[:10]:
    P(_row(c))

combos.sort(key=lambda x: x['mean_nmae'])
P(f"\nTOP 10 BY MEAN NMAE (lower is better):")
P(HDR); P(SEP)
for c in combos[:10]:
    P(_row(c))

rpe1 = sorted([r for r in results if r['holdout'] == 'RPE1'], key=lambda x: -x['pds'])
P(f"\nRPE1-ONLY TOP 10 BY PDS:")
P(f"{'scale':>6} {'esm_b':>6} {'mix':>5} {'zero':>5} {'qt':>5} | "
  f"{'PDS':>6} {'NMAE':>6}")
P(SEP)
for r in rpe1[:10]:
    P(f"{r['scale']:6.1f} {r['esm_boost']:6.2f} {r['shared_mix']:5.2f} "
      f"{str(r['zero_unseen']):>5} {r['q_threshold']:5.1f} | "
      f"{r['pds']:6.3f} {r['nmae']:6.3f}")

h1 = sorted([r for r in results if r['holdout'] == 'H1'], key=lambda x: -x['pds'])
P(f"\nH1-ONLY TOP 10 BY PDS:")
P(f"{'scale':>6} {'esm_b':>6} {'mix':>5} {'zero':>5} {'qt':>5} | "
  f"{'PDS':>6} {'NMAE':>6}")
P(SEP)
for r in h1[:10]:
    P(f"{r['scale']:6.1f} {r['esm_boost']:6.2f} {r['shared_mix']:5.2f} "
      f"{str(r['zero_unseen']):>5} {r['q_threshold']:5.1f} | "
      f"{r['pds']:6.3f} {r['nmae']:6.3f}")

P(f"\nDone. Total configs: {len(configs)}, total evaluations: {len(results)}")
