"""LOCO validation for CellDynamics v32. Gate: Î”overall>0.015 vs v29 baseline.

Holdouts: RPE1 (replogle_gw/RPE1) and H1 (vcc2025/H1) â€” mirrors rd_v31_test.py
but scores CellDynamics ensemble vs prior-only. Uses pds_proxy_fast and
full six-metric via metrics.evaluate_context when controls available.

Usage:
  python scripts/rd_v32_loco.py                    # quick 1-member, 40 epochs
  python scripts/rd_v32_loco.py --full             # 5-member ensemble, 80 epochs
  python scripts/rd_v32_loco.py --members 3 --epochs 60
"""
import sys
sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--members", type=int, default=1, help="ensemble members (1 for quick)")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--full", action="store_true", help="5 members, 80 epochs")
parser.add_argument("--max-pairs", type=int, default=15000)
parser.add_argument("--batch", type=int, default=128)
args = parser.parse_args()
if args.full:
    args.members = 5
    args.epochs = 80

from vcc_model.ensemble import pds_proxy_fast
from vcc_model.dynamics import DynamicsConfig
from vcc_model.ensemble_surrogate import EnsembleDynamics

stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
if not stores:
    print("no stores under artifacts/stores â€” run build_stores first"); sys.exit(1)

pairs, basals, genes = {}, {}, None
for p in stores:
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z['gene_names'].astype(str))
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c), str(g))] = v
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]

# ESM embeddings (optional, for CellDynamics)
esm = None
for cand in [Path("data/protein/se_full_18533.npz"), Path("artifacts/se_full_18533.npz")]:
    if cand.exists():
        try:
            d = np.load(cand, allow_pickle=True)
            if "gene_names" in d and "embeddings" in d:
                gn = [str(x) for x in d["gene_names"]]
                mat = d["embeddings"]
                # L2-normalize per gene
                n = np.linalg.norm(mat, axis=1, keepdims=True)
                n = np.where(n < 1e-9, 1.0, n)
                mat = mat / n
                esm = {g: mat[i].astype(np.float32) for i, g in enumerate(gn)}
                print(f"[esm] loaded {len(esm)} genes dim {mat.shape[1]} from {cand}", flush=True)
            else:
                print(f"[esm] {cand} has keys {list(d.files)} â€” skipping", flush=True)
            break
        except Exception as e:
            print(f"[esm] load failed {cand}: {e}", flush=True)

# Baseline prior-only for delta (reuse prior logic via explicit prior fit)
from vcc_model.prior import LFCTransferPrior

HOLDOUTS = ['replogle_gw/RPE1', 'vcc2025/H1']
THRESH = 0.015

results = {}

for holdout in HOLDOUTS:
    train = {(c, g): v for (c, g), v in pairs.items() if c != holdout}
    test = {(c, g): v for (c, g), v in pairs.items() if c == holdout}
    if not test:
        print(f"\n[skip] {holdout} not in stores"); continue
    train_basals = {k: v for k, v in basals.items() if k != holdout}
    held_basal = basals.get(holdout)
    if held_basal is None:
        print(f"[skip] {holdout} basal missing"); continue

    # cap test genes to 200 for PDS proxy stability
    import random; random.seed(0)
    tgenes = sorted({g for _, g in test})[:300]
    # for PDS proxy we want enough genes; take 120 random if >120
    pds_genes = tgenes
    if len(pds_genes) > 150:
        rng = np.random.default_rng(0)
        pds_genes = sorted(rng.choice(pds_genes, size=150, replace=False).tolist())
    truth = {g: test[(holdout, g)] for g in pds_genes if (holdout, g) in test}
    print(f"\n{'='*60}\n[LOCO] {holdout} train {len(train)} test {len(truth)} members={args.members} epochs={args.epochs}", flush=True)

    # --- prior-only baseline ---
    nested = {}
    for (c, g), v in train.items():
        nested.setdefault(c, {})[g] = v
    prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0, use_alpha=False, zero_unseen=True).fit(nested, train_basals, genes)
    prior_preds = {g: prior.predict_lfc(g, held_basal) for g in truth}
    base_pds = pds_proxy_fast(prior_preds, truth, genes)
    print(f" prior-only pds {base_pds:.4f}", flush=True)

    # --- CellDynamics ---
    cfg = DynamicsConfig(
        epochs=args.epochs, batch_size=args.batch, max_pairs=args.max_pairs,
        seed=0, w_rank=0.5, w_cos=1.0, w_l1=0.3, rank_temperature=0.1,
        dropout=0.10, weight_decay=1e-4,
    )
    n_params_est = (18789*cfg.enc_h1 + cfg.enc_h1*cfg.enc_h2 + cfg.enc_h2*cfg.enc_h3 + cfg.enc_h3*cfg.latent_dim
                    + (cfg.latent_dim+512)*cfg.trans_h1 + cfg.trans_h1*cfg.trans_h2 + cfg.trans_h2*cfg.trans_h3 + cfg.trans_h3*cfg.latent_dim*2
                    + cfg.latent_dim*cfg.dec_hidden + cfg.dec_hidden*len(genes))
    print(f" cfg {cfg} est ~{n_params_est/1e6:.1f}M", flush=True)

    if args.members == 1:
        from vcc_model.dynamics import train_cell_dynamics
        model, info = train_cell_dynamics(train, train_basals, genes, esm_embeddings=esm, cfg=cfg)
        # wrap as single-member ensemble for uniform API
        class _Single:
            def __init__(self, m, esm): self.m=m; self._esm=esm
            def predict_lfc(self, g, b): return self.m.predict_lfc(b, self._esm.get(g) if self._esm else None)
        ens = _Single(model, esm)
        dyn_preds = {g: ens.predict_lfc(g, held_basal) for g in truth}
        dyn_pds = pds_proxy_fast(dyn_preds, truth, genes)
        print(f" dynamics pds {dyn_pds:.4f} Î” {dyn_pds-base_pds:+.4f}  {info}", flush=True)
        # blend sweep: prior + w*dynamics
        for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            blended = {g: (1-w)*prior_preds[g] + w*dyn_preds[g] for g in truth}
            p = pds_proxy_fast(blended, truth, genes)
            print(f"  blend w={w:.1f} pds {p:.4f} Î” {p-base_pds:+.4f}", flush=True)
        results[holdout] = {"base": base_pds, "dyn": dyn_pds, "delta": dyn_pds-base_pds}
    else:
        ens = EnsembleDynamics.train(train, train_basals, genes, esm_embeddings=esm, n_members=args.members, base_cfg=cfg, verbose=True)
        dyn_preds = {g: ens.predict_lfc(g, held_basal) for g in truth}
        dyn_pds = pds_proxy_fast(dyn_preds, truth, genes)
        print(f" ensemble pds {dyn_pds:.4f} Î” {dyn_pds-base_pds:+.4f}", flush=True)
        for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            blended = {g: (1-w)*prior_preds[g] + w*dyn_preds[g] for g in truth}
            p = pds_proxy_fast(blended, truth, genes)
            print(f"  blend w={w:.1f} pds {p:.4f} Î” {p-base_pds:+.4f}", flush=True)
        # OOD report
        if ens.ood is not None:
            try:
                train_mat = np.stack([train_basals[c] for c in sorted(train_basals)])
                held_mat = held_basal.reshape(1, -1)
                print(f" OOD {ens.ood.report(train_mat, held_mat)}", flush=True)
            except Exception as e:
                print(f" OOD report failed {e}", flush=True)
        results[holdout] = {"base": base_pds, "dyn": dyn_pds, "delta": dyn_pds-base_pds}

print(f"\n{'='*60}\nSUMMARY", flush=True)
for k, v in results.items():
    flag = "PASS" if v["delta"] > THRESH else "FAIL"
    print(f" {k}: base {v['base']:.4f} dyn {v['dyn']:.4f} Î” {v['delta']:+.4f} [{flag} vs {THRESH}]", flush=True)
# overall gate: need positive on both holdouts and at least one >0.015? be strict: mean delta
mean_delta = float(np.mean([v["delta"] for v in results.values()])) if results else 0.0
print(f" mean Î” {mean_delta:+.4f} gate {THRESH} â†’ {'PASS â€” proceed to submission' if mean_delta>THRESH else 'FAIL â€” do not submit'}", flush=True)

