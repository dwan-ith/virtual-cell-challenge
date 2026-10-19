"""Train v35 residual ensemble (100M+ class) on ALL contexts. Saves to artifacts/ensemble_v35/.

Usage: python scripts/train_v35_ensemble.py [--members 3] [--epochs 8]
"""
import sys; sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--members", type=int, default=3)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--max-pairs", type=int, default=8000)
parser.add_argument("--panel", type=int, default=128)
parser.add_argument("--out", type=str, default="artifacts/ensemble_v35")
args = parser.parse_args()

from vcc_model.prior import LFCTransferPrior
from vcc_model.dynamics_residual import ResidualDynamicsConfig, ResidualCellDynamics, train_residual_dynamics

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
print(f"[train] {len(pairs)} pairs {len(basals)} ctx", flush=True)

esm = None
for cand in [Path("data/protein/se_full_18533.npz")]:
    if cand.exists():
        d = np.load(cand, allow_pickle=True)
        gn = [str(x) for x in d['gene_names']]; mat = d['embeddings']
        n = np.linalg.norm(mat, axis=1, keepdims=True); n = np.where(n < 1e-9, 1.0, n); mat = mat / n
        esm = {g: mat[i].astype(np.float32) for i, g in enumerate(gn)}
        print(f"[esm] {len(esm)}", flush=True)
        break

nested = {}
for (c, g), v in pairs.items():
    nested.setdefault(c, {})[g] = v
prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0, use_alpha=False, zero_unseen=True).fit(nested, basals, genes)
print(f"[prior] {len(prior.contexts)} ctx alpha {prior.global_alpha:.3f}", flush=True)

# 100M+ config
def make_cfg(seed: int) -> ResidualDynamicsConfig:
    return ResidualDynamicsConfig(
        latent_dim=256, in_proj_dim=2048,
        enc_h1=2048, enc_h2=1024,
        trans_h1=2048, trans_h2=1024,
        dec_hidden=2048,
        epochs=args.epochs, steps_per_epoch=args.steps, panel_size=args.panel,
        max_pairs=args.max_pairs, w_rank=0.5, w_cos=1.0, w_l1=0.3,
        seed=seed,
    )

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
import json
(out / "meta.json").write_text(json.dumps({"n_members": args.members, "n_genes": len(genes)}))
# NOTE: prior is NOT pickled (2.9GB with full LFC dicts); build script refits
# it from stores in seconds. Keeps disk free for submission artifacts.
import gc
for i in range(args.members):
    if (out / f"member_{i}.pt").exists():
        print(f"[skip] member {i} exists (resume)", flush=True)
        continue
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cfg = make_cfg(seed=1000 + i)
    m, info = train_residual_dynamics(prior, pairs, basals, genes, esm_embeddings=esm, cfg=cfg)
    m.save(str(out / f"member_{i}.pt"))
    print(f"[save] member {i} {info} -> {out/f'member_{i}.pt'}", flush=True)
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"[done] ensemble -> {out}", flush=True)
