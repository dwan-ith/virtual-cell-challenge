"""Build v35 submission: residual ensemble output DIRECTLY (no blend weight).

Model output = prior + correction, so no blending needed. Loads
artifacts/ensemble_v35/{prior.pkl, member_*.pt}.

Usage: python scripts/build_v35_submission.py --out artifacts/submission/pred_v35.h5ad
"""
import sys; sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="artifacts/submission/pred_v35.h5ad")
parser.add_argument("--ens", type=str, default="artifacts/ensemble_v35")
args = parser.parse_args()

import torch
from vcc_model.dataset import load_controls_bundle
from vcc_model.dynamics_residual import ResidualCellDynamics
from vcc_model.countsim import sparse_log_mean_cpm
from vcc_model.submission import StreamingSubmissionWriter, validate_h5ad_file_light

ens_dir = Path(args.ens)
import json
# Refit prior from stores (seconds; avoids 2.9GB pickle)
from vcc_model.prior import LFCTransferPrior
stores_p = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
_ppairs, _pbasals, _pgenes = {}, {}, None
for p in stores_p:
    z = np.load(p, allow_pickle=True)
    if _pgenes is None:
        _pgenes = list(z['gene_names'].astype(str))
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        _ppairs[(str(c), str(g))] = v
    for i, c in enumerate(z['basal_ctx']):
        _pbasals[str(c)] = z['basal'][i]
_nested = {}
for (c, g), v in _ppairs.items():
    _nested.setdefault(c, {})[g] = v
prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0, use_alpha=False, zero_unseen=True).fit(_nested, _pbasals, _pgenes)
del _ppairs, _pbasals, _nested
import gc as _gc; _gc.collect()
print(f"[prior] refit {len(prior.contexts)} ctx", flush=True)
members = []
i = 0
while (ens_dir / f"member_{i}.pt").exists():
    m = ResidualCellDynamics.load(str(ens_dir / f"member_{i}.pt"), map_location="cpu")
    members.append(m)
    i += 1
print(f"[v35] {len(members)} members, prior ctx {len(prior.contexts)}", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
for m in members:
    m.to(device).eval()
print(f"[v35] on {device}", flush=True)

# ESM
esm = None
for cand in [Path("data/protein/se_full_18533.npz")]:
    if cand.exists():
        d = np.load(cand, allow_pickle=True)
        gn = [str(x) for x in d['gene_names']]; mat = d['embeddings']
        n = np.linalg.norm(mat, axis=1, keepdims=True); n = np.where(n < 1e-9, 1.0, n); mat = mat / n
        esm = {g: mat[i].astype(np.float32) for i, g in enumerate(gn)}
        print(f"[esm] {len(esm)}", flush=True)
        break

genes = prior.gene_names
# Load contexts ONE at a time (all-at-once harmonize OOMs fragmented RAM)
import anndata as _ad
from vcc_model.dataset import harmonize_to_axis as _harm, detect_columns as _det
_ctrl_files = sorted(Path("data/vcc2026").glob("context_*.h5ad"))[:3]
def _load_one_ctx(fpath):
    a = _ad.read_h5ad(fpath)
    tcol, _ = _det(a.obs)
    x = a.X
    import numpy as _np
    if tcol is not None:
        labels = a.obs[tcol].astype(str)
        keep = (labels == "non-targeting").to_numpy()
        if keep.any() and not keep.all():
            x = x[keep]
    ctx = fpath.stem.removeprefix("context_").removeprefix("context-")
    m = _harm(x, a.var_names, genes).tocsr()
    del a, x
    return ctx, m
print(f"[ctrl] files {[f.name for f in _ctrl_files]}", flush=True)
panel = pd.read_csv("data/vcc2026/pert_counts.csv")
panel_genes = panel[panel.columns[0]].astype(str).tolist()[:300]
print(f"[panel] {len(panel_genes)}", flush=True)

# Strip prior.lfc to panel-only COPIES so the 1.5GB store buffers can free.
_keep = set(panel_genes)
for _c in list(prior.lfc.keys()):
    _d = prior.lfc[_c]
    prior.lfc[_c] = {g: np.array(v, dtype=np.float32) for g, v in _d.items() if g in _keep}
# also strip pooled/consistency to panel genes only
for _k in list(prior.pooled.keys()):
    if _k not in _keep:
        del prior.pooled[_k]
for _k in list(prior.consistency.keys()):
    if _k not in _keep:
        del prior.consistency[_k]
gc.collect()
print(f"[prior] stripped to panel-only", flush=True)

rng = np.random.default_rng(0)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def ensemble_lfc(prior_vec: np.ndarray, basal: np.ndarray, esm_vec: np.ndarray | None) -> np.ndarray:
    G = len(genes)
    pr = torch.from_numpy(np.asarray(prior_vec, dtype=np.float32).reshape(1, G)).to(device)
    b = torch.from_numpy(np.asarray(basal, dtype=np.float32).reshape(1, G)).to(device)
    if esm_vec is None:
        esm_vec = np.zeros(members[0].cfg.esm_dim, dtype=np.float32)
    e = torch.from_numpy(np.asarray(esm_vec, dtype=np.float32).reshape(1, -1)).to(device)
    outs = []
    for m in members:
        corr = m.forward(pr, b, e)
        outs.append((pr + corr).squeeze(0).cpu().numpy())
    mean = np.mean(np.stack(outs), axis=0)
    return np.clip(mean, -8.0, 8.0)

with StreamingSubmissionWriter(out_path, genes) as writer:
    for fpath in _ctrl_files:
        ctx, ctrl = _load_one_ctx(fpath)
        sel = rng.choice(ctrl.shape[0], size=min(4000, ctrl.shape[0]), replace=False)
        basal = sparse_log_mean_cpm(ctrl[sel], 1e6)
        print(f"[ctx {ctx}] basal {basal.mean():.3f}", flush=True)
        for j, gene in enumerate(panel_genes):
            p = prior.predict_lfc(gene, basal)
            ev = esm.get(gene) if esm else None
            lfc = ensemble_lfc(p, basal, ev)
            mult = np.exp2(np.clip(lfc, -6, 6))
            from vcc_model.countsim import poisson_thin as _pt
            for _s0 in range(0, 400, 200):
                pick = rng.choice(ctrl.shape[0], size=200, replace=True)
                base = np.asarray(ctrl[pick].todense(), dtype=np.float32)
                cells = _pt(base, mult, rng)
                writer.add_block(cells, [gene] * 200, [ctx] * 200)
                del base, cells
            if (j + 1) % 50 == 0:
                print(f"  {ctx} {j+1}/{len(panel_genes)}", flush=True)
        print(f"[ctx {ctx}] done", flush=True)
        del ctrl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[writer] nnz {writer._nnz:,} n_obs {writer._n_obs:,}", flush=True)

problems = validate_h5ad_file_light(out_path, genes, contexts=("A", "B", "C"),
                                    perts_per_context=len(panel_genes), cells_per_pert=400)
if problems:
    print("[validate] problems:", problems, flush=True)
    sys.exit(1)
print("[validate] OK", flush=True)
print(f"[save] {out_path} {out_path.stat().st_size/1e6:.1f} MB", flush=True)
import subprocess
vcc_path = out_path.with_suffix(".vcc")
subprocess.run([sys.executable, "scripts/pack_vcc.py", str(out_path), str(vcc_path)], check=True)
print(f"[pack] {vcc_path}", flush=True)
