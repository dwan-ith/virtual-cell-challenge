"""v36 probe: v29-EXACT prior pipeline + top-K sparsity mask.

v29 recipe (from artifacts/loco_config_v29.json):
  prior: temp 0.1, basal_gate 5.0, use_alpha False, gene_conditioned True,
         ESM se_full_18533, shared_mix 0.0, zero_unseen True, cons 0.45
  gain map: build_per_gene_gain_map(cons, scale 0.7, esm_boost 0.15)
  calib: DECalibration(thr 0.0, gain 1.8, power 1.0, q 0.25)
  weights (1,0,0) = prior only. All 8 stores (Tahoe KEPT per rd_notahoe).

NEW (only change vs v29): top-K mask (K=150) on |calibrated LFC| per target.
  Kept genes: poisson-thin normally. Masked genes: COPY resampled control
  cells exactly (no thinning noise -> no false DE calls, no MSE inflation).
  Kept set additionally requires |lfc| > 0 (zero-LFC genes always copied).

Usage: python scripts/build_v36_sparse_probe.py [--K 150] [--out ...]
"""
import sys
sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=150)
parser.add_argument("--out", type=str, default="artifacts/submission/pred_v36.h5ad")
parser.add_argument("--only-contexts", nargs="*", default=None)
parser.add_argument("--append", action="store_true")
parser.add_argument("--pack", action="store_true",
                    help="validate full file and pack to .vcc (run once after final chunk)")
parser.add_argument("--gain", type=float, default=1.8,
                    help="DECalibration gain (v29 used 1.8; <1.8 shrinks magnitude)")
parser.add_argument("--q", type=float, default=0.25, help="DECalibration q_threshold")
parser.add_argument("--no-mask", action="store_true",
                    help="disable top-K masking (plain v29 emission)")
args = parser.parse_args()

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.dataset import load_gene_names_csv
from vcc_model.countsim import sparse_log_mean_cpm, poisson_thin
from vcc_model.submission import StreamingSubmissionWriter, validate_h5ad_file_light
import anndata as ad
from vcc_model.dataset import harmonize_to_axis, detect_columns

# --- training stores (all 8, Tahoe kept) ---
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
print(f"[v36] {len(pairs)} pairs {len(basals)} ctx", flush=True)

esm = load_esm_embeddings(Path("data/protein/se_full_18533.npz"))
print(f"[esm] {len(esm) if esm else 0} genes", flush=True)

nested = {}
for (c, g), v in pairs.items():
    nested.setdefault(c, {})[g] = v
prior = LFCTransferPrior(
    similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
    gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
    zero_unseen=True, default_conservation=0.45,
).fit(nested, basals, genes)
print(f"[prior] {len(prior.contexts)} ctx alpha {prior.global_alpha:.3f}", flush=True)

pg = build_per_gene_gain_map(prior.consistency, genes, default_cons=0.45,
                             scale=0.7, esm_embeddings=esm, esm_boost=0.15)
cal = DECalibration(threshold=0.0, gain=args.gain, power=1.0, q_threshold=args.q, per_gene_gain=pg)
print(f"[calib] gain {args.gain} power 1.0 q {args.q} + per-gene map; mask={'OFF' if args.no_mask else f'K={K}'}", flush=True)

# --- strip to panel genes later (after panel load); free store buffers refs ---
panel = pd.read_csv("data/vcc2026/pert_counts.csv")
panel_genes = panel[panel.columns[0]].astype(str).tolist()[:300]
print(f"[panel] {len(panel_genes)}", flush=True)
_keep = set(panel_genes)
for _c in list(prior.lfc.keys()):
    _d = prior.lfc[_c]
    prior.lfc[_c] = {g: np.array(v, dtype=np.float32) for g, v in _d.items() if g in _keep}
del pairs, nested, _d
gc.collect()

gene_names = genes
rng = np.random.default_rng(0)
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
K = args.K

def topk_mask(lfc: np.ndarray, k: int) -> np.ndarray:
    """Boolean keep-mask: top-k |lfc| AND |lfc| > 0."""
    mag = np.abs(lfc)
    if (mag > 0).sum() <= k:
        return mag > 0
    idx = np.argpartition(mag, -k)[-k:]
    keep = np.zeros_like(mag, dtype=bool)
    keep[idx] = True
    keep &= mag > 0
    return keep

with StreamingSubmissionWriter(out_path, gene_names, append=args.append) as writer:
    _files = sorted(Path("data/vcc2026").glob("context_*.h5ad"))[:3]
    if args.only_contexts:
        _want = {str(c) for c in args.only_contexts}
        _files = [f for f in _files
                  if f.stem.removeprefix("context_").removeprefix("context-") in _want]
    for fpath in _files:
        a = ad.read_h5ad(fpath)
        tcol, _ = detect_columns(a.obs)
        x = a.X
        if tcol is not None:
            labels = a.obs[tcol].astype(str)
            keep = (labels == "non-targeting").to_numpy()
            if keep.any() and not keep.all():
                x = x[keep]
        ctx = fpath.stem.removeprefix("context_").removeprefix("context-")
        ctrl = harmonize_to_axis(x, a.var_names, gene_names).tocsr()
        del a, x
        gc.collect()
        basal = sparse_log_mean_cpm(ctrl, 1e6)
        print(f"[ctx {ctx}] ctrl {ctrl.shape} basal {basal.mean():.3f}", flush=True)
        # batched v29-identical predictions (verified maxdiff 0.0 vs predict_lfc)
        raw_map = prior.predict_many_gc(panel_genes, basal)
        for j, gene in enumerate(panel_genes):
            raw = raw_map[gene]
            lfc = np.asarray(cal.transform(raw, gene=gene), dtype=np.float64)
            if args.no_mask:
                masked_idx = np.empty(0, dtype=int)
            else:
                keep_mask = topk_mask(lfc, K)
                masked_idx = np.flatnonzero(~keep_mask)
            mult = np.exp2(np.clip(lfc, -6, 6)).astype(np.float32)
            # emit in 2x200 chunks; masked columns copied exactly
            for _s0 in range(0, 400, 200):
                pick = rng.choice(ctrl.shape[0], size=200, replace=True)
                base = np.asarray(ctrl[pick].todense(), dtype=np.float32)
                cells = poisson_thin(base, mult, rng)
                if masked_idx.size:
                    cells[:, masked_idx] = base[:, masked_idx]
                writer.add_block(cells, [gene] * 200, [ctx] * 200)
                del base, cells
            if (j + 1) % 100 == 0:
                print(f"  {ctx} {j+1}/{len(panel_genes)}", flush=True)
        print(f"[ctx {ctx}] done", flush=True)
        del ctrl
        gc.collect()
    print(f"[writer] nnz {writer._nnz:,} n_obs {writer._n_obs:,}", flush=True)

if not args.pack:
    print(f"[chunk done] {out_path} {out_path.stat().st_size/1e6:.1f} MB (validation+pack skipped, rerun with --pack after final chunk)", flush=True)
    sys.exit(0)
problems = validate_h5ad_file_light(out_path, gene_names, contexts=("A", "B", "C"),
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
