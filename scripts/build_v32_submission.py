"""Build v32 submission: prior + CellDynamics ensemble (95M) blended.

Usage:
  python scripts/build_v32_submission.py --members 3 --epochs 15
  python scripts/build_v32_submission.py --members 5 --epochs 20 --full  # final
"""
import sys; sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument("--members", type=int, default=3)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--max-pairs", type=int, default=8000)
parser.add_argument("--blend", type=float, default=0.2, help="w for dynamics in (1-w)*prior + w*dyn")
parser.add_argument("--full", action="store_true", help="5 members 20 epochs 15000 pairs")
parser.add_argument("--out", type=str, default="artifacts/submission/pred_v32.h5ad")
args = parser.parse_args()
if args.full:
    args.members = 5
    args.epochs = 20
    args.max_pairs = 15000

from vcc_model.dataset import load_gene_names_csv
from vcc_model.dynamics import DynamicsConfig
from vcc_model.ensemble_surrogate import EnsembleDynamics
from vcc_model.prior import LFCTransferPrior
from vcc_model.countsim import log_mean_cpm
from vcc_model.ensemble import HybridPredictor

# --- load stores ---
stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
pairs, basals, genes = {}, {}, None
for p in stores:
    z = np.load(p, allow_pickle=True)
    if genes is None:
        genes = list(z['gene_names'].astype(str))
    for c,g,v in zip(z['context'], z['target'], z['lfc']):
        pairs[(str(c),str(g))] = v
    for i,c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]
print(f"[v32] loaded {len(pairs)} pairs {len(basals)} contexts {len(genes)} genes", flush=True)

# --- ESM ---
esm = None
for cand in [Path("data/protein/se_full_18533.npz"), Path("artifacts/se_full_18533.npz")]:
    if cand.exists():
        d = np.load(cand, allow_pickle=True)
        gn = [str(x) for x in d['gene_names']]; mat = d['embeddings']
        n = np.linalg.norm(mat, axis=1, keepdims=True); n = np.where(n<1e-9,1.0,n); mat=mat/n
        esm = {g:mat[i].astype(np.float32) for i,g in enumerate(gn)}
        print(f"[esm] {len(esm)} genes", flush=True)
        break

# --- fit prior on ALL data (for submission) ---
nested={}
for (c,g),v in pairs.items():
    nested.setdefault(c,{})[g]=v
prior = LFCTransferPrior(similarity_temperature=0.1, shared_mix=0.0, use_alpha=False, zero_unseen=True).fit(nested, basals, genes)
print(f"[prior] {len(prior.contexts)} contexts alpha {prior.global_alpha:.3f}", flush=True)

# --- train ensemble on ALL data ---
cfg = DynamicsConfig(epochs=args.epochs, batch_size=128, max_pairs=args.max_pairs, w_rank=0.5, w_cos=1.0, w_l1=0.3)
print(f"[ensemble] training {args.members} members {args.epochs} epochs {cfg.enc_h1} latent {cfg.latent_dim} ({cfg})", flush=True)
if args.members == 1:
    from vcc_model.dynamics import train_cell_dynamics
    m, info = train_cell_dynamics(pairs, basals, genes, esm_embeddings=esm, cfg=cfg)
    # wrap
    class _Ens:
        def __init__(self, m, esm): self.m=m; self._esm=esm; self.ood=None
        def predict_lfc(self, g, b): return self.m.predict_lfc(b, self._esm.get(g) if self._esm else None)
    ens = _Ens(m, esm)
else:
    ens = EnsembleDynamics.train(pairs, basals, genes, esm_embeddings=esm, n_members=args.members, base_cfg=cfg, verbose=True)

# free training memory before control loading
import gc
try:
    del pairs
except: pass
gc.collect()
if 'torch' in globals() or True:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except: pass

# --- load validation controls ---
# Find gene_names and controls
gene_names = genes
# Load controls from data/ (context_A/B/C.h5ad or similar)
ctrl_root = Path("data/vcc2026")
# Try vcc data layout
ctrl_files = list(ctrl_root.glob("context_*.h5ad")) + list(ctrl_root.glob("*.h5ad"))
if not ctrl_files:
    ctrl_root = Path("data")
    ctrl_files = list(ctrl_root.glob("context_*.h5ad")) + list(ctrl_root.glob("*.h5ad"))
# Also check artifacts
if not ctrl_files:
    ctrl_files = list(Path("artifacts").glob("*.h5ad"))
print(f"[ctrl] files {ctrl_files[:5]}", flush=True)
# Use dataset helper if available
from vcc_model.dataset import load_controls_bundle
try:
    controls = load_controls_bundle(ctrl_root if ctrl_files else Path("data/vcc2026"), gene_names)
    print(f"[ctrl] loaded {list(controls.keys())} contexts", flush=True)
except Exception as e:
    print(f"[ctrl] load_controls_bundle failed {e}", flush=True)
    # fallback: try direct
    controls = {}
    for f in ctrl_files:
        try:
            a = ad.read_h5ad(f)
            # harmonize
            from vcc_model.dataset import harmonize_to_axis, detect_columns
            tcol, ccol = detect_columns(a.obs)
            ctx = f.stem.replace("context_","")
            mat = harmonize_to_axis(a.X, a.var_names, gene_names)
            controls[ctx] = mat
            print(f"  {f.name} -> {ctx} {mat.shape}", flush=True)
        except Exception as e2:
            print(f"  {f.name} failed {e2}", flush=True)

if not controls:
    print("no controls found — cannot build submission"); sys.exit(1)

# --- load panel ---
panel_path = Path("data/vcc2026/pert_counts.csv")
if not panel_path.exists():
    panel_path = Path("data/vcc2026/gene_names.csv")
    # try alternative
    for cand in [Path("data/pert_counts.csv"), Path("artifacts/pert_counts.csv"), Path("pert_counts.csv")]:
        if cand.exists():
            panel_path = cand; break
if panel_path.exists():
    df = pd.read_csv(panel_path)
    # find gene col
    cols = {c.lower():c for c in df.columns}
    gene_col = next((cols[c] for c in ["target_gene","target","perturbation","gene"] if c in cols), df.columns[0])
    panel_genes = df[gene_col].astype(str).tolist()
    print(f"[panel] {len(panel_genes)} genes from {panel_path}", flush=True)
else:
    # fallback: use all genes in pairs
    panel_genes = sorted({g for _,g in pairs})[:300]
    print(f"[panel] fallback {len(panel_genes)}", flush=True)

# dedupe panel
panel_genes = list(dict.fromkeys(panel_genes))[:300]

# --- generate cells per context ---
from vcc_model.ensemble import HybridPredictor as HP
# We will blend prior + dynamics directly, not via HybridPredictor, to keep dynamics in loop
import numpy as np
rng = np.random.default_rng(0)
from vcc_model.submission import StreamingSubmissionWriter, validate_h5ad_file_light
# use streaming writer like make_submission.py to avoid RAM blowup
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
# need gene_names already, controls already loaded
import tempfile, gc

with StreamingSubmissionWriter(out_path, gene_names) as writer:
    for ctx in sorted(controls.keys())[:3]:
        ctrl_mat = controls[ctx]
        if ctrl_mat.shape[0] > 4000:
            sel = rng.choice(ctrl_mat.shape[0], size=4000, replace=False)
            ctrl_sample = ctrl_mat[sel]
        else:
            ctrl_sample = ctrl_mat
        from vcc_model.countsim import sparse_log_mean_cpm
        basal = sparse_log_mean_cpm(ctrl_sample, 1e6) if hasattr(ctrl_sample, 'tocsr') else log_mean_cpm(np.asarray(ctrl_sample.todense(), dtype=np.float64), 1e6)
        print(f"[ctx {ctx}] ctrl {ctrl_mat.shape} basal mean {basal.mean():.3f} (sample {ctrl_sample.shape[0]})", flush=True)
        for gene in panel_genes:
            p = prior.predict_lfc(gene, basal)
            try:
                d = ens.predict_lfc(gene, basal)
            except Exception:
                d = np.zeros_like(p)
            lfc = (1-args.blend)*p + args.blend*d
            lfc = np.clip(lfc, -6, 6)
            mult = np.exp2(lfc)
            # generate 400 cells at once (float32, 29MB) - streaming writer handles incremental disk write
            pick = rng.choice(ctrl_mat.shape[0], size=400, replace=True)
            base_sparse = ctrl_mat[pick]
            base = np.asarray(base_sparse.todense() if hasattr(base_sparse, 'todense') else base_sparse, dtype=np.float32)
            from vcc_model.countsim import poisson_thin as _pt
            cells = _pt(base, mult, rng)
            writer.add_block(cells, [gene]*400, [ctx]*400)
            del base, base_sparse, cells
            if panel_genes.index(gene) % 50 == 0:
                print(f"  {ctx} {gene} {panel_genes.index(gene)+1}/{len(panel_genes)}", flush=True)
        print(f"[ctx {ctx}] done", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[writer] nnz {writer._nnz:,} n_obs {writer._n_obs:,}", flush=True)

# --- validate ---
from vcc_model.submission import validate_h5ad_file_light
problems = validate_h5ad_file_light(out_path, gene_names, contexts=tuple(sorted(controls.keys())[:3]), perts_per_context=len(panel_genes), cells_per_pert=400)
if problems:
    print("[validate] problems:", problems, flush=True)
    sys.exit(1)
print("[validate] OK", flush=True)
print(f"[save] {out_path} {out_path.stat().st_size/1e6:.1f} MB", flush=True)
# skip duplicate write/pack below by exiting early after validation
import subprocess
try:
    vcc_path = out_path.with_suffix(".vcc")
    subprocess.run([sys.executable, "scripts/pack_vcc.py", str(out_path), str(vcc_path)], check=True)
    print(f"[pack] {vcc_path}", flush=True)
except Exception as e:
    print(f"[pack] skipped {e}", flush=True)
try:
    if hasattr(ens, 'save'):
        ens.save(f"artifacts/ensemble_v32_{args.members}members")
        print(f"[save] ensemble saved", flush=True)
except Exception as e:
    print(f"[save] ensemble save failed {e}", flush=True)
sys.exit(0)
