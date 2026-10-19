"""v40: AtlasShift-style submission on OUR stores (public #82 recipe, 0.1546).

Changes vs v29 (0.09233):
 1. PANEL-MEAN REMOVAL (common_subtract). Per gene, subtract the mean effect
    across all 300 panel targets (self column excluded). [model.py:101-107]
 2. DUAL-MOMENT COUNT CONSTRUCTION. Build a 400x18533 integer matrix matching
    BOTH the per-cell CPM mean (Wilcoxon DE members) and the depth-weighted
    bulk (pds_cosine / expr_mse). Deterministic, so none of the ~6800 spurious
    DE calls our Poisson/multinomial emitters produced. [model.py:332-444]
 3. POOLED CONTROL TEMPLATE. Each emitted cell starts as the mean of 4 real
    control cells, inheriting library size and zero structure.

Bulk amplitude is capped so the requested bulk stays inside the fitter's
feasible projection bound (measured: sd<=0.3 ok, 0.5 rejected).
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\aacer\AppData\Local\Temp\opencode\vcc\atlasshift')
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

from model import dual_moment_counts, seed_for

from vcc_model.prior import LFCTransferPrior
from vcc_model.calibrate import DECalibration
from vcc_model.protein import load_esm_embeddings, build_per_gene_gain_map
from vcc_model.dataset import load_gene_names_csv, harmonize_to_axis, detect_columns
from vcc_model.countsim import log_profile

parser = argparse.ArgumentParser()
parser.add_argument("--gain", type=float, default=1.0)
parser.add_argument("--common", type=float, default=1.0)
parser.add_argument("--amp-log2fc", type=float, default=0.6)
parser.add_argument("--amp-bulk", type=float, default=0.3)
parser.add_argument("--clip", type=float, default=3.0)
parser.add_argument("--pool", type=int, default=1,
                    help="template = mean of N real control cells. MUST be 1: pooling "
                         "inflates gene detection (5924->7281 genes/cell) and drives the "
                         "control-null DE floor from 59 to 8779 calls.")
parser.add_argument("--iterations", type=int, default=25)
parser.add_argument("--limit", type=int, default=0, help="only first N targets")
parser.add_argument("--only-contexts", nargs="*", default=None)
parser.add_argument("--append", action="store_true")
parser.add_argument("--pack", action="store_true")
parser.add_argument("--out", type=str, default="artifacts/submission/pred_v40.h5ad")
args = parser.parse_args()

genes = load_gene_names_csv('data/vcc2026/gene_names.csv')
G = len(genes)
gidx = {g: i for i, g in enumerate(genes)}
stores = [p for p in Path('artifacts/stores').glob('*.npz') if '__controls' not in p.name]
nested, basals = {}, {}
for p in stores:
    z = np.load(p, allow_pickle=True)
    for c, g, v in zip(z['context'], z['target'], z['lfc']):
        nested.setdefault(str(c), {})[str(g)] = v
    for i, c in enumerate(z['basal_ctx']):
        basals[str(c)] = z['basal'][i]
esm = load_esm_embeddings(Path("data/protein/se_full_18533.npz"))
prior = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False,
                         gene_conditioned=True, esm_embeddings=esm, shared_mix=0.0,
                         zero_unseen=True, default_conservation=0.45).fit(nested, basals, genes)
pg = build_per_gene_gain_map(prior.consistency, genes, 0.45, 0.7, esm, 0.15)
cal = DECalibration(0.0, args.gain, 1.0, 0.25, pg)
print(f"[v40] prior ok gain={args.gain} common={args.common}", flush=True)

panel = [str(g) for g in pd.read_csv('data/vcc2026/pert_counts.csv').iloc[:, 0].tolist()[:300]]
if args.limit:
    panel = panel[:args.limit]
rng = np.random.default_rng(0)
TS = 50000.0
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

from vcc_model.submission import StreamingSubmissionWriter, validate_h5ad_file_light

files = sorted(Path('data/vcc2026').glob('context_*.h5ad'))[:3]
if args.only_contexts:
    want = {str(c) for c in args.only_contexts}
    files = [f for f in files if f.stem.replace('context_', '') in want]

with StreamingSubmissionWriter(out_path, genes, append=args.append) as writer:
    for fp in files:
        ctx = fp.stem.replace('context_', '')
        a = ad.read_h5ad(fp)
        tcol, _ = detect_columns(a.obs)
        lab = a.obs[tcol].astype(str).to_numpy()
        X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
        ctrl = np.asarray(X[np.flatnonzero(lab == 'non-targeting')].todense(), dtype=np.float64)
        del a, X
        gc.collect()
        ctrl_prob = ctrl.sum(axis=0)
        ctrl_prob = ctrl_prob / ctrl_prob.sum()
        ctrl_bulk_log = np.log1p(TS * ctrl_prob)

        idx = rng.choice(ctrl.shape[0], size=(400, args.pool), replace=True)
        template = ctrl[idx].mean(axis=1)
        depths = np.clip(np.rint(template.sum(axis=1)).astype(np.int64), 1, 999999)
        template *= (depths[:, None] / np.maximum(template.sum(axis=1, keepdims=True), 1e-9))
        print(f"[{ctx}] ctrl {ctrl.shape} median depth {int(np.median(depths))}", flush=True)

        N = len(panel)
        eff_l2 = np.zeros((N, G))
        eff_bd = np.zeros((N, G))
        ctrl_cpm = ctrl_prob * 1e6
        for j, g in enumerate(panel):
            raw = prior.predict_lfc(g, np.log(np.maximum(ctrl_cpm, 1e-12)))
            lfc = np.asarray(cal.transform(raw, gene=g), dtype=np.float64)
            eff_l2[j] = lfc
            eff_bd[j] = log_profile(np.maximum(ctrl_cpm * np.exp2(np.clip(lfc, -6, 6)), 0), TS) - ctrl_bulk_log
            if (j + 1) % 100 == 0:
                print(f"  [{ctx}] prior {j+1}/{N}", flush=True)

        # ---- panel-mean removal: ONE pass over targets, self column excluded ----
        stack = eff_l2.copy()
        for k in range(N):
            own = gidx.get(panel[k])
            if own is not None:
                stack[k, own] = np.nan
        colmean = np.nan_to_num(np.nanmean(stack, axis=0))
        del stack
        eff_l2 -= args.common * colmean[None, :]
        eff_bd -= args.common * colmean[None, :]
        print(f"[{ctx}] panel-mean removed (mean|cm|={np.abs(colmean).mean():.4f})", flush=True)

        for j, g in enumerate(panel):
            e1 = np.clip(args.amp_log2fc * eff_l2[j], -args.clip, args.clip)
            p = ctrl_prob * np.exp2(e1)
            tot = p.sum()
            p = p / tot if tot > 0 else ctrl_prob.copy()
            e2 = np.clip(args.amp_bulk * eff_bd[j], -args.clip, args.clip)
            bulk = np.expm1(np.maximum(ctrl_bulk_log + e2, 0))
            bulk = bulk / bulk.sum() if bulk.sum() > 0 else ctrl_prob.copy()
            try:
                counts = dual_moment_counts(template, p, bulk, depths=depths,
                                            seed=seed_for(g, 7), iterations=args.iterations)
            except Exception as exc:
                print(f"  [{ctx}/{g}] fallback: {str(exc)[:50]}", flush=True)
                from vcc_model.countsim import multinomial_thin
                counts = multinomial_thin(template.astype(np.int64), np.exp2(e1), rng)
            writer.add_block(counts, [g] * 400, [ctx] * 400)
            del counts
            if (j + 1) % 50 == 0:
                print(f"  [{ctx}] {j+1}/{N}", flush=True)
        print(f"[{ctx}] done", flush=True)
        del ctrl, template, eff_l2, eff_bd
        gc.collect()
    print(f"[writer] nnz {writer._nnz:,} n_obs {writer._n_obs:,}", flush=True)

if not args.pack:
    print("[chunk done]", flush=True)
    sys.exit(0)
problems = validate_h5ad_file_light(out_path, genes, contexts=("A", "B", "C"),
                                    perts_per_context=len(panel), cells_per_pert=400)
print(f"[validate] {problems if problems else 'OK'}", flush=True)
if problems:
    sys.exit(1)
import subprocess
vcc_path = out_path.with_suffix(".vcc")
subprocess.run([sys.executable, "scripts/pack_vcc.py", str(out_path), str(vcc_path)], check=True)
print(f"[pack] {vcc_path}", flush=True)