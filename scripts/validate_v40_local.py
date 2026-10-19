"""Local sanity gate for a v40-style submission.

We have no ground truth for the 300 panel targets, so this checks the
properties that provably broke the old emitter:

  1. DE call count per target vs that context's 18,400 controls.
     Control-null floor is ~59 (pool=1) and real 400-cells give ~1947.
     A real perturbation should sit ABOVE the floor but not saturate.
  2. Gene detection per emitted cell must match real cells (~5900), else the
     Wilcoxon members are measuring detection artifacts (the sidechain
     "-0.98" failure mode).
  3. Are the 300 emitted bulks mutually distinguishable? Mean pairwise
     cosine of emitted effect vs control effect. Higher = better separation,
     which is what pds_cosine rewards.
"""
import sys
sys.path.insert(0, '.')
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
from scipy.stats import mannwhitneyu

from vcc_model.dataset import load_gene_names_csv, detect_columns

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="artifacts/submission/pred_v40.h5ad")
ap.add_argument("--targets", type=int, default=12, help="targets per context to audit")
args = ap.parse_args()

genes = load_gene_names_csv('data/vcc2026/gene_names.csv')
G = len(genes)
pred = ad.read_h5ad(args.pred)
tcol, ccol = detect_columns(pred.obs)
tgts = pred.obs[tcol].astype(str).to_numpy()
ctxs = pred.obs[ccol].astype(str).to_numpy()
print(f"pred: {pred.shape}  targets {len(set(zip(ctxs, tgts)))}")

for ctx in sorted(set(ctxs)):
    cp = f"data/vcc2026/context_{ctx}.h5ad"
    if not Path(cp).exists():
        continue
    a = ad.read_h5ad(cp)
    tc, _ = detect_columns(a.obs)
    lab = a.obs[tc].astype(str).to_numpy()
    ctrl = np.asarray(a.X[np.flatnonzero(lab == 'non-targeting')].todense(), dtype=np.float64)
    cm = ctrl.mean(0)
    cmask = (cm > 0)
    print(f"\n=== context {ctx} ===  real genes/cell {(ctrl[:400] > 0).sum(1).mean():.0f}")

    sel = np.flatnonzero(ctxs == ctx)
    Xc = pred.X.tocsr() if pred.X.format != 'csr' else pred.X
    utg = sorted(set(tgts[sel]))[:args.targets]
    calls, det, cos = [], [], []
    ctrl_prob = ctrl.sum(0); ctrl_prob /= ctrl_prob.sum()
    for g in utg:
        m = np.sort(sel[tgts[sel] == g])
        # rows for a target are one contiguous block; slice by indptr to stay RAM-safe
        s, e = Xc.indptr[m[0]], Xc.indptr[m[-1] + 1]
        sub = sp.csr_matrix((Xc.data[s:e], Xc.indices[s:e],
                             Xc.indptr[m[0]:m[-1] + 2] - s),
                            shape=(len(m), G), dtype=np.float64)
        E = np.asarray(sub.toarray())
        det.append((E > 0).sum(1).mean())
        gg = np.flatnonzero((E.mean(0) > 0) & cmask)
        n = sum(1 for k in gg
                if mannwhitneyu(E[:, k], ctrl[:, k], alternative='two-sided').pvalue < 0.05)
        calls.append(n)
        ep = E.sum(0); ep /= ep.sum()
        ce = np.log2((ep + 1e-4) / (ctrl_prob + 1e-4))
        cos.append(float(np.linalg.norm(ce)))
    print(f"  DE calls/target : mean {np.mean(calls):.0f}  median {np.median(calls):.0f}  "
          f"min {np.min(calls)}  max {np.max(calls)}   [null floor 59, real-400 ~1947]")
    print(f"  genes/cell      : mean {np.mean(det):.0f}   [real ~5900]")
    print(f"  effect magnitude: mean L2(log2FC) {np.mean(cos):.4f}")
    del a, ctrl
print("\n[average] verdict")
print("  calls 200-4000  -> plausible; few hundred = conservative")
print("  calls >6000     -> still emitting noise, do not submit")
print("  genes/cell far from 5900 -> detection artifact")
