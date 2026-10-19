"""Sweep default_conservation x gain x q_threshold on H1 LOCO proxy.

Measures PDS proxy in LFC space + magnitude ratio to find the optimal config
that maximizes PDS while keeping predictions large enough for Wilcoxon DE.
"""
import sys, os, json, time, glob, zlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcc_model.dataset import PseudobulkStore, load_gene_names_csv
from vcc_model.prior import LFCTransferPrior
from vcc_model.ensemble import HybridPredictor, pds_proxy_fast, log_mean_cpm
from vcc_model.calibrate import DECalibration
import pandas as pd

gene_names = load_gene_names_csv(ROOT / "data" / "vcc2026" / "gene_names.csv")
targets = pd.read_csv(ROOT / "data" / "vcc2026" / "pert_counts.csv")["target_gene"].tolist()

print("Loading stores...", flush=True)
t0 = time.time()
store_paths = sorted(glob.glob(str(ROOT / "artifacts" / "stores" / "*.npz")))
store_paths = [p for p in store_paths if "controls" not in p and "calibration" not in p]
stores = {}
for p in store_paths:
    s = PseudobulkStore.load(p)
    ctxs = s.contexts()
    ctx = ctxs[0] if ctxs else Path(p).stem
    stores[ctx] = s
print(f"Loaded {len(stores)} stores in {time.time()-t0:.1f}s: {sorted(stores.keys())}")

holdout = "vcc2025/H1"
train_ctx = [c for c in sorted(stores.keys()) if c != holdout]

# Build train data in prior.fit format: lfc[context][gene] = array
lfc_dict = {}
basal_dict = {}
for c in train_ctx:
    s = stores[c]
    ctx_lfc = {}
    for i, (ctx, gene) in enumerate(s.pairs):
        ctx_lfc[gene] = s.lfc[i].astype(np.float64)
    lfc_dict[c] = ctx_lfc
    basal_dict[c] = s.basal[c].astype(np.float64)

# H1 truth - use ALL H1 targets (not just VCC 300)
h1_pairs = stores[holdout].lfc_pairs()
h1_lfc = {gene: vec for (ctx, gene), vec in h1_pairs.items()}
h1_targets = list(h1_lfc.keys())
print(f"H1 truth targets: {len(h1_lfc)} (using all for proxy)")

# H1 basal for prediction
h1_ctxs = stores[holdout].contexts()
h1_basal = stores[holdout].basal[h1_ctxs[0]]

results = []
for cons in [0.45, 0.70, 0.85, 0.95]:
    for gain in [2.1, 5.0, 10.0, 20.0, 30.0, 50.0]:
        for q in [0.0, 0.2, 0.35]:
            t1 = time.time()
            prior = LFCTransferPrior(
                default_conservation=cons,
                use_alpha=False,
                shared_mix=0.0,
            )
            prior.fit(lfc_dict, basal_dict, gene_names, calibrate_scale=False)
            
            comp = HybridPredictor(prior=prior, gene_names=gene_names)
            comp.weights = (1.0, 0.0, 0.0)
            comp.calibration = DECalibration(power=1.0, gain=gain, threshold=0.0, q_threshold=q)
            
            # Predict LFC for H1 targets
            pred_lfc = {}
            for g in h1_lfc:
                lfc = comp.blend_lfc(g, h1_basal, weights=(1.0, 0.0, 0.0))
                pred_lfc[g] = lfc
            
            # PDS proxy on RAW predictions (before calibration)
            pds_raw = pds_proxy_fast(pred_lfc, h1_lfc, gene_names)
            
            # Apply calibration and measure PDS
            cal = DECalibration(power=1.0, gain=gain, threshold=0.0, q_threshold=q)
            pred_cal = {g: cal.transform(v) for g, v in pred_lfc.items()}
            pds_cal = pds_proxy_fast(pred_cal, h1_lfc, gene_names)
            
            # Magnitude stats
            pred_norms = [np.linalg.norm(v) for v in pred_cal.values()]
            truth_norms = [np.linalg.norm(v) for v in h1_lfc.values()]
            avg_pred = np.mean(pred_norms) if pred_norms else 0
            avg_truth = np.mean(truth_norms) if truth_norms else 0
            ratio = avg_pred / avg_truth * 100 if avg_truth > 0 else 0
            
            # Cosine similarity
            cosines = []
            for g in pred_cal:
                if g in h1_lfc:
                    p, t = pred_cal[g], h1_lfc[g]
                    cos = float(np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t) + 1e-10))
                    cosines.append(cos)
            avg_cos = np.mean(cosines) if cosines else 0
            
            r = {"cons": cons, "gain": gain, "q": q,
                 "pds_raw": pds_raw, "pds_cal": pds_cal,
                 "avg_norm": avg_pred, "truth_norm": avg_truth,
                 "ratio_pct": ratio, "avg_cos": avg_cos}
            results.append(r)
            dt = time.time() - t1
            print(f"cons={cons:.2f} gain={gain:6.1f} q={q:.2f} -> "
                  f"pds_raw={pds_raw:.4f} pds_cal={pds_cal:.4f} "
                  f"ratio={ratio:.1f}% cos={avg_cos:.4f} ({dt:.1f}s)", flush=True)

results.sort(key=lambda r: r["pds_cal"], reverse=True)
print("\n=== TOP 10 by PDS (calibrated) ===")
for r in results[:10]:
    print(f"  cons={r['cons']:.2f} gain={r['gain']:6.1f} q={r['q']:.2f} "
          f"-> pds_cal={r['pds_cal']:.4f} pds_raw={r['pds_raw']:.4f} "
          f"ratio={r['ratio_pct']:.1f}% cos={r['avg_cos']:.4f}")

print("\n=== WORST 5 ===")
for r in results[-5:]:
    print(f"  cons={r['cons']:.2f} gain={r['gain']:6.1f} q={r['q']:.2f} "
          f"-> pds_cal={r['pds_cal']:.4f} ratio={r['ratio_pct']:.1f}%")

with open(ROOT / "artifacts" / "gain_cons_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results")
