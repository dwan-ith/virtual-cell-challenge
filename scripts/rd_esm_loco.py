"""Rigorous LOCO: ESM-conditioned residual vs statistical baseline.
Holdout = replogle_gw/RPE1 and vcc2025/H1 (the only true unseen-line contexts).
Measures pds_proxy_fast and lfc_proxy, not just surrogate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, time, json
from vcc_model.dataset import PseudobulkStore
from vcc_model.ensemble import HybridPredictor
from vcc_model.residual import ResidualConfig
from vcc_model.spec import N_GENES

stores = sorted([p for p in Path("artifacts/stores").glob("*.npz") if "__controls" not in p.name])
merged = None
for p in stores:
    s = PseudobulkStore.load(p)
    if merged is None:
        merged = s
        continue
    assert np.array_equal(merged.gene_names, s.gene_names)
    merged.pairs = merged.pairs + s.pairs
    merged.lfc = np.vstack([merged.lfc, s.lfc])
    merged.n_cells.update(s.n_cells)
    for c,b in s.basal.items():
        if c not in merged.basal:
            merged.basal[c] = b

gene_names = list(merged.gene_names)
all_pairs = {pair: merged.lfc[i] for i,pair in enumerate(merged.pairs)}
basals = dict(merged.basal)
print(f"Total pairs {len(all_pairs)} contexts {sorted(basals.keys())}")

# Holdouts: RPE1 and H1 are the only true unseen lines
holdouts = ["replogle_gw/RPE1", "vcc2025/H1"]
for holdout in holdouts:
    print(f"\n=== LOCO holdout {holdout} ===")
    # Split: train excludes holdout, test is holdout's pairs
    train_pairs = {k:v for k,v in all_pairs.items() if k[0] != holdout}
    test_pairs = {k:v for k,v in all_pairs.items() if k[0] == holdout}
    if not test_pairs:
        print(f"  no test pairs for {holdout}, skipping")
        continue
    print(f"  train {len(train_pairs)} test {len(test_pairs)}")
    # Baseline: stat only, no residual, no ESM
    t0=time.time()
    from vcc_model.prior import LFCTransferPrior
    # Baseline prior
    nested = {}
    for (c,g),v in train_pairs.items():
        nested.setdefault(c,{})[g]=v
    prior_base = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, gene_names)
    # Build predictions for test genes
    # Group test by gene
    test_genes = sorted({g for c,g in test_pairs})
    # Use target basal from holdout (if available) else mean of train basals
    target_basal = basals.get(holdout)
    if target_basal is None:
        target_basal = np.mean(list(basals.values()), axis=0)
    preds_base = {}
    truths = {}
    for (c,g),v in test_pairs.items():
        # predict using prior (leave-one-context-out already via train)
        if g not in preds_base:
            preds_base[g] = prior_base.predict_lfc(g, target_basal)
            truths[g] = v
    # PDS proxy fast requires at least 2 genes
    from vcc_model.ensemble import pds_proxy_fast
    # Need to filter to genes where prediction non-zero
    pds_base = pds_proxy_fast(preds_base, truths, gene_names) if len(preds_base)>=2 else float('nan')
    # LFC proxy aggregate
    from vcc_model.ensemble import lfc_proxy_aggregate
    # Compute per-pair cosine for proxy
    cos_vals = []
    for g in preds_base:
        p = preds_base[g]; t = truths[g]
        # keep panel genes? pds_proxy_fast already masks, for lfc proxy use full
        denom = np.linalg.norm(p)*np.linalg.norm(t)
        cos = float(np.dot(p,t)/denom) if denom>1e-9 else 0.0
        cos_vals.append(cos)
    mean_cos = float(np.mean(cos_vals)) if cos_vals else float('nan')
    print(f"  BASELINE stat-only: pds_proxy={pds_base:.3f} mean_cos={mean_cos:.3f} (n={len(preds_base)})")

    # ESM-conditioned: prior with gene_conditioned + ESM, plus per-gene adaptive gain via calibrate
    # For LOCO we test the prior-level ESM (gene_conditioned) + residual ESM
    try:
        from vcc_model.protein import load_esm_embeddings
        esm = load_esm_embeddings()
        print(f"  ESM {len(esm) if esm else 0} genes")
        if esm:
            prior_esm = LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, gene_names)
            # Need to recompute centroids: prior.fit already does if esm set at construction
            # But our prior was constructed with esm, so centroids already computed in fit
            preds_esm = {}
            for g in preds_base:
                preds_esm[g] = prior_esm.predict_lfc(g, target_basal)
            pds_esm = pds_proxy_fast(preds_esm, truths, gene_names) if len(preds_esm)>=2 else float('nan')
            cos_esm = []
            for g in preds_esm:
                p = preds_esm[g]; t = truths[g]
                denom = np.linalg.norm(p)*np.linalg.norm(t)
                cos_esm.append(float(np.dot(p,t)/denom) if denom>1e-9 else 0.0)
            mean_cos_esm = float(np.mean(cos_esm)) if cos_esm else float('nan')
            print(f"  ESM prior gene-conditioned: pds_proxy={pds_esm:.3f} mean_cos={mean_cos_esm:.3f} dPDS={pds_esm-pds_base:+.3f} dCOS={mean_cos_esm-mean_cos:+.3f}")

            # Residual with ESM
            cfg_esm = ResidualConfig(esm_dim=320, esm_path="data/protein/esm2_t6_8m_panel.npz", epochs=5, max_pairs=500, batch_size=256, seed=0)
            # Fit hybrid with residual ESM
            # Use HybridPredictor with prior_kwargs to get ESM prior + residual
            model_esm = HybridPredictor.fit(
                lfc_pairs=train_pairs, basals={k:v for k,v in basals.items() if k!=holdout},
                gene_names=gene_names, residual_cfg=cfg_esm, fit_residual=True, select_weights=False,
                prior_kwargs={"similarity_temperature":0.1, "basal_gate_cpm":5.0, "use_alpha":False, "gene_conditioned":True, "esm_path":"data/protein/esm2_t6_8m_panel.npz"}
            )
            # Predict with model (prior+residual blend, but weights default 1,0.7,0 - we force stat only for fair vs ESM prior alone, or blend)
            # For this test, use prior+residual with default weights (1,0.7,0)
            # Instead we test prior+residual ESM vs baseline
            preds_resid = {}
            for g in preds_base:
                # blend_lfc with weights (1,0.7,0) would use residual; but to isolate ESM we use prior_esm + residual
                # Simpler: use model_esm.blend_lfc with its weights
                preds_resid[g] = model_esm.blend_lfc(g, target_basal)
                # Apply per-gene? not needed for proxy
            pds_resid = pds_proxy_fast(preds_resid, truths, gene_names) if len(preds_resid)>=2 else float('nan')
            print(f"  ESM residual (prior+resid ESM): pds_proxy={pds_resid:.3f} dVsBase={pds_resid-pds_base:+.3f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ESM test failed: {e}")

print("\nDone")
