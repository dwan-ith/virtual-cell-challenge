import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vcc_model.dataset import PseudobulkStore
from vcc_model.ensemble import HybridPredictor, pds_proxy_fast
from vcc_model.residual import ResidualConfig

stores=[p for p in Path("artifacts/stores").glob("*.npz") if "__controls" not in p.name]
merged=None
for p in stores:
    s=PseudobulkStore.load(p)
    if merged is None:
        merged=s; continue
    merged.pairs=merged.pairs+s.pairs
    merged.lfc=np.vstack([merged.lfc,s.lfc])
    merged.n_cells.update(s.n_cells)
    for c,b in s.basal.items():
        if c not in merged.basal:
            merged.basal[c]=b
gene_names=list(merged.gene_names)
all_pairs={pair: merged.lfc[i] for i,pair in enumerate(merged.pairs)}
basals=dict(merged.basal)
holdout="vcc2025/H1"
train_pairs={k:v for k,v in all_pairs.items() if k[0]!=holdout}
test_pairs={k:v for k,v in all_pairs.items() if k[0]==holdout}
print(f"train {len(train_pairs)} test {len(test_pairs)}")
# Baseline: prior only
print("=== Baseline prior only ===")
model_base=HybridPredictor.fit(lfc_pairs=train_pairs, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=gene_names, residual_cfg=ResidualConfig(epochs=5), fit_residual=False, select_weights=False, prior_kwargs={"similarity_temperature":0.1,"basal_gate_cpm":5.0,"use_alpha":False})
# Evaluate on test
target_basal=basals[holdout]
preds_base={}
truths={}
for (c,g),v in test_pairs.items():
    preds_base[g]=model_base.predict_lfc(g, target_basal)
    truths[g]=v
pds_base=pds_proxy_fast(preds_base, truths, gene_names)
print(f"BASELINE pds {pds_base:.3f} n {len(preds_base)}")
# ESM residual
print("=== ESM residual ===")
cfg_esm=ResidualConfig(esm_dim=5120, esm_path="data/protein/se_full_18533.npz", epochs=30, max_pairs=2000, batch_size=256, seed=0)
model_esm=HybridPredictor.fit(lfc_pairs=train_pairs, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=gene_names, residual_cfg=cfg_esm, fit_residual=True, select_weights=False, prior_kwargs={"similarity_temperature":0.1,"basal_gate_cpm":5.0,"use_alpha":False,"gene_conditioned":True,"esm_path":"data/protein/se_full_18533.npz"})
print(f"ESM model weights {model_esm.weights} prior centroids {len(model_esm.prior._esm_context_centroids) if model_esm.prior._esm_context_centroids else 0}")
# Blend with default weights (1,0.7,0) vs (1,0,0) ? Use prior+resid
preds_esm={}
for g in preds_base:
    # Use blend_lfc which includes residual with weight 0.7 if we set, but fit used default weights (1,0.7,0)?? Actually select_weights False so weights stay (1,0.7,0) default? Check
    # For this test we use prior only vs prior+resid ESM with same prior but with resid
    # Use model_esm.blend_lfc with its weights
    preds_esm[g]=model_esm.predict_lfc(g, target_basal)
pds_esm=pds_proxy_fast(preds_esm, truths, gene_names)
print(f"ESM prior+resid pds {pds_esm:.3f} dPDS {pds_esm-pds_base:+.3f}")
# Also test prior-only ESM (without resid) for comparison
from vcc_model.prior import LFCTransferPrior
from vcc_model.protein import load_esm_embeddings
esm=load_esm_embeddings("data/protein/se_full_18533.npz")
nested={}
for (c,g),v in train_pairs.items():
    nested.setdefault(c,{})[g]=v
prior_esm2=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, gene_names)
preds_prior_esm={g: prior_esm2.predict_lfc(g, target_basal) for g in preds_base}
pds_prior_esm=pds_proxy_fast(preds_prior_esm, truths, gene_names)
print(f"ESM prior-only pds {pds_prior_esm:.3f} dPDS {pds_prior_esm-pds_base:+.3f}")
print("Done")
