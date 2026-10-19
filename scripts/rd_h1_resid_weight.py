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
target_basal=basals[holdout]
# Baseline prior only
model_base=HybridPredictor.fit(lfc_pairs=train_pairs, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=gene_names, residual_cfg=ResidualConfig(epochs=2), fit_residual=False, select_weights=False, prior_kwargs={"similarity_temperature":0.1,"basal_gate_cpm":5.0,"use_alpha":False})
preds_base={g: model_base.predict_lfc(g, target_basal) for _,g in test_pairs}
truths={g: v for (_,g),v in test_pairs.items()}
pds_base=pds_proxy_fast(preds_base, truths, gene_names)
print(f"BASELINE pds {pds_base:.3f}")
# ESM residual with weight 0.3
cfg_esm=ResidualConfig(esm_dim=5120, esm_path="data/protein/se_full_18533.npz", epochs=10, max_pairs=1000, batch_size=256, seed=0)
model_esm=HybridPredictor.fit(lfc_pairs=train_pairs, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=gene_names, residual_cfg=cfg_esm, fit_residual=True, select_weights=False, prior_kwargs={"similarity_temperature":0.1,"basal_gate_cpm":5.0,"use_alpha":False,"gene_conditioned":True,"esm_path":"data/protein/se_full_18533.npz"})
print(f"ESM prior centroids {len(model_esm.prior._esm_context_centroids) if model_esm.prior._esm_context_centroids else 0} resid {model_esm.residual is not None}")
# Force weights to use resid
model_esm.weights=(0.7,0.3,0.0)
preds_esm={g: model_esm.predict_lfc(g, target_basal) for g in preds_base}
pds_esm=pds_proxy_fast(preds_esm, truths, gene_names)
print(f"ESM resid weight 0.3 pds {pds_esm:.3f} dPDS {pds_esm-pds_base:+.3f}")
# Also test prior-only ESM (no resid) with same prior but weight 1,0,0
model_esm_prior_only=HybridPredictor.fit(lfc_pairs=train_pairs, basals={k:v for k,v in basals.items() if k!=holdout}, gene_names=gene_names, residual_cfg=ResidualConfig(epochs=1), fit_residual=False, select_weights=False, prior_kwargs={"similarity_temperature":0.1,"basal_gate_cpm":5.0,"use_alpha":False,"gene_conditioned":True,"esm_path":"data/protein/se_full_18533.npz"})
preds_esm_prior={g: model_esm_prior_only.predict_lfc(g, target_basal) for g in preds_base}
pds_esm_prior=pds_proxy_fast(preds_esm_prior, truths, gene_names)
print(f"ESM prior-only pds {pds_esm_prior:.3f} dPDS {pds_esm_prior-pds_base:+.3f}")
print("Done")
