import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, time
from vcc_model.dataset import PseudobulkStore
from vcc_model.ensemble import pds_proxy_fast

stores = [p for p in Path("artifacts/stores").glob("*.npz") if "__controls" not in p.name]
merged = None
for p in stores:
    s = PseudobulkStore.load(p)
    if merged is None:
        merged = s; continue
    merged.pairs = merged.pairs + s.pairs
    merged.lfc = np.vstack([merged.lfc, s.lfc])
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
# Baseline
from vcc_model.prior import LFCTransferPrior
nested={}
for (c,g),v in train_pairs.items():
    nested.setdefault(c,{})[g]=v
prior_base=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, gene_names)
target_basal=basals[holdout]
preds_base={g: prior_base.predict_lfc(g, target_basal) for _,g in test_pairs}
truths={g: v for (_,g),v in test_pairs.items()}
# Need to deduplicate genes (test has 100 unique genes, but pairs are per context/gene, so 100 unique)
# preds_base currently overwrites duplicates? test_pairs has 100 unique genes for H1, so ok
# But need to ensure we have one pred per gene
# For H1, each gene appears once, so dict is correct
pds_base=pds_proxy_fast(preds_base, truths, gene_names)
cos_base=np.mean([np.dot(preds_base[g],truths[g])/(np.linalg.norm(preds_base[g])*np.linalg.norm(truths[g]) +1e-9) for g in preds_base])
print(f"BASELINE H1: pds={pds_base:.3f} cos={cos_base:.4f}")
# ESM
from vcc_model.protein import load_esm_embeddings
esm=load_esm_embeddings()
print(f"ESM {len(esm) if esm else 0}")
if esm:
    prior_esm=LFCTransferPrior(similarity_temperature=0.1, basal_gate_cpm=5.0, use_alpha=False, gene_conditioned=True, esm_embeddings=esm).fit(nested, {k:v for k,v in basals.items() if k!=holdout}, gene_names)
    preds_esm={g: prior_esm.predict_lfc(g, target_basal) for g in preds_base}
    pds_esm=pds_proxy_fast(preds_esm, truths, gene_names)
    cos_esm=np.mean([np.dot(preds_esm[g],truths[g])/(np.linalg.norm(preds_esm[g])*np.linalg.norm(truths[g]) +1e-9) for g in preds_esm])
    print(f"ESM prior: pds={pds_esm:.3f} cos={cos_esm:.4f} dPDS={pds_esm-pds_base:+.3f} dCOS={cos_esm-cos_base:+.4f}")
    # Check weight difference for a specific gene
    w_base=prior_base.similarity_weights(target_basal)
    w_esm=prior_esm.similarity_weights(target_basal)
    print(f"w_base { {k: f'{v:.3f}' for k,v in list(w_base.items())[:3]} }")
    # Check per-gene weights for a test gene
    test_g=list(preds_base.keys())[0]
    # Use gene_conditioned_weights directly
    # Build protein_sim for test_g
    v=esm[test_g]/np.linalg.norm(esm[test_g]) if test_g in esm else None
    if v is not None and prior_esm._esm_context_centroids:
        ps={}
        for c,cent in prior_esm._esm_context_centroids.items():
            ps[c]=max(0,(float(np.dot(v,cent))+1)/2)
        w_gc=prior_esm.gene_conditioned_weights(target_basal, test_g, protein_sim=ps)
        print(f"gene {test_g} w_gc { {k: f'{v:.3f}' for k,v in w_gc.items()} }")
print("Done H1")
