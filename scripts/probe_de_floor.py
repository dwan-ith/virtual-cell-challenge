"""Control-copy floor test: are our DE metrics negative because of noise, or position?

Builds ContextData from REAL perturbed cells (not thinned) in two ways:
  1. pred = resampled copy of the SAME cells (perfect expression match,
     zero biological signal)  -> isolates the sampling-noise floor
  2. pred = copy of INDEPENDENT control cells (the official baseline strategy)
Then runs the real six-metric evaluator. Negative fid/jac on (1) proves
sampling noise alone destroys those members and no masking can fix it.
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import anndata as ad
from scipy import sparse

from vcc_model.dataset import load_gene_names_csv, harmonize_to_axis, detect_columns
from vcc_model.countsim import poisson_thin
from vcc_model.metrics import ContextData, evaluate_context
from vcc_model import spec

ROOT = Path("data/vcc2026")
genes = load_gene_names_csv(ROOT / "gene_names.csv")
N_PERT = 8
N_CELLS = 400
rng = np.random.default_rng(0)

for ctx in ["A"]:
    a = ad.read_h5ad(ROOT / f"context_{ctx}.h5ad")
    tcol, _ = detect_columns(a.obs)
    labels = a.obs[tcol].astype(str).to_numpy()
    X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
    del a
    ctrl_rows = np.flatnonzero(labels == "non-targeting")
    # perturbed groups present in the control file (usually none)
    pert_labels = sorted(set(labels) - {"non-targeting"})
    print(f"[{ctx}] ctrl cells {len(ctrl_rows)} distinct target labels in file: {len(pert_labels)}", flush=True)
    if not pert_labels:
        print(f"[{ctx}] NOTE: validation controls contain only non-targeting cells; "
              f"no ground-truth perturbed cells available here.", flush=True)
        continue
    real_pert, pred_same, pred_indep = {}, {}, {}
    for nm in pert_labels[:N_PERT]:
        rows = np.flatnonzero(labels == nm)
        if len(rows) < N_CELLS:
            continue
        real_pert[nm] = X[rows[:N_CELLS]]
        pick_same = rng.choice(rows, size=N_CELLS, replace=True)
        pred_same[nm] = X[pick_same]
        pick_ind = rng.choice(ctrl_rows, size=N_CELLS, replace=True)
        pred_indep[nm] = X[pick_ind]
    names = sorted(real_pert)
    print(f"[{ctx}] groups used: {names}", flush=True)

    for label, pmap in [("copy-of-same-cells", pred_same), ("copy-of-controls", pred_indep)]:
        cd = ContextData(np.asarray(genes), X[ctrl_rows], real_pert, pmap)
        raw = evaluate_context(cd, n_control_subsample=4000, seed=0)
        print(f"[{ctx}] {label}: " + "  ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in raw.items()), flush=True)
