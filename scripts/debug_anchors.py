"""Diagnose raw six-metric values + anchors on the synthetic challenge.

Prints RAW metric values (no anchor scaling) for oracle / control-copy /
mean-response, plus the baseline and replicate anchors, so degenerate spans
are visible directly.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.countsim import multinomial_thin
from vcc_model.metrics import (
    ContextData,
    baseline_anchor,
    baseline_counts,
    evaluate_context,
    replicate_anchor,
)
from vcc_model.synthesize import make_synthetic_challenge

ch = make_synthetic_challenge(n_genes=512, n_contexts=6, n_perts=16,
                              n_ctrl_cells=320, n_cells_per_pert=96, seed=11)
held = "line_4"
gene_names = [str(g) for g in ch.gene_names]
ctrl = np.asarray(ch.controls[held], dtype=np.float64)

rng = np.random.default_rng(7)
pick = rng.choice(ctrl.shape[0], size=80, replace=False)
base_cells = ctrl[pick]
ref_ctrl = ctrl[:300]

targets = sorted(t for (c, t) in ch.lfc_truth if c == held)[:12]
true_cells = {t: multinomial_thin(base_cells, np.exp2(ch.lfc_truth[(held, t)]), rng) for t in targets}
copy_cells = {t: base_cells.copy() for t in targets}
G = np.asarray(gene_names)

def show(name, cd):
    raw = evaluate_context(cd)
    pretty = {k: round(float(v), 4) for k, v in raw.items()}
    print(f"{name:>14}: {pretty}")
    return raw

print("=== ORACLE (truth vs truth) ===")
cd_o = ContextData(G, ref_ctrl, true_cells, true_cells)
show("oracle", cd_o)

print("=== CONTROL COPY ===")
cd_c = ContextData(G, ref_ctrl, true_cells, copy_cells)
raw_c = show("copy", cd_c)

print("=== MEAN RESPONSE (official baseline generator) ===")
mr = baseline_counts(cd_c, seed=3)
cd_m = ContextData(G, ref_ctrl, true_cells, mr)
show("mean_resp", cd_m)

b = baseline_anchor(cd_c, seed=3)
r = replicate_anchor(cd_c, n_splits=2, base_seed=3)
print("\nbaseline anchor :", {k: round(float(v), 4) for k, v in b.items()})
print("replicate anchor:", {k: round(float(v), 4) for k, v in r.items()})

# significance scale check
from vcc_model.countsim import normalize_cpm
from vcc_model.de_table import control_gate, de_table_for_group
gate = control_gate(normalize_cpm(ref_ctrl, 1e6))
t0 = de_table_for_group(true_cells[targets[0]], ref_ctrl, gate)
nsig = int(((t0.padj < 0.05) & ~np.isnan(t0.padj)).sum())
t1 = de_table_for_group(copy_cells[targets[0]], ref_ctrl, gate)
nsig_copy = int(((t1.padj < 0.05) & ~np.isnan(t1.padj)).sum())
print(f"\npert {targets[0]}: tested={int(gate.sum())} truth_sig={nsig} copy_sig={nsig_copy} "
      f"max|lfc|={np.nanmax(np.abs(t0.lfc)):.2f}")
