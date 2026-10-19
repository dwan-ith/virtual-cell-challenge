"""SYNTHETIC-TRUTH GATES for emitter variants. NOT A LEADERBOARD PREDICTOR.

v42 board result: 0.09051, versus v41 0.08928 and v38 0.09233 -- no real gain,
even though the synthetic six-metric comparison below showed the depth-dependent
emission driving expr_mse 0.40 -> 0.00 while holding fid/jac. Synthetic truth
built from the SAME control pool and SAME lfc vector for both "real" and
"predicted" cells makes the effect trivially recoverable, which inflates every
metric and hides transfer error. Conclusion recorded here so nobody reuses it
as a submission gate: use it only to catch degenerate emitters (sparsity,
row-depth corruption, zero-call collapse), never to tune a knob for the board.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import anndata as ad
from scipy.stats import mannwhitneyu

from vcc_model.countsim import (
    apply_promoter_ceiling,
    bulk_probability_from_lfc,
    poisson_thin,
    repair_bulk_totals,
)
from vcc_model.dataset import detect_columns, harmonize_to_axis, load_gene_names_csv

rng = np.random.default_rng(11)
genes = load_gene_names_csv("data/vcc2026/gene_names.csv")
G = len(genes)
a = ad.read_h5ad("data/vcc2026/context_A.h5ad")
tcol, _ = detect_columns(a.obs)
lab = a.obs[tcol].astype(str).to_numpy()
X = harmonize_to_axis(a.X, a.var_names, genes).tocsr()
ctrl = np.asarray(X[np.flatnonzero(lab == "non-targeting")].todense(), dtype=np.int64)
del a, X
REF_N = 2000
ref = ctrl[rng.choice(ctrl.shape[0], REF_N, replace=False)]
ctrl_prob = ref.sum(axis=0).astype(np.float64)
ctrl_prob /= ctrl_prob.sum()
print(f"controls: ref {ref.shape}, genes/cell {(ref > 0).sum(1).mean():.0f}", flush=True)

base = ctrl[rng.choice(ctrl.shape[0], 800, replace=False)]
true_genes = rng.choice(G, size=40, replace=False)
results = {}
for amp in [0.0, 0.4, 1.0]:
    lfc = np.zeros(G)
    lfc[true_genes] = rng.choice([-1.0, 1.0], size=len(true_genes)) * amp
    real = poisson_thin(base[:400], np.exp2(lfc), rng)
    pred_plain = poisson_thin(base[400:], np.exp2(lfc), rng)
    q = bulk_probability_from_lfc(ctrl_prob, lfc, bulk_gain=1.0)
    y = poisson_thin(base[400:], np.exp2(lfc), rng)
    d = y.sum(axis=1, dtype=np.int64)
    pred_repair = repair_bulk_totals(y, d[:, None] * q[None, :])
    true_bulk = real.sum(axis=0).astype(np.float64)
    true_bulk /= true_bulk.sum()

    def bulk_l1(E):
        b = E.sum(axis=0).astype(np.float64)
        return float(np.abs(b / b.sum() - true_bulk).sum())

    def calls(E):
        # Fast screen on true genes plus a random background subset.
        cand = np.union1d(true_genes, rng.choice(G, size=1500, replace=False))
        n = 0
        for k in cand:
            if E[:, k].mean() == 0 and ref[:, k].mean() == 0:
                continue
            if mannwhitneyu(E[:, k], ref[:, k], alternative="two-sided").pvalue < 0.05:
                n += 1
        return n

    def sign_acc(E):
        m_real = real.mean(axis=0)
        m_pred = E.mean(axis=0)
        s = np.sign(np.log2((m_pred + 1e-9) / (ref.mean(axis=0) + 1e-9)))
        t = np.sign(np.log2((m_real + 1e-9) / (ref.mean(axis=0) + 1e-9)))
        return float((s[true_genes] == t[true_genes]).mean())

    results[amp] = {
        "bulk_plain": bulk_l1(pred_plain),
        "bulk_repair": bulk_l1(pred_repair),
        "calls_plain": calls(pred_plain),
        "calls_repair": calls(pred_repair),
        "sign_plain": sign_acc(pred_plain),
        "sign_repair": sign_acc(pred_repair),
        "det_repair": float((pred_repair > 0).sum(1).mean()),
    }
    print(f"amp={amp}: {results[amp]}", flush=True)

# Promoter helper unit check.
p = ctrl_prob.copy()
pairs = {"FAKE": [(1, 100.0), (2, 5000.0)]}
out = apply_promoter_ceiling(p, ctrl_prob, "FAKE", pairs, fraction=0.15)
assert abs(out.sum() - 1.0) < 1e-9
assert out[1] <= ctrl_prob[1] * 0.15 + 1e-12
assert out[2] <= ctrl_prob[2] + 1e-12
print("promoter helper ok", flush=True)

ok = True
ok &= results[1.0]["bulk_repair"] < results[1.0]["bulk_plain"]
ok &= results[0.0]["calls_repair"] < 800
ok &= results[1.0]["calls_repair"] > results[0.0]["calls_repair"]
ok &= results[1.0]["sign_repair"] >= 0.6
print(f"GATE {'PASS' if ok else 'FAIL'}", flush=True)
if not ok:
    raise SystemExit(1)

# Exact local six-metric comparison on a small synthetic panel.
from vcc_model.metrics import ContextData, evaluate_context

SUB_G = 2500
sub_true = np.union1d(true_genes, rng.choice(G, size=SUB_G - len(true_genes), replace=False))
sub_true = np.sort(sub_true)
ref_s = ref[:, sub_true]
syn_names = [f"SYN{i}" for i in range(5)]
real_pert, pred_plain_map, pred_repair_map, syn_lfc = {}, {}, {}, {}
for si, nm in enumerate(syn_names):
    slfc = np.zeros(len(sub_true))
    tg = rng.choice(len(sub_true), size=25, replace=False)
    slfc[tg] = rng.choice([-1.0, 1.0], size=len(tg)) * (0.5 + 0.2 * si)
    syn_lfc[nm] = slfc
    full_lfc = np.zeros(G)
    full_lfc[sub_true] = slfc
    rb = ctrl[rng.choice(ctrl.shape[0], 200, replace=False)][:, sub_true]
    pb = ctrl[rng.choice(ctrl.shape[0], 200, replace=False)][:, sub_true]
    real_pert[nm] = poisson_thin(rb, np.exp2(full_lfc[sub_true]), rng)
    plain = poisson_thin(pb, np.exp2(full_lfc[sub_true]), rng)
    pred_plain_map[nm] = plain
    q = bulk_probability_from_lfc(
        (ref_s.sum(axis=0) / ref_s.sum()).astype(np.float64),
        full_lfc[sub_true],
        bulk_gain=1.0,
    )
    d = plain.sum(axis=1, dtype=np.int64)
    pred_repair_map[nm] = repair_bulk_totals(plain, d[:, None] * q[None, :])

cd_plain = ContextData(np.asarray(genes)[sub_true], ref_s, real_pert, pred_plain_map)
cd_repair = ContextData(np.asarray(genes)[sub_true], ref_s, real_pert, pred_repair_map)
m_plain = evaluate_context(cd_plain, n_control_subsample=2000, seed=0)
m_repair = evaluate_context(cd_repair, n_control_subsample=2000, seed=0)
print("plain:", {k: round(float(v), 4) for k, v in m_plain.items()}, flush=True)
print("repair:", {k: round(float(v), 4) for k, v in m_repair.items()}, flush=True)

for boost in [0.3, 0.6, 1.0]:
    bmap = {}
    for si, nm in enumerate(syn_names):
        slfc = syn_lfc[nm]
        pb = ctrl[rng.choice(ctrl.shape[0], 200, replace=False)][:, sub_true]
        d0 = pb.sum(axis=1, dtype=np.float64)
        factor = 1.0 + boost * (d0 / max(d0.mean(), 1.0) - 1.0)
        mult = np.exp2(np.clip(slfc[None, :] * factor[:, None], -6.0, 6.0))
        bmap[nm] = poisson_thin(pb, mult, rng)
    mb = evaluate_context(
        ContextData(np.asarray(genes)[sub_true], ref_s, real_pert, bmap),
        n_control_subsample=2000, seed=0,
    )
    print(f"boost={boost}:", {k: round(float(v), 4) for k, v in mb.items()}, flush=True)
