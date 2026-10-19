import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
r = json.load(open("artifacts/loco_v6.json"))
ctx = r["contexts"]["vcc2025/H1"]
print("weights:", r["weights"], "| n_pairs:", ctx["n_pairs"])
print("calibration:", ctx["calibration"])
print()
short = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
    "de_wilcoxon_lfc_nmae": "nmae",
}


def f(x):
    return ("%9.4f" % x) if isinstance(x, (int, float)) else "     None"


print("%6s %9s %9s %9s %9s" % ("member", "raw", "base_b", "rep_r", "scaled"))
for m, s in short.items():
    print(
        "%6s %9s %9s %9s %9s"
        % (s, f(ctx["raw"].get(m)), f(ctx["baseline_anchor"].get(m)),
           f(ctx["replicate_anchor"].get(m)), f(ctx["scaled"].get(m)))
    )
print()
orc = ctx.get("oracle_scaled", {})
print("oracle scaled:", {k: round(v, 3) for k, v in orc.items()})
