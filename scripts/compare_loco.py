import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
short = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
    "de_wilcoxon_lfc_nmae": "nmae",
}
for v in ["v6", "v7", "v8"]:
    r = json.load(open("artifacts/loco_%s.json" % v))
    ctx = r["contexts"]["vcc2025/H1"]
    print("=== %s: calib=%s" % (v, ctx["calibration"]))
    row = []
    for m, s in short.items():
        sc = ctx["scaled"].get(m)
        row.append("%s=%.2f" % (s, sc) if isinstance(sc, (int, float)) else "%s=None" % s)
    print("   " + "  ".join(row) + "  overall=%.3f" % ctx["scaled"]["overall"])
