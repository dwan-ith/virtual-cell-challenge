"""Project honest-eval RAW member values through OFFICIAL vcc2026 anchors.

The local thinned-truth evaluation has different statistical power than the
real scoring pipeline, so its per-member spans (r-b) differ wildly from the
official ones -- local 'scaled' values are NOT comparable across configs for
span-sensitive members (reach above all). What transfers across pipelines is
the RAW member value; this tool rescales raw values with the OFFICIAL anchors
from the metrics brief (section 8 midpoints) to project the leaderboard-style
score of each configuration.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Official anchors: (baseline b, replicate r) -- brief section 8 midpoints.
ANCHORS = {
    "pds_cosine": (0.500, 0.956),
    "expr_mse_unbiased_capped_norm": (0.989, 0.037),
    "de_wilcoxon_direction_fidelity_yield_raw": (0.514, 0.814),
    "de_wilcoxon_direction_reach_raw": (0.072, 0.968),
    "de_wilcoxon_sig_jaccard": (0.029, 0.400),
    "de_wilcoxon_lfc_nmae": (1.001, 0.400),
}
SHORT = {
    "pds_cosine": "pds",
    "expr_mse_unbiased_capped_norm": "mse",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid",
    "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
    "de_wilcoxon_lfc_nmae": "nmae",
}

for v in sys.argv[1:] or ["v6", "v7", "v8"]:
    try:
        r = json.load(open("artifacts/loco_%s.json" % v))
    except FileNotFoundError:
        continue
    ctx = r["contexts"]["vcc2025/H1"]
    calib = ctx["calibration"]
    parts, proj = [], {}
    for m, (b, rr) in ANCHORS.items():
        u = ctx["raw"].get(m)
        if not isinstance(u, (int, float)) or u != u:
            continue
        if m == "expr_mse_unbiased_capped_norm":
            u = min(max(u, 0.0), 1.0)
        if m == "de_wilcoxon_lfc_nmae":
            u = max(u, -6.0)
        s = (u - b) / (rr - b)
        proj[SHORT[m]] = s
        parts.append(s)
    overall = sum(parts) / len(parts) if parts else float("nan")
    print("=== %s gain=%.2f q=%.2f tau=%.2f" % (
        v, calib["gain"], calib["q_threshold"], calib["threshold"]))
    print("   " + "  ".join("%s=%+.3f" % (k, val) for k, val in proj.items()))
    print("   PROJECTED OFFICIAL OVERALL = %+.4f" % overall)
