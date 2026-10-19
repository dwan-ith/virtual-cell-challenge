"""Reconstruct the gain->metric response curve from every scored submission.

v6/v9/v8 form a clean gain sweep (1.8 / 2.6 / 3.7) on the same pipeline.
v29 is gain 1.8 + ESM gene-conditioning + per-gene gain + zero_unseen + basal gate.
This computes marginal slopes so we can see which direction gain should move.
"""
# (label, gain, pds, nmae, reach, jac, fid, mse, overall)
SUBS = [
    ("v6",   1.8, 0.352,  0.071, 0.133, -0.014, -0.025, 0.000, 0.086),
    ("v9",   2.6, 0.421, -0.100, 0.139, -0.011, -0.023, 0.000, 0.071),
    ("v8",   3.7, 0.464, -0.470, 0.157, -0.005, -0.018, 0.000, 0.021),
    ("v29",  1.8, 0.397,  0.052, 0.142, -0.013, -0.024, 0.000, 0.0923),
    ("v32",  1.8, 0.361,  0.065, 0.124, -0.015, -0.026, 0.000, 0.085),
    ("v35",  1.8, 0.359,  0.045, 0.129, -0.014, -0.030, 0.000, 0.081),
    ("v36",  1.8, 0.243,  0.021, 0.055, -0.056, -0.477, 0.000, -0.036),
]

print("=== overall = mean(6 scaled members)? verify ===")
for lab, g, pds, nmae, reach, jac, fid, mse, ov in SUBS:
    calc = (pds + mse + nmae + reach + jac + fid) / 6.0
    print(f"  {lab:4s} reported {ov:+.4f}  recomputed {calc:+.4f}  {'OK' if abs(calc-ov)<0.002 else 'MISMATCH'}")

print("\n=== marginal slopes along the v6/v9/v8 gain sweep ===")
print("  (per +1.0 gain; positive = metric IMPROVES with more gain)")
for a, b in [(SUBS[0], SUBS[1]), (SUBS[1], SUBS[2])]:
    dg = b[1] - a[1]
    names = ["label", "gain", "pds", "nmae", "reach", "jac", "fid", "mse", "overall"]
    sl = {names[i]: (b[i] - a[i]) / dg for i in range(2, 9)}
    print(f"  {a[0]}({a[1]}) -> {b[0]}({b[1]})  dg={dg:+.1f}")
    for k in ["pds", "nmae", "reach", "jac", "fid", "overall"]:
        print(f"     d{k:7s}/dgain {sl[k]:+.4f}")

print("\n=== what the sweep says PDS is doing ===")
print("  pds rises monotonically with gain: 0.352 -> 0.421 -> 0.464")
print("  official pds is computed on EMITTED CELLS, so it is NOT scale-invariant:")
print("  bigger predicted effect => better signal-to-noise => better discrimination.")
print("  nmae falls monotonically: +0.071 -> -0.100 -> -0.470 (over-magnitude penalty)")
print("  mse is 0 for EVERY submission -> it contributes nothing to any score.")
print("  => the only real tension on the board is pds(needs more) vs nmae(needs less).")

print("\n=== linear extrapolation BELOW gain 1.8 (never measured) ===")
pd_, nd = (SUBS[1][2]-SUBS[0][2])/(SUBS[1][1]-SUBS[0][1]), (SUBS[1][3]-SUBS[0][3])/(SUBS[1][1]-SUBS[0][1])
rr, jr, fr = (SUBS[1][4]-SUBS[0][4])/(SUBS[1][1]-SUBS[0][1]), (SUBS[1][5]-SUBS[0][5])/(SUBS[1][1]-SUBS[0][1]), (SUBS[1][6]-SUBS[0][6])/(SUBS[1][1]-SUBS[0][1])
net = pd_ + nd + rr + jr + fr
print(f"  near 1.8-2.6 marginals: pds {pd_:+.3f} nmae {nd:+.3f} reach {rr:+.4f} jac {jr:+.4f} fid {fr:+.4f}")
print(f"  NET per +1 gain = {net:+.4f}  (negative => lower gain should raise overall)")
for g in [1.6, 1.4, 1.2, 1.0, 0.8]:
    d = g - 1.8
    proj = SUBS[3][8] + net * d / 1.0
    print(f"    gain {g:.1f} (d={d:+.1f}) -> projected overall ~ {proj:+.4f}")
print("  CAVEAT: assumes linearity and symmetric pds response. pds may fall FASTER")
print("  below 1.8 than the trend implies, because at low gain the emitted cells")
print("  approach the controls and sampling noise takes over.")