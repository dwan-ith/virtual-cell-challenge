"""Empirical direction-transfer ceiling between source contexts.

For every context pair sharing targets, measures on shared (target, gene)
entries with |lfc| above a significance-like threshold on BOTH sides:
  - sign agreement rate      (upper bound for fid/reach-style metrics)
  - Pearson / Spearman rho   (transferability of magnitudes)
Also reports same-line cross-dataset pairs (batch/protocol reliability) as a
reference point separating "measurement noise" from "true context change".
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.dataset import PseudobulkStore

STORES = [
    ("k562_gw", "artifacts/stores/replogle_gw_k562.npz"),
    ("k562_ess", "artifacts/stores/replogle_ess_k562.npz"),
    ("rpe1_bulk", "artifacts/stores/replogle_gw_rpe1.npz"),
    ("tahoe_hepg2", "artifacts/stores/tahoe_hepg2.npz"),
    ("tahoe_jurkat", "artifacts/stores/tahoe_jurkat.npz"),
    ("tahoe_k562", "artifacts/stores/tahoe_k562.npz"),
    ("tahoe_rpe1", "artifacts/stores/tahoe_rpe1.npz"),
    ("h1_exact", "artifacts/stores/vcc2025_h1.npz"),
]
THRESH = 0.25  # |lfc| both sides -> "biologically called" gene


def main() -> None:
    data: dict[str, dict[str, np.ndarray]] = {}
    for name, path in STORES:
        s = PseudobulkStore.load(path)
        data[name] = {t: s.lfc[i].astype(np.float64) for i, (_c, t) in enumerate(s.pairs)}
        print(f"{name}: {len(data[name])} targets")

    print(f"\nsign agreement on shared targets (both |lfc|>{THRESH}):\n")
    hdr = f"{'pair':<26}{'n_tgt':>6}{'sign_acc':>9}{'pearson':>9}{'spearman':>9}"
    print(hdr)

    rows = []
    for a, b in combinations([n for n, _ in STORES], 2):
        da, db = data[a], data[b]
        shared = sorted(set(da) & set(db))
        if len(shared) < 10:
            continue
        sa_tot = n_called = 0
        ps, ss = [], []
        for t in shared:
            va, vb = da[t], db[t]
            m = (np.abs(va) > THRESH) & (np.abs(vb) > THRESH) & np.isfinite(va) & np.isfinite(vb)
            if m.sum() < 20:
                continue
            sa_tot += float((np.sign(va[m]) == np.sign(vb[m])).mean()) * m.sum()
            n_called += int(m.sum())
            nz_a, nz_b = np.abs(va) > 0.05, np.abs(vb) > 0.05
            mask_any = nz_a | nz_b
            if mask_any.sum() > 100:
                ps.append(pearsonr(va[mask_any], vb[mask_any])[0])
                ss.append(spearmanr(va[mask_any], vb[mask_any]).statistic)
        if n_called == 0:
            continue
        rows.append((f"{a} vs {b}", len(shared), sa_tot / n_called,
                     float(np.mean(ps)) if ps else np.nan,
                     float(np.mean(ss)) if ss else np.nan))
        print(f"{rows[-1][0]:<26}{len(shared):>6}{rows[-1][2]:>9.3f}"
              f"{rows[-1][3]:>9.3f}{rows[-1][4]:>9.3f}")


if __name__ == "__main__":
    main()
