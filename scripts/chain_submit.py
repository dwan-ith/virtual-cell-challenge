"""Chain: wait for LOCO config -> fit full model -> stream submission -> prep.

Runs unattended. Steps:
  1. Poll for artifacts/loco_config.json (LOCO-selected weights + tuned
     calibration written by train_eval_loco.py --config-out).
  2. Fit HybridPredictor on ALL stores (holdout included) and predict every
     (context A/B/C, target) pair straight to disk via the streaming writer;
     light-validate against the full 2026 rules.
  3. `vcc prep --dry-run` if the CLI is authenticated.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", buffering=1)

ROOT = Path(__file__).resolve().parents[1]
STORES = [
    "replogle_gw_k562", "replogle_ess_k562", "replogle_gw_rpe1",
    "tahoe_hepg2", "tahoe_jurkat", "tahoe_k562", "tahoe_rpe1", "vcc2025_h1",
]
CONFIG_SRC = ROOT / "artifacts/loco_config.json"
SUB_DIR = ROOT / "artifacts/submission"


def wait_for(path: Path, timeout_s: float = 12 * 3600) -> bool:
    t0 = time.time()
    while not path.exists() and time.time() - t0 < timeout_s:
        time.sleep(60)
    return path.exists()


def main() -> None:
    print(f"[chain] waiting for {CONFIG_SRC.name} ...", flush=True)
    if not wait_for(CONFIG_SRC):
        print("[chain] TIMEOUT waiting for LOCO config; aborting", flush=True)
        return

    cfg = json.loads(CONFIG_SRC.read_text())
    print(f"[chain] LOCO config: {cfg}", flush=True)

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/make_submission.py"),
        "--stores", *[str(ROOT / f"artifacts/stores/{s}.npz") for s in STORES],
        "--controls-dir", str(ROOT / "data/vcc2026"),
        "--out", str(SUB_DIR / "prediction.h5ad"),
        "--config", str(CONFIG_SRC),
        "--epochs", "200",
        "--seed", "0",
        "--decoder", "poisson",
    ]
    print("[chain] fitting full model + streaming submission ...", flush=True)
    with open(ROOT / "artifacts/submission_build.log", "w", encoding="utf-8") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[chain] make_submission rc={rc}", flush=True)
    if rc != 0:
        return

    # Dry-run packaging through the official CLI when available.
    import shutil

    vcc = shutil.which("vcc")
    if vcc:
        prep_cmd = [
            vcc, "prep", str(SUB_DIR / "prediction.h5ad"),
            "-g", str(ROOT / "data/vcc2026/gene_names.csv"),
            "--perts", str(ROOT / "data/vcc2026/pert_counts.csv"),
            "-o", str(SUB_DIR / "prediction.vcc"),
            "--dry-run",
        ]
        rc2 = subprocess.call(prep_cmd)
        print(f"[chain] vcc prep dry-run rc={rc2}", flush=True)


if __name__ == "__main__":
    main()
