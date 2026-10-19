"""Submit v10 and v11 sequentially."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {
    **__import__("os").environ,
    "VCC_TOKEN": "vcc_pat_OAidTjJwOLuOWXZIa2TdydMnfci-HWDl3UKoI6WzTsg",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}
POLL_TIMEOUT = 110 * 60

submissions = [
    ("v10", str(ROOT / "artifacts" / "submission" / "prediction_v10.vcc"), "hybrid-v10-gain2.1-q0.35"),
    ("v11", str(ROOT / "artifacts" / "submission" / "prediction_v11.vcc"), "hybrid-v11-gain2.1-q0.35"),
]

for label, vcc_file, model_name in submissions:
    print(f"\n{'='*60}\nSubmitting {label}: {model_name}\n{'='*60}", flush=True)
    cmd = [
        "vcc", "submit", vcc_file, "-m", model_name,
        "-d", f"K562 transfer prior; shared_mix=0; alpha off; gain=2.1; q_threshold=0.35; {label}",
        "--wait", "--poll-interval", "30", "--wait-timeout", str(POLL_TIMEOUT),
    ]
    try:
        proc = subprocess.run(
            cmd, env=ENV, cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=POLL_TIMEOUT + 120,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out[-3000:], flush=True)
        if "final status: published" in out:
            print(f"\n>>> {label} PUBLISHED OK <<<", flush=True)
        else:
            print(f"\n>>> {label} may have issues — check output above <<<", flush=True)
    except subprocess.TimeoutExpired:
        print(f"\n>>> {label} TIMEOUT after {POLL_TIMEOUT}s — may still be scoring <<<", flush=True)
    except Exception as exc:
        print(f"\n>>> {label} ERROR: {exc} <<<", flush=True)
    
    if label != submissions[-1][0]:
        print(f"\nWaiting 60s before next submission...", flush=True)
        time.sleep(60)

print("\nDone submitting all entries.", flush=True)
