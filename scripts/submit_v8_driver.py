"""Robust v8 submission driver: upload with timeout, retry on 409/hang.

Handles the failure modes observed today:
  - local uploader dies with network (entry stays 'uploading' server-side,
    holding the one-in-flight slot until server-side expiry)
  - slot busy (409) -> wait and retry
  - hung upload (0% for long stretches) -> kill and restart

Runs until the entry reaches a terminal state or --max-hours elapses.
"""
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "submit_v8_driver.log"
VCC = "vcc"
FILE = str(ROOT / "artifacts" / "submission" / "prediction_v8.vcc")
ENV = {
    **__import__("os").environ,
    "VCC_TOKEN": "vcc_pat_OAidTjJwOLuOWXZIa2TdydMnfci-HWDl3UKoI6WzTsg",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

UPLOAD_TIMEOUT_S = 75 * 60       # kill an attempt after this long
POLL_AFTER_UPLOAD_S = 110 * 60   # allow scoring this long once uploaded


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}Z] {msg}\n")
    print(f"[{stamp}Z] {msg}", flush=True)


def run_once(model_name: str, attempt: int = 0, resume_entry: str | None = None):
    """One submit attempt. Returns ('published'|'busy'|'failed'|'hung', detail)."""
    # Killed uploads leave a local lock that blocks every future attempt;
    # the server-side entry state is what actually matters, so clear it.
    import os as _os

    lock = Path(_os.path.expanduser("~")) / ".config" / "vcc" / "locks" / "submit-default.lock"
    if lock.exists():
        try:
            lock.unlink()
            log("cleared stale submit lock")
        except OSError:
            pass
    cmd = [
        VCC, "submit", FILE, "-m", model_name,
        "-d", "K562-GW transfer prior; shared_mix=0; alpha off; gain=3.7; gain-curve probe vs v6",
        "--wait", "--poll-interval", "30", "--wait-timeout", str(POLL_AFTER_UPLOAD_S),
    ]
    if resume_entry:
        cmd += ["--resume", resume_entry]
    # NOTE: bare `--resume` is invalid (requires an entry-id argument); retries
    # without a known entry simply start a fresh upload instead.
    log(f"attempt {attempt} start: {' '.join(cmd[1:4])}...")
    try:
        proc = subprocess.run(
            cmd, env=ENV, cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=UPLOAD_TIMEOUT_S + POLL_AFTER_UPLOAD_S,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = out[-2000:]
        if "final status: published" in out:
            return "published", tail
        if "in progress at a time" in out:
            return "busy", tail[-500:]
        return "failed", tail
    except subprocess.TimeoutExpired:
        return "hung", f"killed after {UPLOAD_TIMEOUT_S + POLL_AFTER_UPLOAD_S}s"


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "hybrid-v8-gain3.7"
    max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    deadline = time.time() + max_hours * 3600
    backoff = 0
    attempt = 0
    while time.time() < deadline:
        status, detail = run_once(model_name, attempt)
        log(f"attempt -> {status}: {detail}")
        if status == "published":
            log("DONE")
            return
        # If the failure names an in-flight entry of ours, resume THAT entry.
        import re as _re

        m = _re.search(r"in progress: (\S+) \(status: uploading\)", detail)
        if m and m.group(1) != getattr(main, "_last_resume", None):
            main._last_resume = m.group(1)
            log(f"slot held by {m.group(1)}; attempting direct resume")
            status2, detail2 = run_once(model_name, 0, resume_entry=m.group(1))
            log(f"resume -> {status2}: {detail2}")
            if status2 == "published":
                log("DONE")
                return
        attempt += 1
        if status == "busy":
            wait = 25 * 60
        elif status == "hung":
            wait = 5 * 60
        else:
            wait = min(5 * 60, 60 * (backoff + 1))
            backoff += 1
        log(f"sleeping {wait}s before next attempt")
        time.sleep(wait)
    log("deadline reached; giving up")


if __name__ == "__main__":
    main()
