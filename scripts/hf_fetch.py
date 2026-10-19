"""Persistent resumable downloader for large HF files on flaky networks.

Streams with Range resumes: keeps partial .part files and continues from the
last received byte. Retries forever until --max-hours.
"""
import argparse
import sys
import time
from pathlib import Path

import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fetch_resumable(url: str, local: Path, max_hours: float) -> bool:
    part = local.with_suffix(local.suffix + ".part")
    expected = local.stat().st_size if local.exists() else None
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        done = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if done:
            req.add_header("Range", f"bytes={done}-")
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                total = int(r.headers.get("Content-Length") or 0) + done
                mode = "wb" if not done or "Content-Range" in r.headers else "ab"
                t0 = time.time()
                with open(part, mode) as f:
                    while True:
                        blk = r.read(1 << 22)
                        if not blk:
                            break
                        f.write(blk)
                        done += len(blk)
                        if time.time() - t0 > 30:
                            print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB "
                                  f"({done/max(time.time()-t0+1e-9,1)/1e3:.0f} KB/s)", flush=True)
                            t0 = time.time()
            # completed stream
            if local.exists() and expected and part.stat().st_size == expected:
                print("[skip-already-done]")
            part.rename(local)
            print(f"[done] {local.name} ({local.stat().st_size/1e6:.0f} MB)", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001
            got = part.stat().st_size if part.exists() else 0
            print(f"[retry] {exc}; have {got/1e6:.0f} MB; sleeping 20s", flush=True)
            time.sleep(20)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=24.0)
    args = ap.parse_args()
    ok = fetch_resumable(args.url, Path(args.out), args.max_hours)
    print("SUCCESS" if ok else "GAVE UP")


if __name__ == "__main__":
    main()
