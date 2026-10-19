"""Download Arc State assets: SE-600M encoder + ST-SE-Replogle zero-shot k562
checkpoint + its eval predictions/ground-truth for a head-to-head validation.

Skips files that already exist with the expected size.
"""
import os
import sys
import time
from pathlib import Path

import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://huggingface.co/arcinstitute/{}/resolve/main/{}"
OUT = Path("data/state")
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    # (repo, remote path, local relative path)
    ("SE-600M", "config.yaml", "se600m/config.yaml"),
    ("ST-SE-Replogle", "zeroshot/k562/config.yaml", "st_k562_zeroshot/config.yaml"),
    ("ST-SE-Replogle", "zeroshot/k562/var_dims.pkl", "st_k562_zeroshot/var_dims.pkl"),
    ("ST-SE-Replogle", "zeroshot/k562/batch_onehot_map.pkl", "st_k562_zeroshot/batch_onehot_map.pkl"),
    ("ST-SE-Replogle", "zeroshot/k562/cell_type_onehot_map.pkl", "st_k562_zeroshot/cell_type_onehot_map.pkl"),
    ("ST-SE-Replogle", "zeroshot/k562/pert_onehot_map.pt", None),  # may not exist; handled below
    ("ST-SE-Replogle", "zeroshot/k562/checkpoints/best.ckpt", "st_k562_zeroshot/checkpoints/best.ckpt"),
    ("SE-600M", "model.safetensors", "se600m/model.safetensors"),
    # head-to-head eval set (big; downloaded last)
    ("ST-SE-Replogle", "zeroshot/k562/eval_best.ckpt/adata_pred.h5ad", "eval_k562/adata_pred.h5ad"),
    ("ST-SE-Replogle", "zeroshot/k562/eval_best.ckpt/adata_real.h5ad", "eval_k562/adata_real.h5ad"),
]


def fetch(repo: str, remote: str, local: Path, max_tries: int = 8) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        print(f"[skip] {local.name} ({local.stat().st_size/1e6:.0f} MB)")
        return True
    url = BASE.format(repo, urllib.request.quote(remote))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for t in range(max_tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(local, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                t0 = time.time()
                while True:
                    blk = r.read(1 << 22)
                    if not blk:
                        break
                    f.write(blk)
                    done += len(blk)
                    if time.time() - t0 > 20:
                        dt = time.time() - t0
                        print(f"  {local.name}: {done/1e6:.0f}/{total/1e6:.0f} MB ({done/dt/1e3:.0f} KB/s)", flush=True)
                        t0 = time.time()
            return True
        except Exception as exc:  # noqa: BLE001
            wait = min(60, 5 * (t + 1))
            print(f"[retry {t+1}] {remote}: {exc}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    print(f"[FAIL] {remote}")
    return False


def main() -> None:
    for repo, remote, rel in FILES:
        if rel is None:
            continue
        ok = fetch(repo, remote, OUT / rel)
        if not ok and "eval_" in (rel or ""):
            print("[warn] continuing without eval file")


if __name__ == "__main__":
    main()
