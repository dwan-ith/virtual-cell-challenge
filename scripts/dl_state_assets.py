"""Resumable background downloader for STATE assets (HF CDN throttles curl)."""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from huggingface_hub import hf_hub_download

TARGETS = [
    ("arcinstitute/ST-SE-Replogle", "zeroshot/k562/pert_onehot_map.pt",
     "data/state/st_k562_zeroshot/pert_onehot_map.pt"),
    ("arcinstitute/SE-600M", "config.yaml", "data/state/se600m/config.yaml"),
    ("arcinstitute/SE-600M", "model.safetensors", "data/state/se600m/model.safetensors"),
]

for repo, remote, local in TARGETS:
    dest = Path(local)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {local} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    for attempt in range(1, 9):
        try:
            t0 = time.time()
            got = hf_hub_download(
                repo_id=repo,
                filename=remote,
                local_dir=str(dest.parent),
            )
            size = Path(got).stat().st_size
            print(f"[ok] {remote} -> {local} ({size/1e6:.1f} MB in {time.time()-t0:.0f}s)", flush=True)
            ok = True
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[retry {attempt}] {remote}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(min(30, 5 * attempt))
    if not ok:
        print(f"[FAIL] {remote}", flush=True)
print("[done]", flush=True)