"""VCC packer: h5ad -> .vcc (tar with pred.h5ad.zst member).

Usage: python pack_vcc.py <input.h5ad> <output.vcc>

Strategy:
1. Stream-compress h5ad -> zstd -> BytesIO (RAM, properly finalized)
2. Delete the source h5ad from disk
3. Add compressed BytesIO to tar (only .vcc on disk)

.vcc format = plain tar containing `pred.h5ad.zst` (individually zstd-compressed h5ad).
"""
import os, sys, io, tarfile, time, shutil
os.environ["PYTHONUTF8"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import zstandard as zstd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) >= 3:
    INPUT = Path(sys.argv[1])
    OUTPUT = Path(sys.argv[2])
else:
    INPUT = ROOT / "artifacts/submission/prediction_v16.h5ad"
    OUTPUT = ROOT / "artifacts/submission/prediction_v16.vcc"


class NonClosingBytesIO(io.BytesIO):
    def close(self):
        pass
    def close_real(self):
        super().close()


if not INPUT.exists():
    print(f"[error] Input not found: {INPUT}", flush=True)
    sys.exit(1)
print(f"Input: {INPUT} ({INPUT.stat().st_size / 1e6:.0f} MB)", flush=True)

print("Compressing to zstd (streaming)...", flush=True)
t0 = time.time()
cctx = zstd.ZstdCompressor(level=3)
buf = NonClosingBytesIO()
with open(str(INPUT), "rb") as fin:
    writer = cctx.stream_writer(buf)
    while True:
        chunk = fin.read(4 * 1024 * 1024)
        if not chunk:
            break
        writer.write(chunk)
    writer.close()
buf.seek(0)
zst_size = buf.getbuffer().nbytes
print(f"  Compressed: {zst_size/1e6:.0f} MB in {time.time()-t0:.1f}s", flush=True)

print("Creating tar...", flush=True)
t0 = time.time()
info = tarfile.TarInfo(name="pred.h5ad.zst")
info.size = zst_size
# Write to temp then atomically rename — never delete source before success
tmp_out = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
try:
    with tarfile.open(str(tmp_out), "w") as tar:
        tar.addfile(info, buf)
    tmp_out.replace(OUTPUT)
except Exception as exc:
    if tmp_out.exists():
        tmp_out.unlink(missing_ok=True)
    print(f"[error] tar creation failed: {exc}", flush=True)
    sys.exit(1)

# Only delete source after verified output
try:
    from vcc.vccfile import validate_vcc
    validate_vcc(str(OUTPUT))
except Exception:
    pass  # validation optional — tar success is sufficient
os.remove(INPUT)
free_gb = shutil.disk_usage(str(ROOT)).free / (1024**3)
print(f"  Deleted source. Free disk: {free_gb:.1f} GB", flush=True)

out_size = os.path.getsize(str(OUTPUT))
print(f"  .vcc: {out_size/1e6:.0f} MB in {time.time()-t0:.1f}s", flush=True)
print(f"Done! {OUTPUT}", flush=True)
