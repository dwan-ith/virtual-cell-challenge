"""Second patch: retry file access in vcc prep._write_vcc.

On Windows, a freshly written multi-GB h5ad can stay locked for seconds
(antivirus scan / handle teardown), so the immediate open() for zstd
compression fails with WinError 32. Retry with backoff -- same semantics.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\aacer\AppData\Roaming\Python\Python313\site-packages\vcc\prep.py"
src = open(P, encoding="utf-8").read()

old = """        minimal.write_h5ad(h5ad_path)

        # zstd compress (vendored). Level/threads only affect the compressed
        # bytes, never the decompressed content the server reads.
        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        with open(h5ad_path, "rb") as src, open(zst_path, "wb") as dst:
            cctx.copy_stream(src, dst)
        os.remove(h5ad_path)"""
new = """        minimal.write_h5ad(h5ad_path)

        # zstd compress (vendored). Level/threads only affect the compressed
        # bytes, never the decompressed content the server reads.
        #
        # LOW-RAM/WINDOWS PATCH: a freshly written multi-GB h5ad can stay
        # briefly locked on Windows (AV scan / handle teardown); retry opens.
        import time as _time

        cctx = zstd.ZstdCompressor(level=3, threads=-1)
        last_exc = None
        for attempt in range(8):
            try:
                with open(h5ad_path, "rb") as src_f, open(zst_path, "wb") as dst:
                    cctx.copy_stream(src_f, dst)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
                _time.sleep(3.0 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        os.remove(h5ad_path)"""
assert old in src, "_write_vcc anchor missing"
src = src.replace(old, new)

open(P, "w", encoding="utf-8").write(src)
print("prep.py patched (_write_vcc retry)")
