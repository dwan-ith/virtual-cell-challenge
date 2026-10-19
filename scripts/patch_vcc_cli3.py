"""Extend _write_vcc retry window: AV locks on 3.9 GB fresh files can exceed
the previous ~108 s budget on slow machines."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\aacer\AppData\Roaming\Python\Python313\site-packages\vcc\prep.py"
src = open(P, encoding="utf-8").read()

old = """        last_exc = None
        for attempt in range(8):
            try:
                with open(h5ad_path, "rb") as src_f, open(zst_path, "wb") as dst:
                    cctx.copy_stream(src_f, dst)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
                _time.sleep(3.0 * (attempt + 1))"""
new = """        last_exc = None
        for attempt in range(60):
            try:
                with open(h5ad_path, "rb") as src_f, open(zst_path, "wb") as dst:
                    cctx.copy_stream(src_f, dst)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
                _time.sleep(10.0)"""
assert old in src, "retry anchor missing"
src = src.replace(old, new)

open(P, "w", encoding="utf-8").write(src)
print("prep.py retry window extended (60 x 10s)")
