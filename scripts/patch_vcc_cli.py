"""Patch the installed vcc CLI's prep.py for low-RAM machines.

Three nnz-proportional allocations OOM a 16 GB machine on full-panel
submissions (1.7e9 stored entries):
  1. _all_values_integer: np.modf over the whole data vector (2 x 6.4 GiB)
  2. guess_is_lognorm:    same pattern (only reached on --no-require-counts;
                          patched with exact original control flow preserved)
  3. run_prep:            scipy .astype() copies even for matching dtype
All patches preserve semantics with chunked / no-copy equivalents.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\aacer\AppData\Roaming\Python\Python313\site-packages\vcc\prep.py"
src = open(P, encoding="utf-8").read()

# ---- Patch 1: chunked integrality check in _all_values_integer --------------
old1 = """    if data.size == 0:
        return True
    frac, _ = np.modf(data)
    return not bool(np.any(np.abs(frac) > 1e-6))"""
new1 = """    if data.size == 0:
        return True
    # LOW-RAM PATCH: np.modf over the full stored vector allocates two
    # nnz-sized float arrays (~13 GiB at 1.7e9 entries) and OOMs 16 GB
    # machines. Chunked equivalent, identical semantics and tolerance.
    step = 1 << 26
    for s0 in range(0, data.size, step):
        blk = data[s0 : s0 + step]
        frac = np.modf(blk)[0]
        if bool(np.any(np.abs(frac) > 1e-6)):
            return False
    return True"""
assert old1 in src, "patch1 anchor missing"
src = src.replace(old1, new1)

# ---- Patch 2: chunked modf inside guess_is_lognorm (flow-preserving) --------
old2 = """        if getattr(X, "format", None) not in ("csr", "csc"):
            X = X.tocsr()
        frac, _ = np.modf(X.data)"""
new2 = """        if getattr(X, "format", None) not in ("csr", "csc"):
            X = X.tocsr()
        # LOW-RAM PATCH: chunked equivalent of np.modf(X.data); frac is only
        # consumed via np.any(frac > epsilon), so test per chunk.
        _data = X.data
        has_decimals = False
        for s0 in range(0, _data.size, 1 << 26):
            if bool(np.any(np.modf(_data[s0 : s0 + (1 << 26)])[0] > epsilon)):
                has_decimals = True
                break
        if not has_decimals:
            logger.info("Data appears to be integer counts (no decimal values detected).")
            return False
        max_val = float(_data.max()) if _data.size else 0.0
        min_val = float(_data.min()) if _data.size else 0.0
        if min_val < 0:
            raise PrepError(
                f"Invalid scale: minimum value {min_val:.2f} is negative. "
                "Counts and log1p-normalized data must be >= 0."
            )
        if validate and max_val >= max_threshold:
            raise PrepError(
                f"Invalid scale: maximum value {max_val:.2f} exceeds the log1p threshold "
                f"of {max_threshold}. Expected log1p-normalized values in [0, {max_threshold}); "
                "values this large look like raw counts or a mixed scale. If your data is "
                "raw integer counts, prep will normalize it for you; if it is intentionally "
                "discrete, pass --allow-discrete."
            )
        logger.info(
            "Data appears to be log1p-normalized (decimals detected, range "
            f"[{min_val:.2f}, {max_val:.2f}])."
        )
        return True"""
assert old2 in src, "patch2 anchor missing"
src = src.replace(old2, new2)

# ---- Patch 3: no-copy astype when dtype already matches ---------------------
old3 = "    new_x = X.astype(dtype) if issparse(X) else csr_matrix(np.asarray(X).astype(dtype))"
new3 = """    if issparse(X):
        # LOW-RAM PATCH: scipy astype copies by default even for a matching
        # dtype (~13 GiB extra on a full-panel submission); avoid the copy.
        new_x = X if X.dtype == np.dtype(dtype) else X.astype(dtype)
    else:
        new_x = csr_matrix(np.asarray(X).astype(dtype))"""
assert old3 in src, "patch3 anchor missing"
src = src.replace(old3, new3)

open(P, "w", encoding="utf-8").write(src)
print("prep.py patched (3 sites)")
