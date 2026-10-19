"""Chunked merge of per-context submission h5ads into one panel h5ad.

Streams CSR buffers (data/indices/indptr) and obs categoricals via h5py —
never holds the full matrix in RAM. Output matches StreamingSubmissionWriter
format so downstream validation/pack works unchanged.
"""
import sys
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHUNK_ROWS = 2000


def _decode_strs(arr) -> list[str]:
    out = []
    for v in arr:
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def _read_categorical(obs_g, name: str) -> list[str]:
    g = obs_g[name]
    cats = _decode_strs(g["categories"][:])
    codes = g["codes"][:]
    return [cats[c] if c >= 0 else "" for c in codes]


def main() -> None:
    out = Path(sys.argv[1])
    ins = [Path(p) for p in sys.argv[2:]]
    assert len(ins) >= 2, "need >=2 inputs"

    # Probe first file for dtypes/axis
    with h5py.File(ins[0], "r") as f0:
        n_genes = int(tuple(f0["X"].attrs["shape"])[1])
        dt_data = f0["X"]["data"].dtype
        dt_idx = f0["X"]["indices"].dtype
        dt_ptr = f0["X"]["indptr"].dtype
        var_index_vals = f0["var"]["_index"]["values"][:]
        var_index_mask = f0["var"]["_index"]["mask"][:]

    total_rows, total_nnz = 0, 0
    all_tgt: list[str] = []
    all_ctx: list[str] = []
    counts = []
    for p in ins:
        with h5py.File(p, "r") as f:
            shape = tuple(int(v) for v in f["X"].attrs["shape"])
            assert shape[1] == n_genes, f"axis mismatch: {p}"
            nnz = int(f["X"]["indptr"][-1])
            n = shape[0]
            assert len(f["obs"]["target_gene"]["codes"][:]) == n, f"label/X mismatch: {p}"
            counts.append((n, nnz))
            total_rows += n
            total_nnz += nnz
            all_tgt.extend(_read_categorical(f["obs"], "target_gene"))
            all_ctx.extend(_read_categorical(f["obs"], "context"))
    assert len(all_tgt) == total_rows
    print(f"merge {len(ins)} files: rows={total_rows} nnz={total_nnz}", flush=True)

    if out.exists():
        out.unlink()
    with h5py.File(out, "w") as h:
        xg = h.create_group("X")
        xg.attrs["encoding-type"] = "csr_matrix"
        xg.attrs["encoding-version"] = "0.1.0"
        xg.attrs["shape"] = [total_rows, n_genes]
        d_data = xg.create_dataset("data", shape=(0,), maxshape=(None,),
                                   dtype=dt_data, compression="gzip")
        d_idx = xg.create_dataset("indices", shape=(0,), maxshape=(None,),
                                  dtype=dt_idx, compression="gzip")
        d_ptr = xg.create_dataset("indptr", shape=(1,), maxshape=(None,),
                                  dtype=dt_ptr, compression="gzip")
        d_ptr[0] = 0
        w_nnz, w_rows = 0, 0
        for p, (n, nnz) in zip(ins, counts):
            with h5py.File(p, "r") as f:
                # stream data/indices in row-blocks via indptr slices
                ptr = f["X"]["indptr"][:]
                start = 0
                while start < n:
                    end = min(start + CHUNK_ROWS, n)
                    a, b = int(ptr[start]), int(ptr[end])
                    if b > a:
                        d_data.resize((w_nnz + (b - a),))
                        d_idx.resize((w_nnz + (b - a),))
                        d_data[w_nnz:] = f["X"]["data"][a:b]
                        d_idx[w_nnz:] = f["X"]["indices"][a:b]
                    seg = (ptr[start:end + 1].astype(np.int64) - ptr[start]) + w_nnz
                    d_ptr.resize((w_rows + (end - start) + 1,))
                    d_ptr[w_rows + 1:w_rows + (end - start) + 1] = seg[1:]
                    w_nnz += b - a
                    w_rows += end - start
                    start = end
        assert w_rows == total_rows and w_nnz == total_nnz, (w_rows, w_nnz)
        xg.attrs["shape"] = [w_rows, n_genes]

        # obs categoricals
        obsg = h.create_group("obs")
        obsg.attrs["_index"] = "_index"
        obsg.attrs["column-order"] = np.array(["context", "target_gene"], dtype=object)
        obsg.attrs["encoding-type"] = "dataframe"
        obsg.attrs["encoding-version"] = "0.2.0"
        dt = h5py.special_dtype(vlen=str)

        def _write_cat(name: str, values: list[str]) -> None:
            cats = sorted(set(values))
            pos = {c: i for i, c in enumerate(cats)}
            codes = np.array([pos[v] for v in values], dtype=np.int32)
            g = obsg.create_group(name)
            g.attrs["encoding-type"] = "categorical"
            g.attrs["encoding-version"] = "0.2.0"
            g.attrs["ordered"] = False
            g.create_dataset("codes", data=codes)
            cd = g.create_dataset("categories", shape=(len(cats),), dtype=dt)
            if cats:
                cd[:] = cats

        _write_cat("target_gene", all_tgt)
        _write_cat("context", all_ctx)
        # _index strings (nullable-string-array like the writer emits)
        ig = obsg.create_group("_index")
        ig.attrs["encoding-type"] = "nullable-string-array"
        ig.attrs["encoding-version"] = "0.1.0"
        ig.attrs["na-value"] = "NaN"
        idx_vals = np.array([f"pred_{i}" for i in range(w_rows)], dtype=dt)
        ig.create_dataset("values", data=idx_vals)
        ig.create_dataset("mask", data=np.zeros(w_rows, dtype=bool))

        # var gene names (copied from first input)
        vg = h.create_group("var")
        vg.attrs["_index"] = "_index"
        vg.attrs["column-order"] = np.array([], dtype=np.float64)
        vg.attrs["encoding-type"] = "dataframe"
        vg.attrs["encoding-version"] = "0.2.0"
        vig = vg.create_group("_index")
        vig.attrs["encoding-type"] = "nullable-string-array"
        vig.attrs["encoding-version"] = "0.1.0"
        vig.attrs["na-value"] = "NaN"
        vig.create_dataset("values", data=var_index_vals)
        vig.create_dataset("mask", data=var_index_mask)
        for empty in ("layers", "obsm", "obsp", "raw", "uns", "varm", "varp"):
            eg = h.create_group(empty)
            eg.attrs["encoding-type"] = "dict"
            eg.attrs["encoding-version"] = "0.1.0"
    print(f"wrote {out} rows={w_rows} nnz={w_nnz}", flush=True)


if __name__ == "__main__":
    main()
