import sys
import os
import h5py
import numpy as np

out_path = sys.argv[1]
in_paths = sys.argv[2:]

tmp = out_path + '.tmp'

# Pass 1: gather metadata (shapes, nnz, obs, var)
file_info = []
total_rows = 0
total_nnz = 0
var_vals = None
var_mask = None
n_cols = 0
all_targets = []
all_contexts = []

for p in in_paths:
    print(f"scanning {os.path.basename(p)} ...", flush=True)
    with h5py.File(p, 'r') as f:
        shape = tuple(int(v) for v in f['X'].attrs['shape'])
        nnz = f['X']['data'].shape[0]
        file_info.append({'path': p, 'nrows': shape[0], 'ncols': shape[1], 'nnz': nnz})
        total_rows += shape[0]
        total_nnz += nnz
        n_cols = shape[1]

        cats_t = [c.decode() if isinstance(c, bytes) else str(c) for c in f['obs']['target_gene']['categories'][:]]
        codes_t = f['obs']['target_gene']['codes'][:]
        all_targets.extend([cats_t[c] for c in codes_t])

        cats_c = [c.decode() if isinstance(c, bytes) else str(c) for c in f['obs']['context']['categories'][:]]
        codes_c = f['obs']['context']['codes'][:]
        all_contexts.extend([cats_c[c] for c in codes_c])

        if var_vals is None:
            var_vals = f['var']['_index']['values'][:]
            var_mask = f['var']['_index']['mask'][:]

print(f"total: {total_rows} rows x {n_cols} cols, {total_nnz} nnz", flush=True)

# Pass 2: create output and stream data file by file
print("creating output file ...", flush=True)
with h5py.File(tmp, 'w') as h:
    xg = h.create_group("X")
    xg.attrs["encoding-type"] = "array"
    xg.attrs["encoding-version"] = "0.1.0"
    xg.attrs["shape"] = np.array([total_rows, n_cols])

    # Pre-allocate with chunked datasets for streaming writes
    ds_data = h.create_dataset("X/data", shape=(total_nnz,), dtype='float32',
                               chunks=(min(total_nnz, 1_000_000),), compression=None)
    ds_indices = h.create_dataset("X/indices", shape=(total_nnz,), dtype='int32',
                                  chunks=(min(total_nnz, 1_000_000),), compression=None)
    ds_indptr = h.create_dataset("X/indptr", shape=(total_rows + 1,), dtype='int64',
                                 chunks=(min(total_rows + 1, 100_000),), compression=None)

    data_offset = 0
    row_offset = 0
    indptr_offset = 0

    for fi in file_info:
        p = fi['path']
        print(f"writing {os.path.basename(p)} ({fi['nrows']} rows, {fi['nnz']} nnz) ...", flush=True)
        with h5py.File(p, 'r') as f:
            # Read CSR components in chunks to avoid memory spikes
            nnz = fi['nnz']
            nrows = fi['nrows']
            chunk_size = 5_000_000

            # Write data
            for start in range(0, nnz, chunk_size):
                end = min(start + chunk_size, nnz)
                ds_data[data_offset + start:data_offset + end] = f['X']['data'][start:end].astype(np.float32)

            # Write indices
            for start in range(0, nnz, chunk_size):
                end = min(start + chunk_size, nnz)
                ds_indices[data_offset + start:data_offset + end] = f['X']['indices'][start:end].astype(np.int32)

            data_offset += nnz

            # Write indptr with row offset
            indptr_raw = f['X']['indptr'][:]
            if indptr_offset == 0:
                # First file: write all indptr entries
                chunk_size_ip = 100_000
                for start in range(0, len(indptr_raw), chunk_size_ip):
                    end = min(start + chunk_size_ip, len(indptr_raw))
                    ds_indptr[indptr_offset + start:indptr_offset + end] = indptr_raw[start:end].astype(np.int64)
            else:
                # Subsequent files: skip first entry (it's 0), offset the rest
                shifted = indptr_raw[1:].astype(np.int64) + row_offset
                chunk_size_ip = 100_000
                for start in range(0, len(shifted), chunk_size_ip):
                    end = min(start + chunk_size_ip, len(shifted))
                    ds_indptr[indptr_offset + start:indptr_offset + end] = shifted[start:end]

            indptr_offset += nrows + 1
            row_offset += nrows

    # obs
    print("writing obs ...", flush=True)
    og = h.create_group("obs")
    og.attrs["encoding-type"] = "dataframe"
    og.attrs["encoding-version"] = "0.2.0"

    tg_cats = sorted(set(all_targets))
    tg_map = {c: i for i, c in enumerate(tg_cats)}
    tg = og.create_group("target_gene")
    tg.attrs["encoding-type"] = "categorical"
    tg.attrs["encoding-version"] = "0.1.0"
    tg.create_dataset("categories", data=np.array(tg_cats, dtype=object))
    tg.create_dataset("codes", data=np.array([tg_map[t] for t in all_targets], dtype=np.int8))

    ctx_cats = sorted(set(all_contexts))
    ctx_map = {c: i for i, c in enumerate(ctx_cats)}
    cg = og.create_group("context")
    cg.attrs["encoding-type"] = "categorical"
    cg.attrs["encoding-version"] = "0.1.0"
    cg.create_dataset("categories", data=np.array(ctx_cats, dtype=object))
    cg.create_dataset("codes", data=np.array([ctx_map[c] for c in all_contexts], dtype=np.int8))

    # Generate obs _index in chunks to avoid memory issue
    idx = og.create_group("_index")
    idx.attrs["encoding-type"] = "nullable-string-array"
    idx.attrs["encoding-version"] = "0.1.0"
    idx.attrs["na-value"] = "NaN"
    chunk_size_obs = 100_000
    ds_obs_vals = h.create_dataset("obs/_index/values", shape=(total_rows,), dtype=object,
                                   chunks=(chunk_size_obs,))
    ds_obs_mask = h.create_dataset("obs/_index/mask", shape=(total_rows,), dtype=bool,
                                   chunks=(chunk_size_obs,))
    for start in range(0, total_rows, chunk_size_obs):
        end = min(start + chunk_size_obs, total_rows)
        ds_obs_vals[start:end] = np.array([f"obs_{i}" for i in range(start, end)], dtype=object)
        ds_obs_mask[start:end] = False

    # var
    print("writing var ...", flush=True)
    vg = h.create_group("var")
    vg.attrs["encoding-type"] = "dataframe"
    vg.attrs["encoding-version"] = "0.2.0"
    vg.attrs["_index"] = "_index"
    vig = vg.create_group("_index")
    vig.attrs["encoding-type"] = "nullable-string-array"
    vig.attrs["encoding-version"] = "0.1.0"
    vig.attrs["na-value"] = "NaN"
    vig.create_dataset("values", data=var_vals)
    vig.create_dataset("mask", data=var_mask)

    for empty in ("layers", "obsm", "obsp", "raw", "uns", "varm", "varp"):
        eg = h.create_group(empty)
        eg.attrs["encoding-type"] = "dict"
        eg.attrs["encoding-version"] = "0.1.0"

os.replace(tmp, out_path)
print(f"wrote {out_path} rows={total_rows} nnz={total_nnz}", flush=True)
