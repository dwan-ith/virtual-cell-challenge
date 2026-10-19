"""Convert submission h5ad /X/data to float32 (values unchanged, chunked).

The official `vcc prep` casts X to float32 (-e 32); feeding it int32 forces a
~6.4 GiB allocation per 1e9 nnz that OOMs a 16 GB machine. Whole-number floats
are explicitly legal ("whole numbers stored as floats are fine"), and a
matching dtype turns prep's cast into a no-op.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNK = 1 << 23


def copy_dataset(name: str, ds: h5py.Dataset, fout: h5py.File) -> None:
    comp = ds.compression
    cargs = {"chunks": True, "compression": comp} if comp else {"chunks": True}
    out = fout.create_dataset(name, shape=ds.shape, dtype=ds.dtype, **cargs)
    for k, v in ds.attrs.items():
        out.attrs[k] = v
    if ds.shape[0] == 0:
        return
    step = max(1, (1 << 23) // max(1, int(np.prod(ds.shape[1:]) or 1)))
    for start in range(0, ds.shape[0], step):
        out[start : start + step] = ds[start : start + step]


def copy_group_deep(name: str, obj, fout: h5py.File) -> None:
    if isinstance(obj, h5py.Dataset):
        copy_dataset(name, obj, fout)
    else:
        if name not in fout:
            grp = fout.create_group(name)
        else:
            grp = fout[name]
        for k, v in obj.attrs.items():
            grp.attrs[k] = v


def convert(path: str) -> None:
    src = Path(path)
    tmp = src.with_suffix(".f32.h5ad")
    print(f"[f32] {src.name} ->", flush=True)
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        # root attrs
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        def visitor(name, obj):
            if name == "X" or name.startswith("X/"):
                return
            copy_group_deep(name, obj, fout)

        fin.visititems(visitor)
        # top-level groups that may be empty (visititems skips empty groups)
        for k in fin:
            if k == "X":
                continue
            if k not in fout:
                grp = fout.create_group(k)
                for ak, av in fin[k].attrs.items():
                    grp.attrs[ak] = av

        xg_src = fin["X"]
        xg = fout.create_group("X")
        for k, v in xg_src.attrs.items():
            xg.attrs[k] = v
        comp = xg_src["data"].compression
        n_data = int(xg_src["data"].shape[0])
        cargs = (
            {"chunks": (min(1 << 20, max(n_data, 1)),), "compression": comp}
            if comp else {"chunks": (min(1 << 20, max(n_data, 1)),)}
        )
        d_out = xg.create_dataset("data", shape=(n_data,), dtype="float32", **cargs)
        n_idx = int(xg_src["indices"].shape[0])
        ix_out = xg.create_dataset(
            "indices", shape=(n_idx,), dtype="int32",
            chunks=(min(1 << 20, max(n_idx, 1)),), compression=comp,
        ) if comp else xg.create_dataset(
            "indices", shape=(n_idx,), dtype="int32",
            chunks=(min(1 << 20, max(n_idx, 1)),),
        )
        ip_out = xg.create_dataset("indptr", shape=xg_src["indptr"].shape, dtype="int64")
        ip_out[:] = xg_src["indptr"][:]
        ix_out[:] = xg_src["indices"][:]
        n = xg_src["data"].shape[0]
        done = 0
        for start in range(0, n, CHUNK):
            blk = xg_src["data"][start : start + CHUNK].astype(np.float32)
            d_out[start : start + blk.size] = blk
            done += blk.size
            if done % (1 << 26) < CHUNK:
                print(f"[f32]   {done:,}/{n:,}", flush=True)
        del d_out, ix_out
    src.unlink()
    tmp.rename(src)
    print(f"[f32] done {src.name} ({src.stat().st_size/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
