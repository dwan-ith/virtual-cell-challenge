"""Submission assembly and validation per the official 2026 rules (spec.py).

Rules enforced here (verified against the evaluation page + CLI wiki):
  - raw integer counts, 0 <= count <= 1,000,000 per cell
  - exactly CELLS_PER_PERT cells for every (context, target_gene)
  - no control ("non-targeting") rows — controls are held out server-side
  - gene axis exactly gene_names.csv, in order
  - scipy CSR sparse X; obs columns `target_gene` and `context`
  - stored-entries (nnz) cap DENSITY_CAP_ENTRIES across the file

Package with the official CLI: `vcc prep` (run_vcc_prep below).
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import spec


def _to_csr_counts(block: np.ndarray) -> sparse.csr_matrix:
    arr = np.asarray(block)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D block, got shape {arr.shape}")
    if arr.min(initial=0.0) < 0:
        raise ValueError("negative counts in block")
    if float(arr.max(initial=0.0)) > spec.MAX_COUNTS_PER_CELL:
        raise ValueError(
            f"cell exceeds MAX_COUNTS_PER_CELL={spec.MAX_COUNTS_PER_CELL}"
        )
    if not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError("non-integer counts — submissions must be raw counts")
    return sparse.csr_matrix(arr.astype(np.int32))


def build_submission(
    cells: dict[tuple[str, str], np.ndarray],
    gene_names: list[str],
    out_path: str | Path,
    contexts: tuple[str, ...] = spec.VALIDATION_CONTEXTS,
) -> ad.AnnData:
    """cells: {(context, target_gene): (n_cells, G) integer counts}."""
    blocks: list[sparse.csr_matrix] = []
    ctx_col: list[str] = []
    tgt_col: list[str] = []
    for (ctx, tgt), arr in sorted(cells.items()):
        blocks.append(_to_csr_counts(arr))
        ctx_col.extend([str(ctx)] * arr.shape[0])
        tgt_col.extend([str(tgt)] * arr.shape[0])

    x = sparse.vstack(blocks, format="csr")
    obs = pd.DataFrame(
        {
            spec.TARGET_COL: pd.Series(tgt_col, dtype="category"),
            spec.CONTEXT_COL: pd.Series(ctx_col, dtype="category"),
        }
    )
    var = pd.DataFrame(index=pd.Index([str(g) for g in gene_names]))
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.obs_names = [f"pred_{i}" for i in range(adata.n_obs)]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    problems = validate_submission(
        adata, expected_gene_names=gene_names, contexts=contexts
    )
    if problems:
        raise ValueError("submission failed self-validation:\n" + "\n".join(problems))
    adata.write_h5ad(out_path, compression="gzip")
    return adata


def validate_submission(
    obj: str | Path | ad.AnnData,
    expected_gene_names: list[str] | None = None,
    contexts: tuple[str, ...] = spec.VALIDATION_CONTEXTS,
    perts_per_context: int = spec.N_PERTS_PER_CONTEXT,
    cells_per_pert: int = spec.CELLS_PER_PERT,
) -> list[str]:
    """Return a list of rule violations (empty list == compliant)."""
    problems: list[str] = []
    adata = ad.read_h5ad(obj) if isinstance(obj, (str, Path)) else obj

    if not sparse.issparse(adata.X) or adata.X.format != "csr":
        problems.append("X must be a scipy CSR sparse matrix")

    x = adata.X
    data = np.asarray(x.data)
    if not np.all(np.equal(np.mod(data, 1), 0)):
        problems.append("counts must be integers (raw counts, not normalized)")
    if data.size and data.min() < 0:
        problems.append("negative counts present")
    if data.size and data.max() > spec.MAX_COUNTS_PER_CELL:
        problems.append(f"count value {data.max()} exceeds {spec.MAX_COUNTS_PER_CELL}")
    if x.nnz > spec.DENSITY_CAP_ENTRIES:
        problems.append(f"nnz={x.nnz} exceeds density cap {spec.DENSITY_CAP_ENTRIES}")

    for col in (spec.TARGET_COL, spec.CONTEXT_COL):
        if col not in adata.obs:
            problems.append(f"obs column '{col}' missing")

    if expected_gene_names is not None:
        got = [str(v) for v in adata.var_names]
        want = [str(v) for v in expected_gene_names]
        if len(got) != len(want):
            problems.append(f"n_vars={len(got)} != expected {len(want)}")
        elif got != want:
            bad = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
            problems.append(f"gene axis/order mismatch at position {bad}: {got[bad]} != {want[bad]}")
    elif adata.n_vars != spec.N_GENES:
        problems.append(f"n_vars={adata.n_vars} != official axis {spec.N_GENES}")

    if spec.TARGET_COL in adata.obs and spec.CONTEXT_COL in adata.obs:
        tg = adata.obs[spec.TARGET_COL].astype(str)
        cx = adata.obs[spec.CONTEXT_COL].astype(str)
        ctrl_rows = tg == spec.CONTROL_LABEL
        if ctrl_rows.any():
            problems.append(f"{int(ctrl_rows.sum())} control rows present (forbidden)")
        counts = pd.Series(list(zip(cx, tg))).value_counts()
        wrong = counts[counts != cells_per_pert]
        if len(wrong):
            problems.append(
                f"{len(wrong)} groups lack exactly {cells_per_pert} cells; "
                f"e.g. {dict(wrong.head(3))}"
            )
        missing_ctx = [c for c in contexts if c not in set(cx)]
        if missing_ctx:
            problems.append(f"contexts missing from submission: {missing_ctx}")
        for c in contexts:
            n_perts = int((cx == c).sum()) // max(cells_per_pert, 1)
            if c in set(cx) and n_perts != perts_per_context:
                problems.append(f"context {c}: {n_perts} perturbations != {perts_per_context}")
    return problems


def run_vcc_prep(h5ad_path: str | Path, gene_names_csv: str | Path) -> bool:
    """Invoke the official `vcc prep` packager; returns True on success."""
    exe = shutil.which("vcc")
    if exe is None:
        print("[vcc] CLI not found — install with `pip install arc-vcc` and run "
              "`vcc login`, then package manually:")
        print(f"      vcc prep -i {h5ad_path} -g {gene_names_csv}")
        return False

    attempts = (
        [exe, "prep", "-i", str(h5ad_path), "-g", str(gene_names_csv)],
        [exe, "prep", "--input", str(h5ad_path), "--gene-names", str(gene_names_csv)],
    )
    for cmd in attempts:
        try:
            rc = subprocess.call(cmd)
        except OSError as exc:
            print(f"[vcc] failed to run {cmd[0]}: {exc}")
            return False
        if rc == 0:
            return True
    print("[vcc] `vcc prep` returned non-zero — check `vcc prep --help` for flags.")
    return False


# --------------------------------------------------------------------------
# Streaming writer: full-panel submissions are ~360k cells x 18,533 genes,
# which cannot be held densely in RAM on a 16 GB machine. This writes the
# h5ad incrementally in anndata's native on-disk format (CSR X appended
# block by block), so peak memory stays at one prediction block.
# --------------------------------------------------------------------------


class StreamingSubmissionWriter:
    """Append rule-checked prediction blocks directly into an h5ad file."""

    def __init__(
        self,
        out_path: str | Path,
        gene_names: list[str],
        compression: str | None = "gzip",
        append: bool = False,
    ):
        import h5py

        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.gene_names = [str(g) for g in gene_names]
        self._compression = compression
        if append and self.out_path.exists():
            # Continue a previous chunked run: recover counts and labels
            self._h5 = h5py.File(self.out_path, "r+")
            xg = self._h5["X"]
            shape = tuple(int(v) for v in xg.attrs["shape"])
            self._n_obs = shape[0]
            self._nnz = int(xg["indptr"][-1])
            # Reload obs labels written by previous close()
            obsg = self._h5["obs"]
            def _read_categorical(group_name: str) -> list[str]:
                g = obsg[group_name]
                cats = _decode_strs(g["categories"][:])
                codes = g["codes"][:]
                return [cats[c] if c >= 0 else "" for c in codes]
            try:
                self._tgt_chunks = _read_categorical("target_gene")
                self._ctx_chunks = _read_categorical("context")
            except KeyError as e:
                raise ValueError(
                    f"append mode: previous run left no obs labels ({e}); "
                    "rebuild the file from scratch"
                ) from e
            # NOTE: obs groups are deleted at close() time, NOT here, so a
            # killed run always leaves the previous close's labels intact.
            # Reconcile: a killed run may have left X rows beyond labeled obs.
            # Truncate X back to the labeled row count (blocks are atomic per target).
            n_labeled = len(self._tgt_chunks)
            if self._n_obs > n_labeled:
                xg = self._h5["X"]
                drop = self._n_obs - n_labeled
                keep_indptr = xg["indptr"][: n_labeled + 1]
                keep_nnz = int(keep_indptr[-1])
                xg["data"].resize((keep_nnz,), axis=0)
                xg["indices"].resize((keep_nnz,), axis=0)
                xg["indptr"].resize((n_labeled + 1,), axis=0)
                self._n_obs = n_labeled
                self._nnz = keep_nnz
                xg.attrs["shape"] = [self._n_obs, len(self.gene_names)]
            return
        self._h5 = h5py.File(self.out_path, "w")
        self._n_obs = 0
        self._nnz = 0
        self._tgt_chunks: list[str] = []
        self._ctx_chunks: list[str] = []

        root = self._h5
        root.attrs["encoding-type"] = "anndata"
        root.attrs["encoding-version"] = "0.1.0"

        # Empty auxiliary groups so readers see the standard layout.
        for name in ("layers", "obsm", "obsp", "uns", "raw", "varm", "varp"):
            g = root.create_group(name)
            g.attrs["encoding-type"] = "dict"
            g.attrs["encoding-version"] = "0.1.0"

        # ---- /var ----
        varg = root.create_group("var")
        varg.attrs["_index"] = "_index"
        varg.attrs["column-order"] = []
        varg.attrs["encoding-type"] = "dataframe"
        varg.attrs["encoding-version"] = "0.2.0"
        self._write_string_index(varg, self.gene_names)

        # ---- /obs (columns filled at close) ----
        obsg = root.create_group("obs")
        obsg.attrs["_index"] = "_index"
        obsg.attrs["column-order"] = ["context", "target_gene"]
        obsg.attrs["encoding-type"] = "dataframe"
        obsg.attrs["encoding-version"] = "0.2.0"

        # ---- /X (resizable CSR) ----
        xg = root.create_group("X")
        xg.attrs["encoding-type"] = "csr_matrix"
        xg.attrs["encoding-version"] = "0.1.0"
        xg.attrs["shape"] = np.array([0, len(self.gene_names)])
        cargs = {"chunks": (1 << 20)}
        if compression:
            cargs["compression"] = compression
        xg.create_dataset("data", shape=(0,), maxshape=(None,), dtype="int32", **cargs)
        xg.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="int32", **cargs)
        xg.create_dataset("indptr", shape=(1,), maxshape=(None,), dtype="int64")
        xg["indptr"][0] = 0

    @staticmethod
    def _write_string_index(group, values: list[str]) -> None:
        """anndata >=0.7 nullable-string-array index representation."""
        import h5py

        dt = h5py.special_dtype(vlen=str)
        g = group.create_group("_index")
        g.attrs["encoding-type"] = "nullable-string-array"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["na-value"] = "NaN"
        vals = g.create_dataset("values", shape=(len(values),), dtype=dt)
        if values:
            vals[:] = values
        g.create_dataset("mask", data=np.zeros(len(values), dtype=bool))

    def add_block(self, counts, targets: list[str], contexts: list[str]) -> None:
        """counts: (n_cells, G) integer-valued array or CSR; one group's cells."""
        x = sparse.csr_matrix(counts)
        if x.shape[1] != len(self.gene_names):
            raise ValueError(f"block has {x.shape[1]} genes, expected {len(self.gene_names)}")
        if len(targets) != x.shape[0] or len(contexts) != x.shape[0]:
            raise ValueError("targets/context labels must match block rows")
        data = np.asarray(x.data)
        if data.size and (data.min() < 0 or data.max() > spec.MAX_COUNTS_PER_CELL):
            raise ValueError("counts out of range in block")
        if data.size and not np.all(np.equal(np.mod(data, 1), 0)):
            raise ValueError("non-integer counts in block")
        if any(t == spec.CONTROL_LABEL for t in targets):
            raise ValueError("control rows are forbidden in submissions")

        xg = self._h5["X"]
        n_new = x.shape[0]
        d = xg["data"]; ix = xg["indices"]; ip = xg["indptr"]
        d.resize(d.shape[0] + x.nnz, axis=0)
        d[-x.nnz:] = x.data.astype(np.int32)
        ix.resize(ix.shape[0] + x.nnz, axis=0)
        ix[-x.nnz:] = x.indices.astype(np.int32)
        ip.resize(ip.shape[0] + n_new, axis=0)
        prev = ip[-n_new - 1]
        ip[-n_new:] = prev + x.indptr[1:].astype(np.int64)
        self._nnz += int(x.nnz)
        if self._nnz > spec.DENSITY_CAP_ENTRIES:
            raise ValueError(
                f"nnz={self._nnz} exceeds density cap {spec.DENSITY_CAP_ENTRIES}"
            )
        self._tgt_chunks.extend(str(t) for t in targets)
        self._ctx_chunks.extend(str(c) for c in contexts)
        self._n_obs += n_new
        xg.attrs["shape"] = [self._n_obs, len(self.gene_names)]

    def close(self) -> Path:
        import h5py

        obsg = self._h5["obs"]
        dt = h5py.special_dtype(vlen=str)
        # Delete previous close's obs groups first (append mode); fresh files
        # have none. Deleting here (not at open) keeps labels crash-safe.
        for _name in ("target_gene", "context", "_index"):
            if _name in obsg:
                del obsg[_name]

        def _write_categorical(group_name: str, values: list[str]) -> None:
            cats = sorted(set(values))
            cat_pos = {c: i for i, c in enumerate(cats)}
            codes = np.array([cat_pos[v] for v in values], dtype=np.int32)
            g = obsg.create_group(group_name)
            g.attrs["encoding-type"] = "categorical"
            g.attrs["encoding-version"] = "0.2.0"
            g.attrs["ordered"] = False
            g.create_dataset("codes", data=codes)
            cd = g.create_dataset("categories", shape=(len(cats),), dtype=dt)
            if cats:
                cd[:] = cats

        _write_categorical("target_gene", self._tgt_chunks)
        _write_categorical("context", self._ctx_chunks)
        self._write_string_index(obsg, [f"pred_{i}" for i in range(self._n_obs)])
        self._h5.close()
        return self.out_path

    def __enter__(self) -> "StreamingSubmissionWriter":
        return self
    def __exit__(self, exc_type, exc, tb) -> bool:
        # If the with-block raised, do NOT swallow — propagate after cleanup attempt
        if exc_type is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            return False
        try:
            self.close()
        except Exception:
            # close() failures on successful blocks must surface
            raise
        return False


def _decode_strs(arr) -> list[str]:
    out = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def validate_h5ad_file_light(
    path: str | Path,
    expected_gene_names: list[str],
    contexts: tuple[str, ...] = spec.VALIDATION_CONTEXTS,
    perts_per_context: int = spec.N_PERTS_PER_CONTEXT,
    cells_per_pert: int = spec.CELLS_PER_PERT,
    chunk: int = (1 << 24),
) -> list[str]:
    """Rule-check a large submission h5ad WITHOUT loading X into RAM.

    Streams the CSR datasets in chunks for value checks; reads obs/var
    metadata directly. Returns the same style of problem list as
    validate_submission.
    """
    import h5py

    problems: list[str] = []
    path = Path(path)
    with h5py.File(path, "r") as f:
        xg = f["X"]
        shape = tuple(int(v) for v in xg.attrs["shape"])
        n_obs, n_vars = shape
        indptr = xg["indptr"][:]
        nnz = int(indptr[-1])
        if nnz > spec.DENSITY_CAP_ENTRIES:
            problems.append(f"nnz={nnz} exceeds density cap {spec.DENSITY_CAP_ENTRIES}")

        want = [str(g) for g in expected_gene_names]
        got = _decode_strs(f["var"]["_index"]["values"][:])
        if got != want:
            if len(got) != len(want):
                problems.append(f"n_vars={len(got)} != expected {len(want)}")
            else:
                bad = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
                problems.append(f"gene axis/order mismatch at position {bad}: {got[bad]} != {want[bad]}")

        # stream value checks
        data_ds = xg["data"]
        mn, mx = np.int64(0), np.int64(0)
        row_sums = np.zeros(max(n_obs, 1), dtype=np.int64)
        for start in range(0, data_ds.shape[0], chunk):
            blk = np.asarray(data_ds[start : start + chunk], dtype=np.int64)
            if not blk.size:
                continue
            mn = min(mn, int(blk.min()))
            mx = max(mx, int(blk.max()))
            kk = np.arange(start, start + blk.size)
            rows = np.searchsorted(indptr, kk, side="right") - 1
            row_sums += np.bincount(rows, weights=blk.astype(np.float64), minlength=n_obs).astype(np.int64)
        if mn < 0:
            problems.append("negative counts present")
        if mx > spec.MAX_COUNTS_PER_CELL:
            problems.append(f"count value {mx} exceeds {spec.MAX_COUNTS_PER_CELL}")
        if n_obs and row_sums.max() > spec.MAX_COUNTS_PER_CELL:
            problems.append(
                f"a cell totals {int(row_sums.max())} > {spec.MAX_COUNTS_PER_CELL}"
            )

        # obs labels
        tg_group = f["obs"]["target_gene"]
        cx_group = f["obs"]["context"]
        tg_cats = _decode_strs(tg_group["categories"][:])
        cx_cats = _decode_strs(cx_group["categories"][:])
        tg_codes = tg_group["codes"][:]
        cx_codes = cx_group["codes"][:]
        tg = np.array([tg_cats[c] if c >= 0 else "" for c in tg_codes], dtype=object)
        cx = np.array([cx_cats[c] if c >= 0 else "" for c in cx_codes], dtype=object)

        ctrl_rows = tg == spec.CONTROL_LABEL
        if ctrl_rows.any():
            problems.append(f"{int(ctrl_rows.sum())} control rows present (forbidden)")

        pairs = pd.Series(list(zip(cx, tg))).value_counts()
        wrong = pairs[pairs != cells_per_pert]
        if len(wrong):
            problems.append(
                f"{len(wrong)} groups lack exactly {cells_per_pert} cells; "
                f"e.g. {dict(wrong.head(3))}"
            )
        missing_ctx = [c for c in contexts if c not in set(cx)]
        if missing_ctx:
            problems.append(f"contexts missing from submission: {missing_ctx}")
        for c in contexts:
            n_perts = int((cx == c).sum()) // max(cells_per_pert, 1)
            if c in set(cx) and n_perts != perts_per_context:
                problems.append(f"context {c}: {n_perts} perturbations != {perts_per_context}")
    return problems
