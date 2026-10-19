"""Real-data loading, gene-axis harmonization, and pseudobulk LFC stores.

Expected challenge layout (validation phase, per the official docs):

    data/
      gene_names.csv          # headerless gene symbols, the 18,533-gene axis
      pert_counts.csv         # target_gene[, context, ...] panel to predict
      context_A.h5ad          # ~18,400 unperturbed control cells per context
      context_B.h5ad
      context_C.h5ad

Training data (e.g. Replogle CRISPRi screens, VCC 2025 release) is any set of
.h5ad files with raw integer counts and obs columns identifying the target
gene and source context; everything is harmonized onto the challenge gene
axis and reduced to per-(context, target) LFC vectors in a PseudobulkStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import spec
from .countsim import log2_fold_change

TARGET_COL_CANDIDATES = ("target_gene", "target", "perturbation", "gene", "guide_identity")
CONTEXT_COL_CANDIDATES = ("context", "cell_line", "dataset", "condition", "batch")
CONTROL_CANDIDATES = ("non-targeting", "ntc", "control", "non_targeting", "safe-targeting")


# ------------------------------------------------------------------ basics


def load_gene_names_csv(path: str | Path) -> list[str]:
    """Headerless (or simply-headed) single-column gene symbol list."""
    df = pd.read_csv(path, header=None, dtype=str)
    col = df.iloc[:, 0].astype(str).str.strip()
    first = col.iloc[0].lower()
    if first in {"gene", "gene_name", "gene_names", "gene_symbol", "x", "index", "symbol"}:
        col = col.iloc[1:]
    return [g for g in col if g and g.lower() != "nan"]


def load_panel_csv(path: str | Path) -> pd.DataFrame:
    """Panel of (target_gene, context) pairs to predict; context optional."""
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    gene_col = next(cols[c] for c in TARGET_COL_CANDIDATES if c in cols)
    ctx_col = next((cols[c] for c in CONTEXT_COL_CANDIDATES if c in cols), None)
    out = pd.DataFrame({spec.TARGET_COL: df[gene_col].astype(str)})
    out[spec.CONTEXT_COL] = df[ctx_col].astype(str) if ctx_col else "A"
    return out


def detect_columns(obs: pd.DataFrame) -> tuple[str | None, str | None]:
    lower = {c.lower(): c for c in obs.columns}
    tcol = next((lower[c] for c in TARGET_COL_CANDIDATES if c in lower), None)
    ccol = next((lower[c] for c in CONTEXT_COL_CANDIDATES if c in lower), None)
    return tcol, ccol


def harmonize_to_axis(
    x,
    var_names: list[str] | pd.Index,
    gene_names: list[str],
) -> sparse.csr_matrix:
    """Project a (cells x source-genes) matrix onto the challenge gene axis.

    Missing genes become zero columns; extra source genes are dropped.
    Implemented as a sparse 0/1 projection so streaming chunks stay cheap.
    """
    pos = {g: i for i, g in enumerate(gene_names)}
    rows, cols = [], []
    for src_i, g in enumerate(var_names):
        dst = pos.get(str(g))
        if dst is not None:
            rows.append(src_i)
            cols.append(dst)
    proj = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(var_names), len(gene_names)),
    )
    if sparse.issparse(x):
        return (x @ proj).tocsr()
    return sparse.csr_matrix(np.asarray(x) @ proj)


# ------------------------------------------------------------- controls


def load_controls_bundle(
    root: str | Path,
    gene_names: list[str],
    context_col: str | None = None,
) -> dict[str, sparse.csr_matrix]:
    """Load unperturbed control cells per context, harmonized to the axis.

    Accepts either one file per context (context_A.h5ad, ...) or multi-context
    files split by an obs column (context_col or auto-detected 'context').
    """
    root = Path(root)
    files = sorted(root.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"no .h5ad control files under {root}")

    out: dict[str, sparse.csr_matrix] = {}
    for f in files:
        adata = ad.read_h5ad(f)
        tcol, ccol = detect_columns(adata.obs)
        use_ccol = context_col or ccol
        x = adata.X

        if use_ccol is not None and use_ccol in adata.obs:
            ctx_values = adata.obs[use_ccol].astype(str)
            for ctx in sorted(ctx_values.unique()):
                mask = (ctx_values == ctx).to_numpy()
                out[ctx] = harmonize_to_axis(x[mask], adata.var_names, gene_names)
            continue

        name_ctx = _context_from_filename(f.name)
        out[name_ctx] = harmonize_to_axis(x, adata.var_names, gene_names)
    return out


def _context_from_filename(name: str) -> str:
    stem = Path(name).stem
    for sep in ("context_", "context-", "ctx_"):
        if stem.lower().startswith(sep):
            return stem[len(sep):]
    return stem


# --------------------------------------------------------- pseudobulk store


@dataclass
class PseudobulkStore:
    """Per-(context, target) observed LFC vectors on the challenge axis."""

    gene_names: np.ndarray                       # (G,)
    pairs: list[tuple[str, str]]                 # (context, target_gene)
    lfc: np.ndarray                              # (N, G) float32
    basal: dict[str, np.ndarray] = field(default_factory=dict)  # ctx -> log mean CPM
    n_cells: dict[tuple[str, str], int] = field(default_factory=dict)
    controls: dict[str, "sparse.csr_matrix"] = field(default_factory=dict)  # ctx -> raw counts
    source: str = ""

    def contexts(self) -> list[str]:
        return sorted(self.basal)

    def lfc_pairs(self) -> dict[tuple[str, str], np.ndarray]:
        return {p: self.lfc[i].astype(np.float64) for i, p in enumerate(self.pairs)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ctx_arr = np.array([c for c, _ in self.pairs], dtype=object)
        tgt_arr = np.array([t for _, t in self.pairs], dtype=object)
        n_arr = np.array([self.n_cells.get(p, 0) for p in self.pairs], dtype=np.int32)
        basal_ctx = np.array(sorted(self.basal), dtype=object)
        basal_mat = (
            np.stack([self.basal[c] for c in basal_ctx]).astype(np.float32)
            if len(basal_ctx)
            else np.zeros((0, len(self.gene_names)), np.float32)
        )
        np.savez_compressed(
            path,
            gene_names=self.gene_names.astype(str),
            context=ctx_arr,
            target=tgt_arr,
            lfc=self.lfc.astype(np.float32),
            n_cells=n_arr,
            basal_ctx=basal_ctx,
            basal=basal_mat,
            source=np.array([self.source]),
        )
        if self.controls:
            ctrl_path = path.with_name(path.stem + "__controls.npz")
            keys = sorted(self.controls)
            mats = [sparse.csr_matrix(self.controls[k]) for k in keys]
            vstacked = sparse.vstack(mats, format="csr")
            np.savez_compressed(
                ctrl_path,
                context=np.array(keys, dtype=object),
                indptr=vstacked.indptr,
                indices=vstacked.indices,
                data=vstacked.data,
                shape=np.array(vstacked.shape),
                row_starts=np.array([m.indptr[-1] if m.shape[0] else 0 for m in mats]),
            )

    @classmethod
    def load(cls, path: str | Path) -> "PseudobulkStore":
        z = np.load(path, allow_pickle=True)
        basal = {c: z["basal"][i] for i, c in enumerate(z["basal_ctx"])}
        pairs = list(zip([str(c) for c in z["context"]], [str(t) for t in z["target"]]))
        n_cells = {p: int(n) for p, n in zip(pairs, z["n_cells"])}
        store = cls(
            gene_names=z["gene_names"].astype(str),
            pairs=pairs,
            lfc=z["lfc"].astype(np.float32),
            basal=basal,
            n_cells=n_cells,
            source=str(z["source"][0]) if "source" in z else "",
        )
        # Sanitize once at load: downstream fits assume finite LFCs within
        # +-8 (epsilon blowups on silent control genes can reach |lfc|~30).
        lc = store.lfc
        np.nan_to_num(lc, copy=False, nan=0.0, posinf=np.float32(8.0), neginf=np.float32(-8.0))
        np.clip(lc, -8.0, 8.0, out=lc)
        ctrl_path = Path(path).with_name(Path(path).stem + "__controls.npz")
        if ctrl_path.exists():
            cz = np.load(ctrl_path, allow_pickle=True)
            mat = sparse.csr_matrix(
                (cz["data"], cz["indices"], cz["indptr"]),
                shape=tuple(cz["shape"]),
            )
            starts = np.cumsum([0] + list(cz["row_starts"]))
            for i, c in enumerate(cz["context"]):
                store.controls[str(c)] = mat[starts[i] : starts[i + 1]].tocsr()
        return store


def build_pseudobulk_store(
    sources: list[dict],
    gene_names: list[str],
    min_cells: int = 10,
    chunk_rows: int = 4096,
    max_control_cells: int = 4000,
    control_seed: int = 0,
) -> PseudobulkStore:
    """Stream one or more raw-count .h5ad screens into LFC vectors.

    sources: dicts with keys {path, name, target_col?, context_col?,
    control_label?}. Contexts are namespaced as ``{name}/{context}`` when a
    context column exists, else just ``{name}``. LFCs use arithmetic mean-CPM
    profiles exactly as the scored DE metric defines them.

    When max_control_cells > 0 is requested, control rows are NOT stored by
    this builder anymore; use a dedicated extractor to write the
    ``{store}__controls.npz`` companion consumed by PseudobulkStore.load.
    All accumulation here is sparse-only (no densification), so large deep
    screens stream through in bounded memory.
    """
    gene_names = list(gene_names)
    pairs: list[tuple[str, str]] = []
    lfc_rows: list[np.ndarray] = []
    n_cells: dict[tuple[str, str], int] = {}
    basal: dict[str, np.ndarray] = {}
    controls: dict[str, sparse.csr_matrix] = {}

    for src in sources:
        path = Path(src["path"])
        name = src.get("name", path.stem)
        adata = ad.read_h5ad(path, backed="r")
        obs = adata.obs.copy()
        tcol, ccol = detect_columns(obs)
        if tcol is None:
            raise ValueError(f"{path}: cannot identify target column in {list(obs.columns)}")
        ctrl_label = src.get("control_label", spec.CONTROL_LABEL)

        if src.get("force_single_context"):
            # One file == one biological context (e.g. all Flex batches of one
            # cell line); ignores any batch column so basals pool cleanly.
            ctx_of_row = None
        else:
            use_ccol = src.get("context_col") or ccol
            ctx_of_row = obs[use_ccol].astype(str) if use_ccol and use_ccol in obs else None

        targets = obs[tcol].astype(str)

        def ctx_name(row_ctx) -> str:
            return f"{name}/{row_ctx}" if row_ctx is not None else name

        # ---- Pass 1: control mean-CPM per context (sparse streaming) ----
        ctrl_mask_all = (targets == ctrl_label).to_numpy()
        acc: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for start in range(0, adata.n_obs, chunk_rows):
            stop = min(start + chunk_rows, adata.n_obs)
            if not ctrl_mask_all[start:stop].any():
                continue
            block = harmonize_to_axis(adata[start:stop].X, adata.var_names, gene_names)
            totals = np.asarray(block.sum(axis=1)).ravel()
            ok = totals > 0
            if not ok.any():
                continue
            w = spec.DE_TARGET_SUM / totals[ok]
            sub = block[ok]  # csr row-subset of valid control cells
            if ctx_of_row is not None:
                rows_ctx = ctx_of_row.iloc[start:stop].to_numpy()[ok]
                for rc in pd.unique(rows_ctx):
                    key = ctx_name(rc)
                    sel = rows_ctx == rc
                    part = np.asarray(
                        sub[sel].multiply(w[sel][:, None]).sum(axis=0)
                    ).ravel()
                    acc[key] = acc.get(key, 0.0) + part
                    counts[key] = counts.get(key, 0) + int(sel.sum())
            else:
                part = np.asarray(sub.multiply(w[:, None]).sum(axis=0)).ravel()
                acc[name] = acc.get(name, 0.0) + part
                counts[name] = counts.get(name, 0) + int(ok.sum())

        for key, s in acc.items():
            if counts[key] >= min_cells:
                basal[key] = np.log(s / counts[key] + 1e-9)

        # ---- Pass 2: per-target mean-CPM, then LFC vs context control ----
        tgt_arr = targets.to_numpy()
        if ctx_of_row is not None:
            key_arr = np.array([ctx_name(rc) for rc in ctx_of_row.to_numpy()], dtype=object)
        else:
            key_arr = np.full(adata.n_obs, name, dtype=object)
        group_rows: dict[tuple[str, str], np.ndarray] = {}
        for i in np.flatnonzero(~ctrl_mask_all):
            group_rows.setdefault((str(key_arr[i]), str(tgt_arr[i])), []).append(int(i))

        for (ckey, tgt), row_idx in sorted(group_rows.items()):
            if ckey not in basal or len(row_idx) < min_cells:
                continue
            row_idx_arr = np.sort(np.asarray(row_idx))
            s = np.zeros(len(gene_names), dtype=np.float64)
            n = 0
            for start in range(0, row_idx_arr.size, chunk_rows):
                sel = row_idx_arr[start : start + chunk_rows]
                block = harmonize_to_axis(adata[sel.tolist()].X, adata.var_names, gene_names)
                totals = np.asarray(block.sum(axis=1)).ravel()
                ok = totals > 0
                if not ok.any():
                    continue
                w = spec.DE_TARGET_SUM / totals[ok]
                s += np.asarray(block[ok].multiply(w[:, None]).sum(axis=0)).ravel()
                n += int(ok.sum())
            if n < min_cells:
                continue
            mean_cpm = s / n
            ctrl_mean = np.exp(basal[ckey]) - 1e-9
            pairs.append((ckey, tgt))
            lfc_rows.append(log2_fold_change(mean_cpm, ctrl_mean, spec.DE_EPSILON).astype(np.float32))
            n_cells[(ckey, tgt)] = n

        adata.file.close()

    if not pairs:
        raise RuntimeError("no (context, target) groups passed the cell-count filter")
    return PseudobulkStore(
        gene_names=np.array(gene_names),
        pairs=pairs,
        lfc=np.stack(lfc_rows),
        basal=basal,
        n_cells=n_cells,
        controls=controls,
        source=";".join(s.get("name", Path(s["path"]).stem) for s in sources),
    )
