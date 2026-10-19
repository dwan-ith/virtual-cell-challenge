"""Batch Stack ICL generation for the 2026 panel.

For every (challenge context, target gene) pair this script:
  1. writes a prompt h5ad: perturbed cells for that gene across all source
     contexts (subsampled), with the condition column `target_gene`;
  2. writes one shared query h5ad per challenge context (control subsample);
  3. invokes `stack-generation` (arc-stack CLI) per pair;
  4. imports outputs into the cache layout consumed by vcc_model.stack_adapter
     ({cache_dir}/{context}__{gene}.npy, log1p-CPM deltas on the challenge axis).

Requires: pip install arc-stack; a downloaded checkpoint
(huggingface: arcinstitute/Stack-Large-Aligned, ~217M params — runs on a
6 GB GPU); and raw source h5ads referenced from a stores config.

Usage:
    python scripts/run_stack_generation.py \
        --stores artifacts/stores/*.npz \
        --controls-dir data/vcc2026 \
        --checkpoint models/Stack-Large-Aligned/bc_large_aligned.ckpt \
        --genelist models/Stack-Large-Aligned/basecount_1000per_15000max.pkl \
        --raw data/raw \
        --cache-dir artifacts/stack_cache \
        [--contexts A B C] [--targets-file optional_subset.txt] [--dry-run]

The heavy lifting is offline and resumable: completed pairs are skipped, so
the script can be re-run as checkpoints finish.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model import spec
from vcc_model.countsim import sparse_log_mean_cpm
from vcc_model.dataset import (
    PseudobulkStore,
    detect_columns,
    load_gene_names_csv,
    load_panel_csv,
)
from vcc_model.stack_adapter import StackAdapter


def build_query(controls_path: Path, gene_names: list[str], n_cells: int, seed: int) -> ad.AnnData:
    a = ad.read_h5ad(controls_path)
    tcol, _ = detect_columns(a.obs)
    if tcol is not None:
        keep = (a.obs[tcol].astype(str) == spec.CONTROL_LABEL).to_numpy()
        if keep.any():
            a = a[keep]
    rng = np.random.default_rng(seed)
    if a.n_obs > n_cells:
        a = a[rng.choice(a.n_obs, size=n_cells, replace=False)]
    x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    out = ad.AnnData(
        X=x.astype(np.float32),
        obs=a.obs.iloc[:, :0].copy(),
        var=__import__("pandas").DataFrame(index=[str(v) for v in a.var_names]),
    )
    # Map onto challenge axis order for later delta import.
    assert list(out.var_names) == gene_names or len(out.var_names) == len(gene_names), (
        "query axis mismatch"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", nargs="+", type=Path, required=True,
                    help="for context metadata only; prompts come from --raw files")
    ap.add_argument("--raw", type=Path, required=True,
                    help="directory of raw source h5ads used to cut prompts")
    ap.add_argument("--controls-dir", type=Path, required=True)
    ap.add_argument("--gene-names", type=Path, default=None)
    ap.add_argument("--perts", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--genelist", type=Path, required=True,
                    help="Stack vocab pkl shipped with the checkpoint")
    ap.add_argument("--cache-dir", type=Path, default=Path("artifacts/stack_cache"))
    ap.add_argument("--workdir", type=Path, default=Path("artifacts/stack_work"))
    ap.add_argument("--prompt-cells", type=int, default=1024)
    ap.add_argument("--query-cells", type=int, default=1024)
    ap.add_argument("--contexts", nargs="+", default=None,
                    help="subset of challenge contexts to run")
    ap.add_argument("--targets-file", type=Path, default=None,
                    help="optional newline-delimited subset of targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda | cpu | auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    controls_dir = args.controls_dir
    gene_csv = args.gene_names or controls_dir / "gene_names.csv"
    perts_csv = args.perts or controls_dir / "pert_counts.csv"
    gene_names = load_gene_names_csv(gene_csv)
    panel = load_panel_csv(perts_csv)
    contexts = args.contexts or sorted(panel[spec.CONTEXT_COL].unique())
    targets = sorted(panel[spec.TARGET_COL].unique())
    if args.targets_file:
        subset = {l.strip() for l in args.targets_file.read_text().splitlines() if l.strip()}
        targets = [t for t in targets if t in subset]

    cache = StackAdapter(gene_names=gene_names, cache_dir=args.cache_dir)
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # Shared query file per context.
    queries: dict[str, Path] = {}
    for ctx in contexts:
        qpath = workdir / f"query_{ctx}.h5ad"
        if not qpath.exists():
            q = build_query(controls_dir / f"context_{ctx}.h5ad", gene_names, args.query_cells, args.seed)
            q.write_h5ad(qpath)
            print(f"[stack] wrote query {qpath} ({q.n_obs} cells)")
        queries[ctx] = qpath

    todo = [
        (ctx, g)
        for ctx in contexts
        for g in targets
        if not cache.has_cache(g, ctx)
    ]
    print(f"[stack] {len(todo)} (context, gene) pairs to generate "
          f"(skipping cached)")
    if args.dry_run:
        return

    t0 = time.time()
    done = 0
    for i, (ctx, gene) in enumerate(sorted(todo)):
        prompt_path = workdir / f"prompt_{ctx}_{gene}.h5ad"
        out_dir = workdir / f"gen_{ctx}_{gene}"
        prompt = _cut_prompt(args.raw, gene, args.prompt_cells, args.seed)
        if prompt is None:
            print(f"[stack][warn] no perturbed cells for {gene}; filling zeros delta")
            cache.write_delta(np.zeros(len(gene_names)), gene, ctx)
            continue
        prompt.write_h5ad(prompt_path)

        cmd = [
            "stack-generation",
            "--checkpoint", str(args.checkpoint),
            "--base-adata", str(prompt_path),
            "--test-adata", str(queries[ctx]),
            "--genelist", str(args.genelist),
            "--output-dir", str(out_dir),
            "--split-column", "target_gene",
        ]
        if args.device:
            cmd += ["--device", args.device]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[stack][warn] generation failed rc={rc} for {ctx}/{gene}")
            continue

        delta = _import_generation(out_dir, queries[ctx], gene_names, gene)
        cache.write_delta(delta, gene, ctx)
        done += 1
        if done % 10 == 0:
            rate = done / max(time.time() - t0, 1)
            print(f"[stack] {done}/{len(todo)} pairs ({rate:.2f}/s, eta "
                  f"{(len(todo) - done) / max(rate, 1e-9) / 3600:.1f}h)")

    print(f"[stack] finished {done}/{len(todo)} pairs -> {args.cache_dir}")


def _cut_prompt(raw_dir: Path, gene: str, n_cells: int, seed: int) -> ad.AnnData | None:
    """Collect perturbed cells for one target gene across raw source files."""
    import pandas as pd

    rng = np.random.default_rng(seed + hash(gene) % 99991)
    parts: list[np.ndarray] = []
    var_names: list[str] | None = None
    obs_frames: list[pd.DataFrame] = []
    row_counts: list[int] = []
    for f in sorted(raw_dir.glob("*.h5ad")):
        a = ad.read_h5ad(f, backed="r")
        tcol, ccol = detect_columns(a.obs)
        if tcol is None:
            a.file.close()
            continue
        labels = a.obs[tcol].astype(str)
        mask = (labels == gene).to_numpy()
        if not mask.any():
            a.file.close()
            continue
        idx = np.flatnonzero(mask)
        if len(idx) > n_cells:
            idx = rng.choice(idx, size=n_cells, replace=False)
        block = a[idx.tolist()].to_memory()
        a.file.close()
        if var_names is None:
            var_names = [str(v) for v in block.var_names]
        else:
            block = block[:, [v for v in var_names if v in set(map(str, block.var_names))]].copy()
            missing = [v for v in var_names if v not in set(block.var_names)]
            if missing:
                fill = ad.AnnData(
                    X=np.zeros((block.n_obs, len(missing)), dtype=np.float32),
                    obs=block.obs.iloc[:, :0].copy(),
                    var=pd.DataFrame(index=missing),
                )
                block = __import__("anndata").concat([block, fill], axis=1)
                block = block[:, var_names]
        parts.append(np.asarray(block.X.todense() if hasattr(block.X, "todense") else block.X,
                                dtype=np.float32))
        meta = pd.DataFrame({
            "target_gene": gene,
            "source": f.stem,
        }, index=[f"{f.stem}_{i}" for i in range(len(parts[-1]))])
        obs_frames.append(meta)
        row_counts.append(parts[-1].shape[0])

    if not parts:
        return None
    x = np.vstack([p[:, : len(var_names)] if p.shape[1] >= len(var_names) else
                   np.pad(p, ((0, 0), (0, len(var_names) - p.shape[1]))) for p in parts])
    obs = pd.concat(obs_frames)
    return ad.AnnData(X=x.astype(np.float32), obs=obs,
                      var=pd.DataFrame(index=list(var_names)))


def _import_generation(gen_dir: Path, query_path: Path, gene_names: list[str], gene: str) -> np.ndarray:
    """Read stack-generation output, reduce to mean delta on the challenge axis."""
    outs = sorted(gen_dir.rglob("*.h5ad"))
    if not outs:
        raise RuntimeError(f"no output h5ad under {gen_dir}")
    pred = ad.read_h5ad(outs[0])
    query = ad.read_h5ad(query_path)

    from .countsim import sparse_log_mean_cpm

    ctrl_prof = sparse_log_mean_cpm(query.X, 1e6)
    pred_prof = sparse_log_mean_cpm(pred.X, 1e6)

    pos = {g: i for i, g in enumerate(map(str, pred.var_names))}
    out = np.zeros(len(gene_names), dtype=np.float64)
    hit = np.array([pos.get(g, -1) for g in gene_names])
    m = hit >= 0
    out[m] = (pred_prof - ctrl_prof)[hit[m]]
    return out


if __name__ == "__main__":
    main()
