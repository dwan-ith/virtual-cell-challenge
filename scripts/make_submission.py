"""Build a rule-valid 2026 submission from the challenge controls bundle.

Reads the official controls bundle (context_<X>.h5ad + gene_names.csv +
pert_counts.csv), fits the hybrid on all available training stores, predicts
exactly CELLS_PER_PERT integer-count cells per (context, target), validates
against the full 2026 rules, and writes the h5ad that `vcc prep` packages.

Usage:
    python scripts/make_submission.py \
        --stores artifacts/stores/*.npz \
        --controls-dir data/vcc2026 \
        --out artifacts/submission/prediction.h5ad \
        [--calibration artifacts/calibration.json] \
        [--stack-cache artifacts/stack_cache] [--prep]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model import spec
from vcc_model.calibrate import DECalibration
from vcc_model.countsim import multinomial_thin, poisson_thin, sparse_log_mean_cpm
from vcc_model.countsim import bulk_probability_from_lfc, repair_bulk_totals
from vcc_model.countsim import apply_promoter_ceiling, load_promoter_pairs
from vcc_model.dataset import (
    PseudobulkStore,
    detect_columns,
    harmonize_to_axis,
    load_gene_names_csv,
    load_panel_csv,
)
from vcc_model.ensemble import HybridPredictor
from vcc_model.residual import ResidualConfig
from vcc_model.submission import run_vcc_prep


def load_context_controls(
    path: Path,
    gene_names: list[str],
    control_label: str = spec.CONTROL_LABEL,
) -> tuple[sparse.csr_matrix, int]:
    """Load one context's control cells, harmonized to the challenge axis."""
    adata = ad.read_h5ad(path)
    tcol, _ = detect_columns(adata.obs)
    x = adata.X
    if tcol is not None:
        labels = adata.obs[tcol].astype(str)
        keep = (labels == control_label).to_numpy()
        if keep.any() and not keep.all():
            x = x[keep]
            print(f"[sub] {path.name}: kept {int(keep.sum())}/{len(labels)} control-labeled cells")
    n_cells = x.shape[0]
    return harmonize_to_axis(x, adata.var_names, gene_names).tocsr(), n_cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", nargs="+", type=Path, required=True)
    ap.add_argument("--controls-dir", type=Path, required=True,
                    help="directory with context_*.h5ad (+ optional gene_names.csv / pert_counts.csv)")
    ap.add_argument("--gene-names", type=Path, default=None)
    ap.add_argument("--perts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/submission/prediction.h5ad"))
    ap.add_argument("--calibration", type=Path, default=None,
                    help="JSON {threshold,gain,power} from LOCO tuning; omit for identity")
    ap.add_argument("--config", type=Path, default=None,
                    help="LOCO config JSON ({weights, calibration}) -- takes precedence "
                         "over --weights/--calibration flags")
    ap.add_argument("--weights", type=float, nargs=3, default=None, metavar=("STAT", "RESID", "STACK"),
                    help="blend weights; default (1.0, 0.7, 0.0)")
    ap.add_argument("--stack-cache", type=Path, default=None)
    ap.add_argument("--cells-per-pert", type=int, default=spec.CELLS_PER_PERT)
    ap.add_argument("--gen-pool", type=int, default=1500,                    help="control cells sampled per context as thinning pool")
    ap.add_argument("--shared-mix", type=float, default=None,
                    help="override the prior's shared-program mixing fraction")
    ap.add_argument("--no-alpha", action="store_true",
                    help="prior emits unscaled directions (calibration owns scale)")
    ap.add_argument("--decoder", choices=("poisson", "multinomial"), default="poisson",
                    help="count decoder: poisson is vectorized and adds realistic depth noise")
    ap.add_argument("--bulk-repair", action="store_true",
                    help="after thinning, repair integer cells to the desired pooled bulk "
                         "while preserving every cell depth")
    ap.add_argument("--bulk-gain", type=float, default=1.0,
                    help="multiplies the calibrated LFC only when forming the desired bulk")
    ap.add_argument("--bulk-boost", type=float, default=0.0,
                    help="depth-dependent gain: cell i uses exponent "
                         "1 + boost*(depth_i/mean_depth - 1), so deeper cells carry "
                         "a stronger response and move the pooled bulk more than the "
                         "per-cell mean. 0.0 = off (bit-exact baseline).")
    ap.add_argument("--promoter-pairs", type=Path, default=None,
                    help="CSV with target,neighbor,distance columns for the CRISPRi cis ceiling")
    ap.add_argument("--promoter-fraction", type=float, default=0.15,
                    help="remaining expression fraction at 500 bp before distance ramp")
    ap.add_argument("--panel-mean", type=float, default=0.0,
                    help="AtlasShift common_subtract: subtract this multiple of each "
                         "gene's mean LFC across all panel targets (self excluded). "
                         "0.0 = off (keeps v29/v38 bit-exact).")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-contexts", nargs="*", default=None,
                    help="restrict prediction to these contexts (e.g. --only-contexts A C); "
                         "used to split long builds into per-context chunks for merging")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing --out file from a previous chunked run")
    ap.add_argument("--start-index", type=int, default=0,
                    help="skip the first N targets (replaying RNG draws) to resume a killed chunk")
    ap.add_argument("--end-index", type=int, default=None,
                    help="stop after target N (exclusive); split slow contexts into labeled halves")
    ap.add_argument("--skip-validation", action="store_true",
                    help="skip end-of-run validation (validate only the final chunk)")
    ap.add_argument("--validate-contexts", nargs="*", default=None,
                    help="contexts to validate against (default: the built contexts)")
    ap.add_argument("--prep", action="store_true", help="run `vcc prep` after writing")
    args = ap.parse_args()

    controls_dir = args.controls_dir
    gene_csv = args.gene_names or next(
        (p for p in [controls_dir / "gene_names.csv"] if p.exists()), None
    )
    perts_csv = args.perts or next(
        (p for p in [controls_dir / "pert_counts.csv"] if p.exists()), None
    )
    if gene_csv is None or perts_csv is None:
        raise SystemExit(f"gene_names.csv / pert_counts.csv not found under {controls_dir}; "
                         "pass --gene-names/--perts explicitly")

    gene_names = load_gene_names_csv(gene_csv)
    gene_index = {str(g): i for i, g in enumerate(gene_names)}
    promoter_pairs = load_promoter_pairs(args.promoter_pairs, gene_names) if args.promoter_pairs else {}
    if args.promoter_pairs and not args.bulk_repair:
        raise SystemExit("--promoter-pairs currently applies to the desired bulk, so it requires --bulk-repair")
    panel = load_panel_csv(perts_csv)
    contexts_present = sorted(panel[spec.CONTEXT_COL].unique())
    targets = sorted(panel[spec.TARGET_COL].unique())
    print(f"[sub] axis={len(gene_names)} genes | {len(targets)} targets | contexts={contexts_present}")
    if len(gene_names) != spec.N_GENES:
        print(f"[sub][warn] axis length {len(gene_names)} != official {spec.N_GENES}")

    # ---- fit on all stores (no vstack copy: train_pairs are views into
    # each store's resident lfc, keeping peak RAM at the stores themselves) ----
    _stores: list[PseudobulkStore] = []
    for p in sorted(args.stores):
        s = PseudobulkStore.load(p)
        if _stores:
            assert np.array_equal(_stores[0].gene_names, s.gene_names), f"axis mismatch: {p}"
        _stores.append(s)
    assert _stores, "no training stores"
    merged = _stores[0]
    train_pairs: dict[tuple[str, str], np.ndarray] = {}
    for s in _stores:
        for i, pair in enumerate(s.pairs):
            train_pairs[pair] = s.lfc[i]
    for s in _stores[1:]:
        merged.n_cells.update(s.n_cells)
        for c, b in s.basal.items():
            if c in merged.basal:
                # Average duplicate basal vectors (e.g. same line from two stores) instead
                # of silently keeping the first — both carry information.
                merged.basal[c] = (merged.basal[c] + np.asarray(b, dtype=np.float64)) / 2.0
            else:
                merged.basal[c] = np.asarray(b, dtype=np.float64)
    assert merged is not None and len(train_pairs) > 0, "no training pairs"
    # train_pairs are float32 views into each store's resident lfc (no copies;
    # the 20175x18533 matrix stays resident exactly once across the stores).
    basals = dict(merged.basal)

    stack_fn = None
    if args.stack_cache:
        from vcc_model.stack_adapter import StackAdapter

        adapter = StackAdapter(gene_names=gene_names, cache_dir=args.stack_cache)

        def _make_stack_fn(adapter: "StackAdapter"):
            # Capture adapter; per-call context comes from HybridPredictor context arg
            def _fn(gene: str, basal_vec: np.ndarray, context: str | None = None) -> np.ndarray:
                ctx = context or "target"
                try:
                    return adapter.delta(None, gene, context=ctx).astype(np.float64)
                except RuntimeError:
                    return np.zeros(len(gene_names))
            return _fn
        stack_fn = _make_stack_fn(adapter)

    # Blend weights + calibration: LOCO artifact wins over flags over defaults.
    # Resolved BEFORE fitting so an unused residual net can be skipped.
    weights = tuple(args.weights) if args.weights else (1.0, 0.7, 0.0)
    calibration = DECalibration(**json.loads(Path(args.calibration).read_text())) \
        if args.calibration else DECalibration()
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        w = cfg.get("weights", {})
        weights = (float(w.get("stat", 1.0)), float(w.get("resid", 0.7)), float(w.get("stack", 0.0)))
        c = cfg.get("calibration")
        if c:
            calibration = DECalibration(**c)
        shared_mix_cfg = cfg.get("shared_mix")
        use_alpha_cfg = cfg.get("use_alpha")
        basal_gate_cpm_cfg = cfg.get("basal_gate_cpm")
        james_stein_cfg = cfg.get("james_stein")
        default_conservation_cfg = cfg.get("default_conservation")
        temperature_cfg = cfg.get("similarity_temperature")
        per_gene_gain_cfg = cfg.get("per_gene_gain")
        denoise_rank_cfg = cfg.get("denoise_rank")
        beta_deshared_cfg = cfg.get("beta_deshared")
        adaptive_gain_cfg = cfg.get("adaptive_gain")
        zero_unseen_cfg = cfg.get("zero_unseen")

    # Build prior_kwargs BEFORE fit so james_stein/temperature/basal_gate affect fitting
    prior_kwargs: dict = {}
    if shared_mix_cfg is not None:
        prior_kwargs["shared_mix"] = float(shared_mix_cfg)
    if args.shared_mix is not None:
        prior_kwargs["shared_mix"] = float(args.shared_mix)
    if use_alpha_cfg is not None:
        prior_kwargs["use_alpha"] = bool(use_alpha_cfg)
    if args.no_alpha:
        prior_kwargs["use_alpha"] = False
    if basal_gate_cpm_cfg is not None:
        prior_kwargs["basal_gate_cpm"] = float(basal_gate_cpm_cfg)
    if james_stein_cfg is not None:
        prior_kwargs["james_stein"] = bool(james_stein_cfg)
    if default_conservation_cfg is not None:
        prior_kwargs["default_conservation"] = float(default_conservation_cfg)
    if temperature_cfg is not None:
        prior_kwargs["similarity_temperature"] = float(temperature_cfg)
    if denoise_rank_cfg is not None:
        prior_kwargs["denoise_rank"] = int(denoise_rank_cfg)
    if beta_deshared_cfg is not None:
        prior_kwargs["beta_deshared"] = float(beta_deshared_cfg)
    if zero_unseen_cfg is not None:
        prior_kwargs["zero_unseen"] = bool(zero_unseen_cfg)
    use_consistency_cfg = cfg.get("use_consistency") if args.config else None
    if use_consistency_cfg is not None:
        prior_kwargs["use_consistency"] = bool(use_consistency_cfg)

    t0 = time.time()
    pair_n = {p: int(n) for p, n in merged.n_cells.items()} if hasattr(merged, "n_cells") else {}
    source_weights_cfg = cfg.get("source_weights") if args.config else None
    min_cells_cfg = int(cfg.get("min_cells", 0)) if args.config and cfg.get("min_cells") is not None else 0
    model = HybridPredictor.fit(
        lfc_pairs=train_pairs,
        basals=basals,
        gene_names=gene_names,
        residual_cfg=ResidualConfig(epochs=args.epochs, seed=args.seed),
        select_weights=False,
        fit_residual=weights[1] > 0.0,
        stack_fn=stack_fn,
        prior_kwargs=prior_kwargs,
        pair_n=pair_n,
        source_weights=source_weights_cfg,
        min_cells=min_cells_cfg,
    )
    model.weights = weights
    model.calibration = calibration
    # Any post-fit overrides that were not in prior_kwargs (kept for backward compat)
    # Note: james_stein etc already handled pre-fit; post-fit toggling would not repopulate _gene_var
    if adaptive_gain_cfg is not None and adaptive_gain_cfg:
        # Build per-gene gain map from prior consistency + optional ESM boost
        from vcc_model.protein import build_per_gene_gain_map, load_esm_embeddings
        esm = None
        try:
            esm = load_esm_embeddings()
            if esm:
                print(f"[sub] ESM embeddings loaded: {len(esm)} genes, dim {next(iter(esm.values())).shape}")
        except Exception as e:
            print(f"[sub][warn] ESM load failed: {e}")
        pg = build_per_gene_gain_map(
            model.prior.consistency, gene_names,
            default_cons=model.prior.default_conservation,
            scale=float(adaptive_gain_cfg.get("scale", 0.5)) if isinstance(adaptive_gain_cfg, dict) else 0.5,
            esm_embeddings=esm,
            esm_boost=float(adaptive_gain_cfg.get("esm_boost", 0.15)) if isinstance(adaptive_gain_cfg, dict) else 0.15,
        )
        actual_scale = float(adaptive_gain_cfg.get("scale", 0.5)) if isinstance(adaptive_gain_cfg, dict) else 0.5
        # Merge into calibration
        base_cal = model.calibration
        model.calibration = DECalibration(
            threshold=base_cal.threshold, gain=base_cal.gain,
            power=base_cal.power, q_threshold=base_cal.q_threshold,
            per_gene_gain=pg,
        )
        print(f"[sub] adaptive per-gene gain enabled scale={actual_scale} ({len(pg)} genes)")
    elif per_gene_gain_cfg is not None:
        # Explicit per-gene map from config
        base_cal = model.calibration
        model.calibration = DECalibration(
            threshold=base_cal.threshold, gain=base_cal.gain,
            power=base_cal.power, q_threshold=base_cal.q_threshold,
            per_gene_gain={k: float(v) for k, v in per_gene_gain_cfg.items()},
        )
    w = model.weights
    print(f"[sub] fitted {len(train_pairs)} pairs in {time.time() - t0:.0f}s "
          f"weights=(stat={w[0]:.2f}, resid={w[1]:.2f}, stack={w[2]:.2f}) "
          f"calib={model.calibration.as_dict()} shared_mix={model.prior.shared_mix} "
          f"use_alpha={getattr(model.prior, 'use_alpha', True)} "
          f"basal_gate_cpm={getattr(model.prior, 'basal_gate_cpm', 0.0)} "
          f"james_stein={getattr(model.prior, 'james_stein', False)}")

    # ---- predict per context (streamed straight to disk) ----
    from vcc_model.submission import StreamingSubmissionWriter, validate_h5ad_file_light

    ctx_files = sorted(controls_dir.glob("context_*.h5ad"))
    if not ctx_files:
        raise SystemExit(f"no context_*.h5ad files under {controls_dir}")
    if args.only_contexts:
        want = {str(c) for c in args.only_contexts}
        ctx_files = [f for f in ctx_files
                     if f.stem.removeprefix("context_").removeprefix("context-") in want]
        if not ctx_files:
            raise SystemExit(f"--only-contexts {sorted(want)} matched no files")
        print(f"[sub] restricted to contexts: {[f.stem for f in ctx_files]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with StreamingSubmissionWriter(args.out, gene_names, append=args.append) as writer:
        for f in ctx_files:
            ctx = f.stem.removeprefix("context_").removeprefix("context-")
            ctrl_sparse, n_ctrl = load_context_controls(f, gene_names)
            basal = sparse_log_mean_cpm(ctrl_sparse, spec.DE_TARGET_SUM)
            # Stable seed: hash() is salted per process and would break
            # run-to-run reproducibility of the submission.
            ctx_seed = args.seed + (zlib.crc32(ctx.encode("utf-8")) % 100000)
            rng_ctx = np.random.default_rng(ctx_seed)

            pool_n = min(args.gen_pool, n_ctrl)
            pool_idx = np.sort(rng_ctx.choice(n_ctrl, size=pool_n, replace=False))
            base_pool = np.asarray(ctrl_sparse[pool_idx].todense(), dtype=np.int64)
            ctrl_prob = np.asarray(ctrl_sparse.sum(axis=0), dtype=np.float64).ravel()
            ctrl_prob = ctrl_prob / max(float(ctrl_prob.sum()), 1e-12)

            t_ctx = time.time()
            # Panel-mean removal (AtlasShift common_subtract). Off by default so
            # the v29/v38 path stays bit-exact. v41 isolated this change and did
            # not improve overall score, so it is not by itself the source of
            # v40's higher PDS; treat it as one candidate component only.
            lfc_all = None
            if args.panel_mean != 0.0:
                lfc_all = np.stack([model.predict_lfc(t, basal) for t in targets])
                probe = lfc_all.copy()
                for kk, t in enumerate(targets):
                    gi = gene_index.get(t) if gene_index else None
                    if gi is not None:
                        probe[kk, gi] = np.nan
                colmean = np.nan_to_num(np.nanmean(probe, axis=0))
                lfc_all -= args.panel_mean * colmean[None, :]
                print(f"[sub] {ctx}: panel-mean {args.panel_mean} "
                      f"(mean|cm|={np.abs(colmean).mean():.4f})")

            for k, tgt in enumerate(targets):
                pick = rng_ctx.integers(0, pool_n, size=args.cells_per_pert)
                if k < args.start_index:
                    continue
                if args.end_index is not None and k >= args.end_index:
                    break
                lfc = lfc_all[k] if lfc_all is not None else model.predict_lfc(tgt, basal)
                block = base_pool[pick]
                if args.bulk_boost != 0.0:
                    d0 = block.sum(axis=1, dtype=np.float64)
                    dbar = max(float(d0.mean()), 1.0)
                    factor = 1.0 + args.bulk_boost * (d0 / dbar - 1.0)
                    mult = np.exp2(np.clip(lfc[None, :] * factor[:, None], -6.0, 6.0))
                else:
                    mult = np.exp2(np.clip(lfc, -6.0, 6.0))
                counts = (
                    poisson_thin(block, mult, rng_ctx) if args.decoder == "poisson"
                    else multinomial_thin(block, mult, rng_ctx)
                )
                if args.bulk_repair:
                    desired = bulk_probability_from_lfc(
                        ctrl_prob, lfc, bulk_gain=args.bulk_gain
                    )
                    if promoter_pairs:
                        desired = apply_promoter_ceiling(
                            desired, ctrl_prob, tgt, promoter_pairs,
                            fraction=args.promoter_fraction,
                        )
                    depths = counts.sum(axis=1, dtype=np.int64)
                    counts = repair_bulk_totals(counts, depths[:, None] * desired[None, :])
                writer.add_block(counts, [tgt] * args.cells_per_pert, [ctx] * args.cells_per_pert)
                if (k + 1) % 50 == 0:
                    print(f"[sub] {ctx}: {k + 1}/{len(targets)} targets "
                          f"(nnz so far {writer._nnz:,}) ({time.time() - t_ctx:.0f}s)")
            print(f"[sub] {ctx}: predicted {len(targets)} x {args.cells_per_pert} cells "
                  f"({time.time() - t_ctx:.0f}s)")

    if args.skip_validation:
        print("[sub] validation skipped (--skip-validation)")
    else:
        if args.validate_contexts:
            val_contexts = tuple(args.validate_contexts)
        elif args.only_contexts:
            val_contexts = tuple(
                sorted({f.stem.removeprefix("context_").removeprefix("context-") for f in ctx_files})
            )
        else:
            val_contexts = tuple(contexts_present)
        problems = validate_h5ad_file_light(
            args.out,
            gene_names,
            contexts=val_contexts,
            perts_per_context=len(targets),
            cells_per_pert=args.cells_per_pert,
        )
        if problems:
            raise ValueError("submission failed self-validation:\n" + "\n".join(problems))
    print(f"[sub] wrote + validated streaming submission -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.prep:
        run_vcc_prep(args.out, gene_csv)


if __name__ == "__main__":
    main()
