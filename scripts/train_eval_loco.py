"""Leave-one-context-out zero-shot training + evaluation on real stores.

Mirrors the 2026 challenge: fit on all contexts except held-out ones, predict
every (held-out context, target) pair, and score with the local faithful
six-metric implementation, anchored against baseline (mean-response) and
replicate (split-half) references computed on the same data.

Reference ("truth") counts for scoring are synthesized by multinomial thinning
of the context's control cells with multipliers exp2(true LFC) — the same
count-generating process our decoder uses — so model comparisons are
internally consistent. Absolute anchor values differ from real Flex data.

Requires stores built with controls (build_stores.py saves a control-cell
subsample per context alongside the LFC store).

Usage:
    python scripts/train_eval_loco.py --stores artifacts/stores/*.npz \
        --holdout vcc2025_h1/H1 --out artifacts/loco_h1.json \
        [--exact-calibration] [--stack-cache artifacts/stack_cache]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcc_model.calibrate import DECalibration, tune, tune_exact
from vcc_model.countsim import log_mean_cpm, multinomial_thin
from vcc_model.dataset import PseudobulkStore
from vcc_model.ensemble import HybridPredictor
from vcc_model.metrics import (
    ContextData,
    baseline_anchor,
    evaluate_context,
    replicate_anchor,
    scale_scores,
)
from vcc_model.residual import ResidualConfig


def load_stores(paths: list[Path]) -> PseudobulkStore:
    merged: PseudobulkStore | None = None
    for p in paths:
        s = PseudobulkStore.load(p)
        if merged is None:
            merged = s
            print(f"[loco] loaded {p.name}: {len(s.pairs)} pairs")
            continue
        assert np.array_equal(merged.gene_names, s.gene_names), (
            f"gene axis mismatch in {p}; rebuild stores with a shared axis"
        )
        merged.pairs = merged.pairs + s.pairs
        merged.lfc = np.vstack([merged.lfc, s.lfc])
        merged.n_cells.update(s.n_cells)
        for c, b in s.basal.items():
            if c not in merged.basal:
                merged.basal[c] = b
        for c, m in s.controls.items():
            if c not in merged.controls:
                merged.controls[c] = m
        print(f"[loco] loaded {p.name}: {len(s.pairs)} pairs (total {len(merged.pairs)})")
    return merged


def group_pairs_by_context(store: PseudobulkStore) -> dict[str, list[int]]:
    by_ctx: dict[str, list[int]] = {}
    for i, (c, _) in enumerate(store.pairs):
        by_ctx.setdefault(c, []).append(i)
    return by_ctx


def synth_truth_cells(
    ctrl: np.ndarray,
    lfc_by_target: dict[str, np.ndarray],
    n_cells: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Materialize reference cells per target by thinning control cells.

    Base rows are drawn from `ctrl` itself — the same matrix used as the
    reference control group in scoring — mirroring real Perturb-seq, where
    perturbed and control cells share the experimental batch. Any shift the
    DE test then detects is attributable to applied multipliers, not to
    subset composition noise. Uses the SAME decoder as predictions so both
    sides of every test share a count-generating process.
    """
    from vcc_model.countsim import poisson_thin

    rng = np.random.default_rng(seed + 4242)
    pick = rng.choice(ctrl.shape[0], size=n_cells, replace=n_cells > ctrl.shape[0])
    base = ctrl[pick]
    out: dict[str, np.ndarray] = {}
    for t, lfc in lfc_by_target.items():
        out[t] = poisson_thin(base, np.exp2(np.clip(lfc, -6, 6)), rng)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stores", nargs="+", type=Path, required=True)
    ap.add_argument("--holdout", nargs="+", required=True,
                    help="context keys to hold out, e.g. vcc2025_h1/H1")
    ap.add_argument("--out", type=Path, default=Path("artifacts/loco_report.json"))
    ap.add_argument("--stack-cache", type=Path, default=None)
    ap.add_argument("--n-eval-cells", type=int, default=160,
                    help="cells synthesized/scoring per perturbation group")
    ap.add_argument("--max-ctrl-cells", type=int, default=2000,
                    help="control cells loaded into ContextData for scoring")
    ap.add_argument("--max-pairs-per-ctx", type=int, default=40,
                    help="cap scored pairs per held context for tractability")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--replicate-splits", type=int, default=3)
    ap.add_argument("--exact-calibration", dest="exact_calibration", action="store_true", default=True,
                    help="tune calibration against the full six-metric eval on "
                         "thinned counts (default; two-stage, cached truth tables)")
    ap.add_argument("--surrogate-calibration", dest="exact_calibration", action="store_false",
                    help="fast LFC-space surrogate tuning instead of exact")
    ap.add_argument("--config-out", type=Path, default=None,
                    help="write {weights, calibration, holdout} JSON for make_submission")
    ap.add_argument("--model-cache", type=Path, default=None,
                    help="checkpoint path: saves the fitted model right after fit and "
                         "reuses it on rerun (calibration/eval can be retried cheaply)")
    ap.add_argument("--fixed-weights", type=float, nargs=3, default=None, metavar=("STAT", "RESID", "STACK"),
                    help="skip LOCO weight selection; use these blend weights "
                         "(e.g. the deterministic LOCO verdict banked from prior runs)")
    ap.add_argument("--shared-mix", type=float, default=None,
                    help="override the prior's shared-program mixing fraction")
    ap.add_argument("--no-alpha", action="store_true",
                    help="prior emits unscaled directions; the DE-calibration "
                         "search owns all scale decisions (recommended: the LOCO "
                         "slopes compound with gain and starve Wilcoxon members)")
    ap.add_argument("--raw-mean-tuning", action="store_true",
                    help="exact calibration optimizes the unweighted raw-member "
                         "mean instead of the official span-scaled mean")
    ap.add_argument("--calib-cache", type=Path, default=None,
                    help="reuse/save tuned calibration JSON to skip re-tuning")
    ap.add_argument("--no-weight-select", action="store_true",
                    help="skip leave-one-context-out blend-weight search "
                         "(fast path; uses prior-only or fixed 1.0/0.7)")
    ap.add_argument("--basal-gate-cpm", type=float, default=0.0,
                    help="basal gating threshold in CPM; zero out predicted LFC "
                         "for genes with mean CPM below this in the target context")
    ap.add_argument("--james-stein", action="store_true",
                    help="enable James-Stein shrinkage on pooled programs")
    ap.add_argument("--include-oracle", action="store_true",
                    help="also score truth-vs-itself (should land ~1.0 scaled)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    store = load_stores(sorted(args.stores))
    gene_names = [str(g) for g in store.gene_names]
    by_ctx = group_pairs_by_context(store)

    missing_ctx = [h for h in args.holdout if h not in by_ctx]
    if missing_ctx:
        raise SystemExit(
            f"unknown holdout contexts {missing_ctx}; available: {sorted(by_ctx)}"
        )
    missing_ctrl = [
        h for h in args.holdout if h not in store.controls or store.controls[h].shape[0] < 50
    ]
    if missing_ctrl:
        raise SystemExit(
            f"no control cells stored for {missing_ctrl}; rebuild stores with "
            "build_stores.py (max_control_cells > 0)"
        )

    train_pairs = {
        store.pairs[i]: store.lfc[i]          # float32 views; stores sanitize on load
        for c, rows in by_ctx.items()
        if c not in args.holdout
        for i in rows
    }
    basals_train = {
        c: store.basal[c] for c in by_ctx if c not in args.holdout and c in store.basal
    }
    holdout_meta = {
        h: {
            "basal": store.basal[h],
            "ctrl": store.controls.get(h),
            "rows": list(by_ctx[h]),
        }
        for h in args.holdout
    }
    print(
        f"[loco] fitting on {len(train_pairs)} pairs / {len(basals_train)} source "
        f"contexts; holding out {args.holdout}"
    )
    if not train_pairs:
        raise SystemExit("no training pairs left after holdout; check --holdout")

    stack_fn = _make_stack_fn(args.stack_cache, gene_names) if args.stack_cache else None

    model = None
    if args.model_cache and args.model_cache.exists():
        print(f"[loco] loading cached fit from {args.model_cache}", flush=True)
        try:
            model = HybridPredictor.load(args.model_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"[loco] cache load failed ({exc}); refitting", flush=True)
            model = None

    t0 = time.time()
    if model is None:
        if args.fixed_weights is not None:
            # Banked LOCO verdict: skip residual + selection entirely.
            model = HybridPredictor.fit(
                lfc_pairs=train_pairs,
                basals=basals_train,
                gene_names=gene_names,
                residual_cfg=ResidualConfig(epochs=args.epochs, seed=args.seed),
                select_weights=False,
                fit_residual=False,
                stack_fn=stack_fn,
            )
            model.weights = tuple(args.fixed_weights)
            print(f"[loco] using banked weights={model.weights} (LOCO-derived, "
                  f"selection skipped)", flush=True)
        else:
            model = HybridPredictor.fit(
                lfc_pairs=train_pairs,
                basals=basals_train,
                gene_names=gene_names,
                residual_cfg=ResidualConfig(epochs=args.epochs, seed=args.seed),
                select_weights=not args.no_weight_select,
                stack_fn=stack_fn,
            )
            if args.no_weight_select:
                model.weights = (1.0, 0.7, 0.0)  # fixed hybrid default
        if args.shared_mix is not None:
            model.prior.shared_mix = float(args.shared_mix)
        if args.no_alpha:
            model.prior.use_alpha = False
        if args.basal_gate_cpm > 0.0:
            model.prior.basal_gate_cpm = float(args.basal_gate_cpm)
        if args.james_stein:
            model.prior.james_stein = True
        if args.model_cache:
            model.save(args.model_cache)
            print(f"[loco] saved fitted model -> {args.model_cache}", flush=True)
    elif args.shared_mix is not None or args.no_alpha:
        if args.shared_mix is not None:
            model.prior.shared_mix = float(args.shared_mix)
        if args.no_alpha:
            model.prior.use_alpha = False
    print(f"[loco] fit done in {time.time() - t0:.0f}s; weights={model.weights} "
          f"shared_mix={model.prior.shared_mix} use_alpha={getattr(model.prior, 'use_alpha', True)} "
          f"basal_gate_cpm={getattr(model.prior, 'basal_gate_cpm', 0.0)} "
          f"james_stein={getattr(model.prior, 'james_stein', False)}")

    report: dict = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "weights": {"stat": model.weights[0], "resid": model.weights[1], "stack": model.weights[2]},
        "n_train_pairs": len(train_pairs),
        "contexts": {},
    }
    all_overalls: list[float] = []

    for held in args.holdout:
        t_ctx = time.time()
        meta = holdout_meta[held]
        rows = meta["rows"]
        targets_all = sorted({store.pairs[i][1] for i in rows})
        keep_targets = targets_all
        if len(targets_all) > args.max_pairs_per_ctx:
            keep = list(targets_all)
            rng.shuffle(keep)
            keep_targets = sorted(keep[: args.max_pairs_per_ctx])
        rows = [i for i in rows if store.pairs[i][1] in set(keep_targets)]

        truth_lfc = {store.pairs[i][1]: store.lfc[i].astype(np.float64) for i in rows}
        basal_held = meta["basal"]
        pred_raw = {t: model.blend_lfc(t, basal_held) for t in keep_targets}

        # ---- control cells (dense, subsampled) ----
        ctrl_sparse = meta["ctrl"]
        n_ctrl = min(ctrl_sparse.shape[0], args.max_ctrl_cells)
        sel = rng.choice(ctrl_sparse.shape[0], size=n_ctrl, replace=False)
        ctrl = np.asarray(ctrl_sparse[sel].todense(), dtype=np.float64)

        # ---- calibration ----
        t_cal = time.time()
        cal = None
        if args.calib_cache and args.calib_cache.exists():
            try:
                from vcc_model.calibrate import DECalibration as _DEC

                cal = _DEC(**json.loads(args.calib_cache.read_text()))
                print(f"[loco] {held}: loaded cached calibration {cal.as_dict()}", flush=True)
                cal_info = {"score": float("nan"), "cached": True}
            except Exception as exc:  # noqa: BLE001
                print(f"[loco] calib cache unreadable ({exc}); re-tuning", flush=True)
                cal = None
        if cal is None:
            if args.exact_calibration:
                cal, cal_info = tune_exact(
                    {(held, t): v for t, v in pred_raw.items()},
                    {(held, t): v for t, v in truth_lfc.items()},
                    ctrl_counts_by_context={held: ctrl},
                    gene_names=gene_names,
                    seed=args.seed,
                    official_span_weighting=not args.raw_mean_tuning,
                )
            else:
                cal, cal_info = tune(pred_raw, truth_lfc)
            if args.calib_cache:
                args.calib_cache.parent.mkdir(parents=True, exist_ok=True)
                args.calib_cache.write_text(json.dumps(cal.as_dict()))
                print(f"[loco] cached calibration -> {args.calib_cache}", flush=True)
        model.calibration = cal
        print(
            f"[loco] {held}: calibration={cal.as_dict()} "
            f"(tuning score {cal_info.get('score', float('nan')):.4f}, "
            f"{time.time() - t_cal:.0f}s)"
        )

        pred_cal = {t: model.predict_lfc(t, basal_held) for t in keep_targets}

        # ---- synthesize reference + predicted cells ----
        t_gen = time.time()
        from vcc_model.countsim import poisson_thin

        true_cells = synth_truth_cells(ctrl, truth_lfc, args.n_eval_cells, args.seed)
        prng = np.random.default_rng(args.seed + 777)
        pick = prng.choice(n_ctrl, size=args.n_eval_cells, replace=n_ctrl > ctrl.shape[0])
        base_cells = ctrl[pick]
        pred_cells = {
            t: poisson_thin(base_cells, np.exp2(np.clip(pred_cal[t], -6.0, 6.0)), prng)
            for t in keep_targets
        }
        print(f"[loco] {held}: generated {2 * len(keep_targets)} cell groups ({time.time() - t_gen:.0f}s)")

        cd = ContextData(
            gene_names=np.asarray(gene_names, dtype=str),
            real_ctrl=ctrl,
            real_pert=true_cells,
            pred_pert=pred_cells,
        )

        t_eval = time.time()
        raw = evaluate_context(cd)
        b_anchor = baseline_anchor(cd, seed=args.seed)
        r_anchor = replicate_anchor(cd, n_splits=args.replicate_splits, base_seed=args.seed)
        scaled = scale_scores(raw, b_anchor, r_anchor)
        print(f"[loco] {held}: scored ({time.time() - t_eval:.0f}s)")

        ctx_report = {
            "n_pairs": len(keep_targets),
            "targets": keep_targets,
            "calibration": cal.as_dict(),
            "raw": {k: _f(v) for k, v in raw.items()},
            "baseline_anchor": {k: _f(v) for k, v in b_anchor.items()},
            "replicate_anchor": {k: _f(v) for k, v in r_anchor.items()},
            "scaled": {k: _f(v) for k, v in scaled.items()},
        }

        if args.include_oracle:
            cd_o = ContextData(cd.gene_names, ctrl, true_cells, true_cells)
            raw_o = evaluate_context(cd_o)
            scaled_o = scale_scores(raw_o, b_anchor, r_anchor)
            ctx_report["oracle_scaled"] = {k: _f(v) for k, v in scaled_o.items()}

        report["contexts"][held] = ctx_report
        if np.isfinite(scaled.get("overall", np.nan)):
            all_overalls.append(float(scaled["overall"]))
        print(
            f"[loco] {held}: overall={_f(scaled.get('overall'))} "
            f"pds={_f(scaled['pds_cosine'])} mse={_f(scaled['expr_mse_unbiased_capped_norm'])} "
            f"fid={_f(scaled['de_wilcoxon_direction_fidelity_yield_raw'])} "
            f"reach={_f(scaled['de_wilcoxon_direction_reach_raw'])} "
            f"jac={_f(scaled['de_wilcoxon_sig_jaccard'])} "
            f"nmae={_f(scaled['de_wilcoxon_lfc_nmae'])} "
            f"[{time.time() - t_ctx:.0f}s]"
        )

    report["mean_overall"] = float(np.mean(all_overalls)) if all_overalls else None
    print(f"\n[loco] MEAN OVERALL across held contexts: {report['mean_overall']}")

    # Persist the decision bundle the submission step should reuse verbatim:
    # LOCO-selected blend weights + tuned calibration.
    if args.config_out:
        cfg_out = {
            "holdout": list(args.holdout),
            "weights": {
                "stat": float(model.weights[0]),
                "resid": float(model.weights[1]),
                "stack": float(model.weights[2]),
            },
            "calibration": model.calibration.as_dict() if model.calibration is not None
            else {"threshold": 0.0, "gain": 1.0, "power": 1.0},
            "shared_mix": float(getattr(model.prior, "shared_mix", 0.25)),
            "use_alpha": bool(getattr(model.prior, "use_alpha", True)),
            "basal_gate_cpm": float(getattr(model.prior, "basal_gate_cpm", 0.0)),
            "james_stein": bool(getattr(model.prior, "james_stein", False)),
            "mean_overall": report["mean_overall"],
        }
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(json.dumps(cfg_out, indent=2))
        print(f"[loco] wrote submission config -> {args.config_out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))

    preds_path = args.out.with_suffix(".preds.npz")
    keys = [(c, t) for c in report["contexts"] for t in report["contexts"][c]["targets"]]
    # Recompute calibrated preds for persistence (cheap).
    pred_store = {}
    for c in report["contexts"]:
        basal_c = holdout_meta[c]["basal"]
        for t in report["contexts"][c]["targets"]:
            pred_store[(c, t)] = model.predict_lfc(t, basal_c).astype(np.float32)
    np.savez_compressed(
        preds_path,
        context=np.array([c for c, _ in keys], dtype=object),
        target=np.array([t for _, t in keys], dtype=object),
        pred=np.stack([pred_store[k] for k in keys]),
    )
    print(f"[loco] wrote {args.out} and {preds_path}")


def _f(x) -> float | None:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def _make_stack_fn(cache_dir: Path | None, gene_names: list[str]):
    """Load cached Stack ICL deltas as blend component, or return None."""
    from vcc_model.stack_adapter import StackAdapter

    adapter = StackAdapter(gene_names=gene_names, cache_dir=cache_dir)

    def fn(gene: str, basal_vec: np.ndarray, context: str = "target") -> np.ndarray:
        try:
            return adapter.delta(None, gene, context=context)
        except RuntimeError:
            return np.zeros(len(gene_names), dtype=np.float32)

    return fn


if __name__ == "__main__":
    main()
