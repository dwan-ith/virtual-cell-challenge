# VCC 2026 — Zero-Shot Perturbation Response Prediction

System for the **Arc Institute Virtual Cell Challenge 2026**: predict CRISPRi
knockdown responses in three unseen cell lines (contexts A/B/C validation,
D/E/F final) from their unperturbed control cells alone — 300 targets × 400
cells × 3 contexts, raw integer counts on an 18,533-gene axis.

Timeline: final test set released Oct 22, submissions due Nov 5, 2026.

---

## 0. What changed in the v2 repair pass (2026-08-23)

The v1 pipeline never produced an honest real-data number: it OOMed at fit
(20075 x 18533 dense copies on a 16 GB machine) and its proxy benchmark showed
a diagnostic pattern — **pds scaled ~1.0 while every DE member scored below the
baseline**. Root causes found and fixed:

1. **Double shrinkage killed all DE calls.** Measured leave-one-context-out
   transfer slopes onto truly unseen lines are tiny (H1: 0.067, K562 GW:
   0.057); the prior multiplied them in AND the calibration grid topped out at
   gain 1.5, so every predicted |lfc| landed under Wilcoxon call threshold.
   Scale authority now lives in one place: the exact six-metric calibration
   search (gain grid extended down to 0.15), not the prior's alpha. PDS is
   cosine-scale-free, which is exactly why it alone survived — the benchmark
   pattern was the tell.
2. **Consistency shrinkage was inert.** 272/300 panel targets appear in exactly
   ONE source context, so cross-context sign agreement was vacuously 1.0.
   Replaced with two-tier consistency: within-line dataset agreement (K562 GW
   vs essentialome vs Tahoe-K562) measures reliability; cross-line agreement
   measures conservation; single-observation programs get a calibrated default
   trust instead of implicit full marks.
3. **Weight selection was blind to PDS** and its result was then ignored by
   make_submission (hardcoded weights). Selection now folds an official-style
   PDS rank proxy into the LOCO objective, also scans the shared-program mix
   (a common additive component drags PDS toward the 0.5 tie baseline), and
   the submission consumes the LOCO artifact verbatim (--config).
4. **Calibration tuned against a misaligned surrogate.** Four of six members
   are count-level Wilcoxon tests; whether a shift becomes a "call" depends on
   per-gene power no LFC-space surrogate models. tune_exact is now a two-round
   exact search (small-budget screen of the full grid -> refined rescoring of
   the top candidates) with cached reference DE tables, plus a per-target
   adaptive floor (q_threshold) because program magnitudes vary hugely.
5. **Memory/scale engineering** (all fatal at full panel scale here):
   residual training subsamples pairs into one bounded float32 block; lazy
   basal-correlation neighbor features; cached per-context mean programs;
   float32 views instead of float64 row copies; Poisson-resampling decoder
   (vectorized, realistic depth noise, density-cap safe); streaming h5ad
   writer — the old builder accumulated all 900 dense blocks (~53 GB) before
   writing; now blocks go straight to disk in anndata's native CSR layout with
   a light no-full-load rule validator; deterministic seeds everywhere
   (hash() was salted).

Honest holdout status: see artifacts/loco_v5.json once the run completes
(holdout vcc2025/H1 — the only true unseen-cell-line context with exact truth).

---

## 1. What the scoring actually rewards (and how we exploit that)

Scored with `cell-eval2` (`vcc2026` profile): six members, each scaled as
`s = (u − b)/(r − b)` where **b** = mean-response baseline and **r** =
split-half replicate of the real experiment, then averaged per context.
0 ≈ baseline, 1 ≈ replicate; unclamped (except mse ∈ [0,1], nmae floor −6).

| member | b | r | what moves it |
|---|---|---|---|
| `pds` | .500 | .93–.98 | perturbation-*unique* delta directions |
| `mse` | ~.99 | .03–.05 | pseudobulk accuracy beyond mean-response |
| `fid` | ~.51 | .80–.83 | correct-sign significant calls ≥ n_real |
| `reach` | ~.07 | .96–.98 | depth of ≥90%-pure ranked prefix |
| `jac` | ~.03 | .38–.42 | exact significant-set overlap |
| `nmae` | ~1.0 | ~.40 | LFC magnitudes on ref-significant genes |

Consequences that drive every design choice:

1. **Four of six members are Wilcoxon tests of OUR emitted cells vs reference
   controls.** With 400-vs-~18k cells the power is huge, so any consistent
   nonzero shift gets called "significant". False positives come from noise in
   our predicted LFC vector → we shrink it where it's unreliable
   (`vcc_model/calibrate.py`, tuned against held-out source contexts).
2. **PDS needs perturbation-specific signal**; a shared program scores exactly
   0.5 (= baseline). Transfer priors + residual net supply specificity;
   magnitude calibration sharpens cosine geometry (TransPert's lever).
3. **mse is unsolved on the live board** (a rank-2 entry had mse = 0). Our
   thinning decoder preserves realistic dispersion, earning jackknife credit.
4. Never optimize one metric — 2025's leaderboard was broken by variance-
   inflating transforms on a narrow metric surface.

## 2. Architecture

```
public Perturb-seq (VCC-2025 H1, Replogle K562/RPE1, ...)
        │  streamed pseudobulk: per-(context,target) observed LFC vectors
        ▼
┌───────────────────────────── HybridPredictor (LFC space) ───────────────────┐
│  stat prior      similarity-weighted pooling across source contexts,       │
│  (TransPert-class) sign-consistency shrinkage, LOCO magnitude α            │
│  residual net    MLP[basal-PCs ∥ prior-LFC basis coords] + gene embedding  │
│                  → SVD-basis coefficients (context-conditioned correction) │
│  Stack ICL       cached cross-context deltas from arc-stack generation     │
│                  (prompt=perturbed source cells, query=target controls)    │
│  blend weights   leave-one-context-out selection over the 3 components     │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼  DECalibration  lfc' = sign(lfc)·gain·max(|lfc|−τ,0)^p  (LOCO-tuned)
        ▼  decoder: multinomial thinning of sampled control cells by 2^lfc'
           (preserves library sizes, support, covariance, density cap)
        ▼  submission.py: rule-checked AnnData (400/pert, CSR int, no ctrls)
```

Evidence base: all three 2025 winners were hybrids (BioMap xTrimoSCPerturb,
XLearning Lab pseudo-bulk FCN, TransPert pure-statistical 3rd-of-1200);
PRiMeFlow (Generalist) amplified perturbation deltas for PDS; Arc shipped
**Stack** (217M-param ICL foundation model, `pip install arc-stack`) three
weeks before launch with a prompt/query pattern identical to this task.

## 3. Layout

```
vcc_model/
  spec.py         official constants (panel shape, metric params) — verified
  countsim.py     CPM / pseudobulk / log profiles / BH / thinning decoders
  de_table.py     Wilcoxon DE table exactly as section 3 of the cell-eval brief
  metrics.py      faithful local six-metric implementation + baseline/replicate anchors
                  (+ build_real_pairs/attach_pred_table for cached-truth tuning)
  prior.py        cross-context statistical transfer (two-tier consistency,
                  lazy neighbor fallback, PDS-aware shared-program mix)
  residual.py     basal-conditioned low-rank residual network (memory-bounded fit)
  ensemble.py     HybridPredictor: blend, PDS-aware LOCO weight selection, decode
  calibrate.py    DECalibration (soft-threshold/gain/power/q_threshold)
                  + two-round exact six-metric tuner
  dataset.py      streaming h5ad -> PseudobulkStore (+ control subsamples)
  submission.py   rule validation, in-memory writer (small runs),
                  StreamingSubmissionWriter + light validator (full panel)
  synthesize.py   sparse multi-context synthetic CRISPRi (harness realism)
scripts/
  download_data.py          vcc CLI controls + atlas fetch guide
  build_stores.py           raw h5ads -> stores/*.npz (+ __controls.npz)
  build_bulk_stores.py      Replogle bulk pseudobulk h5ads -> exact stores
  build_tahoe_stores.py     replogle-nadig deltas -> pseudo-LFC stores
  build_h1_store.py         2025 Test mirror -> exact H1 store
  train_eval_loco.py        fit on N-1 contexts, score held-out with anchors;
                            writes artifacts/loco_config.json for submission
  make_submission.py        full-data fit -> predict A/B/C -> streamed h5ad
  chain_submit.py           wait for LOCO config -> unattended submission build
  smoke_test.py             end-to-end synthetic proof (asserts copy<hybrid<oracle)
```

## 4. Run it

```bash
pip install -r requirements.txt

# 0) sanity: whole pipeline on synthetic data (~2 min CPU)
python scripts/smoke_test.py

# 1) challenge inputs (needs vcc login once)
python scripts/download_data.py --controls-dir data/vcc2026

# 2) training stores from public screens (see script docstrings for paths)
python scripts/build_bulk_stores.py       # Replogle K562 GW/essential + RPE1
python scripts/build_tahoe_stores.py      # replogle-nadig deltas, 4 lines
python scripts/build_h1_store.py          # 2025 Test mirror (exact truth)

# 3) honest zero-shot benchmark + config selection (weights + calibration)
python scripts/train_eval_loco.py --stores artifacts/stores/*.npz \
    --holdout vcc2025/H1 --out artifacts/loco_v5.json \
    --config-out artifacts/loco_config.json --exact-calibration

# 4) submission (consumes the LOCO config verbatim; streams to disk)
python scripts/make_submission.py --stores artifacts/stores/*.npz \
    --controls-dir data/vcc2026 --out artifacts/submission/prediction.h5ad \
    --config artifacts/loco_config.json

# 5) package + submit
vcc prep artifacts/submission/prediction.h5ad -g data/vcc2026/gene_names.csv \
    --perts data/vcc2026/pert_counts.csv -o artifacts/submission/prediction.vcc
vcc submit artifacts/submission/prediction.vcc -m "hybrid-v2" --wait
```

Optional Stack member: `pip install arc-stack`, download
`arcinstitute/Stack-Large-Aligned` from HuggingFace, run
`scripts/run_stack_generation.py ... --cache-dir artifacts/stack_cache`, then
pass `--stack-cache` to training/submission scripts — LOCO picks its weight.

## 5. Validation discipline (the part that wins)

* **Local truth is synthesized** (thinning control cells by true LFCs) so the
  six-metric harness runs offline; oracle lands ≈1 and control-copy <0 in the
  smoke test — the same signatures the official anchors produce.
* **LOCO on real contexts is the decision signal.** The live leaderboard
  (3 lines, daily limit, one-in-flight) is only a sanity check.
* Anchor spans on pathological data collapse and explode scaled scores; the
  debug path prints raw values + anchors side-by-side before anything is trusted.
* Any gain that looks too good gets checked for metric gaming (variance
  inflation, label swaps, leakage via context columns).

## 6. Known gaps / next steps

- Real-data LOCO run pending store downloads (Replogle K562/RPE1 + VCC-2025).
- Stack cache empty until GPU generation run; weight auto-selects to 0 meanwhile.
- Baseline anchor locally omits Arc's knockdown-efficiency filter (minor skew).
- `expr_mse` may need a dedicated magnitude head if it stays ≤0 after LOCO.
- Hybrid `fid` ran negative in smoke (calibration trades coverage for precision);
  if it persists on real LOCO, re-tune with `--exact-calibration`, which
  optimizes the true six-metric mean including fid directly.
- Final phase (D/E/F): rerun steps 4–5 with the new bundle; labels differ.
