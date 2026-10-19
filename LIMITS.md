# Critical analysis & repair log — VCC 2026 pipeline (2026-08-23)

Deep review of the zero-shot CRISPRi prediction system: every limitation found,
what was fixed, what was measured, and what remains open.

---

## 1. The problem, stated precisely

Predict, for three hidden cell lines (A/B/C validation; D/E/F final), the
Perturb-seq response to 300 CRISPRi targets from the unperturbed control cells
alone. Submission = 400 integer-count cells per (context, target) on a fixed
18,533-gene axis. Score = mean of six members, each scaled `s=(u-b)/(r-b)`
against a mean-response baseline (b) and a split-half replicate anchor (r):

| member | b | r | upside available |
|---|---|---|---|
| pds | 0.500 | 0.93–0.98 | large IF predictions stay perturbation-specific |
| mse | 0.99 (clamped [0,1]) | 0.03–0.05 | ≈ 0 (bounded both ways) |
| fid | 0.51 | 0.80–0.83 | moderate |
| reach | 0.05–0.10 | 0.96–0.98 | moderate |
| jac | 0.02–0.04 | 0.38–0.42 | largest relative headroom |
| nmae | 1.00 | 0.37–0.43 | small |

Key structural facts that drive everything:
- **4 of 6 members are Wilcoxon tests of our emitted cells vs reference
  controls.** At 400-vs-~18k cells, any consistent nonzero shift is called.
  Call selection = shaping the predicted LFC vector.
- **mse is clamped with baseline at ~0.99**: max scaled contribution ≈ +0.01,
  downside bounded near −1/6 of overall. Do not chase it.
- **PDS is cosine-based** → global scale cancels; specificity is all that matters.

## 2. Measured evidence base (real data)

- Panel coverage: **272/300 targets observed in exactly one source context**
  (Replogle K562 genome-wide); 28 in none; 8 also in H1. Cross-context
  consistency for panel targets was therefore vacuous.
- Transfer ceiling between sources (sign agreement on shared strong calls):
  same-line datasets 0.99 (K562-ess vs Tahoe-K562), cross-line 0.59–0.70,
  spearman 0.04–0.22. Measurement noise is NOT the limiting factor;
  biology is.
- LOCO magnitude slopes onto an unseen line are tiny: **H1 0.067**, K562-GW
  0.057 (vs 0.45–0.69 for within-line Tahoe folds). Any pipeline that
  pre-shrinks by this factor before call-thresholding emits nothing callable.

## 3. Limitations found — and their fixes

### M1. Double shrinkage killed all DE members (critical)
Prior applied α≈0.067·(consistency≈0.59); calibration grid gains topped out at
1.5 → post-transform |lfc| ≪ call threshold everywhere. Proxy benchmark
confirmed: pds scaled ≈ 1.0 while fid/reach/jac/nmae were negative (worse than
baseline). **Fix:** scale authority moved entirely to the exact six-metric
calibration search; gain grid extended down to 0.15 so the tuner can choose the
true end-to-end scale.

### M2. Consistency shrinkage inert for single-source targets (critical)
`consistency = sign-agreement across contexts` = 1.0 when only one context has
the target. **Fix:** two-tier consistency — within-line dataset agreement =
reliability; cross-line agreement = conservation; single-observation programs
get a calibrated default trust (`default_conservation=0.45`) instead of full
marks.

### M3. Weight selection blind to PDS, then ignored (critical)
LOCO selection optimized a per-pair LFC proxy that cannot see ranking collapse
(a shared additive component drives every pairwise distance together → PDS→0.5
while per-pair cosine looks fine). And `make_submission` hardcoded weights
anyway. **Fix:** selection objective mixes an official-style PDS rank proxy
(computed exactly as scored, in LFC space) with the pair proxy; scans the
shared-program mixing fraction jointly; submission consumes the LOCO artifact
verbatim via `--config`.

### M4. Calibration tuned against a misaligned surrogate (major)
Four members are count-level Wilcoxon tests; whether a shift becomes a call
depends on per-gene expression/power that no LFC-space surrogate models.
**Fix:** two-round exact tuner — screen the full grid (72 candidates) through
the real count-generation + local metric pipeline at small budget, refine top-5
at larger budget; reference DE tables computed once per round (cached truth);
per-target adaptive quantile floor added because program magnitudes vary hugely
across targets.

### M5. Fatal memory/scale engineering bugs (blocking)
- Residual fit stacked ALL pairs twice (2 × 20075×18533 float arrays) → OOM on
  16 GB. Fix: stratified subsample into ONE bounded float32 block
  (`max_pairs=12000`), no second copy, freed after training.
- Submission builder accumulated all 900 dense (400×18533) blocks ≈ **53 GB**
  before writing — never survivable. Fix: `StreamingSubmissionWriter` appends
  CSR blocks directly into anndata-native h5ad (roundtrip-verified) +
  `validate_h5ad_file_light` streams value checks without loading X.
- float64 row copies (~3 GB waste) → float32 views.
- ~600 MB basal-correlation matrix → lazy per-gene computation.
- Per-call re-stacking of whole contexts for shared programs (~730 MB churn per
  call!) → cached per-context means.
- Similarity weights recomputed 3× per prediction and per gene → hoisted to
  per-context `transfer_params` + batched predict paths.
- Weight-selection scoring recomputed blends per combo → component cache +
  batched residual inference + capped proxy gene samples.

### M6. Decoder realism/speed (moderate)
Per-cell Python multinomial loop froze library sizes and was slow at panel
scale. **Fix:** vectorized Poisson resampling — same expectation, realistic
depth noise, support-shrinking (density-cap safe), used identically for truth
and prediction in evaluation so tests stay comparable.

### M7. Determinism/reproducibility (minor but nasty)
`hash(ctx)` seeding is salted per process → submissions not reproducible.
**Fix:** crc32-based stable seeds; fixed RNG streams documented.

### M8. Stale/dangerous leftovers (minor)
`predict_vcc.py` references v1 classes (dead code); coverage_report ignored the
Replogle stores (misleading 0/300 coverage printout — actual: 272/300);
smoke test `--no-oracle` path KeyError; checkpoint loads broke when new prior
fields were added (backfill added). All fixed or flagged.

## 4. Honest validation design

- Local metrics follow `.research/vcc2026_metrics_brief.md` exactly (six
  members, anchors included); smoke test asserts control_copy < hybrid < oracle.
- Decision signal = LOCO on real stores with holdout `vcc2025/H1` (the only
  true unseen-cell-line context with exact pseudobulk truth). Calibration is
  tuned on the held-out fold's own thinning-generated truth; nothing from the
  holdout enters fitting except the final weight/calibration choice.
- Caveat: LOCO truth is Poisson-thinned controls, so absolute anchored numbers
  are optimistic on span-limited members; relative model comparisons remain
  valid.

## 5. Results

### 5.1 Honest local evaluation (holdout vcc2025/H1, thinned truth)

Three calibration points on the gain curve (`artifacts/loco_v6/v7/v8.json`):

| config | gain | local scaled overall | note |
|---|---|---|---|
| v6 | 1.8 | −0.150 | under-calling: fid coverage-punished |
| v7 | 5.0 | −1.173 | over-calling: nmae floored at −6 |
| v8 | 3.7 | −0.846 | span-weighted tuner's pick |

Local anchors mislead across configs because local spans differ from official
spans (reach: local 0.15 vs official 0.90). Projecting RAW member values
through OFFICIAL anchors (`scripts/project_official.py`) reordered the configs:
v6 +0.058, v8 +0.045, v7 −0.019 — which the live board then confirmed
qualitatively.

### 5.2 Official leaderboard (validation phase)

| submission | gain x q | Overall | pds | nmae | reach | jac | fid | rank |
|---|---|---|---|---|---|---|---|---|
| hybrid-v6 | 1.8 x 0 | **+0.086** | 0.352 | +0.071 | 0.133 | −0.014 | −0.025 | **46** |
| hybrid-v9 | 2.6 x 0.7 | +0.071 | 0.421 | −0.100 | 0.139 | −0.011 | −0.023 | 79 |
| hybrid-v8 | 3.7 x 0 | +0.021 | 0.464 | −0.470 | 0.157 | −0.005 | −0.018 | 122 |

The official gain response surface is smooth and interpretable: pds/reach
rise with gain, nmae falls, overall peaks near gain ~2.0-2.2. Projection
accuracy from local raw values through official anchors was directionally
correct (+0.058 predicted vs +0.086 actual for v6).

Leaderboard meta (ranks 11-20 at overall 0.10-0.13): pds 0.45-0.62 is the
differentiator; mse is 0.000 for everyone (clamp); jac/fid/nmae hover near
zero. Calibration tuning alone saturates near ~+0.09-0.10; beyond that the
binding constraint is transfer quality itself.

Shape-lever experiments on H1 fold (`scripts/test_shape_levers.py`):
low-rank program denoising DEGRADES pds (0.91 -> 0.64); de-shared subtraction
is flat. Rejected with evidence.

Official-space PDS evaluator (`scripts/test_pds_official.py`): local thinned-
truth PDS = 0.82 vs live-board 0.35 -- the gap is competition crowding and
real-context biology; local pds optimization beyond this point is blind and
must be steered by probes.

## 6. Continual-improvement loop (current phase)

Budget: 2 submissions/day, renewing midnight UTC, one in flight.

### 6.1 Calibration surface (exhausted)
Official gain curve fully mapped (v6/v9/v8 above); peak at gain ~2.0-2.2.
v10 probe (gain 2.1 x q 0.35) config written for the next slot. Expected
ceiling from calibration alone: ~+0.09-0.10.

### 6.2 Structural track: STATE foundation model integration (primary)

Discovery: Arc's entire State ecosystem is public and is the intended VCC
baseline -- almost certainly what top teams run:
  - `pip install arc-state` (PyPI v0.11.1) with CLI
    `state tx infer --model-dir <run> --adata controls.h5ad --pert-col gene`
  - `arcinstitute/ST-SE-Replogle` / `st-se-replogle-full`: ST checkpoints
    trained on Replogle-Nadig CRISPRi Perturb-seq (our source family)
  - `arcinstitute/SE-600M` (0.7B): cell encoder producing the HVG-space
    embeddings ST conditions on; `state emb transform` featurizes controls
  - Arc's own "Train STATE for the Virtual Cell Challenge" Colab

Integration plan:
  1. Install arc-state; download SE-600M + an ST checkpoint.
  2. Build per-context control AnnData in model gene space; embed.
  3. Infer responses for all 300 targets x A/B/C.
  4. Map to the 18,533 axis (non-model genes <- control levels), decode via
     our thinning pipeline with the mapped gain calibration, submit as probe.

Hardware: SE-600M fp16 ~1.4 GB VRAM; ST ckpt size TBD; CPU fallback exists.

### 6.3 Parked / rejected (with evidence)
  - Program-shape levers: denoising hurts pds (0.91->0.64); de-sharing flat.
  - Local pds optimization past ~0.82 proxy value: blind (competition
    crowding + real-context biology absent from thinned truth).
  - Tahoe-100M identification of A/B/C: dataset is DRUG perturbation
    (380 small molecules), not CRISPRi -- no same-line CRISPRi programs there.
  - Residual net: LOCO awarded it zero weight on real data.
