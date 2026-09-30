# Task 6X — Frozen SAM2.1 Proposal Refinement Audit

> Task: `handoff/TO_DSH.md` (Task 6X) · Base commit: `3de2142` · Predecessor: Task 6W →
> `QUALITY_FILTER_NOT_HELPFUL`
> **Verdict: `SAM2_REFINEMENT_NOT_HELPFUL`** · Frozen option: **X-C0 (no refinement)** · **STOP rule triggered**
> Tests: `tests/test_task6x_sam2_refinement.py` · Evidence: `evaluation/task6x_*.json`

Task 6X tests one **training-free**, foundation-model-based alternative to Task 6W's learned quality
classifier: refine each eligible U-C1 YOLO proposal mask from its geometric box prompt with the frozen
official SAM2.1 image predictor and then run the literal largest/smallest rule on the refined instance
masks. Support infrastructure only — not a claimed research novelty. **No model was trained**, and because
the baseline won the train-only calibration the task **stopped immediately** without touching
RefValUnique, MiniVal240 or PairedVal20.

## 1. Recorded Task 6W findings (Task 6W artifacts not mutated)

Verdict `QUALITY_FILTER_NOT_HELPFUL`.

**W0 oracle-quality mechanism** (train-only U-Calib200): deterministic U-C1 reference mIoU `0.4789276` →
oracle `q ≥ 0.50` filter mIoU `0.6575339`; smallest `0.3544718 → 0.6707174`; selection wrong `76 → 23`;
abstention `0`. The abstract mechanism *"remove invalid proposals before extreme-area selection"* is valid.

**Learned quality estimator**: holdout AUROC `0.8438013`, F1@0.50 `0.8251182`, but RefVal reference mIoU
`0.3793127` (below W-S0 `0.4289355`), selection wrong `65` (worse than W-S0 `53`), downstream answered mIoU
`0.2745576` (below W-S0 `0.3141364`). ProposalQualityEstimator v0.1 is therefore not part of the primary
resolver.

## 2. Frozen assets (Part B)

`evaluation/task6x_frozen_asset_audit.json` = **`FROZEN_ASSETS_VERIFIED`**.

| Asset | Verification |
|---|---|
| SAM2.1 Hiera Base+ | `local_cache/models/sam2.1_hiera_base_plus.pt`, SHA256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` **exact**; official `sam2.sam2_image_predictor.SAM2ImagePredictor`; config `configs/sam2.1/sam2.1_hiera_b+.yaml`; `trainable_parameters = 0`; not retrained |
| box-prompt probe | box `[10,10,200,200]` → masks `(3, 512, 512)` with scores `[0.8870, 0.4559, 0.8629]`; `point_coords = point_labels = mask_input = None`, `return_logits = False` |
| U-C1 proposals | `imgsz 640 / conf 0.05 / max_det 300 / default NMS / no TTA / no tiling`; YOLO SHA256 `ef852b58…61f474` exact; not retrained |
| ProgramHead / B3 / field v0.2 | hashes recorded, unchanged; the ranker and quality estimator are explicitly **excluded** from the Task 6X resolver |

## 3. Four predeclared options (Parts C-D)

`evaluation/task6x_calibration_refinement.json` — U-Calib200 (200 references over 195 tiles):

| Option | Description | overall mIoU | smallest mIoU | largest mIoU | Pr@0.5 | centroid median | abstain | SAM2 calls/tile | wall time/tile |
|---|---|---|---|---|---|---|---|---|---|
| **X-C0** | no refinement (U-C1 baseline) | **0.4789** | **0.3545** | **0.6034** | **0.5800** | 0.0078 | 0.0000 | 0.00 | 0.000 s |
| X-C1 | exact box, single-mask SAM2 | 0.4744 | 0.3502 | 0.5986 | 0.5700 | 0.0061 | 0.0000 | 9.67 | 0.172 s |
| X-C2 | exact box, multimask (max predicted quality) | 0.4284 | 0.3277 | 0.5291 | 0.5250 | 0.0075 | 0.0000 | 9.67 | 0.126 s |
| X-C3 | 10 %-expanded box, multimask | 0.3741 | 0.3008 | 0.4481 | 0.4523 | 0.0098 | 0.0050 | 9.67 | 0.134 s |

No empty refined masks and no abstentions for X-C1/X-C2 (X-C3 abstains once). Baseline buckets:
`REFERENCE_SELECTION_WRONG` 84, `REFERENCE_OK` 116, no coverage/eligibility failures (U-Calib200 is the
easy train-side split).

Section 12 priority → ranking **`['X-C0', 'X-C1', 'X-C2', 'X-C3']`** →
`evaluation/task6x_frozen_refinement_option.json` freezes **X-C0**, i.e.
`baseline_is_selected = true`. Selection inputs were U-Calib200 only (RefValUnique / MiniVal240 /
PairedVal20 / test all `false`), and the option is immutable afterwards.

## 4. STOP rule (section 13.1) — no downstream evaluation

Because X-C0 won, Task 6X **stopped**: `evaluation/task6x_refval_refinement.json`,
`task6x_downstream_minival240.json`, `task6x_downstream_pairedval20.json` and
`task6x_hardened_parser_integration.json` were intentionally **not produced**, and the verdict artifact
records their absence as a required property (`stop_rule.forbidden_files_absent = true`).

`evaluation/task6x_verdict.json`:

* `INVALID_EXPERIMENT` — no: frozen modules unchanged, exactly four options, no test use, no banned model.
* `FROZEN_ASSET_UNAVAILABLE` — no: SAM2.1 and U-C1 hashes exact.
* **`SAM2_REFINEMENT_NOT_HELPFUL`** — X-C0 wins the train-only ranking; the best refinement option (X-C1)
  reaches overall mIoU 0.4744 vs baseline 0.4789 (−0.0045), and X-C2/X-C3 lose far more
  (−0.0505 / −0.1048). ← **verdict**
* `SAM2_REFINEMENT_PARTIAL` / `SAM2_REFINEMENT_USEFUL` — not applicable.

## 5. Measured interpretation boundary

The comparison is a **strictly controlled, train-only** measurement: same frozen U-C1 proposals, same
Task 6Q eligibility re-applied to the refined masks, same literal largest/smallest semantics with the
predeclared tie-break (SAM2 predicted quality → YOLO confidence → original index), and no SAM-score
threshold anywhere. Refinement measurably improves the *median centroid error* for X-C1 (0.0078 → 0.0061),
i.e. the refined masks are marginally better centred, but the refined **areas** shift the extreme
selection enough to lose overall IoU; multi-mask and expanded-box variants lose substantially more,
consistent with SAM2 preferring "object-like" regions that are not the literal area extremes. DSH reports
this as measured and does not prescribe a remedy: no point/mask prompt, no box-expansion change, no quality
threshold, no TTA/tiling and no further reference module was added.

## 6. Interpretation boundary

The refiner is support infrastructure, **not** a claimed novelty; no SAM quality threshold was used; the
SAM2 score appears only as a recorded selection tie-break; U-C1 was not changed; YOLO/SAM2/ProgramHead/B3
were not retrained; the ranker and quality estimator were not used; no nearest/L3 execution was started and
no Task 6Y was chosen.

## 7. Reproduce

```text
python scripts/task6x_calibrate_refinement.py --device 0   # Parts B + D-E (asset audit, four options, freeze)
python scripts/task6x_report.py                            # Parts J-K (verdict + STOP rule)
```

On the STOP path no other script is run. SAM2 image predictor refinements are recomputed per tile (no cache
file is required); run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`.
