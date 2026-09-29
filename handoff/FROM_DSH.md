<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 6Q Report: Frozen Proposal Reference Resolver Audit

_This file holds the Task 6Q report. The Task 6P report is preserved in git history at commit
`b80f3cc`; Task 6O at `595e7bb`; Task 6N at `90f3735`; Task 6M.1 at `5e52d95`; Task 6M at `b9f49f8`._

Full design notes and the literature-position note: `docs/task6q_frozen_proposal_reference_resolver.md`.

## 1. Verdict

**`REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`** — section 14 priority order applied literally:

1. `INVALID_EXPERIMENT` — no.
2. `PROPOSAL_CHECKPOINT_UNAVAILABLE` — no: the frozen Task 6M.1 checkpoint is present and its SHA256
   matches `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` exactly.
3. **`REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`** — section 11 proposal-coverage gates fail (eligible
   coverage@0.50 overall 0.6895 < 0.70; smallest 0.5413 < 0.60). ← **verdict**
4. `DETERMINISTIC_REFERENCE_SELECTOR_INSUFFICIENT` — not reached.
5. `REFERENCE_ERROR_PROPAGATION_SEVERE` — not reached (three of its four sub-conditions actually pass).
6. `PROPOSAL_REFERENCE_CHAIN_FEASIBLE` — no.

No training, no threshold tuning, no test split. The resolver is treated as supporting infrastructure
and is **not** claimed as novelty.

## 2. Literature-position note (fixed)

RRSIS work already uses explicit object/relation/position decomposition with candidate/graph reasoning
(e.g. SRGFormer), and reasoning-segmentation work commonly decouples semantic reasoning from
grounding/segmentation through foundation-model proposals/prompts (e.g. Think2Seg-RS). Proposal-based
reference grounding is therefore treated here only as a reliable support module; the project's
candidate method contribution remains the reference-conditioned geometric relation field guiding dense
visual target segmentation, later combined with explicit relation-level supervision. No "first-ever"
claim.

## 3. Frozen proposal configuration (Part B)

Checkpoint `artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt`, SHA256
verified **exactly** before inference. YOLO26m-seg, imgsz **640**, conf **0.10**, max_det **100**,
default NMS, no TTA, no tiling, no threshold sweep. Masks are restored to source 512×512 and binarized
with the **canonical Task 6M** segmentation threshold (Ultralytics retina mask `> 0` through
`normalize_mask`); no new threshold was invented. Every tile was run through the model exactly once and
the selected references were cached in a gitignored resolver cache, so nothing was recomputed.

## 4. Deterministic resolver (Part C)

| Family | Eligibility | Selection |
|---|---|---|
| `largest` | NOT `touches_border` and `bbox_extent_ratio ≤ 0.20` (tiny **not** rejected) | maximum predicted area |
| `smallest` | the above and `area_px ≥ 150` | minimum predicted area |

`touches_border` = any positive pixel on row 0, row 511, col 0 or col 511. Tie-break on equal area:
higher confidence, then lower original proposal index. Zero proposals or zero eligible proposals →
explicit abstention. **No GT and no target identity ever entered eligibility, ranking or tie-breaks.**

## 5. Reference-level audit (Part D) — RefValUnique, 219 references over 212 tiles

`evaluation/task6q_reference_resolver_val.json`,
`evaluation/task6q_reference_failure_attribution.json`.

| Eligible coverage | @0.25 | @0.50 | @0.75 |
|---|---|---|---|
| overall | 0.7397 | **0.6895** | 0.4566 |
| largest | 0.9000 | **0.8364** | 0.6000 |
| smallest | 0.5780 | **0.5413** | 0.3119 |

(all-proposal coverage is 0.7489 / 0.6941 / 0.4612 overall)

| Selected-reference quality | overall | largest | smallest |
|---|---|---|---|
| mIoU | **0.413164** | 0.537910 | 0.287274 |
| Dice | 0.469992 | 0.608308 | 0.330408 |
| Pr@0.5 | 0.511415 | 0.645455 | 0.376147 |
| centroid error median | **0.015607** | 0.007302 | 0.065959 |
| centroid error p90 | **0.393517** | 0.317427 | 0.428458 |
| median area ratio | 1.081301 | 1.030109 | 1.162356 |
| abstentions (rate) | 6 (0.027397) | 1 (0.009091) | 5 (0.045872) |

Failure attribution (one category per record, fixed priority order): `REFERENCE_OK` **111** (largest 70 /
smallest 41), `REFERENCE_NOT_COVERED_IOU50` **62** (17 / 45), `EXTREME_SELECTION_WRONG` **39** (21 / 18),
`NO_ELIGIBLE_PROPOSALS` 3 (0 / 3), `NO_PROPOSALS` 3 (1 / 2), `SELECTED_MASK_GEOMETRY_POOR` 1 (1 / 0).
Resolver-cache statistics: 212 tiles, 0–51 proposals per tile (mean 8.72), 2 tiles with zero proposals,
10 abstained (tile, family) entries, 414 cached selections.

## 6. Downstream target propagation (Part E)

`evaluation/task6q_target_val.json` — resolved proposal reference → frozen GeometricRelationField
**v0.2** → frozen Task 6O **B3** → target mask (oracle reference mask, target GT, candidate target masks
and target instance id are never inputs; the GT target is scoring only):

| Metric | Value |
|---|---|
| target mIoU (234 answered of 240) | **0.304581** |
| Dice | 0.384945 |
| Precision@0.5 | 0.635327 |
| target abstentions | 6 (rate **0.025000**) |
| per relation (left / right / above / below) | 0.3353 / 0.2471 / 0.3291 / 0.3034 |
| per reference family (largest / smallest) | 0.3350 / 0.2731 |
| border target (n=110) | 0.304968 |
| tiny target (n=4) | ≈0 |

Comparison: Task 6O oracle B3 mIoU **0.4299680351479113** (delta **−0.125387**); Task 6P dense predicted
reference **0.24096754293919803** (delta **+0.063614**).

`evaluation/task6q_target_paired_val.json` — exact frozen PairedVal20, with exactly the same resolved
reference mask reused for pairs sharing image + reference source (only the relation changes):

| Metric | This task | Oracle B3 | Task 6P dense predicted |
|---|---|---|---|
| pass | **10/20** | 14/20 | 0/20 |
| mean own IoU | 0.277946 | 0.398969 | 0.064774 |
| mean cross IoU | 0.004245 | 0.001773 | 0.061155 |
| own − cross margin | **+0.273700** | +0.397196 | +0.003619 |
| pairs with reference abstention | 0 | — | — |

## 7. Gates (Part F)

* **Section 11 proposal coverage — FAIL**: eligible coverage@0.50 overall 0.6895 < 0.70; largest 0.8364
  ≥ 0.75 ✓; smallest 0.5413 < 0.60.
* **Section 12 reference-resolver adequacy — FAIL**: mIoU 0.413164 ≥ 0.35 ✓; median centroid error
  0.015607 ≤ 0.05 ✓; **p90 centroid error 0.393517 > 0.12**; abstention rate 0.027397 ≤ 0.10 ✓.
* **Section 13 downstream chain retention — FAIL** (only because section 12 fails): target mIoU
  0.304581 ≥ 0.3009776246035379 ✓; PairedVal 10/20 ≥ 10 ✓; margin +0.273700 ≥ 0.20 ✓.

No gate or threshold was altered.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **696 passed, 1 skipped** (Task 6P ended at 658 passed / 1 skipped; no
prior passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism
check that needs the proposal env). `tests/test_task6q_frozen_proposal_reference_resolver.py` covers the
38 section-15 checks: Task 6P artifacts unchanged, Task 6M.1 inference config unchanged, proposal
checkpoint SHA exact, conf exactly 0.10, imgsz exactly 640, max_det exactly 100, no threshold sweep,
masks restored to 512×512, exact border predicate, merge threshold exactly 0.20, tiny threshold exactly
150, exact largest/smallest eligibility, deterministic largest/smallest selection, exact tie-break, no
GT in reference selection, no target id in reference selection, RefValUnique reused, MiniVal240 reused,
PairedVal20 reused, no test split, v0.2 field unchanged, frozen B3 checkpoint unchanged, B3 not
retrained, no oracle reference in the downstream chain, GT reference only for evaluation, GT target only
for evaluation, same reference reused in the paired same-reference case, no YOLO training, no dense
reference-head retraining, no `[REF]`, no GRCL/SCL, no nearest/L3, no graph transformer, no 4B, no new
dataset/download/install/GUI, previous suite preserved.

Not committed: the YOLO checkpoint, the proposal/resolver cache, the SAM2 checkpoint, the feature cache,
source images/vectors, `.conda`, large caches. Committed: code, small JSON artifacts, docs, tests,
handoff.

Watt was **not needed** in Task 6Q: this task downloaded nothing (no new weights, packages or datasets)
and installed nothing. The pre-existing Watt instance is transport-only, is not owned by this project,
and was left running, per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 9. Interpretation boundary (section 21)

DSH reports measurements only: it does not claim the proposal reference resolver is novel, does not
decide to keep proposals permanently, does not decide a new learned reference architecture, and does not
start MLLM integration, GRCL or nearest/L3.

## 10. Recommended next step

等待 ChatGPT 根据 Task 6Q 的 frozen-proposal reference resolver 结果决定 Task 6R，不自行修改 reference 架构或开始 MLLM/GRCL/nearest/L3。

## 11. STOP

Task 6Q stops here: no other reference architecture, no MLLM hidden-state fusion, no joint training, no
GRCL, no nearest, no L3, no full-dataset training, no GUI. Waiting for the ChatGPT audit.
