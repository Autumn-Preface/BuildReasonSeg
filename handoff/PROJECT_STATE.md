# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 5C._

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

The block above is machine-checked against
`evaluation/build_spatial_reason_artifact_index.json` by
`scripts/check_artifact_consistency.py`. Do not hand-edit numbers anywhere else.

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset | **BuildSpatialReason** |
| Current dataset version | **v0.1.1** (validated, usable) |
| Legacy dataset version | **v0.1** (frozen, superseded, retained on disk) |
| Source dataset | WHU Building Dataset — `Satellite dataset II (East Asia)` |
| Semantic visibility policy | **1.0** (`spatial_reasoning/semantic_policy.py`) |
| Independent audit oracle | **1.0** (`spatial_reasoning/semantic_oracle.py`) |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components, 0 conflicts) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (32,284 records) |
| 5 | v0.1 validator + semantic quality audit | done → `FAIL_REQUIRES_REVISION` |
| 5B | v0.1.1 corrective regeneration + acceptance audit | done → `PASS` |
| 5C | **Acceptance hardening + artifact consistency** | **done → `PASS`** |

## Frozen relation rules (unchanged since Task 3B)

- Predicate convention: **`relation(subject, object)` = subject satisfies the
  relation w.r.t. the object** (ADR-006).
- Relation set: `left_of` `right_of` `above` `below` `leftmost` `rightmost`
  `topmost` `bottommost` `nearest` `largest` `smallest`.
- Thresholds (frozen in `configs/spatial_relations_v1.yaml`, never overridden):
  direction preset `medium` (`alpha = 1.2`, `tau = 0.04`), presets monotonic;
  `size_rank.ratio_margin = 1.10`; `nearest` anchor + target non-border;
  `extreme.margin_px = 4.0`; `tiny_component` < 150 px²;
  `suspected_large_merge` when bbox extent > 0.20.
- Ambiguity policy: **discard, never tie-break** (ADR-004).
- Scope: **tile-relative** spatial reasoning.

## Semantic visibility policy (ADR-010)

Natural language is defined over **all visible components**. Eligibility may
**reject** a sample; it may never silently **change** its answer, and no runner-up
is ever substituted. Implemented in `spatial_reasoning/semantic_policy.py` and
shared by the generator, the validator recomputation and the acceptance audit.

## Independent acceptance oracle (ADR-011, new in Task 5C)

`spatial_reasoning/semantic_oracle.py` is a second implementation of the same five
semantics that **never imports `semantic_policy`**. It resolves all 25,229
records independently and must agree on reference, direction candidate set,
nearest, final target and the ambiguity decision.

| Task 5C gate | Result |
|---|---|
| Independent-oracle target match | **25,229 / 25,229** |
| Independent semantic violations | **0** |
| Production hidden violations | **0** |
| Artifact consistency violations | **0** |
| Provenance digests vs disk | **8 / 8 match** |
| v0.1 / v0.1.1 JSONL hashes | **unchanged** |

## Task 5B count correction (Task 5C section 3.1)

The Task 5B report published Level-2 Type A/B as 2,272/2,764 and Level-3
trivial/nontrivial as 1,261/1,657. `statistics.json`, `manifest.json` and the
quality JSON all said **2,275/2,761** and **1,256/1,662**, and a direct recount
from the JSONL confirms them. The **dataset was always correct; the Task 5B
report text was wrong.** Corrected, and now gated.

## Canonical artifact paths (post-Task 5C)

```
datasets/build_spatial_reason/v0.1.1/manifest.json
datasets/build_spatial_reason/v0.1.1/statistics.json
evaluation/build_spatial_reason_artifact_index.json
evaluation/build_spatial_reason_v0.1.1_quality.json
evaluation/build_spatial_reason_v0.1.1_consistency.json
evaluation/build_spatial_reason_v0.1.1_samples/
docs/build_spatial_reason_v0.1.1_quality_audit.md
```

The historical `v011` evaluation artifacts no longer exist.

## Tests

**138/138 checks passed across 8 suites, 0 failed, 0 skipped, all exit 0.**

`test_component_conversion` 11 · `test_geometry` 17 · `test_relations` 35 ·
`test_annotator` 22 · `test_dataset_validator` 18 · `test_v011_acceptance` 19 ·
`test_semantic_oracle` 8 · `test_artifact_consistency` 8.

## Current blockers

**None.**

Known limitations (non-blocking): border truncation caps `largest`/`smallest`
coverage; `scene_level_split_leakage = unverified`; `peak_memory_mb` unconfirmed;
the oracle shares `geometry.component_box_distance` with production; oracle
independence is structural, not adversarial (see `FROM_DSH.md` §11).

## Recommended next task

**ChatGPT review of the Task 5C commit.** If approved, proceed to the next planned
stage (Task 5.5 / MLLM integration) as a **separate** task, consuming
`datasets/build_spatial_reason/v0.1.1/` and reading counts from
`evaluation/build_spatial_reason_artifact_index.json` rather than hand-copying
them.

Full detail: `handoff/FROM_DSH.md`,
`docs/build_spatial_reason_v0.1.1_quality_audit.md`,
`evaluation/build_spatial_reason_v0.1.1_quality.json`,
`evaluation/build_spatial_reason_v0.1.1_consistency.json`.
