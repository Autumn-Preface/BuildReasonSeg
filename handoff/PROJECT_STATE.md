# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 5B._

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset | **BuildSpatialReason** |
| Current dataset version | **v0.1.1** (usable) |
| Legacy dataset version | **v0.1** (frozen, superseded, retained on disk) |
| Source dataset | WHU Building Dataset — `Satellite dataset II (East Asia)` |
| Semantic visibility policy | **1.0** (`spatial_reasoning/semantic_policy.py`) |

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
| 5B | **v0.1.1 corrective regeneration + acceptance audit** | **done → `PASS`** |

## Frozen relation rules (unchanged by Task 5B)

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

## Semantic visibility policy (new, ADR-010)

Natural language is defined over **all visible components**. Eligibility may
**reject** a sample; it may never silently **change** its answer, and no runner-up
is ever substituted. Implemented once in `spatial_reasoning/semantic_policy.py`
and shared by the generator, the validator recomputation and the acceptance audit,
so generator and verifier cannot disagree about what a question means.

## Task 5B verdict

**`PASS`** — v0.1.1 passed the acceptance audit with **zero** blocking counts and
**zero** issues of any severity.

| Check | v0.1 | v0.1.1 |
|---|---|---|
| hidden semantic violations | 7,086 | **0** |
| reasoning ID leakage | 32,284 (159,014 mentions) | **0** |
| reference in distractors | 12,918 | **0** |
| target recomputation | 32,284/32,284 | **25,229/25,229 (100%)** |
| `template_id` verified | absent | **25,229/25,229** |
| back-reference in reasoning | n/a | **0** |

## Dataset v0.1.1 counts

```
total    : 25229
by_split : train 15592 | val 3884 | test 5753
by_level : L1 17275 | L2 5036 | L3 2918
level2   : reference_to_nearest 2272 + reference_to_direction 2764 = 5036 (invariant holds)
level3   : trivial 1261 | nontrivial 1657
per image: min 1 | median 6 | max 12
images with samples: 3920
```

Top discard reasons: `semantic_target_ineligible` 10,283 ·
`no_direction_candidate` 9,501 · `semantic_ambiguous` 5,362 ·
`multiple_direction_candidates` 4,265 · `ambiguous` 1,657 ·
`level3_nearest_semantic_ineligible` 1,217 · `nearest_semantic_ineligible` 692.

## Tests

**122/122 passed, 0 failed, 0 skipped, all exit 0.**

`test_component_conversion` 11 · `test_geometry` 17 · `test_relations` 35 ·
`test_annotator` 22 · `test_dataset_validator` 18 · `test_v011_acceptance` 19.

## Current blockers

**None.** v0.1.1 is validated and usable as MLLM supervision.

Known limitations (non-blocking): border truncation caps `largest`/`smallest`
coverage; `scene_level_split_leakage = unverified`; `peak_memory_mb` unconfirmed
(Windows ctypes fallback returns 0); v0.1 is no longer byte-regenerable by the
current code path (preserved on disk and verified unchanged).

## Recommended next task

**ChatGPT review of the v0.1.1 commit**, then — if approved — the next planned
stage (MLLM integration / reasoning segmentation MVP) as a **separate** task.

Non-blocking dataset follow-ups: quantify template diversity per query type using
the new `template_id` field; decide whether recovering `largest`/`smallest`
coverage needs richer (non-truncated) source annotations.

Full detail: `handoff/FROM_DSH.md`,
`docs/build_spatial_reason_v0.1.1_quality_audit.md`,
`evaluation/build_spatial_reason_v0.1.1_quality.json`.
