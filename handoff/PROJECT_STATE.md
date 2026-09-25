# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 5._

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset | **BuildSpatialReason** |
| Current dataset version | **v0.1** (frozen) |
| Next dataset version | v0.1.1 (recommended, not started) |
| Source dataset | WHU Building Dataset — `Satellite dataset II (East Asia)` |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components, 0 conflicts) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done (predicate convention fixed, presets monotonic, `ratio_margin` 1.10) |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (**32,284 records**) |
| 5 | v0.1 dataset validator + semantic quality audit | **done → verdict `FAIL_REQUIRES_REVISION`** |

## Frozen relation rules

- Predicate convention: **`relation(subject, object)` = subject satisfies the
  relation w.r.t. the object** (`left_of(A,B)` ⇔ "A is left of B"). ADR-006.
- Relation set: `left_of` `right_of` `above` `below` `leftmost` `rightmost`
  `topmost` `bottommost` `nearest` `largest` `smallest`.
  Not used: `overlap` `contain` `inside` `adjacent_to` `near` `far`.
- Thresholds (frozen in `configs/spatial_relations_v1.yaml`, never overridden by the generator):
  - direction preset `medium` — `alpha = 1.2`, `tau = 0.04`; presets are
    monotonic strict → medium → loose;
  - `size_rank.ratio_margin = 1.10`;
  - `nearest`: **anchor non-border AND target non-border**;
  - `extreme.margin_px = 4.0`; `tiny_component` < 150 px²;
    `suspected_large_merge` when bbox extent > 0.20.
- Ambiguity policy: **discard, never tie-break** (ADR-004).
- Scope: **tile-relative** spatial reasoning.

## Task 5 verdict

**`FAIL_REQUIRES_REVISION`** — v0.1 must not be used as MLLM supervision as-is.

What passed (all 32,284 records): target re-computation **100%**, mask selectors,
structured programs, candidate/distractor integrity, template reconstruction
(39/39 templates used), language parity, no duplicate ids, **0** exact cross-split
image duplicates, instructions **ID-free**.

Blocking defects:

1. **Hidden eligibility makes the natural-language answer false/misleading for
   7,086 records (21.9%)** — `largest` 2,180 (20.2%), `smallest` 4,246 (56.9%),
   `nearest` 1,058 (22.5%), Level-3 nearest 291 (7.3%). The engine answers over an
   eligibility-filtered subset while the instruction reads as being over all
   visible components.
2. **Internal component IDs leak into reasoning text in 100% of records**
   (159,014 mentions). Instructions are clean; only reasoning text is affected.

Also found: `statistics.json` / `manifest.json` report Level-2 Type A as 8,700,
but the true value is **4,707**; the generator's
`endswith("_to_nearest")` counter also swept in all 3,993 Level-3 records
(4,707 + 3,993 = 8,700). Level/​split totals are correct.

Full detail: `handoff/FROM_DSH.md`,
`docs/build_spatial_reason_v0.1_quality_audit.md`,
`evaluation/build_spatial_reason_v0.1_quality.json`.

## Current blockers

1. Semantic eligibility defect must be fixed before v0.1 can supervise a model
   (recommended fix: drop the 7,086 affected records — Option A).
2. Reasoning text must be regenerated ID-free.
3. Level-2 Type-A statistic must be corrected.

Not a blocker but unverified: **scene-level split leakage = `unverified`** — the
derived dataset carries no scene/geographic grouping metadata and val was a random
20% subset of the original train pool.

## Recommended next task

**`BuildSpatialReason-v0.1.1`** (not Task 5.5). The audit is complete and
decisive, the root causes are identified, and the corrections are mechanical:

1. drop the 7,086 semantically untruthful records (retain 25,198 = 78.1%);
2. regenerate reasoning text with no `component N` identifiers;
3. fix the Type-A counter to key on `level == 2`;
4. decide and document the reference/distractor policy;
5. add a per-sample `template_id` for language-diversity auditing.

Acceptance gate for v0.1.1 (all validator-checkable):
`hidden_eligibility_* = 0`, `component_id_leak_reasoning = 0`,
Type A + Type B = Level 2 total, target recomputation stays 100%.
