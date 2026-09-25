# FROM_DSH — Task 5 Report: BuildSpatialReason-v0.1 Dataset Validator & Semantic Quality Audit

**Date:** 2026-08-04
**Auditor:** DSH
**Audit target:** `datasets/build_spatial_reason/v0.1/` (32,284 records), treated as **frozen**
**Verdict: `FAIL_REQUIRES_REVISION`**

---

## 1. Audit Verdict

**`FAIL_REQUIRES_REVISION`**

The dataset is **internally consistent and fully machine-verifiable**, but it is
**not semantically sound** for a substantial minority of records, and its
reasoning text is **not directly usable as MLLM supervision**.

Two independent defect classes were established:

| Class | Severity | Scale |
|---|---|---|
| Hidden eligibility makes the natural-language answer false/misleading | **BLOCKING** | **7,086 records (21.9%)** |
| Internal component IDs leak into reasoning text | WARNING (universal) | **32,284 records (100%)** |

Verdict logic (`dataset_validator.decide_verdict`) returns
`FAIL_REQUIRES_REVISION` because blocking codes fired, specifically
`hidden_eligibility_largest` / `_smallest` / `_nearest` /
`_level3_nearest` — all of which are in `BLOCKING_CODES`.

**v0.1 was NOT modified.** No file under `datasets/build_spatial_reason/v0.1/`
was written, regenerated or repaired. Recommended revision version:
**`BuildSpatialReason-v0.1.1`**.

---

## 2. Files Created / Modified

**Created (5 new files, all committable — verified not ignored):**

| File | Purpose |
|---|---|
| `spatial_reasoning/dataset_validator.py` | Validator core: per-image cached context, independent recomputation, semantic audit, template reconstruction, image-hash leakage |
| `scripts/validate_build_spatial_reason.py` | Audit CLI |
| `scripts/build_spatial_reason_samples.py` | Audit visualization pack + contact sheet |
| `tests/test_dataset_validator.py` | 18 validator tests |
| `evaluation/build_spatial_reason_v0.1_quality.json` | Machine-readable audit results (full run) |
| `docs/build_spatial_reason_v0.1_quality_audit.md` | Readable audit report |
| `evaluation/build_spatial_reason_v0.1_samples/` | 37 PNGs + `contact_sheet.png` (8.98 MB) |
| `handoff/FROM_DSH.md` (this file), `handoff/PROJECT_STATE.md` | Handoff outputs |

**Modified:** none. **No existing file was edited**, including
`handoff/TO_DSH.md` (left intact for audit history).

**Not modified:** `../WHU_Building_Segment/` (legacy, read-only) and every file
under `datasets/build_spatial_reason/v0.1/`.

---

## 3. Independent Dataset Counts

Recounted directly from the three JSONL files, independent of all generator
bookkeeping:

| Quantity | Value |
|---|---|
| **Total records** | **32,284** |
| `train` | 19,977 |
| `val` | 4,981 |
| `test` | 7,326 |
| **Level 1** | **19,366** |
| **Level 2** | **8,925** |
| **Level 3** | **3,993** |
| Level 2 Type A (`reference → nearest`) | **4,707** |
| Level 2 Type B (`reference → direction`) | **4,218** |
| Level 3 trivial | **1,804** |
| Level 3 nontrivial | **2,189** |
| Images with samples | 3,921 |
| Samples/image | min 2, p5 3, median 9, p95 12, max 12 |

By split × level:

| split | L1 | L2 | L3 |
|---|---|---|---|
| train | 11,994 | 5,541 | 2,442 |
| val | 3,018 | 1,345 | 618 |
| test | 4,354 | 2,039 | 933 |

`total_records`, `by_split`, `by_level`, `trivial` and `nontrivial` all **agree**
with `statistics.json` and `manifest.json`.

---

## 4. Existing Statistics Consistency

**One inconsistency found, and it is a genuine generator bug — the known
discrepancy in §2 of the task file is confirmed.**

Task 4 reported Level 2 = 8,925, Type A = 8,700, Type B = 4,218. Since
8,700 + 4,218 = 12,918 ≠ 8,925, these could not all be true.

**Authoritative recount:**

```
Level 2 total (level == 2)        = 8925
Level 2 Type A (largest_to_nearest + smallest_to_nearest) = 2235 + 2472 = 4707
Level 2 Type B (count("_to_") == 1 and not endswith("_to_nearest")) = 4218
4707 + 4218 = 8925  ==  Level 2 total   ✓ internally consistent
```

**Root cause (identified exactly):** the generator computed Type A with
`query_type.endswith("_to_nearest")`. **Every Level-3 query type also ends in
`_to_nearest`** (`largest_to_right_of_to_nearest`), so the filter swept all of
Level 3 into the Type-A count:

```
8700 = 4707  (true Level 2 Type A)
     + 3993  (ALL of Level 3)
```

| Statistic | Claimed | True | Status |
|---|---|---|---|
| `statistics.json` → `level2.reference_to_nearest` | 8,700 | **4,707** | **WRONG** |
| `manifest.json` → `sample_counts.level2.reference_to_nearest` | 8,700 | **4,707** | **WRONG** |
| `statistics.json` → `level2.reference_to_direction` | 4,218 | 4,218 | correct (by luck of the filter) |
| `by_level`, `by_split`, `total_samples` | — | — | correct |

The defect is confined to the reported **decomposition**; the dataset records
themselves are unaffected. Pinned by
`test_independent_counts_match_generator_bookkeeping`.

---

## 5. Full Target Re-computation

Every one of the 32,284 records was independently recomputed from its intended
structured program, using source component metadata, the frozen relation
definitions and the frozen thresholds. The stored `target_component_id` was
**never used as an input**.

| Metric | Result |
|---|---|
| Checked | **32,284** |
| Pass | **32,284** |
| Fail | **0** |
| Match rate | **100.000%** |

Recomputation was independent in two respects:

1. The Level-2 Type-B target was re-derived from the frozen direction predicate,
   not read from the stored `candidate_component_ids`.
2. Filter sets were re-derived via `filter_relation_candidates` and compared
   against the stored lists.

Additionally, `reasoning_steps` were re-executed end-to-end by the validator's
own operation interpreter (`recompute_target_independent`), including
re-applying the nearest margin policy, with **zero** mismatches.

**Conclusion: the structured-program → target chain is sound in v0.1.**

---

## 6. Largest Semantic Audit

`hidden_eligibility_largest`: **2,180 violations** of 10,785 applicable
(**20.2%**).

Method: for every record involving `largest`, compute the global argmax of
`area_px` over **all visible components**, then compare with the component the
dataset uses as "largest".

Example `buildsr_train_1_10028_1_largest_6ec47d449c40`:

```
instruction_en       : Locate and segment the building region with the largest area.
global_largest_id    : 10
global_largest_flags : ["touches_image_border"]
engine_reference_id  : 5      <- actual dataset target
```

Component 10 is genuinely the largest visible building region, but it is
border-truncated, so the engine silently answered component 5. A reader cannot
see the hidden filter and would reasonably choose 10.

Visualized at
`evaluation/build_spatial_reason_v0.1_samples/031_semantic_failure_buildsr_train_1_10028_1_largest_6ec47d449c40.png`.

By split: train 1,390 / val 366 / test 424.

---

## 7. Smallest Semantic Audit

`hidden_eligibility_smallest`: **4,246 violations** of 7,468 applicable
(**56.9%**) — the worst semantic defect in the dataset.

Method: global argmin of `area_px` over all visible components vs the engine's
eligible smallest.

The rate is far higher than `largest` because `smallest` has **two** filters
stacked: border-truncation **and** `tiny_component` exclusion. Small components
are frequently both border-touching and tiny, so the visible global smallest is
very often not the engine's answer.

Example `buildsr_train_1_10028_1_smallest_1a8466520a90`:

```
global_smallest_id    : 11
global_smallest_flags : ["touches_image_border"]
engine_reference_id   : 4
```

By split: train 2,607 / val 640 / test 999.

---

## 8. Nearest Semantic Audit

`hidden_eligibility_nearest`: **1,058 violations** of 4,707 applicable
(**22.5%**).

Method: compute boundary distance from the reference to **all other visible
components**, take the semantic nearest before target-eligibility filtering, and
compare with the dataset target.

Example `buildsr_train_1_0_2_largest_to_nearest_062c35a80063`:

```
semantic_nearest_id    : 3   gap = 48.65 px
engine_target_id       : 2   gap = 229.46 px
semantic_nearest_flags : ["touches_image_border"]
```

The dataset's answer (component 2) is **4.7× farther** from the reference than the
component a reader would pick (component 3). This is the most visibly misleading
violation class: the instruction says "nearest", and the selected component is
demonstrably not the nearest visible one.

By split: train 686 / val 141 / test 231.

---

## 9. Level-3 Hidden Eligibility Audit

`hidden_eligibility_level3_nearest`: **291 violations** of 3,993 applicable
(**7.3%**).

Method: obtain all direction candidates via the frozen direction predicate, then
find the true nearest within that direction set **before** nearest-eligibility
filtering, and compare with the dataset target.

Example `buildsr_train_1_10028_3_largest_to_above_to_nearest_a988dcbdac2a`:

```
relation               : above
direction_candidates   : [7, 8, 9, 10, 11]
semantic_nearest_id    : 10   gap = 13.04 px
engine_target_id       : 7    gap = 78.61 px
semantic_nearest_flags : ["touches_image_border"]
```

Note this is a **nontrivial** Level-3 sample, so the violation survives even in
the subset that a future reasoning metric would prefer. By split: train 164 /
val 47 / test 80.

---

## 10. Component-ID Leakage

| Field group | Samples affected | Total mentions |
|---|---|---|
| **`reasoning_zh` / `reasoning_en`** | **32,284 / 32,284 = 100%** | **159,014** |
| `instruction_zh` / `instruction_en` | **0** | 0 |

Uniform across levels: L1 19,366 / L2 8,925 / L3 3,993.

Example (`buildsr_train_1_0_1_leftmost_a9c23eba2157`):

```
instruction_zh : 分割图像中最左侧的建筑区域。              <- clean
instruction_en : Segment the leftmost building region...    <- clean
reasoning_zh   : 最左侧的建筑区域是 component 2。            <- LEAK
reasoning_en   : The leftmost building region is component 2. <- LEAK
```

**Classification:** this is **reasoning-text leakage only**. IDs inside
`reasoning_steps` are structured machine fields and are **not** leakage.

**Suitability assessment — NOT suitable as-is for MLLM supervision.** The input
image does not display any `component N` label. Training a model to emit such
identifiers teaches it to produce annotation-internal artifacts it cannot
ground, rather than spatial reasoning. Any model trained on this reasoning text
would be pushed toward hallucinating identifiers.

**Proposed v0.1.1 ID-free reasoning style** (this is a proposal, **not applied**
to v0.1):

- 中文：首先找到图像中面积最大的建筑区域，将其作为参考区域。随后比较其右侧候选建筑与参考区域的距离，选择距离最近的区域作为目标。
- English: First identify the largest building region as the reference. Then
  compare the distances of the candidate regions to its right and select the
  closest one as the target.

---

## 11. Structured Program Integrity

Validated on all 32,284 records. **Zero failures** in every category:

| Check | Result |
|---|---|
| Continuous step numbering (`step == index`) | 0 failures |
| Program length matches level (L1=1, L2=2, L3=3 operations) | 0 failures |
| Operation sequence matches level/query_type | 0 failures |
| Every referenced component exists in the source image | 0 failures |
| No duplicate IDs inside any step candidate list | 0 failures |
| Output target appears only at the appropriate step | 0 failures |
| Level-3 `filter_relation` present with `argmin_boundary_distance` | 0 failures |

---

## 12. Candidate / Distractor Integrity

| Check | Result |
|---|---|
| Target present in `distractor_component_ids` | **0** |
| Duplicate IDs in `distractor_component_ids` | **0** |
| Distractor IDs that do not exist | **0** |
| Level-2 Type-B with exactly one direction candidate | **4,218 / 4,218** ✓ |
| Level-3 `nearest_eligible_component_ids ⊆ candidate_component_ids` | **0 violations** |
| Level-3 stored direction set == independently derived set | **0 mismatches** |
| Level-3 stored eligible set == independently derived set | **0 mismatches** |
| Reference present in `distractor_component_ids` | **12,918 (40.0%)** — WARNING |

The distractor finding is a **definitional ambiguity, not a data error**: the
reference is legitimately not the target, so listing it is not incorrect. But a
distractor set meant to represent "plausible wrong answers" probably should
exclude the named reference. Flagged for v0.1.1 to decide and document.

---

## 13. Mask Selector Integrity

Full-dataset validation (all 32,284 records):

| Check | Result |
|---|---|
| `target_mask.component_id == target_component_id` | **32,284 / 32,284** |
| `target_mask.representation == "component_map_selector"` | **32,284 / 32,284** |
| Target component exists in the referenced component map | **32,284 / 32,284** |
| Target mask yields **non-empty** binary mask | **32,284 / 32,284** |

---

## 14. Template Distribution

All **39 declared templates were used; zero unused**. Reconstruction succeeded
for **32,284 / 32,284** records.

| Family | Templates | Uses | Balance |
|---|---|---|---|
| `l1.leftmost` | 3 | 3,519 | 1,154 / 1,140 / 1,225 |
| `l1.rightmost` | 3 | 3,527 | 1,183 / 1,226 / 1,118 |
| `l1.topmost` | 3 | 3,491 | 1,152 / 1,130 / 1,209 |
| `l1.bottommost` | 3 | 3,494 | 1,156 / 1,128 / 1,210 |
| `l1.largest` | 3 | 2,566 | 861 / 878 / 827 |
| `l1.smallest` | 3 | 2,769 | 928 / 927 / 914 |
| `l2.nearest` | 6 | 4,707 | 716 – 852 |
| `l2.dir` | 12 | 4,218 | 327 – 375 |
| `l3.chain` | 3 | 3,993 | 1,315 / 1,299 / 1,379 |

**No severe imbalance.** Every family's per-template spread is within roughly
±10% of its mean, which is consistent with the SHA256-bucketed selector being
unbiased.

---

## 15. Language Parity

| Check | Result |
|---|---|
| zh and en reconstruct to the **same** template id | **32,284 / 32,284** |
| zh/en from *different* templates (mismatch) | **0** |
| Raw `{placeholder}` left in text | **0** |
| Literal `None` / `null` in text | **0** |
| Empty language field | **0** |
| Doubled punctuation (`。。`, `，，`, `..`) | **0** |

Language parity holds operationally: a zh/en pair always resolves to the same
`template_id`, and a mismatched cross-family pair is correctly rejected
(`test_template_reconstruction_rejects_mismatched_pair`).

---

## 16. Split / Exact Duplicate Audit

| Check | Result |
|---|---|
| `train ∩ val` image ids | **∅** |
| `train ∩ test` image ids | **∅** |
| `val ∩ test` image ids | **∅** |
| Duplicate `sample_id` | **0** |
| Duplicate canonical semantic key (image + program + target) | **0** |
| **`exact_image_duplicate_cross_split`** | **0** |

Source images hashed: train 2,424 / val 614 / test 883 (3,921 distinct images
referenced by the dataset). **No byte-identical image appears in two splits.**

### Scene-level leakage

**`scene_level_split_leakage = unverified`**

Reason: the derived YOLO dataset carries **no scene / geographic grouping
metadata**, and val was a **random 20% subset of the original WHU train pool**.
Scene-level disjointness therefore cannot be established from available
metadata. Exact image duplication is ruled out (0); geographic adjacency between
train and val tiles is **not** ruled out and must not be claimed absent.

---

## 17. Distribution Audit

Level proportions are tightly matched across splits:

| split | n | L1 | L2 | L3 |
|---|---|---|---|---|
| train | 19,977 | 60.04% | 27.74% | 12.22% |
| val | 4,981 | 60.59% | 27.00% | 12.41% |
| test | 7,326 | 59.43% | 27.83% | 12.74% |

Max deviation between splits is ~0.6 percentage points — no level skew.

Target component distributions (descriptive):

| metric | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| `area_px` | 66 | 207 | 1,198.5 | 4,940.1 | 131,787 |
| `centroid_x_norm` | 0.001 | 0.028 | 0.488 | 0.969 | 0.997 |
| `centroid_y_norm` | 0.001 | 0.029 | 0.496 | 0.971 | 0.997 |

Targets are centred on the image and cover the full area range; no positional or
scale collapse. The dataset was **not** resplit in this task.

---

## 18. Trivial / Nontrivial Level-3 Audit

| Metric | Reported | Independently verified |
|---|---|---|
| trivial | 1,804 | **1,804** ✓ |
| nontrivial | 2,189 | **2,189** ✓ |
| total | 3,993 | **3,993** ✓ |

Verification: `trivial_selection == true` was checked **iff** the
direction-filtered **nearest-eligible** candidate set contains exactly one
component, recomputed independently from the frozen predicate with **0**
mismatches.

**Recommendation (as requested):** trivial Level-3 samples **should NOT** be
included in the primary multi-hop reasoning metric. With a single eligible
candidate after filtering, the direction filter alone determines the answer and
no distance comparison is required, so such samples do not test multi-hop
reasoning. Report them separately.

Caveat worth noting for v0.1.1: because the semantic audit found 291 Level-3
violations, the nontrivial subset is the one that must be cleaned first — it is
exactly where a "nearest" claim can be visibly wrong.

---

## 19. Visualization Pack

| Item | Count |
|---|---|
| **visualizations_generated** | **37** (36 per spec + contact sheet) |
| Composition | 6 Level-1, 8 Level-2 (4 nearest + 4 direction), 10 nontrivial Level-3, 4 trivial Level-3, 8 semantic-failure cases |
| `contact_sheet.png` | 1 |
| Total pack size | **8.98 MB** (<10 MB target) |
| Location | `evaluation/build_spatial_reason_v0.1_samples/` |

Each panel shows: source image (grayscale), reference outline (amber), target
outline + fill (green), candidate IDs as an **audit overlay only**, plus
query_type, level, reference, target, candidate list and the English instruction.

**`manual_visual_inspection`: PARTIAL.** I opened and inspected **2** panels
directly (one semantic-failure case and the pack listing); I did **not** visually
inspect all 37. Per the task's instruction, generated count is **not** claimed as
inspected count. The pack exists specifically so a human can complete the review
quickly.

---

## 20. Issues Found

| # | Issue | Severity | Scale | Evidence |
|---|---|---|---|---|
| 1 | **Hidden eligibility changes the natural-language answer** | **BLOCKING** | 7,086 records / 21.9% | §6–9; 4 flags |
| 2 | **Internal component IDs in reasoning text** | WARNING (universal) | 32,284 records / 100% | §10 |
| 3 | **`statistics.json` / `manifest.json` Level-2 Type-A count wrong** | WARNING | 2 files | §4: 8,700 vs true 4,707 = 4,707 + 3,993 (all L3) |
| 4 | Reference listed in `distractor_component_ids` | WARNING (definitional) | 12,918 records / 40.0% | §12 |
| 5 | Scene-level split leakage unverifiable | INFO | — | §16 |
| 6 | Trivial Level-3 pollutes multi-hop metric if pooled | INFO | 1,804 records | §18 |

**No other defects found.** In particular: target recomputation, mask selectors,
structured programs, candidate/distractor integrity, template reconstruction,
language parity and exact-duplicate detection are all **clean**, and no
instruction-level ID leakage, placeholder residue or empty fields exist.

---

## 21. Recommended Fixes

For **`BuildSpatialReason-v0.1.1`** (do not patch v0.1 in place):

1. **Apply Option A — drop semantically untruthful samples.** Keep only records
   where hidden eligibility does not change the expressed answer. Measured
   effect: discard **7,086 (21.9%)**, retain **25,198 (78.1%)**. No language
   change needed. `leftmost` / `rightmost` / `topmost` / `bottommost` are
   unaffected and fully retained.
   - Rejected alternative (Option B) — making eligibility explicit in the
     instruction ("among building regions that do not touch the image boundary
     ...") — is **not recommended**: tile-edge truncation is an artifact of how
     the WHU tiles were cropped, not a property of buildings or of spatial
     reasoning, so it would teach a non-transferable dataset quirk and make the
     instructions stylistically inconsistent with the tile-relative extremes.
2. **Regenerate reasoning text ID-free.** Replace `component N` with referring
   descriptions (see §10 for the proposed style). This is a prerequisite for
   using the reasoning text as MLLM supervision at all.
3. **Fix the Type-A counter** to key on `level == 2` rather than
   `query_type.endswith("_to_nearest")`, then regenerate `statistics.json` and
   `manifest.json`.
4. **Decide and document the distractor policy** for the reference component.
5. **Add a `template_id` field per sample** so language-diversity auditing does
   not require text reverse-engineering (currently only `template_version` is
   recorded).
6. **Consider one uniform eligibility policy for `smallest`.** Its 56.9%
   violation rate suggests the two stacked filters (border + tiny) are too
   aggressive for a natural-language "smallest".

---

## 22. Ready for v0.1.1 or Task 5.5?

**Recommend `BuildSpatialReason-v0.1.1` directly — Task 5.5 is not required.**

Rationale: the audit is complete and decisive. Both defect classes are
**fully characterised with exact counts, root causes and representative
evidence**, and the corrective actions are mechanical rather than investigative:

- semantic filtering needs no new analysis, only the measured filter (§21.1);
- the ID-free reasoning rewrite is a templating change in
  `spatial_reasoning/templates.py` / `annotator.py`;
- the counter bug has an identified one-line root cause.

There is **no open question that further auditing would resolve**, so an
intermediate Task 5.5 would add cost without information. The one item that
genuinely needs a human decision — the distractor/reference policy (§21.4) — is
small enough to settle inside v0.1.1.

Before v0.1.1 is accepted, the following must be true (all are validator-checkable):

- `hidden_eligibility_*` counts = **0**;
- `component_id_leak_reasoning` = **0**;
- `statistics.json` Type A = true Type A, and Type A + Type B = Level 2 total;
- target recomputation remains **100%**.

**Do not begin Task 5.5 or v0.1.1 in this session** — as instructed, this task
stops here.
