# BuildSpatialReason v0.1 — Independent Quality Audit

**Verdict: `FAIL_REQUIRES_REVISION`**

Audited by `spatial_reasoning/dataset_validator.py` via
`scripts/validate_build_spatial_reason.py`, over all **32,284** records of
`datasets/build_spatial_reason/v0.1/`. Runtime 261 s.

Full machine-readable results: `evaluation/build_spatial_reason_v0.1_quality.json`
Visualization pack: `evaluation/build_spatial_reason_v0.1_samples/`

> **v0.1 was NOT modified.** It is a frozen audit target. Corrections belong to a
> future `BuildSpatialReason-v0.1.1`.

---

## 1. What passed

| Check | Result |
|---|---|
| Full target re-computation | **32,284 / 32,284 = 100%** |
| Mask selector `target_mask.component_id == target_component_id` | **32,284 / 32,284** |
| Target mask non-empty | **32,284 / 32,284** |
| Template reconstruction (39/39 templates) | **32,284 / 32,284** |
| Duplicate sample ids | **0** |
| Duplicate canonical semantic keys | **0** |
| Exact cross-split image duplicates | **0** |
| Structured-program integrity (numbering, ops, references, dup ids) | **0 failures** |
| Level-2 Type-B uniqueness | **0 failures** |
| Level-3 `nearest_eligible ⊆ direction candidates` | **0 failures** |
| Instructions free of internal component ids | **0 leaks** |
| Empty language fields / raw `{placeholder}` / literal `None` | **0** |
| Cross-split `image_id` leakage | **0** |

These establish that the dataset is **internally consistent and machine-verifiable**.
Every target really is derivable from its own structured program, and every mask
selector resolves.

---

## 2. What failed

Two independent defect classes, both systematic rather than sporadic.

### 2.1 Hidden eligibility changes the natural-language answer (BLOCKING)

Natural language such as "the largest building" / "the nearest building" is
normally read over the **visible** components in the image. The engine answers
over an **eligibility-filtered** subset (border-truncated, suspected-merge and
tiny components are excluded from `largest` / `smallest` / `nearest`). Where the
two disagree, the instruction is **false or misleading** for a reader who cannot
see the hidden filter.

| Flag | Count | Share of applicable |
|---|---|---|
| `hidden_eligibility_smallest` | **4,246** | 4,246 / 7,468 = **56.9%** |
| `hidden_eligibility_largest` | **2,180** | 2,180 / 10,785 = **20.2%** |
| `hidden_eligibility_nearest` | **1,058** | 1,058 / 4,707 = **22.5%** |
| `hidden_eligibility_level3_nearest` | **291** | 291 / 3,993 = **7.3%** |

Representative case (`buildsr_train_1_10028_1_largest_6ec47d449c40`):

```
instruction_en : Locate and segment the building region with the largest area.
global_largest_id      : 10
global_largest_flags   : ["touches_image_border"]
engine_reference_id    : 5      <- dataset target
```

Component 10 is the largest by area but touches the tile edge, so the engine
silently answered with component 5. A human or an MLLM looking at the image would
reasonably select component 10. See
`evaluation/build_spatial_reason_v0.1_samples/031_semantic_failure_*.png`.

The asymmetry between `largest` (20.2%) and `smallest` (56.9%) has a clear cause:
border truncation affects large components disproportionately, and `smallest`
*additionally* excludes tiny components, so two filters bite instead of one.

**This alone forces `FAIL_REQUIRES_REVISION`** per the audit contract: such
samples make the natural-language answer false or misleading.

### 2.2 Internal component IDs leak into reasoning text (WARNING, but universal)

**100% of records (32,284 / 32,284)** contain internal annotation IDs in
`reasoning_zh` / `reasoning_en`, e.g.:

```
reasoning_zh : 最左侧的建筑区域是 component 2。
reasoning_en : The leftmost building region is component 2.
```

1,59,014 mentions in total. Breakdown by level: L1 19,366 / L2 8,925 / L3 3,993 —
i.e. every level, uniformly.

Instructions are **clean** (0 leaks); the leak is confined to reasoning text.
IDs inside `reasoning_steps` are **not** leakage — they are structured machine
fields.

**Why this matters:** the input image does not display these IDs. An MLLM trained
on this reasoning text would be taught to emit identifiers it has no way to
observe, which is an annotation artifact rather than spatial reasoning. The
reasoning text in its current form is **not directly usable as v0.1.1 MLLM
supervision**.

### 2.3 Reference appears in the distractor set (WARNING)

12,918 records (40.0%) list the reference component inside
`distractor_component_ids`. The reference is legitimately not the target, so this
is not wrong, but a distractor set intended to mean "wrong answers the model might
choose" should arguably exclude the *named reference*. Flagged as a definitional
ambiguity for v0.1.1 to settle, not as a data error.

---

## 3. Known statistics inconsistency — resolved

Task 4 reported Level 2 = 8,925, Type A = 8,700, Type B = 4,218, which cannot all
be true (8,700 + 4,218 = 12,918 > 8,925).

**Independent recount is authoritative:**

| Quantity | True value | Source of truth |
|---|---|---|
| Level 2 total | **8,925** | counted from JSONL by `level == 2` |
| Level 2 Type A (`reference → nearest`) | **4,707** | `largest_to_nearest` 2,235 + `smallest_to_nearest` 2,472 |
| Level 2 Type B (`reference → direction`) | **4,218** | 8 query types × `_to_{left,right,above,below}` |
| Type A + Type B | **8,925 = Level 2 total** ✓ | internally consistent |

**Root cause:** the generator counted Type A with
`query_type.endswith("_to_nearest")`. Every **Level-3** query type also ends in
`_to_nearest` (`largest_to_right_of_to_nearest`), so the filter swept them in:

```
8700 = 4707  (true Level 2 Type A)
     + 3993  (all Level 3)
```

`statistics.json` → `level2.reference_to_nearest = 8700` and the corresponding
`manifest.json` value are therefore **both wrong**. `reference_to_direction =
4218` happens to be correct. The `by_level` counts (19,366 / 8,925 / 3,993) and
`total_samples = 32,284` are all **correct**.

This defect is purely in the reported *decomposition*, not in the dataset
records, and is pinned by
`tests/test_dataset_validator.py::test_independent_counts_match_generator_bookkeeping`.

---

## 4. Recommended correction strategy

### Option A — filter to semantically-truthful samples (RECOMMENDED)

Keep only samples where hidden eligibility does **not** change the answer the
instruction expresses; discard the rest. No language change.

Measured retention:

| | Records | Share |
|---|---|---|
| Total | 32,284 | 100% |
| **Affected by >=1 semantic flag** (to discard) | **7,086** | **21.9%** |
| **Clean** (to keep) | **25,198** | **78.1%** |

Per flag (a record can trip more than one, so the flags sum to 7,775 > 7,086):

| Flag | Applicable | Violations | Share |
|---|---|---|---|
| `hidden_eligibility_largest` | 10,785 | 2,180 | 20.2% |
| `hidden_eligibility_smallest` | 7,468 | 4,246 | 56.9% |
| `hidden_eligibility_nearest` | 4,707 | 1,058 | 22.5% |
| `hidden_eligibility_level3_nearest` | 3,993 | 291 | 7.3% |

Discarding ~22% of the dataset is significant but correct: those samples are
semantically **wrong**, not merely noisy. `leftmost` / `rightmost` / `topmost` /
`bottommost` are entirely unaffected, because tile-relative extremes have no
eligibility filter.

### Option B — state the eligibility in the instruction

e.g. "among building regions that do not touch the image boundary, segment the
largest one".

**Why this is worse:** it teaches the model an annotation artifact. "Does not
touch the tile edge" is a property of *how the WHU tiles were cropped*, not of
buildings or of spatial reasoning. A model trained to condition on it would learn
a dataset-specific quirk that does not transfer, and the instruction becomes a
verbose hedge rather than a spatial description. It would also make v0.1's
instructions inconsistent in style with the tile-relative extremes, which need no
such qualifier.

### Recommendation

**Adopt Option A** for `BuildSpatialReason-v0.1.1`, plus:

1. **Drop internal IDs from reasoning text** and regenerate it ID-free, e.g.
   *"首先找到图像中面积最大的建筑区域，将其作为参考区域。随后比较其右侧候选建筑与参考区域的距离，选择距离最近的区域作为目标。"*
   / *"First identify the largest building region as the reference. Then compare
   the distances of the candidate regions to its right and select the closest one
   as the target."*
2. **Fix the Type-A counter** to count only `level == 2`, so `statistics.json`
   and `manifest.json` become self-consistent.
3. **Decide the distractor policy** for the reference component and document it.

---

## 5. Scene-level leakage

`scene_level_split_leakage = unverified`

The derived YOLO dataset carries no scene / geographic grouping metadata, and the
val split was a random 20% subset of the original WHU *train* pool. Scene-level
disjointness therefore **cannot be established from the available metadata**.
What *was* verified: no `image_id` overlap across splits, and **0** byte-identical
cross-split source images (3,921 images hashed).

Exact image duplication is therefore ruled out; scene-level adjacency is not.

---

## 6. Scope of this audit

Established by this audit:

* program-internal consistency — **verified sound**;
* geometry truth of targets and masks — **verified sound**;
* natural-language semantic truth — **FAILED** for **7,086 records (21.9%)**
  (7,775 flag instances across 2,180 + 4,246 + 1,058 + 291);
* split / duplication integrity — **verified sound**, scene level unverifiable;
* future MLLM-supervision suitability — **NOT suitable** as-is, because reasoning
  text leaks internal IDs in 100% of records.

Not established (and not claimed): whether the queries are *pedagogically*
sufficient for multi-hop spatial reasoning, and whether the template-generated
wording is linguistically varied enough for robust language generalisation.
