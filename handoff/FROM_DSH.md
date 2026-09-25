# FROM_DSH — Task 5B Report: BuildSpatialReason-v0.1.1 Corrective Regeneration & Acceptance Audit

**Date:** 2026-08-04
**Actor:** DSH
**Task:** Task 5B (from `handoff/TO_DSH.md`)
**Verdict: `PASS`**

---

## 1. Verdict

**`PASS`** — BuildSpatialReason-v0.1.1 was generated and passed the full
acceptance audit with **zero** blocking counts and **zero** issues of any
severity (no errors, no warnings).

| | v0.1 | v0.1.1 |
|---|---|---|
| Verdict | `FAIL_REQUIRES_REVISION` | **`PASS`** |
| Hidden semantic violations | 7,086 records (21.9%) | **0** |
| Reasoning ID leakage | 32,284 records (100%) | **0** |
| Reference in distractors | 12,918 records | **0** |
| Target recomputation | 32,284/32,284 | **25,229/25,229** |
| `template_id` verification | field absent | **25,229/25,229** |

v0.1 remains byte-identical and frozen. No Task 5.5 or later work was started.

---

## 2. Scope / Safety Compliance

| Constraint | Status |
|---|---|
| Writes confined to `BuildReasonSeg/` | **complied** |
| `../WHU_Building_Segment/` read-only | **complied** — 11/11 core SHA256 unchanged, 9,014 files |
| No package installation | **complied** |
| No model/dataset download | **complied** |
| No system/registry/PATH changes | **complied** |
| No unrelated file deletion | **complied** |
| No model training | **complied** |
| `datasets/build_spatial_reason/v0.1/` not modified | **complied** — verified by SHA256, see §14 |
| Task 5.5 / later tasks not started | **complied** |

**Full Access note.** The session ran under `danger-full-access` because the
`workspace-write` sandbox runner could not start on this Windows host (its
per-run temp directory under `C:\Users\ROG\AppData\Local\Temp` did not exist and
lies outside the workspace, so file tools could not create it). Full Access was
used purely as a technical workaround; all writes stayed inside `BuildReasonSeg/`
plus ordinary temp files.

---

## 3. Files Created / Modified

**Created:**

| File | Purpose |
|---|---|
| `spatial_reasoning/semantic_policy.py` | Single shared semantic visibility policy (generator + validator + recomputation) |
| `configs/build_spatial_reason_v0.1.1.yaml` | v0.1.1 generation config, incl. `semantic_visibility_policy_version: "1.0"` |
| `tests/test_v011_acceptance.py` | 19 v0.1.1 acceptance tests |
| `docs/build_spatial_reason_v0.1.1.md` | v0.1.1 dataset documentation |
| `docs/build_spatial_reason_v0.1.1_quality_audit.md` | Acceptance audit report |
| `evaluation/build_spatial_reason_v0.1.1_quality.json` | Machine-readable acceptance audit |
| `evaluation/build_spatial_reason_v0.1.1_samples/` | 28 PNGs + `contact_sheet.png` (6.67 MB) |
| `datasets/build_spatial_reason/v0.1.1/{train,val,test}.jsonl`, `manifest.json`, `statistics.json` | The dataset (JSONL git-ignored) |

**Modified:**

| File | Change |
|---|---|
| `spatial_reasoning/annotator.py` | Semantic policy integration; ID-free reasoning renderer; `template_id`; distractor rule; `SAMPLE_ID_VERSION` separation; new discard reasons |
| `spatial_reasoning/templates.py` | `ROLE_PHRASE`, `DIRECTION_RELATIVE_TO_REFERENCE`; grammar fixes |
| `spatial_reasoning/dataset_validator.py` | Version parameterization; semantic-policy recomputation; `verify_v01_unchanged`; `decide_verdict` moved here |
| `scripts/build_spatial_reason.py` | Version-aware manifest + fixed Level-2 counter + invariant assertion |
| `scripts/validate_build_spatial_reason.py` | `--version`; new counters; `template_id` check; Level-3 checks; machine-readable fields |
| `scripts/build_spatial_reason_samples.py` | Version parameterization and pack composition |
| `tests/test_annotator.py` | Updated 3 assertions to v0.1.1 semantics |
| `tests/test_dataset_validator.py` | Recomputation now targets v0.1.1 |
| `docs/architecture_decisions.md` | **ADR-010** added |

**Not modified:** `datasets/build_spatial_reason/v0.1/**`, `../WHU_Building_Segment/**`,
`handoff/TO_DSH.md` (left intact for audit history, per §22).

---

## 4. Generator Changes

1. **Semantic-first resolution.** `generate_level1/2/3` no longer ask the relation
   engine for an eligibility-filtered answer. They call
   `semantic_policy.resolve_size_extreme` / `resolve_nearest` /
   `direction_candidates_over_visible`, which compute the answer over **all
   visible components** and report whether it is admissible. A new
   `_resolve_reference` helper replaces the three duplicated
   `image_relations.size_rank.get(...)` blocks.
2. **Discard instead of substitute.** When the semantic answer is ineligible or
   ambiguous the query is dropped, with a specific discard reason
   (`semantic_target_ineligible`, `semantic_ambiguous`,
   `nearest_semantic_ineligible`, `level3_nearest_semantic_ineligible`).
3. **Level-3 uses the full direction set** for the nearest comparison, so a
   closer-but-ineligible direction candidate is never skipped.
4. **ID-free reasoning renderer.** `render_reasoning` emits role-based prose
   (`面积最大的建筑区域` / `the largest building region`) and never an id.
5. **`template_id` recorded** on every `CandidateQuery` and sample. It was
   missing from Level-2 Type-B queries in v0.1.
6. **Distractors exclude target and all references.**
7. **Level-2 counter fixed**: both halves now require `level == 2`, plus a hard
   assertion that `TypeA + TypeB == by_level["2"]`.
8. **Reproducibility split from version.** `SAMPLE_ID_VERSION` stays `"v0.1"`
   (historical sample ids stay reproducible) while `GENERATOR_VERSION` is now
   `"v0.1.1"` and is recorded per sample and in the manifest.

**No threshold was changed.** The generator asserts at startup that its config
agrees with the frozen `ratio_margin` and direction preset.

---

## 5. Semantic Policy Changes

**Rule:** natural language is defined over **all visible components**. Eligibility
may reject a sample; it may never silently change its answer.

| Question | Semantic universe | Kept only if |
|---|---|---|
| `largest` / `smallest` | global argmax/argmin of `area_px` over all visible | that exact component is eligible **and** `larger/smaller >= 1.10` |
| `<ref>_to_nearest` | boundary distance to all other visible | true nearest is nearest-eligible **and** passes the margin |
| `<ref>_to_<dir>` | frozen direction predicate over all visible | exactly one candidate |
| `<ref>_to_<dir>_to_nearest` | nearest within the **full** direction-valid set | that nearest is eligible **and** passes the margin |

Implemented once in `spatial_reasoning/semantic_policy.py` and shared by the
generator, the validator's recomputation and the acceptance audit, so the
generator and the verifier cannot disagree about what a question means — the
root cause of the v0.1 defect.

---

## 6. v0.1.1 Counts (actual, not target-driven)

```
total    : 25229
by_split : train 15592 | val 3884 | test 5753
by_level : L1 17275 | L2 5036 | L3 2918
level2   : reference_to_nearest 2272 + reference_to_direction 2764 = 5036 (invariant holds)
level3   : trivial 1261 | nontrivial 1657
per image: min 1 | p5 3 | median 6 | p95 11 | max 12
images with samples: 3920
```

No count target was imposed. The reduction from 32,284 to 25,229 is the
**intended consequence** of discarding semantically untruthful samples.

Discard reasons:

| Reason | Count |
|---|---|
| `semantic_target_ineligible` | 10,283 |
| `no_direction_candidate` | 9,501 |
| `semantic_ambiguous` | 5,362 |
| `multiple_direction_candidates` | 4,265 |
| `ambiguous` | 1,657 |
| `level3_nearest_semantic_ineligible` | 1,217 |
| `nearest_semantic_ineligible` | 692 |
| `quota_downsample` | 539 |
| `nearest_margin_fail` | 360 |
| `single_component_image` | 116 |

---

## 7. v0.1 vs v0.1.1 Comparison

| metric | v0.1 | v0.1.1 | change |
|---|---|---|---|
| total | 32,284 | 25,229 | −7,055 (−21.9%) |
| train | 19,977 | 15,592 | −4,385 |
| val | 4,981 | 3,884 | −1,097 |
| test | 7,326 | 5,753 | −1,573 |
| Level 1 | 19,366 | 17,275 | −2,091 |
| Level 2 | 8,925 | 5,036 | −3,889 |
| Level 3 | 3,993 | 2,918 | −1,075 |
| L2 Type A | 4,707 (reported as 8,700) | 2,272 | |
| L2 Type B | 4,218 | 2,764 | |
| L3 trivial | 1,804 | 1,261 | criterion refined |
| L3 nontrivial | 2,189 | 1,657 | |
| hidden semantic violations | 7,086 | **0** | |
| reasoning id leaks | 32,284 | **0** | |
| reference in distractors | 12,918 | **0** | |
| `template_id` present | no | **yes (100% verified)** | |
| verdict | FAIL_REQUIRES_REVISION | **PASS** | |

Note the v0.1 Type-A figure: the historical report said 8,700, which was the
buggy counter. The true v0.1 Type A was **4,707** (verified in Task 5).

---

## 8. Target Recomputation

| Metric | Value |
|---|---|
| checked | **25,229** |
| pass | **25,229** |
| fail | **0** |
| match rate | **100.000%** |

Every record was recomputed under the semantic policy from source component
metadata, the frozen relations and the frozen thresholds. The stored
`target_component_id` was never used as an input; Level-2 Type-B targets were
re-derived from the direction predicate, and all filter sets were re-derived and
compared.

---

## 9. Hidden Eligibility Audit

| Flag | v0.1 | v0.1.1 |
|---|---|---|
| `hidden_eligibility_largest` | 2,180 | **0** |
| `hidden_eligibility_smallest` | 4,246 | **0** |
| `hidden_eligibility_nearest` | 1,058 | **0** |
| `hidden_eligibility_level3_nearest` | 291 | **0** |
| semantic violation flag instances | 7,775 | **0** |
| `semantic_violation_unique_records` | 7,086 | **0** |
| `semantic_clean_records` | 25,198 | **25,229** |

The audit is deliberately stricter than engine eligibility: it recomputes the
global semantic answer over all visible components and compares it with the
dataset's choice. Zero means no stored instruction can be read as false.

---

## 10. Reasoning Leakage Audit

| Counter | v0.1 | v0.1.1 |
|---|---|---|
| `leak_reasoning_records` | 32,284 (was reported as `leak_reasoning_samples`) | **0** |
| `leak_reasoning_fields` | — | **0** |
| `leak_reasoning_mentions` | 159,014 | **0** |
| `leak_instruction_records` | 0 | **0** |

The ambiguous legacy field `leak_reasoning_samples` is no longer produced. The
three counters requested by §11 are emitted, all zero.

Sample of the new reasoning style:

```
reasoning_zh : 首先确定图像中面积最大的建筑区域，将其作为参考区域。随后筛选位于参考区域右侧的建筑区域。最后比较这些候选区域与参考区域的边界距离，选择距离最近的区域作为目标。
reasoning_en : First identify the largest building region as the reference region. Then identify the building regions on the right side of the reference region. Finally compare their boundary distances to the reference region and select the nearest one as the target.
```

`reasoning_steps` still carries numeric ids (`output_component_id`,
`reference_component_id`, `candidate_component_ids`) for machine verification.

---

## 11. Template ID Audit

| Metric | Value |
|---|---|
| checked | **25,229** |
| verified | **25,229** |
| mismatch | **0** |
| missing | **0** |
| not reconstructed | **0** |
| zh/en semantic parity | **100%** (zero mismatches) |

`template_id` is deterministic, identical for the zh/en pair, and independently
re-verified by re-rendering both languages and matching them back to a template
pair. All 39 declared templates are used.

Language hygiene: zero raw `{placeholder}`, zero literal `None`/`null`, zero
empty language fields, zero doubled punctuation.

---

## 12. Distractor Policy Audit

| Check | v0.1 | v0.1.1 |
|---|---|---|
| target in distractors | 0 | **0** |
| **reference in distractors** | 12,918 | **0** |
| duplicate distractors | 0 | **0** |
| missing distractor components | 0 | **0** |

`distractor_component_ids` now excludes both the target and every explicit
reference. Internal candidate sets are separate and retain the members the
reasoning logic needs.

---

## 13. Statistics Consistency

The v0.1 defect is fixed and now guarded by an assertion.

| Statistic | v0.1 | v0.1.1 |
|---|---|---|
| `level2.reference_to_nearest` | 8,700 (**wrong**; true 4,707) | **2,272** |
| `level2.reference_to_direction` | 4,218 | **2,764** |
| sum vs `by_level["2"]` | 12,918 ≠ 8,925 (**violated**) | 5,036 = 5,036 ✓ |
| `invariant_holds` field | absent | **true** |

Root cause (v0.1): the counter omitted the `level == 2` guard, so
`endswith("_to_nearest")` also matched all 3,993 Level-3 types
(4,707 + 3,993 = 8,700). `scripts/build_spatial_reason.py` now restricts both
halves to `level == 2` and **raises** if the invariant fails.

---

## 14. Split / Duplicate Audit

| Check | Result |
|---|---|
| `train ∩ val` image ids | **∅** |
| `train ∩ test` image ids | **∅** |
| `val ∩ test` image ids | **∅** |
| duplicate `sample_id` | **0** |
| duplicate canonical semantic key | **0** |
| **`exact_cross_split_duplicates`** | **0** |
| images hashed | train 2,423 / val 614 / test 883 |

**v0.1 frozen integrity:** all five artefacts still hash to their recorded
values — `verify_v01_unchanged` returns `unchanged: True`. A mismatch would raise
`v01_modified` as a blocking error; it did not fire.

**Scene-level leakage:** `unverified` (see §18).

---

## 15. Level-3 Trivial / Nontrivial

| Subset | Count |
|---|---|
| nontrivial | **1,657** |
| trivial | **1,261** |
| total | 2,918 |

`trivial_selection` was independently re-verified against
`semantic_policy.admissible_nearest_ids`: **0** mismatches. The criterion was
refined during this task — it now means "exactly one **admissible** candidate
after the direction filter and the frozen margin rule", rather than the raw
direction-set size. This makes the flag consistent between generator and
validator and correctly classifies chains whose direction set is larger but whose
admissible set is a singleton.

Trivial samples are **not** deleted; the primary multi-hop metric should use the
nontrivial subset.

---

## 16. Visualization Pack

| Item | Value |
|---|---|
| `visualizations_generated` | **28** + contact sheet = 29 files |
| Composition | 4 Level-1, 6 Level-2, 8 nontrivial Level-3, 4 trivial Level-3, 6 warning/failure cases (none required — pack size driven by spec) |
| `contact_sheet.png` | present |
| Size | **6.67 MB** |
| Location | `evaluation/build_spatial_reason_v0.1.1_samples/` |

`manual_visual_inspection`: **NOT claimed.** I did not visually inspect the v0.1.1
panels individually. Component ids appear only in the audit overlay, never in the
natural-language supervision.

---

## 17. Test Execution Summary

Exact commands and results (run from the repository root):

```
python tests/test_component_conversion.py   ->  11/11 checks passed   exit 0
python tests/test_geometry.py               ->  17/17 checks passed   exit 0
python tests/test_relations.py              ->  35/35 checks passed   exit 0
python tests/test_annotator.py              ->  22/22 checks passed   exit 0
python tests/test_dataset_validator.py      ->  18/18 checks passed   exit 0
python tests/test_v011_acceptance.py        ->  19/19 checks passed   exit 0
```

| Metric | Value |
|---|---|
| total collected | **122** |
| passed | **122** |
| failed | **0** |
| skipped | **0** |
| exit code | **0** |

Also run:

```
python scripts/build_spatial_reason.py --gen-config configs/build_spatial_reason_v0.1.1.yaml    -> exit 0
python scripts/validate_build_spatial_reason.py --version v0.1.1                               -> exit 0, verdict PASS
```

Three v0.1-era test assertions were updated because the v0.1.1 semantics changed
deliberately:

1. `test_annotator.py` determinism no longer compares against the **on-disk
   v0.1** artefact (produced under the superseded policy); the two-run
   byte-identity assertion remains, and the on-disk equivalent is covered for
   v0.1.1 in `test_v011_acceptance.py`.
2. `test_annotator.py` now requires the Level-3 nearest step to cover the **full
   direction set** rather than the eligible subset.
3. `test_annotator.py` trivial-flag test and
   `test_dataset_validator.py` recomputation test now use the v0.1.1 semantics.

These are test expectation updates, not suppressions: every underlying property is
still asserted, in the form the corrected policy defines.

---

## 18. Known Limitations

1. **Border truncation reduces coverage.** Because the semantic answer must itself
   be eligible, an image whose largest/smallest building touches the tile edge
   produces **no** query of that type. This is why Level 2 dropped most
   (`semantic_target_ineligible` 10,283). Coverage is traded for correctness
   deliberately.
2. **`scene_level_split_leakage = unverified`.** The derived dataset carries no
   scene/geographic grouping metadata and val was a random 20% subset of the
   original train pool, so geographic adjacency between splits cannot be excluded.
   Exact image duplication is ruled out (0). WHU was **not** resplit.
3. **Source annotation is connected components**, not verified physical buildings;
   touching buildings were merged upstream.
4. **No building-function semantics** exists in the source data.
5. **No overlap / containment relations** — components are disjoint by
   construction.
6. **Language is template-generated**; `template_id` is now recorded so diversity
   is measurable, but wording variety is still bounded by 39 templates.
7. **Tile-edge truncation is a cropping artifact**, not a property of buildings —
   which is why Option B (naming eligibility in the instruction) was rejected.
8. **`peak_memory_mb` is unreliable.** The Windows ctypes fallback for peak RSS
   returned 0; peak memory was not confirmed and must not be read as "zero".
9. **v0.1 is no longer regenerable byte-identically** by the current code path,
   because the semantic policy changed. v0.1 is preserved on disk and verified
   unchanged; regeneration is intentionally not attempted (§12 backward safety).

---

## 19. Git Status / Commit / Push

Pre-staging inspection confirmed: no dataset JSONL, no weights/checkpoints, and no
out-of-repo changes staged; `../WHU_Building_Segment/` untouched.

```
commit : fix: regenerate BuildSpatialReason v0.1.1
branch : main
push   : origin/main
```

Commit hash and push result are recorded in the final DSH chat summary.

---

## 20. Recommended Next Task

**ChatGPT review of the pushed commit**, then — if approved — move on to the
next planned stage (MLLM integration / reasoning segmentation MVP) as a
**separate** task.

Within the dataset line, two small follow-ups remain worth doing at some point,
neither blocking:

1. **Quantify template diversity** per query type using the new `template_id`
   field, to judge whether 39 templates are enough for language generalisation.
2. **Consider whether the `largest`/`smallest` coverage loss** (Level 2 fell from
   8,925 to 5,036) is acceptable, or whether a richer source annotation with
   non-truncated buildings would recover it. This is a data-sourcing question, not
   a policy one — relaxing the policy would reintroduce false instructions.

**Task 5.5 was not started. No later task was started.**
