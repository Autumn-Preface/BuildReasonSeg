# TO_DSH — Task 5: BuildSpatialReason-v0.1 Dataset Validator & Semantic Quality Audit

> Status: **ACTIVE**
>
> Canonical repository: `BuildReasonSeg`
>
> This file is the authoritative task handoff from ChatGPT to DSH.

## 0. Before starting

1. Work only inside the current `BuildReasonSeg` workspace.
2. Treat `../WHU_Building_Segment/` as read-only legacy evidence.
3. Do **not** train any model, download data, install packages, or modify the external legacy project.
4. Do **not** modify or regenerate any existing file under:
   `datasets/build_spatial_reason/v0.1/`
5. Treat BuildSpatialReason-v0.1 as a frozen audit target. If problems are found, report them; do not silently repair v0.1.
6. The next corrected dataset version, if required, will be `BuildSpatialReason-v0.1.1`.

## 1. Goal

Independently validate the final 32,284-record BuildSpatialReason-v0.1 dataset from source component metadata and the frozen relation engine.

The validator must distinguish:

- program-internal consistency;
- geometry truth;
- natural-language semantic truth;
- split / duplication integrity;
- future MLLM-supervision suitability.

Do not simply trust `reasoning_steps`, `target_component_id`, `statistics.json`, `manifest.json`, or generator-produced intermediates.

Create:

- `spatial_reasoning/dataset_validator.py`
- `scripts/validate_build_spatial_reason.py`
- `tests/test_dataset_validator.py`
- `evaluation/build_spatial_reason_v0.1_quality.json`
- `docs/build_spatial_reason_v0.1_quality_audit.md`

Keep audit visualizations under:
`evaluation/build_spatial_reason_v0.1_samples/`

## 2. Independent dataset statistics

Read the final JSONL files directly and independently recompute:

- total records;
- split counts;
- Level 1 / Level 2 / Level 3 counts;
- query_type counts;
- trivial / nontrivial Level 3;
- samples per image.

Compare these numbers with `manifest.json`, `statistics.json`, and existing generated docs.

### Known inconsistency to verify independently

Task 4 reported:

- Level 2 total = 8,925;
- Type A reference→nearest = 8,700;
- Type B reference→direction = 4,218.

These cannot all be true.

Do **not** copy a number from this task file. Recount from JSONL and identify the true source-of-truth values and which existing report/statistic is stale or wrong.

## 3. Full target recomputation — all samples

For **every record**, independently execute its intended structured program from:

- source component metadata;
- frozen relation definitions;
- frozen thresholds;
- query_type semantics.

Compute:

`recomputed_target_component_id`

and compare against:

`target_component_id`.

Do not use the stored target as an input to the recomputation.

Report:

- checked count;
- pass count;
- failures;
- exact sample IDs and reasons.

Target should be 100% match.

## 4. Global natural-language semantic audit

This section is intentionally stricter than engine eligibility.

Natural language such as “largest building” and “nearest building” is normally interpreted over the visible components in the image, not over a hidden eligibility-filtered subset.

Audit whether hidden eligibility rules change the answer seen by a human/model.

### 4.1 largest

For every sample involving `largest`:

1. compute global argmax of `area_px` over all visible components in the image;
2. compare it with the dataset reference/target used as “largest”.

If the true global largest was excluded because of border/merge eligibility and the dataset uses another component, flag:

`hidden_eligibility_largest`.

### 4.2 smallest

Analogously compute global argmin over all visible components.

If the natural-language “smallest” differs from the engine-selected eligible smallest, flag:

`hidden_eligibility_smallest`.

### 4.3 nearest

For all reference→nearest samples:

1. compute boundary distance from the reference to **all other visible components**;
2. identify the semantic nearest before target-eligibility filtering;
3. compare with dataset target.

If a closer border/ineligible component exists and the dataset selects a farther eligible component, flag:

`hidden_eligibility_nearest`.

### 4.4 Level-3 nearest-within

For:

`reference → direction filter → nearest within filtered set`

first obtain all direction candidates using the frozen direction predicate. Then find the true nearest among that direction set **before nearest eligibility filtering**.

If dataset target differs because the closest directional candidate is ineligible, flag:

`hidden_eligibility_level3_nearest`.

For every violation record:

- image_id;
- sample_id;
- reference;
- all directional/semantic candidates;
- selected target;
- semantic target;
- exclusion reason.

## 5. Recommended semantic correction strategy

Do not modify v0.1.

Evaluate two correction strategies for v0.1.1:

### Option A — preferred unless evidence argues otherwise

Keep only samples where hidden eligibility filtering does **not** change the answer expressed by the natural-language instruction.

Examples:

- global largest is eligible → keep;
- global largest is ineligible → discard the query;
- semantic nearest is eligible → keep;
- semantic nearest is ineligible → discard the query.

### Option B

Make hidden eligibility explicit in language, e.g.:
“among building regions that do not touch the image boundary ...”

Discuss why this may teach annotation artifacts rather than useful spatial reasoning.

Recommend one option based on audit evidence.

## 6. Internal component-ID leakage audit

Scan:

- `instruction_zh`
- `instruction_en`
- `reasoning_zh`
- `reasoning_en`

for internal annotation IDs such as:

- `component 3`
- `component 12`
- Chinese equivalents such as `组件3`.

Differentiate:

- instruction leakage;
- reasoning-text leakage.

IDs inside structured machine fields such as `reasoning_steps` are **not** leakage.

Assess whether current natural-language reasoning can be used directly as future MLLM supervision when the input image does not display these IDs.

If unsuitable, propose a v0.1.1 ID-free reasoning style, e.g.:

Chinese:
“首先找到图像中面积最大的建筑区域，将其作为参考区域。随后比较其右侧候选建筑与参考区域的距离，选择距离最近的区域作为目标。”

English:
“First identify the largest building region as the reference. Then compare the distances of the candidate regions to its right and select the closest one as the target.”

Do not change v0.1 in this task.

## 7. Candidate / distractor integrity

Full-dataset checks:

- target must not occur in `distractor_component_ids`;
- reference must not be incorrectly included as target/distractor;
- all candidate IDs exist;
- candidate lists contain no duplicates;
- Level-2 Type-B requires exactly one direction candidate;
- Level-3:
  `nearest_eligible_component_ids ⊆ candidate_component_ids`;
- recompute both candidate sets independently and compare.

## 8. Mask selector integrity

For all records, check:

`target_mask.component_id == target_component_id`.

Resolve the referenced component map and verify the target ID exists and yields a non-empty binary mask.

Prefer full-dataset validation at current scale.

## 9. Reference integrity

- Level 1: `reference_component_ids == []`;
- Level 2/3: reference exists in source image;
- reference != target;
- structured-program reference agrees with `reference_component_ids`;
- Level 3 step 1 output equals the reference.

## 10. Structured program integrity

Validate for every sample:

- continuous step numbering;
- operation sequence matches level/query_type;
- every referenced component exists;
- no circular/self-invalid dependency;
- candidate lists have no duplicate IDs;
- output target appears only at the appropriate step;
- Level 1 uses the expected one-step direct operation;
- Level 2 uses the expected two logical operations;
- Level 3 uses the expected three logical operations.

## 11. Template distribution and language parity

Using `templates.py` and the deterministic selector, reconstruct the expected template choice from the semantic key.

Quantify for every `query_type × language`:

- template usage counts;
- template shares;
- unused templates;
- severe imbalance, if any.

Verify Chinese and English instructions correspond to the same semantic template/program.

Programmatically scan for:

- raw placeholders;
- `{...}`;
- `None`;
- `null`;
- doubled punctuation;
- obvious repeated articles;
- empty sentences;
- unexpected Chinese in English template;
- unexpected English in Chinese template.

Do not use an LLM for validation.

## 12. Cross-split image duplicate audit

Beyond image_id separation, hash the underlying source image files.

Check byte-identical cross-split duplicates across train/val/test.

Report:

`exact_image_duplicate_cross_split = N`.

If no exact duplicates, record 0.

### Scene-level leakage

Do not claim scene-level leakage is absent unless it can actually be established from available metadata.

If original scene/tile grouping cannot be reliably reconstructed, report:

`scene_level_split_leakage = unverified`.

Explain why.

## 13. Distribution audit

Compare train/val/test for:

- level proportions;
- query_type proportions;
- target area distribution;
- target centroid distribution;
- source components/image.

Simple descriptive differences are enough.

Do not resplit the dataset in this task.

## 14. Trivial / nontrivial Level-3 audit

Independently verify the reported:

- trivial selection count;
- nontrivial selection count.

Check:

`trivial_selection == true`

iff the direction-filtered nearest candidate set has exactly one eligible candidate.

Recommend that trivial Level-3 not be included in the primary multi-hop reasoning metric.

## 15. Visualization audit pack

Generate a lightweight pack:

- 6 Level-1;
- 8 Level-2;
- 10 nontrivial Level-3;
- 4 trivial Level-3;
- if semantic violations exist, up to 8 representative failure cases.

Each audit visualization should show:

- source image;
- reference outline;
- target outline;
- candidate IDs as **audit overlay only**;
- instruction;
- query_type.

Generate:

`evaluation/build_spatial_reason_v0.1_samples/contact_sheet.png`

Keep the whole pack reasonably small (preferably <10 MB).

Do not claim manual visual inspection unless images were actually inspected.

## 16. Audit verdict

The final status must be exactly one of:

- `PASS`
- `PASS_WITH_WARNINGS`
- `FAIL_REQUIRES_REVISION`

Default to `FAIL_REQUIRES_REVISION` if any of the following is non-zero:

- target recomputation failures;
- instruction semantic mismatches;
- hidden eligibility violations that make the natural-language answer false or misleading.

If revision is required, recommend `BuildSpatialReason-v0.1.1`.

Do not overwrite v0.1.

## 17. Tests

Add validator tests that cover at least:

- independent count reconstruction;
- target recomputation;
- hidden-largest mismatch fixture;
- hidden-smallest mismatch fixture;
- hidden-nearest mismatch fixture;
- Level-3 hidden nearest fixture;
- component-ID leakage detection;
- candidate/distractor checks;
- mask selector integrity;
- template reconstruction;
- exact image-hash leakage helper.

Run all existing tests plus the new validator tests.

## 18. Final DSH report

Write the complete report to:

`handoff/FROM_DSH.md`

Use these sections:

1. Audit Verdict
2. Files Created / Modified
3. Independent Dataset Counts
4. Existing Statistics Consistency
5. Full Target Re-computation
6. Largest Semantic Audit
7. Smallest Semantic Audit
8. Nearest Semantic Audit
9. Level-3 Hidden Eligibility Audit
10. Component-ID Leakage
11. Structured Program Integrity
12. Candidate / Distractor Integrity
13. Mask Selector Integrity
14. Template Distribution
15. Language Parity
16. Split / Exact Duplicate Audit
17. Distribution Audit
18. Trivial / Nontrivial Level-3 Audit
19. Visualization Pack
20. Issues Found
21. Recommended Fixes
22. Ready for v0.1.1 or Task 5.5?

Be explicit about every mismatch and do not hide failed checks.

## 19. Update project handoff state

At the end, update:

`handoff/PROJECT_STATE.md`

with a concise current state:

- project codename;
- current dataset version;
- completed tasks;
- frozen relation rules;
- Task 5 verdict;
- current blockers;
- recommended next task.

Do not put the full report there.

## 20. Git handoff

After all work and tests are complete:

1. run `git status`;
2. ensure no large ignored/generated dataset binaries were accidentally staged;
3. commit all intended Task 5 code/docs/audit outputs plus:
   - `handoff/FROM_DSH.md`
   - `handoff/PROJECT_STATE.md`
4. commit message:
   `audit: validate BuildSpatialReason v0.1`
5. push to `origin/main`.

Do **not** modify `handoff/TO_DSH.md` except, optionally, to add a final line saying the task has been completed. Prefer leaving the task text intact for audit history.

If push is blocked by permission/network policy, complete all local work and report the exact blocker in `handoff/FROM_DSH.md`.
