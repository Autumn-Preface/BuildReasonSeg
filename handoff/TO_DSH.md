# TO_DSH — Task 5B: BuildSpatialReason-v0.1.1 Corrective Regeneration & Acceptance Audit

> Status: ACTIVE
> Repository: `BuildReasonSeg`
> Goal: create and validate **BuildSpatialReason-v0.1.1** as the corrected successor to frozen v0.1.

## 0. User-facing language

Only the DSH web/chat output shown to the user must be Chinese:
- progress summaries;
- permission/escalation explanations;
- warnings;
- final summary.

Commands, code identifiers, paths, logs, field names may remain English.

`handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` may remain English because they are primarily for ChatGPT review.

## 1. Full-access boundary

DSH is running with Full Access only because the normal workspace-write sandbox cannot start on this Windows host. Full Access is a technical workaround, not broader task authorization.

Allowed writes:
- only inside `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\`
- ordinary system temp directories if required by the runtime.

Allowed read-only external access:
- `..\WHU_Building_Segment\` only when existing code needs legacy source images, labels, or frozen baseline evidence.

Allowed network/Git:
- `git status`, `diff`, `log`, `add`, `commit`, `fetch`, `pull`, `push origin main`.

Forbidden:
- modifying anything outside `BuildReasonSeg`;
- modifying `../WHU_Building_Segment/`;
- package installation (`pip`, `conda`, `npm`, etc.);
- downloading models, datasets, checkpoints, scripts, or external code;
- changing system settings, registry, PATH, shell profiles;
- deleting unrelated files;
- accessing unrelated user directories;
- model training;
- Task 5.5 or later tasks.

If a required action conflicts with these limits, do not perform it. Explain the blocker in Chinese in the DSH UI and record it in `handoff/FROM_DSH.md`.

## 2. v0.1 freeze

`datasets/build_spatial_reason/v0.1/` is immutable historical evidence.

You may read it but must not modify, regenerate, rename, overwrite, or delete it.

All corrected output must go to:

`datasets/build_spatial_reason/v0.1.1/`

Verify at the end that v0.1 is unchanged.

## 3. Task 5 defects to correct

Historical audit findings:
- v0.1 total: 32,284;
- target recomputation: 32,284/32,284;
- hidden semantic eligibility defects:
  - largest 2,180;
  - smallest 4,246;
  - nearest 1,058;
  - Level-3 nearest 291;
- previous report said 7,086 unique records hit >=1 semantic defect;
- reasoning text had component-ID leakage in 100% of records;
- instructions had 0 ID leakage;
- Level-2 Type-A was reported as 8,700 but true v0.1 Type A was 4,707 because 3,993 Level-3 records were accidentally included;
- references appeared in distractors for 12,918 records;
- scene-level leakage remains unverified.

These numbers are historical evidence only. Do NOT hard-code rejection IDs or expected v0.1.1 counts.

## 4. Core semantic policy

Natural-language semantics are defined over **all visible components**, not a hidden eligibility-filtered subset.

Eligibility may reject a sample; it may never silently change the answer.

### 4.1 largest / smallest

For every sample using `largest` or `smallest`, including when used only as a Level-2/3 reference:

1. compute the global semantic extreme over all visible components:
   - largest = global argmax of rasterized `area_px`;
   - smallest = global argmin of rasterized `area_px`;
2. apply the frozen size ambiguity/margin rule;
3. if that exact semantic extreme is eligible, it may be used;
4. if it is excluded by border/tiny/merge quality logic, discard the query;
5. never substitute the second-ranked eligible component;
6. if ambiguous under frozen `ratio_margin = 1.10`, discard.

Do not change frozen relation thresholds.

### 4.2 Level-2 reference -> nearest

For `largest_to_nearest` / `smallest_to_nearest`:

1. get the semantic reference using 4.1;
2. compute boundary distance from that reference to **all other visible components**;
3. semantic nearest = closest visible component before nearest eligibility filtering;
4. keep only if that exact nearest is nearest-eligible and passes the frozen nearest margin/ambiguity rule;
5. otherwise discard;
6. never replace it with a farther eligible component.

### 4.3 Level-2 reference -> direction

For `largest_to_left_of`, `smallest_to_above`, etc.:

- reference must first satisfy 4.1;
- evaluate the frozen direction predicate over visible components;
- keep only when the intended uniqueness/ambiguity conditions hold;
- never replace an invalid semantic reference.

### 4.4 Level-3 reference -> direction -> nearest

For `largest_to_{direction}_to_nearest`:

1. obtain true global semantic largest;
2. if invalid under 4.1, discard;
3. evaluate the frozen direction predicate over all visible components;
4. form the complete direction-valid set;
5. compute the true nearest within that full set by frozen boundary distance;
6. keep only if that exact nearest is nearest-eligible and passes the frozen nearest ambiguity rule;
7. otherwise discard;
8. never remove a closer ineligible component and choose the second-nearest.

## 5. Correct generator, do not just post-filter v0.1

Preferred implementation:
- change generator logic so semantic validity is checked before final sample acceptance/quota selection;
- deterministically generate v0.1.1 from source component geometry and frozen relation rules;
- do not load v0.1 and delete a hard-coded list of sample IDs.

The corrected generator itself must become the source of truth.

## 6. Version/config

Create:

`configs/build_spatial_reason_v0.1.1.yaml`

Generate under:

`datasets/build_spatial_reason/v0.1.1/`

Include provenance/version fields sufficient to identify:
- dataset version;
- generator version;
- relation config version;
- component representation version;
- semantic visibility policy version.

Recommended:
`semantic_visibility_policy_version: "1.0"`

Do not rename the project, WHU source dataset, or frozen relation config.

## 7. ID-free natural-language reasoning

Structured `reasoning_steps` may keep numeric component IDs for machine verification.

But these natural-language fields must contain zero internal IDs:
- `reasoning_zh`
- `reasoning_en`

Forbidden examples:
- `component 3`
- `component 12`
- `组件3`
- `构件 7`

Use role-based prose, e.g.:

Chinese:
`首先找到图像中面积最大的建筑区域，将其作为参考区域。随后筛选位于参考区域右侧的建筑区域。最后比较这些候选区域与参考区域的边界距离，选择最近的区域作为目标。`

English:
`First identify the largest building region as the reference. Then identify the building regions to the right of the reference. Finally compare their boundary distances to the reference and select the nearest region as the target.`

The prose should explain the operation, not expose annotation IDs.

## 8. Explicit template_id

Add per-sample:

`template_id`

Requirements:
- deterministic;
- same semantic ID for zh/en pair;
- validator confirms rendered zh/en matches recorded `template_id`;
- keep existing `template_version` if useful.

## 9. Distractor policy

Adopt this rule:

`distractor_component_ids` excludes both:
- target;
- every explicit reference component.

Reference is reasoning context, not a target distractor.

Candidate sets used internally for reasoning are separate and may retain algorithmically necessary members.

## 10. Fix Level-2 statistics

Type A:
- `level == 2`
- and query is `*_to_nearest`

Type B:
- `level == 2`
- and one-hop reference->direction query.

Add invariant:

`level2.reference_to_nearest + level2.reference_to_direction == by_level["2"]`

Generation/audit must fail if false.

## 11. Fix validator counter semantics

Do not reuse the ambiguous old field `leak_reasoning_samples`.

Use:
- `leak_reasoning_records`
- `leak_reasoning_fields`
- `leak_reasoning_mentions`

Acceptance target for all three: 0.

Also add machine-readable:
- `semantic_violation_unique_records`
- `semantic_clean_records`

Do not derive the unique semantic union only in prose.

## 12. Backward safety

Refactor narrowly so v0.1.1 uses the new semantic policy without touching v0.1.

Preserve:
- deterministic seed behavior;
- frozen relation thresholds;
- current component representation.

If exact historical v0.1 regeneration becomes awkward, do not risk v0.1. It is sufficient that on-disk v0.1 stays frozen and current code correctly generates v0.1.1.

Document compatibility limitations if any.

## 13. ADR/docs

Update `docs/architecture_decisions.md` with a new ADR (e.g. ADR-010) covering:

1. natural-language semantic universe = all visible components;
2. eligibility may reject a sample but may not silently change its answer;
3. quality/GT geometry remains annotation/supervision logic, not inference input;
4. reference components are excluded from distractors;
5. natural-language reasoning is ID-free while structured fields retain IDs.

Create/update:

`docs/build_spatial_reason_v0.1.1.md`

Include:
- changes from v0.1;
- frozen rules;
- actual sample counts;
- semantic policy;
- known limitations.

## 14. Regenerate v0.1.1

Generate full:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `manifest.json`
- `statistics.json`

under `datasets/build_spatial_reason/v0.1.1/`.

Full JSONL files remain local/ignored by Git.

Do NOT force final count to 32,284, 25,198, or any predetermined value. Let the corrected deterministic generator and quota logic determine it.

Report actual counts.

## 15. Acceptance validator

Refactor validator/CLI so it explicitly supports v0.1.1, e.g.:

`python scripts/validate_build_spatial_reason.py --version v0.1.1`

Run on all v0.1.1 records.

Blocking counts that must be zero:
- target recomputation failures;
- hidden_eligibility_largest;
- hidden_eligibility_smallest;
- hidden_eligibility_nearest;
- hidden_eligibility_level3_nearest;
- instruction semantic mismatch;
- reasoning component-ID leakage;
- target in distractors;
- reference in distractors;
- missing/empty target mask;
- template_id mismatch;
- duplicate sample IDs;
- structured-program errors.

Required positive checks:
- target recomputation = 100%;
- template_id verification = 100%;
- Type A + Type B = Level 2 total;
- exact cross-split byte duplicate count = 0;
- zh/en semantic parity = 100%;
- v0.1 unchanged.

Final v0.1.1 verdict may be:
- `PASS`
- `PASS_WITH_WARNINGS`

If `FAIL_REQUIRES_REVISION`, stop and do not proceed.

## 16. Scene-level leakage

If geographic/scene grouping still cannot be reconstructed:

`scene_level_split_leakage = unverified`

Do not claim absence and do not resplit WHU in this task.

## 17. Level-3 policy

Continue to distinguish:
- trivial Level 3;
- nontrivial Level 3.

Recompute counts for v0.1.1.

Primary multi-hop metric should use nontrivial Level 3; trivial Level 3 should be reported separately.

Do not delete trivial samples merely because they are trivial.

## 18. Visualization pack

Create a compact pack under:

`evaluation/build_spatial_reason_v0.1.1_samples/`

Suggested:
- 4 Level 1;
- 6 Level 2;
- 8 nontrivial Level 3;
- 4 trivial Level 3;
- representative warning cases if warnings remain;
- `contact_sheet.png`.

Candidate/component IDs may appear in audit overlays only, never in natural-language supervision.

Do not claim full manual inspection unless actually done.

## 19. Tests and execution report

Add/update tests for at least:

1. global largest is never silently replaced;
2. global smallest is never silently replaced;
3. nearest visible target is never replaced by farther eligible target;
4. Level-3 nearest uses the full direction-valid set;
5. semantic-invalid samples are rejected before acceptance;
6. natural-language reasoning has no internal IDs;
7. structured reasoning_steps may retain IDs;
8. template_id stored and matches zh/en;
9. reference excluded from distractors;
10. target excluded from distractors;
11. Level-2 Type-A + Type-B invariant;
12. validator version parameterization;
13. exact image-hash duplicate helper;
14. deterministic regeneration on a representative subset;
15. v0.1 remains unmodified where practical.

Run the full existing test suite plus all new tests.

`handoff/FROM_DSH.md` must explicitly record:
- exact test command(s);
- total collected;
- passed;
- failed;
- skipped;
- exit code.

Do not merely state “tests passed”.

## 20. Machine-readable audit output

Create:

`evaluation/build_spatial_reason_v0.1.1_quality.json`

Include at least:
- verdict;
- total/split/level/query_type counts;
- Level-2 Type-A/Type-B;
- Level-3 trivial/nontrivial;
- target recomputation checked/pass/fail;
- hidden semantic counts;
- semantic_violation_unique_records;
- semantic_clean_records;
- reasoning leakage records/fields/mentions;
- distractor integrity;
- template_id integrity;
- exact cross-split duplicates;
- scene-level leakage status;
- test summary if practical.

## 21. Handoff files

Write full technical report to:

`handoff/FROM_DSH.md`

It may remain English.

Suggested sections:
1. Verdict
2. Scope / Safety Compliance
3. Files Created / Modified
4. Generator Changes
5. Semantic Policy Changes
6. v0.1.1 Counts
7. v0.1 vs v0.1.1 Comparison
8. Target Recomputation
9. Hidden Eligibility Audit
10. Reasoning Leakage Audit
11. Template ID Audit
12. Distractor Policy Audit
13. Statistics Consistency
14. Split / Duplicate Audit
15. Level-3 Trivial / Nontrivial
16. Visualization Pack
17. Test Execution Summary
18. Known Limitations
19. Git Status / Commit / Push
20. Recommended Next Task

Update `handoff/PROJECT_STATE.md` concisely. It may also remain English.

## 22. Git commit/push for ChatGPT review

ChatGPT will inspect the pushed repository directly.

Commit and push all new:
- source code;
- configs;
- tests;
- docs/ADR;
- v0.1.1 manifest/statistics;
- quality JSON;
- lightweight audit visualizations;
- `handoff/FROM_DSH.md`;
- `handoff/PROJECT_STATE.md`;
- this `handoff/TO_DSH.md` if currently modified/uncommitted.

Do NOT commit ignored full JSONL datasets, weights, checkpoints, or unrelated files.

Before staging:
1. `git status --short`
2. inspect changed files
3. confirm no unrelated/large generated binaries are staged
4. confirm no out-of-repo writes occurred.

Recommended commit:

`fix: regenerate BuildSpatialReason v0.1.1`

Then:

`git push origin main`

If push fails, record exact reason in `FROM_DSH.md` and explain it in Chinese in the DSH UI.

## 23. Final DSH web/chat response — Chinese only

At completion, the visible DSH response to the user must be concise Chinese and include only:
- whether v0.1.1 was generated;
- final validator verdict;
- final sample count;
- test summary;
- commit hash;
- push success/failure;
- any blocker requiring user action.

Do not paste the full English technical report into the UI.

## 24. Stop

After Task 5B, STOP.

Do not start:
- Task 5.5;
- MLLM integration;
- model training;
- Spatial Relation Encoder;
- Spatial Consistency Loss;
- any later task.

Wait for ChatGPT review of the pushed GitHub commit.
