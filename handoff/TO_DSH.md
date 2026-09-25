# TO_DSH — Task 5C: Acceptance Audit Hardening & Artifact Consistency

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Goal: harden the acceptance audit for **BuildSpatialReason-v0.1.1**, fix stale/inconsistent repository artifacts, and establish a genuinely independent semantic oracle before Task 5.5.
>
> Important: **Do not regenerate the v0.1.1 JSONL dataset unless the independent oracle finds a real data defect.**

## 0. User-facing language

Only the DSH web/chat output shown to the user must be in **Chinese**:
- progress summaries;
- warnings;
- permission explanations;
- blocker explanations;
- final summary.

Commands, paths, code identifiers, raw logs, and field names may remain English.

`handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` may remain English.

## 1. Full-access boundary

DSH is still running with Full Access only because the normal workspace-write sandbox cannot start correctly on this Windows host.

This does **not** expand the task scope.

Allowed writes:
- only inside `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\`
- ordinary runtime temp directories when technically required.

Allowed external reads:
- `..\WHU_Building_Segment\` may be read only when required for frozen legacy/source verification.

Allowed Git/network operations:
- `git status`
- `git diff`
- `git log`
- `git add`
- `git commit`
- `git fetch`
- `git pull`
- `git push origin main`

Forbidden:
- modifying files outside `BuildReasonSeg`;
- modifying `../WHU_Building_Segment/`;
- installing packages;
- downloading models, datasets, scripts, checkpoints, or external code;
- changing system settings, registry, PATH, shell profiles;
- deleting unrelated files;
- starting model training;
- starting Task 5.5;
- starting any later MLLM/model task.

If any required action conflicts with these limits, stop that action and explain the blocker in Chinese in the DSH UI.

## 2. Dataset freeze

Treat both dataset versions as frozen historical artifacts:

- `datasets/build_spatial_reason/v0.1/`
- `datasets/build_spatial_reason/v0.1.1/`

Do **not** modify or regenerate any JSONL file in either directory during this task.

Only if the new independent semantic oracle finds a real defect in v0.1.1 may you stop and report that a regeneration task is required.

Do not silently repair data in Task 5C.

## 3. Known inconsistencies to fix

The previous Task 5B commit produced a valid-looking v0.1.1 dataset, but several repository artifacts are inconsistent.

### 3.1 Stale counts in handoff/state docs

Machine-readable sources currently indicate the authoritative v0.1.1 values:

- total = 25,229
- Level 2 total = 5,036
- Level-2 Type A = **2,275**
- Level-2 Type B = **2,761**
- Level 3 total = 2,918
- Level-3 trivial = **1,256**
- Level-3 nontrivial = **1,662**

However, `handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md` still contain stale values such as:

- Type A 2,272 / Type B 2,764
- trivial 1,261 / nontrivial 1,657

Fix all stale references so repository documents agree with authoritative machine-readable artifacts.

### 3.2 Output path mismatch

The validator currently converts:

`v0.1.1 -> v011`

and writes:

- `evaluation/build_spatial_reason_v011_quality.json`
- `evaluation/build_spatial_reason_v011_samples/`

while docs refer to:

- `evaluation/build_spatial_reason_v0.1.1_quality.json`
- `evaluation/build_spatial_reason_v0.1.1_samples/`

Unify this.

Preferred canonical naming:

- `evaluation/build_spatial_reason_v0.1.1_quality.json`
- `evaluation/build_spatial_reason_v0.1.1_samples/`

Do not leave duplicate stale artifacts unless clearly marked deprecated and intentionally preserved.

### 3.3 Missing semantic policy provenance

`spatial_reasoning/semantic_policy.py` is now a critical generator/validator dependency but is not included in `manifest.json -> generator_file_sha256`.

Add it to the provenance hash set.

If other newly critical source files also materially affect generation and are missing, document and include them if appropriate.

### 3.4 Stale wording in manifest/docs

Update v0.1-era wording that is now inaccurate, including examples such as:
- template selection being “implicitly recorded” when `template_id` is now explicit;
- references to “Current v0.1” inside v0.1.1 limitations/docs;
- any other version-specific wording that is clearly stale.

Do not rewrite documents broadly; keep the patch narrow.

## 4. Main hardening requirement: independent semantic oracle

This is the core of Task 5C.

The current generator and validator both depend on:

`spatial_reasoning/semantic_policy.py`

That is useful for consistency, but not sufficiently independent for a final acceptance proof.

Create a second implementation used **only for audit/testing** that does **not import or call**:

- `semantic_policy.py`;
- generator semantic helper functions that internally call it.

Suggested file:

`spatial_reasoning/semantic_oracle.py`

or:

`evaluation/semantic_oracle.py`

The name is flexible, but the separation must be clear.

### Oracle requirements

The oracle must independently implement the semantics directly from:
- source component geometry;
- component quality flags;
- frozen relation thresholds/config;
- direct geometry functions;
- frozen direction predicate definition.

It must independently resolve:
1. `largest`
2. `smallest`
3. reference -> nearest
4. reference -> direction
5. reference -> direction -> nearest

### Critical independence rule

The oracle may reuse low-level neutral primitives such as:
- component metadata structures;
- `geometry.py` distance/area helpers;
- threshold config loading;
- component quality classification;
- low-level direction predicate evaluation if it is already frozen and not part of semantic acceptance policy.

But it must **not** call:
- `SP.resolve_size_extreme`
- `SP.resolve_nearest`
- `SP.direction_candidates_over_visible`
- `SP.admissible_nearest_ids`
- any wrapper that simply delegates to these.

The point is to have two implementations:
- production semantic policy;
- independent audit oracle.

## 5. Full-dataset oracle audit

Run the independent oracle over **all 25,229 v0.1.1 records**.

For every record, recompute:
- semantic reference;
- direction candidate set where applicable;
- semantic nearest where applicable;
- final target;
- ambiguity/admissibility decision.

Compare against the stored record.

Report:
- total checked;
- exact target matches;
- semantic-program mismatches;
- candidate-set mismatches;
- reference mismatches;
- ambiguity-policy mismatches.

Acceptance target:
- **25,229 / 25,229 exact semantic target match**
- **0 semantic mismatches**

If any mismatch is found:
- verdict must be `FAIL_REQUIRES_REVISION`;
- do not modify JSONL;
- stop before Task 5.5;
- list exact sample IDs and root cause.

## 6. Independent hidden-eligibility audit

Using the independent oracle, re-check that no stored v0.1.1 sample violates visible-component semantics.

Machine-readable counts must be zero:
- `hidden_eligibility_largest`
- `hidden_eligibility_smallest`
- `hidden_eligibility_nearest`
- `hidden_eligibility_level3_nearest`
- `semantic_violation_unique_records`

Do not derive these by calling the production semantic policy.

## 7. Repository consistency gate

Add a dedicated consistency checker/test that prevents future drift between:
- `manifest.json`
- `statistics.json`
- quality JSON
- `handoff/PROJECT_STATE.md`
- `handoff/FROM_DSH.md`
- version docs
- expected evaluation artifact paths

Preferred approach:
- create `scripts/check_artifact_consistency.py`
- add corresponding tests.

At minimum verify machine-readable consistency for:
- total count;
- split counts;
- level counts;
- Level-2 Type A / Type B / total;
- Level-3 trivial / nontrivial / total;
- dataset version;
- semantic policy version;
- generator version;
- evaluation quality path;
- evaluation sample-pack path.

For Markdown/handoff docs, avoid fragile full-text parsing if unnecessary. It is acceptable to verify key declared values and canonical paths through a small structured block or clearly delimited fields.

If useful, add a short machine-readable summary file used by docs/handoff generation rather than hard-coding counts repeatedly.

## 8. Canonical artifact paths

After Task 5C, use exactly:

- `evaluation/build_spatial_reason_v0.1.1_quality.json`
- `evaluation/build_spatial_reason_v0.1.1_samples/`
- `docs/build_spatial_reason_v0.1.1_quality_audit.md`

Update all references accordingly.

Remove or rename the old `v011` artifacts in the repository if safe.

Do not keep two competing quality JSONs that appear equally authoritative.

## 9. Provenance hardening

Update:

`datasets/build_spatial_reason/v0.1.1/manifest.json`

so `generator_file_sha256` includes:

- `spatial_reasoning/semantic_policy.py`

Also review whether the following materially affect generation and should be included if not already:
- generator entrypoint;
- annotator;
- templates;
- relations;
- geometry;
- thresholds;
- component quality.

Do not over-expand provenance to unrelated files.

Then verify hashes match current repository contents.

## 10. Test hardening

Add tests for at least:
1. oracle never imports/calls `semantic_policy`;
2. oracle largest matches direct area argmax semantics;
3. oracle smallest matches direct area argmin semantics;
4. oracle nearest uses all visible candidates before eligibility;
5. oracle Level-3 nearest uses full direction-valid candidate set;
6. full-dataset oracle target match;
7. artifact consistency checker;
8. canonical evaluation path naming;
9. manifest provenance includes `semantic_policy.py`;
10. v0.1 and v0.1.1 JSONL hashes remain unchanged.

Run the full existing suite plus the new tests.

Final report must include:
- exact test commands;
- total tests/checks;
- passed;
- failed;
- skipped;
- exit codes.

## 11. v0.1.1 frozen integrity

Record hashes for:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

before Task 5C changes.

Verify the exact same hashes after Task 5C.

Also verify v0.1 frozen hashes remain unchanged.

If any JSONL hash changes, this task fails unless the user explicitly authorized regeneration, which they have not.

## 12. Quality JSON update

Canonical file:

`evaluation/build_spatial_reason_v0.1.1_quality.json`

Add or preserve:
- verdict;
- total/split/level/query_type counts;
- Level-2 Type A / Type B;
- Level-3 trivial / nontrivial;
- production-policy recomputation result;
- independent-oracle recomputation result;
- hidden semantic counts from independent oracle;
- reasoning leakage counts;
- template_id integrity;
- distractor integrity;
- exact cross-split duplicate count;
- v0.1 integrity;
- v0.1.1 JSONL integrity;
- scene-level leakage status;
- consistency-check result.

Recommended new fields:

```text
independent_oracle:
  checked
  target_match
  target_mismatch
  candidate_set_mismatch
  reference_mismatch
  semantic_violation_unique_records
```

## 13. Verdict

Task 5C final verdict must be exactly one of:
- `PASS`
- `PASS_WITH_WARNINGS`
- `FAIL_REQUIRES_REVISION`

Use `FAIL_REQUIRES_REVISION` if:
- any independent-oracle target mismatch exists;
- any independent hidden semantic violation exists;
- any frozen JSONL hash changes;
- critical artifact counts remain inconsistent;
- critical provenance hashes are missing/mismatched.

A documentation-only warning may result in `PASS_WITH_WARNINGS`, but all machine facts must agree before approval.

## 14. Handoff files

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

These may remain English.

They must use the final authoritative values from machine-readable artifacts.

Do not manually preserve stale numbers.

Recommended final sections in `FROM_DSH.md`:
1. Verdict
2. Files Modified
3. Dataset Freeze Verification
4. Canonical Artifact Paths
5. Independent Oracle Design
6. Full-Dataset Oracle Audit
7. Hidden Semantic Audit
8. Artifact Consistency Audit
9. Provenance Audit
10. Test Summary
11. Remaining Limitations
12. Git Commit / Push
13. Ready for Task 5.5?

## 15. Git commit and push

ChatGPT will review the pushed commit directly.

Before staging:
1. `git status --short`
2. inspect all changed files;
3. ensure no JSONL dataset file is modified/staged;
4. ensure no weights/checkpoints/large unrelated files are staged;
5. confirm no out-of-repo writes occurred.

Recommended commit:

`audit: harden BuildSpatialReason v0.1.1 acceptance`

Push:

`git push origin main`

If push fails, report it exactly.

## 16. Final DSH web/chat response — Chinese only

The final visible DSH response should be concise Chinese and include:
- Task 5C verdict;
- independent oracle checked/matched count;
- whether v0.1.1 JSONL hashes stayed unchanged;
- test summary;
- commit hash;
- push success/failure;
- any blocker requiring user action.

Do not paste the full English handoff report into the UI.

## 17. Stop

After Task 5C:

**STOP.**

Do not begin:
- Task 5.5;
- external model/dataset research;
- MLLM integration;
- training;
- architecture implementation;
- any later task.

Wait for ChatGPT to review the pushed GitHub commit.
