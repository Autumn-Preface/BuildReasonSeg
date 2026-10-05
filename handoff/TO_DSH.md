# TO_DSH — MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-mask01-f1-r1-corrective-forensics`
> Required starting HEAD: `e8f1315c9afdaac818126db6a2ac76b758027f70`
> New task branch: `audit/task8b3-mask01-f2-locked-success-artifacts`

## 0. CHATGPT AUDIT DISPOSITION

Task `MASK01_F1_R1_CORRECTIVE_FORENSICS` is PARTIALLY ACCEPTED.

Freeze as accepted:

```text
SOURCE_SUCCESS_CONTRACT = ACCEPTED
PADDING_FORENSICS = ACCEPTED

POST_INFERENCE_GATES = 3
1. empty_target_mask
2. mask_only_in_padding
3. direction_constraint_violated

PADDING_GATE_CONCLUSION = PADDING_GATE_INEFFECTIVE
PADDING_LEAKAGE_ESTABLISHED = false

min_mask_pixels = NOT_IMPLEMENTED
min_mask_frac   = NOT_IMPLEMENTED
```

ChatGPT independently verified that:

```text
context_to_global()
```

crops context padding before the original-image `mask_full` is produced, while:

```text
_non_padding_mask()
```

returns an all-True original-image mask. Therefore `mask_only_in_padding` is not an effective independent gate
after the preceding empty-mask check. This does NOT establish successful padding leakage.

Not accepted from F1-R1:

```text
HISTORICAL_FINAL_MASK_EVIDENCE
FAILURE_TAXONOMY
OBSERVABLE_SIGNAL_INVENTORY
```

The previous report did not inspect the actual R4B final-mask files and left the taxonomy effectively empty.

This task closes only that remaining evidence gap.

NO product repair is authorized.

---

## 1. GIT PRE-FLIGHT

Before any write verify exactly:

```text
branch = audit/task8b3-mask01-f1-r1-corrective-forensics
HEAD   = e8f1315c9afdaac818126db6a2ac76b758027f70
```

Allowed initial worktree state:

```text
clean
```

or only the user replacement of:

```text
handoff/TO_DSH.md
```

No other change is allowed.

If incompatible:

```text
STOP
```

Do not reset, rebase, amend, stash, clean, force-push, or discard unknown work.

After preflight create:

```text
audit/task8b3-mask01-f2-locked-success-artifacts
```

directly from the required starting HEAD.

---

## 2. TASK PURPOSE

Read-only forensic characterization of the three frozen Task 8B.3-R4B full-pipeline runtime-success cases:

```text
A1
A3
A4
```

The questions are:

```text
What final-mask material was actually saved?
What exact descriptive properties do those masks have?
Which current runtime-observable signals distinguish runtime validity from semantic correctness?
What evidence exists, and what evidence is still missing, before ChatGPT can decide a MASK-01 repair?
```

Do NOT design or implement a repair.

---

## 3. FROZEN HISTORICAL FACTS

Use these versioned files as the authoritative provenance sources:

```text
docs/task8b3_six_image_demo_suite.md
docs/task8b3_d1_demo_failure_forensics.md
```

Frozen R4B runtime results:

```text
A1 = SUCCESS, reference_id=48, mask_area=101
A3 = SUCCESS, reference_id=1,  mask_area=1439
A4 = SUCCESS, reference_id=30, mask_area=359
```

Frozen visual/manual semantic audit:

```text
A1 = semantic FAIL
A3 = semantic FAIL
A4 = semantic FAIL
```

Do NOT reinterpret or replace those verdicts.

D1 additionally characterizes A1/A4 as tiny/incomplete target fragments under the frozen visual audit.

No GT identity is to be invented.

---

## 4. EXACT EXTERNAL ARTIFACTS

Inspect these exact paths READ ONLY.

### A1

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\masks\A1_mask_001.png
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\overlays\A1_overlay_001.png
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\A1_001
```

### A3

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\masks\A3_mask.png
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\overlays\A3_overlay.png
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\A3
```

### A4

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\masks\A4_mask.png
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\overlays\A4_overlay.png
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\A4
```

For each diagnostics directory:

```text
list immediate children only
```

Then inspect only these existing files if present:

```text
result.json
maps.npz
```

Do not inspect unrelated diagnostics directories.

Do not recursively search `inference/output`.

Do not modify any external file.

---

## 5. ARTIFACT IDENTITY

For each exact mask, overlay, result.json and maps.npz that exists, record:

```text
path
exists
bytes
sha256
```

For diagnostics directories record immediate child filenames only.

If an expected file is missing, record:

```text
MISSING
```

and continue.

Do not regenerate it.

---

## 6. FINAL MASK EXACT CHARACTERIZATION

For each existing final mask PNG:

1. load it read-only;
2. record image shape and dtype;
3. record unique pixel values;
4. define foreground exactly as:

```text
pixel_value != 0
```

5. compute only descriptive, threshold-free statistics:

```text
foreground_pixel_count
foreground_fraction_of_full_image
foreground_bbox = [top, left, bottom, right]
foreground_bbox_height
foreground_bbox_width
foreground_bbox_area
bbox_fill_ratio = foreground_pixel_count / foreground_bbox_area
centroid_row
centroid_col
touches_image_border
connected_component_count using 8-connectivity
connected_component_areas sorted descending
largest_component_area
largest_component_fraction_of_mask
```

These are forensic measurements only.

Do NOT propose cutoffs.

Do NOT classify a mask as valid/invalid from any newly computed numeric value.

Verify:

```text
foreground_pixel_count == result.json.mask_area
```

when both values exist.

Verify computed centroid against `result.json.target_centroid` within exact/rounding tolerance appropriate to the stored float.

Record MATCH/MISMATCH.

No morphology operation may modify the mask.

---

## 7. RESULT.JSON FACTS

For each A1/A3/A4 `result.json`, record exactly if present:

```text
status
error_code
reason
prompt
parsed.program
language_mode
reference_mode
reference_id
effective_reference_id
reference_override
reference_area
reference_confidence
reference_bbox
direction
target_centroid
mask_area
context_padding
directional_guard
raw_proposal_count
merged_proposal_count
field_mass
output_paths
timings
```

If absent:

```text
NOT_PRESENT
```

Do not infer absent values.

Explicitly verify that each case reached:

```text
status = SUCCESS
```

and had no runtime failure reason.

---

## 8. MAPS.NPZ — READ-ONLY INVENTORY

If `maps.npz` exists for a case:

record:

```text
keys
shape per key
dtype per key
finite/nonfinite counts
min
max
mean
```

for numeric arrays.

If keys include likely decoder confidence/probability/logit material, report those statistics descriptively.

Do NOT:

```text
fit thresholds
search cutoffs
compare candidate thresholds
alter masks
reconstruct a new mask
```

The purpose is only to know what evidence is currently persisted.

---

## 9. FROZEN SEMANTIC LABEL LINKAGE

Link each artifact to the frozen D1/manual verdict:

```text
A1 -> TAX_SEMANTIC_TARGET_MISMATCH
A3 -> TAX_SEMANTIC_TARGET_MISMATCH
A4 -> TAX_SEMANTIC_TARGET_MISMATCH
```

Additionally, because D1 explicitly characterizes A1 and A4 as tiny/incomplete target fragments:

```text
A1 -> TAX_RUNTIME_VALID_QUALITY_POOR
A4 -> TAX_RUNTIME_VALID_QUALITY_POOR
```

For A3:

```text
TAX_RUNTIME_VALID_QUALITY_POOR
```

must NOT be assigned unless a versioned source explicitly supports that same characterization.

Do not invent GT.

Do not convert the frozen visual verdict into IoU or other numeric truth.

---

## 10. FAILURE TAXONOMY

Use only:

```text
TAX_RUNTIME_STRUCTURAL
TAX_CONTEXT_PADDING_COORDINATE
TAX_UPSTREAM_REFERENCE_RELATION
TAX_SEMANTIC_TARGET_MISMATCH
TAX_RUNTIME_VALID_QUALITY_POOR
TAX_UNKNOWN_EVIDENCE_GAP
```

For each A1/A3/A4 produce a structured record:

```text
case
runtime_status
semantic_verdict
taxonomy
supporting_versioned_source
supporting_saved_artifact
established_facts
not_established
```

Important distinction:

- historical D1 says A1/A3/A4 had reference semantic/proposal-quality issues in the R4B run;
- later REF-01 work used a different locked supported-domain four-case set and must not be retroactively substituted for A1/A3/A4;
- do not claim later REF repairs would have fixed these R4B masks without evidence.

---

## 11. OBSERVABLE-SIGNAL INVENTORY

Create a complete inventory for future ChatGPT design.

For each signal record:

```text
signal
current_runtime_source_or_artifact
availability
present_in_A1
present_in_A3
present_in_A4
raw_value_or_summary
can_establish_runtime_validity
can_establish_semantic_target_correctness
notes
```

Availability must be exactly one of:

```text
RUNTIME_ALWAYS_AVAILABLE
RUNTIME_CONDITIONALLY_AVAILABLE
PERSISTED_ARTIFACT_AVAILABLE
REQUIRES_GT_OR_MANUAL_REVIEW
NOT_CURRENTLY_AVAILABLE
```

At minimum classify:

```text
mask_full nonempty
mask_area
mask fraction
mask bbox
mask bbox fill ratio
mask connected components
largest component fraction
target centroid
direction hard-check result
reference_mode
reference_id
reference_area
reference_confidence
reference_bbox
direction
context_padding
directional_guard
raw proposal count
merged proposal count
field_mass
saved decoder probability/logit maps
GT IoU
manual semantic correctness
```

Rules:

- `mask_area`, target centroid, direction, reference fields that are in `result.json` are persisted runtime facts.
- connected-component metrics are derivable from `mask_full`, but if they are not currently computed by the product, mark the signal itself `NOT_CURRENTLY_AVAILABLE` and note `DERIVABLE_FROM_RUNTIME_MASK`.
- manual semantic correctness is `REQUIRES_GT_OR_MANUAL_REVIEW`.
- do not claim any runtime signal establishes semantic correctness unless evidence truly supports it.

---

## 12. CROSS-CASE TABLE

Create one compact A1/A3/A4 table containing at least:

```text
case
status
semantic_verdict
image_size
mask_area
mask_fraction
bbox_size
bbox_fill_ratio
component_count
largest_component_fraction
target_centroid
direction
reference_id
reference_area
reference_confidence
context_padding.applied
raw_proposals
merged_proposals
```

Missing fields must be `NOT_PRESENT`.

This table is descriptive only.

---

## 13. NO THRESHOLD / NO REPAIR SEARCH

Absolutely forbidden:

```text
best threshold
candidate threshold
area cutoff
relative-area cutoff
component cutoff
probability cutoff
confidence cutoff
weighted score
heuristic search
rule fitting
case-specific repair
```

There are only three frozen runtime-success cases and all three have a frozen semantic FAIL verdict.

This task MUST explicitly state:

```text
NO_POSITIVE_SEMANTIC_SUCCESS_CONTROL_IN_R4B_SUCCESS_SET = true
```

Therefore these three cases alone cannot justify a discriminative numeric validity threshold.

---

## 14. REQUIRED INTERPRETATION QUESTIONS

The report must answer factually:

1. Do A1/A3/A4 final mask PNGs actually exist?
2. Do their saved pixel counts match R4B/D1 `mask_area`?
3. Are the final masks structurally non-empty and direction-valid according to the stored runtime result?
4. Which mask morphology/statistics are currently available or derivable without GT?
5. Which evidence of semantic failure comes only from the frozen manual/visual audit?
6. Is there any positive semantic-success R4B runtime-success case available as a control?
7. Can the current three-case evidence justify a numeric threshold that separates good from bad masks?

For question 7, unless contrary evidence exists in the exact authorized sources, expected factual answer:

```text
NO
```

because all three R4B runtime-success samples are frozen semantic FAIL cases and no positive control is present.

DSH must not choose the next repair.

---

## 15. REQUIRED OUTPUTS

Create exactly:

```text
docs/task8b3_mask01_f2_locked_success_artifacts.md
evaluation/task8b3_mask01_f2_locked_success_artifacts.json
```

Modify exactly:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other repository path may change.

No external path may be written.

---

## 16. REQUIRED TERMINAL FIELDS

The evidence JSON and FROM_DSH summary must contain:

```text
task_id = MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS

source_contract_reaudit = NOT_PERFORMED
padding_reaudit = NOT_PERFORMED

historical_cases = [A1, A3, A4]

full_final_mask_evidence =
FOUND
or
PARTIAL
or
NOT_FOUND

no_positive_semantic_success_control_in_r4b_success_set = true

numeric_threshold_design_justified =
false

product_source_modified = false
tests_modified = false
inference_executed = false
training_executed = false
external_write_performed = false

repair_decision = DEFER_TO_CHATGPT
next_gate = CHATGPT_MASK01_F2_DESIGN_REVIEW
```

Task status must be exactly one of:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

---

## 17. COMMIT SHA RULE

Inside files committed in this task record:

```text
starting_head = e8f1315c9afdaac818126db6a2ac76b758027f70
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```

Do not attempt self-referential final-SHA insertion.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

ChatGPT will independently audit GitHub.

Do not create a second commit.

Do not amend.

---

## 18. DIFF / COMMIT / PUSH

Before staging verify only the four authorized repository paths changed.

For `COMPLETE` or `COMPLETE_WITH_EVIDENCE_GAPS`, stage exactly:

```text
docs/task8b3_mask01_f2_locked_success_artifacts.md
evaluation/task8b3_mask01_f2_locked_success_artifacts.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Commit exactly once:

```text
git commit -m "docs(rc1): characterize locked successful masks"
```

Push:

```text
audit/task8b3-mask01-f2-locked-success-artifacts
```

No force push.

Verify remote/local HEAD match and worktree is clean.

Then STOP.

---

## 19. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify product source
modify tests
modify configs
modify manifests
run predict.py
run detector
run Qwen
run SAM/SAM2
run D-B1
run model inference
run training
regenerate proposals
regenerate masks
edit external delivery
rerun Demo
use Assisted Mode
tune thresholds
design a repair
implement a repair
enter Task 8B.4
reset
rebase
amend
stash
clean
force push
```

---

## 20. SUCCESS DEFINITION

This task is complete only when ChatGPT has trustworthy saved-mask evidence for A1/A3/A4 and can decide whether:

```text
MASK-01 should receive an automatic runtime validity repair,
a status/semantic-contract repair,
or no threshold-based repair at all.
```

DSH must not make that decision.

After commit + push:

```text
STOP
```
