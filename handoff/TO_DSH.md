# TO_DSH — MASK01_D1_SUCCESS_SEMANTICS_HARDENING

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-mask01-f2-locked-success-artifacts`
> Required starting HEAD: `a98ccecce20585dc37520523cd32b49b0a684248`
> New task branch: `fix/task8b3-mask01-success-semantics`

## 0. CHATGPT F2 AUDIT / FROZEN DECISION

Task:

```text
MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS
```

is accepted for the engineering decision:

```text
MASK01_F2 = ACCEPTED_FOR_DECISION
```

Git facts independently verified by ChatGPT:

```text
branch = audit/task8b3-mask01-f2-locked-success-artifacts
HEAD   = a98ccecce20585dc37520523cd32b49b0a684248
parent = e8f1315c9afdaac818126db6a2ac76b758027f70
commit = docs(rc1): characterize locked successful masks
diff   = only the four authorized F2 report/handoff files
```

Frozen evidence:

```text
A1 status = SUCCESS
A1 frozen semantic verdict = FAIL
A1 mask_area = 101

A3 status = SUCCESS
A3 frozen semantic verdict = FAIL
A3 mask_area = 1439

A4 status = SUCCESS
A4 frozen semantic verdict = FAIL
A4 mask_area = 359
```

All three saved final mask PNGs exist and their pixel counts match the stored `result.json.mask_area`.

Current normal-predict post-inference runtime gates remain:

```text
1. non-empty mask
2. mask_only_in_padding branch
3. direction-centroid hard constraint
```

Frozen padding interpretation:

```text
PADDING_GATE_CONCLUSION = PADDING_GATE_INEFFECTIVE
PADDING_LEAKAGE_ESTABLISHED = false
```

Frozen design decision:

```text
REJECT_THRESHOLD_BASED_MASK_VALIDITY_REPAIR
```

No mask area, fraction, connected-component, confidence, probability, logit, IoU, or other scalar threshold may be introduced from the three R4B cases.

Reason:

```text
NO_POSITIVE_SEMANTIC_SUCCESS_CONTROL_IN_R4B_SUCCESS_SET = true
```

A1/A4 being small does not justify an area cutoff; A3 is substantially larger and is still a frozen semantic FAIL.

The accepted repair direction is:

```text
SUCCESS_SEMANTICS_HARDENING_V1
```

This task implements that design exactly.

---

## 1. PURPOSE

The existing normal inference `SUCCESS` status is retained as an engineering/process status for compatibility.

It MUST no longer be presented as if it established semantic target correctness.

The product contract after this task must be:

```text
status = SUCCESS
```

means:

```text
the RC1 normal inference pipeline completed and passed its current runtime structural checks
```

and does NOT mean:

```text
the final mask has been verified to correspond to the user's intended semantic target
```

The machine-readable result and CLI must communicate this explicitly.

This is a status/semantic-contract hardening task.

It is NOT:

```text
a model repair
a segmentation-quality repair
a threshold repair
a reference repair
a detector repair
an architecture change
```

---

## 2. GIT PRE-FLIGHT

Before any write verify exactly:

```text
current branch =
audit/task8b3-mask01-f2-locked-success-artifacts

HEAD =
a98ccecce20585dc37520523cd32b49b0a684248
```

Allowed initial worktree:

```text
clean
```

or only:

```text
M handoff/TO_DSH.md
```

from the user replacing the task book.

Any other mutation:

```text
STOP
```

Do not:

```text
reset
rebase
amend
stash
clean
force-push
discard unknown work
```

After successful preflight create:

```text
fix/task8b3-mask01-success-semantics
```

directly from the required starting HEAD.

---

## 3. DESIGN — EXACT MACHINE-READABLE SUCCESS SEMANTICS

In canonical:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
```

define one stable, centralized normal-success semantics contract.

Use exactly these values:

```text
validity_scope = "RUNTIME_STRUCTURAL_ONLY"
semantic_status = "NOT_EVALUATED"
```

Use this exact English semantic note:

```text
SUCCESS means the RC1 runtime completed and passed its current structural checks; semantic target correctness is not established.
```

Preferred implementation:

```python
SUCCESS_VALIDITY_SCOPE = "RUNTIME_STRUCTURAL_ONLY"
SUCCESS_SEMANTIC_STATUS = "NOT_EVALUATED"
SUCCESS_SEMANTIC_NOTE = (
    "SUCCESS means the RC1 runtime completed and passed its current structural checks; "
    "semantic target correctness is not established."
)

def success_semantics() -> dict:
    return {
        "validity_scope": SUCCESS_VALIDITY_SCOPE,
        "semantic_status": SUCCESS_SEMANTIC_STATUS,
        "semantic_note": SUCCESS_SEMANTIC_NOTE,
    }
```

Equivalent formatting is allowed, but the keys and exact string values above are frozen.

For the NORMAL `predict_one()` success payload, before diagnostics/result finalization, the saved payload MUST contain:

```json
{
  "status": "SUCCESS",
  "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
  "semantic_status": "NOT_EVALUATED",
  "semantic_note": "SUCCESS means the RC1 runtime completed and passed its current structural checks; semantic target correctness is not established."
}
```

These fields must therefore be present in the normal-success `result.json`.

Do not add these fields to a FAILED result as if semantic validation had been performed.

Do not change error codes.

---

## 4. COMPATIBILITY — STATUS MUST NOT CHANGE

Do NOT replace:

```text
status = SUCCESS
```

with a new status enum/string.

Do NOT change:

```text
PipelineResult.ok
```

semantics.

Do NOT change successful exit code behavior.

The following compatibility contract is frozen:

```text
PipelineResult.status == "SUCCESS"
PipelineResult.ok == True
normal single-image successful process exit code == 0
batch success accounting still uses result.ok
```

This task hardens the meaning around SUCCESS without breaking existing machine status consumers.

---

## 5. INSPECT-PROPOSALS MODE

`--inspect-proposals` is not a final target segmentation result.

Do not reinterpret it as a semantic mask result.

For this task:

```text
inspect_proposals status behavior = UNCHANGED
```

Do not add normal target semantic fields to inspect mode.

No inspect-mode redesign.

---

## 6. CLI — SINGLE IMAGE

Modify canonical:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
```

For a successful NORMAL single-image inference, retain the existing first status line:

```text
Result       : SUCCESS
```

Immediately after it print exactly:

```text
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
```

Also print exactly this user-facing note:

```text
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

Then retain the existing mask / overlay / diagnostics / reference / mask-area output.

Do not print this normal-target note for `--inspect-proposals`.

Do not change failure reporting.

---

## 7. CLI — BATCH MODE

For each successful batch item, replace the bare presentation:

```text
... SUCCESS
```

with:

```text
... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

Do not change the actual `result.ok` logic.

At batch summary, change the label:

```text
Success:
```

to:

```text
Runtime success:
```

Keep:

```text
Failed:
Total:
```

and exit-code logic unchanged.

This is presentation semantics only.

---

## 8. ABSOLUTELY NO NEW VALIDITY HEURISTIC

Do NOT add:

```text
min_mask_pixels
min_mask_frac
mask_area cutoff
relative-area cutoff
connected-component cutoff
bbox-fill cutoff
confidence cutoff
probability cutoff
logit cutoff
IoU cutoff
field-mass cutoff
weighted quality score
learned quality head
proposal-overlap gate
target-proposal matching gate
```

Do NOT change:

```text
empty_target_mask
mask_only_in_padding
direction_constraint_violated
```

behavior in this task.

Do NOT remove `_non_padding_mask()` or the `mask_only_in_padding` branch in this task.

The padding redundancy is already documented and is not the scope of D1.

---

## 9. NO ALGORITHM / MODEL CHANGES

Do not modify:

```text
detector.py
context.py
imageio.py
core.py
ProgramHead/Qwen
SAM/SAM2
D-B1
GRF/relation fields
reference selection
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
proposal configuration
model weights
configs/inference.yaml thresholds/settings
```

Do not execute inference or training.

---

## 10. DOCUMENTATION HARDENING

Update these canonical delivery documents only as needed to make the normal-success semantics explicit:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
```

The docs must consistently state:

```text
SUCCESS = runtime structural success
semantic target correctness = NOT_EVALUATED by RC1 runtime
```

They must NOT state or imply that:

```text
SUCCESS proves the requested building was correctly identified/segmented
```

Do not rewrite unrelated sections.

Do not change scientific metrics.

Do not change final-test interpretation.

---

## 11. TESTS — REQUIRED

Authorized test files:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

Add focused CPU-only/no-model tests that prove the frozen contract.

At minimum test:

### A. centralized semantics contract

Verify exact machine values:

```text
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
exact semantic_note string
```

### B. normal-success payload contract

Without model inference, unit-test the pure semantics helper/constant path used by normal success.

If direct `predict_one()` construction would require model runtimes, do NOT mock the full model chain merely for this test. Test the centralized helper/contract directly and verify source call/use in the smallest deterministic way.

### C. single-result CLI presentation

Using a synthetic/fake successful result object and direct call of the reporting helper, verify output contains exactly:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

No model loading.

### D. failure presentation does not claim semantic success

Synthetic failed result must not print the success semantics block.

### E. inspect mode contract remains unchanged

Existing inspect behavior must not acquire the normal-target semantic fields/output.

### F. compatibility

Verify:

```text
PipelineResult(status="SUCCESS", ...).ok is True
```

No new success enum is introduced.

Tests must not encode A1/A3/A4 mask areas as acceptance thresholds.

---

## 12. TEST EXECUTION — CANONICAL ONLY

After implementation, from canonical delivery root run targeted tests:

```text
python -m pytest tests/test_task8b_runtime.py tests/test_cli_contract.py -q
```

Then run the full canonical delivery suite:

```text
python -m pytest tests/ -q
```

No root/research test suite is required.

No model inference.

No training.

If a test fails because of this task's changes, fix only within authorized scope.

If an unrelated pre-existing failure appears, STOP and report; do not broaden scope.

---

## 13. SOURCE MANIFEST — GIT CANONICAL BYTES ONLY

Because canonical delivery files change, update:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Identity basis MUST remain:

```text
GIT_CANONICAL_BLOB_BYTES
```

Do not use Windows working-tree byte counts/hashes as canonical identity.

Required workflow:

1. finish code/docs/tests edits;
2. run tests;
3. stage all changed files under:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

except `source_manifest.json`;
4. compute bytes + SHA256 for the manifest-listed changed canonical paths from the GIT INDEX / canonical staged bytes, not working-tree CRLF-expanded bytes;
5. update only the corresponding manifest entries;
6. stage `source_manifest.json`;
7. perform an all-entry staged-manifest validation.

For staged canonical identity, a valid basis is equivalent to:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/<relative path>
```

The all-entry validation must prove:

```text
manifest entries = unchanged file-count policy
every manifest path exists in staged/index canonical state
every bytes value matches canonical staged blob bytes
every sha256 matches canonical staged blob bytes
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Do not add runtime output files to the manifest.

Do not add model weights.

Do not change the copy policy.

Record the final manifest file count and validation result.

---

## 14. EXTERNAL RC1 — NO WRITE IN D1

This task must NOT synchronize or modify:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

After the new canonical commit is created, a read-only external comparison is allowed:

```text
python scripts/sync_advisor_rc1_delivery.py \
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 \
  --check
```

Expected outcome is that changed manifest-listed files may be reported as mismatched.

This read-only comparison is optional but preferred.

Do NOT run sync without `--check`.

External controlled sync belongs to the next gate.

---

## 15. REQUIRED ENGINEERING REPORTS

Create:

```text
docs/task8b3_mask01_d1_success_semantics_hardening.md
evaluation/task8b3_mask01_d1_success_semantics_hardening.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report at minimum:

```text
task
status
starting branch/head
task branch

design =
SUCCESS_SEMANTICS_HARDENING_V1

threshold_based_repair =
REJECTED

product files changed
test files changed
docs changed

normal success status changed =
NO

PipelineResult.ok changed =
NO

exit-code semantics changed =
NO

machine fields:
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
semantic_note = exact frozen string

single CLI wording
batch CLI wording

targeted tests
full canonical delivery suite

manifest identity basis
manifest file count
manifest all-entry validation

external write =
NO

inference executed =
NO

training executed =
NO
```

The JSON must structurally contain the same facts.

Use:

```text
repair_decision =
IMPLEMENTED_PENDING_EXTERNAL_SYNC

next_gate =
MASK01_D2_EXTERNAL_SYNC_AND_REGRESSION
```

Task status exactly one of:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

---

## 16. ALLOWED REPOSITORY CHANGES — EXACT SET

Only these paths may change:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py

delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md

delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py

delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json

docs/task8b3_mask01_d1_success_semantics_hardening.md
evaluation/task8b3_mask01_d1_success_semantics_hardening.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

A listed documentation path does not need to change if no edit is required, but no path outside this set may change.

No dependency files may change.

---

## 17. DIFF GATE

Before commit run:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Every changed/staged path must be in the exact allowlist above.

Any unexpected path:

```text
STOP
```

Do not clean/discard it.

---

## 18. COMMIT / PUSH

For `COMPLETE` or `COMPLETE_WITH_EVIDENCE_GAPS`:

Commit exactly once:

```text
git commit -m "fix(rc1): harden runtime success semantics"
```

Push exactly:

```text
fix/task8b3-mask01-success-semantics
```

No force push.

The committed reports must use:

```text
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```

Do not self-reference the commit SHA.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Verify local/remote HEAD equal and worktree clean.

Then STOP.

Do not execute D2.

---

## 19. ABSOLUTE PROHIBITIONS

Do NOT:

```text
change status SUCCESS to another enum/string
change PipelineResult.ok
change success exit codes
add mask thresholds
add quality thresholds
add connected-component gates
add probability/logit gates
add confidence gates
change current post-inference gates
fix/remove padding gate
modify detector/reference/model architecture
run inference
run training
rerun A1/A3/A4
rerun fixed Demo
write external RC1
sync external RC1
modify weights
modify generated inference outputs
modify dependencies
merge to main
reset
rebase
amend
stash
clean
force push
execute D2
```

---

## 20. SUCCESS DEFINITION

D1 succeeds only if canonical RC1 now makes this distinction explicit and test-protected:

```text
PROCESS/RUNTIME:
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY

SEMANTIC:
semantic_status = NOT_EVALUATED
semantic target correctness is not established
```

while preserving all existing algorithm/model behavior.

Required terminal state:

```text
THRESHOLD REPAIR = REJECTED
SEMANTICS HARDENING = IMPLEMENTED
ALGORITHM CHANGE = NONE
MASK CHANGE = NONE
MODEL CHANGE = NONE
EXTERNAL RC1 WRITE = NONE
NEXT = MASK01_D2_EXTERNAL_SYNC_AND_REGRESSION
```

Then STOP.
