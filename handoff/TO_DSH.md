请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0 — Locked Replay Saved-Artifact Audit**

# TO_DSH — Task 8B.3-REF01-E3C0: Locked Replay Saved-Artifact Audit

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required base branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required base HEAD: `1197d860b8b1a3503c6cba730eda33ccbff4e1c5`
> New audit branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. CHATGPT DECISION

Task 8B.3-REF01-E3B2 is formally CLOSED.

The next technical gate is:

```text
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

However, ChatGPT has NOT yet established that the exact saved proposal artifacts needed for a zero-detector replay are still present and sufficiently complete.

Therefore E3C0 is a **read-only saved-artifact audit**.

It MUST NOT execute the replay itself.
It MUST NOT rerun detector/model inference.
It MUST NOT regenerate proposals.

The sole purpose is to answer:

```text
Do exact saved artifacts exist that allow the four locked cases
(right/left/above/below)
to be replayed deterministically without any detector/model call?
```

# 1. FROZEN LOCKED CASES / HISTORICAL FACTS

The only four cases in scope are:

```text
right:
tile = 1010
historical raw / merged / eligible = 6 / 6 / 4
historical production largest selection = proposal 1
historical classification = REFERENCE_SELECTED_CORRECT

left:
tile = 1003
historical raw / merged / eligible = 66 / 53 / 42
historical production largest selection = proposal 14
historical best-covered reference proposal = 30
historical classification = REFERENCE_SELECTION_WRONG_COVERED

above:
tile = 1008
historical raw / merged / eligible = 9 / 9 / 4
historical production largest selection = proposal 4
historical best-covered reference proposal = 5
proposal 5:
  mask_area = 5059
  confidence = 0.7026934027671814
  touches_image_border = false
  bbox_extent_ratio = 0.296875
  IoU_to_GT_reference = 0.9032501889644747
proposal 3:
  mask_area = 11474
  touches_image_border = true
  bbox_extent_ratio = 0.39453125
historical classification = REFERENCE_ELIGIBILITY_BLOCKED

below:
tile = 1009
historical raw / merged / eligible = 7 / 6 / 3
historical production largest selection = proposal 1
historical best-covered reference proposal = 2
historical classification = REFERENCE_SELECTION_WRONG_COVERED
```

Frozen repair design:

```text
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
```

Locked expected post-repair reference selection, for a later task only:

```text
right: 1 -> 1
left: 14 -> 14
above: 4 -> 5
below: 1 -> 1
```

E3C0 MUST NOT claim these post-repair results have been replay-validated.
They are only the frozen expected result for the future replay.

# 2. EXECUTOR CONTRACT

DSH has ZERO algorithm/design discretion.

Allowed operations ONLY:

1. verify exact base branch/head and working-tree state;
2. create the exact new audit branch;
3. perform the exact bounded read-only searches in §§4–8;
4. inspect only candidate artifact files allowed by §3;
5. hash candidate artifact files;
6. classify artifact readiness mechanically using §9;
7. create exact evidence/report/FROM_DSH;
8. make exactly one audit commit;
9. push the audit branch once;
10. STOP.

Forbidden:
- NO detector call;
- NO YOLO/Ultralytics inference;
- NO SAM/SAM2 inference;
- NO Qwen/MLLM inference;
- NO `predict.py`;
- NO pipeline inference;
- NO proposal regeneration;
- NO image inference;
- NO final Demo run;
- NO threshold tuning;
- NO source/test/manifest/helper modification;
- NO external RC1 write;
- NO dataset scan outside the explicitly allowed roots;
- NO opening/reading model weights;
- NO `.pt/.pth/.ckpt/.safetensors/.bin` load;
- NO pickle execution / unpickle;
- NO package install/update;
- NO environment change;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO actual production replay;
- NO NEXT.

If a search root does not exist, record `ROOT_MISSING` and continue.
If a candidate file cannot be safely inspected under these rules, record metadata/hash only.
Do NOT invent a workaround.

# 3. EXACT ALLOWED SEARCH ROOTS / FILE TYPES

## 3.1 Repository roots

Read-only search is allowed under exactly:

```text
evaluation\
docs\
handoff\
scripts\
```

This includes tracked and untracked local files under those roots.

Do NOT recursively search:
- datasets;
- model directories;
- `.conda`;
- `.git`;
- delivery model assets;
- arbitrary workspace parent directories.

## 3.2 External RC1 roots

Read-only search is allowed under exactly:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\docs
```

Also allowed at external RC1 root level ONLY:
- `*.json`
- `*.md`
- `*.txt`
- `*.csv`

Do NOT recurse into external:
- `model\`
- `datasets\`
- `runs\`
- `logs\`

## 3.3 Inspectable formats

Text inspection allowed only for files <= 20 MiB with extensions:

```text
.json
.jsonl
.md
.txt
.csv
.yaml
.yml
```

Structured binary metadata inspection allowed only for:

```text
.npy
.npz
```

Rules:
- `.npy/.npz` may be inspected ONLY with NumPy `allow_pickle=False`;
- no object-array loading;
- record keys/shapes/dtypes only unless scalar numeric arrays are clearly proposal records;
- no image/model execution.

For these extensions, metadata/hash only; DO NOT deserialize:

```text
.pkl
.pickle
.joblib
.pt
.pth
.ckpt
.safetensors
.bin
```

# 4. GIT / BRANCH GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= 1197d860b8b1a3503c6cba730eda33ccbff4e1c5
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

Create exactly:

```text
audit/task8b3-ref01-locked-replay-artifacts
```

After branch creation require:

```text
git branch --show-current
= audit/task8b3-ref01-locked-replay-artifacts

git rev-parse HEAD
= 1197d860b8b1a3503c6cba730eda33ccbff4e1c5
```

No other branch operation.

# 5. TRACKED-REPOSITORY EXACT TEXT SEARCH

From repository root run these commands exactly, in this order.

## 5.1 Filename inventory anchors

```text
git ls-files evaluation docs handoff scripts
```

Capture the complete output for audit analysis.

## 5.2 Tile-anchor grep

Run:

```text
git grep -n -I -E "tile.?1010|tile.?1003|tile.?1008|tile.?1009|tile_id.?1010|tile_id.?1003|tile_id.?1008|tile_id.?1009" -- evaluation docs handoff scripts
```

Exit 0 or 1 is allowed.
- exit 0 = matches found;
- exit 1 = no matches;
- any other exit => STOP.

## 5.3 Task/provenance-anchor grep

Run:

```text
git grep -n -I -E "P1D12|P1D11|REF01|REF-01|eligibility.forensics|locked.demo.proposal|proposal.gate|REFERENCE_ELIGIBILITY_BLOCKED|REFERENCE_SELECTION_WRONG_COVERED" -- evaluation docs handoff scripts
```

Exit 0 or 1 allowed only.

## 5.4 Known-proposal-value grep

Run:

```text
git grep -n -I -E "0\.9032501889644747|0\.7026934027671814|0\.296875|5059|11474|0\.39453125" -- evaluation docs handoff scripts
```

Exit 0 or 1 allowed only.

Do NOT broaden grep to the whole repository.

# 6. LOCAL REPOSITORY ALLOWED-ROOT FILESYSTEM SEARCH

Use PowerShell from repository root.

Run exactly:

```powershell
$roots = @("evaluation","docs","handoff","scripts")
$patterns = @("1010","1003","1008","1009","P1D12","REF01","REF-01","proposal","eligibility","locked")
Get-ChildItem $roots -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object {
    $n = $_.Name
    ($patterns | Where-Object { $n -match [regex]::Escape($_) }).Count -gt 0
  } |
  Select-Object FullName,Length,LastWriteTime |
  Sort-Object FullName
```

Capture all output.

Then run text search exactly:

```powershell
Get-ChildItem $roots -Recurse -File -Include *.json,*.jsonl,*.md,*.txt,*.csv,*.yaml,*.yml -ErrorAction SilentlyContinue |
  Where-Object { $_.Length -le 20MB } |
  Select-String -Pattern 'tile.?1010|tile.?1003|tile.?1008|tile.?1009|P1D12|REF01|REF-01|0\.9032501889644747|0\.7026934027671814|0\.296875|5059|11474|0\.39453125' |
  Select-Object Path,LineNumber,Line
```

Capture all output.

Do NOT edit any found file.

# 7. EXTERNAL RC1 BOUNDED READ-ONLY SEARCH

## 7.1 Root existence

Check only:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\docs
```

Record `PRESENT` or `ROOT_MISSING`.

## 7.2 Allowed recursive filename search

For each PRESENT root, run PowerShell:

```powershell
$patterns = @("1010","1003","1008","1009","P1D12","REF01","REF-01","proposal","eligibility","locked")
Get-ChildItem "<ROOT>" -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object {
    $n = $_.Name
    ($patterns | Where-Object { $n -match [regex]::Escape($_) }).Count -gt 0
  } |
  Select-Object FullName,Length,LastWriteTime |
  Sort-Object FullName
```

Replace `<ROOT>` with the exact root, one command per PRESENT root.

## 7.3 External root-level files only

At:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

list ONLY non-recursive:

```powershell
Get-ChildItem "C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1" -File -Include *.json,*.md,*.txt,*.csv -ErrorAction SilentlyContinue |
  Select-Object FullName,Length,LastWriteTime |
  Sort-Object FullName
```

## 7.4 Text anchor search

For each PRESENT recursive root:

```powershell
Get-ChildItem "<ROOT>" -Recurse -File -Include *.json,*.jsonl,*.md,*.txt,*.csv,*.yaml,*.yml -ErrorAction SilentlyContinue |
  Where-Object { $_.Length -le 20MB } |
  Select-String -Pattern 'tile.?1010|tile.?1003|tile.?1008|tile.?1009|P1D12|REF01|REF-01|0\.9032501889644747|0\.7026934027671814|0\.296875|5059|11474|0\.39453125' |
  Select-Object Path,LineNumber,Line
```

No external write is allowed.

# 8. CANDIDATE ARTIFACT INSPECTION CONTRACT

A file becomes a **candidate artifact** only if at least one of the bounded searches above matches one of these locked anchors:

```text
tile 1010 / 1003 / 1008 / 1009
P1D12
REF01 / REF-01
known above proposal values
```

For EVERY candidate artifact:

1. record absolute or repo-relative path;
2. record bytes;
3. compute SHA256 using read-only hashing;
4. record whether tracked by Git;
5. record file type;
6. record which locked case(s) it supports;
7. record which required replay fields are actually present.

For text files, read only the relevant sections/records needed to inventory schema.

For `.npy/.npz`, use `allow_pickle=False` and record:
- keys;
- shapes;
- dtypes;
- scalar field names if visible.

DO NOT infer missing fields from names.
DO NOT deserialize pickle-like formats.

# 9. MECHANICAL REPLAY-READINESS CLASSIFICATION

For each locked case separately, inventory whether the saved artifacts contain these exact categories.

## A. Proposal identity/list

Required:
- stable `proposal_id`;
- complete merged proposal list for the case, not only the winning proposal.

## B. Production-selection scalar fields

Required for every merged proposal:
- `mask_area`;
- `confidence`;
- `touches_image_border`;
- `bbox_extent_ratio`;
- nonempty-mask fact, OR exact saved mask allowing nonempty to be read.

## C. Exact mask/object replay material

For `FULL_PRODUCTION_REPLAY_READY`, also require for every merged proposal:
- exact saved mask / mask_crop / reconstructable serialized data sufficient to instantiate the production proposal object without detector regeneration;
- any additional constructor fields actually required by current `GlobalProposal`.

## D. Forensic correctness linkage

Required to classify semantic correctness:
- saved per-proposal IoU to the locked GT reference, OR
- exact GT-reference mask/linkage plus proposal masks sufficient to recompute IoU with no detector/model call.

Historical prose summaries alone may support provenance comparison, but do NOT satisfy C.

## Overall readiness enum

Set exactly one:

### `FULL_PRODUCTION_REPLAY_READY`

Only if ALL FOUR cases satisfy A+B+C+D.

Meaning:
- future task may invoke the current production selector on exact saved proposal objects/records;
- detector/model calls remain zero.

### `RECORD_LEVEL_REPLAY_READY`

Only if ALL FOUR cases satisfy A+B+D, but at least one case lacks C.

Meaning:
- a deterministic locked proposal-record replay is possible;
- actual production-object replay is NOT yet proven possible.

### `LOCKED_REPLAY_ARTIFACTS_PARTIAL`

If at least one relevant artifact exists, but one or more cases fail A or B or D.

### `LOCKED_REPLAY_ARTIFACTS_NOT_FOUND`

If no relevant candidate artifact is found beyond prose/task summaries that do not contain replay records.

DSH MUST NOT choose a different enum.

# 10. REQUIRED CASE CONSISTENCY CHECKS

Where the saved artifact contains the relevant facts, compare them to the frozen historical facts.

Record each as:

```text
MATCH
MISMATCH
NOT_AVAILABLE
```

Required checks:

```text
right:
raw=6
merged=6
eligible=4
pre_selected=1

left:
raw=66
merged=53
eligible=42
pre_selected=14
best_covered=30

above:
raw=9
merged=9
eligible=4
pre_selected=4
best_covered=5
proposal5 area=5059
proposal5 confidence=0.7026934027671814
proposal5 border=false
proposal5 extent=0.296875
proposal5 IoU=0.9032501889644747
proposal3 area=11474
proposal3 border=true
proposal3 extent=0.39453125

below:
raw=7
merged=6
eligible=3
pre_selected=1
best_covered=2
```

A MISMATCH does NOT authorize correction or detector rerun.
It must be reported as a fact.

# 11. EVIDENCE JSON

Create exactly:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
```

Required top-level schema:

```json
{
  "task": "8B.3-REF01-E3C0",
  "base_head": "1197d860b8b1a3503c6cba730eda33ccbff4e1c5",
  "branch": "audit/task8b3-ref01-locked-replay-artifacts",
  "scope": "READ_ONLY_LOCKED_REPLAY_SAVED_ARTIFACT_AUDIT",
  "detector_model_calls": 0,
  "proposal_regeneration_performed": false,
  "product_source_changed": false,
  "external_write_performed": false,
  "search_roots": {
    "repo": [
      "evaluation",
      "docs",
      "handoff",
      "scripts"
    ],
    "external": [
      "C:\\D\\DeepSeekHarness\\delivery\\BuildReasonSeg_Advisor_RC1\\inference\\output",
      "C:\\D\\DeepSeekHarness\\delivery\\BuildReasonSeg_Advisor_RC1\\docs",
      "C:\\D\\DeepSeekHarness\\delivery\\BuildReasonSeg_Advisor_RC1 (root-level text files only)"
    ]
  },
  "candidate_artifacts": [],
  "cases": {
    "right": {},
    "left": {},
    "above": {},
    "below": {}
  },
  "overall_readiness": "<ONE ENUM FROM SECTION 9>",
  "production_object_replay_possible_without_detector": "<true or false>",
  "record_level_replay_possible_without_detector": "<true or false>",
  "historical_consistency_mismatches": [],
  "overall_outcome": "REF01_LOCKED_REPLAY_ARTIFACT_AUDIT_COMPLETE",
  "next_gate": "<VALUE FROM SECTION 12>"
}
```

Populate `candidate_artifacts` with factual discovered records only.

Each candidate record must contain exactly:

```json
{
  "path": "...",
  "bytes": 0,
  "sha256": "...",
  "tracked_by_git": true,
  "format": "...",
  "supports_cases": [],
  "fields_present": [],
  "inspection_status": "INSPECTED|METADATA_ONLY"
}
```

Each `cases.<case>` object must contain:

```json
{
  "tile_id": 0,
  "artifact_paths": [],
  "complete_proposal_list_present": false,
  "proposal_scalar_fields_complete": false,
  "exact_mask_or_object_material_present": false,
  "forensic_correctness_linkage_present": false,
  "replay_record_count": null,
  "consistency_checks": {}
}
```

Use tile IDs:
- right 1010
- left 1003
- above 1008
- below 1009

Do not add technical conclusions outside this schema.

# 12. NEXT GATE MAPPING

If:

```text
overall_readiness =
FULL_PRODUCTION_REPLAY_READY
```

then:

```text
next_gate =
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

If:

```text
overall_readiness =
RECORD_LEVEL_REPLAY_READY
```

then:

```text
next_gate =
REF01_LOCKED_RECORD_REPLAY_DESIGN
```

If:

```text
overall_readiness =
LOCKED_REPLAY_ARTIFACTS_PARTIAL
or
LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```

then:

```text
next_gate =
REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN
```

No other next gate.

# 13. REPORT

Create exactly:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
```

Required sections:

```text
# Task 8B.3-REF01-E3C0 — Locked Replay Saved-Artifact Audit

## Scope
## Search roots
## Candidate artifacts
## Per-case replay readiness
## Frozen historical consistency checks
## Readiness classification
## Prohibitions respected
## Outcome / NEXT
```

Required conclusions must explicitly state:
- detector/model calls = 0;
- proposal regeneration = NO;
- product modification = NO;
- external write = NO;
- no final Demo run;
- exact overall readiness enum;
- exact next gate;
- PROP-01 remains `PROP01_OPEN_ENGINEERING_DEFECT`;
- REF-01 remains ACTIVE;
- no claim that the repair is replay-validated yet.

# 14. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Active handoff must contain:

```text
Task: 8B.3-REF01-E3C0
Status: COMPLETE
Branch: audit/task8b3-ref01-locked-replay-artifacts
Starting HEAD: 1197d860b8b1a3503c6cba730eda33ccbff4e1c5
Design selected by: CHATGPT
DSH algorithm choice performed: NO
Detector/model calls: 0
Proposal regeneration performed: NO
Product source changed: NO
External write performed: NO
Locked cases audited: right/1010; left/1003; above/1008; below/1009
Candidate artifact count: <observed integer>
Overall replay readiness: <ENUM>
Production-object replay possible without detector: YES / NO
Record-level replay possible without detector: YES / NO
Historical consistency mismatch count: <observed integer>
Evidence: evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
Report: docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
REF-01 status: ACTIVE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Next gate: <SECTION 12 VALUE>
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 15. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 1197d860b8b1a3503c6cba730eda33ccbff4e1c5
```

Allowed ONLY:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path allowed.

Require:

```text
git rev-list --count 1197d860b8b1a3503c6cba730eda33ccbff4e1c5..HEAD
= 0
```

before final commit.

# 16. STATUS / COMMIT / PUSH

This task is an audit. A finding of PARTIAL or NOT_FOUND is still a successful audit completion if all searches were executed correctly.

If §§4–15 execute correctly:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): audit locked replay artifacts
```

Only use STOP if the audit procedure itself could not be completed safely.

STOP commit message:

```text
docs(rc1): record locked replay artifact audit stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 1197d860b8b1a3503c6cba730eda33ccbff4e1c5..HEAD
= 1
```

Push exactly:

```text
audit/task8b3-ref01-locked-replay-artifacts
```

once.

No force push.
Do not update main.
Do not execute NEXT.

Then STOP.

# 17. COMPLETE DEFINITION

COMPLETE means:
- exact base branch/head;
- exact audit branch;
- bounded searches only;
- zero detector/model calls;
- zero proposal regeneration;
- zero external writes;
- no product/test/manifest/helper changes;
- candidate files hashed and schema-inventoried;
- four locked cases mechanically classified;
- readiness enum chosen only by §9;
- exact evidence/report/FROM_DSH created;
- only four allowed repo paths changed;
- exactly one audit commit with exact message;
- audit branch pushed;
- no replay/NEXT executed;
- STOP.
