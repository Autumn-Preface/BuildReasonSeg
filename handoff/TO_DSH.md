请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0-R1 — Bounded Artifact Audit Correction**

# TO_DSH — Task 8B.3-REF01-E3C0-R1: Bounded Artifact Audit Correction

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `bde13bd15bbab3e455ea1d3ef10b3f6740fc110d`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. CHATGPT AUDIT DISPOSITION

E3C0 is **NOT APPROVED**.

The prior commit is preserved in history. Do NOT rewrite or amend it.

Observed defects that R1 must correct:

1. The previous audit searched forbidden repository roots:
   - `...\BuildReasonSeg\artifacts`
   - `...\BuildReasonSeg\inference`

2. The previous audit searched forbidden external root:
   - `...\BuildReasonSeg_Advisor_RC1\logs`

3. The evidence schema did not follow the prescribed E3C0 schema.

4. The previous audit invented unapproved readiness/outcome enums such as:
   - `SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU`
   - `REFERENCE_IOU_REPLAY_READY_END_TO_END_NOT_READY`
   - `REF01_LOCKED_REPLAY_ARTIFACT_COMPLETION`

5. The report path was wrong:
   - created: `docs/task8b3_ref01_locked_replay_artifact_audit.md`
   - required: `docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md`

6. The report and FROM_DSH discussed end-to-end replay gaps that were outside this audit decision tree.

R1 must discard all conclusions derived from forbidden roots and redo the audit using ONLY the roots and classification rules in this taskbook.

# 1. SCIENTIFIC / ENGINEERING STATE

Frozen repair design:

```text
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
```

Locked cases:

```text
right  -> tile 1010
left   -> tile 1003
above  -> tile 1008
below  -> tile 1009
```

Frozen historical facts:

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
proposal5:
  mask_area=5059
  confidence=0.7026934027671814
  touches_image_border=false
  bbox_extent_ratio=0.296875
  IoU=0.9032501889644747
proposal3:
  mask_area=11474
  touches_image_border=true
  bbox_extent_ratio=0.39453125

below:
raw=7
merged=6
eligible=3
pre_selected=1
best_covered=2
```

Frozen expected later replay result:

```text
right: 1 -> 1
left: 14 -> 14
above: 4 -> 5
below: 1 -> 1
```

R1 MUST NOT claim this post-repair result has been replay-validated.

# 2. EXECUTOR CONTRACT

DSH has ZERO technical/design discretion.

Allowed operations ONLY:

1. verify exact branch/head and working-tree state;
2. perform the exact bounded read-only searches in §§5–7;
3. inspect only candidate files allowed by §4;
4. hash those candidate files;
5. classify readiness mechanically using §8;
6. compare only the frozen facts in §1;
7. delete the wrong prior report path;
8. overwrite the E3C0 evidence JSON with the corrected exact schema;
9. create the correct E3C0 report path;
10. overwrite FROM_DSH with the corrected result while preserving ARTIFACT-FACTS exactly;
11. leave this taskbook as `handoff/TO_DSH.md`;
12. make exactly one R1 commit;
13. push once;
14. STOP.

Forbidden:
- NO detector call;
- NO YOLO/Ultralytics inference;
- NO SAM/SAM2 inference;
- NO Qwen/MLLM inference;
- NO `predict.py`;
- NO proposal regeneration;
- NO image inference;
- NO actual replay;
- NO final Demo;
- NO threshold tuning;
- NO source/test/manifest/helper edit;
- NO external RC1 write;
- NO search under repository `artifacts\`;
- NO search under repository `inference\`;
- NO search under external `logs\`;
- NO search under datasets/model/runs;
- NO arbitrary workspace-parent search;
- NO `.pt/.pth/.ckpt/.safetensors/.bin` load;
- NO pickle/unpickle;
- NO package/environment change;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO NEXT.

Any uncovered condition => STOP and report facts. Do not invent a workaround.

# 3. CURRENT PRODUCTION OBJECT CONTRACT — READ ONLY

For deciding whether full production-object replay is possible, use the current committed `GlobalProposal` constructor contract only.

The current constructor fields are:

```text
proposal_id
source_tile_id
tile_index
confidence
mask_crop
global_bbox
mask_area
touches_image_border
border_clearance
centroid
raw_index
pad_mask_empty
image_size
```

The `bbox_extent_ratio` is derived from `global_bbox`.

For `FULL_PRODUCTION_REPLAY_READY`, saved artifacts must be sufficient to reconstruct every merged proposal for all four cases with all constructor fields required by the current code, including exact `mask_crop`, without detector/model regeneration.

A visualization PNG is NOT an exact proposal mask serialization.
A prose statement is NOT mask material.

# 4. EXACT ALLOWED ROOTS / FORMATS

## 4.1 Repository roots

Read-only search is allowed ONLY under:

```text
evaluation\
docs\
handoff\
scripts\
```

Repository paths explicitly forbidden:

```text
artifacts\
inference\
datasets\
model\
.conda\
.git\
```

## 4.2 External RC1

Recursive read-only search is allowed ONLY under:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\docs
```

At the external RC1 root itself, only a NON-RECURSIVE listing/inspection of:

```text
*.json
*.md
*.txt
*.csv
```

is allowed.

Explicitly forbidden external recursion:

```text
logs\
model\
datasets\
runs\
```

## 4.3 Safe inspectable types

Text content inspection only for files <=20 MiB:

```text
.json
.jsonl
.md
.txt
.csv
.yaml
.yml
```

For `.npy/.npz`:
- inspect only with NumPy `allow_pickle=False`;
- no object arrays;
- record keys/shapes/dtypes;
- numeric/scalar record values may be read only if clearly part of the candidate proposal/GT artifact.

For:
```text
.png
.jpg
.jpeg
.tif
.tiff
```
metadata/hash only. Do NOT treat a visualization image as proposal mask/object replay material unless the file is explicitly documented by an allowed text artifact as a machine-readable exact mask AND this exact fact is recorded. Otherwise C=false.

For:
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
metadata/hash only; do not deserialize/load.

# 5. GIT GATE

Require:

```text
git branch --show-current
= audit/task8b3-ref01-locked-replay-artifacts

git rev-parse HEAD
= bde13bd15bbab3e455ea1d3ef10b3f6740fc110d
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No branch creation/switching.

Require before final commit:

```text
git rev-list --count bde13bd15bbab3e455ea1d3ef10b3f6740fc110d..HEAD
= 0
```

# 6. REPOSITORY BOUNDED SEARCH — EXACT

From repository root run exactly:

```text
git ls-files evaluation docs handoff scripts
```

Then:

```text
git grep -n -I -E "tile.?1010|tile.?1003|tile.?1008|tile.?1009|tile_id.?1010|tile_id.?1003|tile_id.?1008|tile_id.?1009" -- evaluation docs handoff scripts
```

Then:

```text
git grep -n -I -E "P1D12|P1D11|REF01|REF-01|eligibility.forensics|locked.demo.proposal|proposal.gate|REFERENCE_ELIGIBILITY_BLOCKED|REFERENCE_SELECTION_WRONG_COVERED" -- evaluation docs handoff scripts
```

Then:

```text
git grep -n -I -E "0\.9032501889644747|0\.7026934027671814|0\.296875|5059|11474|0\.39453125" -- evaluation docs handoff scripts
```

For each grep, exit 0 or 1 is allowed; any other exit => STOP.

Local untracked search ONLY under these same roots:

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

Then:

```powershell
Get-ChildItem $roots -Recurse -File -Include *.json,*.jsonl,*.md,*.txt,*.csv,*.yaml,*.yml -ErrorAction SilentlyContinue |
  Where-Object { $_.Length -le 20MB } |
  Select-String -Pattern 'tile.?1010|tile.?1003|tile.?1008|tile.?1009|P1D12|REF01|REF-01|0\.9032501889644747|0\.7026934027671814|0\.296875|5059|11474|0\.39453125' |
  Select-Object Path,LineNumber,Line
```

Do NOT search any other repository root.

# 7. EXTERNAL BOUNDED SEARCH — EXACT

Check existence ONLY for:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\docs
```

Record `PRESENT` or `ROOT_MISSING`.

For each PRESENT root run:

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

Then for each PRESENT root:

```powershell
Get-ChildItem "<ROOT>" -Recurse -File -Include *.json,*.jsonl,*.md,*.txt,*.csv,*.yaml,*.yml -ErrorAction SilentlyContinue |
  Where-Object { $_.Length -le 20MB } |
  Select-String -Pattern 'tile.?1010|tile.?1003|tile.?1008|tile.?1009|P1D12|REF01|REF-01|0\.9032501889644747|0\.7026934027671814|0\.296875|5059|11474|0\.39453125' |
  Select-Object Path,LineNumber,Line
```

External root-level NON-RECURSIVE command only:

```powershell
Get-ChildItem "C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1" -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in ".json",".md",".txt",".csv" } |
  Select-Object FullName,Length,LastWriteTime |
  Sort-Object FullName
```

Do NOT run any command against external `logs`.

# 8. MECHANICAL READINESS RULE

For each case, determine A/B/C/D independently.

## A — complete proposal list

A=true only if an allowed artifact establishes:
- stable `proposal_id`;
- complete merged proposal list;
- record count equals frozen merged count:
  - right 6
  - left 53
  - above 9
  - below 6

Otherwise A=false.

## B — production selection scalars

B=true only if EVERY merged proposal record for the case contains:
- `proposal_id`;
- `mask_area`;
- `confidence`;
- `touches_image_border`;
- `bbox_extent_ratio` OR `global_bbox` sufficient to derive it exactly;
- explicit nonempty-mask fact OR exact saved mask material from which nonempty is directly readable.

Do NOT assume `mask_area > 0` is equivalent to `mask_crop.any()` unless an allowed artifact explicitly records nonempty or exact mask material exists.

Otherwise B=false.

## C — exact production object/mask material

C=true only if EVERY merged proposal can be reconstructed as current `GlobalProposal` without detector/model inference, including exact `mask_crop` and required constructor fields.

`global_proposals.png` alone never satisfies C.

Otherwise C=false.

## D — forensic correctness linkage

D=true only if, for every merged proposal needed to establish selected/best-covered reference correctness, allowed artifacts contain either:
- saved per-proposal IoU to the locked GT reference; OR
- exact proposal mask + exact GT-reference mask/linkage sufficient to recompute IoU with zero model calls.

Historical prose summaries do not satisfy D by themselves.

Otherwise D=false.

## Overall enum — EXACTLY ONE

```text
FULL_PRODUCTION_REPLAY_READY
```
iff A+B+C+D are true for all four cases.

```text
RECORD_LEVEL_REPLAY_READY
```
iff A+B+D are true for all four cases and C is false for at least one case.

```text
LOCKED_REPLAY_ARTIFACTS_PARTIAL
```
iff at least one relevant allowed artifact exists but at least one case fails A or B or D.

```text
LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```
iff no relevant candidate replay artifact exists beyond prose summaries.

NO OTHER ENUM IS PERMITTED.

Boolean mapping:

```text
production_object_replay_possible_without_detector =
true only for FULL_PRODUCTION_REPLAY_READY

record_level_replay_possible_without_detector =
true for FULL_PRODUCTION_REPLAY_READY or RECORD_LEVEL_REPLAY_READY
false otherwise
```

# 9. CANDIDATE ARTIFACT RECORDING

A candidate artifact is only a file found from the allowed searches in §§6–7 that materially supports A/B/C/D or a frozen consistency check.

For every candidate record exactly:

```json
{
  "path": "...",
  "bytes": 0,
  "sha256": "...",
  "tracked_by_git": true,
  "format": "...",
  "supports_cases": [],
  "fields_present": [],
  "inspection_status": "INSPECTED"
}
```

Use `"METADATA_ONLY"` instead of `"INSPECTED"` if content was not safely inspected.

For external files:
```text
tracked_by_git = false
```

Do NOT include files discovered only through the forbidden prior searches.

# 10. CONSISTENCY CHECKS — EXACT KEYS

For each case use only:
`MATCH`, `MISMATCH`, `NOT_AVAILABLE`.

right:
```text
raw_count
merged_count
eligible_count
pre_selected
```

left:
```text
raw_count
merged_count
eligible_count
pre_selected
best_covered
```

above:
```text
raw_count
merged_count
eligible_count
pre_selected
best_covered
proposal5_mask_area
proposal5_confidence
proposal5_touches_image_border
proposal5_bbox_extent_ratio
proposal5_iou
proposal3_mask_area
proposal3_touches_image_border
proposal3_bbox_extent_ratio
```

below:
```text
raw_count
merged_count
eligible_count
pre_selected
best_covered
```

No extra interpretation.

# 11. CORRECTED EVIDENCE JSON — EXACT TOP LEVEL

Overwrite exactly:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
```

Use exactly these top-level keys:

```json
{
  "task": "8B.3-REF01-E3C0-R1",
  "base_head": "bde13bd15bbab3e455ea1d3ef10b3f6740fc110d",
  "branch": "audit/task8b3-ref01-locked-replay-artifacts",
  "scope": "READ_ONLY_LOCKED_REPLAY_SAVED_ARTIFACT_AUDIT_CORRECTION",
  "detector_model_calls": 0,
  "proposal_regeneration_performed": false,
  "product_source_changed": false,
  "external_write_performed": false,
  "forbidden_prior_roots_discarded": [
    "C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\artifacts",
    "C:\\D\\DeepSeekHarness\\workspace\\project\\BuildReasonSeg\\inference",
    "C:\\D\\DeepSeekHarness\\delivery\\BuildReasonSeg_Advisor_RC1\\logs"
  ],
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
  "overall_readiness": "<EXACT ENUM>",
  "production_object_replay_possible_without_detector": false,
  "record_level_replay_possible_without_detector": false,
  "historical_consistency_mismatches": [],
  "overall_outcome": "REF01_LOCKED_REPLAY_ARTIFACT_AUDIT_CORRECTED",
  "next_gate": "<EXACT MAPPED VALUE>"
}
```

Each `cases.<case>` object must contain exactly:

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

Tile IDs:
- right 1010
- left 1003
- above 1008
- below 1009

No alternate schema.

# 12. NEXT GATE — EXACT MAPPING

If:

```text
FULL_PRODUCTION_REPLAY_READY
```

then:

```text
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

If:

```text
RECORD_LEVEL_REPLAY_READY
```

then:

```text
REF01_LOCKED_RECORD_REPLAY_DESIGN
```

If:

```text
LOCKED_REPLAY_ARTIFACTS_PARTIAL
```

or:

```text
LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```

then:

```text
REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN
```

No other next gate.

# 13. REPORT PATH CORRECTION

Delete exactly:

```text
docs/task8b3_ref01_locked_replay_artifact_audit.md
```

Create exactly:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
```

Required headings exactly:

```text
# Task 8B.3-REF01-E3C0-R1 — Locked Replay Saved-Artifact Audit Correction

## Scope
## Search roots
## Candidate artifacts
## Per-case replay readiness
## Frozen historical consistency checks
## Readiness classification
## Prohibitions respected
## Outcome / NEXT
```

Required explicit statements:
- detector/model calls = 0;
- proposal regeneration = NO;
- actual replay = NO;
- product modification = NO;
- external write = NO;
- forbidden prior-root findings were discarded;
- no final Demo;
- exact overall readiness enum;
- exact mapped next gate;
- PROP-01 remains `PROP01_OPEN_ENGINEERING_DEFECT`;
- REF-01 remains ACTIVE;
- post-repair `right1/left14/above5/below1` has NOT been replay-validated.

Do NOT discuss generic end-to-end language/SAM2/D-B1/final-mask gaps unless such a fact is directly required to classify A/B/C/D. This audit is about locked reference-selection replay inputs only.

# 14. FROM_DSH

Preserve the ARTIFACT-FACTS block byte-for-byte.

Replace the active handoff with this exact field format:

```text
Task: 8B.3-REF01-E3C0-R1
Status: COMPLETE
Branch: audit/task8b3-ref01-locked-replay-artifacts
Starting HEAD: bde13bd15bbab3e455ea1d3ef10b3f6740fc110d
Design selected by: CHATGPT
DSH algorithm choice performed: NO
Detector/model calls: 0
Proposal regeneration performed: NO
Actual replay performed: NO
Product source changed: NO
External write performed: NO
Forbidden prior-root findings discarded: YES
Locked cases audited: right/1010; left/1003; above/1008; below/1009
Candidate artifact count: <observed integer>
Overall replay readiness: <EXACT ENUM>
Production-object replay possible without detector: YES / NO
Record-level replay possible without detector: YES / NO
Historical consistency mismatch count: <observed integer>
Evidence: evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
Report: docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
REF-01 status: ACTIVE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Next gate: <EXACT MAPPED VALUE>
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 15. PROTECTED FILE GATE

Require no R1 change to:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
scripts/sync_advisor_rc1_delivery.py
docs/task8b3_ref01_eligibility_repair_impl.md
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

No external RC1 file may be written.

# 16. FINAL DIFF GATE

Before commit, relative to R1 starting HEAD:

```text
git diff --name-only bde13bd15bbab3e455ea1d3ef10b3f6740fc110d
```

Allowed ONLY these five paths:

```text
docs/task8b3_ref01_locked_replay_artifact_audit.md
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

The first path MUST be DELETED.
The second path MUST be ADDED.
The remaining three may be MODIFIED.

No other path.

Require before commit:

```text
git rev-list --count bde13bd15bbab3e455ea1d3ef10b3f6740fc110d..HEAD
= 0
```

# 17. COMMIT / PUSH / STOP

If all audit procedure gates PASS, regardless of which allowed readiness enum is observed:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): correct locked replay artifact audit
```

If the correction audit itself cannot be completed safely:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record locked replay artifact audit correction stop
```

Exactly one commit.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count bde13bd15bbab3e455ea1d3ef10b3f6740fc110d..HEAD
= 1
```

Push current branch once:

```text
audit/task8b3-ref01-locked-replay-artifacts
```

No force push.
Do not execute NEXT.
Then STOP.

# 18. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- no forbidden root search in R1;
- forbidden prior-root findings excluded from corrected evidence;
- zero detector/model calls;
- zero proposal regeneration;
- zero actual replay;
- zero external writes;
- zero product/test/manifest/helper changes;
- candidate artifacts come only from allowed roots;
- A/B/C/D evaluated mechanically for all four cases;
- exact readiness enum used;
- exact next-gate mapping used;
- exact evidence schema used;
- wrong report path deleted;
- correct report path created;
- FROM_DSH uses exact field format;
- only five allowed paths differ;
- exactly one R1 commit with exact message;
- push succeeds;
- NEXT not executed;
- STOP.
