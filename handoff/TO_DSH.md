# TO_DSH — Task 8B.3-P1D12-R1: Proposal-Gate Evidence Closure (NO RERUN)

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `75e01defde3ad6e9f8a247481028536e68ef4ee4`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> RC1 Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

P1D12 result is **PROVISIONAL PASS — evidence closure required**.

Already accepted from the committed report:

```text
right raw/merged/eligible = 6/6/4
left  raw/merged/eligible = 66/53/42
above raw/merged/eligible = 9/9/4
below raw/merged/eligible = 7/6/3
all four exits = 0
all four status = SUCCESS
candidate replacement = NO
manual visual inspection = NO
GT access = NO
```

However, the committed report did not record all task-book evidence needed for final approval:

```text
1. exact source-raster SHA256 / 512x512 / RGB identity;
2. current external setup + 135/135 source/config integrity;
3. frozen detector constants;
4. exact inspect-only timing fields showing SAM2 / relation-fields / D-B1 = 0;
5. candidate diagnostics contents proving no core-chain artifacts;
6. no newly created candidate mask/overlay;
```

Additionally, the executed source paths include:

```text
...\1. The cropped image data and raster labels\test\image\<tile>.tif
```

while the earlier task book used a shortened dataset path.

This task does **not** rerun any candidate. Exact raster bytes (SHA256) are authoritative.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 75e01defde3ad6e9f8a247481028536e68ef4ee4
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- rerun `predict.py`;
- rerun `--inspect-proposals`;
- run any model/detector inference;
- run Qwen / ProgramHead;
- run SAM2;
- run D-B1;
- run reference selection;
- run pytest;
- modify canonical RC1;
- modify external RC1 source/config/model assets;
- modify or delete candidate diagnostics;
- inspect `global_proposals.png` visually;
- inspect source images visually for quality;
- access GT masks/polygons;
- compute GT IoU;
- replace any candidate;
- change thresholds/config/model;
- run write sync;
- update main;
- force push.

Only read existing files plus the setup/integrity checks explicitly authorized below.

# 3. Allowed tracked changes

Only:

```text
docs/task8b3_p1d12_locked_demo_proposal_gate.md
docs/task8b3_p1d12_r1_evidence_closure.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No source/test/config/model tracked file may change.

# 4. Current external integrity check

Run the approved sync helper in read-only mode:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Require:

```text
checked = 135
match = 135
missing = 0
mismatch = 0
exit code = 0
```

Then run from the external RC1 root with the specified RC1 Python:

```text
python check_setup.py
```

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

This is an integrity check only. No candidate execution.

If either gate fails:
- do not continue;
- STOP.

# 5. Exact locked raster identity — authoritative bytes

For the exact four source paths recorded in the P1D12 report, read metadata only.

Expected authoritative identities:

```text
right / 1010.tif
SHA256 = 1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

left / 1003.tif
SHA256 = eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

above / 1008.tif
SHA256 = 0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

below / 1009.tif
SHA256 = c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

For each record:

```text
absolute path
exists
byte size
SHA256
width
height
PIL/image mode
channel count
```

Require all:

```text
SHA256 = exact expected value
width = 512
height = 512
RGB-readable = YES
```

Path spelling/subdirectory may differ from the shortened task-book path **only if the exact SHA256 lock matches**.

If any SHA differs:
- outcome = EVIDENCE_FAILED;
- do not substitute another raster;
- STOP.

# 6. Frozen detector constants — read only

Read current external:

```text
buildreasonseg/runtime/detector.py
```

Require exactly:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
DUPLICATE_IOU = 0.50
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
FROZEN_THRESHOLD = 0.5
```

Record:

```text
DETECTOR_CONSTANTS = MATCH
```

If not exact:
- STOP.

# 7. Existing diagnostics — NO rerun

Read only these already-created directories:

```text
external/inference/output/diagnostics/1010
external/inference/output/diagnostics/1003
external/inference/output/diagnostics/1008
external/inference/output/diagnostics/1009
```

For each directory require it exists and contains exactly the inspect-mode artifacts already reported:

```text
global_proposals.png
parsed_program.json
prompt.txt
proposals.json
result.json
```

No need to hash/read `global_proposals.png` image content.

Require absence of:

```text
selected_reference.png
reasoning_context.png
reference_context_mask.png
direction_field.png
nearest_field.png
relation_weight.png
prototype_similarity.png
maps.npz
```

Do not delete anything.

# 8. Re-read result/proposal evidence

For each candidate, read only:

```text
result.json
proposals.json
```

Recompute and record:

```text
status
mode
raw_proposal_count
merged_proposal_count
proposals.count
number of proposals.items
tile_count
tile_size
overlap
timings.detector
timings.sam2
timings.relation_fields
timings.db1
prompt field value / presence
parsed field value / presence
result.json SHA256
proposals.json SHA256
```

Expected P1D12 counts:

```text
right = raw 6 / merged 6 / eligible 4
left  = raw 66 / merged 53 / eligible 42
above = raw 9 / merged 9 / eligible 4
below = raw 7 / merged 6 / eligible 3
```

Require for all four:

```text
status = SUCCESS
mode = inspect
merged_proposal_count = proposals.count = len(items)
tile_count = 1
tile_size = 512
overlap = 128
timings.sam2 = 0.0
timings.relation_fields = 0.0
timings.db1 = 0.0
prompt absent or ""
parsed absent, null, or {}
```

# 9. Recompute mechanical eligible count

From `proposals.json` only, recompute:

```text
eligible_largest =
    mask_area > 0
    AND touches_image_border == false
    AND bbox_extent_ratio <= 0.20
```

Require exact:

```text
right = 4
left = 42
above = 4
below = 3
```

No ranking. No reference selection. No visual judgement.

# 10. Mask / overlay non-production evidence

For the four stems:

```text
1010
1003
1008
1009
```

Check external:

```text
inference/output/masks
inference/output/overlays
```

The P1D12 inspect-only run must not have created candidate-specific mask/overlay files.

Because prior unrelated files could theoretically exist, use timestamps only if needed to distinguish pre-existing files.
Do not delete/modify anything.

Record one of:

```text
P1D12 candidate mask/overlay produced = NO
```

or, if attribution cannot be established:

```text
P1D12 candidate mask/overlay attribution = NOT ESTABLISHED
```

If there is affirmative evidence that P1D12 produced a candidate mask/overlay:
- outcome = EVIDENCE_FAILED;
- STOP.

# 11. No re-execution evidence

Record explicitly:

```text
candidate inference invocations in R1 = 0
predict.py invocations in R1 = 0
detector inference in R1 = 0
candidate diagnostics rewritten = NO
```

# 12. Final classification

## A — `PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_CLOSED`

Require all:

```text
external source/config integrity = 135/135
external setup = READY
4/4 locked raster SHA = exact match
4/4 raster dimensions = 512x512
detector constants = MATCH
existing result/proposals evidence reproduces P1D12 counts
eligible counts = 4 / 42 / 4 / 3
inspect-only timing proof = PASS
core-chain artifacts = NONE
no affirmative candidate mask/overlay production
no candidate rerun
```

Then freeze:

```text
P1D12 = APPROVABLE
PROP01_LOCKED_DEMO_PROPOSAL_GATE_PASS
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

Do not execute NEXT.

## B — `PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_FAILED`

For any contradiction in immutable raster identity, counts, inspect-only proof, or integrity.

Then:

```text
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_EVIDENCE_RECOVERY
```

Do not execute NEXT.

# 13. Report

Create:

```text
docs/task8b3_p1d12_r1_evidence_closure.md
```

Append a short R1 evidence-closure section to:

```text
docs/task8b3_p1d12_locked_demo_proposal_gate.md
```

Required sections:

1. starting HEAD
2. no-rerun declaration
3. external 135/135 + setup READY
4. exact four raster paths + SHA/size/dimensions/mode
5. detector constants
6. per-candidate existing diagnostics file set
7. per-candidate result/proposals fields
8. recomputed eligible counts
9. SAM2/relation/D-B1 zero-timing proof
10. no core artifacts
11. mask/overlay attribution
12. exact outcome
13. exact NEXT.

# 14. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D12-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 75e01defde3ad6e9f8a247481028536e68ef4ee4
Candidate rerun: NO
predict.py executed: NO
Detector inference executed: NO
External sync check: 135/135 PASS / FAIL
External setup: READY / NOT READY
Locked raster SHA identities: 4/4 PASS / other
Locked raster dimensions: 4/4 512x512 / other
Locked raster RGB readability: 4/4 PASS / other
Detector constants: MATCH / other
Existing diagnostics dirs: 4/4 PRESENT / other
Existing diagnostics inspect-only file set: 4/4 PASS / other
Core-chain artifacts present: NO / YES
SAM2/relation/DB1 timings: 0/0/0 for all four / other
right raw/merged/eligible: 6/6/4 / other
left raw/merged/eligible: 66/53/42 / other
above raw/merged/eligible: 9/9/4 / other
below raw/merged/eligible: 7/6/3 / other
Candidate mask/overlay produced by P1D12: NO / YES / NOT ESTABLISHED
Candidate replacement: NO
Visual inspection: NO
Ground-truth access: NO
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d12_r1_evidence_closure.md
Next action: Awaiting ChatGPT audit; do not run reference selection or full inference.
```

# 15. Commit / push

If COMPLETE:

```text
docs(rc1): close locked demo proposal evidence
```

If STOP/FAILED:

```text
docs(rc1): record locked demo proposal evidence stop
```

Push normally.
No force push.

# 16. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no candidate rerun;
- no model inference;
- no source/config/model external modification;
- current external integrity is 135/135 and setup READY;
- all four exact raster hashes match immutable locks;
- dimensions/RGB status match;
- detector constants match;
- existing diagnostics reproduce all P1D12 counts;
- eligible counts reproduce exactly;
- inspect-only timings prove no SAM2/relation/D-B1 execution;
- no core-chain diagnostics exist;
- no affirmative mask/overlay creation by P1D12;
- exact outcome assigned;
- next gate not executed;
- only allowed tracked docs/handoff committed;
- push succeeds;
- STOP.
