# TO_DSH — Task 8B.3-P1D12: Locked Supported-Domain Demo Proposal Gate

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `6651c4bf89663e5fff3e7854c9e2115d084ae003`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. Frozen entering state

ChatGPT approves P1D11B-R2 as the completed external Git-canonical migration / policy sync.

Freeze:

```text
external manifest-listed source/config = 135/135 Git-canonical match
external source_manifest = Git-canonical match
external setup = READY
supported-domain policy = synchronized
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
A2 provenance/domain = NOT ESTABLISHED
A2 = documented persistent non-detection / stress case
locked candidate status = FINAL_METADATA_LOCK
locked candidate runtime status = NOT YET RUN
```

This task runs the four locked supported-domain candidates through the **proposal-only** inspection path.

It does NOT run:
- natural-language parsing;
- Qwen;
- SAM2;
- relation fields;
- D-B1;
- target segmentation;
- manual visual evaluation;
- ground truth comparison.

# 1. Locked candidates — immutable

The four candidates are fixed and MUST NOT be replaced.

## right

```text
sample_id =
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91

program =
largest_to_right_of_to_nearest

image =
C:\D\resources\Satellite dataset Ⅱ (East Asia)\test\image\1010.tif

sha256 =
1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2
```

## left

```text
sample_id =
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3

program =
largest_to_left_of_to_nearest

image =
C:\D\resources\Satellite dataset Ⅱ (East Asia)\test\image\1003.tif

sha256 =
eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38
```

## above

```text
sample_id =
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

program =
largest_to_above_to_nearest

image =
C:\D\resources\Satellite dataset Ⅱ (East Asia)\test\image\1008.tif

sha256 =
0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd
```

## below

```text
sample_id =
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450

program =
largest_to_below_to_nearest

image =
C:\D\resources\Satellite dataset Ⅱ (East Asia)\test\image\1009.tif

sha256 =
c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

Candidate replacement is forbidden regardless of runtime outcome.

# 2. Scientific-use disclosure

These candidates are qualitative Demo candidates from the frozen BuildSpatialReason v0.2 TEST split selected
deterministically from metadata after Task 7J final frozen-architecture test metrics were already consumed.

Their use here:
- does not change any Task 7J metric;
- does not select a model;
- does not select a seed;
- does not change a threshold;
- does not change architecture;
- does not replace the historical A1/A2/A3/A4/B1/B2 diagnostic evidence.

No performance metric is being recomputed in this task.

# 3. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 6651c4bf89663e5fff3e7854c9e2115d084ae003
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 4. Strict prohibitions

Do NOT:
- modify canonical RC1;
- modify external RC1 source/config/model assets;
- run sync helper in write mode;
- run pytest;
- run training/fine-tuning/export/download;
- provide any prompt to `predict.py`;
- run normal `predict.py` inference;
- run Qwen / ProgramHead;
- run SAM2;
- run relation-field computation;
- run D-B1;
- produce target mask/overlay;
- use `--reference-id`;
- inspect `global_proposals.png` visually;
- inspect the source images manually to decide quality;
- use ground truth polygons/masks;
- compute proposal-vs-GT IoU;
- compute Task7J/validation metrics;
- select or replace candidates;
- change detector conf/imgsz/max_det/tiling/NMS/merge thresholds;
- modify REF-01 or MASK-01;
- enter Task 8B.4 / 8C;
- update main;
- force push.

Normal diagnostic output produced by `--inspect-proposals` under external `inference/output/diagnostics` is authorized.

# 5. Allowed tracked repository changes

Only:

```text
docs/task8b3_p1d12_locked_demo_proposal_gate.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No source/test/config/model tracked file may change.

# 6. External integrity preflight

From external RC1 run:

```text
python check_setup.py
```

using the specified RC1 Python.

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

Then run the approved sync helper in read-only mode:

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
```

If either gate fails:
- do not run any candidate;
- STOP.

# 7. Candidate source identity preflight

For all four locked rasters, before any inference:

Require:
- file exists;
- SHA256 exactly matches §1;
- width = 512;
- height = 512;
- readable as RGB optical TIFF.

Record exact:
- path;
- SHA256;
- byte size;
- width/height;
- image mode / channel count.

If any raster identity differs:
- do not replace it;
- do not run another candidate;
- STOP.

# 8. Frozen detector configuration audit

Read only the synchronized external:

```text
buildreasonseg/runtime/detector.py
```

Record and require:

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

Do not modify any value.

If any value differs:
- STOP before candidate execution.

# 9. Clean first-run output gate

The locked candidates have frozen runtime status `NOT YET RUN`.

Before running, require these four diagnostics directories do NOT already exist:

```text
inference/output/diagnostics/1010
inference/output/diagnostics/1003
inference/output/diagnostics/1008
inference/output/diagnostics/1009
```

Also require no corresponding inspect-created diagnostic suffix directories:

```text
1010_001, 1010_002, ...
1003_001, 1003_002, ...
1008_001, 1008_002, ...
1009_001, 1009_002, ...
```

If any candidate-specific diagnostics directory already exists:
- do NOT delete or overwrite it;
- STOP and report exact paths.

Do not use mask/overlay presence as the only history check, because inspect mode does not create a mask.

# 10. Proposal-only execution path

Run in this exact order:

```text
right → left → above → below
```

For each candidate, execute exactly once from external RC1:

```text
<RC1_PYTHON> predict.py --image "<LOCKED_ABSOLUTE_IMAGE_PATH>" --inspect-proposals
```

No `--prompt`.
No `--reference-id`.

Expected code path:

```text
model-package verification
→ image load
→ DetectorRuntime.detect_global
→ proposal merge
→ diagnostics save
→ return

NO language parse
NO reference selection
NO SAM2
NO relation fields
NO D-B1
NO final mask
```

For each invocation:
- record command;
- record exit code;
- record wall time;
- record generated diagnostics directory.

If a process exits non-zero:
- STOP immediately;
- no retry;
- do not run remaining candidates.

A zero-proposal result is NOT a process failure; inspect mode may still exit 0.
Do not stop early merely because raw/merged proposal count is zero; run all four unless the process itself fails.

# 11. Required per-candidate evidence

For each candidate read:

```text
<diagnostics_dir>/result.json
<diagnostics_dir>/proposals.json
```

Do not visually inspect PNG previews.

Record:

```text
sample_id
program label
source image path
source SHA256
process exit code
result status
raw_proposal_count
merged_proposal_count
proposal item count
tile_count
tile_size
overlap
detector timing
diagnostics directory
result.json SHA256
proposals.json SHA256
```

Require internal consistency:

```text
result.status = SUCCESS
merged_proposal_count = proposals.count
proposals.count = number of items in proposals.json
tile_count = 1
tile_size = 512
overlap = 128
```

# 12. Compute mechanical frozen eligibility count

Do NOT run reference selection.

From `proposals.json` items only, compute:

```text
eligible_largest =
    mask_area > 0
    AND touches_image_border == false
    AND bbox_extent_ratio <= 0.20
```

For each candidate record:

```text
eligible_largest_count
```

This is a mechanical audit of the frozen proposal/reference interface only.

Do NOT:
- choose the largest candidate;
- rank proposal quality;
- judge whether a proposal is visually a building;
- inspect the preview image;
- use GT.

# 13. Core-chain non-execution proof

For every `result.json`, require:

```text
mode = inspect
prompt absent or empty
parsed program absent/empty
timings.sam2 = 0.0
timings.relation_fields = 0.0
timings.db1 = 0.0
```

Require no candidate diagnostics directory contains:

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

Require no candidate-specific mask/overlay was newly produced by this task.

If any core-chain artifact exists from this run:
- outcome = FAILED;
- STOP.

# 14. Gate classifications

After all four candidates run, classify exactly one outcome.

## A — `PROP01_LOCKED_DEMO_PROPOSAL_GATE_PASS`

Require for all four:

```text
exit = 0
status = SUCCESS
raw_proposal_count > 0
merged_proposal_count > 0
eligible_largest_count > 0
core chain not executed
```

Interpretation:

```text
The locked supported-domain Demo path does not reproduce A2-style proposal-zero failure.
The historical A2 engineering defect remains documented and OPEN.
Proposal availability is sufficient to proceed to reference-stage forensics.
```

Freeze:

```text
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
demo-policy proposal path = PASS
NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

Do not execute NEXT.

## B — `PROP01_LOCKED_DEMO_PROPOSAL_ZERO_OR_EMPTY`

Use if any candidate has:

```text
raw_proposal_count == 0
OR
merged_proposal_count == 0
```

Freeze:
- candidate remains locked;
- no replacement authorized;
- PROP-01 remains blocking for the supported-domain Demo path.

```text
NEXT = PROP01_LOCKED_CANDIDATE_FAILURE_DECISION
```

Do not execute NEXT.

## C — `PROP01_LOCKED_DEMO_REF_ELIGIBILITY_BLOCKED`

Use only if:
- all four have raw > 0 and merged > 0;
- at least one candidate has `eligible_largest_count == 0`.

Interpretation:

```text
proposal generation is present;
the blocker has moved to the frozen eligibility/reference interface.
```

Freeze:

```text
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Do not execute NEXT.

## D — `PROP01_LOCKED_DEMO_PROPOSAL_GATE_RUNTIME_FAILED`

Use for process/runtime/integrity failure unrelated to a zero proposal count.

```text
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_GATE_RECOVERY
```

No retries in this task.

# 15. No claim inflation

Regardless of outcome, do NOT state:
- four cases prove generalization;
- cross-city generalization;
- arbitrary aerial-image robustness;
- detector problem solved globally;
- historical A2 fixed;
- reference quality is good;
- final target segmentation succeeds;
- Demo is release-ready.

This is a proposal-availability gate only.

# 16. Report

Create:

```text
docs/task8b3_p1d12_locked_demo_proposal_gate.md
```

Required sections:

1. task / starting HEAD
2. external integrity gate
3. four locked candidate identities
4. frozen detector constants
5. clean first-run output gate
6. execution commands and exit codes
7. per-candidate raw/merged counts
8. per-candidate eligible_largest_count
9. inspect-only / no-core-chain proof
10. no visual / no GT / no candidate replacement confirmation
11. exact outcome
12. PROP-01 status
13. exact NEXT.

Include one compact table:

```text
candidate | sample_id | raw | merged | eligible | exit | inspect-only
```

Do not include qualitative visual judgments.

# 17. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D12
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 6651c4bf89663e5fff3e7854c9e2115d084ae003
External setup preflight: READY / NOT READY
External sync check: 135/135 PASS / FAIL
Locked raster identities: 4/4 PASS / other
Frozen detector constants: MATCH / other
Pre-existing candidate diagnostics: NONE / PRESENT
Candidate replacement: NO
Visual inspection: NO
Ground-truth access: NO
Qwen/ProgramHead execution: NONE
SAM2 execution: NONE
D-B1 execution: NONE
Reference selection execution: NONE
right raw/merged/eligible: <n>/<n>/<n>
left raw/merged/eligible: <n>/<n>/<n>
above raw/merged/eligible: <n>/<n>/<n>
below raw/merged/eligible: <n>/<n>/<n>
All four process exits: 0 / other
Core-chain artifacts produced: NO / YES
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d12_locked_demo_proposal_gate.md
Next action: Awaiting ChatGPT audit; do not select references or run full inference.
```

# 18. Commit / push

If the task reaches a classified outcome A/B/C:

```text
docs(rc1): record locked demo proposal gate
```

If runtime/integrity failure D or earlier STOP:

```text
docs(rc1): record locked demo proposal gate stop
```

Push current branch normally.
No force push.

# 19. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- canonical/external source/config/model assets are not modified;
- external integrity preflight passes;
- all four source raster hashes match immutable locks;
- detector constants match frozen values;
- no prior candidate diagnostics exist;
- each authorized candidate is run at most once;
- only `--inspect-proposals` path is used;
- no Qwen/SAM2/D-B1/reference selection/GT/manual visual evaluation;
- raw/merged/eligible counts are recorded;
- exact outcome enum is assigned;
- candidate replacement does not occur;
- PROP-01 remains historically OPEN;
- next gate is not executed;
- report/handoff committed and pushed;
- STOP.
