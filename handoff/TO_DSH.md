# TO_DSH — Task 8B.3-P1D10-R2: Audit BuildSpatialReason v0.2 and Lock Outcome-Independent Demo Candidates

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `a3a0d01f5f3f49f249c4a56b45d47fc42d6cbd48`

# 0. Audit disposition

P1D10-R1 corrected the circular A2 domain claim, but is NOT YET APPROVED.

Two remaining issues:

1. The R1 audit did not actually audit the canonical relation dataset requested by the task:
   `datasets/build_spatial_reason/v0.2/{val,test}.jsonl`.
   The tracked v0.2 manifest/statistics already establish:
   - 28,108 samples;
   - `scene_disjoint_v1`;
   - 20 programs;
   - explicit reference/target provenance;
   - all four frozen level-3 programs are present in val/test.

2. R1 selected `artifacts/task6m1_demo` as the primary pool, but its metadata includes prior model outputs
   (`parsed_program`, `status`, `proposal_count`, outputs/selection). A selection contract that requires a completed
   status or parsed model output is not cleanly outcome-independent.

This R2 must audit the canonical BuildSpatialReason v0.2 records and, if valid, lock the future Demo candidates
BEFORE any detector/manual result is observed.

This is DOCS-ONLY / METADATA-ONLY. No model or image-quality selection is allowed.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = a3a0d01f5f3f49f249c4a56b45d47fc42d6cbd48
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any detector/model/inference;
- run `predict.py`, `--inspect-proposals`, Qwen, SAM2, D-B1;
- run pytest/check_setup;
- train/fine-tune/export/download;
- modify runtime/tests/manifests/checkpoints/config;
- modify/copy/replace Demo inputs;
- inspect any detector output, proposal count, success/failure status, mask, overlay or manual visual quality when
  selecting candidates;
- use `task6m1_demo` / `task6m_demo` model outcome fields as eligibility criteria;
- change frozen research metrics;
- enter REF-01/MASK-01/Task 8B.4/8C;
- modify main;
- force push.

Reading JSON/JSONL/YAML/metadata and checking that referenced image files exist/read correctly is allowed.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do not commit copied images or a new dataset.

# 4. Canonical v0.2 provenance gate

Inspect and record:

```text
datasets/build_spatial_reason/v0.2/manifest.json
datasets/build_spatial_reason/v0.2/statistics.json
configs/build_spatial_reason_v0.2.yaml
scripts/task6l_build_v0_2.py
```

Require evidence consistent with:

```text
dataset = BuildSpatialReason v0.2
source = WHU Building Dataset / Satellite dataset II (East Asia)
source representation = whu-native-vector-v1.0
split_view = scene_disjoint_v1
total samples = 28108
train / val / test = 12778 / 9111 / 6219
program_count = 20
```

Record the four relevant query counts in both val and test:

```text
largest_to_right_of_to_nearest
largest_to_left_of_to_nearest
largest_to_above_to_nearest
largest_to_below_to_nearest
```

If the canonical evidence disagrees materially:
- do not lock candidates;
- STOP.

# 5. Locate actual v0.2 sample records

Require local availability of:

```text
datasets/build_spatial_reason/v0.2/val.jsonl
datasets/build_spatial_reason/v0.2/test.jsonl
```

These files may be intentionally untracked large artifacts; local read-only use is allowed.

Record for each:

```text
exists
bytes
SHA256
line count
JSON parse failures
```

Required line counts:

```text
val = 9111
test = 6219
```

If `test.jsonl` is absent, corrupt, or count-mismatched:
- no candidate lock;
- `REPLACEMENT_SELECTION_POLICY_NOT_READY`;
- document and STOP after commit/push.

Do NOT regenerate the dataset in this task.

# 6. Establish v0.2 record schema

From metadata only, record the field names and types necessary for selection.

At minimum establish whether records contain:

```text
split
image_id
query_type
level
target_component_id
reference_component_ids
target_mask
trivial_selection
native_vector.dataset_name
native_vector.dataset_version
native_vector.tile_id
native_vector.target
native_vector.references
target_geometry_ref
image metadata/path reference
```

For one record of each of the four target query types from the TEST split, record only:
- stable identity fields;
- program/query fields;
- reference/target metadata fields;
- image-path resolution fields.

Do NOT record or consult detector/model results.

# 7. Resolve source images without model output

Using the v0.2 record and its frozen reasoning-view/native-vector metadata, prove that the underlying source tile can
be resolved deterministically.

For the TEST split record pool establish:

```text
image root / metadata path
image dimensions
image format
readable count for candidate-bearing images
missing/unreadable count
```

Do not inspect visual quality.

A candidate is eligible only if its referenced source image exists and decodes successfully.

# 8. Historical-use / leakage audit

Audit documentation for historical use of:

```text
BuildSpatialReason v0.2 val
BuildSpatialReason v0.2 test
```

Record whether each was used for:

```text
training
model/architecture/threshold selection
validation
final frozen Task7J evaluation
post-test rescue/tuning
```

Choose exactly one per split:

```text
CLEAN_DEMO_SELECTION_POOL
USABLE_WITH_DISCLOSURE
MODEL_SELECTION_LEAKAGE_RISK
POOL_STATUS_INCOMPLETE
```

Expected decision logic:
- a split used for model selection cannot be called clean;
- a frozen final-test split may be `USABLE_WITH_DISCLOSURE` for qualitative Demo selection only if candidate locking is
  metadata-only, deterministic, and does not change any reported test result.

# 9. Reclassify historical demo-output pools

For:

```text
artifacts/task6m1_demo
artifacts/task6m_demo
```

record whether eligibility fields such as these exist:

```text
parsed_program
status
proposal_count
selected
outputs
```

Classify each exactly as one:

```text
MODEL_OUTPUT_CONTAMINATED_POOL
METADATA_ONLY_POOL
POOL_STATUS_INCOMPLETE
```

These pools MUST NOT be the primary replacement source if model-output-contaminated.

# 10. Preferred pool

Choose the preferred future success-Demo pool using this priority:

1. BuildSpatialReason v0.2 TEST if `CLEAN_DEMO_SELECTION_POOL` or `USABLE_WITH_DISCLOSURE`;
2. BuildSpatialReason v0.2 VAL only if test is unusable and val is not outcome-contaminated;
3. otherwise no pool.

Record:

```text
BEST_POOL = <exact path/split or NONE>
BEST_POOL_CLASSIFICATION = <enum>
```

No candidate may be chosen from a model-output-contaminated demo bundle.

# 11. Exact metadata-only candidate-lock rule

If BEST_POOL exists, use exactly this relation order:

```text
1. largest_to_right_of_to_nearest
2. largest_to_left_of_to_nearest
3. largest_to_above_to_nearest
4. largest_to_below_to_nearest
```

Eligibility uses ONLY canonical ground-truth/generator metadata:

```text
split == chosen split
query_type == required query
level == 3
target_component_id is present
reference_component_ids is non-empty
native_vector.target is present
native_vector.references are present
target_geometry_ref is present
referenced source image exists and decodes
```

Forbidden eligibility fields:

```text
detector output
proposal count
parser output
runtime status
predicted mask
manual visual quality
historical Demo success/failure
confidence
```

Do NOT require a prior model `status == completed`.

# 12. Deterministic ordering

Prefer an existing immutable `sample_id` if present.

If no `sample_id` exists, define canonical sort key exactly as:

```text
(
  str(image_id),
  str(query_type),
  int(target_component_id),
  tuple(int(x) for x in reference_component_ids)
)
```

For each relation in the fixed order:
- sort all eligible records by the canonical key;
- choose the first record whose `image_id` has not already been locked for another relation.

No retries based on model outcome are ever allowed.

If a relation has no eligible distinct-image record:
- do not loosen criteria;
- selection policy = NOT READY;
- STOP.

# 13. Lock exactly four candidate records

Metadata-only candidate locking IS AUTHORIZED in this R2.

For each locked candidate record:

```text
relation/program
stable key / sample_id
split
image_id
resolved source image path
image bytes
image SHA256
dimensions
format
target_component_id
reference_component_ids
native-vector target identity
native-vector reference identities
trivial_selection
```

Do NOT:
- copy it into RC1;
- run the detector;
- inspect prediction outputs;
- inspect manual visual quality.

Once recorded and committed, these four candidate IDs are frozen for the next gate. They may not be replaced after
future runtime results without a new ChatGPT decision that explicitly acknowledges the failed locked candidate.

# 14. Support-scope wording

Define support positively from canonical provenance, not from A2 failure.

Allowed scope basis:

```text
WHU Building Dataset — Satellite dataset II (East Asia)
native-vector building instances
scene-disjoint_v1
tile-relative spatial reasoning
```

Explicit limitation from v0.2 manifest:

```text
scene separation does not establish cross-city or broad geographic generalization
```

A2 statement must remain:

```text
A2 provenance/domain = NOT ESTABLISHED
A2 = documented persistent non-detection/stress case
```

Do NOT call A2 proven out-of-domain.

# 15. Re-evaluate policy decision

Choose exactly:

```text
DEMO_POLICY_PATH_READY
DEMO_POLICY_PATH_NOT_READY
DEMO_POLICY_STATUS_INCOMPLETE
```

`READY` requires:
- canonical relation metadata;
- a non-output-contaminated usable pool;
- four locked candidates selected only by §11–§12;
- positive support scope;
- disclosure of test reuse if TEST is chosen.

Then choose:

```text
PROP01_RESOLUTION_DEMO_POLICY
PROP01_RESOLUTION_BLOCKED
```

Detector adaptation remains `DETECTOR_ADAPTATION_NOT_READY`.

PROP status remains:

```text
PROP01_OPEN_ENGINEERING_DEFECT
```

Do NOT mark it closed or reclassified as out-of-domain in this task.

# 16. Release wording

Write ≤120 Chinese characters.

Must state:
- positive verified support scope;
- A2 is a documented non-detection case with unknown provenance;
- no arbitrary aerial-image robustness claim.

If TEST is the chosen Demo pool, also add a separate disclosure sentence in the report:

```text
The qualitative Demo cases are deterministically selected from the frozen v0.2 test split after final test metrics
were already consumed; they do not alter or replace the reported Task7J metrics.
```

# 17. Next gate

If ready:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION
```

If not ready:

```text
NEXT = PROP01_DECISION_EVIDENCE_RECOVERY
```

Do not execute it.

# 18. Report update

Append `P1D10-R2` to:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
```

Required sections:

1. ChatGPT R2 audit reason
2. canonical v0.2 provenance
3. val/test JSONL identity/counts
4. sample schema
5. source-image resolution
6. val/test historical-use classifications
7. historical demo-pool contamination classifications
8. chosen BEST_POOL
9. exact metadata-only selection rule
10. four locked candidate records
11. positive support scope and limitation
12. Demo-policy feasibility
13. primary resolution
14. PROP-01 status
15. corrected release wording
16. test-reuse disclosure if applicable
17. exact next gate
18. no model/test execution
19. no functional modification.

Mark the R1 `task6m1_demo` primary-pool statement as SUPERSEDED if it is model-output-contaminated.

# 19. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D10-R2
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: a3a0d01f5f3f49f249c4a56b45d47fc42d6cbd48
Model/test execution: NONE
Functional files modified: NO
v0.2 test JSONL: AVAILABLE / UNAVAILABLE
v0.2 test records: <n>
v0.2 test classification: <enum>
v0.2 val classification: <enum>
task6m1_demo classification: <enum>
task6m_demo classification: <enum>
Best Demo pool: <path/split / NONE>
Replacement selection policy: READY / NOT READY
Locked candidate count: 4 / other
Locked right candidate: <stable id / NONE>
Locked left candidate: <stable id / NONE>
Locked above candidate: <stable id / NONE>
Locked below candidate: <stable id / NONE>
A2 domain classification: NOT ESTABLISHED
Detector adaptation feasibility: DETECTOR_ADAPTATION_NOT_READY
Demo policy feasibility: <enum>
Primary resolution: <enum>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Scientific freeze preserved: YES / NO / NOT ESTABLISHED
Next gate: <enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d10_prop01_resolution_decision.md
Next action: Awaiting ChatGPT audit; locked candidates must not be run or replaced.
```

# 20. Commit / push

If COMPLETE:

```text
docs(rc1): lock outcome-independent prop01 demo pool
```

If STOP/FAILED:

```text
docs(rc1): record prop01 demo-pool audit stop
```

Push normally. No force push.

# 21. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test execution;
- no functional modification;
- canonical v0.2 val/test are actually audited;
- actual test records are present and count-correct;
- source images resolve;
- historical split use is classified;
- model-output-contaminated historical demo pools are not used as primary pool;
- selection eligibility contains no model/manual outcome field;
- exactly four candidates are locked before any model execution;
- support scope is positive and evidence-based;
- A2 remains NOT ESTABLISHED rather than declared out-of-domain;
- report/handoff committed/pushed;
- STOP.
