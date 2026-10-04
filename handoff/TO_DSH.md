# TO_DSH — Task 8B.3-P1D11A-R1: Line-Ending-Safe Manifest Gate and Supported-Domain Policy

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `9f91d953b9ef5bb3993be4b7434b174de9856648`

# 0. ChatGPT audit correction

P1D11A correctly STOPPED because its written preflight failed.

However ChatGPT independently audited the remote Git tree and found that the failure is very likely a
**working-tree line-ending false positive**, not stale canonical content.

P1D11A local working-tree report:

```text
path                                      manifest/Git bytes    local bytes    delta
runtime/core.py                           11064                 11310          +246
runtime/detector.py                       20300                 20772          +472
runtime/outputs.py                        9166                  9367           +201
tests/test_task8b_runtime.py              24983                 25552          +569
```

Remote Git at current branch still reports exactly:

```text
runtime/core.py              11064
runtime/detector.py          20300
runtime/outputs.py           9166
tests/test_task8b_runtime.py 24983
```

and the first three current Git blob IDs are identical to their blobs at commit
`c432ec41f2d8bad9ddc67d65d0dbc033724e482a`, where `source_manifest.json` was updated for the compact-proposal
implementation.

Therefore:

```text
DO NOT update the four runtime/test manifest entries merely to match CRLF-expanded working-tree bytes.
```

This task must first prove the line-ending diagnosis using Git canonical/index bytes.

If proven, continue the original supported-domain policy implementation.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 9f91d953b9ef5bb3993be4b7434b174de9856648
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
- modify runtime implementation files;
- modify runtime test files;
- change the four compact-proposal manifest entries unless this task proves the Git canonical bytes themselves
  disagree with the manifest;
- normalize/renormalize the entire repository;
- run `git add --renormalize .`;
- change global/local Git configuration;
- copy/run/inspect locked Demo candidates;
- replace any locked candidate;
- modify external delivery;
- run external write sync;
- mark PROP-01 CLOSED;
- call A2 proven out-of-domain;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

# 3. Allowed tracked changes

Only:

```text
docs/task8b3_p1d11a_policy_implementation.md
docs/task8b3_p1d11_supported_domain_policy.md
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked file may change.

# 4. Diagnose the four P1D11A false mismatches

For exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

record:

```text
git status --porcelain for path
manifest bytes
manifest sha256
HEAD/index canonical bytes
HEAD/index canonical sha256
working-tree bytes
working-tree sha256
HEAD LF count
HEAD CRLF count
working-tree LF count
working-tree CRLF count
```

Canonical bytes MUST be obtained from Git, not from the checkout representation, e.g. by capturing:

```text
git show HEAD:<repo-relative-path>
```

or, after staging later in the task:

```text
git show :<repo-relative-path>
```

Important:
- use binary stdout capture;
- hash those exact bytes;
- do not allow shell text decoding/newline conversion before hashing.

Classify each exactly:

```text
GIT_CANONICAL_MATCH_WORKTREE_EOL_ONLY
GIT_CANONICAL_MANIFEST_MISMATCH
OTHER_MISMATCH
```

`GIT_CANONICAL_MATCH_WORKTREE_EOL_ONLY` requires:
- Git canonical bytes/hash match manifest exactly;
- the tracked path has no semantic `git diff`;
- working-tree mismatch is fully explained by line-ending representation.

If ANY of the four is not `GIT_CANONICAL_MATCH_WORKTREE_EOL_ONLY`:
- no policy edits;
- record STOP;
- push;
- STOP.

# 5. Authoritative 135-entry preflight — Git canonical bytes

Do NOT use direct `Path.read_bytes()` on the Windows checkout as the authoritative manifest check.

Parse the current manifest and for all 135 entries compute bytes/SHA256 from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<manifest path>
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
manifest entries = 135
Git-canonical manifest self-check = 135/135 PASS
README.md entry count = 1
docs/model_card.md entry count = 1
```

If not 135/135:
- STOP;
- no policy edits.

Append a correction to:

```text
docs/task8b3_p1d11a_policy_implementation.md
```

stating that the original P1D11A STOP was caused by checkout representation if and only if this gate proves it.

# 6. Frozen policy decision

Retain:

```text
primary resolution =
PROP01_RESOLUTION_DEMO_POLICY

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

A2 provenance/domain =
NOT ESTABLISHED

A2 =
documented persistent non-detection/stress case

locked candidate status =
FINAL_METADATA_LOCK

locked candidate runtime status =
NOT YET RUN
```

Exact four candidate IDs remain immutable:

```text
right =
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91

left =
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3

above =
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

below =
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
```

# 7. Create authoritative policy document

Create:

```text
docs/task8b3_p1d11_supported_domain_policy.md
```

Required policy:

## Verified research/evaluation domain

```text
BuildSpatialReason v0.2
over WHU-EA-NativeVector v1.0
source = WHU Building Dataset — Satellite dataset II (East Asia)
split view = scene_disjoint_v1
modality = RGB optical overhead/aerial imagery
instance concept = WHU native-vector building instances
reasoning = tile-relative spatial reasoning
formal RC1 programs =
  largest -> left_of  -> nearest
  largest -> right_of -> nearest
  largest -> above    -> nearest
  largest -> below    -> nearest
```

Explicitly state:

```text
software format/size acceptance != demonstrated generalization domain
scene-disjoint split != proof of cross-city generalization
```

## Non-claims

```text
cross-city generalization = NOT ESTABLISHED
broad geographic generalization = NOT ESTABLISHED
cross-sensor generalization = NOT ESTABLISHED
arbitrary aerial-image robustness = NOT ESTABLISHED
unrestricted natural language = NOT ESTABLISHED
SAR / infrared / raw multispectral = NOT SUPPORTED
```

## A2

State:
- provenance/domain NOT ESTABLISHED;
- fixed historical persistent non-detection/stress case;
- remains in Task 8B.3 historical six-case evidence;
- P1D1-P1D9 did not fix it;
- MUST NOT be called proven out-of-domain;
- MUST NOT be silently rewritten as success.

Bounded evidence only:
- active YOLO26 tiled/full-frame zero;
- active YOLO26 diagnostic conf=0.001 zero;
- epoch-18 YOLO26 zero;
- independent WHU YOLOv8m zero;
- one predeclared validation-moment affine rescue zero.

## Demo sets

Separate:

```text
Historical diagnostic suite:
A1/A2/A3/A4/B1/B2

Locked supported-domain qualitative candidates:
the four exact v0.2 TEST sample IDs above
```

Locked cases:
- deterministic metadata-only selection;
- selected after Task7J final metrics consumed;
- no detector/parser/runtime/manual visual outcome used;
- immutable without new ChatGPT decision;
- NOT YET RUN;
- NOT yet claimed successful.

## Required English disclosure

Include exactly:

```text
The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.
```

Also provide faithful Chinese equivalent.

# 8. Update canonical README

Modify only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
```

Required:
- remove/replace the ambiguous `RGB 光学遥感影像（正式输入域）`;
- distinguish software-readable modality from verified evaluation domain;
- add `### 已验证数据域与 Demo 边界`;
- state WHU East Asia / v0.2 / native-vector / scene_disjoint_v1;
- state scene split != cross-city evidence;
- state arbitrary aerial input success is not guaranteed;
- state A2 is persistent non-detection with provenance unknown, not proven out-of-domain;
- preserve historical A1–B2 audit;
- list/point to the four locked TEST candidates as NOT YET RUN;
- disclose qualitative TEST reuse;
- do not change runtime commands/algorithm/metrics.

# 9. Update canonical model card

Modify only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Required:
- clarify verified WHU East Asia / native-vector / scene_disjoint_v1 / tile-relative domain;
- preserve four formal L3 programs;
- add limitations: scene split != cross-city proof, arbitrary aerial/cross-sensor robustness not established;
- A2 persistent non-detection, provenance NOT ESTABLISHED, not proven out-of-domain;
- add Demo-policy subsection distinguishing historical six-case audit vs four locked not-yet-run TEST candidates;
- include TEST reuse disclosure;
- no frozen metric value may change.

# 10. Stage target documents before computing manifest hashes

Because this checkout may expand LF→CRLF in the working tree, manifest identities must be calculated from the
**Git index representation**.

After editing README/model_card:

Stage ONLY:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Then obtain canonical bytes via:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/README.md
git show :delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Compute exact SHA256/bytes from those binary byte streams.

Do NOT derive manifest values from direct Windows working-tree `Path.read_bytes()`.

# 11. Update source_manifest narrowly

Update exactly:

```text
README.md
docs/model_card.md
```

with the Git-index canonical:

```text
bytes
sha256
```

Do NOT modify the four runtime/test entries from §4.
Do NOT modify any other entry.
Do NOT add/remove paths.

Stage:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

# 12. Post-edit canonical manifest check — INDEX aware

Parse the STAGED manifest bytes from:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

For every one of its 135 entries, hash canonical source bytes from:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/<manifest path>
```

Require:

```text
staged/index canonical manifest check = 135/135 PASS
entry count = 135
```

This is the authoritative post-edit gate.

Also require:
- the four runtime/test entries retain their original manifest bytes/SHA256;
- only README/model_card manifest records differ from starting HEAD.

If not:
- STOP;
- do not write external.

# 13. External boundary

External delivery remains untouched.

Do NOT run write sync.

A read-only check is allowed:

```text
python scripts/sync_advisor_rc1_delivery.py
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
  --check
```

Because this script compares working-tree canonical vs external, interpret line endings carefully.

Expected semantic content drift from this task is:
- README.md;
- docs/model_card.md;
- canonical `source_manifest.json` itself is intentionally not an entry copied by the sync helper.

If the check reports unrelated runtime/test mismatches, determine whether they are merely the same EOL checkout
representation issue before drawing any conclusion.

No external write.

# 14. Stage remaining authorized files and diff gate

Update/stage:

```text
docs/task8b3_p1d11a_policy_implementation.md
docs/task8b3_p1d11_supported_domain_policy.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Require final staged/commit diff contains ONLY the seven authorized paths listed in §3.

Explicitly require NO diff in:
- runtime files;
- tests;
- configs;
- model package;
- locked inputs.

# 15. Final status

If COMPLETE:

```text
P1D11A original STOP root cause =
WORKTREE_EOL_FALSE_POSITIVE

Git canonical preflight =
135/135 PASS

supported-domain policy =
IMPLEMENTED_IN_CANONICAL_RC1_DOCS

A2 domain =
NOT ESTABLISHED

A2 policy =
DOCUMENTED_PERSISTENT_NON_DETECTION

locked candidate status =
FINAL_METADATA_LOCK

locked candidate runtime status =
NOT YET RUN

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

scientific freeze =
PRESERVED

NEXT =
PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Do NOT execute next gate.

# 16. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11A-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 9f91d953b9ef5bb3993be4b7434b174de9856648
Model/test execution: NONE
Functional runtime files modified: NO
External delivery modified: NO
Original P1D11A STOP root cause: WORKTREE_EOL_FALSE_POSITIVE / REAL_MANIFEST_DRIFT / OTHER
Four flagged Git-canonical entries: 4/4 MATCH / other
Git-canonical pre-edit manifest: 135/135 PASS / FAIL
Post-edit staged/index manifest: 135/135 PASS / FAIL / NOT RUN
Manifest entry count: 135 / other
Runtime/test manifest entries modified: NO
README manifest entry updated: YES / NO
model_card manifest entry updated: YES / NO
Canonical policy doc: docs/task8b3_p1d11_supported_domain_policy.md
Canonical README policy: UPDATED / NOT UPDATED
Canonical model card policy: UPDATED / NOT UPDATED
A2 domain classification: NOT ESTABLISHED
A2 policy status: DOCUMENTED_PERSISTENT_NON_DETECTION
Locked candidate status: FINAL_METADATA_LOCK
Locked candidate runtime status: NOT YET RUN
Supported-domain policy: IMPLEMENTED_IN_CANONICAL_RC1_DOCS / NOT IMPLEMENTED
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Scientific freeze preserved: YES / NO
External read-only sync check: <result / NOT RUN>
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Next action: Awaiting ChatGPT audit; do not run candidates or sync external.
```

# 17. Commit / push

If COMPLETE:

```text
docs(rc1): implement prop01 supported-domain policy
```

If STOP/FAILED:

```text
docs(rc1): record prop01 policy recovery stop
```

Push normally.
No force push.

# 18. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test execution;
- no functional/runtime/test modification;
- four false mismatches are proven EOL-only against Git canonical bytes;
- Git canonical preflight = 135/135;
- policy document is created;
- README/model_card are evidence-bounded;
- only their two manifest entries change;
- post-edit INDEX-aware manifest check = 135/135;
- A2 remains provenance/domain NOT ESTABLISHED;
- four locks remain immutable and NOT YET RUN;
- external delivery untouched;
- PROP-01 remains OPEN;
- next gate not executed;
- only authorized paths committed;
- commit/push succeeds;
- STOP.
