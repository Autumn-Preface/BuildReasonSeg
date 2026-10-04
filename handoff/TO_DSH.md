# TO_DSH — Task 8B.3-P1D11B-R2: Execute the Authorized External Git-Canonical Migration

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `1f5641a0c0a1f43956f55da8f6a28262d52ee574`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. ChatGPT audit decision

P1D11B-R1 STOP is accepted.

R1 satisfied the task-book pre-sync gate:

```text
checked = 135
match = 97
missing = 0
mismatch = 38
```

ChatGPT independently verified that the 38 mismatches are exactly the P1D11M1 normalized-entry set:

```text
README.md
buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md
buildreasonseg/runtime/_frozen/__init__.py
buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py
buildreasonseg/runtime/_frozen/mvp/grcl_directional.py
buildreasonseg/runtime/_frozen/mvp/native_vector_adapter.py
buildreasonseg/runtime/_frozen/mvp/task6m_eval.py
buildreasonseg/runtime/_frozen/mvp/task6m_structured.py
buildreasonseg/runtime/_frozen/mvp/task6n_relation_decoder.py
buildreasonseg/runtime/_frozen/mvp/task6p_reference_head.py
buildreasonseg/runtime/_frozen/mvp/task6q_reference_resolver.py
buildreasonseg/runtime/_frozen/mvp/task6s_directional_pipeline.py
buildreasonseg/runtime/_frozen/mvp/task6u_common.py
buildreasonseg/runtime/_frozen/mvp/task6v_family_reference_resolver.py
buildreasonseg/runtime/_frozen/mvp/task6w_quality_reference_resolver.py
buildreasonseg/runtime/_frozen/mvp/task6x_sam2_reference_refiner.py
buildreasonseg/runtime/_frozen/mvp/task6z_field_composition.py
buildreasonseg/runtime/_frozen/mvp/task6z_l3_decoder.py
buildreasonseg/runtime/_frozen/mvp/task7a_l3_pipeline.py
buildreasonseg/runtime/_frozen/mvp/task7d_data.py
buildreasonseg/runtime/_frozen/mvp/task7e_l3_decoder_adapter.py
buildreasonseg/runtime/_frozen/mvp/whu_vector_audit.py
buildreasonseg/runtime/pipeline.py
check_setup.py
docs/command_grammar.md
docs/model_card.md
docs/runtime_mapping.md
environment.yml
model/buildreasonseg_advisor/metadata.json
model/buildreasonseg_advisor/model.yaml
model/components/program_head/qwen_asset_manifest.json
predict.py
requirements.txt
tests/test_cli_contract.py
tests/test_language_contract.py
tests/test_model_package.py
tests/test_setup_checker.py
tests/test_task8b1_fallback_ux.py
```

The four paths below are NOT mismatches and MUST NOT be added to the migration gate:

```text
buildreasonseg/runtime/core.py
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

They already match Git-canonical bytes in external delivery.

## Critical execution rule

Do NOT invent any additional CRLF/worktree-derived gate.

The only authorized pre-sync content gate is:

```text
97 match / 38 mismatch
AND exact mismatch path set above
```

If that exact gate passes, proceed.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 1f5641a0c0a1f43956f55da8f6a28262d52ee574
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify canonical RC1;
- modify canonical source_manifest.json;
- modify sync helper or its tests;
- run pytest;
- run predict.py;
- run detector/Qwen/SAM2/D-B1 inference;
- run/copy/inspect locked Demo candidates;
- replace any locked candidate;
- modify A1/A2/A3/A4/B1/B2;
- change thresholds/models/tiling/merge/reference policy;
- delete external files/directories;
- modify model weights/checkpoints;
- modify main;
- force push;
- add any extra pre-sync gate not explicitly written here.

# 3. Allowed tracked repository changes

Only:

```text
docs/task8b3_p1d11b_git_canonical_external_sync.md
docs/task8b3_p1d11b_r2_external_git_identity_migration.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

External RC1 is intentionally modified by the controlled sync in this task.

# 4. Canonical Git-identity preflight

Require canonical manifest:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

Verify every manifest entry against exact binary Git HEAD bytes:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

Require:

```text
canonical Git identity = 135/135 PASS
missing = 0
duplicate paths = 0
```

Record exact SHA256 of:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Expected historical value from R1:

```text
c72c8ed88b9b62ec49ca7f29f44a2a2b68af820a27b29441a5e7876c9aab9604
```

If actual Git blob SHA differs from this historical value, do not assume failure solely from that.
Require only that the value is recorded and the 135-entry Git-identity gate passes.

# 5. Pre-sync helper gate — ONLY authorized mismatch gate

Run exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Require:

```text
checked = 135
match = 97
missing = 0
mismatch = 38
```

Require exact mismatch path set = the 38 paths listed in §0.

Do NOT compare this set against:
- current working-tree CRLF-expanded paths;
- all paths where worktree differs from Git;
- any derived EOL set;
- any other self-created set.

If 97/38 and the exact §0 path set hold:

```text
PRE_SYNC_GATE = PASS
```

and proceed.

Otherwise STOP.

# 6. Pre-sync external setup

Use:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe
```

From external RC1 root run exactly:

```text
python check_setup.py
```

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

If not READY:
- do not sync;
- STOP.

No inference.

# 7. Protected external snapshot

Before sync record existence, bytes and SHA256 for:

```text
model/buildreasonseg_advisor/decoder.pt
model/buildreasonseg_advisor/detector.pt
model/components/sam2/sam2.1_hiera_base_plus.pt
model/components/sam2/sam2.1_hiera_b+.yaml
model/components/program_head/program_parser_l3_rehearsal_v1.pt
```

For:

```text
model/components/program_head/Qwen3-VL-2B-Instruct
```

record:

```text
exists
recursive regular-file count
sorted relative file names
total bytes
```

For protected user/runtime directories:

```text
inference/input
inference/output
logs
runs
datasets
```

record:

```text
exists
recursive regular-file count
total bytes
```

`qwen_integrity_cache.json`, if created/updated by `check_setup.py`, is generated cache state and is NOT a model-weight
identity. Record it separately if it changes; do not confuse it with Qwen downloaded model files.

# 8. Authorized one-time 135-file write sync

Run exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

No second attempt.

Require:

```text
copied = 135
verified = 135
failures = 0
exit code = 0
```

The helper is already approved to source Git HEAD canonical blob bytes for the real RC1 manifest.

If sync fails:
- STOP;
- do not rerun automatically.

# 9. Separate source_manifest.json synchronization

After §8 succeeds, overwrite exactly:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

with exact binary bytes from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Requirements:

```text
binary capture
binary write
no JSON regeneration
no text newline conversion
no working-tree source
```

Require:

```text
external source_manifest SHA256 == canonical Git source_manifest SHA256
```

# 10. Post-sync helper gate

Run exactly once:

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

Also require:

```text
external source_manifest Git SHA match = YES
```

# 11. External policy verification

Read only external:

```text
README.md
docs/model_card.md
source_manifest.json
```

Require README meaning:

```text
software input modality is distinguished from verified domain
"正式输入域" is absent
"已验证数据域与 Demo 边界" is present
A2 provenance/domain = NOT ESTABLISHED
A2 = persistent non-detection/stress case
locked candidates = NOT YET RUN
```

Require model card meaning:

```text
contains software-input-modality wording
contains separate verified-domain wording
A2 provenance/domain = NOT ESTABLISHED
contains Demo policy
locked candidates not claimed successful
```

Require external source manifest:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

Do not inspect candidate images.

# 12. Protected post-sync comparison

Repeat §7 snapshot.

Require:

```text
decoder.pt unchanged = YES
detector.pt unchanged = YES
SAM2 checkpoint unchanged = YES
SAM2 YAML unchanged = YES
ProgramHead checkpoint unchanged = YES
Qwen downloaded asset file names/count/total bytes unchanged = YES
inference/input unchanged = YES
inference/output unchanged = YES
logs unchanged except explicitly documented setup-generated log/cache behavior = YES
runs unchanged = YES
datasets unchanged = YES
```

If a protected model/downloaded asset changed:
- STOP;
- no ad-hoc repair.

Generated `qwen_integrity_cache.json` may be recorded separately and does not by itself fail this gate.

# 13. Post-sync external setup

Run exactly:

```text
python check_setup.py
```

from external RC1 using the same environment Python.

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

Do not run pytest or inference.

# 14. Scientific/product state

If all gates pass, record:

```text
external source/config identity = GIT_CANONICAL 135/135
supported-domain policy external sync = COMPLETE
A2 domain = NOT ESTABLISHED
A2 = DOCUMENTED_PERSISTENT_NON_DETECTION
locked candidates = FINAL_METADATA_LOCK
locked candidate runtime status = NOT YET RUN
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
REF-01 = OPEN
MASK-01 = OPEN
scientific freeze = PRESERVED
```

This task does NOT claim PROP-01 fixed.

# 15. Outcome

Choose exactly one.

If all gates pass:

```text
Outcome = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_COMPLETE
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Otherwise:

```text
Outcome = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_STOP
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_RECOVERY
```

Do not execute NEXT.

# 16. Reports

Create:

```text
docs/task8b3_p1d11b_r2_external_git_identity_migration.md
```

Append a short R2 section to existing:

```text
docs/task8b3_p1d11b_git_canonical_external_sync.md
```

Preserve R1 STOP history.

Required report facts:

1. starting HEAD
2. R1 self-invented cross-check explicitly superseded
3. canonical Git 135/135
4. PRE_SYNC_GATE exact 97/38 + exact path-set PASS
5. pre-sync setup
6. protected pre-snapshot
7. one write-sync invocation result
8. separate source_manifest copy result
9. post-sync 135/135
10. external policy verification
11. protected post comparison
12. post-sync setup
13. pytest/model/inference/locked candidates = NOT RUN
14. PROP/REF/MASK states
15. exact outcome
16. exact NEXT.

# 17. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11B-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 1f5641a0c0a1f43956f55da8f6a28262d52ee574
Model/inference execution: NONE
Pytest: NOT RUN
Canonical RC1 modified: NO
External delivery modified: YES — CONTROLLED GIT-CANONICAL MIGRATION/SYNC ONLY / NO
Canonical Git manifest: 135/135 PASS / FAIL
Pre-sync helper gate: 97/135; exact authorized 38 mismatch paths / other
Self-invented CRLF cross-check: NOT USED
Pre-sync external setup: READY / NOT READY
Protected pre-snapshot: RECORDED / NOT RECORDED
Write sync invocations: 1 / other
Manifest-listed files copied/verified: 135/135 / other
External source_manifest Git SHA match: YES / NO / NOT RUN
Post-sync external check: 135/135 PASS / FAIL / NOT RUN
External README policy: VERIFIED / NOT VERIFIED
External model-card policy: VERIFIED / NOT VERIFIED
Protected model/component assets changed: NO / YES / NOT CHECKED
Qwen downloaded asset directory changed: NO / YES / NOT CHECKED
Protected runtime/user dirs changed: NO / YES / NOT CHECKED
Post-sync external setup: READY / NOT READY / NOT RUN
A2 domain classification: NOT ESTABLISHED
A2 policy status: DOCUMENTED_PERSISTENT_NON_DETECTION
Locked candidate status: FINAL_METADATA_LOCK
Locked candidate runtime status: NOT YET RUN
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: OPEN
MASK-01 status: OPEN
Scientific freeze preserved: YES / NO
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d11b_r2_external_git_identity_migration.md
Next action: Awaiting ChatGPT audit; do not run locked candidates.
```

# 18. Commit / push

If COMPLETE:

```text
docs(rc1): record completed external git identity sync
```

If STOP/FAILED:

```text
docs(rc1): record external git identity sync stop
```

Push normally.
No force push.

# 19. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no canonical RC1 modification;
- exact authorized 97/38 pre-sync gate passes;
- NO additional CRLF/worktree-derived gate is used;
- pre-sync setup READY;
- protected pre-snapshot recorded;
- exactly one helper write sync invocation;
- copied/verified = 135/135;
- source_manifest separately synchronized from Git blob bytes;
- post-sync helper check = 135/135;
- external policy spot-check passes;
- protected model/Qwen/runtime-user state preserved;
- post-sync setup READY;
- no pytest/model/inference/locked candidate run;
- PROP-01 remains OPEN;
- next gate not executed;
- report/handoff committed and pushed;
- STOP.
