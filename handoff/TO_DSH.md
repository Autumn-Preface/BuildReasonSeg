# TO_DSH — Task 8B.3-P1D11B-R1: Controlled External Git-Canonical Migration and Policy Sync

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `894163a082e332de377bab8635e4257bfb956326`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. ChatGPT audit decision

P1D11S1-R3 is approved as the sync-helper implementation.

Frozen helper evidence:

```text
STATIC_PATCH_GATE = PASS
PATCH_DIFF_GATE = PASS
dedicated pytest = 28 passed
helper source basis = GIT_CANONICAL_BLOB_BYTES
legacy no-basis behavior = preserved
manifest identity mismatch blocks write = tested
unsupported basis = tested
canonical RC1 modified by helper task = NO
external RC1 modified by helper task = NO
```

The R3 task itself STOPPED because the real external read-only check was:

```text
checked = 135
match = 97
missing = 0
mismatch = 38
```

ChatGPT independently compared those 38 mismatch paths with the exact 38 identities normalized in P1D11M1.

They match exactly.

Therefore:

```text
97/38 is the EXPECTED pre-migration state.

The 38 mismatches are not new source drift.
They are the external delivery's historical CRLF-era copies of the exact 38 entries
whose manifest identities were migrated from working-tree bytes to Git-canonical bytes.

No further sync-helper redesign is authorized.
```

This task performs the one-time controlled migration of all 135 manifest-listed external source/config files to the
now-authoritative Git-canonical bytes, while also transferring the already-approved README/model-card policy updates.

No model inference or locked Demo candidate execution is allowed.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 894163a082e332de377bab8635e4257bfb956326
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify canonical RC1 content;
- modify canonical `source_manifest.json`;
- modify sync-helper code/tests;
- modify model weights/checkpoints;
- delete any external file or directory;
- modify unlisted external runtime/user content;
- run pytest;
- run `predict.py`;
- run detector/Qwen/SAM2/D-B1 inference;
- run/copy/inspect the four locked Demo candidates;
- modify or replace A1/A2/A3/A4/B1/B2;
- change thresholds/models/tiling/merge/reference policies;
- fix REF-01 or MASK-01;
- enter Task 8B.4 / 8C;
- update main;
- force push.

`check_setup.py` is authorized before and after sync.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d11b_r1_external_git_identity_migration.md
docs/task8b3_p1d11b_supported_domain_policy_external_sync.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

The external RC1 delivery is intentionally modified only through the controlled sync defined below.

# 4. Canonical integrity preflight

Read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

For all 135 entries, verify manifest identity against exact Git HEAD canonical bytes:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

Require:

```text
canonical Git identity = 135/135 PASS
missing paths = 0
duplicate paths = 0
```

Also compute and record the SHA256 of the exact Git blob bytes for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

If any canonical gate fails:
- do not write external;
- STOP.

# 5. Reproduce and classify the pre-sync external state

Run exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Require exactly:

```text
checked = 135
match = 97
missing = 0
mismatch = 38
```

The exact 38 mismatch paths MUST equal the P1D11M1 normalized-entry set below, with no extra and no missing path:

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

Classify:

```text
PRE_SYNC_EXTERNAL_STATE =
EXPECTED_GIT_IDENTITY_MIGRATION_38
```

If counts or path set differ at all:
- do not sync;
- STOP.

# 6. Pre-sync external setup readiness

Use:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe
```

From external RC1 root, run:

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

# 7. Protected external state snapshot

Before sync record exact existence, bytes and SHA256 for:

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
regular file count
sorted relative file names
total bytes
```

Also snapshot directory state for protected unlisted runtime/user locations:

```text
inference/input
inference/output
logs
runs
datasets
```

At minimum record:
- exists;
- recursive file count;
- total bytes.

Do not hash every runtime/user file unless already cheap.
Do not modify them.

# 8. Controlled Git-canonical 135-file migration

Run the approved helper exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Because the real manifest basis is `GIT_CANONICAL_BLOB_BYTES`, the helper must source exact Git HEAD canonical blobs,
not Windows working-tree CRLF bytes.

Require:

```text
copied = 135
verified = 135
failures = 0
exit code = 0
```

Do not rerun automatically if this fails.

# 9. Synchronize source_manifest.json separately

`source_manifest.json` is intentionally not one of the 135 entries.

After §8 succeeds, overwrite exactly:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

with the exact binary bytes from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Requirements:
- binary capture/write;
- no JSON regeneration;
- no text-mode newline conversion;
- no working-tree source.

Require:

```text
external source_manifest SHA256
==
canonical Git source_manifest SHA256
```

# 10. Post-sync helper integrity gate

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

# 11. External policy spot-check

Read only:

```text
README.md
docs/model_card.md
source_manifest.json
```

Require README:
```text
contains software-input-modality wording
does NOT contain "正式输入域"
contains "已验证数据域与 Demo 边界"
A2 provenance/domain meaning = NOT ESTABLISHED
locked candidates meaning = NOT YET RUN
```

Require model card:
```text
contains "软件输入模态"
contains separate verified-domain wording
A2 = persistent non-detection / provenance NOT ESTABLISHED
contains Demo policy
```

Require source manifest:
```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

No candidate images may be inspected.

# 12. Protected-state post comparison

Repeat §7 snapshot.

Require:

```text
protected model/component files unchanged = YES
Qwen directory names/count/total bytes unchanged = YES
inference/input unchanged = YES
inference/output unchanged = YES
logs unchanged = YES
runs unchanged = YES
datasets unchanged = YES
```

If anything protected changes unexpectedly:
- STOP immediately;
- record exact difference;
- do not attempt ad-hoc repair.

# 13. Post-sync external setup readiness

Run external:

```text
python check_setup.py
```

with the same environment Python.

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

Do NOT run pytest or inference.

# 14. One-time migration interpretation

If all gates pass, record explicitly:

```text
The 38 pre-sync mismatches were the exact historical entries whose manifest identities
were normalized from Windows working-tree bytes to Git-canonical blob bytes in P1D11M1.

This task migrated the external copies of those source/config files to the same Git-canonical identity.
No research/runtime algorithm was changed by the migration itself.
```

Do NOT describe this as a detector/model fix.

# 15. Outcome

Choose exactly one.

## A — `PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_COMPLETE`

Require all gates pass.

Freeze:

```text
external manifest-listed source/config = 135/135 Git-canonical match
external source_manifest = Git-canonical match
external supported-domain policy = SYNCHRONIZED
protected external assets = UNCHANGED
external setup = READY
locked candidates = FINAL_METADATA_LOCK / NOT YET RUN
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
```

Next:

```text
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Do not execute.

## B — `PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_STOP`

For any failed gate.

Next:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_RECOVERY
```

Do not execute.

# 16. Reports

Create:

```text
docs/task8b3_p1d11b_r1_external_git_identity_migration.md
```

Also append a short R1 correction/result section to:

```text
docs/task8b3_p1d11b_supported_domain_policy_external_sync.md
```

The original P1D11B STOP history must remain.

Required report facts:

1. starting HEAD
2. ChatGPT correction of the old 133/2 expectation
3. 38-path equivalence with P1D11M1 normalized set
4. canonical 135/135 Git identity
5. pre-sync 97/38 exact gate
6. pre-sync setup
7. protected pre-snapshot
8. one controlled 135-file migration result
9. separate Git-canonical source_manifest write
10. post-sync 135/135
11. external policy spot-check
12. protected post comparison
13. post-sync setup
14. pytest/inference/locked candidates = NOT RUN
15. PROP-01 remains OPEN
16. exact outcome
17. exact next gate.

# 17. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11B-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 894163a082e332de377bab8635e4257bfb956326
Model/inference execution: NONE
Pytest: NOT RUN
Canonical RC1 modified: NO
External delivery modified: YES — CONTROLLED GIT-CANONICAL MIGRATION/SYNC ONLY / NO
Canonical Git manifest: 135/135 PASS / FAIL
Canonical source_manifest Git SHA256: <sha>
Pre-sync external check: 97/135; mismatch exact P1D11M1 normalized 38 / other
Pre-sync mismatch-set classification: EXPECTED_GIT_IDENTITY_MIGRATION_38 / other
Pre-sync external setup: READY / NOT READY
Manifest-listed files copied/verified: 135/135 / other
External source_manifest Git SHA match: YES / NO / NOT RUN
Post-sync external check: 135/135 PASS / FAIL / NOT RUN
External README policy: VERIFIED / NOT VERIFIED
External model-card policy: VERIFIED / NOT VERIFIED
Protected model/component assets changed: NO / YES / NOT CHECKED
Qwen asset directory changed: NO / YES / NOT CHECKED
Protected runtime/user dirs changed: NO / YES / NOT CHECKED
Post-sync external setup: READY / NOT READY / NOT RUN
A2 domain classification: NOT ESTABLISHED
A2 policy status: DOCUMENTED_PERSISTENT_NON_DETECTION
Locked candidate status: FINAL_METADATA_LOCK
Locked candidate runtime status: NOT YET RUN
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: OPEN
MASK-01 status: OPEN
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d11b_r1_external_git_identity_migration.md
Next action: Awaiting ChatGPT audit; do not run locked candidates.
```

# 18. Commit / push

If COMPLETE:

```text
docs(rc1): record git canonical external policy sync
```

If STOP/FAILED:

```text
docs(rc1): record external git identity migration stop
```

Push current branch normally.
No force push.

# 19. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no canonical RC1 modification;
- pre-sync mismatch set is exactly the 38 P1D11M1-normalized paths;
- pre-sync setup READY;
- protected snapshot recorded;
- exactly one controlled helper write sync succeeds 135/135;
- source_manifest separately copied from Git canonical bytes;
- post-sync helper check = 135/135;
- policy spot-check passes;
- protected model/Qwen/runtime-user state unchanged;
- post-sync setup READY;
- no pytest/model/inference/locked candidate execution;
- PROP-01 remains OPEN;
- next gate not executed;
- report/handoff committed and pushed;
- STOP.
