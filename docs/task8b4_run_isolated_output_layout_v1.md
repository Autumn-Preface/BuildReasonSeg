# Task 8B.4 — run-isolated output layout

Status: IN_PROGRESS — Supervisor PARTIAL_ACCEPT / CONTINUE_AUTHORIZED; external precheck explained.

## Authority and starting state

Task: `TASK8B4_RUN_ISOLATED_OUTPUT_LAYOUT_V1`. Exact clean predecessor and fetched remote: `audit/task8b3-a2-ground-truth-case-validity-v1` at `c6d09e77d3c403771a92d096fe5977ebaa61db67`. Task branch: `fix/task8b4-run-isolated-output-layout-v1`. Supervisor task book installed byte-for-byte; startup reads and original source/hash evidence are in the JSON report.

## Implemented behavior

Each output-owning image run exclusively creates the first available `inference/output/<stem[_NNN]>/` and its `diagnostics/`, `masks/`, `overlays/` children. Failed or empty runs permanently occupy their root; mask existence never selects the suffix. Filenames retain `<stem>_mask[_NNN].png` / `<stem>_overlay[_NNN].png`. Diagnostics filenames remain unchanged and live directly inside run-local diagnostics.

Inspect mode allocates immediately after successful image load, before detector execution. Disabling diagnostics suppresses its writes. Inspect never writes a mask or overlay. Payload fields, CLI labels, status/exit and SUCCESS semantics stay unchanged. The optional no-argument `output_dirs()` query retains its old locations; allocation always supplies the new run root.

Legacy shared directories and files are neither reused nor migrated. Any occupied root entry, including a file or `masks`, `overlays`, `diagnostics`, advances the suffix.

## Canonical targeted validation

| Gate | Result |
| --- | --- |
| `tests/test_task8b_runtime.py` | 62 passed, exit 0 |
| `tests/test_cli_contract.py` (unchanged) | 24 passed, exit 0 |

New synthetic/stub contracts cover T1–T10, first/second paths, failure reruns, empty and file collisions, legacy preservation, parallel allocation, mask dtype/dimensions/alpha, diagnostic names, no-save behavior and pre-load failure timing. The first runtime attempt had two fixture errors from using an invalid parse source; the fixture now uses the existing `user_supplied` source. All attempts and outputs are retained in JSON.

## Prior manifest and STOP integrity record (historical)

Starting manifest validation against Git blobs: 135 checked, 135 match, 0 missing, 0 mismatch. The five changed manifest-listed files are outputs.py, pipeline.py, test_task8b_runtime.py, README.md and inference/README.md. Their identities were calculated from committed Git blobs at `7965a99743f02d5427c55be429e09349faa3605e`, not Windows working-tree bytes. Updated manifest validates 135/135; exactly five entries changed, with no path addition/removal.

Read-only external inventories recorded before implementation: 1,543 output-history files, 6 input files and 20 protected model-asset files. Full byte lengths, SHA256 values and timestamps are retained in JSON. No external writes or sync have occurred.

Full canonical suite: **127 passed, 17 failed, 6 errors; exit 1** (37.58 s). No failing node is in runtime or CLI contracts. The complete stdout/stderr, commands, failed/error nodes and classifications are retained in JSON.

The original five failing test-module sources, check_setup.py and package/frontend dependencies are byte-identical to the accepted predecessor. Canonical `VERSION`, decoder/detector, SAM2, ProgramHead/Qwen weights and fallback fixtures are absent both in the accepted Git snapshot and current canonical tree; they exist in complete external RC1. `inference/input` is absent in canonical. Six setup fixture errors and the relocation test stop when copying missing VERSION. Package/language tests require absent assets; fallback tests require absent historical log fixtures. Two READY assertions cannot pass with missing full-delivery prerequisites. A further unchanged setup assertion fails because sandbox access to AppData/Roaming/Ultralytics/settings.json is denied (WinError 5), suppressing its provenance label; this is preserved as an L0 environment observation, not a product regression claim.

The original task-book sections 16/24 required a green full canonical gate at checkpoint 2 (superseded by the Supervisor continuation below). GOV-D007 explicitly freezes the lightweight canonical / complete external distinction. A green canonical complete-delivery gate cannot be produced within the authorized file scope without adding excluded prerequisites or changing gate/test acceptance. The executor has made neither change. **Checkpoint 2 is not accepted and READY_FOR_SUPERVISOR_AUDIT is not claimed.** Supervisor must reconcile the required gate and authorize the next action.

External helper pre-sync check, controlled sync, manifest write, post-check, setup and external targeted/full tests are **NOT_RUN**. No external write has occurred. Read-only STOP integrity checks verify exact before/after equality (including file timestamps) for all 1,543 historical output files, 6 inputs and 20 protected model assets. All 135 external source files and the external manifest are unchanged. Recorded starting external source identities also match the accepted predecessor manifest 135/135. These checks do not establish post-sync acceptance.

Implementation checkpoint was committed/pushed at `7965a99743f02d5427c55be429e09349faa3605e`. The updated manifest, failure evidence and STOP handoff are saved in a separate follow-on checkpoint. Final SHA is resolved from actual Git after push; the files intentionally contain the preceding observed checkpoint rather than a fabricated self-referential SHA. The Supervisor-provided CURRENT_TASK remains byte-for-byte installed, with STOP outcome recorded in EXECUTOR_STATE.

## Boundaries

No real detector/Qwen/SAM/D-B1 inference, A2, historical image suite, locked Demo cases, model/threshold/algorithm changes, scientific claim changes, legacy cleanup or protected-asset changes. This milestone may establish only engineering output isolation and delivery integrity. Final acceptance remains with ChatGPT Supervisor.

## Supervisor continuation and external precheck

The Supervisor remotely reviewed and accepted implementation, canonical targeted contracts (62 runtime / 24 CLI), canonical manifest 135/135 and the authorized diff at `7e0e8b3115c2b3660d57ffa5170c4fb5393be39c`. Ordered governance/handoff reads, fetch, exact local/remote task HEAD and clean worktree were reverified; no new branch was created.

```text
canonical_full_suite_disposition = WAIVED_AS_INVALID_BY_SUPERVISOR_UNDER_GOV_D007
```

The historical 127 passed / 17 failed / 6 errors execution remains intact and is not relabeled PASS. No excluded asset/fixture/VERSION/input directory or test weakening is introduced in canonical. GOV-D007 assigns the complete full-suite acceptance gate to external complete RC1.

The existing read-only helper reported checked=135, match=130, missing=0, mismatch=5 (exit 1 expected). Differences are exactly README.md, runtime/outputs.py, runtime/pipeline.py, inference/README.md and tests/test_task8b_runtime.py. Each external identity equals its recorded pre-Task8B4 identity and each target equals the accepted committed canonical identity. There is no unrelated source/config mismatch.

External manifest is byte-identical to the Supervisor-accepted STOP inventory. Its stale entry identities (including earlier accepted repair entries) are enumerated in JSON and explained by the accepted unsynchronized control manifest. The authorized P1D11B write will install the exact current Git canonical manifest. All output/input/model inventories still equal the prior STOP snapshot; fresh closure baseline also includes historical logs and the complete file inventory.

Checkpoint 3 persists this explained precheck before any external write. Controlled sync and all post-sync acceptance gates remain pending at this checkpoint. No product/source changes are introduced during continuation.
