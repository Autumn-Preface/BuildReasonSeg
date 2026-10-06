# Task 8B.4 — run-isolated output layout

Status: **READY_FOR_SUPERVISOR_AUDIT**. Next gate: `CHATGPT_TASK8B4_OUTPUT_LAYOUT_REMOTE_AUDIT`. Executor work stops here; milestone approval remains with ChatGPT Supervisor and Final Demo has not started.

## Result and scope

Each output-owning image run exclusively reserves the first available `inference/output/<stem[_NNN]>/`, with `diagnostics/`, `masks/`, `overlays/`. A failed or empty run permanently occupies its root. Allocation never depends on mask existence; existing directories/files and legacy root names advance the suffix. Filenames retain `<stem>_mask[_NNN].png` and `<stem>_overlay[_NNN].png`. Diagnostics retain their original names directly inside run-local diagnostics.

Inspect reserves after successful image load and before detector execution. Disabling diagnostics suppresses diagnostic files; inspect never writes a mask/overlay. Payload fields, CLI labels, exit/status and SUCCESS semantics remain frozen. Historical shared outputs are preserved without migration or cleanup. No product/source change was made during continuation.

## Authority and checkpoints

Initial accepted predecessor: `audit/task8b3-a2-ground-truth-case-validity-v1` at `c6d09e77d3c403771a92d096fe5977ebaa61db67`. Task branch: `fix/task8b4-run-isolated-output-layout-v1`. Initial task book was installed byte-for-byte; original sources, Git blobs, SHA256 identities and call references remain in JSON.

Implementation plus green targeted tests was committed/pushed at `7965a99743f02d5427c55be429e09349faa3605e`. Manifest and canonical failure STOP evidence was committed/pushed at `7e0e8b3115c2b3660d57ffa5170c4fb5393be39c`.

The Supervisor remotely audited that exact task HEAD and issued **PARTIAL_ACCEPT / CONTINUE_AUTHORIZED**, an explicit L2 disposition under GOV-D007. Startup reread the five required governance/handoff files in order, fetched the exact remote HEAD, verified the local branch/HEAD and clean worktree, and continued the same branch. No new branch or destructive Git operation occurred.

Explained pre-sync evidence was committed/pushed at `37f2d681ef5140080b342c20168de82907adfbc4` before any external write. Sync/setup/targeted gates were committed/pushed at `482bfe37fe622a41546578f7115e11127123d68b`. The final audit-ready checkpoint follows that observed commit; actual final local/remote SHA is resolved after push, without writing a fabricated self-referential SHA into its files.

## Canonical validation and preserved failure history

| Gate | Result |
| --- | --- |
| Canonical runtime targeted | 62 passed, exit 0; Supervisor accepted |
| Canonical CLI targeted (unchanged) | 24 passed, exit 0; Supervisor accepted |
| Canonical manifest | 135 checked / 135 match / 0 missing / 0 mismatch |
| Historical canonical full suite | 127 passed / 17 failed / 6 errors, exit 1; preserved, not PASS |

```text
canonical_full_suite_disposition = WAIVED_AS_INVALID_BY_SUPERVISOR_UNDER_GOV_D007
```

The complete-delivery gate is invalid for the lightweight canonical source/config snapshot. The Supervisor classified the prior 127/17/6 execution as inapplicable, not a Task 8B.4 product regression. Full commands/stdout/stderr and all failed/error node classifications remain unchanged in JSON. Canonical lacks excluded VERSION, model/Qwen/SAM2 assets, historical log fixtures and input directory; an earlier setup attempt also encountered sandbox Ultralytics settings access denial. No excluded prerequisites were added and no tests were weakened. The first targeted attempt's two invalid parse-source fixture errors and their local fixture correction are also retained.

Contract tests cover T1–T10, first/second paths, diagnostics-only failure reruns, empty/file/legacy collisions, concurrent allocation, mask dimensions/dtype/values/alpha, diagnostic fields, no-save behavior and pre-load error timing. They use synthetic images and injected detector/model stubs with temporary output roots.

Exactly five manifest entries changed: README.md, runtime/outputs.py, runtime/pipeline.py, inference/README.md and tests/test_task8b_runtime.py. Their byte lengths/SHA256 values were derived from committed Git canonical blobs at the implementation checkpoint. Schema, identity basis and all 135 path entries remain unchanged; no CRLF-expanded working-tree identity was used.

## External closure

External root: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`.

| Gate | Exact result |
| --- | --- |
| Existing helper read-only precheck | 135 checked / 130 match / 0 missing / 5 mismatch; exit 1 expected; EXPLAINED |
| Controlled source/config sync | One invocation; 135 copied / 135 verified / 0 failures; exit 0 |
| External manifest | Exact Git canonical binary write under P1D11B; byte/hash MATCH |
| Helper post-check | 135 checked / 135 match / 0 missing / 0 mismatch; exit 0 |
| External check_setup | READY; exit 0 |
| External runtime targeted | 62 passed; exit 0 |
| External CLI targeted | 24 passed; exit 0 |
| External complete full suite | 150 passed; exit 0 (55.393 s) |
| Final helper after tests | 135 checked / 135 match / 0 missing / 0 mismatch; exit 0 |

Precheck mismatches were exactly the five authorized changed entries. Every external identity matched its recorded pre-Task8B4 source identity and every target matched the accepted Git blob. The external control manifest was unchanged from the accepted STOP snapshot; its stale entry differences, including earlier accepted repairs, are enumerated in JSON. No unrelated source/config drift was found. The helper copied only its 135 listed paths and deleted nothing; the control manifest was separately written from Git bytes. External manifest SHA256: `df9a870d72a25421311698f7a8865e07b4a4e11e9bfe52df18975da17b3bcf7e`.

The established `.conda/buildreasonseg-mvp` Python environment was used. Unique temporary Ultralytics/matplotlib caches and offline asset flags avoided the prior user-settings sandbox issue without changing source/test contracts. Setup verifies imports, paths and asset hashes. The full suite reads metadata/assets and historical fixtures, uses temporary synthetic setup fixtures, stub detector/pipeline tests, and CLI help/error/report contracts. It runs no real inference or historical image suite.

## Preservation evidence

Fresh continuation pre-sync inventories and final inventories are retained in JSON, together with initial preimplementation and STOP history evidence.

| Protected content | Result |
| --- | --- |
| Historical inference/output files | All 1,543 byte-identical; path membership and timestamps unchanged |
| inference/input files | All 6 byte-identical; path membership and timestamps unchanged |
| Protected model/Qwen/SAM2 files | All 20 byte lengths/SHA256 values and path membership unchanged |
| Historical logs/fixtures | All pre-existing contents unchanged |
| Real output directories through full suite | Unchanged; no new run directory |
| External file deletions | 0 |
| Natural test additions | 12 error logs, retained |

The authorized helper rewrote manifest-listed model metadata/config files with identical content. Existing full setup verification refreshed the derived Qwen integrity cache with identical bytes/hash. These timestamp-only changes are enumerated separately; no protected content change was accepted or hidden. All external inventory changes are explained by listed source/config sync, exact control-manifest write, the identical cache refresh and natural test logs. No evidence was deleted.

## Claim boundary

Established: `RUN_ISOLATED_OUTPUT_LAYOUT_IMPLEMENTED` and delivery integrity under the authorized gates. No detector/Qwen/SAM/D-B1 inference, A2, historical six-case suite, locked Demo execution, threshold/model/algorithm/scientific-contract modification or historical output migration occurred. PROP-01 and REF-01 conclusions remain unchanged; SUCCESS still asserts runtime structural validity only, not semantic target correctness. Governance files remain unchanged. Final milestone acceptance belongs to the Supervisor.
