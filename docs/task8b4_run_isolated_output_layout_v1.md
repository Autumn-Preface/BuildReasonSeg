# Task 8B.4 — run-isolated output layout

Status: IN_PROGRESS — implementation checkpoint; final Supervisor acceptance pending.

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

## Manifest and external integrity

Starting manifest validation against Git blobs: 135 checked, 135 match, 0 missing, 0 mismatch. The five changed manifest-listed files are outputs.py, pipeline.py, test_task8b_runtime.py, README.md and inference/README.md. Their identities will be calculated from the implementation commit, not Windows working-tree bytes. No manifest paths will change.

Read-only external inventories recorded before implementation: 1,543 output-history files, 6 input files and 20 protected model-asset files. Full byte lengths, SHA256 values and timestamps are retained in JSON. No external writes or sync have occurred.

Full canonical and all external gates remain pending. Canonical is the lightweight snapshot described in GOV-D007; missing assets/fixtures will be assessed from actual full-suite output without broadening this task or weakening tests.

## Boundaries

No real detector/Qwen/SAM/D-B1 inference, A2, historical image suite, locked Demo cases, model/threshold/algorithm changes, scientific claim changes, legacy cleanup or protected-asset changes. This milestone may establish only engineering output isolation and delivery integrity. Final acceptance remains with ChatGPT Supervisor.
