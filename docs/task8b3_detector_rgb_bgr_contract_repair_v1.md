# DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1

Status: READY_FOR_SUPERVISOR_AUDIT.

The canonical detector now converts internal RGB tiles to an independent, contiguous BGR array at the Ultralytics NumPy API boundary. The original RGB tile is preserved. All detector invocation arguments and downstream mask/proposal extraction remain unchanged.

Startup reads followed AGENTS.md order. Actual starting branch was `audit/task8b3-prop01-a2-channel-contract-causal-test-v1`, HEAD `4e3019c77984dcb27c7a10651ccc7df631abf0da`, with a clean working tree. Authorized task branch: `fix/task8b3-detector-rgb-bgr-contract-v1`. The Supervisor attachment was installed byte-for-byte and reread; SHA256 `b72d3101c79ce56da95930a2164dff11752f57e6b3cee7baa3bb42440801435e`.

## Canonical gate

`test_task8b_runtime.py`: 47 passed, including 5 new contract cases. Tests cover distinct RGB/BGR pixels, non-contiguous read-only input, byte immutability, independent API storage, shape/dtype preservation, C-contiguity, exact frozen kwargs for CPU/CUDA, strict mask threshold and proposal confidence ordering/fields. The first default-temp run had 38 passed / 9 setup errors from `PermissionError [WinError 5]` on the existing pytest temp directory. L0 remediation used a unique task temp directory; the unchanged suite then passed. No product or test weakening occurred.

Manifest identity is `GIT_CANONICAL_BLOB_BYTES`; all 135 entries matched initial Git blobs and external actual source. Only detector.py and the directly relevant existing test file have identity updates. The controlled sync requires a committed canonical checkpoint before it reads HEAD blobs.

## Controlled external sync and preserved evidence

External root: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`. Before sync, the evidence JSON records the full external file size/mtime inventory and SHA256 of every non-manifest-listed file, including all model assets, locked images, historical logs and outputs, and the external manifest. Existing external manifest metadata drift is preserved. Source/config sync used only `scripts/sync_advisor_rc1_delivery.py`; no manual external source edits.

| Controlled gate | Result |
|---|---|
| Before sync | checked=135 match=133 missing=0 mismatch=2 |
| Sync | copied=135 verified=135 failures=0 |
| After sync and final source identities | checked=135 match=135 missing=0 mismatch=0 |
| All 1964 pre-existing non-source content hashes | unchanged after sync and final regression |

Only detector.py and the existing runtime test file changed content. The helper refreshes metadata of all manifest-listed files when writing their committed Git blob bytes. The source manifest itself is not copied by the existing helper; its pre-existing external metadata drift is recorded and preserved.

## Six locked-case real detector regression

Exactly one product `detect_global` call per case was executed, in A1/A2/A3/A4/B1/B2 order, using one unchanged YOLO26m-seg model and the existing `select_reference(family="largest")`. No Qwen, SAM2, GRF, segmentation chain or full predict CLI inference was performed. Reference-stage results below are observed detector/reference facts, not full pipeline or semantic claims.

A1–A4 comparison uses accepted historical suite counts; the later accepted compact-mask memory gate supersedes the original B1/B2 E502 results. Baseline source paths, identities and raw records are retained in JSON. Proposal IDs are local to each regenerated proposal set; changes in ID alone do not measure semantic reference correctness.

| Case | Baseline raw/merged | Corrected raw/merged | Baseline → corrected reference ID | Detector/reference stage |
|---|---:|---:|---|---|
| A1 | 133 / 52 | 241 / 94 | 48 → 68 | completed; reference available |
| A2 | 0 / 0 | 0 / 0 | none → none | completed; existing E401 condition |
| A3 | 7 / 6 | 24 / 13 | 1 → 4 | completed; reference available |
| A4 | 216 / 77 | 292 / 74 | 30 → 50 | completed; reference available |
| B1 | 6578 / 3066 | 7130 / 3292 | 243 → 1453 | completed; reference available |
| B2 | 7864 / 3740 | 9207 / 4357 | 862 → 4233 | completed; reference available |

All 374 product tile predict calls passed API-source BGR/contiguity/shape/dtype checks, exact unchanged kwargs checks, accepted effective library argument checks, and actual preprocessing tensor equality to intended RGB. Original tile and full image bytes remained unchanged. All nine A2 actual tensor hashes equal the accepted prior corrected-arm hashes. All observed library raw box counts equal product extracted proposal counts. Model SHA256 is `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`; device is the established CUDA RTX 5080 Laptop GPU with the same accepted libraries/precision. Frozen policy remains tile=512, overlap=128, stride=384, imgsz=640, conf=0.05, max_det=300, duplicate IoU=0.50, mask threshold=0.5; default NMS and no TTA remain unchanged.

No model-loading issue, crash, E5xx failure or gross proposal disappearance occurred in this detector scope. Count changes and changed reference IDs are evidence only. A2 remains raw=0: `PROP01_REMAINS_OPEN`, with no rescue attempt.

## Broader regression and final integrity

Complete external command: established `.conda/buildreasonseg-mvp/python.exe -B -m pytest -q -p no:cacheprovider --basetemp <unique task scratch directory> --tb=short`, cwd external RC1. Result: **135 passed in 63.81s; 0 failed, 0 errors, failing nodes=[]**. The previously accepted external suite had 130 tests; this repair adds five directly relevant contract cases. No skip or test weakening was introduced.

Final inventory grew from 2099 to 2105 files only because the existing error-contract tests generated six new `logs/error_20261006_*.log` files; historical logs/outputs remain byte-identical. Existing `check_setup.py:205` calls `manifest.verify(full=True)`, whose unchanged `manifest.py:106-108` refreshes `qwen_integrity_cache.json` with the same verified hash. This cache's timestamp changed; its 110 bytes and SHA256 did not. All 1964 pre-existing non-source hashes, including detector/ProgramHead/Qwen/SAM assets, six locked inputs, historical logs, outputs and external manifest, remain identical. All 135 external source identities still equal committed canonical bytes.

An initially stricter diagnostic timestamp assertion flagged that known same-content derived cache refresh. Source inspection explained it; no product change or second repair occurred. Original observation and resolved classification are retained. Likewise, the initial ordinary-permission inventory could not see five existing `.pytest_cache` files; the full permission-view baseline added those files after verifying their old timestamps and all original 1959 hashes. No sync had run when that view guard fired. Final verification uses the same complete permission view.

The whole detector AST equals the starting version after removing only the new API conversion assignment and changing `source=api_tile` back to `source=tile_rgb`. This verifies unchanged parameters, ranking, extraction, tiling and merge code independently of green tests. It does not confer final approval; the Supervisor is the final approver.

## Reproduction and persistence

The evidence JSON embeds the exact temporary evidence runner source, phase commands/results, per-tile actual tensor/source identities, all regenerated proposal metadata, and original/final protected inventories and hashes. The runner changes no installed library file; its process-local observers return the original predictions/tensors and are restored after regression. Library caches and test fixtures use task scratch directories. A first detector-process settings fallback created a known new repository-local `Ultralytics/settings.json`; it was recorded and removed after process exit, never staged. Later phases use precreated task cache directories.

Recoverable checkpoints pushed: `8e9f5712f4edb79bb47f584d3b13774e2c3fafa7` (canonical repair/tests), `07aa62e` (controlled sync), `99a559b2383540ff44eee115c6e4a42c43b2d315` (six-case regression). Git HTTPS/shell-helper limitations were resolved using the installed curl helper and existing Git Credential Manager; credentials were used only in process memory, never logged or persisted. Final commit is created/pushed after this handoff; its actual local and remote SHA is reported directly from Git, without a self-referential follow-up commit.

## Claim boundary

A2 raw = 0 is not a failure criterion. PROP-01 remains OPEN. A2 case validity is NOT_EVALUATED_IN_THIS_TASK. No threshold sweep, parameter tuning, model/weight change, A2 special handling, ranking, architecture, or acceptance-rule change was performed. The engineering conclusion is only `DETECTOR_RGB_BGR_INPUT_CONTRACT = CORRECTED`. Detector scientific quality, semantic target correctness, segmentation correctness, REF-01 closure and architecture/performance improvement are not established. Final acceptance belongs to the Supervisor.

CURRENT_TASK and EXECUTOR_STATE are READY_FOR_SUPERVISOR_AUDIT. Machine-readable evidence: `evaluation/task8b3_detector_rgb_bgr_contract_repair_v1.json`. STOP for `CHATGPT_DETECTOR_CHANNEL_REPAIR_REMOTE_AUDIT`; no deeper PROP-01, A2 provenance/ground-truth audit, tuning or replacement follows automatically.
