# DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1

Status: IN_PROGRESS — canonical gate passed; external validation pending.

The canonical detector now converts internal RGB tiles to an independent, contiguous BGR array at the Ultralytics NumPy API boundary. The original RGB tile is preserved. All detector invocation arguments and downstream mask/proposal extraction remain unchanged.

Startup reads followed AGENTS.md order. Actual starting branch was `audit/task8b3-prop01-a2-channel-contract-causal-test-v1`, HEAD `4e3019c77984dcb27c7a10651ccc7df631abf0da`, with a clean working tree. Authorized task branch: `fix/task8b3-detector-rgb-bgr-contract-v1`. The Supervisor attachment was installed byte-for-byte and reread; SHA256 `b72d3101c79ce56da95930a2164dff11752f57e6b3cee7baa3bb42440801435e`.

## Canonical gate

`test_task8b_runtime.py`: 47 passed, including 5 new contract cases. Tests cover distinct RGB/BGR pixels, non-contiguous read-only input, byte immutability, independent API storage, shape/dtype preservation, C-contiguity, exact frozen kwargs for CPU/CUDA, strict mask threshold and proposal confidence ordering/fields. The first default-temp run had 38 passed / 9 setup errors from `PermissionError [WinError 5]` on the existing pytest temp directory. L0 remediation used a unique task temp directory; the unchanged suite then passed. No product or test weakening occurred.

Manifest identity is `GIT_CANONICAL_BLOB_BYTES`; all 135 entries matched initial Git blobs and external actual source. Only detector.py and the directly relevant existing test file have identity updates. The controlled sync requires a committed canonical checkpoint before it reads HEAD blobs.

## External plan and preserved evidence

Before sync, the evidence JSON records the entire external file size/mtime inventory and SHA256 of every non-manifest-listed file, including all model assets, locked images, historical logs and outputs, and the external manifest. Existing external manifest metadata drift is preserved. Source/config sync uses only `scripts/sync_advisor_rc1_delivery.py`; no manual external source edits.

Six-case detector regression will use exact locked A1/A2/A3/A4/B1/B2 identities and unchanged CUDA/model/parameters, execute product `detect_global`, observe actual preprocessing tensors, and use the existing `select_reference`. Reference-stage results are not full pipeline or semantic claims. A1–A4 comparison uses accepted historical suite counts; the later accepted compact-mask memory gate supersedes the original B1/B2 E502 results (B1 raw/merged 6578/3066; B2 7864/3740).

Then run the complete external regression suite, preserve exact pass/fail counts and failing nodes, and verify protected hashes and source identities again. Stop on any task-defined condition; no second repair is authorized.

## Claim boundary

A2 raw = 0 is not a failure criterion. PROP-01 remains OPEN. A2 case validity is NOT_EVALUATED_IN_THIS_TASK. No threshold sweep, parameter tuning, model/weight change, A2 special handling, ranking, architecture, or acceptance-rule change is authorized. Final acceptance belongs to the Supervisor.

Machine-readable evidence: `evaluation/task8b3_detector_rgb_bgr_contract_repair_v1.json`. Next gate: `CHATGPT_DETECTOR_CHANNEL_REPAIR_REMOTE_AUDIT`.
