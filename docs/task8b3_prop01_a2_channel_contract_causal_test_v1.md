# PROP01 A2 Channel Contract Causal Test V1

Status: **READY_FOR_SUPERVISOR_AUDIT**.
Task: `PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1`.
Next gate: `CHATGPT_PROP01_CHANNEL_CAUSAL_TEST_REMOTE_AUDIT`. STOP; no product repair.

## Result and allowed causal conclusion

Primary classification: **C2_CHANNEL_CORRECTION_STILL_ZERO**.

| Paired arm | Total raw boxes | Network tensor contract |
|---|---:|---|
| Existing RGB NumPy baseline | 0 | Source RGB with R/B reversed; confirmed |
| API-facing contiguous BGR counterfactual | 0 | Intended source RGB; confirmed |

Exactly one paired experiment completed: 9 baseline + 9 corrected case predict calls. All baseline guards and tensor assertions passed. The real channel-contract defect was corrected at the network tensor in this diagnostic, but that correction was **not sufficient to recover A2 raw proposals under the frozen setup**. The valid corrected A2 detector path still returned zero boxes on all nine tiles.

This result does not establish that the channel defect has no possible contribution in every setting. It does not establish domain gap, model inadequacy, sole scientific root cause, semantic correctness, full pipeline correctness or a successful product repair. No threshold experiment follows. PROP-01 remains OPEN in unchanged governance files.

## Authorized start and exact identities

Five canonical files were read in order: AGENTS, PROJECT_STATE, DECISIONS, CURRENT_TASK, EXECUTOR_STATE. Actual Git then matched required starting branch `audit/task8b3-prop01-a2-zero-proposals-forensics-v1`, HEAD `59296d25195a656e6a75e303e3cbb43fa9df8b8b`, and clean working tree. Task branch: `audit/task8b3-prop01-a2-channel-contract-causal-test-v1`.

Supervisor task attachment was copied byte-for-byte and re-read; SHA256 of attachment and installed task was `1b99bc499174ba10db15f5cbadf226576a2e0f4983356b11c5afc261225d4c4a`. At completion only its authorized Status value changes.

External root: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`.
Exact input: `inference/input/A2.png`, RGB uint8, 1024 x 1024, 1,677,040 bytes, SHA256 `10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f`. Runtime decode equals direct Pillow RGB decode. All nine source-space tile hashes match the accepted previous forensic evidence. There is no substituted image or additional case.

Existing detector: `model/buildreasonseg_advisor/detector.pt`, YOLO26m-seg, segmentation task, 54,480,241 bytes, SHA256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`. The model identity, model.yaml, actual external source files, installed input-contract library source hashes and library versions match the accepted forensic evidence. Previous report/evidence/diagnostic files remain unchanged.

## Frozen setup and single independent variable

Frozen values: tile 512, overlap 128, stride 384, imgsz 640, conf 0.05, max_det 300, duplicate IoU 0.50, mask threshold 0.5. Actual detector calls retain retina_masks false and CUDA. Duplicate merge and runtime mask extraction are not executed in this raw-library experiment; their frozen constants are checked and unchanged.

For every original RGB tile, in row-major order:

```python
with torch.no_grad():
    baseline = model.predict(source=RGB_tile, imgsz=640, conf=0.05, max_det=300,
                             verbose=False, retina_masks=False, device="cuda")
BGR_api_tile = np.ascontiguousarray(RGB_tile[..., ::-1])
with torch.no_grad():
    corrected = model.predict(source=BGR_api_tile, imgsz=640, conf=0.05, max_det=300,
                              verbose=False, retina_masks=False, device="cuda")
```

Only the API-facing NumPy channel representation changes. The corrected ndarray has the same shape/dtype, is contiguous and reverses back to the exact original RGB tile. Invocation arguments and observed library arguments are equal between arms and match accepted frozen defaults: iou 0.7, agnostic_nms false, classes null, augment false, rect true, batch 1, half null, save/save_txt/save_crop false, end2end null. The diagnostic loads one existing model and reuses the same predictor/backend objects for every pair. No training, weight replacement, NMS or TTA modification occurs.

Environment: established `.conda/buildreasonseg-mvp/python.exe`; Python 3.11.16; Ultralytics 8.4.164; torch 2.13.0+cu132; torchvision 0.28.0+cu132; CUDA 13.2; cuDNN 92000; NumPy 2.4.6; Pillow 12.3.0; OpenCV 5.0.0.93; NVIDIA GeForce RTX 5080 Laptop GPU. CUDA/GPU/library identities match accepted forensics. Network tensors are float32 [1,3,640,640] for both arms. No device or precision substitution.

## Tensor observation and guards

A process-local wrapper observes the actual tensor returned by unchanged Ultralytics BasePredictor.preprocess and returns that same tensor to inference. It restores the original method after the experiment; no installed library or product source file is edited.

The intended RGB comparison tensor is independently constructed from the exact original RGB tile using the same unchanged library letterboxing, layout and normalization. Comparison tensors are diagnostic-only, never passed into inference. Baseline must exactly equal this intended tensor with R/B reversed; corrected must exactly equal intended RGB. RGB and reversed RGB must be distinct. Assertions run before the observer returns the production tensor to inference.

All baseline actual tensor hashes reproduce the previous accepted actual tensor hashes. All corrected actual tensor hashes equal the previous accepted intended RGB tensor hashes. Hashes, API-source hashes, full invocation/default arguments and assertions are recorded per pair in JSON.

If baseline is nonzero, the script stops immediately with C3_BASELINE_NOT_REPRODUCIBLE and does not interpret counterfactual results. If a tensor assertion fails, it stops with C4_COUNTERFACTUAL_INVALID (guard condition INPUT_CONTRACT_COUNTERFACTUAL_INVALID). Neither condition occurred. Each completed pair is persisted before the next, so partial results survive interruption.

## Per-tile paired results

All source tiles are 512 x 512 x 3 uint8 with no padding. Each arm returned one Ultralytics result object with zero boxes, no masks, mask count zero and empty confidences. Raw counts mean library returned boxes after the frozen library policy, not pre-confidence/NMS scores.

| Tile | top | left | Baseline raw | Corrected raw | Baseline/corrected tensor guards |
|---|---:|---:|---:|---:|---|
| tile_0000_r000_c000 | 0 | 0 | 0 | 0 | PASS / PASS |
| tile_0001_r000_c001 | 0 | 384 | 0 | 0 | PASS / PASS |
| tile_0002_r000_c002 | 0 | 512 | 0 | 0 | PASS / PASS |
| tile_0100_r001_c000 | 384 | 0 | 0 | 0 | PASS / PASS |
| tile_0101_r001_c001 | 384 | 384 | 0 | 0 | PASS / PASS |
| tile_0102_r001_c002 | 384 | 512 | 0 | 0 | PASS / PASS |
| tile_0200_r002_c000 | 512 | 0 | 0 | 0 | PASS / PASS |
| tile_0201_r002_c001 | 512 | 384 | 0 | 0 | PASS / PASS |
| tile_0202_r002_c002 | 512 | 512 | 0 | 0 | PASS / PASS |

There are exactly 18 case model.predict calls in one process; library warmup is not counted as a case call. No repeat experiment, A1/control rerun, detect_global/merge, full predict CLI, Qwen, SAM2 or D-B1 inference occurs.

## External integrity, validation and reproduction

Before/after complete external inventory: 2,094 files, size/mtime digest unchanged, no added/removed/modified paths. All 144 relevant protected content hashes remain unchanged: 135 manifest-listed source files plus manifest, locked inputs/results, historical suite/probe/transcript, detector weights and configuration. This does not claim all large external assets were content-hashed. Existing external manifest metadata drift remains untouched; actual protected source matches accepted evidence.

Library settings and plotting caches are redirected to a disposable temporary directory; Python bytecode writes are disabled. No image/log/cache/config/output is written to external RC1. There is no external sync or asset download.

Syntax/import, JSON/YAML parse, saved paired-evidence assertions, baseline replay hashes, corrected intended-RGB hashes, frozen argument equality, external final identity checks, task byte preservation and five-authorized-path scope checks pass. No full pytest suite is needed or run. Saved-evidence finalization makes zero inference calls. These checks do not substitute for Supervisor final acceptance.

From repository root:

```powershell
# Preparation: zero inference; writes the new task JSON.
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_channel_contract_causal_test.py --phase prepare
# Exactly one paired experiment; replaces the new task JSON.
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_channel_contract_causal_test.py --phase run
# Validate existing paired evidence, without inference.
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_channel_contract_causal_test.py --phase finalize
```

In this execution, --phase run ran once. The later script change only strengthened saved-evidence validation against accepted tensor hashes; it did not change inference or trigger another run. Reproduction commands are documented for audit, not an authorization to continue autonomous experiments.

Recoverable checkpoints already pushed: `3aac94923214942b3411e73d0222814bf9710776` (pre-experiment design/identity) and `9ea5938dcdb0d7f126cf5d3d8843d7cc45d0daba` (paired observations/C2). Final handoff is committed/pushed afterward; actual local/remote final HEAD is read from Git after push, never written as a self-referential SHA.

## STOP disposition

Repair candidate only: **API-compatible RGB→BGR conversion before passing NumPy tile to Ultralytics**. This experimental correction did not recover A2 raw proposals and is not implemented in detector.py. Any product change requires a new Supervisor task.

No product source, governance, existing evidence, tests, configs, model or external RC1 changes. No sweep/tuning, acceptance-rule change, full pipeline run or product repair. CURRENT_TASK and EXECUTOR_STATE are READY_FOR_SUPERVISOR_AUDIT. STOP for CHATGPT_PROP01_CHANNEL_CAUSAL_TEST_REMOTE_AUDIT.
