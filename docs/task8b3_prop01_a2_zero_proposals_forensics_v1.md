# PROP01 A2 Zero Proposals Forensics V1

Status: **READY_FOR_SUPERVISOR_AUDIT**.
Task: `PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1`.
Next gate: `CHATGPT_PROP01_FORENSICS_V1_REMOTE_AUDIT`. STOP; PROP-01 remains OPEN.

## Finding and claim boundary

Primary classification: **P1_INPUT_OR_PREPROCESSING_DEFECT**, using the exact task section 12 predicate "wrong/corrupted/transformed pixels reach detector". Exact A2 decoding and all nine NumPy tiles preserve source RGB pixels, but RC1 passes those RGB arrays to a BGR-assuming Ultralytics API. Every actual A2 network input tensor has reversed R/B channels relative to the source RGB image: a verified input contract defect.

**The earliest observed zero-proposal count is the raw Ultralytics result.** All nine library results already have zero boxes; actual detect_tile, accumulation, compact and merge counts are zero. No downstream transition loses an A2 proposal.

**The channel defect's causal contribution to zero detections is NOT_ESTABLISHED.** No corrected-input inference was performed; channel conversion is not proven to recover proposals. P2's valid exact input prerequisite is not satisfied at the network tensor because channel order is wrong. The primary P1 label classifies this verified defect, without claiming a proven counterfactual explanation for the zero count. P3/P4/P5 loss predicates are not met on the measured A2 path. No domain-gap, model-quality, below-confidence distribution, threshold-lowering or model-replacement conclusion follows.

## Startup and exact provenance

The five canonical files were read in Governance order and actual Git was verified: starting branch `docs/governance-v1-1-idle-state-semantics`, HEAD `504268128b7a9b058580b7702c9429b27705c053`, working tree clean. Task branch: `audit/task8b3-prop01-a2-zero-proposals-forensics-v1`.

The Supervisor attachment was copied byte-for-byte to CURRENT_TASK and re-read; initial attachment and task SHA256 both `cd270805140da288a30d8ae67953241763b9de51e089c82a2e5f6ad9b8103a80`. Only the authorized final Status field was subsequently changed.

External root: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`.

| Input | A2 | A1 positive control |
|---|---|---|
| Relative path | inference/input/A2.png | inference/input/A1.png |
| Dimensions/mode/dtype | 1024 x 1024 / RGB / uint8 | 1024 x 1024 / RGB / uint8 |
| File bytes | 1,677,040 | 1,607,301 |
| SHA256 | 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f | 8a4b459d65773a7dfb0ffcc509c26b5d3a7cea23cd759cdadf94cd46be84c227 |
| Program | largest_to_left_of_to_nearest | largest_to_right_of_to_nearest |
| Historical raw / merged | 0 / 0 | 133 / 52 |

`logs/task8b3_suite_results.json` binds these hashes to the exact cases, programs and counts. The existing `inference/output/diagnostics/A2/result.json` records FAILED/E401 and 9 tiles; its program/counts agree with the suite. The formal suite records diagnostic directory A2_001; the base A2 diagnostic independently agrees. Runtime decode equals direct Pillow RGB decode byte-for-byte. Historical artifact hashes and full input lock records are preserved in JSON. No guessed/substituted image was used. Only A1 was run as the control.

## Frozen detector and environment

Existing checkpoint: external `model/buildreasonseg_advisor/detector.pt`, YOLO26m-seg, 54,480,241 bytes, SHA256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`, matches model.yaml. CUDA device matches the historical A2/A1 recorded device; no CPU substitution.

Frozen constants: tile 512, overlap 128, stride 384, imgsz 640, conf 0.05, max_det 300, mask threshold 0.5, duplicate IoU 0.50. Each tile invokes the actual existing adapter:

```python
model.predict(source=tile_rgb, imgsz=640, conf=0.05, max_det=300,
              verbose=False, retina_masks=False, device="cuda")
```

Unchanged library defaults recorded: iou 0.7, agnostic_nms false, classes null, augment false, rect true, batch 1, half null, save/save_txt/save_crop false. No NMS, TTA, ranking or acceptance policy change.

Established environment `.conda/buildreasonseg-mvp/python.exe`: Python 3.11.16, Ultralytics 8.4.164, torch 2.13.0+cu132, torchvision 0.28.0+cu132, CUDA 13.2, cuDNN 92000, NumPy 2.4.6, Pillow 12.3.0, OpenCV 5.0.0.93, NVIDIA GeForce RTX 5080 Laptop GPU. Current installed library source hashes are preserved. Historical evidence did not lock every installed package file hash, so byte-identical historical package identity is not asserted; both A2 and A1 counts reproduce.

## Method and per-tile evidence

One actual DetectorRuntime.detect_global call for A2 and one for A1: exactly 18 tile model.predict calls. Process-local observers wrap actual predict, detect_tile, _compact_mask, merge_proposals and BasePredictor.preprocess, forwarding original arguments/results unchanged and restoring observers afterward. No product files are edited. No Qwen, SAM2, D-B1, full predict CLI or Demo chain runs.

Every actual source argument is checked against an independently extracted tile and the direct source RGB slice. The returned unchanged preprocessing tensor is recorded. Diagnostic-only channel comparisons use the same library pre_transform (letterboxing); comparison tensors are never fed into the detector. There is no alternate-input inference. The detector receives the original runtime tensor. Library warmup is not counted as a case predict call.

All A2 tiles: shape 512 x 512 x 3, uint8, no padding. Each library result list contains one result, zero boxes, absent masks, zero mask count, empty confidences, and actual adapter output count zero. Full tile/content/tensor SHA256 values are in JSON.

| Tile | top | left | Raw boxes | Masks present | Actual detect_tile |
|---|---:|---:|---:|---|---:|
| tile_0000_r000_c000 | 0 | 0 | 0 | false | 0 |
| tile_0001_r000_c001 | 0 | 384 | 0 | false | 0 |
| tile_0002_r000_c002 | 0 | 512 | 0 | false | 0 |
| tile_0100_r001_c000 | 384 | 0 | 0 | false | 0 |
| tile_0101_r001_c001 | 384 | 384 | 0 | false | 0 |
| tile_0102_r001_c002 | 384 | 512 | 0 | false | 0 |
| tile_0200_r002_c000 | 512 | 0 | 0 | false | 0 |
| tile_0201_r002_c001 | 512 | 384 | 0 | false | 0 |
| tile_0202_r002_c002 | 512 | 512 | 0 | false | 0 |

## Stage counts and source conditions

| Stage | A2 | A1 control |
|---|---:|---:|
| Raw Ultralytics returned boxes | 0 | 133 |
| Actual detect_tile / runtime raw_proposal_count | 0 | 133 |
| Entries reaching compact (actual compact calls) | 0 | 133 |
| Compact-valid | 0 | 133 |
| Accumulated entries (actual stored list) | 0 | 133 |
| Merge input | 0 | 133 |
| Merged | 0 | 52 |

The product has no separate pre-compact accumulated list: compact validation occurs before append. JSON accumulated means the actual stored list observed at merge; accumulated_before_compact means compact calls. Raw library boxes and product raw_count are separately observed measures. Raw Ultralytics result means after the frozen library detector policy, not pre-conf/NMS.

Exact conditions in `buildreasonseg/runtime/detector.py`:

- L220: detect_tile passes RGB NumPy source without channel conversion.
- L227: return empty if masks is None OR boxes is None OR len(boxes) == 0. A2 has zero boxes and absent masks; this is an empty-input return, not loss of nonzero boxes.
- L256: detect_global raw_count increases by actual adapter length, zero nine times.
- L272/L276: compact then append valid entries. A2 enters neither call nor append.
- L282: merge receives an actual empty list; merge_proposals L373 returns empty naturally.

All A2 downstream transitions drop **0** proposals. A1's 133 to 52 reduction is ordinary frozen duplicate merging, not an A2 loss mechanism. A1 shows proposal capability under this model/environment and the same channel mismatch; it cannot establish the effect of correcting that mismatch.

Input defect chain: imageio.py L89/L93 preserves RGB; detector.py L220 passes it; installed ultralytics/data/loaders.py L571 documents NumPy BGR and L590 accepts the three channels unchanged; engine/predictor.py L197 flips channels intending BGR to RGB. All A2 tensors are float32 [1,3,640,640], equal normalized letterboxed source with reversed channels, unequal source RGB. All A1 tensors show the same condition. This is dynamic tensor evidence plus source inspection.

## Integrity, validation and reproducibility

All 135 actual external manifest-listed files match canonical Git blobs and canonical manifest hashes. Detector source SHA256: `934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883`; imageio: `b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878`.

External source_manifest contains eight stale entries: README, pipeline, model_card, runtime_mapping, inference README, predict, CLI tests, runtime tests. Actual files match current canonical source. Metadata drift is recorded and untouched; it does not establish model/input mismatch.

The complete external 2,094-file size/mtime inventory is unchanged. All 144 protected content hashes are unchanged (135 manifest entries plus manifest, locked inputs/results, suite/probe/transcript, detector weights/config). Every large external asset was not content-hashed. Library settings/caches were redirected to a disposable temporary directory and bytecode writes disabled; no external artifacts were emitted.

Provenance, frozen inference, saved-evidence assertions, syntax/import, JSON/YAML parse, task preservation, whitespace and five-path scope checks passed. No full pytest suite was required/run. Checks do not establish semantic correctness or independent milestone acceptance. Supervisor retains final approval.

From repository root (commands replace the authorized JSON):

```powershell
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_zero_proposals.py --phase provenance
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_zero_proposals.py --phase inference
& '.conda/buildreasonseg-mvp/python.exe' -B scripts/diagnose_prop01_a2_zero_proposals.py --phase finalize
```

Finalize alone validates/classifies saved observations with zero inference calls. Inference ran once in this task; later harness changes only hardened an invocation assertion and added saved-evidence classification/finalization. No second model run occurred.

Recoverable pushed checkpoints: `075f98733177e91c5c4ec5d546daeafb241c4a93` (provenance), `471a93e8147166e3028a992d88be1a1677fffea3` (frozen stage/tensor evidence). Final commit contains the ready handoff; actual final SHA is resolved from Git after push rather than written into its own commit. Local Git lacked its HTTPS helper; an existing bundled Git with a process-local PATH completed push without installation/config edits.

## Supervisor disposition

A future separately authorized task could test API-compatible channel conversion. This is a hypothesis only; no correction or prediction of recovered A2 proposals is established. No product source edit, tuning/sweep, alternative input inference, parameter/weight/model change, scientific conclusion change or external sync occurred. Only the five authorized report/evidence/diagnostic/handoff paths changed. CURRENT_TASK and EXECUTOR_STATE are READY_FOR_SUPERVISOR_AUDIT. PROP-01 remains OPEN. STOP; no automatic repair.
