# TO_DSH — Task 6M: Native-Vector Proposal Model + Structured Demo Gate

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Predecessor: Task 6L → `VECTOR_DATASET_MIGRATION_PASS`
>
> Goal: train a modern instance-segmentation proposal model on `WHU-EA-NativeVector v1.0`, evaluate the native-vector version of the Task 6J structured grounding chain under the primary `scene_disjoint_v1` split, and produce a CMD-runnable structured Demo path.
>
> This task is about **proposal quality + end-to-end structured inference**. It is NOT the final `[REF]` / SRE / SCL paper architecture.

## 0. User-facing language
All DSH narrative/UI output must be Chinese. Code/model names/metric keys/program ids may remain English.

## 1. Frozen inputs
Treat these as frozen:
- `WHU-EA-NativeVector v1.0`
- `BuildSpatialReason v0.2`
- `scene_disjoint_v1`
- frozen Task 3B relation semantics
- 20 canonical programs
- Task 6J J0/J2/J3 historical artifacts
- Task 6K/6K.1/6L audit artifacts
- historical YOLOv8m-seg baseline and pseudo-instance artifacts

Do NOT rewrite v0.1.1 or v0.2.

Task 6L measured:
- 17,388 canonical tiles;
- 41,186 clipped native instances;
- train/val/test tiles = 10,044 / 3,618 / 3,726;
- v0.2 samples = 12,778 / 9,111 / 6,219;
- all 20 programs supported in every split;
- 1,235 native instances <50 px retained;
- 35.4% instances touch tile border.

## 2. Fix the Task 6L paired-evaluation gap first
Task 6L currently reports `paired_counterfactual_availability: null`.

Before training, freeze deterministic native-vector evaluation packs.

Validation:
- 120 records
- 20 same-image paired counterfactual pairs
- stratified over L1/L2/L3 and program families
- each pair: same tile, different native target instances/masks

Test:
- 120 records
- 20 same-image paired counterfactual pairs
- same construction policy
- freeze before tuning
- do not inspect test results until model + inference thresholds are frozen on validation

Write:
- `evaluation/task6m_val_fixed120.json`
- `evaluation/task6m_val_paired20.json`
- `evaluation/task6m_test_fixed120.json`
- `evaluation/task6m_test_paired20.json`
- `evaluation/task6m_eval_pack_manifest.json`

# PART A — Proposal framework

## 3. Primary model: YOLO26m-seg
Use **Ultralytics YOLO26m-seg** as the primary proposal model.

Rationale already checked by ChatGPT against current official Ultralytics docs:
- YOLO26 is the current released family recommended for new projects;
- instance segmentation is officially supported;
- official COCO segmentation table reports YOLO26m-seg at 23.6M fused params and 44.1 mask mAP50-95;
- YOLO26 training includes Small-Target-Aware Label Assignment (STAL), relevant to retained small buildings;
- RTX 5080 Laptop 16GB should be sufficient for a controlled `m` experiment, but local VRAM must be measured.

This is a Demo engineering choice, not paper novelty and not proof of global optimality.

Do not run a broad framework bake-off in Task 6M.

## 4. License/provenance
Current official Ultralytics guidance places the open-source stack/models under AGPL-3.0, with enterprise licensing for proprietary/commercial use.

Record:
- exact Ultralytics package version;
- package source;
- pretrained weight source;
- pretrained weight SHA256;
- license note in Task 6M docs.

Do NOT change the repository license automatically.
Do NOT make commercial-licensing claims.

# PART B — Environment

## 5. New project-local environment
Do not use the historical editable `yolo_sam_env` for new training.

Create if needed:
`.conda/buildreasonseg-proposal`

Rules:
- Conda mandatory;
- no modification to base/jupyter/yolo_sam_env;
- `.conda/` remains gitignored;
- official released packages only;
- pin exact versions;
- do not clone/edit Ultralytics source.

It is acceptable to clone the working Task 6A project env and add official Ultralytics.

## 6. Download authorization
No dataset downloads.

Authorized for Task 6M only:
- official released `ultralytics` package/dependencies if missing;
- official `yolo26m-seg.pt`;
- optionally `yolo26s-seg.pt` only for smoke testing.

Use verified HTTPS and established Watt ownership rules.
Record source, size, SHA256.

No other downloads.

# PART C — Native-vector training export

## 7. Derived Ultralytics export
Create gitignored:
`artifacts/task6m_yolo_native/`

Use `scene_disjoint_v1`:
- train = train1
- val = train2
- test = test

Requirements:
- retain all empty tiles with valid empty labels;
- retain all native instances including tiny;
- no `<50` filter;
- no connected-component conversion;
- source images hardlinked if safe/possible; otherwise documented non-destructive fallback;
- never modify/move source images.

Ultralytics polygon TXT cannot preserve holes. For derived training export only:
- use required exterior polygon form;
- record affected hole instances;
- canonical GT stays unchanged;
- all evaluation uses canonical exact native masks.

Quantify export fidelity:
- mean/median/p1 IoU vs canonical masks;
- tiny-instance fidelity;
- hole-instance fidelity;
- malformed/degenerate count.

Gate:
- mean IoU >= 0.995
- no non-hole instance silently missing
- 0 malformed labels

Otherwise verdict `TRAINING_EXPORT_INVALID` and STOP.

# PART D — Smoke

## 8. M0 smoke
Before full training:
1. load official YOLO26 segmentation checkpoint;
2. 2-image train forward/backward;
3. deterministic 500–1000 tile subset, ~2 epochs;
4. validate mask decoding with canonical evaluator;
5. verify empty-image handling;
6. verify tiny labels enter training.

`yolo26s-seg` may be used for smoke only if materially faster.

No architecture conclusions from M0.

# PART E — Full training

## 9. M1 YOLO26m-seg
Train on `scene_disjoint_v1`.

Initial config:
- COCO pretrained checkpoint
- imgsz 640
- max epochs 80
- patience 15
- fixed seed
- AMP if stable
- device RTX 5080 Laptop
- conservative Windows-safe workers
- best + last checkpoints
- deterministic settings where supported
- no test usage during training/tuning

Batch:
- short memory probe first
- choose largest stable fixed batch

If OOM:
1. reduce batch
2. gradient accumulation if supported
3. do NOT silently reduce imgsz before reporting

Record:
- wall time
- max VRAM
- losses
- box/mask metrics
- best epoch
- early stop
- NaN/Inf
- checkpoint SHA256

# PART F — Proposal evaluation

## 10. Native-vector metrics
GT = canonical native masks, not training TXT.

On full val compute:
- recall @ IoU 0.25 / 0.50 / 0.75
- mask precision/recall
- mask AP50 / AP50-95 where available
- mean/median best GT→proposal IoU
- proposals/tile
- empty-tile false-proposal rate
- tiny-instance recall
- border-truncated recall
- dense-tile recall
- size breakdown

Matching is diagnostic only; GT must never repair proposals.

## 11. Validation-only threshold tuning
Sweep only a small declared set of:
- confidence
- max_det
- NMS/e2e mode if safely supported

Selection hierarchy:
1. target recall @0.50
2. oracle-program structured selected-mask performance
3. reasonable proposal burden

Freeze before test:
`evaluation/task6m_inference_config_frozen.json`

# PART G — Structured chain

## 12. Native J1-v2
Oracle program + predicted proposals.

Run on:
- full val
- fixed val120
- val paired20

Report:
- strict selected-mask mIoU
- Dice
- abstentions/reasons
- target selection success
- paired own-vs-cross mask performance
- proposal recall

Development gate:
- overall recall@0.50 >= 0.92
- tiny recall@0.50 >= 0.60
- fixed120 mIoU >= 0.50
- paired pass >= 14/20
- abstentions <= 20/120

If fail, do failure attribution; do not auto-train l/x or switch framework.

## 13. Program parser on v0.2
First evaluate frozen Task 6J Qwen3-VL-2B ProgramHead on v0.2 val:
- accuracy
- macro F1
- fixed120
- full val

Instruction text only.

If checkpoint missing or val accuracy <0.95:
- retrain same 2B text-only ProgramHead on v0.2 train only
- no image tokens
- no query_type leakage
- freeze before test

Do not upgrade to 4B.

## 14. Native J4-v2
After proposal config + parser freeze, run ONCE on:
- full test
- fixed test120
- test paired20

No GT in inference.

Chain:
instruction → Qwen2B ProgramHead → canonical program

image → YOLO26m-seg proposals → mask/bbox/centroid/area

program + proposal geometry → deterministic relation executor → selected proposal mask

GT only for evaluation.

Report:
- parser accuracy
- proposal recall
- strict selected-mask mIoU/Dice
- abstention
- paired pass
- L1/L2/L3
- per program
- tiny/border/dense breakdown

# PART H — Demo CLI

## 15. CMD-runnable inference
Create:

```text
python predict_structured.py ^
  --image path\to\image.tif ^
  --prompt "分割面积最大的建筑物右侧最近的建筑物" ^
  --proposal-checkpoint checkpoints\proposal\best.pt ^
  --parser-checkpoint checkpoints\program_parser\best.pt ^
  --out-dir outputs\demo
```

Outputs:
- parsed canonical program
- compact reasoning/program trace
- proposal count
- selected proposal
- selected mask PNG
- overlay PNG
- JSON result
- explicit abstention reason

Must not require GT or annotation files.

Unsupported instruction/program:
- explicit failure
- do not map to unrelated program

# PART I — Verdict

## 16. Exactly one
`STRUCTURED_DEMO_READY` if:
- export/training valid
- val development gate passes
- parser ready
- test fixed120 J4 mIoU >= 0.40
- test paired pass >= 14/20
- CLI works without GT on >=10 audited images

`PROPOSAL_MODEL_NEEDS_IMPROVEMENT`
`PROGRAM_PARSER_NEEDS_IMPROVEMENT`
`TRAINING_EXPORT_INVALID`
`INVALID_EXPERIMENT`

# PART J — Required artifacts

Create:
- `evaluation/task6m_eval_pack_manifest.json`
- `evaluation/task6m_val_fixed120.json`
- `evaluation/task6m_val_paired20.json`
- `evaluation/task6m_test_fixed120.json`
- `evaluation/task6m_test_paired20.json`
- `evaluation/task6m_training_export_audit.json`
- `evaluation/task6m_environment_manifest.json`
- `evaluation/task6m_training_summary.json`
- `evaluation/task6m_proposal_val.json`
- `evaluation/task6m_inference_config_frozen.json`
- `evaluation/task6m_j1v2_val.json`
- `evaluation/task6m_parser_v02.json`
- `evaluation/task6m_j4v2_test.json`
- `evaluation/task6m_error_attribution.json`
- `evaluation/task6m_demo_cli_audit.json`
- `evaluation/task6m_verdict.json`
- `docs/task6m_native_vector_proposal_demo.md`

Add scripts/config/tests.

Checkpoints and large exports stay gitignored. Record local paths + SHA256 in small manifests.

# PART K — Tests

At least:
1. canonical native dataset unchanged
2. v0.2 unchanged
3. v0.1.1 unchanged
4. no source mutation
5. export preserves all non-hole instances
6. empty labels valid
7. no `<50` filter
8. hole loss only recorded in derived export
9. correct scene_disjoint split
10. zero feature leakage
11. test pack frozen before tuning
12. no test metrics before frozen inference config
13. GT never repairs proposals
14. parser text-only
15. no query_type in parser input
16. no GT in J4
17. executor gets predicted geometry only
18. pairs same-image/different-native-target
19. all intended programs represented
20. CLI works without annotations
21. eval-mode determinism
22. unsupported program explicit failure
23. model hashes recorded
24. no large weights staged
25. no old editable Ultralytics fork used
26. no 4B
27. no `[REF]`, SRE, SCL
28. no GUI

Run:
`python -m pytest tests/ -q`

# PART L — Git / Watt
Commit code/config/tests/small eval/docs/handoff only.

Do not commit:
- `.conda`
- pretrained/trained weights
- derived image export
- caches
- source imagery/vector data

Recommended commit:
`feat: train native-vector proposal model for structured demo`

Use Watt ownership rules. If DSH starts Watt, close it after network work; if pre-existing, leave it.

# PART M — DSH model
Default:
- DeepSeek V4.1 Flash + High

Escalate to:
- V4.1 Flash + Max only for a genuine difficult runtime/causal bug

Do not use V4 Pro by default.

# PART N — Handoff
Update `handoff/FROM_DSH.md` and `handoff/PROJECT_STATE.md`.

Include:
1. Verdict
2. Fixed Eval Packs
3. YOLO26 Provenance
4. Export Audit
5. Environment
6. Smoke
7. Full Training
8. Proposal Metrics
9. Frozen Inference Config
10. J1-v2
11. Program Parser
12. J4-v2 Test
13. Failure Attribution
14. Demo CLI
15. License Note
16. Tests
17. Git/Watt
18. Recommended Next Step

# 17. STOP
After Task 6M STOP.

Do not automatically:
- train YOLO26l/x
- switch frameworks
- upgrade Qwen to 4B
- add `[REF]`, SRE or SCL
- download another dataset
- start formal final-model training
- build GUI

Wait for ChatGPT review.
