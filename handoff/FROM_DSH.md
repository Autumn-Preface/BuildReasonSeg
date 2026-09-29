<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 6M.1 Report: Proposal Convergence + Correct Demo Gates

_This file holds the Task 6M.1 report. The Task 6M report is preserved in git history at commit
`b9f49f8` and in `docs/task6m_native_vector_proposal_demo.md`; every Task 6M artifact in
`evaluation/task6m_*.json` is byte-identical after this task._

Full design notes: `docs/task6m1_proposal_convergence_and_demo_fix.md`.

## 1. Verdict

**`PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`** — failed gates
`validation_j1_gate_passed`, `test_fixed120_miou_ge_0_40`, `test_paired_pass_ge_14`. Passing gates:
source-checkpoint verified, safe resume, **normal training completion (early stop)**, validation
evaluation ran, corrected Demo-CLI gate.

Convergence **did** help every metric, but not enough to clear the validation development gate, so per
section 10 no test evaluation was run, no model/config was changed and no threshold was lowered.

## 2. Preserved Task 6M evidence (Part A)

`evaluation/task6m1_source_checkpoint_audit.json` = **`SOURCE_CHECKPOINT_VERIFIED`**: `best.pt`
`fd407db634a8a7ef…dbf4ea44` and `last.pt` `ea998bda37dd2dcb…5513f860e` were recomputed and matched
exactly, then **copied** (never moved) with `results.csv` and run metadata into
`artifacts/checkpoints/task6m1/source_epoch18_snapshot/` and re-verified; the snapshot is never
overwritten. After the whole continuation the live hashes were checked again — **still unchanged**. The
four frozen Task 6M eval packs and the parser checkpoint were reused byte-for-byte (SHA256 compared
against `task6m_eval_pack_manifest.json`).

## 3. Safe continuation (Part A/B)

`evaluation/task6m1_resume_preflight.json` = **`SAFE_RESUME_READY` → `SAFE_RESUME_COMPLETED`**. Because
Ultralytics `resume=True` restores the checkpoint's stored `train_args` (whose `project`/`name`/
`save_dir` point at the Task 6M directory), the continuation resumed from a patched copy
`last_resume.pt` whose stored args point at
`artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/`, with `save_dir` also passed as an
explicit resume override — so no Task 6M evidence can be written. Verified before training: source
hashes unchanged, snapshot matches, stored epoch 17 (0-based) → **next epoch 19**, optimizer and EMA
state present, `save_dir` Task 6M.1-local, frozen config unchanged (imgsz 640, batch 16, workers 4,
seed 20260812, epochs 80, patience 15, AMP, deterministic, derived native-vector export). The log
confirms `Resuming training …last_resume.pt` and the first resumed epoch is **19/80**.

## 4. Continuation training (Part B)

`evaluation/task6m1_training_summary.json` — combined Task 6M + 6M.1 lineage.

* **stop = `early_stopping_fired`** (patience 15) → normal completion; epochs **19 → 55** (37 epochs);
* **2.42 h** wall clock, **235.5 s** mean epoch, peak VRAM **11.2 GB**, **no NaN/Inf**;
* combined **best epoch 40**: mask mAP50 **0.73742**, mask mAP50-95 **0.40475** (Ultralytics fitness
  proxy / `best.pt` box mAP50-95 0.44891); Task 6M epoch 18 was 0.68266 / 0.35437 (+8.0 % / +14.2 %);
* final epoch 55: mask mAP50 0.72415 / mAP50-95 0.39172; final `best.pt`/`last.pt` SHA256 recorded;
* no configuration change of any kind (same family, imgsz, batch, seed, split, loss, augmentation).

## 5. Validation proposal metrics (Part C)

`evaluation/task6m1_proposal_val.json` (full validation, canonical native masks, test untouched):

| Metric | Task 6M (ep 18) | Task 6M.1 (converged) |
|---|---|---|
| recall @ 0.25 / 0.50 / 0.75 | 0.7772 / 0.6333 / 0.3404 | **0.7902 / 0.6578 / 0.3928** |
| mask mAP50 / mAP50-95 | 0.6819 / 0.3637 | **0.7367 / 0.4025** |
| proposals / tile | 6.38 | **5.39** |
| empty-tile false-proposal rate | 0.1264 | **0.0303** |
| tiny / border / dense recall@0.50 | 0.0023 / 0.5461 / 0.6322 | **0.0068 / 0.5753 / 0.6590** |
| small / medium / large recall@0.50 | 0.1252 / 0.5299 / 0.7269 | **0.1621 / 0.5574 / 0.7505** |

Sweep used the **same** declared grid (conf ∈ {0.05, 0.10, 0.25} × max_det ∈ {100, 300}); frozen
config **conf 0.10, max_det 100** in `evaluation/task6m1_inference_config_frozen.json` with
`frozen_before_test: true` and `test_metrics_inspected_before_freezing: false`. No test metric was read
before that file existed.

## 6. J1-v2 validation (Part D)

`evaluation/task6m1_j1v2_val.json` — oracle program + converged predicted proposals + the frozen Task
6M val fixed120/paired20 packs, GT for scoring only:

| Gate | Threshold | Task 6M | Task 6M.1 | Result |
|---|---|---|---|---|
| overall recall@0.50 | ≥ 0.92 | 0.6333 | **0.6578** | FAIL |
| tiny recall@0.50 | ≥ 0.60 | 0.0023 | **0.0068** | FAIL |
| fixed120 strict mIoU | ≥ 0.50 | 0.2801 | **0.3506** | FAIL |
| paired pass | ≥ 14/20 | 4/20 | **7/20** | FAIL |
| fixed120 abstentions | ≤ 20/120 | 35 | **27** | FAIL |

Full validation mIoU **0.3541** (+0.0569), abstention rate 0.2579, by level L1 0.3805 / L2 0.2669 /
L3 0.3322; paired own IoU 0.3144 vs cross 0.0367 (8.6× margin — the executor discriminates, the
proposal is the limit). Thresholds were not lowered.

## 7. Test evaluation (Part E)

**Not run** — section 10 forbids it when a validation gate fails, and forbids model/config changes or
retuning. Task 6M's single earlier test inspection (`evaluation/task6m_j4v2_test.json`) remains the only
test evidence for this lineage; there is no `task6m1_j4v2_test.json`. No pristine-unseen-test claim is
made anywhere.

## 8. Corrected Demo CLI (Part F)

`predict_structured.py` now evaluates a deterministic closed-Demo domain gate **before ProgramHead and
before the proposal model**: a prompt must contain a building/object anchor (`建筑`, `建筑物`,
`建筑区域`, `房屋`, `楼`, `building`, `buildings`, `structure`) **and** a supported relation/selection
anchor, otherwise the CLI writes `result.json` with `status = unsupported_instruction`,
`abstention_reason = out_of_domain_prompt`, `parsed_program = null`, `proposal_count = null` and
**exits 4** — with no parser construction, no Ultralytics import and no inference.

`evaluation/task6m1_domain_gate_check.json` = **PASS**: all six required prompts rejected with exit 4
(`Write a poem about the sea.`, `今天天气怎么样？`, `请总结这张图片。`, `检测道路。`,
`segment the airplane`, empty/whitespace) and all five required positive prompts enter the parser path.
This is a closed-Demo guard, not open-domain OOD detection.

**Measured vocabulary repair.** The first audit run found a real defect: the v0.2 templates phrase
`topmost`/`bottommost` as **位置最高 / 位置最低** and English `leftmost`/`rightmost`/`largest` as
*furthest left/right*, *furthest to the left/right*, *greatest*, *highest/lowest*, which were missing
from the section-13 anchor list — **899 of 9,111 validation records (9.9 %)** and 9 of 20 program
templates were falsely rejected. Section 13 permits exactly this repair, so the relation vocabulary was
completed with those concept-equivalent anchors; object anchors were not widened and no free-form
acceptance was added, so all six required out-of-domain prompts are still rejected.
`evaluation/task6m1_domain_gate_coverage.json`: every distinct v0.2 template in train/val/test is now
accepted in both languages — val **9,111/9,111 zh (100 %)** and **9,111/9,111 en (100 %)**, **0 falsely
rejected templates**, all 20 programs reachable.

## 9. Representative CLI audit (Part G)

`evaluation/task6m1_demo_cli_audit.json` = **gate PASS** (checkpoint
`task6m1_continuation_best`, frozen conf 0.10 / max_det 100, annotation files unavailable to the CLI):

* **12 supported prompts** — 4 L1 / 4 L2 / 4 L3, **12 distinct canonical programs**, **12/12 expected
  program matches**, 9 answered with a selected mask + overlay, 3 explicit abstentions, **no ground
  truth used**;
* **all 6 out-of-domain prompts** exit 4 before parser/proposal inference; no unsupported prompt is
  silently mapped to a program;
* families covered include `smallest`, `leftmost`, `rightmost`, `topmost` (gate-completed vocabulary),
  `smallest_to_right_of`, `smallest_to_nearest`, `largest_to_right_of`, `smallest_to_below` and the
  compositional L3 set `largest_to_{above,right_of,left_of,below}_to_nearest`.

## 10. Error attribution (Part H)

`evaluation/task6m1_error_attribution.json` — diagnosis
**`CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`** (the declared rule fires because fixed120 mIoU improved
by +0.0705 ≥ 0.05). Deltas, Task 6M epoch 18 → Task 6M.1: overall recall@0.50 **+0.0246**;
recall@0.25/0.75 +0.0130/+0.0524; tiny/small/medium/large +0.0045/+0.0369/+0.0275/+0.0236; border
+0.0292; dense +0.0268; mask mAP50/mAP50-95 +0.0548/+0.0388; empty-tile false-proposal rate **−0.0962**;
J1 fixed120 mIoU **+0.0705**; J1 paired **+3**; J1 fixed120 abstentions **−8**; J1 full-val mIoU
+0.0569.

**Residual gap.** Recall is 0.7505 for buildings ≥ 1000 px but 0.1621 for 50–200 px and 0.0068 for
< 50 px (tiny recall 88× below its gate). No architecture change and no small-object strategy was
chosen in this task; that is a research decision for the next one.

## 11. Tests

`python -m pytest tests/ -q` → **569 passed, 1 skipped** (Task 6M ended at 547 passed / 1 skipped; no
previously passing test was reduced or removed, and the one skip is the Ultralytics-only eval-mode
determinism check that needs the proposal env). `tests/test_task6m1_convergence_demo_fix.py` covers the
22 section-18 checks: Task 6M artifacts unchanged, source hashes match, snapshot matches, source data
and v0.2 unchanged, resume starts at epoch 19, frozen training config unchanged, no test read before
the 6M.1 freeze, test skipped when the val gate fails, fixed packs reused byte-for-byte, no GT in
inference, unsupported prompt exits 4, no parser call, no proposal-model call, all 6 OOD prompts
rejected, all 5 positive prompts accepted, v0.2 template coverage 100 % in both languages, CLI audit
4/4/4, ≥ 8 programs, no 4B, no model-family/imgsz/loss change, no GUI, no new dataset/download.

## 12. Git / storage

Commits: `fix: reject unsupported structured demo prompts`, then
`eval: complete native-vector proposal convergence audit`, then a docs handoff commit. Not committed:
checkpoints, run directories, `.conda`, the derived image export, caches, source imagery/vector data.

Watt was **not needed** in Task 6M.1: this task downloaded nothing (no new weights, packages or
datasets) and installed nothing, so no network transport tool was involved. The pre-existing Watt
instance is transport-only, is not owned by this project, and was left running, per the ownership
rule; no proxy, host, certificate or TLS setting was read or modified.

## 13. Recommended next step

The same configuration has now been trained to a normal early stop and still misses the proposal gate,
with the gap concentrated almost entirely in **small and tiny buildings**. The next task should decide
between: (a) an explicit small-object strategy for the same family (higher input resolution and/or
tiled inference — explicitly *not* chosen here), (b) a different proposal family or a small-object-
oriented architecture, or (c) re-scoping the structured task so tiny instances are out of the graded
target set. No such decision is taken in this task.

## 14. STOP

Task 6M.1 stops here: no Task 6N, no small-object strategy selection, no imgsz change, no other
backbone, no `[REF]`, no SRE, no GRCL/SCL, no GUI. Waiting for the ChatGPT audit and research decision.
