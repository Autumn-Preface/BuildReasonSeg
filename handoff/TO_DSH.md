# TO_DSH — Task 6M.1: Complete Proposal Training + Correct Demo Gates

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `b9f49f8431f941b7f4e0196fca2a275d257816a8`
>
> Predecessor verdict: Task 6M = `PROPOSAL_MODEL_NEEDS_IMPROVEMENT`
>
> This is a narrow corrective continuation. Do not redesign the architecture. Do not start Task 6N.

## 0. Execution role
Follow this file literally. Do not make research-direction decisions. If an unexpected issue would require changing the model family, dataset, loss, image size, split, parser architecture, executor semantics, or success thresholds, STOP and report it instead of improvising.

All user-facing DSH output must be Chinese.

## 1. Why this task exists
Task 6M is valid but incomplete:

- YOLO26m-seg completed only **18 / 80** configured epochs because the DSH session wall-clock budget ended.
- The curve was still improving.
- validation proposal recall@0.50 = **0.6333**.
- tiny recall@0.50 = **0.0023**.
- J1-v2 fixed120 mIoU = **0.2801**, paired pass = **4/20**.
- J4-v2 fixed120 mIoU = **0.2575**, paired pass = **5/20**.
- Parser is already ready at **1.0000 val accuracy / macro-F1**.
- CLI inference works without GT, but unsupported instruction handling is wrong: `"Write a poem about the sea."` was mapped to `leftmost` with exit code 0.

The purpose of 6M.1 is exactly:

1. preserve Task 6M evidence;
2. finish the same YOLO26m-seg configuration to its planned convergence horizon;
3. re-evaluate on validation;
4. run a new test evaluation only if the validation development gate passes;
5. fix and re-audit unsupported-instruction rejection;
6. produce the strongest honest structured Demo baseline before research Task 6N.

## 2. Frozen assets — MUST NOT change
Do not modify:

- `datasets/whu_native_vector/v1.0/`
- `datasets/build_spatial_reason/v0.2/`
- `datasets/build_spatial_reason/v0.1.1/`
- `evaluation/task6m_*.json`
- Task 6J / 6K / 6K.1 / 6L artifacts
- Task 3B relation semantics
- 20 canonical programs
- `scene_disjoint_v1`
- Task 6M fixed eval packs
- Task 6M parser checkpoint
- Task 6M inference/test results

Task 6M remains historical evidence and must stay reproducible.

## 3. No architecture changes
Keep exactly:

- proposal family: `YOLO26m-seg`
- package: existing Task 6M project-local proposal environment
- pretrained lineage: Task 6M YOLO26m-seg run
- imgsz: **640**
- batch: **16**
- workers: **4**
- seed: **20260812**
- AMP: enabled
- deterministic: enabled
- max total epoch horizon: **80**
- patience: **15**
- train = train1
- val = train2
- test = test

Do NOT use YOLO26l/x, another framework, another imgsz, tiling/window inference, custom loss weights, a different augmentation policy, tiny-instance filtering, 4B, `[REF]`, SRE, GRCL/SCL, or GUI.

If this exact configuration is still inadequate after convergence, report that result. A later task will decide what to change.

# PART A — Preserve the 18-epoch state

## 4. Snapshot local Task 6M checkpoints before continuation
Current Task 6M recorded:

- best.pt SHA256:
  `fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44`
- last.pt SHA256:
  `ea998bda37dd2dcb2cc7bb5e19f6d15b7a205a137c9cc2866c508a45513f860e`

Before any training:

1. recompute both hashes;
2. require exact match;
3. copy, never move, both files plus `results.csv` and minimum resume metadata into:
   `artifacts/checkpoints/task6m1/source_epoch18_snapshot/`
4. verify copied hashes;
5. never overwrite this snapshot.

Write:
`evaluation/task6m1_source_checkpoint_audit.json`

If either original hash does not match, STOP with `SOURCE_CHECKPOINT_MISMATCH`.

## 5. Safe continuation rule
Preferred source = Task 6M **last.pt**.

Use Ultralytics resume only if it can continue from the copied Task 6M.1 snapshot/run state without mutating original Task 6M evidence.

Before long training, verify:

- next epoch is 19;
- optimizer/scheduler state restored;
- imgsz/batch/seed/split remain frozen;
- output path is Task 6M.1-local;
- original Task 6M best/last hashes remain unchanged.

If safe resume cannot satisfy these requirements, STOP with `SAFE_RESUME_UNAVAILABLE`.

Do NOT silently convert this into a fresh run or new fine-tuning schedule.

# PART B — Continue training

## 6. Training horizon
Continue from epoch 18 toward total cap 80.

Stop only when:
1. early stopping fires with patience 15; or
2. epoch 80 completes; or
3. NaN/Inf / unrecoverable OOM / source integrity problem occurs.

Do not stop merely because an arbitrary DSH subtask duration has elapsed.

If the orchestration environment itself imposes a hard job/session limit:
- preserve the latest completed state;
- record the latest completed epoch;
- do not claim convergence;
- return `CONTINUATION_INTERRUPTED`;
- do not run downstream graded evaluation.

Create:
`evaluation/task6m1_training_summary.json`

Required fields:
- source checkpoint hashes;
- start/end epoch;
- stop reason;
- best epoch in combined lineage;
- best/final mask mAP50 and mAP50-95;
- precision/recall;
- losses;
- wall time;
- mean epoch time;
- peak VRAM;
- NaN/Inf;
- final best/last paths and SHA256.

# PART C — Validation-only evaluation

## 7. Proposal metrics
Only after normal training completion/early-stop, use canonical native masks on full validation.

Compute:
- recall@0.25 / 0.50 / 0.75;
- mask AP50 / AP50-95;
- mean/median best GT→proposal IoU;
- proposals/tile;
- empty-tile false-proposal rate;
- tiny recall@0.50;
- border recall@0.50;
- dense recall@0.50;
- small / medium / large breakdown.

Write:
`evaluation/task6m1_proposal_val.json`

## 8. Threshold sweep
Use the SAME grid only:

- confidence ∈ `{0.05, 0.10, 0.25}`
- max_det ∈ `{100, 300}`

Selection hierarchy:
1. target recall@0.50;
2. oracle-program structured performance;
3. lower proposal burden when otherwise tied.

Freeze:
`evaluation/task6m1_inference_config_frozen.json`

No test metric may be read before this file exists.

# PART D — J1-v2 validation

## 9. Run existing protocol
Use oracle program + new predicted proposals + canonical native GT for scoring only + exact frozen Task 6M val fixed120/paired20 packs.

Write:
`evaluation/task6m1_j1v2_val.json`

Development gate remains:

- overall proposal recall@0.50 >= **0.92**
- tiny recall@0.50 >= **0.60**
- fixed120 strict mIoU >= **0.50**
- paired pass >= **14/20**
- abstentions <= **20/120**

Do NOT lower thresholds after seeing results.

## 10. Branch
If all five gates pass: proceed to Part E.

If any gate fails:
- do NOT rerun J4/test;
- do NOT change model/config;
- write failure attribution;
- final proposal verdict = `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`;
- proceed only to CLI fix, tests, docs, commit, STOP.

# PART E — Test only after val gate pass

## 11. Parser
Reuse existing Task 6M v0.2 parser checkpoint:

`eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3`

Do not retrain. If file missing/hash mismatch, STOP.

## 12. J4-v2 test
Task 6M already inspected test once for the earlier 18-epoch proposal model. Therefore this is not a pristine unseen-test claim.

If validation passes, run exactly one Task 6M.1 test evaluation of the new converged model and label it:

`second_checkpoint_test_evaluation_after_predeclared_val_gate`

No test-driven tuning afterward.

Write:
`evaluation/task6m1_j4v2_test.json`

Demo thresholds:
- fixed120 mIoU >= **0.40**
- paired pass >= **14/20**

Do not describe this as a paper-final untouched test.

# PART F — Fix unsupported instructions

## 13. Deterministic domain gate
Fix `predict_structured.py` so unsupported/out-of-domain prompts fail **before ProgramHead and YOLO inference**.

A prompt must contain:

### A. at least one building/object anchor
Chinese:
`建筑`, `建筑物`, `建筑区域`, `房屋`, `楼`

English:
`building`, `buildings`, `structure`

AND

### B. at least one supported relation/selection anchor
Chinese:
`最大`, `最小`, `最左`, `最右`, `最上`, `最下`,
`最靠左`, `最靠右`, `最靠上`, `最靠下`,
`最近`, `左侧`, `右侧`, `上方`, `下方`,
`左边`, `右边`, `上面`, `下面`

English:
`largest`, `smallest`, `leftmost`, `rightmost`, `topmost`, `bottommost`,
`nearest`, `closest`, `left of`, `right of`, `above`, `below`

If A or B is absent:

- do not call ProgramHead;
- do not run YOLO;
- write `result.json`;
- status = `unsupported_instruction`;
- abstention_reason = `out_of_domain_prompt`;
- exit code = **4**.

This is a closed-Demo grammar/domain guard, not open-domain OOD detection.

## 14. Required OOD checks
MUST reject with exit 4:

- `Write a poem about the sea.`
- `今天天气怎么样？`
- `请总结这张图片。`
- `检测道路。`
- `segment the airplane`
- empty/whitespace prompt

MUST enter ProgramHead path:

- `分割面积最大的建筑物。`
- `找出最左侧的建筑区域。`
- `分割面积最大的建筑物右侧最近的建筑物。`
- `segment the building nearest to the right of the largest building`
- `找出最小建筑物上方的建筑。`

If a required positive prompt is rejected, fix only this deterministic gate vocabulary/logic. Do not retrain parser.

# PART G — Representative CLI audit

## 15. Audit design
Use validation images only. Annotation files must be unavailable to CLI.

Audit at least 12 supported prompts:

- 4 L1
- 4 L2
- 4 L3
- >= 8 distinct canonical program ids
- include largest/smallest, directional, nearest, and compositional L3 examples

Take expected programs from already-frozen Task 6M validation packs.

Also run all 6 OOD prompts.

Write:
`evaluation/task6m1_demo_cli_audit.json`

CLI correctness gate:

- supported prompts reach parser path;
- 12/12 expected program matches;
- no GT used;
- output/abstention explicit;
- every OOD prompt exits 4 before parser/proposal inference;
- no unsupported prompt silently maps to a program.

# PART H — Error attribution

## 16. If proposal gate fails
Write:
`evaluation/task6m1_error_attribution.json`

Compare Task 6M epoch18 vs Task 6M.1:

- overall recall@0.50 delta;
- tiny/small/medium/large recall delta;
- border/dense delta;
- J1 fixed120 mIoU delta;
- J1 paired delta;
- abstention delta.

Allowed diagnosis only:

- `CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`
- `CONVERGENCE_DID_NOT_HELP_MATERIALLY`
- `CONVERGENCE_PASSES_GATE`

Do not choose a new architecture in this task.

# PART I — Final verdict

## 17. Exactly one
Allowed:

- `STRUCTURED_DEMO_READY`
- `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`
- `CONTINUATION_INTERRUPTED`
- `SOURCE_CHECKPOINT_MISMATCH`
- `SAFE_RESUME_UNAVAILABLE`
- `INVALID_EXPERIMENT`

`STRUCTURED_DEMO_READY` requires:

- normal training completion/early-stop;
- all five validation J1 gates pass;
- Task 6M.1 test fixed120 mIoU >= 0.40;
- Task 6M.1 test paired >= 14/20;
- corrected CLI gate passes.

Write:
`evaluation/task6m1_verdict.json`

# PART J — Tests

## 18. Focused tests
At minimum:

1. all Task 6M tracked artifacts unchanged;
2. source checkpoint hashes match Task 6M;
3. snapshot hashes match originals;
4. source data unchanged;
5. v0.2 unchanged;
6. resume starts next epoch 19;
7. frozen training config unchanged;
8. no test read before 6M.1 config freeze;
9. test skipped if val gate fails;
10. old fixed packs reused byte-for-byte;
11. no GT in inference;
12. unsupported prompt exits 4;
13. unsupported prompt does not call parser;
14. unsupported prompt does not call proposal model;
15. all 6 OOD prompts rejected;
16. all 5 positive prompts enter parser path;
17. CLI audit = 4 L1 / 4 L2 / 4 L3;
18. >=8 program ids;
19. no 4B;
20. no model-family/imgsz/loss change;
21. no GUI;
22. no new dataset/download.

Run:
`python -m pytest tests/ -q`

Task 6M ended at 547 passed, 1 skipped. Do not reduce passing tests.

# PART K — Required artifacts

Always:
- `evaluation/task6m1_source_checkpoint_audit.json`
- `evaluation/task6m1_training_summary.json`
- `evaluation/task6m1_proposal_val.json` if normal training completion
- `evaluation/task6m1_inference_config_frozen.json` if normal training completion
- `evaluation/task6m1_j1v2_val.json` if normal training completion
- `evaluation/task6m1_demo_cli_audit.json`
- `evaluation/task6m1_error_attribution.json` if proposal evaluation runs
- `evaluation/task6m1_verdict.json`
- `docs/task6m1_proposal_convergence_and_demo_fix.md`

Only if validation gate passes:
- `evaluation/task6m1_j4v2_test.json`

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

# PART L — Git/storage
Do not commit checkpoints, `.conda`, run dirs, source imagery/vector files, derived export, caches.

Commit only code/tests/small JSON/docs/handoff.

Recommended commits:
1. `fix: reject unsupported structured demo prompts`
2. `eval: complete native-vector proposal convergence audit`
3. optional docs handoff commit

# PART M — Model policy
Default:
- DeepSeek V4.1 Flash + High

Only use Flash + Max for a genuine implementation/runtime bug.

Do not use V4 Pro by default.

# PART N — STOP
After Task 6M.1 STOP.

Do not:
- start Task 6N;
- choose a small-object strategy;
- change imgsz;
- train another backbone;
- add `[REF]`;
- add SRE;
- add GRCL/SCL;
- build GUI.

Wait for ChatGPT audit and research decision.
