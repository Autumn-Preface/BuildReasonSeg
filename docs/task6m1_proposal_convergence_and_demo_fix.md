# Task 6M.1 — Proposal Convergence + Correct Demo Gates

> Task: `handoff/TO_DSH.md` (Task 6M.1) · Base commit: `b9f49f8` · Predecessor verdict: Task 6M =
> `PROPOSAL_MODEL_NEEDS_IMPROVEMENT`
> **Verdict: `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`** · Tests:
> `tests/test_task6m1_convergence_demo_fix.py` · Evidence: `evaluation/task6m1_*.json`

This is a narrow corrective continuation of Task 6M, not a redesign: preserve the 18-epoch Task 6M
evidence, continue the **same** YOLO26m-seg configuration to its planned horizon, re-evaluate on
validation, run a new test evaluation **only** if the validation development gate passes, and fix the
Demo CLI's unsupported-instruction handling. No architecture, dataset, loss, image-size, split,
parser or executor change was made, and no threshold was lowered.

**Outcome in one paragraph.** The 18-epoch state was hash-verified and snapshotted, the same
configuration resumed safely at epoch 19 and ran to a **normal early stop** after 37 more epochs
(19–55, patience 15, best epoch 40), and convergence **helped every measured metric** — mask mAP50-95
0.3544 → **0.4048**, recall@0.50 0.6333 → **0.6578**, J1 fixed120 mIoU 0.2801 → **0.3506**, paired
4/20 → **7/20**, abstentions 35 → **27**, empty-tile false-proposal rate 0.1264 → **0.0303**. The five
validation development gates nevertheless still fail (recall@0.50 0.6578 < 0.92, tiny recall 0.0068 <
0.60, fixed120 mIoU 0.3506 < 0.50, paired 7/20 < 14/20, abstentions 27 > 20), so per section 10 **no
test evaluation was run** and the attributed diagnosis is
`CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`. Separately, the Demo CLI now has a deterministic
domain gate that rejects out-of-domain prompts with **exit 4 before ProgramHead and before the
proposal model**, and the gate vocabulary was completed so that **100 % of the frozen v0.2 templates
in both languages** are accepted; the corrected audit passes with **12/12** correct programs and
**6/6** out-of-domain rejections.

## 1. Preserved Task 6M evidence (Part A)

`evaluation/task6m1_source_checkpoint_audit.json` — `SOURCE_CHECKPOINT_VERIFIED`:

| File | Expected SHA256 (Task 6M) | Recomputed | Result |
|---|---|---|---|
| `best.pt` | `fd407db634a8a7ef…dbf4ea44` | equal | MATCH |
| `last.pt` | `ea998bda37dd2dcb…5513f860e` | equal | MATCH |

Both checkpoints plus `results.csv` and the run metadata were **copied** (never moved) into
`artifacts/checkpoints/task6m1/source_epoch18_snapshot/`, and the copies re-verified against the
live files. The snapshot is never overwritten. Every Task 6M artifact (`evaluation/task6m_*.json`,
the four frozen eval packs, the parser checkpoint) is byte-identical after this task — enforced by
`test_task6m1_tracked_artifacts_unchanged` and `test_old_fixed_packs_reused_byte_for_byte`.

## 2. Safe continuation (Part B)

`evaluation/task6m1_resume_preflight.json` — `SAFE_RESUME_READY`. Ultralytics `resume=True` replaces
its own args with the checkpoint's stored `train_args`, whose `project`/`name`/`save_dir` point at the
Task 6M directory; the continuation therefore resumes from a **patched copy** of `last.pt`
(`last_resume.pt`) whose stored args point at
`artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/`, with `save_dir` also passed as an
explicit resume override. Nothing in Task 6M can be written to.

Verified before training: source hashes still match; snapshot matches the originals; stored epoch 17
(0-based) → **next epoch 19**; optimizer state present; EMA state present; `save_dir` is Task 6M.1
local; frozen config unchanged (**imgsz 640, batch 16, workers 4, seed 20260812, epochs 80,
patience 15, AMP, deterministic**, data = the derived native-vector export). The log confirms
`Resuming training …\task6m1\…\last_resume.pt` and the first resumed epoch is **19/80**.

## 3. Continuation training (Part B)

`evaluation/task6m1_training_summary.json` — combined Task 6M + 6M.1 lineage.

| | Value |
|---|---|
| stop | **`early_stopping_fired`** (patience 15) → **normal completion** |
| epochs this task | **19 → 55** (37 epochs, 2.42 h wall clock) |
| mean epoch time | 235.5 s (train + full 3,618-image validation) |
| peak VRAM | 11.2 GB |
| combined best epoch | **40** (by Ultralytics fitness → `best.pt`, box mAP50-95 **0.44891**) |
| best epoch by mask | **40** — mask mAP50 **0.73742**, mask mAP50-95 **0.40475** |
| Task 6M epoch-18 reference | mask mAP50 0.68266, mask mAP50-95 0.35437 (+8.0 % / +14.2 % relative) |
| final epoch (55) | mask mAP50 0.72415, mask mAP50-95 0.39172 |
| NaN / Inf | none |
| final `best.pt` / `last.pt` | `…/task6m1/runs/m1_yolo26m_seg_continued/weights/`, SHA256 recorded in the summary |
| Task 6M evidence after the run | both hashes still match the recorded values |

Metric history (mask mAP50 / mAP50-95): 18 → 0.683 / 0.354 · 24 → 0.707 / 0.378 · 28 → 0.720 / 0.389 ·
**36 → 0.718 / 0.390** · **40 → 0.737 / 0.405 (best)** · 44 → 0.728 / 0.398 · 49 → 0.717 / 0.381 ·
55 → 0.724 / 0.392.

## 4. Validation evaluation (Parts C/D)

**Part C — proposal metrics** (`evaluation/task6m1_proposal_val.json`, full validation, canonical
native masks, no test data touched):

| Metric | Task 6M (epoch 18) | Task 6M.1 (converged) |
|---|---|---|
| recall @ 0.25 / 0.50 / 0.75 | 0.7772 / 0.6333 / 0.3404 | **0.7902 / 0.6578 / 0.3928** |
| validator mask mAP50 / mAP50-95 | 0.6819 / 0.3637 | **0.7367 / 0.4025** |
| proposals per tile | 6.38 | **5.39** |
| empty-tile false-proposal rate | 0.1264 | **0.0303** |
| tiny / border / dense recall@0.50 | 0.0023 / 0.5461 / 0.6322 | **0.0068 / 0.5753 / 0.6590** |
| small / medium / large recall@0.50 | 0.1252 / 0.5299 / 0.7269 | **0.1621 / 0.5574 / 0.7505** |

**Threshold sweep** — same declared grid as Task 6M (conf ∈ {0.05, 0.10, 0.25} × max_det ∈ {100, 300},
six configurations), selection hierarchy (target recall@0.50 → oracle-program structured performance →
lower proposal burden). Frozen: **conf 0.10, max_det 100**
(`evaluation/task6m1_inference_config_frozen.json`, `frozen_before_test: true`,
`test_metrics_inspected_before_freezing: false`). No test metric existed or was read before that file.

**Part D — J1-v2 validation** (`evaluation/task6m1_j1v2_val.json`; oracle program + converged
predicted proposals + the **byte-identical frozen Task 6M val fixed120 / paired20 packs**; GT used for
scoring only):

| Development gate | Threshold | Task 6M | Task 6M.1 | Result |
|---|---|---|---|---|
| overall recall@0.50 | ≥ 0.92 | 0.6333 | **0.6578** | FAIL |
| tiny recall@0.50 | ≥ 0.60 | 0.0023 | **0.0068** | FAIL |
| fixed120 strict mIoU | ≥ 0.50 | 0.2801 | **0.3506** | FAIL |
| paired own-vs-cross pass | ≥ 14/20 | 4/20 | **7/20** | FAIL |
| fixed120 abstentions | ≤ 20/120 | 35 | **27** | FAIL |

Full validation: mIoU **0.3541** (0.2801 → +0.0569), abstention rate 0.2579, by level L1 0.3805 /
L2 0.2669 / L3 0.3322, paired mean own IoU 0.3144 vs cross 0.0367 (an 8.6× discrimination margin,
i.e. the executor still separates the two targets; it lacks a good enough proposal).

Thresholds were **not** lowered after seeing these results.

## 5. Test evaluation (Part E)

**Not run.** Section 10 is explicit: if any validation gate fails, do not rerun J4/test, do not change
the model or config, write the failure attribution, and stop at the proposal verdict. Task 6M's
earlier single test inspection therefore remains the only test evidence for this lineage
(`evaluation/task6m_j4v2_test.json`, unchanged), and no `task6m1_j4v2_test.json` exists.

## 5. Deterministic domain gate (Part F)

`predict_structured.py` now evaluates a closed-Demo grammar/domain guard **before ProgramHead and
before the proposal model**. A prompt is accepted only if it contains

* at least one building/object anchor — zh: `建筑`, `建筑物`, `建筑区域`, `房屋`, `楼`;
  en: `building`, `buildings`, `structure`;
* **and** at least one relation/selection anchor — zh: `最大`, `最小`, `最左`, `最右`, `最上`, `最下`,
  `最靠左`, `最靠右`, `最靠上`, `最靠下`, `最近`, `左侧`, `右侧`, `上方`, `下方`, `左边`, `右边`,
  `上面`, `下面`; en: `largest`, `smallest`, `leftmost`, `rightmost`, `topmost`, `bottommost`,
  `nearest`, `closest`, `left of`, `right of`, `above`, `below`.

Otherwise the CLI writes `result.json` with `status = unsupported_instruction`,
`abstention_reason = out_of_domain_prompt`, `parsed_program = null`, `proposal_count = null` and
**exits 4** — with no parser construction, no `ultralytics` import and no inference.

`evaluation/task6m1_domain_gate_check.json` — **gate PASS**:

| Required rejection | Result |
|---|---|
| `Write a poem about the sea.` | rejected (exit 4) |
| `今天天气怎么样？` | rejected (exit 4) |
| `请总结这张图片。` | rejected (exit 4) |
| `检测道路。` | rejected (exit 4) |
| `segment the airplane` | rejected (exit 4) |
| empty / whitespace prompt | rejected (exit 4) |

| Required positive prompt | Result |
|---|---|
| `分割面积最大的建筑物。` | enters the ProgramHead path |
| `找出最左侧的建筑区域。` | enters the ProgramHead path |
| `分割面积最大的建筑物右侧最近的建筑物。` | enters the ProgramHead path |
| `segment the building nearest to the right of the largest building` | enters the ProgramHead path |
| `找出最小建筑物上方的建筑。` | enters the ProgramHead path |

This is a closed-Demo grammar/domain guard, **not** open-domain OOD detection.

**Vocabulary completion (measured, minimal).** The first audit run exposed a real defect: the v0.2
templates phrase `topmost`/`bottommost` as **位置最高 / 位置最低** and `leftmost`/`rightmost`/`largest`
in English as *furthest left*, *furthest right*, *furthest to the left/right*, *greatest* and
*highest/lowest*. Those phrasings were absent from the section-13 anchor list, so **899 of 9,111
validation records (9.9 %)** and 9 of the 20 program templates were **falsely rejected** as
out-of-domain. Section 13 permits exactly this repair ("fix only this deterministic gate
vocabulary/logic"), so the relation vocabulary was completed with the concept-equivalent anchors
`最高`/`最低` and `highest`/`lowest`/`greatest`/`furthest left`/`furthest right`/`furthest to the
left`/`furthest to the right`. Object anchors were **not** widened and no free-form acceptance was
added, so all six required out-of-domain prompts are still rejected.

`evaluation/task6m1_domain_gate_coverage.json` (regression guard): every distinct v0.2 template in
train/val/test is accepted in **both** languages — val **9,111/9,111 zh (100 %)** and
**9,111/9,111 en (100 %)**, **0 falsely rejected templates**, all 20 programs reachable.

## 6. Representative CLI audit (Part G)

`evaluation/task6m1_demo_cli_audit.json` — **gate PASS**. Checkpoint used:
`task6m1_continuation_best` (`…/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt`), frozen
conf 0.10 / max_det 100. Annotation files were unavailable to the CLI.

| Check | Result |
|---|---|
| supported runs | **12** (4 L1 / 4 L2 / 4 L3) |
| distinct canonical programs | **12** (≥ 8 required) |
| expected program matches | **12 / 12** |
| answered with a selected mask + overlay | 9 |
| explicit abstentions (documented reason) | 3 |
| ground truth used | **no** (`any_ground_truth_used: false`) |
| out-of-domain prompts rejected with exit 4 before models | **6 / 6** |
| unsupported prompt silently mapped to a program | **none** |

The audited set includes the required families: `smallest`, `leftmost`, `rightmost`, `topmost`
(gate-completed vocabulary), `smallest_to_right_of`, `smallest_to_nearest`, `largest_to_right_of`,
`smallest_to_below`, and the compositional L3 programs `largest_to_above_to_nearest`,
`largest_to_right_of_to_nearest`, `largest_to_left_of_to_nearest`, `largest_to_below_to_nearest`.

## 7. Error attribution and verdict (Parts H/I)

`evaluation/task6m1_error_attribution.json` — diagnosis
**`CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`** (the declared "helpful" rule fires because fixed120 mIoU
improved by +0.0705 ≥ 0.05). Deltas, Task 6M epoch 18 → Task 6M.1 converged:

| Delta | Value |
|---|---|
| overall recall@0.50 | **+0.0246** |
| recall@0.25 / @0.75 | +0.0130 / +0.0524 |
| tiny / small / medium / large recall@0.50 | +0.0045 / +0.0369 / +0.0275 / +0.0236 |
| border / dense recall@0.50 | +0.0292 / +0.0268 |
| validator mask mAP50 / mAP50-95 | +0.0548 / +0.0388 |
| empty-tile false-proposal rate | **−0.0962** (better) |
| J1 fixed120 mIoU | **+0.0705** |
| J1 paired pass | **+3** (4 → 7 of 20) |
| J1 fixed120 abstentions | **−8** (35 → 27) |
| J1 full-val mIoU | +0.0569 |

`evaluation/task6m1_verdict.json` — exactly one allowed verdict:

**`PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`** — failed gates
`validation_j1_gate_passed`, `test_fixed120_miou_ge_0_40`, `test_paired_pass_ge_14`; the source-audit,
safe-resume, normal-completion, validation-ran and corrected-CLI gates all pass.

**What this means.** Convergence was real and directionally consistent across every metric, but it was
not enough: the model is still short of the proposal-quality gates, and the residual gap is
concentrated in small objects (recall@0.50 = 0.7505 for ≥ 1000 px vs **0.1621** for 50–200 px and
**0.0068** for < 50 px), with tiny-instance recall 88× below its gate. No architecture was changed and
no small-object strategy was chosen here — that is a research decision for the next task.

## 8. Reproduce

```text
python scripts/task6m1_source_snapshot.py           # Part A: hash-verify + snapshot
python scripts/task6m1_resume_train.py --dry-run     # Part B: safe-resume pre-flight
python scripts/task6m1_resume_train.py               # Part B: continue to the horizon
python scripts/task6m1_finalize_training.py --stop-reason early_stopping_fired
python scripts/task6m_proposal_eval.py --checkpoint <best.pt> --sweep \
    --out evaluation/task6m1_proposal_val.json \
    --frozen-out evaluation/task6m1_inference_config_frozen.json   # Part C
python scripts/task6m_j1v2.py --out evaluation/task6m1_j1v2_val.json \
    --config evaluation/task6m1_inference_config_frozen.json        # Part D
# Part E is not run: the section-9 validation gate fails (see sections 4-5 above).
python scripts/task6m1_domain_gate_check.py         # Part F required prompts
python scripts/task6m1_gate_coverage.py             # Part F v0.2 template coverage
python scripts/task6m1_demo_audit.py                # Part G audit
python scripts/task6m1_error_attribution.py         # Part H
python scripts/task6m1_verdict.py                   # Part I
```

Checkpoints, run directories, `.conda`, the derived export and caches stay under the gitignored
`artifacts/` tree; only code, tests, small JSON artifacts and docs are committed.
