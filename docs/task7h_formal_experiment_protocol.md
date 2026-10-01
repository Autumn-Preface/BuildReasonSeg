# Task 7H — Formal Experiment Protocol (BuildReasonSeg-DevFreeze-2026-10)

> Frozen by Task 7H · **not executed by Task 7H** · machine-readable form:
> `evaluation/task7h_formal_experiment_protocol.json` · test lock: `evaluation/task7h_test_lock.json`
> Architecture freeze: `docs/task7h_development_architecture_freeze.md`

## 1. Data policy

Only the active canonical assets: **WHU-EA-NativeVector v1.0**, **BuildSpatialReason v0.2**,
**`scene_disjoint_v1`**.

* **train** — gradient updates only;
* **val** — checkpoint selection / early stopping / frozen threshold-free model selection only;
* **test** — only after every architecture and training choice is frozen.

**Historical access disclosure.** The `scene_disjoint_v1` test split was accessed once in **Task 6M** for an
earlier proposal-baseline J4-v2 audit. Future reporting must therefore **not** call the project test split
"never previously viewed" or "completely untouched"; the permitted phrasing is
**`final frozen-architecture test evaluation`**, because post-6M architecture selection did not use test
metrics. Task 7H itself read no test record.

## 2. Formal L3 population

The four canonical L3 programs `largest_to_{left_of,right_of,above,below}_to_nearest`.

| Split | Population | Expected |
|---|---|---|
| train | all valid v0.2 train records of the four programs | **1344** (left 323 / right 347 / above 338 / below 336) |
| val | all valid v0.2 val L3 records | **936** |

Both totals were re-verified in Task 7H from the tracked train/val splits (no test access).

## 3. Formal D-B1 training

* architecture: the exact frozen Task 7D D-B1 (deterministic field-weighted prototype), initialized from
  **fresh random trainable weights**, SAM2 frozen, exact field formulas, **BCE + Dice only**, no new loss and
  no architecture tuning;
* schedule copied verbatim from the authoritative `evaluation/task7d_training.json`:
  **optimizer AdamW · lr 0.0003 · weight decay 0.0001 · batch 8
  · max epochs 25 · early-stopping patience 5 ·
  AMP bfloat16 autocast + GradScaler (Task 7D D2 trainer) · checkpoint selection `MiniVal240 mIoU`**;
* formal seeds **[20261001, 20261002, 20261003]** — all three run only in a later training task;
* checkpoint selection: **val mIoU only**, each seed selected independently, never by test.

## 4. Reference policy in formal evaluation

Two results must be reported **separately** and never mixed into one headline number:

```text
practical predicted-reference chain:  U-C1 -> deterministic largest selector -> D-B1
oracle-reference diagnostic:          GT reference -> D-B1
```

The Task 7F F-R1 oracle-selected proposal is **not** a production result.

## 5. Baselines and ablations

| ID | System | Definition | Role |
|---|---|---|---|
| B-L3-0 | Z-B3 | visual + P_dir + P_near + direction embedding | baseline/ablation |
| B-L3-1 | D-B1 | deterministic relation-conditioned prototype | main architecture candidate |
| historical | Task 6Z visual-only / directional-only / nearest-only / geometry-only · Task 7D D-B2 · Task 7D D-B4 | frozen development controls | historical ablations only |

Documentation must keep **historical development ablations**, **future formally retrained seeds** and
**diagnostic oracle-reference values** clearly apart. Task 7H invents no new retrospective ablation.

## 6. Required metrics

* **mask**: mIoU, Dice, Pr@0.5;
* **counterfactual**: pair pass rate, own IoU, cross IoU, own-cross margin;
* **reference**: selected-reference mIoU, reference Pr@0.5, abstention rate, `NO_PROPOSALS`, `NO_ELIGIBLE`,
  `NOT_COVERED`, `SELECTION_WRONG`, `GEOMETRY_POOR`, `REFERENCE_OK`;
* **per relation**: left, right, above, below;
* **stratification**: target-area quartiles, reference-quality bins, boundary-distance quartiles;
* **efficiency**: train wall time, peak VRAM, inference time/tile, trainable parameter count.

## 7. Three-seed reporting

Each seed is reported separately, with mean ± standard deviation on val and on the final test; reporting only
the best test seed is forbidden. A crashed run is documented and resumed only from that run's own
checkpoint/state — the seed is never replaced.

## 8. Test lock

```text
status                     = LOCKED
architecture_head          = D-B1
reference_policy           = U-C1 deterministic largest
parser_role                = controlled-language/canonical interface
formal_seeds               = [20261001, 20261002, 20261003]
test_execution_authorized  = false
unlock_condition           = ChatGPT audit after formal train/val completion
```

Any future DSH task must read `evaluation/task7h_test_lock.json` before executing a test evaluation. Task 7H
does not unlock it.

## 9. Frozen limitations carried into the formal protocol

| ID | Limitation | Status |
|---|---|---|
| L-01 | Reference selection | unresolved practical bottleneck |
| L-02 | Proposal coverage | secondary unresolved bottleneck |
| L-03 | Proposal-mask geometry | not a major current bottleneck |
| L-04 | Free-form L3 language | controlled-language interface only |
| L-05 | nearest-only L2 | not validated as standalone final capability |
| L-06 | unseen-city/domain generalization | not established |

## 10. Claim statuses (no novelty or "first" claim)

| Claim | Status |
|---|---|
| C1 | SUPPORTED |
| C2 | SUPPORTED_WITH_LIMITATION |
| C3 | SUPPORTED |
| C4 | NOT_SUPPORTED |
| C5 | NOT_SUPPORTED |
| C6 | SUPPORTED_WITH_LIMITATION |
| C7 | NOT_SUPPORTED |
| C8 | NOT_SUPPORTED |
| C9 | NOT_SUPPORTED |

Final recommendation (Task 7H, Part L):

`等待 ChatGPT 审核 Task 7H 的开发版架构冻结与正式实验协议；在审核通过前不启动正式三种子训练，不解锁 test。`
