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

# FROM_DSH — Task 6G Report: Dense Query–Visual Spatial Grounding Map v0.1

_This file holds the Task 6G report. The Task 6F report is preserved in git history and in
`docs/task6f_target_aware_box_query.md`; Task 6E in `docs/task6e_explicit_spatial_tokens.md`._

Full design notes: `docs/task6g_dense_spatial_grounding.md`, ADR-018.

## 1. Verdict

**`DENSE_GROUNDING_IMPLEMENTATION_FAILED`.**

The grid oracle and the head implementation are verified, but **G0 cannot pass its 20-record
overfit gate after an implementation audit**, so per section 11 the task stops before G1. G0 ends
at 1500 steps with point-inside-own **0.10** (gate ≥ 19/20), paired point selection **0/10** (gate ≥
9/10) and heatmap Dice **0.125** (gate ≥ 0.80). The implementation audit
(`evaluation/task6g_g0_audit.json`) found **no defect**: train/eval query hiddens are bit-identical,
heatmap logits are healthy (std 3.3, no saturation), BCE and Dice flow comparable gradients into the
head, SAM2 stays bit-frozen through a training step, and the one-step smoke passed before training.
The frozen recipe genuinely cannot make the dot-product map localize even 20 memorized records: the
heatmap converges to a near-flat field (sigmoid mean 0.013 ≈ the target occupancy, peakiness 0.086)
whose argmax is image-content noise (errors 44–225 px; for one image both instructions pinned the
**identical** cell). This is the same query-signal bottleneck Task 6F measured (instruction
correlation r ≈ 0.48), now in map space: four readouts (6D.1 `[SEG]`, 6E tokens, 6F box head, 6G
dense map) fail identically because the query's instruction-dependent component is too weak to
select the target. G1, the representation diagnosis and G2 were **not run**; no SAM2 training and no
GT geometry in any prompt.

## 2. Frozen Task 6F Evidence (section 1)

Not rerun: F0 box IoU 0.9046 / paired 10/10; F1 epoch 8 val box IoU 0.0250, center-inside 0.017,
coordinate MAE 0.1543, geometry paired 0/20; query same-image centered cosine 0.626, different-image
0.018, hidden-vs-GT-box correlation 0.476; `[SEG]` emission 1.0, EOS 1.0. Also frozen: Oracle BOX
0.7506 / 20/20 and Oracle POINT 0.4876 / 18/20. The stronger claim "8 epochs were simply too few"
is **not** frozen.

## 3. Grid Oracle (sections 2–3)

The frozen Task 6D interior point, snapped to each grid's nearest cell centre, through the official
frozen SAM2 positive-point prompt on the fixed 120 val + 20 paired material
(`evaluation/task6g_grid_oracle.json`):

| Grid | strict mIoU | Dice | paired | mean displacement (512 px) |
|---|---|---|---|---|
| continuous | 0.4876 | — | 18/20 | — |
| 64 | 0.4950 | 0.6151 | 17/20 | 2.23 |
| 128 | 0.4894 | 0.6065 | 17/20 | 1.02 |
| 256 | **0.4883** | 0.6061 | **18/20** | 0.50 |

## 4. Selected Spatial Feature Level

**256×256** — the smallest grid with paired ≥ 18/20 and mIoU ≥ 0.4576 (the only qualifying grid:
64/128 pass the mIoU bar but drop one paired image). Measured SAM2 levels, by spatial size (never
list order): 64×64 = the 256-channel main image embedding; 128×128 = the 64-channel high-res level;
256×256 = the **32-channel** high-res level used by the head.

## 5. DenseSpatialGroundingHead (sections 5–6)

`q = LayerNorm(2048) → Linear(2048,128)` on the reused `[BOX]` hidden; `K = Conv1x1(32,128)` on the
frozen SAM2 256×256 feature; `heatmap_logits = dot(q, K)/sqrt(128) + scalar_bias` (270 593
parameters). No transformer block, no relation module, no decoder. SAM2 encoder + feature tensors
frozen: the one-step smoke records `sam2_max_abs_delta: 0.0` while the `[BOX]`/`[SEG]` rows and all
head parameters move with non-zero gradient and 16 ordinary rows + the base table stay exactly
unchanged (`evaluation/task6g_token_setup.json`, `passed: true`).

## 6. Target Heatmap / Loss (sections 7–8)

GT = the instruction-selected target mask, area-downsampled to 256×256 soft occupancy (mass > 0
asserted per sample). `L_heatmap = 1.0·BCEWithLogits + 1.0·SoftDice(sigmoid)`;
`L_total = 1.0·L_reasoning + 2.0·L_heatmap`; both raw losses recorded; no SmoothL1/GIoU/mask-loss/
contrastive/weight sweep. Predicted point = argmax cell centre (deterministic first-maximum rule);
soft-argmax reported as a diagnostic only; no threshold tuning and no GT repair.

## 7. Trainable/Frozen Parameters (section 10)

Train: text LoRA (392 tensors / 17 432 576), `{[SEG], [BOX]}` rows (1 tensor / 4 096),
DenseSpatialGroundingHead (270 593). Frozen: Qwen base, visual tower, all SAM2, Task 6F box head,
Task 6D grounding head, Projection MLP, no `<loc_*>` rows present. Optimizer coverage recorded:
lora 1e-4 / token 3e-4 / decoder (dense head) 3e-4, weight decay 0 on the rows, clip 1.0, cosine over
the true step budget.

## 8. G0 Overfit (section 11)

20 records / 10 pairs, ≤1500 steps, corrected horizon, real query-to-heatmap evaluation.

| Step | inside own | heatmap Dice | paired selection | distinct points |
|---|---|---|---|---|
| 500 | 0.05 | 0.103 | 0/10 | 2/10 |
| 1000 | 0.10 | 0.121 | 0/10 | 6/10 |
| 1500 | 0.10 | 0.125 | 0/10 | 7/10 |

Reasoning CE converged to 0.0000; BCE ≈ 0.02; the **Dice term never moved** (0.96–0.99 throughout).
Errors 44–225 px (mean 139 px); both instructions of one image pinned the identical cell.
`evaluation/task6g_g0_overfit.json` + `evaluation/task6g_g0_audit.json`.

## 9. G1 Training Curve

**Not run** — section 11 stops the task when G0 cannot pass after the audit.

## 10. Spatial Localization Metrics

G0 stage only (20 records): inside 0.10, heatmap Dice 0.125, binary-IoU@0.5 0.10,
mean 512-px point error 139, mean peakiness 0.086. `evaluation/task6g_spatial_eval.json` is
deliberately absent (it is a G1 product).

## 11. Paired Point Probe

G0 stage (10 pairs, `evaluation/task6g_paired_probe.json`): paired point selection **0/10**,
distinct points **7/10**, mean same-image point distance 0.336, mean same-image heatmap IoU 0.227 —
the heatmap responds to the instruction (7/10 distinct argmax cells) but is image-dominated and
wrong.

## 12. Representation/Fusion Diagnostics

**Not run at G1** (G1 did not run). The G0 evidence stands in: a near-flat heatmap whose argmax is
image-content noise, consistent with Task 6F's frozen query diagnosis (instruction correlation
0.476, same-image centered cosine 0.626). The question "does the weak target-specific component
become usable when matched against a dense spatial feature map?" is answered **no** at G0 scale.

## 13. G2 Segmentation

**Not run** (section 16 gates it on G1). `evaluation/task6g_segmentation_eval.json` is deliberately
absent. Frozen ceilings: Task 6C P_C 0.10604 / 0-20; continuous Point Oracle 0.4876 / 18-20;
continuous Box Oracle 0.7506 / 20-20.

## 14. Paired Mask Probe

**Not run** (same reason); the paired probe records `mask_ran: false`.

## 15. L1/L2/L3 + Query Breakdown

G0 stage: L1 inside 0.20, L2/L3 inside 0.0 — the failure is uniform across levels and families
(diffuse heatmaps everywhere), so no breakdown can be over-read.

## 16. Error / Data Adequacy Analysis

`evaluation/task6g_error_analysis.json` (G0 stage, 20 records): 18/20 outside; dominant
`diffuse_no_localization_heatmap` (18/20), then `heatmap_on_wrong_building` (0), tiny_target on
17/18, `whu_data_quality_dominates` true. Honest reading: the WHU flags describe the material but
cannot explain a diffuse heatmap; **WHU is not the binding constraint of this result** — the query's
instruction signal is, and simple L1 cases fail too (0.20).

## 17. Reusable Inference API

Not created (section 21 gates it on G1). The verified query-path plumbing (`predict_heatmap` in
`buildreasonseg_mvp/dense_eval.py`) remains available to the next task.

## 18. Runtime / VRAM

Grid oracle: 3 grids × (120 + 40) decodes, ~7 min. G0: 1500 steps + 6 query-path evaluations, 535 s
wall clock; ~14 GiB VRAM, batch 1, bf16 autocast, gradient checkpointing, Task 6C.7 visual-feature
cache plus the SAM2 CPU feature cache (frozen encoder outputs only; nothing downstream of the
trainable projection is cached).

## 19. Tests

`tests/test_task6g_dense_grounding.py` adds **21** tests covering the section 23 list: cell-centre
snapping (3 grids) and determinism, the oracle selection rule (smallest grid with paired ≥ 18/20 and
mIoU ≥ 0.4576 → 256), area-interpolation downsampling + mass assertion, head shape/determinism,
gradients on both projections, deterministic argmax cell centres, the frozen interior-point
convention, feature-level selection by spatial size (never list order), the recorded one-step smoke
(SAM2 bit-frozen, rows/head moved, ordinary rows unchanged, optimizer coverage), no forbidden
components, Task 6E/6F machinery retired in all 6G sources, config hygiene, the true-total-step
scheduler horizon, the paired→inside→Dice selection rule, G2 predicted-point-only, no GT in the
inference path, deterministic JSON writing.

Full suite: **343 passed** in 679 s (`python -m pytest tests/ -q`, including the artifact-consistency
gate) — 322 pre-existing tests plus the 21 Task 6G tests, with no regressions.

## 20. Git / Watt

**Commit: `{{COMMIT_HASH}}` — `feat: add dense spatial grounding map`** (staged only code, tests,
docs and `evaluation/**` artifacts: no checkpoints, no weights, no cached feature tensors, no
`.conda`, no dataset edits; `artifacts/**` stays gitignored and only
`evaluation/task6g_checkpoint_manifest.json` records the checkpoint hashes).

**Watt Toolkit handling:** the pre-existing Watt processes (`Steam++.exe`,
`Steam++.Accelerator.exe`) were left running untouched for the whole task and are used for the final
push only under the established ownership rules — never closed, never force-killed, no hosts-file,
certificate-store or TLS changes anywhere in this task.

## 21. Recommended next architecture task

Four readouts have now failed identically — 6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box
regression, 6G dense map — each verified-implementation-clean, each collapsing to an
image-dominated query. The measured bottleneck is the **query's instruction-dependent signal**, not
the readout format. Recommended next step, in order, each measured against the frozen gates
(F0-style overfit → 480-paired → point/box oracle ceilings):

1. **Strengthen the query itself (recommended).** Candidates: (a) a small set of instruction-aware
   query tokens / query refinement steps over the visual features (still no `[REF]`, no new
   dataset); (b) an auxiliary objective that explicitly maximizes the same-image/
   different-instruction separation of the query (a *representation* objective, not SRE/SCL); (c)
   counterfactual supervision — the paired `P` subset already provides two targets per image, and
   none of the four readouts exploited it explicitly; (d) revisit the spec's training budgets
   honestly under the corrected scheduler, since the "8 epochs were simply too few" reading of
   Task 6F remains unfrozen and G0's failure is a signal-strength failure at 1500 steps.
2. **Only after the query becomes target-specific**: re-test the dense map (it is the cheapest
   readout to re-verify and now has a frozen 0.4883/18-20 point-oracle ceiling), then G2.
3. **Not indicated by anything measured here:** `[REF]`, SRE, SCL, 4B, dataset migration, full
   training, GUI.
