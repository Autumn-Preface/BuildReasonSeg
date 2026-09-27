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

# FROM_DSH — Task 6H.1 Report: Spatial-Softmax Point Supervision + Bounded Counterfactual Grounding

_This file holds the Task 6H.1 report. The Task 6H report is preserved in git history and in
`docs/task6h_counterfactual_pair_grounding.md`; Task 6G in `docs/task6g_dense_spatial_grounding.md`._

Full design notes: `docs/task6h1_bounded_point_counterfactual.md`, ADR-020.

## 1. Verdict

**`BOUNDED_POINT_OBJECTIVE_FAILED`.**

The corrected objective is implemented faithfully, verified, and **healthy** — it is not the Task 6H
degeneracy. Over 1500 pair steps on the 10 memorized H0 pairs the point cross-entropy falls
**11.0965 → 4.7427** (chance = `ln 65536` = 11.090), the spatial-softmax entropy falls **11.09 → 6.21**
nats, the target cell gains ~2 500× probability, the argmax point error falls **0.4658 → 0.1673**, and
the bounded own-vs-cross preference margin becomes clearly positive (**−0.0000 → +0.1385**, own mass
0.184 vs cross 0.045). Yet **H0-R fails its gates**: inside-own **7/20** (gate 18), paired point
**2/10** (gate 9), bounded pair ranking **8/10** (gate 9), final mean |logit| 13.2 (limit 10). The
audit explains why: the largest **non-target** cell holds 0.0532 probability against a 0.0515 mean
target probability, so top-1 is 25 % and the mode is not reliably the target. Per section 13 the task
**stops at H0-R**: H1-R and H2-R were not run, no SAM2 training, no `[REF]`/SRE/SCL/4B/dataset
migration/GUI.

## 2. Frozen Task 6H Evidence

Not rerun and **not rewritten**: H0 pair ranking 10/10, strict margin 10/10, mean logit margin +5.58,
inside-own 3/20, paired point 0/10, heatmap Dice 0.134; audit: `L_cf` 1.3133 → 0.0212, mean |logit|
0.148 → 16.32, probability margin 0.1124, point-inside 0.15; machine verdict
`COUNTERFACTUAL_QUERY_SIGNAL_FAILED`. Also frozen: 256-grid snapped point oracle 0.4883 mIoU / 18-20.

## 3. Corrected Causal Interpretation (section 3)

Task 6H successfully learned the specified own-vs-cross **logit** ranking, but the loss was
**scale-degenerate**: it could be minimized by magnifying logits without moving the heatmap peak onto
the target. Query learnability therefore remained **unresolved** until a bounded, localization-aligned
objective was tested — which Task 6H.1 supplies. The 6H verdict remains its original machine verdict;
the new interpretation is explicit here and in `handoff/PROJECT_STATE.md`.

## 4. Point-Cell Target (section 4)

Frozen Task 6D interior point → Task 6G 256-cell snap → `target_index = y_cell*256 + x_cell`. Verified
in `evaluation/task6h1_objective_setup.json`: indices 29418 / 18943 for the first canonical pair
(different and in range), 12 candidate cells → 65 536 classes, GT used for supervision only, inference
remains `argmax(heatmap_logits)` with no GT repair.

## 5. Spatial-Softmax Objective (sections 5, 8-9)

`L_point = CrossEntropy(H.flatten(), target_index)` with no class weighting, no temperature and no
label smoothing; diagnostics per record: point CE, target-cell probability, top-1/top-5/top-25 cell
accuracy, spatial entropy, max non-target probability, mean |logit|, logit std.
**Zero-gradient retirement is verified**: Task 6G's BCE+Dice and Task 6H's raw-logit ranking are built
detached (`bce_dice_requires_grad: false`, `logit_ranking_requires_grad: false`, recorded gradient
contribution 0.0) and are logged as historical diagnostics only.

## 6. Bounded Counterfactual Mass Loss (sections 6-7)

`P = softmax(H.flatten())` (sums to 1), `mass(P, M) = sum(P * M)` — **not** divided by target area and
bounded in [0,1]; `p_AA/p_AB/p_BB/p_BA` recorded with own/cross margins and own/(own+cross) ratios;
`L_cf = 0.5*(-log((p_own+eps)/(p_own+p_cross+2eps)))` with eps 1e-8 and **no margin hyperparameter**;
pair preference passes iff `p_AA > p_AB` and `p_BB > p_BA`.

## 7. Pair-Step Semantics (sections 10-11)

One optimizer step = one canonical pair: one shared frozen SAM2 256×256 feature, query forward for A,
query forward for B (both graphs retained), one backward, one optimizer step; scheduler horizon counts
**pair** steps (240/epoch). `L_total = 0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_point_A+L_point_B) +
1.0*L_cf`, weights fixed. Gradient coverage of both terms into the query projection, query norm, visual
1×1, Qwen LoRA and token rows is recorded; SAM2 and the shared feature stay bit-identical; `[BOX]` stays
causally clean; the Task 6G/6H architecture is unchanged (level `[1,32,256,256]`, head 270 593 params).

## 8. H0-R Overfit (sections 12-13)

Clean Task 6G/6H initialization (never the scale-exploded Task 6H checkpoint), same 10 canonical pairs,
1500 pair steps.

| Pair step | inside-own | top-1 cell | paired point | bounded ranking | own mass | cross mass | \|logit\| |
|---|---|---|---|---|---|---|---|
| 500 | 4/20 | 1 | 1/10 | 5/10 | — | — | 5.65 |
| 750 | 2/20 | 1 | 0/10 | 6/10 | — | — | 9.73 |
| 1000 | 5/20 | 3 | 1/10 | 8/10 | 0.0030 | 0.0062 | 11.20 |
| 1250 | 7/20 | 4 | 2/10 | 8/10 | — | — | 12.95 |
| **1500** | **7/20** | **5** | **2/10** | **8/10** | **0.1835** | **0.0456** | **13.21** |

Gate (inside ≥ 18/20, paired point ≥ 9/10, ranking ≥ 9/10, own mass > cross mass, |logit| < 10.0):
**failed**. `evaluation/task6h1_h0r_overfit.json`, checkpoint `artifacts/checkpoints/task6h1/H0R/`.

## 9. H0-R Audit (section 14)

`evaluation/task6h1_h0r_audit.json` (clean init → H0-R checkpoint): point CE 11.0965 → **4.7427**;
target-cell probability 0.00002 → **0.0515**; max non-target probability 0.00002 → **0.0532**; top-1
0.00 → 0.25; inside 0.00 → 0.35; normalized point error 0.4658 → **0.1673**; entropy 11.090 → **6.205**;
own mass 0.0072 → **0.1838**; cross mass 0.0072 → **0.0453**; probability margin −0.0000 → **+0.1385**;
bounded `L_cf` 0.854 → 0.243; A/B hidden distance 27.4 → 135.7; A/B projected-query distance 1.49 →
636.8; point-CE gradients reach every trainable group (query projection 0.881); pair identities and
target indices verified; shared feature bit-identical. The objective is healthy; the mode is still not
reliably the target — that is a **query-signal** limit, not an objective-limit.

## 10. H1-R Training

**Not run** — section 13 stops the task at the failed H0-R. `evaluation/task6h1_h1r_training.json` and
`evaluation/task6h1_spatial_eval.json` are deliberately absent.

## 11. Point Localization Metrics

H0-R stage (20 records at 1500 pair steps): inside-own **7/20**, top-1 cell **5/20**, top-5 cell
8/20, mean target-cell probability 0.0515, normalized point error 0.1673 (≈86 px at 512), mean spatial
entropy 6.21 nats, mean |logit| 13.2. Full per-record rows in the H0-R artifact and error analysis.

## 12. Pair Preference Metrics

H0-R stage (10 pairs): bounded ranking **8/10**, own mass 0.1835 vs cross 0.0456, margin **+0.1380**,
bounded `L_cf` 0.243 at the end (0.854 at initialization). Task 6H for comparison: ranking 10/10 but a
logit-space margin of +5.58 with probability margin only +0.112.

## 13. Representation/Entropy Diagnostics

Recorded in the audit: entropy 11.09 → 6.21 nats (real sharpening), target-cell probability ×2 500,
max non-target probability 0.0532 ≈ mean target probability 0.0515 (the decisive number), A/B
hidden/projected distances 27.4 → 135.7 and 1.49 → 636.8 (separation grows, but the peak still does
not land on the target). A full H1-R-level representation artifact was not produced because H1-R was
not run.

## 14. H2-R Segmentation

**Not run** (section 20 gates it on H1-R). `evaluation/task6h1_segmentation_eval.json` is deliberately
absent. Frozen ceilings: Task 6C `P_C` 0.10604 / 0-20; point oracle 0.4876 / 18-20; box oracle
0.7506 / 20-20.

## 15. Paired Mask Probe

**Not run**; the paired probe records `mask_ran: false` with the H0-R 10-pair stage.

## 16. Error/Data Adequacy Analysis

`evaluation/task6h1_error_analysis.json` (H0-R stage): pair preference 8/10, both points inside 2/10,
dominant failure class `tiny_target`; record-level inside rate 0.350, top-1 0.250; the WHU quality
flags are reported alongside. The residual is localization precision on WHU-scale targets — the
objective, the pair construction and the frozen backbone are all verified, so **WHU difficulty is a
contributing factor but not the binding constraint of this result** (the same 10 memorized pairs are
only 35 % inside).

## 17. Runtime/VRAM

Objective setup + one-pair smoke: one model load, ~6 min. H0-R: 1500 pair steps + 6 evaluations in
**~31 min** wall clock; ~14 GiB VRAM, batch 1, bf16 autocast, gradient checkpointing, Task 6C.7
visual-feature cache plus the SAM2 CPU feature cache (frozen encoder outputs only, shared by both
instructions of a pair). H0-R audit: two model states in one process, ~8 min.

## 18. Tests

`tests/test_task6h1_point_objective.py` adds **24** tests covering the section 26 list: the frozen
256-grid target-cell convention and the deterministic interior point, the flat `y*grid+x` index and its
inverse, spatial softmax summing to 1 and shift invariance, bounded region mass that is explicitly not
area-normalised, own/cross mass correctness, pair-preference invariance to additive logit shifts,
`L_cf` monotonicity in own and cross mass, the absence of a margin parameter, entropy/top-k helpers,
the recorded zero-gradient retirement of BCE+Dice and the logit ranking, two-forwards-one-backward in
the pair step with an objective that contains only the point CE and bounded `L_cf`, the pair-step
scheduler horizon, the recorded architecture/freeze facts, the H0-R verdict with its non-degeneracy
evidence, Task 6E/6F retirement, H2-R predicted-point-only, config hygiene, no test split, forbidden
components, deterministic JSON, and the section 3 wording correction with unchanged 6H numbers.

Full suite: **389 passed** in 680 s (`python -m pytest tests/ -q`, including the artifact-consistency
gate) — 365 pre-existing tests plus the 24 Task 6H.1 tests, with no regressions.

## 19. Git/Watt

**Commit: `2879136` — `fix: align counterfactual grounding with point localization`** (staged
only code, tests, docs and `evaluation/**` artifacts: no weights/checkpoints, no feature caches, no
hidden dumps, no `.conda`, no dataset changes; `artifacts/**` stays gitignored and only
`evaluation/task6h1_checkpoint_manifest.json` records the checkpoint hash).

**Watt Toolkit handling:** the pre-existing Watt processes (`Steam++.exe`,
`Steam++.Accelerator.exe`) were left running untouched for the whole task and are used for the final
push only under the established ownership rules — never closed, never force-killed, no hosts-file,
certificate-store or TLS changes anywhere in this task.

## 20. Recommended next architecture decision

Per section 23 the clean H0-R failure means **stop redesigning scalar losses**. Six readouts have now
failed with verified implementations; the last two rule out the readout format (6G) and the objective's
scale/sign (6H.1). Recommended, in order, for review:

1. **Instruction-aware query representation (recommended)**: multiple query tokens or a small number of
   refinement steps over the frozen visual features, so one instruction can select among competing
   buildings; the canonical pairs, the frozen 0.4883/18-20 point oracle and the point-CE + bounded-mass
   objective are ready to measure it.
2. **Stronger MLLM representation** (larger Qwen) — a scale question to pose explicitly against the
   measured bottleneck rather than by default.
3. **Architecture-level reference/relation grounding** only after a target-selection gate passes.
4. **Not indicated**: another scalar loss, dataset migration, full training, GUI.
