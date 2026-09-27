# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6I._

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

The block above is machine-checked against
`evaluation/build_spatial_reason_artifact_index.json` by
`scripts/check_artifact_consistency.py`. Do not hand-edit numbers anywhere else.

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset | **BuildSpatialReason v0.1.1** (frozen, audited, PASS) |
| Legacy dataset version | **v0.1** (frozen, superseded — never use for training) |
| MVP environment | **`.conda/buildreasonseg-mvp`** (conda `--prefix`, Python 3.11.16, PyTorch 2.13.0+cu132) |
| MVP stack (measured) | **Qwen3-VL-2B-Instruct + SAM2.1 Hiera Base+ + `[SEG]`** (ADR-013) |
| Design stack (unmeasured) | Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large (ADR-012) |
| Network posture | model/data execution is offline from local assets (`HF_HUB_OFFLINE=1`); Watt is transport-only when needed, and ownership rules determine whether it is left running or closed; no hosts edit, no cert-store edit, no insecure TLS flag |
| Reproducibility | **strict deterministic mode, cross-process bit reproducible** (Task 6C) |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (32,284 records) |
| 5 | v0.1 validator + semantic quality audit | done → `FAIL_REQUIRES_REVISION` |
| 5B | v0.1.1 corrective regeneration + acceptance audit | done → `PASS` |
| 5C | Acceptance hardening + artifact consistency | done → `PASS` |
| 5.5 | External research, model-stack verification, MVP design freeze | done → ADR-012 |
| 6A | Native-Windows env bootstrap + 2B `[SEG]` MVP smoke/overfit | done → `PASS` |
| 6B | Network cleanup + 2B real mini-train + first generalization audit | done → `FAIL_REQUIRES_DEBUG` |
| Watt | Standalone Watt Toolkit lifecycle test (3 rounds) | done → `FULL_AUTO_OK` |
| 6C | **Paired counterfactual training × neutral SAM prompt 2×2 ablation** | **done → `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`** |
| 6C.5 | **Batch-1 training-pipeline throughput audit + value-preserving optimization** | **done → `OPTIMIZATION_PARTIAL`** |
| 6C.6 | **Launch-overhead candidates + formal-path integration of the 6C.5 winner** | **done → `OPTIMIZATION_PARTIAL`** |
| 6C.7 | **Frozen Qwen visual-feature cache + remaining batch-1 sync audit** | **done → `OPTIMIZATION_PARTIAL`** |
| 6D | **Spatial Grounding Bridge v0.1 (oracle diagnostic + geometry head)** | **done → `GROUNDING_REPRESENTATION_FAILED`** (`VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND`) |
| 6D.1 | **Corrective G0 rerun + `[SEG]` spatial-decodability audit** | **done → `PRACTICALLY_NOT_DECODABLE_GEOMETRY`** |
| 6E | **Explicit Spatial Token Grounding v0.1 (`[BOX]` + 256 `<loc_*>` tokens)** | **done → `EXPLICIT_SPATIAL_TOKENS_FAILED`** (E0 passes 20/20; E1 structural 0/120) |
| 6F | **Target-Aware `[BOX]` Query Grounding v0.1 (pre-reasoning query + box head)** | **done → `TARGET_AWARE_QUERY_FAILED`** (F0 passes 0.9046/10-10; F1 paired 0/20) |
| 6G | **Dense Query–Visual Spatial Grounding Map v0.1 (query × frozen SAM2 map → heatmap point)** | **done → `DENSE_GROUNDING_IMPLEMENTATION_FAILED`** (grid-256 oracle 0.4883/18-20; G0 inside 0.10) |
| 6H | **Counterfactual Pair-Aligned Dense Grounding v0.1 (pair step + own-vs-cross ranking)** | **done → `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`** (H0 ranking 10/10 but inside-own 3/20) — **causal wording narrowed in Task 6H.1 §3: the logit ranking was scale-degenerate; query learnability was unresolved** |
| 6H.1 | **Spatial-Softmax Point Supervision + Bounded Counterfactual Grounding** | **done → `BOUNDED_POINT_OBJECTIVE_FAILED`** (H0-R point CE 11.10→4.74, inside-own 7/20) |
| 6I | **Visual Query Refinement Block v0.1 (single query cross-attends frozen 64×64 SAM2 embedding once)** | **done → `VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`** (I0 inside-own 7→13/20, paired point 2→5/10; attention peaks at only ~2.5 % target mass) |

## Task 6I measured results

The frozen Task 6H.1 objective and paired training are reused byte-for-byte; the only addition is
one `VisualQueryRefinementBlock` (q0 → LayerNorm → Linear(2048,256) → one cross-attention over the
frozen 64×64 SAM2 embedding, embed 256 / 4 heads, residual + FFN(256→512→256) → q1 → scores the
frozen 256×256/32-ch feature). Full detail: `docs/task6i_visual_query_refinement.md`, ADR-021,
`evaluation/task6i_*.json`.

| | Value |
|---|---|
| Architecture proof | F64 `[1,256,64,64]`, F256 `[1,32,256,256]` (by size), exactly 1 cross-attention layer, block 1,129,985 params, scorer = dot(q1,K256)/√256 + bias (bit-identical to the combined forward) |
| Freeze proof | gradients reach query/coarse/attention/FFN/fine projections + LoRA + token rows with `frozen_or_other == 0.0`; SAM2 and the frozen Task 6G head bit-identical, zero grad; shared F64/F256 bit-identical |
| Causal proof | `[BOX]` bit-identical under future-text mutation in **fp32** (bf16 sdpa rounds the shared prefix per sequence length: 0.31 delta vs 0.0 fp32) |
| **I0** (10 pairs, 1500 pair steps) | inside-own **13/20**, top-1 9/20, paired point **5/10**, bounded ranking **10/10**, norm-err **0.0838**, own mass **0.4166** vs cross **0.0236**, mean \|logit\| 36.1 → **failed** (gates 18/20, 9/10, 9/10, <0.08) |
| I0 audit | point CE 11.0876 → **3.1833**, CF loss 0.8527 → 0.0967, target-cell prob 1.5e-5 → 0.268, top-1 rate 0 → 0.45, q1 same-image L2 2.2 → **1755.6** vs q0 27.4 → 162.6; **attention target mass 0.0079 → 0.0250** (peaked but not on target) |
| I1 / I2 | **not run** (sections 8/14 stop at the failed I0) |

1. **The refinement step works but under-shoots.** One visual look raises the single-query
   localization roughly 2× (inside 7→13, paired point 2→5, own mass 0.18→0.42, error 0.167→0.084) —
   a real, verified improvement over the frozen 6H.1 baseline — but the gates are far away.
2. **The attention cannot aim itself.** The 1×4096 cross-attention peaks (entropy 8.31→1.46 nats)
   with only ~2.5 % mass inside the target's 64×64 region, and that share barely moves during
   training. A single query slot has no target hypothesis to test against the map, so it locks onto
   a salient but task-irrelevant location and the scorer amplifies it (q1 L2 → 1756, \|logit\| → 36).
3. **Accepted verdict (section 13): `VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`** → clean failure,
   task stops before I1/I2. Recommended for review (section 14): **multiple learned query slots**
   (OMG-Seg/OMG-LLaVA-style object queries), a stronger/larger MLLM, or architecture-level
   reference/relation grounding — none implemented automatically. No SAM2 training, no second
   refinement layer, no multiple queries, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.
4. **Two methodology fixes adopted for all later inference paths**: inference runs the Qwen model
   in **eval mode** (LoRA dropout p=0.05 is training-only noise), and causal bit-identity probes
   run in **fp32**.

## Task 6H.1 measured results

Architecture frozen at Task 6G/6H; only the spatial supervision target changed: a 65,536-class
spatial cross-entropy on the deterministic target **point-cell** plus a **bounded** own-vs-cross
target **probability-mass** preference, with Task 6G's BCE+Dice and Task 6H's raw-logit ranking
detached to zero gradient. Full detail: `docs/task6h1_bounded_point_counterfactual.md`, ADR-020,
`evaluation/task6h1_*.json`.

| | Value |
|---|---|
| Objective | `0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf`, `L_point` = CE over 65 536 spatial classes, `mass(P,M)=sum(P*M)` (bounded, no area division), `L_cf = -log(p_own/(p_own+p_cross))`, eps 1e-8, **no margin** |
| Retirement proof | `bce_dice_requires_grad: false`, `logit_ranking_requires_grad: false`, gradient contribution 0.0 |
| Gradient coverage | point-CE reaches query projection 0.881, query norm 0.0136, visual 1×1 0.109, LoRA 0.109, token rows 0.322; bounded `L_cf` reaches all five groups too |
| **H0-R** (10 pairs, 1500 pair steps) | inside-own **7/20**, top-1 cell 5/20, paired point **2/10**, bounded ranking **8/10**, own mass **0.1835** > cross **0.0456**, mean \|logit\| 13.2 → **failed** (gates 18/20, 9/10, 9/10, <10.0) |
| H0-R audit | point CE **11.0965 → 4.7427** (chance 11.090), entropy **11.09 → 6.21**, target-cell probability ×2 500, point error **0.4658 → 0.1673**, bounded margin **−0.000 → +0.1385**, but max **non-target** probability 0.0532 vs target 0.0515 |
| H1-R / H2-R | **not run** (section 13 stops at the failed H0-R) |

1. **The corrected objective is healthy and not degenerate**: the point cross-entropy falls from
   exactly chance to 4.74, the softmax sharpens (entropy 11.09 → 6.21 nats), the target cell gains
   ~2 500× probability, the argmax point error falls 0.466 → 0.167, and the bounded preference margin
   becomes clearly positive with own mass 4× cross mass. Task 6H's magnification shortcut is gone.
2. **H0-R still fails**: the largest non-target cell holds 0.0532 probability against a 0.0515 mean
   target probability, so top-1 is 25 % and inside-target 35 % — far from the 90 % gate. The single
   `[BOX]` query does not carry enough target-specific spatial signal even under a directly aligned,
   bounded objective.
3. **Task 6H's verdict and numbers are preserved**; only its causal wording was narrowed (section 3):
   6H learned the specified logit ranking but the loss was scale-degenerate, so it did not establish
   that the query signal is unlearnable — 6H.1 supplies that bounded test and shows real but
   insufficient spatial signal.
4. **Accepted verdict (section 22): `BOUNDED_POINT_OBJECTIVE_FAILED`** → per section 23 this is a
   clean failure and the task stops for review: the next candidates are representation-level
   (instruction-aware multi-query / iterative refinement, stronger MLLM, reference/relation
   grounding), **not another scalar loss**, and none may be implemented automatically. No SAM2
   training, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.

## Task 6H measured results

Task 6G's architecture is frozen; Task 6H changed only the training semantics: one optimizer step per
canonical same-image counterfactual pair (240 pairs / 240 images from the Task 6C `P` subset) plus an
own-vs-cross region-ranking loss. Full detail: `docs/task6h_counterfactual_pair_grounding.md`,
ADR-019, `evaluation/task6h_*.json`.

| | Value |
|---|---|
| Canonical pairs | **240 / 240 images**, all assertions enforced, identity hash `c56f0507da41fd55…`, 0/240 pairs overlap, no pair discarded |
| Pair step | one shared frozen SAM2 feature (bit-identical through the step), two query forwards with retained graphs, **one** backward; horizon counts **pair** steps (240/epoch) |
| Loss | `0.5*(reasoning_A+reasoning_B) + 1.0*(heatmap_A+heatmap_B) + 2.0*L_cf`, `L_cf` margin 1.0 on `s_XY = sum(H_X·M_Y)/sum(M_Y)` (logits) |
| Architecture check | identical to Task 6G (level `[1,32,256,256]`, head 270 593 params); `[BOX]` still causally clean |
| **H0** (10 pairs, 1500 pair steps) | pair ranking **10/10**, strict margin 10/10, mean margin **+5.58** — but point-inside-own **3/20** and paired point **0/10** → **failed** |
| H0 audit | `L_cf` 1.3133 → 0.0212, margin (logits) −0.0001 → +5.5788, **mean abs logit 0.148 → 16.32**, probability margin only **+0.112**, peakiness 1.6e-05 → 0.403, point-inside 0.00 → **0.15** |
| H1 / H2 | **not run** (section 12 stops on H0 failure) |

1. **The pair machinery is correct and verified**: canonical pair construction with hash and
   zero-overlap audit, one shared frozen feature, single backward per pair, correct loss sign,
   `L_cf` gradients reaching every trainable group (query projection, query norm, visual 1×1, Qwen
   LoRA, token rows), SAM2 bit-frozen through the step, Task 6G head unchanged.
2. **The specified ranking objective is scale-degenerate.** H0 satisfies ranking 10/10 and margin
   +5.58 by growing the heatmap logit scale ~110× while the probability-space margin stays +0.112
   and the argmax lands inside the target 15 % of the time. Error classification: 10/10 failing
   pairs are `pair_ranking_correct_point_outside`.
3. **Five readouts now fail with verified implementations** (6D.1 `[SEG]` probes, 6E coordinate
   tokens, 6F box regression, 6G dense map, 6H ranked dense map). Task 6H's contribution is the
   diagnosis: a future pair/contrastive objective for this project must be defined on a **bounded**
   scale (probability/softmax) or carry an explicit peak-sharpness term, otherwise the ranking gate
   keeps passing while localization stays at chance.
4. **WHU is not the binding constraint** of this result (the failure is uniform at 10/10 pairs on
   memorized records). No SAM2 training, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.
5. **Machine verdict (section 20): `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`** — kept as the task's
   original verdict. **Corrected causal interpretation (Task 6H.1 section 3):** Task 6H successfully
   learned the specified own-vs-cross *logit* ranking, but the loss was **scale-degenerate** — it could
   be minimized by magnifying logits without moving the heatmap peak onto the target. Query
   learnability therefore remained **unresolved** until a bounded, localization-aligned objective was
   tested; Task 6H.1 provides that test.

## Task 6G measured results

Task 6F's global box regression is retired; Task 6G keeps the causally clean pre-reasoning `[BOX]`
query but matches its hidden against the frozen SAM2 dense spatial features to produce a target
heatmap whose argmax is the predicted point. Full detail: `docs/task6g_dense_spatial_grounding.md`,
ADR-018, `evaluation/task6g_*.json`.

| | Value |
|---|---|
| Grid oracle (snapped point → frozen SAM2) | 64: 0.4950/17-20; 128: 0.4894/17-20; **256: 0.4883/18-20** (selected: smallest with paired ≥ 18/20 and mIoU ≥ 0.4576) |
| Selected level | 256×256 = the **32-channel** SAM2 high-res feature (by spatial size, never list order) |
| Head | `LayerNorm→Linear(2048,128)` on `[BOX]`; `Conv1x1(32,128)` on frozen features; `dot/√128` + scalar bias (270 593 params) |
| Loss | `1.0·L_reasoning + 2.0·(1.0·BCEWithLogits + 1.0·SoftDice)` on the area-downsampled target mask |
| One-step smoke | `[BOX]`/`[SEG]` rows + all head params move with gradient; **SAM2 bit-frozen through a training step**; ordinary rows exactly unchanged; causal bit-identity re-verified |
| **G0** (20 records, 1500 steps) | inside-own **0.10**, heatmap Dice **0.125**, paired point selection **0/10** — gate failed at budget exhaustion |
| G0 implementation audit | **no defect**: train/eval query hiddens bit-identical; logits healthy (std 3.3); BCE/Dice gradient norms comparable (0.395/0.456); heatmap = near-flat image-dominated field (peakiness 0.086, errors 44–225 px) |
| G1 / G2 | **not run** (section 11 stops on G0 failure) |

1. **The point-oracle pathway is verified at every grid**: snapping the frozen interior point to
   256×256 cell centres costs nothing (0.4883 vs 0.4876 continuous, paired 18/20, 0.50 px mean
   displacement). The downstream frozen SAM2 point path is not the problem.
2. **The dense head is implemented faithfully and audited, not assumed**: train/eval bit-equivalence,
   healthy logits, comparable BCE/Dice gradients, SAM2 bit-frozen — yet **G0 cannot overfit 20
   records**. The heatmap converges to a near-flat image-dominated field (Dice term never moves from
   ~0.97), and both instructions of one image pin the identical argmax cell.
3. **Four readouts now fail identically**: 6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box head, 6G
   dense map — each implementation-verified, each collapsing to an image-dominated query. The
   measured bottleneck is the **query's instruction-dependent signal**, not the readout format.
4. **WHU is not the binding constraint** (diffuse heatmaps cannot be explained by target size/border
   flags; L1 simple cases fail too). No SAM2 training, no `[REF]`/SRE/SCL/4B/dataset change/GUI.
5. **Accepted verdict (section 18): `DENSE_GROUNDING_IMPLEMENTATION_FAILED`** (G0 cannot pass after
   the implementation audit; task stops before G1 per section 11).

## Task 6F measured results

Task 6E's coordinate vocabulary is retired; Task 6F inserts a fixed learned `[BOX]` query before
reasoning and regresses the box from its hidden state. Full detail:
`docs/task6f_target_aware_box_query.md`, ADR-017, `evaluation/task6f_*.json`.

| | Value |
|---|---|
| Vocabulary | `[SEG]` 151 669 + exactly one new `[BOX]` 151 670 (no `<loc_*>` tokens added) |
| Trainables | text LoRA (17 432 576) + `{[SEG], [BOX]}` rows (4 096) + `TargetAwareBoxHead` (1 055 236) |
| Loss | `1.0 · L_reasoning + 5.0 · L_box` (SmoothL1 vs Task 6D tight box), no mask loss |
| Causal placement | `[BOX]` at `prompt_length`, never predicted; hidden **bit-identical** under future-text mutation |
| F0 (20 records, ≤1500 steps) | train box IoU **0.9046**, paired **10/10**, non-identical same-image boxes (L1 0.2285) at step 1000 → `F0_PASS` |
| F1 (480 paired, ≤8 epochs) | train box IoU **0.050** (never converged), val box IoU **0.025**, center-inside **0.017**, paired **0/20** at epoch 8/8 |

1. **The query mechanism itself is verified, not assumed.** `[BOX]` sits at the generation prefix
   and is never a prediction target; replacing the entire reasoning tail leaves its hidden
   **bit-identical** (recorded on the real model). The one-step smoke shows both token rows and all
   six head parameters moving with non-zero gradient while ordinary rows and the base table stay
   exactly unchanged, and SAM2 stays frozen.
2. **F0 passes decisively** (train box IoU **0.9046**, paired **10/10**, same-image non-identical
   boxes with L1 0.2285 at step 1000), so the implementation is not the problem.
3. **F1 underfits rather than overfits**: the box head never fits even its own 480 training records
   (train SmoothL1 plateaus ≈0.003 ≈ 8 % RMS coordinate error; per-step train IoU 0.0 at most logged
   steps), the predictions are canonical and spread over the image but wrongly placed (val coord MAE
   0.154, box IoU 0.025, paired 0/20). Same pattern as Task 6E, now on the regression side: 20
   records memorize, 480 do not fit inside the spec's budget.
4. **Representation answer (section 13):** the query is no longer template-dominated, but it is
   **image-dominated** — same-image centered cosine 0.626 vs 0.018 across images — with moderate
   target-geometry variation (hidden-vs-GT-box distance correlation 0.476 vs ≈0 for the legacy
   `[SEG]`). Effective rank 7.07 vs legacy 8.33.
5. **Language compatibility is perfect** (exactly-one `[SEG]` 1.000, EOS 1.000, known template
   1.000), and **F2 was correctly not run** (section 15 gate). Error analysis: 120/120 failures,
   dominant class `wrong_target_valid_box` (98 %), `tiny_target` on 93 % — the localization-precision
   difficulty of WHU-scale targets is real, but the F1 failure is primarily budget-limited
   under-fitting, not a data defect.
6. **Accepted verdict (section 17): `TARGET_AWARE_QUERY_FAILED`**, with no escalation attempted
   (no `[REF]`, SRE, SCL, 4B, dataset change, full training, GUI) and SAM2 never trained. Task 6E's
   coordinate machinery stayed retired throughout.

## Task 6E measured results

The box became an explicit autoregressive target (`reasoning [BOX] <loc_x1> <loc_y1> <loc_x2>
<loc_y2> [SEG]`) over one shared 256-token location family, with `L = 1.0·L_assistant +
5.0·L_location` and no mask loss. Full detail: `docs/task6e_explicit_spatial_tokens.md`, ADR-016,
`evaluation/task6e_*.json`.

| | Value |
|---|---|
| Quantized oracle, `B = 32 / 64 / 128 / 256` | 0.5400 / 0.6473 / 0.7129 / **0.7343** mIoU, all paired **20/20** |
| Selected bin count | **`B = 256`** (smallest with paired 20/20 and mIoU ≥ `0.7506 − 0.03`) |
| Added vocabulary | `[BOX]` + `<loc_000>…<loc_255>` = **257** tokens; **258** trainable rows ([SEG] + [BOX] + all loc) |
| E0 (20 records, ≤1500 steps) | structural **20/20**, exact four-token **20/20**, box IoU **0.9166**, paired **10/10** at step 500 → `E0_PASS` |
| E1 (480 paired records, 3 epochs × 480 steps) | structural **0/120** at every epoch, paired **0/20** |
| E1 teacher-forced | reasoning 1.000, `[BOX]` 1.000, EOS 1.000, **loc 0.075, `[SEG]` 0.000** |
| E1 free generation | exactly one `[BOX]` in 120/120, then a **51–81-token location run**, `[SEG]` count 0 in 120/120 |
| E1 train location CE | 9.64 → **5.15** (256-way uniform floor `ln 256 = 5.545`) |
| E2 strict e2e segmentation | **not run** (section 14 gates it on the E1 gate) |
| Error classification | 120 structural, **0 localization**, 0 quantization-limited, `whu_data_quality_dominates: null` |

1. **The pathway is verified up to the box, and the box is where it stops.** Given a 256-bin
   quantized box, the frozen SAM2 prompt path still reaches **0.7343 mIoU / paired 20/20** (97.8 % of
   the continuous-oracle 0.7506). The tokenizer, the 258 trainable rows, the label/loss plumbing and
   the token-id parser are verified by a real one-step smoke and by E0's 20/20 exact-token overfit.
2. **E1's failure is a vocabulary-learning failure, not a formatting or leakage failure.** Teacher
   forcing reaches 1.000 on reasoning, `[BOX]` and EOS but only 0.075 on the location values and
   **0.000** on `[SEG]`; free generation therefore emits `[BOX]` and then a runaway location run with
   no `[SEG]`, so no sequence is structurally valid.
3. **The same code, loss and schedule overfit the first 20 of those same 480 records to 20/20** (E0),
   so this is an optimization/budget limit of the fixed ≤3-epoch schedule on a new 256-way vocabulary,
   not an implementation defect. The location CE does not even fit the *training* set (5.15 vs 5.545
   uniform floor).
4. **Neither the dataset nor SAM2 is the constraint.** 0/120 localization failures, 0 samples whose
   quantized GT box is below 0.5 IoU, and every failure is structural.
5. **Accepted verdict (section 16): `EXPLICIT_SPATIAL_TOKENS_FAILED`**, with no escalation attempted
   (no `[REF]`, SRE, SCL, 4B, dataset change, full training, GUI) and SAM2 never trained.


## Task 6C measured results

Valid, bit-reproducible 2×2 on the frozen ADR-013 stack. Same recipe, same seed, same validation material;
only the training subset (U/P) and the SAM bridge (C/L) differ. Full detail:
`docs/task6c_prompt_ablation.md`, ADR-015, `evaluation/task6c_comparison.json`.

| Arm | strict e2e mIoU | Dice | emission | paired | mean margin | projected effective rank | top-1 variance |
|---|---|---|---|---|---|---|---|
| `U_C` (Task 6B baseline) | 0.10516 | 0.17360 | 120/120 | **0/20** | −0.000001 | 1.516 | 0.9023 |
| `U_L` (language bridge) | 0.10872 | 0.17846 | 120/120 | **0/20** | −0.003530 | 1.491 | 0.9087 |
| `P_C` (paired data) | 0.10604 | 0.17851 | 120/120 | **0/20** | +0.000007 | **4.100** | **0.5892** |
| `P_L` (both) | 0.09284 | 0.15626 | 120/120 | **0/20** | −0.000003 | **3.371** | **0.6430** |

## Task 6C.5 measured results

Performance-only task; it changed no model, loss, optimizer, scheduler, data or sample order, and it does not
reinterpret Task 6C. Full detail: `docs/task6c5_training_optimization.md`,
`evaluation/task6c5_variants.json`, `evaluation/task6c5_final_benchmark.json`.

| | B0 (current) | Adopted (`skip_grad_norm_instrumentation`) |
|---|---|---|
| Throughput (interleaved, 2 rounds, 8 warmup + 64 steps, batch 1) | 2.152 samples/s | **2.373 samples/s → +10.24 %** |
| Repeat spread | 4.96 % | 0.47 % |
| GPU utilization | 37.3 % | 37.5 % (unchanged) |
| CPU utilization | 84.4 % | 92.5 % |
| Single clean sweep | 2.506 samples/s | 2.847 samples/s (+13.63 %) |
| Peak RSS / reserved VRAM | 3.30 GiB / 8.64 GiB | 3.29 GiB / 8.64 GiB |

1. **The CPU≈100 % / GPU≈40 % cause is per-operator launch overhead inside Qwen forward and backward, not
   data preparation.** Synchronized profiling puts `qwen_forward` 50.1 %, `backward` 38.5 %, `optimizer_step`
   7.7 % against **all host-side preparation ≈2.4 %** (processor 0.76 %, image I/O 0.62 %, SAM feature
   lookup 0.52 %, target-mask I/O 0.24 %, H2D 0.21 %). The processor costs 2.21 ms/sample against a ≈400 ms
   step.
2. **Every data-side candidate was measured and rejected**: source cache −5.80 %, Qwen preprocessing cache
   −2.92 % (it halves CPU utilization 90 → 56 % and still loses), pinned + non-blocking −12.98 %, prefetch
   with 2/4/8 threads −5.46/−5.69/−9.31 %. The preprocessing cache is the clean proof that the hypothesis was
   wrong: CPU utilization is not the binding constraint.
3. **The adopted change is one switch**: drop the per-step 528-tensor gradient-norm sweep that the Task 6C
   loop computes and discards. Default stays `collect_grad_norms=True`, so Task 6A/6B and their tests are
   unaffected. Bit-equivalent gate: prepared tensors (0 mismatches / 16 samples), 12-step losses, gradient
   fingerprints and post-step parameter fingerprints all identical, baseline control reproducible. A
   separate refactor control proves the audit's own restructuring of `train_step` (optional stage timers,
   split `torch.autocast` regions) is value-neutral against the pre-6C.5 step body.
4. **Rejected caches were still value-preserving.** `source_cache + preprocessed_cache + skip` is also
   bit-equivalent (second gate artifact) and 2.29 % faster than the adopted variant — inside the benchmark's
   own 4.96 % spread — so three extra switches were not taken for an unresolvable difference (section 20's
   ~5 % complexity rule, recorded as `adoption_rule`).
5. **Gradient checkpointing stays ON** (−8.58 % when off) and **strict determinism stays ON** (the
   determinism tax is only +1.78 %, well below the gate).
6. **Remaining bottleneck is batch-1 launch serialization**; data-side work is closed out. Raising arithmetic
   intensity per launch (batch > 1) is the only evidence-backed lever and is explicitly a different
   experiment, not a Task 6C.5 optimization.
7. **Measurement defect found and fixed**: the LoRA adapters' `dropout = 0.05` consumes the global RNG, so an
   equivalence gate must **re-seed per run** — with a single start-up seed the baseline did not reproduce
   itself. Any future equivalence claim needs the same per-run re-seeding rule.
8. **Network / Watt for 6C.5**: the accelerator was **pre-existing** (`watt_preexisting = true`) and was only
   used to push; nothing was closed, force-killed, or reconfigured, no hosts file was edited and no TLS
   verification was disabled. The MVP stack stayed offline (`HF_HUB_OFFLINE=1`) throughout.

## Task 6C.6 measured results

Performance-only task. The one runtime change it adopts is wiring Task 6C.5's winner into the formal
training loop; every other candidate was measured and rejected. Full detail:
`docs/task6c6_launch_optimization.md`, `evaluation/task6c6_*.json`.

| | Value |
|---|---|
| Formal-path integration | `training.collect_grad_norms: false` in `configs/mvp/task6c_2b_ablation.yaml`, passed explicitly by `scripts/task6c_train.py` |
| Integration equivalence | **`BIT_EQUIVALENT`** (prepared tensors, 12-step losses, gradients, post-step parameters) |
| Integrated baseline B0.6 | **2.505 samples/s** (final head-to-head mean; 2.545 / 2.465) |
| Pre-integration control | 2.184 samples/s (2.188 / 2.179; spread 0.41 %) |
| Integration gain (interleaved) | **+14.7 %** (+21.5 % in the 4-run baseline group) |
| New candidates adopted | **none** |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 3.295 GiB RSS, no OOM |

1. **The step is dispatch/launch dominated, now measured rather than inferred**: **57,341 CUDA kernels
   per step** at a **2.1 µs median**, **2,545 host↔device synchronizations per step**, the CUDA launch
   path consuming **31.9 % of CPU self time**, and the GPU only ~33 % utilized in unprofiled runs. Host
   data preparation is ≈2.4 % of the step (Task 6C.5) and is not a factor. The gradient-norm sweep that
   Task 6C.5 removed accounted for 3,589 kernels and 2,048 synchronizations per step.
2. **`torch.compile` cannot run on this install.** The inductor backend fails with `TritonMissing`
   (no Triton package, no MSVC). Everything that does run is slower: `cudagraphs` −24.7 %, combined
   `aot_eager` −21.2 %, decoder tail −11.3 %, Qwen `aot_eager` −10.4 %, Qwen `eager` −9.2 %. All compile
   candidates are also `NOT_EQUIVALENT` (max loss difference 0.042–0.234 against a 1e-3 tolerance), all
   inflate reserved VRAM by ~4 GiB, and `combined_aot_eager` exceeds the 14 GiB budget at 14.41 GiB.
   `backend="eager"` and `cudagraphs` additionally fail on a **sequence-length change** (291/321/295/318
   tokens) with a tensor-size mismatch — CUDA graphs are unsafe here without fixed-length padding, which
   is out of scope.
3. **Optimizer/clipping is already optimal.** `_default_to_fused_or_foreach` resolves to
   `foreach=True` for the 528 fp32 parameters, so `foreach=True` is a no-op; `fused=True` is inside the
   noise band and `NOT_EQUIVALENT`; `foreach=False` is slower. Two *code-identical* controls
   (`adamw_foreach`, `clip_foreach_true`) measured −2.85 % and −3.37 %, which fixes the **noise floor of
   a sequential group at 3.37 %** on this machine.
4. **SDPA is already mixed and correct.** Per step, 472 attention calls split into **248
   memory-efficient CUTLASS FMHA** and **224 math**, with 0 flash and 0 cuDNN. FlashAttention is not
   compiled into this PyTorch build; forcing flash or mem-efficient fails with `No available kernel`, and
   forcing math changes the numerics (9.4481 vs 9.4255) and is ~9.7 % slower. No accidental fallback to
   fix, no win available.
5. **Gradient checkpointing stays ON.** Interleaved ON/OFF gave OFF −2.03 % with the two rounds
   disagreeing in sign (spread ON 5.76 %, OFF 1.32 %) → Task 6C.5's −8.58 % single-sweep figure does not
   survive interleaving.
6. **This laptop's absolute throughput is not a stable quantity** (up to 19 % between identical runs;
   the first run in a process is systematically fastest). The reproducible quantity is the interleaved
   *ratio*, which is why every comparison brackets its candidates with reference runs and compares
   against the interpolated reference, and why a non-resolvable gain is never adopted.
7. **Methodology carried forward**: A/B/A/B interleaving for headline numbers, bracketed+interpolated
   references for screening, per-run re-seeding in every equivalence gate, and `PYTHONUTF8=1` when
   reading `torch.compile` errors on this Windows locale (otherwise the real `TritonMissing` cause is
   hidden behind a `UnicodeDecodeError`).

## Task 6D.1 measured results

Corrective audit of Task 6D. Full detail: `docs/task6d1_corrective_grounding_audit.md`,
`evaluation/task6d1_*.json`.

| | Value |
|---|---|
| Scheduler defect | horizon was one epoch (480) while two ran → **epoch 2 LR exactly 0**; end-of-epoch-1 factor 1.17e-05 |
| Scheduler fixed | horizon 960 → epoch-2 start factor 0.517, terminal only at the final step (5.4e-06 of peak) |
| G0-R (scheduler fix only) | emission **120/120** ✅, geometry paired **0/20** ❌, box IoU **0.0088** ❌ → G1 not run |
| Probe A (linear, raw) | overfit-20 0.903, train-480 0.027, val 0.004, paired 0/20 |
| Probe B (raw MLP, no LayerNorm) | overfit-20 0.645, train-480 0.036, val 0.008 |
| Probe C (Task 6D LayerNorm head) | overfit-20 0.778, train-480 0.051, val 0.007 |
| Label-shuffled control | real ≈ shuffled at 480 (0.027 vs 0.020; 0.036 vs 0.020); shuffled also fits 20 samples to loss 0 |
| LayerNorm | **not a confound** — with a fair LR the LayerNorm head matches/beats the raw MLP |
| Representation | norm ≈112, effective rank **8.33**, top-1 31.6 %; centered cosine same-image/diff-template **0.077** vs diff-image/same-template 0.531 |
| Verdict | **Case F — `PRACTICALLY_NOT_DECODABLE_GEOMETRY`** |

1. **The scheduler defect was real but not causal.** Task 6D built the cosine scheduler with one epoch as
   its horizon and ran two, so epoch 2 trained at LR 0 (which is why its epoch-1/epoch-2 metrics were
   bit-identical). The corrective rerun G0-R fixes only that and fails the same gate.
2. **LayerNorm is not a confound, and neither was the head.** The first probe run appeared to show
   LayerNorm destroying the signal; that was **my probe failing to converge** (plus unscaled norm-112
   inputs). After fixing both, all three readouts converge and the LayerNorm head performs as well as the
   raw MLP.
3. **The 20-sample overfit is not evidence**: a shuffled-label readout fits 20 permuted boxes to zero loss
   too. The informative number is the 480-sample fit, where the real fit is within a few IoU points of the
   shuffled fit.
4. **Accepted statement (weaker, per the spec's wording rule)**: after the implementation audit, the frozen
   `[SEG]` representation of the Task 6C `P_C` checkpoint contains **no practically decodable target
   geometry** under this training setup. This is not an information-theoretic absence claim, and it
   replaces Task 6D's "carries no target location" wording.
5. **The residual encodes the instruction wording, not the target**: with the shared component removed, the
   two instructions of one image are nearly orthogonal (centered cosine 0.077) while two different images
   with the same template are much more similar (0.531), and the within-image token variance (0.00999) is
   ~30× smaller than the between-target box L1 (0.306).
6. **Oracle conclusion unchanged**: correct geometry → 0.7506 mIoU, paired 20/20. The downstream machinery
   works; the failure is in what the read-out can express.

## Task 6D measured results

Architecture task: the first explicit spatial-grounding mechanism, and the first measurement that
localizes the Task 6C failure to a specific component. Full detail:
`docs/task6d_spatial_grounding_bridge.md`, `evaluation/task6d_*.json`.

| | Value |
|---|---|
| Oracle point (GT point prompt → SAM2) | mIoU **0.4876**, Dice 0.6055, paired **18/20** |
| Oracle box (GT box prompt → SAM2) | mIoU **0.7506**, Dice 0.8507, paired **20/20**, IoU(pred_A, pred_B) **0.0000** |
| Section 5 geometry choice | **BOX** (point missed the 0.50 mIoU gate; box passed it and led by +0.2630) |
| G0 (2 epochs × 480 paired records) | emission **120/120** ✅, geometry paired **0/20** ❌, box IoU **0.0082** ❌ |
| G1 | **not run** (section 9: stop on G0 failure, do not compensate) |
| Free-generation strict e2e mIoU | **0.0066** (Task 6C `P_C` reference 0.10604) |
| Paired mask probe | **0/20**, own ≈ cross ≈ 4e-10, IoU(pred_A, pred_B) **0.919** |

1. **The segmenter was never the problem.** Given the correct geometry, SAM2 recovers the target at
   **0.7506 mIoU** (box) against Task 6C's **0.10604**, with a paired probe of 20/20 and *zero* overlap
   between the two targets' predicted masks. The Task 6C failure was entirely in prompt generation.
2. **Superseded wording (Task 6E §2.1; accepted statement in Task 6D.1 §16): under the audited Task 6C
   training setup, the frozen `[SEG]` representation contains no *practically decodable* target geometry
   with the tested readouts. This is not an information-theoretic absence claim.** The Task 6D evidence: a
   1.05M-parameter head supervised on
   480 GT boxes did not fit even the training boxes; by epoch 2 it collapsed to the single constant box
   `(0.457, 0.455, 0.547, 0.539)` with per-coordinate spread `[0.000, 0.000, 0.001, 0.000]` — the optimal
   constant predictor under SmoothL1 for an uninformative read-out.
3. **The read-out is template-dominated, not merely instruction-independent.** Four-condition control on
   `[SEG]` hidden cosines: different image + **same template** 0.99988 (most similar) vs same image +
   different template 0.99898 vs different image + different template 0.99869 (least similar). Changing
   the whole image moves the hidden state *less* than changing the instruction wording.
4. **The language objective cannot fix it.** LM CE saturates at 0.0000 because the assistant target is one
   of only 21 distinct reasoning templates (Task 6C's finding), so it is memorized long before it could
   pressure the read-out to encode *which* building.
5. **This reproduces Task 6C with a different read-out.** Task 6C measured the projected prompt at
   effective rank 1.5 with same-image cosine > 0.9999; Task 6D measures a *supervised geometry head* on
   the raw `[SEG]` hidden and finds the same collapse. Two independent read-outs agreeing makes the
   diagnosis strong.
6. **Dataset is not the binding constraint (yet).** The 116/120 failures carry `tiny_target` (93 %) and
   `border_truncation` (30 %) flags, but the artifact's degeneracy guard shows all of them fail from one
   constant mask regardless of content, so a dataset-selection task is not indicated by this evidence.

## Task 6C.7 measured results

Performance-only task; two changes adopted, and the model/loss/data/optimizer semantics are untouched.
Full detail: `docs/task6c7_visual_cache_optimization.md`, `evaluation/task6c7_*.json`.

| | Value |
|---|---|
| Section 3 Phase-B duplicate H2D | **removed** (exact; also fixed a `del moved` `UnboundLocalError` the refactor introduced) |
| Frozen Qwen visual-feature cache | **adopted** — `BIT_EQUIVALENT`, **+9.94 %** warm paired throughput |
| Cache footprint | 4.0 MiB/image (`pooler_output` 1 MiB + 3 × `deepstack_features` 1 MiB, bf16), bound 512 images = **2.0 GiB** |
| Kernel / sync change | **−8.44 %** CUDA kernels, **−43.47 %** host↔device sync ops per step |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 4.751 GiB RSS + 128 MiB cache, no OOM |
| Config | `training.visual_feature_cache: true`, `visual_feature_cache_max_images: 512` |

1. **The visual tower is provably cacheable.** 315 visual parameters, all bf16 and all frozen; zero LoRA
   modules in the tower (`lora.text_only: true`); no dropout, so it consumes no RNG; recomputation is
   bit-identical (max abs difference 0.0); nothing in the tower moves during a real optimizer step
   (parameters *and* buffers checked); and 8 same-image instruction pairs produce identical
   `pixel_values` **and** identical visual features — the tower is instruction-independent.
2. **The cached boundary is the narrowest available**: `Qwen3VLModel.get_image_features(...)` output,
   i.e. `pooler_output` + `deepstack_features`, before they are mixed with trainable text hidden states.
   The wrapper is an instance-level replacement on the project's own Qwen module with the original bound
   method preserved, so the miss path is byte-identical. Nothing downstream of a trainable parameter is
   cached, and `last_hidden_state` is deliberately not cached (the language model never reads it).
3. **Equivalence is bit-exact** over 16 samples' features and a 12-step gate with per-run re-seeding, for
   both key strategies (source-image identity and pixel-content hash). First-step loss identical.
4. **The gain is real but only visible under a drift-robust design.** Whole-variant runs drift by −24.6 %
   and the four-run interleaved V1/V2 comparison produced rounds disagreeing in sign (+15.7 % V1, then
   +8.9 % V2). A paired ablation — alternating 8-step cache-off/cache-on blocks inside one runtime — gives
   **+10.41 % (all pairs) / +9.94 % (warm), 4/4 pairs in favour**. Warming the cache while it is disabled
   stores nothing, and that mistake alone had reported +2.31 % and would have rejected the change.
5. **The remaining ~1,051 scalar read-backs are PyTorch's, not ours.** Scoped windows plus a
   `sys.setprofile` `c_call` tracer attribute 99.5 % of them to `optimizer.step()` →
   `torch/optim/adam.py:770-776` converting each parameter's **CPU-hosted** `step` counter twice per step
   (`_get_value` → `.item()`); the upstream comment states the CPU hosting is deliberate. The optimizer
   window issues 1,024 reads with **one** sync op, so they are cheap; backward contributes 5 reads;
   disabling gradient checkpointing changes nothing. `fused=True` would remove them, but Task 6C.6
   measured it at −0.71 % and it is `NOT_EQUIVALENT`, and section 4 forbids patching third-party
   internals. Project-owned reads are ~7/step (<1 %), so V3 (deferred scalar logging) was not implemented.
6. **Batch-1 optimization is now exhausted on evidence**, which is what makes the batching recommendation
   earned: the redundant H2D is gone, the sync traffic is localized and dismissed, and the last large
   removable win is adopted. What remains is execution shape (~52,500 kernels/step, ~2 µs median, GPU
   ~40 %).

## What Task 6C changed in the project's understanding

1. **Neither of Task 6B's two confounds is the cause.** All four arms are 0/20 on the paired unseen probe
   with mean margins within ±0.0036 of zero. Paired counterfactual training, the removal of the fixed
   centre point, and their combination all fail to create instruction-conditional mask selection.
2. **But the factors are not inert.** Paired training materially changes the prompt representation
   (effective rank 1.50 → 3.74, top-1 variance 0.905 → 0.616) without changing the masks. The projection
   uses more dimensions; two instructions on one image still produce the same mask.
3. **The fixed positive centre point is not the culprit.** Its norm is 11.39 against a projected language
   norm of 266.8 — a **23.4×** ratio in the language vector's favour — and removing it (the `L` bridge)
   changes little. "The centre point overrides the language" is not supported by the measurement.
4. **The collapse is directional and low-rank, not constant.** Same-image projected cosine > 0.9999 with
   top-1 variance 0.59–0.91 and a non-zero effective rank. Task 6B's looser wording is corrected.
5. **The remaining problem is after the prompt**, i.e. in how the `[SEG]` hidden state is formed and how
   SAM's decoder turns a prompt direction into a region. Prompt normalisation, a multi-token prompt,
   auxiliary point supervision or `[REF]` are justified now, and were not before.
6. **Determinism is real now.** `training.deterministic` is consumed, strict deterministic algorithms are
   active with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and two independent processes produce identical losses,
   gradients and post-step parameters. The `U_C` arm was run before and after the cache fix and is
   bit-identical, which also proves the fix value-preserving.

## Correctness fixes carried forward (do not regress)

1. Determinism (above).
2. Per-arm checkpoint directories come from `cfg["paths"]["checkpoints"]` — no module-level constant.
3. The operation-chain lookup is built from the **train split only**; validation text never extends it.
4. The SAM2 CPU feature cache holds all 480 training images inside the budget by sharing the two
   image-independent tensors once (7.508 GiB instead of 11.25 GiB), and it never mutates stored entries.

## Measured limitations (carry into Task 6I)

1. **Instruction conditioning of the mask is still absent**, and Tasks 6D + 6D.1 localize it precisely:
   the frozen `[SEG]` hidden state is norm-112, effective-rank 8.33, and its instruction-dependent residual
   (centered cosine 0.077 between the two instructions of one image) is **not practically decodable**
   into target geometry — three converging readouts cannot fit even the 480 training samples better than a
   label-shuffled control. Task 6D's original wording ("carries no target location") is superseded by this
   audit (§16 of Task 6D.1).
2. **Five geometry readouts fail with verified implementations, and Task 6H identifies the objective as
   the culprit, not the plumbing.** The legacy `[SEG]` probes (6D.1), the coordinate vocabulary (6E:
   structural 0/120), the global box regression (6F: val box IoU 0.025), the dense map (6G: G0 inside
   0.10) and the counterfactually ranked dense map (6H: ranking 10/10 but inside-own 3/20) all collapse
   to an image-dominated query. Task 6H's audit shows the specified own-vs-cross margin on **unbounded
   logits** is minimized by magnifying the field (mean |logit| 0.148 → 16.32, probability margin only
   +0.112). A future ranking/contrastive term must be bounded or carry a peak-sharpness term, and the
   query's instruction dependence must be strengthened deliberately.
3. **Mask quality is not usable**: strict end-to-end mIoU 0.093–0.109, L3 nontrivial 0.076–0.088.
4. **Language metrics are template metrics**: `reasoning_zh` has 21 distinct values in the whole training
   mini-set, so exact match and operation-chain accuracy cannot support a reasoning claim.
5. **More prompt diversity did not become mask diversity**: effective rank 3.4–4.1 still yields
   IoU(pred_A, pred_B) 0.999.
6. **The best mIoU arm is not the most diverse arm** (`U_L` 0.1087 vs `P_L` 0.0928), so mIoU alone remains
   a misleading selection signal.
7. **Training throughput is launch-bound at batch 1** and the removable overhead is now exhausted:
   the adopted runtime is Task 6C.6's integration plus Task 6C.7's frozen visual-feature cache, worth
   +9.94 % on top (paired ablation). Task 6H adds one datapoint: a pair step (two query forwards, one
   shared frozen feature, one backward) costs ~1.6× a single-sample step, so 1500 pair steps ≈ 29 min.

## Current blockers

**The single `[BOX]` query does not carry enough target-specific spatial signal.** Six readouts have
failed with verified implementations (6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box regression,
6G dense map, 6H logit-ranked dense map, 6H.1 point-aligned bounded dense map). The last two rule out
the readout format (6G: image-dominated feature map) and the objective's scale/sign (6H.1: point CE
falls to 4.74 and the bounded margin is clearly positive, yet the max non-target cell still rivals the
target cell). What remains is the query representation itself and the strength of the 2B MLLM's
instruction-conditioned spatial signal. Per Task 6H.1 section 23, further scalar-loss redesign is
**not** indicated.

## Recommended next task

**ChatGPT review of the pushed Task 6H.1 results**, then **exactly one** representation-level task,
with the evidence Tasks 6D.1–6H.1 produced:

1. **Instruction-aware query refinement (recommended).** Multiple query tokens (or a small number of
   refinement steps) over the frozen visual features, so the instruction can select among competing
   buildings in one image; the canonical pair material, the frozen point oracle ceiling and the
   point-CE + bounded-mass objective are ready-made measurement infrastructure.
2. **Stronger MLLM representation** (e.g. a larger Qwen) if the 2B hidden is the binding constraint —
   a scale question that the project has so far refused to answer by default and should now be posed
   explicitly against the measured bottleneck.
3. **Architecture-level reference/relation grounding** (`[REF]`-style reference tokens, SRE) only
   after a target-selection gate passes; the current evidence still does not justify SRE/SCL.
4. **Not indicated**: another scalar loss, dataset migration, full training, GUI.

Full detail: `handoff/FROM_DSH.md`, `docs/task6h1_bounded_point_counterfactual.md`,
`docs/task6h_counterfactual_pair_grounding.md`, `docs/task6g_dense_spatial_grounding.md`,
`evaluation/task6h1_verdict.json`, `evaluation/task6h1_objective_setup.json`,
`evaluation/task6h1_h0r_overfit.json`, `evaluation/task6h1_h0r_audit.json`,
`evaluation/task6h1_error_analysis.json`, `evaluation/task6h_*.json`, `evaluation/task6g_*.json`,
`docs/architecture_decisions.md` (ADR-016 … ADR-020).
