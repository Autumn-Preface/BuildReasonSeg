# TO_DSH — Task 6I: Visual Query Refinement v0.1

> Status: ACTIVE
>
> Repository: BuildReasonSeg
>
> Goal: replace the failed direct single-query readout with one explicit instruction-conditioned visual query refinement step, while keeping the proven Task 6H.1 point-cell objective and paired supervision unchanged.
>
> Accepted evidence:
> - 256-grid snapped point -> frozen SAM2: mIoU 0.4883, paired 18/20.
> - Task 6H.1 fixed the previous scale-degenerate loss: point CE 11.0965 -> 4.7427, point error 0.4658 -> 0.1673, own mass 0.1835 vs cross 0.0456.
> - Yet H0-R still failed: inside 7/20, paired point 2/10.
>
> Hypothesis: the Qwen [BOX] query needs one explicit opportunity to inspect frozen visual features before high-resolution localization.
>
> Candidate:
>
> image + instruction
> -> Qwen pre-reasoning [BOX] query q0
> -> cross-attend q0 to frozen SAM2 64x64 feature
> -> refined query q1
> -> q1 scores frozen SAM2 256x256 feature
> -> heatmap -> argmax point -> frozen SAM2
>
> No 4B, [REF], SRE, SCL, dataset migration or GUI.

## 0. UI language

Narrative DSH output in Chinese. Code/paths/metric keys may be English.

## 1. Freeze Task 6H.1 objective

Keep exactly:

L_total =
  0.5*(L_reasoning_A + L_reasoning_B)
  + 1.0*(L_point_A + L_point_B)
  + 1.0*L_cf_bounded

Where:
- L_point = 65,536-way point-cell CE on the 256x256 grid.
- L_cf_bounded = spatial-softmax own-vs-cross target probability-mass preference.

Do not re-enable:
- Task 6G BCE+Dice,
- Task 6H raw-logit ranking,
- box SmoothL1,
- coordinate-token CE.

No loss-weight sweep.

Keep one optimizer step = one canonical pair, same 240 train pairs, same deterministic pair order, same corrected pair-step scheduler.

## 2. Keep the initial query

Reuse Task 6F/6G/6H causally clean pre-reasoning [BOX] token:

image + instruction -> fixed [BOX] -> q0

q0 must not attend to future reasoning or GT.

Do not add new language special tokens.

## 3. Coarse visual feature

Use frozen SAM2 64x64 main image embedding.

Verify exact shape; expected approximately [B,256,64,64].

Flatten to 4096 visual tokens.

No gradient into SAM2.

## 4. VisualQueryRefinementBlock

Implement exactly one refinement block.

Recommended structure:

q0 [B,2048]
-> LayerNorm
-> Linear(2048,256)
-> q [B,1,256]

F64
-> Conv1x1(256,256)
-> flatten [B,4096,256]
-> LayerNorm

CrossAttention:
- query=q
- key/value=F64 tokens
- embed_dim=256
- num_heads=4

Residual:
q_ref = q + cross_attn_output

Then:
q_ref
-> LayerNorm
-> FFN(256 -> 512 -> 256, GELU)
-> residual
-> q1

Constraints:
- exactly one cross-attention refinement layer;
- no second refinement step;
- no self-attention stack;
- no extra learned image queries;
- no coordinate channels;
- no deformable attention.

## 5. High-resolution scorer

Use frozen SAM2 256x256 / 32-channel feature.

F256 -> Conv1x1(32,256) -> K256

heatmap_logits[y,x] = dot(q1, K256[:,y,x]) / sqrt(256)

Optional scalar bias allowed.

The candidate forward must use q1, not q0 and not the old Task 6G head.

## 6. Trainables

Train:
- text-only Qwen LoRA;
- [BOX] row;
- [SEG] row;
- VisualQueryRefinementBlock;
- F64 projection;
- F256 scorer projection.

Freeze:
- Qwen base;
- Qwen visual tower;
- all SAM2;
- Task 6G old DenseSpatialGroundingHead;
- Task 6F box head;
- Task 6D grounding head;
- Task 6E loc-token path.

No visual LoRA.

Verify optimizer coverage and frozen-backbone bit identity.

## 7. Attention diagnostics

Expose the 1x4096 cross-attention weights for diagnostics.

Report:
- attention entropy;
- top-1 visual cell;
- top-10 attention mass;
- attention mass inside target building after downsampling target mask to 64x64;
- attention mass inside paired other target.

Diagnostics only; do not supervise attention directly.

## 8. I0: 10-pair overfit

Use the same 10 canonical H0 pairs.

Clean initialization only; do not load failed Task 6H.1 H0-R weights.

Maximum 1500 pair optimizer steps.

Use frozen Task 6H.1 point CE + bounded pair mass objective exactly.

At 250/500/750/1000/1500 report:
- point-inside-own /20;
- paired point /10;
- bounded pair ranking /10;
- target-cell top-1/top-5 /20;
- target-cell probability;
- own/cross target probability mass;
- normalized and 512px point error;
- spatial entropy;
- mean abs high-res logit;
- cross-attention entropy;
- target vs cross attention mass;
- same-image q0 distance;
- same-image q1 distance;
- same-image predicted-point distance.

I0 gate:
- inside >= 18/20;
- paired point >= 9/10;
- bounded pair ranking >= 9/10;
- mean normalized point error < 0.08.

No hard gate on exact target-cell top-1.

If I0 fails after implementation audit:
VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT
and STOP.

## 9. I0 audit if needed

Audit only:
- q0 causal cleanliness;
- same-image different-instruction cross-attention outputs differ when appropriate;
- F64/F256 shared features bit-identical for pair A/B;
- gradients reach Qwen LoRA, [BOX], attention Q/K/V/out, FFN, F64 projection and F256 scorer;
- no gradient reaches SAM2;
- point CE and bounded pair loss decrease;
- attention/logits finite;
- q1 separation vs q0 separation.

Do not add a second refinement layer during audit.

## 10. I1: 240-pair mini-train

Only if I0 passes.

Train:
- 240 canonical pairs/epoch;
- max 8 epochs;
- deterministic order;
- corrected pair-step scheduler;
- validate each epoch.

No mid-run architecture or loss changes.

Model selection:
1. paired point /20;
2. bounded pair ranking /20;
3. 120-record point-inside rate.

I1 gate:
- paired point >= 14/20;
- pair ranking >= 14/20;
- 120-record point-inside >= 0.60.

If best epoch fails:
VISUAL_QUERY_REFINEMENT_NO_GENERALIZATION
and STOP before segmentation.

## 11. Representation diagnosis

At best I1 checkpoint compare q0 vs q1 on same-image/different-instruction pairs.

q0:
- raw cosine;
- centered cosine;
- L2.

q1:
- raw cosine;
- centered cosine;
- L2;
- effective rank if practical.

Also report:
- q0 distance vs GT-point-distance correlation;
- q1 distance vs GT-point-distance correlation;
- probability-mass margin vs localization correctness;
- attention target mass vs localization correctness.

Main question:
Does one visual refinement step turn the weak instruction signal in q0 into a target-specific q1?

## 12. I2: frozen-SAM2 segmentation

Only if I1 passes.

Inference:

image + instruction
-> q0
-> visual refinement q1
-> 256x256 heatmap
-> argmax predicted point
-> official frozen SAM2 positive-point prompt
-> frozen SAM2 decoder
-> mask

No GT and no SAM2 training.

Report on fixed 120 val + 20 paired:
- strict e2e mIoU;
- Dice;
- mask paired /20;
- own-target IoU;
- cross-target IoU;
- own-minus-cross mask margin;
- IoU(pred_A,pred_B);
- L1/L2/L3;
- query-family breakdown.

Compare:
- Task 6C P_C 0.10604 / 0/20;
- selected-grid point oracle 0.4883 / 18/20;
- box oracle 0.7506 / 20/20.

## 13. Verdicts

Use exactly one:

VISUAL_QUERY_REFINEMENT_FIX_FOUND
- I0 pass;
- I1 pass;
- I2 mask paired >=14/20;
- strict e2e mIoU >=0.20;
- mask own-minus-cross >0.05;
- no GT leakage.

VISUAL_QUERY_REFINEMENT_PARTIAL
- real target-specific localization improvement, but not full gate.

VISUAL_QUERY_REFINEMENT_NO_GENERALIZATION
- I0 passes, I1 fails.

VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT
- implementation clean, I0 fails.

INVALID_EXPERIMENT
- causal/freeze/shape/loss/pair/split correctness failure.

## 14. Decision rule

If I0 fails cleanly:
STOP. Recommend one for review:
- stronger/larger MLLM representation;
- multiple learned query slots;
- architecture-level reference/relation grounding.

Do not implement automatically.

If I0 passes but I1 fails:
STOP. Review training-data scale, dataset suitability, 2B vs larger MLLM, stronger query architecture.

## 15. Required artifacts

Create as applicable:

evaluation/task6i_architecture_setup.json
evaluation/task6i_i0_overfit.json
evaluation/task6i_i0_audit.json
evaluation/task6i_i1_training.json
evaluation/task6i_spatial_eval.json
evaluation/task6i_representation.json
evaluation/task6i_attention_diagnostics.json
evaluation/task6i_segmentation_eval.json
evaluation/task6i_paired_probe.json
evaluation/task6i_error_analysis.json
evaluation/task6i_checkpoint_manifest.json
docs/task6i_visual_query_refinement.md

Weights/checkpoints/caches local and gitignored.

## 16. Required tests

Cover at least:
1. q0 is causally clean pre-reasoning [BOX];
2. exact F64 shape verified;
3. exact F256 shape verified;
4. exactly one cross-attention refinement layer;
5. attention dim 256 / heads 4;
6. q0->q1 residual correct;
7. FFN residual correct;
8. high-res scorer uses q1;
9. Task 6H.1 point CE unchanged;
10. bounded pair mass unchanged;
11. old BCE/Dice and raw-logit ranking remain zero-gradient;
12. one pair = one optimizer step;
13. scheduler counts pair steps;
14. shared SAM features identical for A/B;
15. SAM2 bit-frozen;
16. Qwen visual tower frozen;
17. attention diagnostics do not inject GT into inference;
18. Task 6G old head absent from candidate forward;
19. Task 6F box head absent;
20. Task 6E loc path absent;
21. no test split;
22. I2 uses predicted point only;
23. no 4B/[REF]/SRE/SCL;
24. strict determinism.

Run:
python -m pytest tests/ -q

## 17. Git / Watt

Do not stage weights/checkpoints/caches/hidden or attention dumps/.conda/dataset JSONL edits.

Recommended commit:
feat: add visual query refinement

Use established Watt ownership rules for final push only.

## 18. FROM_DSH

Include:
1. Verdict
2. Frozen Task 6H.1 Evidence
3. Architecture
4. Trainables/Frozen Parameters
5. Attention Diagnostics
6. I0 Overfit
7. I0 Audit if needed
8. I1 Training
9. Point Localization
10. Pair Preference
11. q0 vs q1 Representation
12. I2 Segmentation
13. Paired Mask Probe
14. L1/L2/L3 + Query Breakdown
15. Error/Data Adequacy
16. Runtime/VRAM
17. Tests
18. Git/Watt
19. Recommended next architecture decision

## 19. Final DSH UI — Chinese only

Report:
- verdict;
- I0 inside /20;
- I0 paired point /10;
- I0 bounded pair ranking /10;
- q0 vs q1 same-image separation;
- target vs cross attention mass;
- if I1 ran: best epoch, paired point /20, pair ranking /20, 120-val inside;
- if I2 ran: strict mIoU, mask paired /20, mask own-cross margin;
- whether visual refinement solved the single-query bottleneck;
- dominant remaining failure;
- tests;
- commit/push;
- Watt handling.

## 20. STOP

After Task 6I STOP.

Do not automatically add:
- second refinement layer;
- multiple query tokens;
- [REF];
- SRE/SCL;
- 4B;
- dataset migration;
- full training;
- GUI.

Wait for ChatGPT review.
