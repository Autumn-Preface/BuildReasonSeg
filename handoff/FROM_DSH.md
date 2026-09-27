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

# FROM_DSH — Task 6I Report: Visual Query Refinement Block v0.1

_This file holds the Task 6I report. The Task 6H.1 report is preserved in git history and in
`docs/task6h1_bounded_point_counterfactual.md`; Task 6H in `docs/task6h_counterfactual_pair_grounding.md`;
Task 6G in `docs/task6g_dense_spatial_grounding.md`._

Full design notes: `docs/task6i_visual_query_refinement.md`, ADR-021.

## 1. Verdict

**`VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`.**

The first explicitly allowed architecture change since 6G is implemented faithfully and verified:
the causally clean `[BOX]` query q0 cross-attends the frozen SAM2 64×64 embedding exactly once
(embed 256, 4 heads, residual + FFN 256→512→256), the refined q1 scores the frozen 256×256 /
32-channel feature, and the frozen Task 6H.1 point-cell CE + bounded probability-mass pair
objective drives training. The refinement step **works but under-shoots**: over 1500 pair steps on
the 10 memorized H0 pairs, inside-own **7 → 13/20** (gate 18), paired point **2 → 5/10** (gate 9),
bounded pair ranking **8 → 10/10** (gate 9 ✓), normalized point error **0.167 → 0.084** (gate
< 0.08), own mass **0.184 → 0.417** vs cross mass 0.046 → 0.024. The audit shows the mechanism:
the 1×4096 cross-attention peaks (entropy 8.31 → 1.46 nats) but only ~2.5 % of its mass lands on
the target, so the single query slot — even with one allowed look at the visual features — cannot
**aim itself** at the target. Per sections 8/14 the task **stops at I0**: I1 (240-pair mini-train)
and I2 (frozen-SAM2 segmentation) were not run; no SAM2 training, no second refinement layer, no
multiple queries, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.

## 2. Frozen Task 6H.1 Evidence

Not rerun and **not rewritten**: H0-R inside-own 7/20, paired point 2/10, bounded ranking 8/10;
point CE 11.0965 → 4.7427, point error 0.4658 → 0.1673, own mass 0.1835 vs cross 0.0456, margin
−0 → +0.1385. The Task 6H.1 scalar objective (weights 0.5 / 1.0 / 1.0, eps 1e-8, no margin) is
reused **byte-for-byte** in Task 6I — the 6I pair step imports the same `spatial_objective`
functions and constants. Also frozen: grid-256 snapped point oracle 0.4883 mIoU / 18-20, box
oracle 0.7506 / 20-20, Task 6C `P_C` 0.10604 / 0-20.

## 3. Architecture

```
image + instruction
  -> Qwen pre-reasoning [BOX] query q0                 [B, 2048]  (causally clean)
  -> q0 -> LayerNorm -> Linear(2048, 256) -> q         [B, 1, 256]
  -> F64 [B,256,64,64] -> Conv1x1(256,256) -> flatten -> LayerNorm   [B, 4096, 256]
  -> q_ref = q + CrossAttention(q, F64)   (embed 256, 4 heads, exactly ONE layer)
  -> q1 = q_ref + FFN(LayerNorm(q_ref))    (256 -> 512 -> 256, GELU)
  -> F256 [B,32,256,256] -> Conv1x1(32,256) -> K256
  -> heatmap_logits[y,x] = dot(q1, K256[:,y,x]) / sqrt(256) + scalar bias
  -> argmax point -> frozen SAM2 positive-point prompt (I2, not reached)
```

Verified: F64 is exactly the main image embedding `[1,256,64,64]`; F256 is the 256×256 high-res
level selected **by size** `[1,32,256,256]`; exactly one `nn.MultiheadAttention`; q1 feeds the
scorer (bit-identical to the combined forward); 1,129,985 block parameters. The old Task 6G
`DenseSpatialGroundingHead` stays installed as frozen evidence only and is absent from the
candidate forward. The 1×4096 attention weights are exposed for diagnostics only (never
supervised).

## 4. Trainables / Frozen Parameters

Train: text LoRA (17,432,576 params, 392 tensors, lr 1e-4), the refinement block (1,129,985
params, 21 tensors, lr 3e-4, weight decay 0.01), the `[BOX]`/`[SEG]` rows (4,096 params, lr 3e-4,
no decay). Frozen: Qwen base, Qwen visual tower, all SAM2, the Task 6G head, the Task 6F box
head, the Task 6D grounding head, the Projection MLP, and any `<loc_*>` rows (none exist).

Verified on the real model (`evaluation/task6i_architecture_setup.json`, all pass): point-CE and
bounded-CF gradients reach query_proj / coarse_proj / cross_attn / ffn / fine_proj / Qwen LoRA /
token rows with `frozen_or_other == 0.0`; one pair step moves every block parameter and both token
rows, leaves the frozen Task 6G head **bit-identical** with zero gradient, SAM2 **bit-identical**
with zero gradient, the shared F64/F256 features **bit-identical**, and ordinary embedding rows
exactly unchanged; retired BCE/Dice + raw-logit ranking carry zero gradient.

## 5. Attention Diagnostics

Clean initialization: uniform attention, entropy 8.31 nats (= ln 4096), top-10 mass 0.0035,
target mass ≈ background mass ≈ 0.008. After I0: entropy **1.46**, top-10 mass concentrated, but
mean attention mass inside the target's 64×64 region only **0.025** (pair-level own mass 0.0125)
with the cross-target mass collapsed to ~1e-5. The attention peaks, but not on the target — it is
diagnosed, not supervised, throughout.

## 6. I0 Overfit (10 pairs)

Clean initialization, 1500 pair steps, eval every 250 steps via `argmax(refined heatmap)` in eval
mode: inside-own 4/7/11/13/13/13 at steps 250/500/750/1000/1250/1500; top-1 0/3/8/9/9/9; paired
point 1/3/4/5/5/5; bounded ranking 2/8/9/10/10/10; normalized error 0.200/0.225/0.096/0.076/
0.084/0.084; own mass 0.098/0.183/0.293/0.374/0.410/0.417 vs cross 0.067/0.052/0.041/0.028/
0.025/0.024; mean |logit| 5.0/10.7/17.9/30.5/35.7/36.1; attention entropy ≈ 1.4; q0 same-image L2
156 → 159, q1 same-image L2 394 → 1742. The trajectory plateaus from step 1000. Gate: **FAILED**
(13/20, 5/10, 10/10, 0.0838). Verdict `VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`, 1377 s.

## 7. I0 Audit (implementation clean)

`evaluation/task6i_i0_audit.json`: q0 causal bit-identity holds (fp32 probe, delta 0.0); F64/F256
shared features bit-identical; gradients reach attention Q/K/V/out, FFN, F64 projection, F256
scorer, Qwen LoRA and `[BOX]` with **no gradient into SAM2**; point CE 11.0876 → 3.1833 and
bounded CF loss 0.8527 → 0.0967 both decrease; target-cell probability 1.53e-5 → 0.2683, top-1
rate 0.00 → 0.45, inside rate 0.05 → 0.65, spatial entropy 11.09 → 4.21, max non-target
probability 1.66e-5 → 0.0580; q1 same-image L2 2.2 → 1755.6 vs q0 27.4 → 162.6; attention target
mass 0.0079 → 0.0250; attention/logits finite. Two procedure findings were fixed: inference now
runs in **eval mode** (LoRA dropout p=0.05 is training-only noise — measured 0.56–1.39 same-batch
delta in train mode vs 0.0 in eval), and the causal probe runs in **fp32** (bf16 sdpa rounds the
shared prefix differently for different sequence lengths — 0.31 delta vs 0.0 in fp32; kernel
rounding, not information flow).

## 8. I1 Training

**Not run.** Per sections 10/14 the 240-pair mini-train is gated on I0 passing; I0 failed cleanly
after the audit.

## 9. Point Localization

I0 final (eval mode, argmax path): inside-own **13/20**, target-cell top-1 9/20, top-5 10/20,
normalized point error **0.0838**, 512px error from `evaluation/task6i_i0_overfit.json`. The
6H.1 H0-R baselines were 7/20 inside and 0.1673 error — the refinement block roughly doubles the
localization signal but does not reach the 18/20 gate.

## 10. Pair Preference

Bounded pair ranking **10/10** at I0 final (6H.1 H0-R: 8/10); own mass 0.4166 vs cross mass
0.0236 (6H.1: 0.1835 vs 0.0456); paired point selection **5/10** (gate 9). Preference is
region-level correct while point placement misses — see the error analysis.

## 11. q0 vs q1 Representation

The full §11 representation diagnosis belongs to the best I1 checkpoint; since I1 did not run, the
I0-level evidence stands in: same-image q0 L2 159.4 (q0 barely moves from 155.7 across training)
vs q1 L2 **1742.4** (growing monotonically 394 → 1742). One refinement step massively amplifies
the instruction contrast in q1 — but the amplification follows the wrong spatial hypothesis (see
§5). The main §11 question is answered provisionally: **no** — a single visual refinement step
does not turn the weak instruction signal into a target-specific q1, because the attention cannot
aim itself at the target.

## 12. I2 Segmentation

**Not run** (gated on I1, which is gated on I0).

## 13. Paired Mask Probe

Not run (I2 did not run). `evaluation/task6i_paired_probe.json` records the I0-stage point-side
probe: preference 10/10, paired point 5/10.

## 14. L1/L2/L3 + Query Breakdown

Not run (I1/I2 did not run). The I0 error analysis (`evaluation/task6i_error_analysis.json`)
provides the level/quality breakdown instead: dominant pair failure
`pair_preference_correct_point_outside`; `attention_misses_target` 100 % of records (attention
mass inside target < 0.5); WHU data-quality flags cover > 50 % of failing records
(`whu_data_quality_dominates: true`).

## 15. Error / Data Adequacy

The failure is representation-level, not objective- or data-level alone: the pair preference is
perfect (10/10) while points miss, so the model prefers the right target region and cannot resolve
where inside it the point goes; the cross-attention (its one chance to look) is peaked at
~2.5 % target mass. Data quality contributes to the residual but is not the dominant mechanism:
the same 10 pairs are the ones 6H.1 audited in depth.

## 16. Runtime / VRAM

One pair step ≈ 0.9 s on the RTX 5080 Laptop (bf16 autocast, gradient checkpointing, frozen
visual-feature cache, strict determinism on); I0 = 1500 pair steps + 6 evaluations ≈ 1377 s.
Block adds 1.13 M trainable parameters; optimizer groups LoRA 1e-4 / decoder 3e-4 / token 3e-4,
cosine over the pair-step budget, warmup 20.

## 17. Tests

`python -m pytest tests/ -q` → **418 passed** (32 warnings), including 28 new Task 6I tests
(`tests/test_task6i_visual_query_refinement.py`): single cross-attention layer, attention
geometry, exact F64/F256 shape validation, q0→q1 and FFN residual exactness, scorer-uses-q1,
attention diagnostics, frozen 6H.1 constants, detached legacy terms, one-pair-one-step, pair-step
scheduler horizon, shared SAM features, SAM2/visual-tower freeze, no-GT-in-inference, old-head/
box-head/loc-path absence, no test split, predicted-point-only I2, no `[REF]`/SRE/SCL/4B, strict
determinism, checkpoint round-trip for the block, parameter-group bucket, and the
`evaluation/task6i_architecture_setup.json` pass gate.

## 18. Git / Watt

Task commit `ee0e589` (`feat: add visual query refinement`) followed by the `docs:` handoff
commit; weights/checkpoints/caches and hidden/attention dumps stay gitignored; only `evaluation/`
manifests and JSONs are tracked. Watt was already running from earlier tasks and is used as
transport-only for the push; ownership rules apply — no hosts/cert/TLS edits, no insecure flags.

## 19. Recommended Next Architecture Decision

**Multiple learned query slots** (an object-query set in the style of OMG-Seg/OMG-LLaVA): the
I0 evidence shows the single `[BOX]` slot cannot direct its own cross-attention at the target —
one free look at the whole map peaks on salient-but-wrong regions (2.5 % target mass). A small set
of learned queries with the 6H.1 point-aligned bounded objective would let each slot specialize
and let the head select the most peaked refined heatmap. Alternatives: a stronger/larger MLLM, or
architecture-level reference/relation grounding. Per section 14 none of these may be implemented
without review; the task stops here.
