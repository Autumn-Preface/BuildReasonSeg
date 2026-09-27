# Task 6I — Visual Query Refinement Block v0.1

> Task: `handoff/TO_DSH.md` · Verdict: **`VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`**
> Evidence: `evaluation/task6i_*.json` · ADR-021 · Tests: `tests/test_task6i_visual_query_refinement.py`

## 1. Hypothesis

Task 6H.1 proved the scalar objective is not the problem: the point-aligned bounded objective is
healthy (point CE 11.10 → 4.74, point error 0.466 → 0.167, bounded margin +0.139) yet the argmax
point still misses the target (H0-R inside-own 7/20, paired point 2/10). The bottleneck is the
**query representation**: the single pre-reasoning `[BOX]` token has exactly one hidden vector to
carry image + instruction conditioning into the dense map, with no opportunity to *inspect* visual
content before localization.

Task 6I gives the query exactly **one** explicit opportunity to inspect frozen visual features:

```
image + instruction
  -> Qwen pre-reasoning [BOX] query q0                      [B, 2048]  (causally clean)
  -> VisualQueryRefinementBlock (ONE cross-attention layer):
       q0  -> LayerNorm -> Linear(2048, 256) -> q           [B, 1, 256]
       F64 [B,256,64,64] -> Conv1x1(256,256) -> flatten -> LayerNorm   [B, 4096, 256]
       q_ref = q + CrossAttention(q, F64)     (embed 256, 4 heads, exactly ONE layer)
       q1   = q_ref + FFN(LayerNorm(q_ref))   (256 -> 512 -> 256, GELU)
  -> q1 scores the frozen SAM2 256x256 / 32-channel feature:
       F256 -> Conv1x1(32,256) -> K256
       heatmap_logits[y,x] = dot(q1, K256[:,y,x]) / sqrt(256) (+ optional scalar bias)
  -> argmax point -> frozen SAM2 positive-point prompt
```

The scalar objective is the frozen Task 6H.1 one, byte-for-byte:

```
L_total = 0.5*(L_reasoning_A + L_reasoning_B)
        + 1.0*(L_point_A + L_point_B)      # 65,536-class point-cell CE
        + 1.0*L_cf                          # bounded own-vs-cross probability-mass preference
```

Task 6G's BCE+Dice and Task 6H's raw-logit ranking remain detached diagnostics (zero gradient).
The old Task 6G `DenseSpatialGroundingHead` is installed only as frozen evidence and is **never**
part of the candidate forward.

## 2. Verified architecture facts (sections 3-5)

Recorded in `evaluation/task6i_architecture_setup.json` (all checks pass):

| Fact | Value |
|---|---|
| Coarse feature | `features.image_embeddings` — exactly `[1, 256, 64, 64]` |
| Fine feature | 256×256 high-res level selected **by size** — exactly `[1, 32, 256, 256]` |
| Cross-attention layers | **1** (asserted and counted) |
| Attention geometry | embed 256, 4 heads, 1×4096 weights (averaged over heads), rows sum to 1 |
| Residuals | `q_ref = q + attn_out`, `q1 = q_ref + FFN(LayerNorm(q_ref))` (unit-tested exact) |
| Scorer | `dot(q1, K256)/sqrt(256)` + scalar bias; bit-identical to the combined forward |
| Block parameters | 1,129,985 (query proj, coarse Conv1×1, attention Q/K/V/out, FFN, fine Conv1×1, norms, bias) |
| `[BOX]` causal cleanliness | bit-identical under future-text mutation (**fp32 probe**; see §6) |
| Frozen | Qwen base + visual tower, all SAM2, old Task 6G head, Task 6F box head, Task 6D grounding head, Projection MLP |
| Trainable | text LoRA (17.4 M), `[BOX]`/`[SEG]` rows, the block (1.13 M) |
| One pair = one optimizer step | shared F64/F256 for A/B; single backward |

## 3. Gradient audit (section 9, run at setup)

Fresh-forward gradients, measured per loss on the real first canonical pair:

| Group | point-CE grad | bounded-CF grad |
|---|---|---|
| `refine_block.query_proj` | 0.856 | 0.0616 |
| `refine_block.coarse_proj` (F64) | 0.118 | 0.0085 |
| `refine_block.cross_attn` (Q/K/V/out) | 0.167 | 0.0118 |
| `refine_block.ffn` | 0.167 | 0.0120 |
| `refine_block.fine_proj` (F256) | 0.129 | 0.0084 |
| Qwen LoRA | 0.213 | 0.0133 |
| token rows | 0.861 | 0.0374 |
| **frozen_or_other** | **0.0** | **0.0** |

One pair step: `[BOX]`/`[SEG]` rows move (Δ 3.0e-4), every block parameter moves, the frozen Task
6G head stays **bit-identical** with zero gradient, SAM2 stays **bit-identical** with zero gradient,
the shared F64/F256 features stay **bit-identical**, ordinary embedding rows are exactly unchanged,
and the retired BCE/Dice + logit ranking carry `requires_grad=False` (zero gradient).

## 4. Attention diagnostics (section 7)

The 1×4096 attention weights are exposed for diagnostics only (never supervised). At clean
initialization the attention is uniform (entropy 8.31 nats = ln 4096; top-10 mass 0.0035; target
mass ≈ background mass ≈ 0.008). After I0 the attention is **peaked but not target-specific**:
entropy 1.46 nats, top-10 mass concentrated, yet only ~0.013–0.025 of the mass falls inside the
64×64 target region while the cross-target mass collapses to ~1e-5.

## 5. I0 — 10-pair overfit (section 8)

Same 10 canonical H0 pairs, clean initialization, ≤ 1500 pair steps, eval every 250 steps through
the real `argmax` path (eval runs the Qwen model in **eval mode**, see §6):

| step | inside /20 | top-1 /20 | top-5 /20 | paired point /10 | ranking /10 | norm-err | own mass | cross mass | mean \|logit\| | attn entropy | q0 l2 | q1 l2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250  | 4  | 0  | 3  | 1  | 2  | 0.2004 | 0.098 | 0.067 | 5.0  | 1.61 | 155.7 | 394.0 |
| 500  | 7  | 3  | 7  | 3  | 8  | 0.2250 | 0.183 | 0.052 | 10.7 | 1.47 | 157.6 | 883.6 |
| 750  | 11 | 8  | 10 | 4  | 9  | 0.0959 | 0.293 | 0.041 | 17.9 | 1.45 | 159.2 | 1277.3 |
| 1000 | 13 | 9  | 10 | 5  | 10 | **0.0760** | 0.374 | 0.028 | 30.5 | 1.42 | 159.8 | 1579.5 |
| 1250 | 13 | 9  | 10 | 5  | 10 | 0.0838 | 0.410 | 0.025 | 35.7 | 1.43 | 159.5 | 1715.9 |
| 1500 | 13 | 9  | 10 | 5  | 10 | 0.0838 | 0.417 | 0.024 | 36.1 | 1.43 | 159.4 | 1742.4 |

I0 gate (inside ≥ 18/20, paired point ≥ 9/10, ranking ≥ 9/10, norm-err < 0.08): **FAILED** at step
1500 (`budget_exhausted`). The trajectory plateaus from step 1000.

**Progress vs the frozen baselines.** Inside-own 7 → **13**/20 (6H.1 H0-R), paired point 2 → **5**/10,
bounded ranking 8 → **10**/10, own mass 0.184 → **0.417** vs cross mass 0.046 → 0.024, normalized
point error 0.167 → **0.084**. The refinement block clearly amplifies the query's spatial signal —
but not enough to pass the gate, and the mean |logit| grows to 36 (the point-CE keeps sharpening
the map; the sharpening stops helping once the wrong peak wins).

## 6. I0 audit (section 9) and the two procedure findings

`evaluation/task6i_i0_audit.json` (clean initialization → loaded I0 checkpoint):

| Quantity | before | after |
|---|---|---|
| mean point CE | 11.0876 | **3.1833** |
| mean bounded CF loss | 0.8527 | **0.0967** |
| mean target-cell probability | 1.53e-5 | **0.2683** |
| target-cell top-1 rate | 0.00 | **0.45** |
| point-inside rate | 0.05 | **0.65** |
| mean spatial entropy | 11.09 | **4.21** |
| mean max non-target probability | 1.66e-5 | 0.0580 |
| mean q0 same-image L2 | 27.4 | 162.6 |
| mean q1 same-image L2 | 2.2 | 1755.6 |
| mean attention entropy | 8.31 | 1.46 |
| mean attention target mass | 0.0079 | 0.0250 |
| mean attention cross mass | 0.0079 | 1.0e-5 |
| p_AA / p_AB / p_BB / p_BA | .009/.005/.005/.009 | .317/.035/.515/.010 |
| bounded margin | −3.5e-6 | **+0.3935** |

All audit requirements pass: q0 causal cleanliness (bit-identical), point CE and bounded CF
decrease, gradients reach every block component with zero gradient into SAM2, attention/logits
finite, q1 separation (1755.6) ≫ q0 separation (162.6).

Two procedure defects were found and fixed during the audit (recorded here for transparency):

1. **Inference must run in eval mode.** LoRA dropout (p=0.05) is training-only noise; with trained
   weights the dropout perturbation is no longer bf16-roundable to zero (measured same-batch delta
   0.56–1.39 in train mode vs **0.0 in eval mode**). `forward_refined` therefore runs the Qwen
   model in eval mode and restores the training flag afterwards.
2. **The causal bit-identity probe must run in fp32.** In bf16, sdpa rounds the shared prefix
   differently for different sequence lengths (measured 0.31 max-abs delta vs **0.0 in fp32** for
   the same future-text mutation). That is kernel rounding, not information flow; the probe runs
   in fp32 so the causal claim is exact.

## 7. Why I0 fails: the refinement attends, but not to the target

The audit shows the mechanism precisely. The cross-attention becomes sharply peaked (entropy 8.31 →
1.46) yet only ~2.5 % of its mass lands inside the target's 64×64 region, and that share barely
moves during training (0.0079 → 0.0250). In other words the single pre-reasoning query, even after
being allowed to *look* at the frozen visual features, cannot decide **where** to look: it has no
target hypothesis to test against the map, so it locks onto some salient but task-irrelevant
location and the 256×256 scorer then amplifies whatever it got (q1 L2 same-image distance grows to
1756 while q0 stays at 163; the logit scale grows to 36).

Error classification (`evaluation/task6i_error_analysis.json`) agrees: pair preference is perfect
(10/10) but the dominant failure is **pair_preference_correct_point_outside** — the model prefers
the right target *region* and still misses the point inside it; every record's attention mass
inside the target is below 50 % (`attention_misses_target` 1.000), and WHU data-quality flags cover
more than half of the failing records.

## 8. Verdict

**`VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`** (section 13): the implementation is clean (setup and
audit pass every correctness check), yet I0 fails its gates (inside-own 13/20, paired point 5/10;
ranking 10/10 passes; norm-err 0.0838). Per sections 8 and 14 the task **stops before I1 and I2**:
no 240-pair mini-train, no frozen-SAM2 segmentation, no SAM2 training, no second refinement layer,
no multiple queries, no `[REF]`/SRE/SCL/4B, no dataset migration, no GUI.

**Recommendation for review (section 14):** the single query's attention cannot be *self-directed*
toward the target — one free cross-attention over the whole map is not enough for the 2B model to
find its own target hypothesis. Review candidates, in order: (a) **multiple learned query slots**
(an object-query set like OMG-Seg/OMG-LLaVA's, so each query can specialize and the head picks the
slot whose refined heatmap is most peaked); (b) **a stronger/larger MLLM** representation; or (c)
**architecture-level reference/relation grounding**. None of these may be implemented without
review.

## 9. Runtime and reproduction

* One pair step ≈ 0.9 s (RTX 5080 Laptop, bf16 autocast, gradient checkpointing, visual-feature
  cache); I0's 1500 pair steps + 6 evaluations ≈ 23 min. Strict determinism on.
* Block parameters 1,129,985; optimizer groups: LoRA 1e-4, decoder (block) 3e-4, token rows 3e-4,
  cosine over the pair-step budget (warmup 20).
* `python -m pytest tests/ -q` → 418 passed (28 new Task 6I tests).

## 10. File map

| Piece | Path |
|---|---|
| Block + attention diagnostics + inference/eval path | `buildreasonseg_mvp/query_refine.py` |
| Runtime plumbing (install / freeze / pair step) | `buildreasonseg_mvp/runtime.py` |
| Parameter-group + checkpoint plumbing | `buildreasonseg_mvp/model.py`, `checkpointing.py` |
| Config | `configs/mvp/task6i_visual_query_refinement.yaml` |
| Stage scripts | `scripts/task6i_*.py` |
| Evidence | `evaluation/task6i_*.json` |
| Tests | `tests/test_task6i_visual_query_refinement.py` |
