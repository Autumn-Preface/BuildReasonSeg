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

# FROM_DSH — Task 6H Report: Counterfactual Pair-Aligned Dense Grounding v0.1

_This file holds the Task 6H report. The Task 6G report is preserved in git history and in
`docs/task6g_dense_spatial_grounding.md`; Task 6F in `docs/task6f_target_aware_box_query.md`._

Full design notes: `docs/task6h_counterfactual_pair_grounding.md`, ADR-019.

## 1. Verdict

**`COUNTERFACTUAL_QUERY_SIGNAL_FAILED`.**

The pair machinery is correct and fully verified, and the specified counterfactual ranking objective
is *learned* — after 1500 pair steps on 10 pairs the own-vs-cross ranking is **10/10** with mean
margin **+5.58** (gate ≥ 9/10 and > 0.5) — but the objective is **scale-degenerate**: the audit
shows the mean absolute logit growing 0.148 → **16.32** while the probability-space margin reaches
only **+0.112** and the argmax point lands inside its own target **3/20** (gate ≥ 18/20), with
paired point selection **0/10** (gate ≥ 9/10). The ranking gate is passed by magnifying the logit
field, not by localizing. Per section 12 the task **stops at H0**: H1, the H1 representation
diagnosis and H2 were not run; no SAM2 training; no `[REF]`/SRE/SCL/4B/dataset migration/GUI.

## 2. Frozen Task 6G Evidence (section 2)

Not rerun: selected grid 256×256 / 32 channels; point oracle 0.4883 mIoU / paired 18-20;
`DenseSpatialGroundingHead` = `LayerNorm(2048)→Linear(128)` query, `Conv1x1(32,128)` visual,
dot-product heatmap (270 593 parameters); G0 after 1500 steps: inside-own 0.10, paired point 0/10,
heatmap Dice 0.125; implementation audit found no train/eval, gradient, frozen-backbone or
saturation defect. Accepted conclusion: with only per-sample BCE+Dice the query does not become
instruction-specific enough even on 20 records.

## 3. Canonical Pair Construction (section 4)

`evaluation/task6h_pair_manifest.json`: **240 pairs / 240 images**, assertions all enforced (same
source image, different sample ids, different target component ids, different masks, deterministic
Task 6C `P` payload order, no sample reuse, no test split). Canonical identity hash
`c56f0507da41fd5506ffd0d3c37b056d017de7d48c6f4cc448941a9aaf997c3b`. Every pair carries its
downsampled soft-target overlap diagnostic and **no pair was discarded**: 0/240 pairs overlap (max
soft IoU 0.000).

## 4. Pair-Step Semantics (section 6)

One optimizer step = one pair: one shared frozen SAM2 256×256 feature → query forward for A → query
forward for B (both graphs retained) → both heatmaps → per-sample BCE+Dice + `L_cf` → **one**
backward and optimizer step. Scheduler horizon counts **pair** steps (240/epoch), not record steps.
`evaluation/task6h_token_setup.json` verifies: architecture identical to Task 6G (level
`[1,32,256,256]`, head 270 593 parameters), SAM2 and the shared feature **bit-identical** through a
pair step, `[BOX]`/`[SEG]` rows and all head parameters moved with non-zero gradient, 16 ordinary
rows + the base table exactly unchanged, visual tower carries zero LoRA, `[BOX]` still causally clean
(hidden bit-identical under future-text mutation).

## 5. Counterfactual Loss (sections 7–9)

`s_XY = sum(H_X·M_Y)/max(sum(M_Y), eps)` on the 256×256 soft target masks (logits, as specified);
`L_cf = 0.5*(softplus(1-(s_AA-s_AB)) + softplus(1-(s_BB-s_BA)))`, margin 1.0 fixed;
`L_total = 0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_heatmap_A+L_heatmap_B) + 2.0*L_cf`. All
weights and the margin are constants; nothing was swept. Recorded components per step: reasoning CE,
BCE and Dice per sample, the four region scores, both margins, `L_cf`, total.

## 6. Trainables/Frozen Parameters (section 5/10)

Train: text LoRA (392 tensors / 17 432 576), `{[SEG], [BOX]}` rows (1 tensor / 4 096), the Task 6G
`DenseSpatialGroundingHead` (270 593). Frozen: Qwen base, visual tower, all SAM2, Task 6F box head,
Task 6D grounding head, Projection MLP, no `<loc_*>` rows. Optimizer groups recorded: lora 1e-4 /
token 3e-4 / decoder (dense head) 3e-4, weight decay 0 on the rows, clip 1.0, cosine over the true
pair-step budget.

## 7. H0 Overfit (sections 11–12)

1500 **pair** steps on the same 10 pairs as Task 6G G0, clean Task 6G initialization convention.

| Pair step | pair ranking | strict margin | inside-own | paired point | mean margin | heatmap Dice |
|---|---|---|---|---|---|---|
| 250 | 10/10 | 10/10 | 0/20 | 0/10 | +3.68 | 0.003 |
| 500 | 10/10 | 10/10 | 5/20 | 1/10 | +4.94 | 0.008 |
| 1000 | 10/10 | 10/10 | 4/20 | 1/10 | +5.37 | 0.127 |
| 1500 | **10/10** | **10/10** | **3/20** | **0/10** | **+5.58** | **0.134** |

Gate: ranking ✅, margin ✅, inside-own ❌ (3/20), paired point ❌ (0/10) → **H0 failed**.
`evaluation/task6h_h0_overfit.json`.

## 8. H0 Audit (section 13)

`evaluation/task6h_h0_audit.json` — clean initialization vs the H0 checkpoint:

| Quantity | Before | After |
|---|---|---|
| `L_cf` (mean over the 10 pairs) | 1.3133 | **0.0212** |
| own−cross margin (logits) | −0.0001 | **+5.5788** |
| own−cross margin (probabilities) | −0.0000 | **+0.1124** |
| mean abs logit | 0.148 | **16.32** |
| argmax peakiness | 1.6e-05 | 0.403 |
| point-inside rate | 0.00 | **0.15** |
| A/B hidden distance | 27.4 | 167.7 |
| A/B projected-query distance | 1.49 | 779.5 |

Sign correct (own-score up lowers `L_cf`, cross-score up raises it); `L_cf` decreased over the
history (1.28 → 0.009); `L_cf` gradients reach **every** trainable group (query projection 0.031,
query norm 0.0004, visual 1×1 0.0041, Qwen LoRA 0.0164, token rows 0.0710). Shared feature
bit-identity re-confirmed; pair ordering and target-mask distinctness re-confirmed against the
manifest. Diagnosis: **the specified logit-space margin is scale-degenerate** — it is minimized by
magnifying the field, not by sharpening/relocating the peak.

## 9. H1 Training Curve

**Not run** — section 12 stops the task at the failed H0. `evaluation/task6h_h1_training.json` and
`evaluation/task6h_spatial_eval.json` are deliberately absent.

## 10. Pair-Ranking Metrics

H0 stage (10 pairs): ranking 10/10, strict margin 10/10, mean margin +5.5788 (min +4.41, max +8.24),
probability-space margin +0.112. `evaluation/task6h_paired_probe.json` (stage H0) records the full
per-pair table.

## 11. Point Localization Metrics

H0 stage: point-inside-own 3/20 (0.15), paired point selection 0/10, mean same-image point distance
0.498, **same-image heatmap A/B IoU 0.0032** (the two instructions' thresholded maps barely overlap —
they differ by scale/offset, not by location), heatmap Dice 0.134 (diagnostic), mean 512-px point
error from the record table. The point metrics are exactly the gates that expose the degenerate
ranking solution.

## 12. Representation Diagnostics

Recorded inside the audit (before/after): A/B hidden distance 27.4 → 167.7, projected-query distance
1.49 → 779.5. The query/output separation grew by **magnitude**, not into target specificity — the
opposite of the section 17 hypothesis. A full H1-level representation artifact was not produced
because H1 was not run.

## 13. H2 Segmentation

**Not run** (section 18 gates it on H1). `evaluation/task6h_segmentation_eval.json` is deliberately
absent. Frozen ceilings: Task 6C `P_C` 0.10604 / 0-20; point oracle 0.4876 / 18-20; box oracle
0.7506 / 20-20.

## 14. Paired Mask Probe

**Not run**; the paired probe records `mask_ran: false`.

## 15. L1/L2/L3 + Query Breakdown

H0 stage only; the record table in `evaluation/task6h_error_analysis.json` carries per-record level,
query family, point-inside, 512-px error, peakiness and heatmap Dice. No level or family localizes.

## 16. Error / Data Adequacy Analysis

`evaluation/task6h_error_analysis.json` (H0 stage): all **10/10** failing pairs classified
`pair_ranking_correct_point_outside` — the dominant failure is "ranking correct, point outside",
i.e. the objective's own degeneracy, not data quality. Record-level inside rate 0.150; the WHU
quality flags (tiny target / border / pseudo-instance ambiguity) are reported alongside but cannot
explain a failure that is uniform at 10/10 pairs on memorized records. **WHU is not the binding
constraint of this result.**

## 17. Reusable Inference API

Not created (section 23 gates it on H1). The verified 6G/6H plumbing (`predict_heatmap`,
`forward_heatmap`, `evaluate_pair`) remains available.

## 18. Runtime / VRAM

Pair manifest: seconds (no GPU). Token setup + one-pair smoke: one model load, ~5 min. H0: 1500 pair
steps + 6 query-path evaluations in **~29 min** wall clock (pair steps cost ~1.6× a single-sample
step); ~14 GiB VRAM, batch 1, bf16 autocast, gradient checkpointing, Task 6C.7 visual-feature cache
plus the SAM2 CPU feature cache (frozen encoder outputs only, shared by both instructions of a
pair). Audit: two model states in one process, ~8 min.

## 19. Tests

`tests/test_task6h_counterfactual_grounding.py` adds **22** tests covering the section 25 list: 240
canonical pairs in the recorded manifest, same-image / different-sample / different-target /
different-mask assertions, deterministic order + stable identity hash, builder rejection cases
(wrong image, identical targets, reused samples), region scores on the correct masks, `L_cf` sign
(both analytic and numeric), fixed weights/margin, the recorded architecture identity (Task 6G
unchanged), SAM2 and shared-feature bit-identity, both-forwards-one-backward in the pair step,
pair-step scheduler horizon, the retired Task 6E/6F paths, `[BOX]` causal cleanliness, H2
predicted-point-only, forbidden components, config hygiene, no test split, deterministic JSON, and
the neutral Watt wording with preserved history.

Full suite: **365 passed** in 678 s (`python -m pytest tests/ -q`, including the artifact-consistency
gate) — 343 pre-existing tests plus the 22 Task 6H tests, with no regressions.

## 20. Git / Watt

**Commit: `b4e986e` — `feat: add counterfactual pair grounding loss`** (staged only code,
tests, docs and `evaluation/**` artifacts: no checkpoints, no weights, no feature caches, no
`.conda`, no dataset changes; `artifacts/**` stays gitignored and only
`evaluation/task6h_checkpoint_manifest.json` records the checkpoint hashes).

**Watt Toolkit handling:** the pre-existing Watt processes (`Steam++.exe`,
`Steam++.Accelerator.exe`) were left running untouched for the whole task and are used for the final
push only under the established ownership rules — never closed, never force-killed, no hosts-file,
certificate-store or TLS changes anywhere in this task. The neutral `PROJECT_STATE.md` identity-row
wording (section 3) is now in place and no historical per-task Watt record was altered.

## 21. Recommended next architecture step

Five readouts have now failed with verified implementations, and Task 6H supplies the sharpest
diagnosis: **the objective, not the plumbing, is degenerate**. Recommendations, in order, each
measurable against the frozen gates and the paired `P` material:

1. **Re-define the counterfactual objective on a bounded scale (recommended).** The same pair
   supervision with `s_XY` computed on `sigmoid(H)` (or a softmax over the map) removes the
   magnification shortcut, and the audit's probability margin (+0.112) is the honest number to
   push. Add a peak/coverage term only if the bounded version still fails.
2. **Strengthen the query signal itself** — instruction-aware query refinement over the visual
   features, or an explicit representation objective that separates the two instructions of one
   image (still no `[REF]`, no new dataset).
3. **Only after a localization gate passes**: re-test the dense readout, then H2-style frozen-SAM2
   segmentation and geometry-verifiable supervision.
4. **Not indicated by anything measured here:** `[REF]`, SRE, SCL, 4B, dataset migration, full
   training, GUI.
