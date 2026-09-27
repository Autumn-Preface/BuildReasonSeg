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

# FROM_DSH — Task 6D Report: Spatial Grounding Bridge v0.1

_This file holds the Task 6D report. The Task 6C.7 report is preserved in git history and in
`docs/task6c7_visual_cache_optimization.md`; Task 6C is in `docs/task6c_prompt_ablation.md`._

## 1. Verdict

**`GROUNDING_REPRESENTATION_FAILED`.**

The oracle diagnostic proved that the segmentation machinery was never the problem — with correct
geometry, SAM2 recovers the target at **0.7506 mIoU** against Task 6C's **0.10604** — and then the first
explicit spatial-grounding stage showed *where* the problem actually is: **the `[SEG]` hidden state does
not carry the target's location.** A 1.05M-parameter head supervised on 480 ground-truth target boxes did
not fit even the training boxes; it collapsed to the single constant box that minimises SmoothL1.

Per section 9, a failed G0 means **stop, do not compensate with 4B or extra modules**, so **G1 was not
run**. Every task boundary held: no 4B, no `[REF]`, no Spatial Relation Encoder, no Spatial Consistency
Loss, no dataset change, no true batching, no full training, no GUI.

| Stage | Result |
|---|---|
| Oracle point | mIoU 0.4876, paired 18/20 |
| Oracle box | mIoU **0.7506**, paired **20/20** |
| Section 5 geometry choice | **BOX** (point missed the 0.50 mIoU gate; box passed it and led by +0.2630) |
| G0 grounding proof | **failed**: emission 120/120 ✅, geometry paired **0/20** ❌, box IoU **0.0082** ❌ |
| G1 joint segmentation | **not run** (section 9: stop on G0 failure) |
| Final free-generation evaluation | strict e2e mIoU **0.0066**, paired mask **0/20**, IoU(pred_A, pred_B) 0.919 |

## 2. Frozen Baseline (section 3)

Task 6C's `P_C` arm, not rerun: strict end-to-end mIoU **0.10604**, paired mask probe **0/20**,
same-image prediction-to-prediction IoU ≈**0.999**, valid `[SEG]` emission **120/120**
(`evaluation/task6c_comparison.json`, arm `P_C`). The Task 6C.7 runtime was reused unchanged
(`collect_grad_norms=false`, frozen visual-feature cache, no redundant Phase-B transfer), so the
architecture comparison is not confounded by the performance work.

## 3. Oracle Point (section 4)

One deterministic interior point per GT target mask — the maximum of the Euclidean distance transform,
ties broken by row-major `argmax`, normalized to a pixel centre — through the **official** SAM2 prompt
encoder as a single positive point, on the fixed 120-record validation set and the 20 paired images:

| strict mIoU | Dice | paired | mean own IoU | mean cross IoU | margin | IoU(pred_A, pred_B) |
|---|---|---|---|---|---|---|
| 0.4876 | 0.6055 | **18/20** | 0.4944 | 0.0205 | +0.4739 | 0.0949 |

## 4. Oracle Box (section 4)

The tight GT bounding box through the official SAM2 box path:

| strict mIoU | Dice | paired | mean own IoU | mean cross IoU | margin | IoU(pred_A, pred_B) |
|---|---|---|---|---|---|---|
| **0.7506** | **0.8507** | **20/20** | 0.7069 | 0.0000 | +0.7069 | **0.0000** |

The two oracle rows are the cleanest diagnosis in the whole project so far: *given the right geometry,
SAM2 finds the right building*, and the paired probe separates the two targets perfectly. The Task 6C
failure was therefore in **prompt generation**, not in the segmenter, and not in the oracle's coordinate
convention — the same code path produces 0.75 mIoU when handed GT geometry.

## 5. Geometry Choice (section 5)

Section 5's rule applied verbatim and recorded in the artifact:

1. point paired 18/20 ✅ but mIoU 0.4876 **< 0.50** ❌ → rule 1 does not apply;
2. box paired 20/20 ✅ and mIoU 0.7506 ≥ 0.50 ✅ → **BOX**.

Box also improves mIoU by **+0.2630** over point, so the choice is not a tie-break and the prettier
result was not silently picked. Combined point+box was not used (section 5 forbids it).
Artifact: `evaluation/task6d_oracle_prompt_diagnostic.json`.

## 6. SpatialGroundingHead (section 6)

```text
LayerNorm(2048) -> Linear(2048, 512) -> GELU -> Linear(512, 4) -> sigmoid    # BOX (D = 4)
```

1,055,236 parameters. The four sigmoid outputs are treated as two corners and ordered with `min`/`max`,
which canonicalizes `x1 ≤ x2`, `y1 ≤ y2` **differentiably** (gradients still reach all four units). The
candidate path is the spec's chain exactly, and the old projected language vector is never built:

```text
image + instruction -> Qwen3-VL (text LoRA) -> reasoning + [SEG] -> [SEG] hidden
   -> SpatialGroundingHead -> predicted box -> official SAM2 prompt encoder -> SAM2 mask decoder -> mask
```

**Coordinate convention, verified in the installed source.** `PromptEncoder._embed_points` /
`_embed_boxes` add 0.5 and call `pe_layer.forward_with_coords(points, self.input_image_size)`, which
divides by the image size — so the official encoder expects **input-image pixels** (the 1024×1024 frame),
not normalized coordinates. The head predicts normalized geometry (resolution independent) and
`geometry_to_sam_coords` scales by 1024; Task 6D data is square 512×512, so the map is linear. The oracle
result is the end-to-end proof that this is right.

## 7. G0 Training (section 9)

2 epochs × 480 paired `P` records (the section 9 maximum), 198 s + 262 s. Trainable: text LoRA, the
`[SEG]` machinery and the head (399 tensors, 18,489,860 parameters). Frozen: Qwen base, visual tower,
SAM2 image encoder, **SAM2 mask decoder** (G0 trains no mask path at all), prompt encoder, and the unused
projection MLP. Loss `2.0 × LM CE + 5.0 × SmoothL1(box)`, with the raw components logged.

| | Epoch 1 | Epoch 2 |
|---|---|---|
| LM CE | → **0.0000** | 0.0000 |
| grounding SmoothL1 (train) | 0.0819 → 0.0126 | 0.0649 → 0.0126 |
| train box IoU (logged samples) | 0.0548 mean | **0.0009 mean** |
| predicted-box spread (std per coordinate) | [0.053, 0.048, 0.051, 0.062] | **[0.000, 0.000, 0.001, 0.000]** |
| GT box spread (std per coordinate) | [0.255, 0.187, 0.257, 0.186] | same |

## 8. G0 Metrics (section 9)

| Gate | Requirement | Measured |
|---|---|---|
| valid `[SEG]` | ≥ 90 % | **120/120 = 100 %** ✅ |
| geometry paired | ≥ 14/20 | **0/20** ❌ |
| mean GT-box IoU | ≥ 0.35 | **0.0082** ❌ |

**Mechanism: the head collapsed to a constant box.** By the end of epoch 2 every sample predicted

```text
(0.457, 0.455, 0.547, 0.539)     spread across samples: [0.000, 0.000, 0.001, 0.000]
```

which is the mean box — the *optimal constant predictor* under SmoothL1 when the read-out carries no
location information. Three independent corroborations:

* the head never fit the **training** boxes either (0.0548 → 0.0009 mean box IoU), i.e. it moved towards
  the constant rather than towards the data;
* teacher-forced and free-generation metrics are identical and **unchanged between epochs**
  (box IoU 0.0082, centre-inside 1/120), which can only happen if the prediction no longer depends on
  the input;
* LM CE saturated at 0.0000 while the language target is one of 21 distinct reasoning templates (Task 6C's
  finding), so the language objective was memorized long before it could pressure the read-out to encode
  *which* building, and it supplies no useful grounding gradient.

This is a representation result, not a wiring error: the same prompt path reaches 0.7506 mIoU with GT
geometry, and `tests/test_task6d_grounding.py` proves the grounding loss produces gradients in the head,
the LoRA adapters and the `[SEG]` row. It also **reproduces Task 6C's independent measurement** —
projected effective rank 1.5, same-image cosine > 0.9999 — with a completely different read-out (a
supervised geometry head instead of a projected vector), which is what makes the diagnosis strong: the
collapse is in the `[SEG]` hidden state itself.

## 9. G1 Training (section 10)

**Not run.** Section 9 says a failed G0 must stop with `GROUNDING_REPRESENTATION_FAILED` and must not be
compensated with 4B or extra modules, and section 10 gates G1 on G0 passing. Training a mask decoder
against a constant prompt could only degrade a decoder that the oracle shows is already capable, so
running it would have produced a worse number for a question that G0 already answered.

## 10. Strict E2E (section 11)

Free generation is the primary setting: image + instruction → generated reasoning + exactly one `[SEG]` →
generated `[SEG]` hidden → predicted box → official prompt encoder → mask. From the G0 checkpoint
(`G0_epoch2.pt`, step 960; the load path is verified because these numbers reproduce G0's own
validation exactly):

| | Value |
|---|---|
| emission | **120/120** (100 %) |
| strict end-to-end mIoU | **0.0066** |
| strict end-to-end Dice | 0.0106 |
| conditional mIoU (valid only) | 0.0066 |
| mean predicted-box IoU vs GT box | 0.0081 |
| predicted-box centre inside target | 1/120 = 0.83 % |
| distinct predicted masks | 120 bitwise, but all are the same constant box with ~0.001 coordinate jitter |

Task 6C's `P_C` reference is 0.10604, i.e. the candidate is **worse than the bridge it replaces** — which
is expected and informative: a constant centred box is a worse spatial prompt than Task 6C's fixed-centre
prior on this data, because the constant box is an *average* box that lands on no particular building.

## 11. Paired Geometry Probe (section 12)

Free generation, all 20 pairs, `IoU(pred_A, GT_A) > IoU(pred_A, GT_B)` and symmetric:

**geometry paired 0/20**, though the gate for a plateaued representation is the more telling number:
the predicted geometry of the two instructions of one image differ by **0.0008 L1** (distance ≈0.0017),
i.e. the two instructions produce the *same* box. This is the exact failure mode Task 6C described, now
at the geometry level.

## 12. Paired Mask Probe (section 12)

| Metric | Value |
|---|---|
| mask paired | **0/20** |
| mean own-target IoU | 4.02e-10 |
| mean cross-target IoU | 4.03e-10 |
| own-minus-cross margin | −4.5e-13 |
| **IoU(pred_A, pred_B)** | **0.919** |

The two instructions of one image produce a 92 %-identical mask (not 100 % only because the constant box
retains ~0.001 coordinate jitter), and neither mask overlaps any GT target — the "same prediction for
both instructions" defect of Task 6C is reproduced by a completely different mechanism, and the paired
probe still discriminates: it is the gate that a real fix must move.

## 13. Representation Diagnostics (section 14)

| Metric | Value |
|---|---|
| `[SEG]` hidden cosine, same image / two instructions | **0.99898** |
| `[SEG]` hidden L2, same image | 10.50 |
| head penultimate cosine, same image | 0.99910 |
| predicted geometry L1, same image | **0.00076** |
| mask IoU(pred_A, pred_B) | **0.919** |

**Four-condition control (same/different image × same/different template):**

| Condition | hidden cosine |
|---|---|
| different image, **same template** | **0.99988** ← most similar |
| same image, different template | 0.99898 |
| different image, different template | 0.99869 ← least similar |

**Answer to section 14's key question: no.** Instruction variation does not produce spatially different
predicted geometry, so it cannot produce different masks. The control is sharper than
"instruction-independent": **changing the whole image moves the `[SEG]` hidden state *less* than changing
the instruction wording**, so the read-out at that position is dominated by the reasoning **template**
and the target contributes only a small residual. The head's collapse to the mean box is the optimal
response to exactly that input — which is why the fix has to change where the geometry comes from, not
how the head is built.

## 14. WHU / Pseudo-instance Error Analysis (section 16)

`evaluation/task6d_error_analysis.json` classifies each free-generation record from the GT component map
(neighbouring components, border contact, connectivity, target size) and from the predicted mask's
component coverage:

| Flag | Records | Share of the 116/120 failures |
|---|---|---|
| tiny target (< 1 % of the tile) | 108 | 93 % |
| border truncation | 35 | 30 % |
| insufficient density (≤ 2 components in tile) | 6 | 5 % |
| touching neighbours (adjacent WHU components) | 2 | 2 % |
| merged prediction (≥ 2 components ≥ 5 % of the mask) | 0 | 0 % |
| ambiguous boundary (multi-component **and** IoU < 0.3) | 0 | 0 % |
| disconnected target | 0 | 0 % |

**These failures are not attributable to the dataset.** The artifact's degeneracy guard records that the
model emitted essentially one constant mask, so all 116 failures occur regardless of record content, and
the flag distribution describes the failure population rather than a dataset defect. The dataset is
unchanged (section 1), and a dataset-selection task is **not** indicated by this evidence — the
limitation that matters is in the read-out, not in WHU's pseudo-instances.

## 15. Runtime / VRAM

| | Value |
|---|---|
| GPU | RTX 5080 Laptop, 15.894 GiB |
| G0 epoch 1 / epoch 2 training time | 198 s / 262 s for 480 steps (≈0.41–0.55 s/step, teacher-forced) |
| Free-generation evaluation (120 records + 20 pairs + panels) | ≈12 min |
| Oracle diagnostic (120 records × 2 geometries + 20 pairs) | ≈9 min |
| Peak VRAM during G0 training | recorded in `evaluation/task6d_g0.json` (`vram` per logged step) |
| Task 6C.7 runtime reused | frozen visual-feature cache ON, `collect_grad_norms=false`, one Phase-B transfer |
| Checkpoints | local and gitignored (`artifacts/checkpoints/task6c/task6d_G0/`), 100 MB per epoch, hashes in `evaluation/task6d_checkpoint_manifest.json` |

## 16. Tests

`python -m pytest tests/ -q` → **251 passed** (238 before plus the 13 new Task 6D tests).

`tests/test_task6d_grounding.py` covers all 16 section-18 items: the deterministic interior point lies
inside the mask, the box tightly encloses it, normalized geometry is valid and canonical, GT geometry is
used only in supervision/oracle (AST-checked on both paths), free inference uses predicted geometry only,
the grounding loss reaches the head/LoRA/`[SEG]` (model-backed), the visual tower and SAM2 image encoder
stay frozen, paired samples have different targets, both paths use the official SAM2 prompt encoder, the
old arbitrary language sparse prompt is absent from the candidate, the visual cache stays value-preserving,
no test split is used, no 4B/`[REF]`/SRE/SCL appears, and strict determinism is retained.

## 17. Git / Watt

* Committed and pushed to `Autumn-Preface/BuildReasonSeg` on `main`: commit **`db54e55`**
  (`feat: add spatial grounding bridge`) plus a follow-up `docs:` commit recording this hash and the push
  result (`4b46045..db54e55  main -> main`). Remote `main` was at `4b46045` before this task.
* Artifacts: `evaluation/task6d_oracle_prompt_diagnostic.json`, `task6d_grounding_targets.json`,
  `task6d_g0.json`, `task6d_paired_probe.json`, `task6d_representation.json`,
  `task6d_error_analysis.json`, `task6d_checkpoint_manifest.json`, `task6d_panels/*.png`; report
  `docs/task6d_spatial_grounding_bridge.md`. `task6d_g1.json` does not exist by design (G0 failed).
  No weights, no checkpoints, no `.conda`, no `local_cache`, no dataset edits are staged.
* **Watt Toolkit: `watt_preexisting = true`** (`Steam++.exe` since 2026-09-26 14:16:23). Used for the
  push if needed and **left running**; nothing was force-killed, no hosts file was edited, no certificate
  or TLS setting was changed. All model runs were offline from local cache.

## 18. Recommendation for next architecture task

The evidence points at one specific defect and rules out several alternatives, so the next task should
attack the read-out rather than the prompt bridge or the segmenter:

1. **Give the model a spatially supervised output token, not a free hidden vector.** The `[SEG]` hidden
   state is a *consequence* of the reasoning template, and the language loss saturates at 0.0000 before it
   can impose any spatial content. Emitting the target geometry **as text tokens** (quantized box
   coordinates, e.g. a small vocabulary of grid/coordinate tokens appended to the reasoning) makes the
   geometry a first-class prediction target of the existing LM head: the loss then differs *per sample*
   instead of per template, and the model must attend to the target to reduce it. This needs no new
   module, no 4B and no `[REF]`.
2. **Or supervise the read-out directly.** Keep the head, but add an explicit objective on the `[SEG]`
   hidden state itself (e.g. a learned spatial code trained with the same oracle geometry), so that a
   constant output cannot be optimal. The Task 6D result is exactly the evidence this needs: the
   constant-box optimum is what an unconstrained head converges to.
3. **Keep the oracle diagnostic as the ceiling check for every future hypothesis.** It costs ~9 minutes
   and separates "the segmenter cannot" from "the model does not say where", which is the distinction that
   redirected this task.
4. **Do not re-run Task 6D's G1, and do not re-open the geometry choice**: box is chosen, and training a
   mask decoder against a collapsed prompt cannot answer a question that G0 already answered.
5. **Dataset work is not indicated yet.** The failure population is model-caused (one constant mask for
   120 records); the WHU pseudo-instance flags are recorded for future reference, but nothing in this
   task's evidence says the dataset is the binding constraint.
