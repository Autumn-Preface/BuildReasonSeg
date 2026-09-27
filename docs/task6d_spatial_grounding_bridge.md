# Task 6D — Spatial Grounding Bridge v0.1

**Verdict:** _pending G0/G1 and the final evaluation_ — see §1.

This is the first Task 6D architecture task: it returns from performance engineering to the model
problem Task 6C characterised. Nothing here adds `[REF]`, a Spatial Relation Encoder, a Spatial
Consistency Loss, 4B parameters, a new dataset or true batching, and nothing reinterprets a Task 6C
model-quality result.

## 1. Verdict

| | Value |
|---|---|
| Oracle point | mIoU **0.4876**, Dice 0.6055, paired **18/20**, own 0.4944 vs cross 0.0205 |
| Oracle box | mIoU **0.7506**, Dice 0.8507, paired **20/20**, own 0.7069 vs cross 0.0000 |
| Section 5 selection | **BOX** (point misses the 0.50 mIoU gate; box passes it and leads by +0.2630) |
| G0 (grounding proof) | _pending_ |
| G1 (joint segmentation) | _pending_ |
| Final verdict | _pending_ |

## 2. Frozen baseline (section 3)

Task 6C's `P_C` arm is the failure baseline and was deliberately not rerun:

| | Task 6C `P_C` |
|---|---|
| strict end-to-end mIoU | 0.10604 |
| paired mask probe | **0/20** |
| same-image prediction-to-prediction IoU | ≈0.999 |
| valid `[SEG]` emission | 120/120 |

Source: `evaluation/task6c_comparison.json` (arm `P_C`), ADR-015, `docs/task6c_prompt_ablation.md`.
The failure is not mask quality but **target selection**: two instructions on one image produce the
same mask, because the sparse prompt was an arbitrary projected language vector rather than a spatial
prediction.

## 3. Oracle point (section 4)

Geometry is derived from the GT target mask only — never from a prediction — and fed through the
official SAM2 prompt encoder:

* **point**: the maximum of the Euclidean distance transform inside the GT mask, ties broken by
  row-major `argmax`, normalized to a pixel centre in `[0,1]²`. On the 120-record validation set:
  **strict mIoU 0.4876**, Dice 0.6055, **paired 18/20**, own 0.4944 vs cross 0.0205
  (margin +0.4739), mean IoU(pred_A, pred_B) 0.0949.

## 4. Oracle box (section 4)

* **box**: the tight GT bounding box (normalized `x1,y1,x2,y2`, `x1 ≤ x2`, `y1 ≤ y2`) through the
  official SAM2 box path: **strict mIoU 0.7506**, Dice 0.8507, **paired 20/20**, own 0.7069 vs cross
  0.0000 (margin +0.7069), mean IoU(pred_A, pred_B) **0.0000**, every pair guaranteed to have
  different geometry.

That comparison is the whole diagnosis in one line: SAM2 recovers the target at **0.75 mIoU when the
prompt is right** against **0.106** in Task 6C. The bottleneck was the prompt, not the segmenter.

## 5. Geometry choice (section 5)

The rule was applied verbatim and is recorded in the artifact:

1. point paired 18/20 ✅ **but mIoU 0.4876 < 0.50** ❌ → rule 1 does not apply;
2. box paired 20/20 ✅ and mIoU 0.7506 ≥ 0.50 ✅ → **BOX** chosen.

Box also beats point by **+0.2630 mIoU**, so the choice is not a tie-break. Combined point+box was not
used (section 5 forbids it). Artifact: `evaluation/task6d_oracle_prompt_diagnostic.json`.

## 6. SpatialGroundingHead (section 6)

```text
LayerNorm(2048) -> Linear(2048, 512) -> GELU -> Linear(512, D) -> sigmoid
```

`D = 4` for the selected BOX geometry. Canonicalization is differentiable: the four sigmoid outputs are
treated as two corners and ordered with `min`/`max`, so `x1 ≤ x2` and `y1 ≤ y2` hold for every input and
gradients still reach all four units. 1,055,236 parameters.

The candidate path is exactly the spec's chain — `image + instruction → Qwen3-VL + text LoRA →
reasoning + [SEG] → [SEG] hidden → SpatialGroundingHead → predicted box → official SAM2 prompt encoder
→ SAM2 mask decoder → mask` — and it never builds the old projected language vector.

**Coordinate convention, verified in the installed source.** `PromptEncoder._embed_points`/`_embed_boxes`
add 0.5 and call `pe_layer.forward_with_coords(points, self.input_image_size)`, which divides by the
image size: the official encoder wants **input-image pixels** (the 1024×1024 frame), not normalized
coordinates. The head predicts normalized `[0,1]` geometry (resolution independent) and
`geometry_to_sam_coords` scales it by 1024. Task 6D data is square 512×512, so the mapping is linear.

## 7. G0 training (section 9)

**G0 ran and failed its gate.** 2 epochs × 480 paired `P` records (the maximum section 9 allows),
198 s + 262 s of training, 399 trainable tensors / 18,489,860 trainable parameters: text LoRA, the
`[SEG]` machinery and the head. Frozen: Qwen base, visual tower, SAM2 image encoder, SAM2 mask decoder,
prompt encoder, and the projection MLP (unused by the candidate). Loss
`2.0 × LM CE + 5.0 × SmoothL1(box)`; the raw components are recorded per logged step.

| | Epoch 1 | Epoch 2 |
|---|---|---|
| LM CE | → **0.0000** | 0.0000 |
| grounding SmoothL1 (train) | 0.0819 → 0.0126 | 0.0649 → 0.0126 |
| **train** box IoU (logged samples) | 0.0548 mean | **0.0009 mean** |
| predicted-box spread (std per coordinate) | [0.053, 0.048, 0.051, 0.062] | **[0.0, 0.0, 0.001, 0.0]** |
| GT box spread (std per coordinate) | [0.255, 0.187, 0.257, 0.186] | same |

## 8. G0 metrics (section 9) and the failure mechanism

| Gate | Requirement | Measured | Pass |
|---|---|---|---|
| valid `[SEG]` | ≥ 90 % | **120/120 = 100 %** | ✅ |
| geometry paired | ≥ 14/20 | **0/20** | ❌ |
| mean GT-box IoU | ≥ 0.35 | **0.0082** | ❌ |

Verdict: **`GROUNDING_REPRESENTATION_FAILED`**, and section 9 says to stop rather than compensate with
4B or extra modules, so **G1 was not run**.

**Mechanism — the head collapsed to a constant box, because the `[SEG]` hidden state carries no target
location.** By the end of epoch 2 every sample produced the same prediction, to three decimals:

```text
predicted box (every validation and training sample): (0.457, 0.455, 0.547, 0.539)
spread across samples:                                [0.000, 0.000, 0.001, 0.000]
```

That is the *optimal constant predictor* under `SmoothL1`: with a read-out that is uninformative about
where the target is, the loss minimum is the mean box, and the head found it. The corroborating numbers:

* the head never even fit the **training** boxes — 0.0548 mean box IoU in epoch 1 and 0.0009 in epoch 2,
  i.e. it moved *towards* the constant, not towards the samples;
* teacher-forced and free-generation metrics are identical and **unchanged between epochs**
  (box IoU 0.0082, centre-inside 1/120), which only happens if the prediction no longer depends on the
  input;
* the free-generation paired probe gives a geometry distance of ~0.0017 between the two instructions of
  one image — the same box for both, so 0/20 with a margin of ~0;
* **LM CE saturated at 0.0000** while the language target is one of 21 distinct reasoning templates
  (Task 6C's finding): the language objective is memorized long before it could pressure the read-out to
  encode *which* building, so it supplies no useful gradient for grounding.

This is a **representation** result, not a wiring or geometry-convention error: the same official prompt
path with GT geometry reaches 0.7506 mIoU (§4), and the head demonstrably receives gradient (§16 tests).
It also reproduces Task 6C's independent measurement — the projected prompt had effective rank 1.5 and
same-image cosine > 0.9999 — now with a completely different read-out (a supervised geometry head instead
of a projected vector), which is what makes the diagnosis strong: **the collapse is in the `[SEG]` hidden
state itself**.

## 9. G1 training (section 10)

**Not run.** Section 9 says a failed G0 must stop with `GROUNDING_REPRESENTATION_FAILED` and must not be
compensated with 4B or extra modules, and section 10 gates G1 on G0 passing. Training a mask decoder
against a constant prompt could only degrade a decoder that the oracle shows is already capable — it
would produce a worse number for a question G0 has already answered.

## 10. Strict end-to-end evaluation (section 11)

Free generation is the primary setting: image + instruction → generated reasoning + exactly one `[SEG]` →
generated `[SEG]` hidden → predicted box → official prompt encoder → mask. Measured from the G0
checkpoint (`G0_epoch2.pt`, step 960); the checkpoint load path is verified because these numbers
reproduce G0's own validation exactly (0.0066 / 0.0082).

| | Value |
|---|---|
| emission | **120/120** (100 %) |
| strict end-to-end mIoU | **0.0066** |
| strict end-to-end Dice | 0.0106 |
| conditional mIoU (valid only) | 0.0066 |
| mean predicted-box IoU vs GT box | 0.0081 |
| predicted-box centre inside target | 1/120 = 0.83 % |
| distinct predicted masks (bitwise) | 120 — but all are the same constant box with ~0.001 coordinate jitter |

Task 6C's `P_C` reference is 0.10604, so the candidate is **worse than the bridge it replaces**. That is
expected and informative rather than embarrassing: a constant *average* box is a worse spatial prompt on
this data than Task 6C's fixed-centre prior, because the average box lands on no particular building.

## 11. Paired geometry probe (section 12)

Free generation, all 20 pairs, `IoU(pred_A, GT_A) > IoU(pred_A, GT_B)` and the symmetric condition:
**geometry paired 0/20**. The more telling number is that the predicted geometry of the two instructions
of one image differs by **0.0008 L1** (distance ≈0.0017) — the two instructions produce the same box.

## 12. Paired mask probe (section 12)

| Metric | Value |
|---|---|
| mask paired | **0/20** |
| mean own-target IoU | 4.0e-10 |
| mean cross-target IoU | 4.0e-10 |
| own-minus-cross margin | −4.5e-13 |
| **IoU(pred_A, pred_B)** | **0.919** |

The two instructions of one image produce a 92 %-identical mask — Task 6C's "same prediction for both
instructions" defect (IoU ≈ 0.999 there) reproduced by a completely different mechanism (a collapsed
geometry head instead of a collapsed projection). The paired probe still discriminates perfectly:
own ≈ cross ≈ 0, margin ≈ 0.

## 13. Representation diagnostics (section 14)

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

**Answer to section 14's key question: no** — instruction variation does not produce spatially different
geometry, so it cannot produce different masks. The control says something sharper than
"instruction-independent": **changing the whole image moves the `[SEG]` hidden state *less* than changing
the instruction wording**, so the read-out at that position is dominated by the reasoning **template**,
with the target contributing a small residual. The head's collapse to the mean box is the optimal
response to exactly that input.

## 14. WHU / pseudo-instance error analysis (section 16)

`evaluation/task6d_error_analysis.json`, computed from the GT component map (WHU polygon components) and
the predicted masks for all 120 free-generation records:

| Flag | Records | Share of the 116 failures |
|---|---|---|
| tiny target (< 1 % of the tile) | 108 | 93 % |
| border truncation | 35 | 30 % |
| insufficient density (≤ 2 components) | 6 | 5 % |
| touching neighbours | 2 | 2 % |
| merged prediction (≥ 2 components ≥ 5 %) | 0 | 0 % |
| ambiguous boundary | 0 | 0 % |
| disconnected target | 0 | 0 % |

**These failures are not attributable to the dataset.** The artifact records
`prediction_degeneracy.degenerate = true` (one constant mask, mean own IoU 0.0066), so all 116 records
fail regardless of their content and the flag distribution describes the failure population rather than a
dataset defect. The dataset is unchanged (section 1) and a dataset-selection task is **not** indicated by
this evidence: the binding constraint is the read-out, not WHU's pseudo-instances.

## 15. Runtime / VRAM

| | Value |
|---|---|
| GPU | RTX 5080 Laptop, 15.894 GiB |
| G0 epoch 1 / epoch 2 | 198 s / 262 s for 480 steps each (≈0.41–0.55 s/step, teacher-forced) |
| Oracle diagnostic | ≈9 min (120 records × 2 geometries + 20 pairs) |
| Final free-generation evaluation | ≈12 min (120 records + 20 pairs + 6 panels) |
| Runtime reused | Task 6C.7 accepted configuration: frozen visual-feature cache ON, `collect_grad_norms=false`, one Phase-B transfer, strict determinism |
| Checkpoints | local and gitignored, 100 MB per epoch, hashes in `evaluation/task6d_checkpoint_manifest.json` |
| Qualitative panels | `evaluation/task6d_panels/` — 6 paired images showing image, GT A/B, predicted mask A/B |

## 16. Tests

`python -m pytest tests/ -q` → see `handoff/FROM_DSH.md` §16 for the recorded result.

`tests/test_task6d_grounding.py` covers all 16 section-18 items: the deterministic interior point lies
inside the mask, the box tightly encloses it, normalized geometry is valid and canonical (including the
`x1 > x2` rejection), GT geometry is used only in supervision/oracle (AST-checked on both paths), free
inference uses predicted geometry only, the grounding loss reaches the head/LoRA/`[SEG]` (model-backed),
the visual tower and SAM2 image encoder stay frozen, paired samples have different targets, both paths use
the official SAM2 prompt encoder, the old arbitrary language sparse prompt is absent from the candidate,
the visual cache stays value-preserving, no test split is used, no 4B/`[REF]`/SRE/SCL appears, and strict
determinism is retained.

## 17. Reproduce

```bash
python scripts/task6d_oracle.py                 # sections 4-5: oracle diagnostic + geometry choice
python scripts/task6d_train.py --stage G0       # section 9:  grounding proof (<= 2 epochs)
python scripts/task6d_evaluate.py --checkpoint artifacts/checkpoints/task6c/task6d_G0/G0_epoch2.pt
python scripts/task6d_evaluate.py --checkpoint <ckpt> --representation-only   # section 14 control
python scripts/task6d_evaluate.py --targets-only            # GT geometry artifact
```

G1 (`--stage G1 --init-from <G0 checkpoint>`) is implemented and gated, but was **not run** because G0
failed. `PYTHONUTF8=1` is required on this machine for readable traces; checkpoints stay local and
gitignored, and only hashes/manifests are committed.
