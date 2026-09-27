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

# FROM_DSH — Task 6D.1 Report: Corrective G0 Rerun + `[SEG]` Spatial-Decodability Audit

_This file holds the Task 6D.1 report. The Task 6D report is preserved in git history and in
`docs/task6d_spatial_grounding_bridge.md`; Task 6C.7 is in `docs/task6c7_visual_cache_optimization.md`._

## 1. Verdict

**Case F — `PRACTICALLY_NOT_DECODABLE_GEOMETRY`.**

Both Task 6D confounds were real, and resolving them does **not** rescue the result:

* the **scheduler horizon was wrong** (epoch 2 trained at LR exactly 0). The corrective rerun **G0-R**
  fixes only that and **still fails the gate**: emission 120/120, geometry paired **0/20**, box IoU
  **0.0088** → `GROUNDING_REPRESENTATION_FAILED` again, so G1 was not run;
* **LayerNorm is not the confound**. Given a fair learning rate the Task 6D head converges and decodes as
  well as the raw MLP (20-sample 0.778 vs 0.645; validation 0.0072 vs 0.0075). The first probe run that
  looked like a LayerNorm effect was **my probe failing to converge**, not a property of the
  representation.

After an implementation audit that found and fixed **two real defects** (unscaled norm-112 inputs, and an
unconverged LayerNorm probe), all three frozen readouts converge and all fail identically: they cannot fit
even the **480 training samples**, and the fit is **no better than a label-shuffled control**. The oracle
result is untouched. The accepted statement is deliberately the weaker one — *practically* not decodable,
not an information-theoretic absence.

## 2. Task 6D Evidence Preserved (section 1)

Frozen, not rerun: Oracle Point mIoU **0.4876** / paired **18/20**; Oracle Box mIoU **0.7506** / paired
**20/20**; geometry choice **BOX**. Meaning: given correct target geometry the current SAM2 prompt encoder
+ decoder select and segment the intended building well — an upper-bound diagnostic for this validation
material, not a claim that every SAM failure is solved.

The original G0 artifact is **not erased**. `evaluation/task6d_g0.json` now carries

```
status: VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND
status_reason: <the horizon defect, the epoch-2 zero-LR consequence, and the bit-identical
                epoch-1/epoch-2 metrics it explains>
superseded_by: evaluation/task6d1_g0_corrected.json
```

## 3. Scheduler Defect and Correction (section 2)

`scripts/task6d_train.py` built the scheduler with `steps_per_epoch = 480` and then ran two epochs.
Reproduced with the project's own factor formula in `evaluation/task6d1_scheduler_audit.json`:

| checkpoint | defective horizon 480 | corrected horizon 960 |
|---|---|---|
| first optimizer step | 0.050 | 0.050 |
| after warmup | 1.000 | 1.000 |
| **end of epoch 1** | **1.166e-05** | **0.518** |
| **start of epoch 2** | **0.000** | **0.517** |
| final optimizer step | 0.000 | 2.79e-06 |

Observed in G0-R (three optimizer groups at the start of epoch 2):
`[5.167e-05, 1.550e-04, 1.550e-04]` — non-zero as intended, decaying smoothly to a final ratio of
**5.4e-06 of peak** at the last scheduled step. This also explains Task 6D's bit-identical epoch-1 and
epoch-2 validation metrics: **epoch 2 performed no optimisation at all.**

The correction: `total_optimizer_steps = planned_epochs × steps_per_epoch`, where `planned_epochs` is
derived from the loop (so a truncated smoke run uses its real budget). Seven LR checkpoints plus a 20-step
trace are recorded per run; a regression test asserts the horizon, the non-zero epoch-2 LR and the
terminal-only-at-the-end property.

## 4. G0-R: Corrective Rerun (section 3)

Exactly the Task 6D model on the same 480 `P` records for 2 epochs: same layer structure, LRs,
`lambda_ground = 5.0`, `2.0 LM CE + 5.0 SmoothL1`, BOX geometry, text-only LoRA + `[SEG]` + head
trainable, SAM decoder frozen, strict determinism, Task 6C.7 visual cache. **The only change is the
scheduler horizon.** Artifact: `evaluation/task6d1_g0_corrected.json`.

Training trace: at epoch-1 step 160 the head had **train box IoU 0.617**, then decayed to 0.000 for the
remainder of training and stayed there through epoch 2. The readout drifts to the mean-box optimum and
never recovers — the same failure mode as Task 6D, now with a correct LR schedule.

## 5. G0-R Gate (section 3)

| Gate | Requirement | G0-R |
|---|---|---|
| valid `[SEG]` | ≥ 90 % | **120/120 = 100 %** ✅ |
| geometry paired | ≥ 14/20 | **0/20** ❌ |
| mean GT-box IoU | ≥ 0.35 | **0.0088** ❌ |

**`GROUNDING_REPRESENTATION_FAILED` again; G1 not run** (section 3 gates it on a passing G0-R).

## 6. Frozen Representation Source (section 5)

`evaluation/task6d1_hidden_extract_manifest.json`. Source: the **Task 6C `P_C` checkpoint before any
Task 6D grounding training**, hash-verified against the Task 6C manifest (`e2f55087…`, **matches**).
Extraction is `torch.no_grad`, Qwen frozen, teacher-forced assistant text, read-out at
`batch.seg_position`.

| split | vectors | shape | dtype | images | templates |
|---|---|---|---|---|---|
| train (`P`) | 480 | [480, 2048] | float32 | 240 | 21 |
| val (fixed 120) | 120 | [120, 2048] | float32 | 120 | 21 |
| paired (20 × 2) | 40 | [40, 2048] | float32 | 20 | 21 |

**Implementation-audit finding:** the 20 paired images are a **different subset of the val split** than the
fixed 120-record set (overlap **0**). They are now extracted separately; without that, the paired probe
silently matched zero pairs. Tensors stay local/gitignored; the manifest commits shape, dtype, sample-id
SHA256 and file SHA256.

## 7. Probe Implementation Audit (section 4's Case D protocol)

Two defects were found and fixed **before** any conclusion was drawn:

1. **Unscaled inputs.** Raw hiddens have norm ≈ **112**, so an unscaled readout saturates immediately: the
   loss *increased* and predicted boxes collapsed to zero width. Fixed with one **fixed scalar divisor**
   (`1/112.03`, from the training split; no learnable parameters, no per-sample mean subtraction — it is
   not LayerNorm).
2. **Unconverged LayerNorm probe.** At the first learning rate the loss rose (0.0443 → 0.1115) with a
   constant output. Three learning rates were then attempted and the best-converging one provides the
   metrics, with every attempt recorded.

An unconverged readout says nothing about a representation, which is exactly why the Case D protocol
exists in the spec.

## 8. Probes A/B/C (sections 6-8)

Qwen frozen; only the readout trains (AdamW full-batch, `weight_decay=0`, fixed seed, 6,000 steps,
20-sample overfit first, then 480-sample fit, then validation and the paired probe). GT boxes are probe
supervision only.

| Probe | structure | overfit 20 | train 480 | val | paired |
|---|---|---|---|---|---|
| A | `Linear(2048,4)`, no LayerNorm | 0.903 | 0.027 | 0.0042 | 0/20 |
| B | raw MLP, **no LayerNorm** | 0.645 | 0.036 | 0.0075 | 0/20 |
| C | **Task 6D head** (LayerNorm MLP) | 0.778 | 0.051 | 0.0072 | 0/20 |

**LayerNorm is not a confound**: with a fair LR, Probe C matches or beats the LayerNorm-free Probe B at
every budget (train 0.051 vs 0.036, val 0.0072 vs 0.0075). The apparent LayerNorm effect was my probe's
learning rate.

## 9. Label-Shuffled Control (why the 20-sample overfit is not evidence)

| Probe | real train-480 IoU | shuffled train-480 IoU | real − shuffled |
|---|---|---|---|
| A | 0.027 | 0.020 | **+0.007** |
| B | 0.036 | 0.020 | **+0.015** |
| C | 0.051 | 0.010 | +0.041 |

The shuffled readout fits **20 randomly permuted boxes to zero loss** (final loss 0.0000), so
"a 1M MLP can overfit 20 frozen vectors" proves nothing about the representation. At 480 samples the real
fit is within a few IoU points of the shuffled fit for the raw probes: **no measurable decodable signal**.

## 10. Probe D, Expanded Control and SVD (sections 9-10)

**The statistics LayerNorm discards carry no box information.** Across 480 hiddens: mean
`0.0637 ± 0.0025`, std `2.4747 ± 0.0041`, norm `112.027 ± 0.185` (range 111.53–112.43). Pearson with box
centre-x: 0.019 / 0.052 / 0.052; with box area: 0.013 / 0.056 / 0.057. An in-sample linear map from those
three statistics to the box reaches **R² = 0.0044** (train) / 0.0179 (val). LayerNorm had nothing to
destroy.

**Expanded control (19 same-image pairs, 19 different-image pairs):**

| Condition | cosine | **centered cosine** | LayerNorm cosine | L2 |
|---|---|---|---|---|
| same image, different template | 0.99677 | **0.0770** | 0.99677 | 8.97 |
| different image, same template | 0.99814 | **0.5311** | 0.99814 | 6.81 |

After removing the shared component, the instruction-dependent residual is nearly **orthogonal** between
the two instructions of one image (0.077) while two different images with the *same* template are much
more similar (0.531): **the residual tracks the instruction wording, not the target.** LayerNorm cosine ≈
raw cosine, again confirming it does not remove the shared component. Within-image variance **0.00999**
against a between-target box L1 of **0.306** — the token moves ~30× less than the targets differ.

**SVD (480 × 2048):** effective rank (participation ratio) **8.33**; top-1 explains **31.6 %**, top-5
52.5 %, top-10 61.5 %, top-50 80.9 %; 111 components for 90 %. A low-dimensional, largely shared subspace.

## 11. Interpretation Matrix (section 11)

| Case | Resolution |
|---|---|
| A corrected G0-R passes | **no** — G0-R fails the same gate |
| B `LAYER_NORM_READOUT_CONFOUND` | **no** — with a fair LR the LayerNorm probe matches the raw MLP |
| C memorizable but not generalizable | **no** — the raw MLP does not fit the 480 training samples either |
| D readout/feature-identity bug | **cleared** — two real defects found and fixed; all readouts now converge |
| E overfits 20 and 480, val collapses | **no** — the 480 fit never happens; the 20-sample fit also fits shuffled labels |
| F cannot fit even 480 after verification | **YES** |

## 12. Accepted Causal Conclusion (section 16)

Replacing Task 6D's wording, as the spec requires:

> Task 6D's first grounding readout collapsed; the original result is confounded by a scheduler-horizon
> defect and by the use of LayerNorm before the readout. Spatial decodability is under corrective audit.

with the audit now complete:

> After the implementation was audited and every readout converged, the frozen `[SEG]` representation of
> the Task 6C `P_C` checkpoint contains **no practically decodable target geometry** under this training
> setup. The scheduler-horizon defect is real but not causal; LayerNorm is not a confound; the
> representation is norm-112, low-effective-rank, and its instruction-dependent residual encodes the
> reasoning template rather than the target.

This is a **practical-decodability** statement, not an information-theoretic absence claim. The oracle
conclusion (correct geometry → 0.7506 mIoU, paired 20/20) remains frozen.

## 13. Recommended Next Direction (section 12)

**Exactly one option: explicit coordinate/grid tokens (Option 1).** Make target location a first-class
supervised *sequence* target — quantized spatial tokens emitted by the language model — instead of
expecting the generic `[SEG]` hidden state to encode geometry. Evidence: the frozen hidden is norm-112 and
effective-rank 8.33; its instruction-dependent residual is nearly orthogonal between instructions
(centered cosine 0.077) yet carries no decodable box (real ≈ shuffled at 480 samples, paired 0/20); the LM
objective saturates at CE 0.0000 on 21 reasoning templates, so nothing pressures that residual to encode
*which* building; and the oracle box proves the downstream machinery is already capable.

Option 2 (a dedicated target-aware query token) stays a fallback; Option 3 (`[REF]`) is not indicated by
any evidence collected so far; 4B is not a response to this result. **Nothing here implements any of
them** (section 12: do not implement automatically).

## 14. Dataset Policy (section 13)

Unchanged: no dataset change. The oracle box result shows the current validation material is sufficient to
test this pathway. WHU limitations remain recorded and open (pseudo-instance ambiguity, touching/merged
buildings, border truncation, tiny-target prevalence — quantified in
`evaluation/task6d_error_analysis.json`: tiny target 108/120, border truncation 35/120 records), with the
caveat that Task 6D's own failure population was model-caused rather than dataset-caused.

## 15. Runtime / VRAM

| | Value |
|---|---|
| GPU | RTX 5080 Laptop, 15.894 GiB |
| G0-R | 2 epochs × 480 steps plus per-epoch validation (free generation on 120 records + 20 paired) |
| Hidden extraction | 640 teacher-forced forwards, `torch.no_grad`, ≈2 min |
| Probes A/B/C | 6,000 steps each on pre-extracted features (full batch) plus the shuffled controls: ≈4 min total |
| Representation stats | CPU only (SVD of a 480×2048 matrix, control statistics) |
| Checkpoints | local and gitignored; the corrective run's are under `artifacts/checkpoints/task6c/task6d1_G0_corrected/`, and `evaluation/task6d_checkpoint_manifest.json` records that the Task 6D bytes were replaced |

## 16. Tests

`python -m pytest tests/ -q` → **263 passed** (251 before plus the 12 new Task 6D.1 tests).

`tests/test_task6d1_corrective_audit.py` covers all 15 section-15 items: the scheduler horizon equals the
actual optimizer-step budget; epoch-2 initial LR is non-zero for the standard 2×480 run; the terminal
cosine factor occurs only at the final full-run step (by ratio, since it is 5.4e-06 of peak, not exactly
zero); smoke/max-step mode uses its actual budget; G0-R differs from Task 6D only by the scheduler
correction; extraction never updates Qwen and uses the hash-verified checkpoint; Probes A and B are
LayerNorm-free while Probe C includes it; probe labels are GT boxes used only in probe supervision; no test
split; strict determinism; no 4B/`[REF]`/SRE/SCL; no dataset change; and no G1 without a passing corrected
gate. The odd-looking `VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND` marker on the Task 6D artifact is
asserted too, so the history cannot be silently rewritten.

## 17. Git / Watt

* Committed and pushed to `Autumn-Preface/BuildReasonSeg` on `main` with the recommended message
  `fix: audit spatial grounding representation`; the exact hash and push result are recorded in a
  follow-up `docs:` commit and in the DSH turn response. Remote `main` was at `35a327f` before this task.
* Artifacts: `evaluation/task6d1_scheduler_audit.json`, `task6d1_g0_corrected.json`,
  `task6d1_hidden_extract_manifest.json`, `task6d1_probe_linear.json`, `task6d1_probe_raw_mlp.json`,
  `task6d1_probe_layernorm_mlp.json`, `task6d1_representation_stats.json`,
  `task6d1_decodability_summary.json`; report `docs/task6d1_corrective_grounding_audit.md`. No
  `task6d1_g1.json` / `task6d1_paired_probe.json` (G1 not run, by gate). No weights, checkpoints, hidden
  tensors, `.conda`, `local_cache` or dataset edits are staged.
* **Watt Toolkit: `watt_preexisting = true`** (`Steam++.exe` since 2026-09-26 14:16:23). Used for the push
  if needed and **left running**; nothing was force-killed, no hosts file was edited, no certificate or
  TLS setting was changed. All model runs were offline from local cache.

## 18. Artifacts (section 14)

```text
evaluation/task6d1_scheduler_audit.json           before/after horizons + observed LR verification
evaluation/task6d1_g0_corrected.json              G0-R (LR traces, losses, box IoU, gate)
evaluation/task6d1_hidden_extract_manifest.json   frozen feature provenance (3 splits, hash-verified)
evaluation/task6d1_probe_linear.json              Probe A (+ shuffled control)
evaluation/task6d1_probe_raw_mlp.json             Probe B (+ shuffled control, no LayerNorm)
evaluation/task6d1_probe_layernorm_mlp.json       Probe C (+ LR attempts)
evaluation/task6d1_representation_stats.json      Probe D, expanded control, SVD
evaluation/task6d1_decodability_summary.json      interpretation matrix resolution
docs/task6d1_corrective_grounding_audit.md        report
```
