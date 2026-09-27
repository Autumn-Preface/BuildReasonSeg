# Task 6D.1 — Corrective G0 Rerun + `[SEG]` Spatial-Decodability Audit

**Verdict: Case F — `PRACTICALLY_NOT_DECODABLE_GEOMETRY`.**

Two Task 6D confounds were real and are now resolved, and neither of them explains the failure:

1. **The scheduler horizon was wrong.** Task 6D built the AdamW scheduler with one epoch as its horizon
   and ran two, so epoch 2 trained at **LR exactly 0** and epoch 1 had decayed to ~1e-5 of peak before its
   own end. The corrective rerun (**G0-R**) fixes only that — and **still fails its gate**
   (emission 120/120, geometry paired **0/20**, box IoU **0.0088**).
2. **LayerNorm is not the confound.** With a fair learning rate the Task 6D head *does* converge, and it
   decodes as well as the raw MLP (20-sample IoU 0.778 vs 0.645; validation 0.0072 vs 0.0075). The first
   probe run that appeared to show LayerNorm destroying the signal was **my probe failing to converge**,
   not a property of the representation.

After the implementation audit, three frozen readouts (linear, raw MLP, LayerNorm MLP) all **converge**
and all fail the same way: they cannot fit even the 480 training samples, and the fit is **no better than
a label-shuffled control**. The oracle result is untouched and remains the ceiling check.

## 1. Task 6D evidence preserved (section 1)

Frozen and not rerun: Oracle Point mIoU **0.4876** / paired **18/20**; Oracle Box mIoU **0.7506** /
paired **20/20**; geometry choice **BOX**. This establishes exactly one thing: *given correct target
geometry, the current SAM2 prompt encoder + decoder can select and segment the intended building well*.
It is an upper-bound diagnostic for this validation material, not a claim that every SAM failure is
solved.

The original G0 artifact is **not erased**: `evaluation/task6d_g0.json` is marked

```
status: VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND
```

with the reason and a pointer to the corrective run.

## 2. Scheduler defect and correction (section 2)

Confirmed in `scripts/task6d_train.py`: `make_optimizer(runtime, steps_per_epoch)` with
`steps_per_epoch = 480` and then two epochs. Reproduced analytically in
`evaluation/task6d1_scheduler_audit.json` using the project's own factor formula:

| checkpoint | defective horizon 480 | corrected horizon 960 |
|---|---|---|
| first optimizer step | 0.050 | 0.050 |
| after warmup | 1.000 | 1.000 |
| **end of epoch 1** | **1.166e-05** | **0.518** |
| **start of epoch 2** | **0.000** | **0.517** |
| final optimizer step | 0.000 | 2.79e-06 |

Observed in the corrective run (three optimizer groups, `lr` at start of epoch 2):
`[5.167e-05, 1.550e-04, 1.550e-04]` — non-zero, as intended; the LR trace now falls smoothly across both
epochs and reaches zero only at the final step. This also **explains the bit-identical epoch-1/epoch-2
validation metrics** Task 6D reported: epoch 2 performed no optimisation at all.

## 3. G0-R: corrective replication (section 3)

Exactly the Task 6D model — `[SEG] hidden → LayerNorm → Linear(2048,512) → GELU → Linear(512,4) →
sigmoid/canonical box` — on the same 480 `P` records, 2 epochs, same LRs, `lambda_ground = 5.0`,
`2.0 LM CE + 5.0 SmoothL1`, box geometry, strict determinism, Task 6C.7 visual cache, SAM decoder frozen.
The **only** change is the scheduler horizon. Artifact: `evaluation/task6d1_g0_corrected.json`.

| Gate | Requirement | G0-R |
|---|---|---|
| valid `[SEG]` | ≥ 90 % | **120/120 = 100 %** ✅ |
| geometry paired | ≥ 14/20 | **0/20** ❌ |
| mean GT-box IoU | ≥ 0.35 | **0.0088** ❌ |

**Verdict `GROUNDING_REPRESENTATION_FAILED` again**, so G1 was not run.

The training trace is informative: at epoch-1 step 160 the head had **train box IoU 0.617**, then decayed
to 0.000 for the rest of training and stayed there through epoch 2 — the readout drifts to the
mean-box optimum and never recovers, exactly as in Task 6D, now with a correct LR schedule.

Full LR traces, train raw grounding loss, train box IoU, train predicted-box spread, validation box IoU,
geometry paired, emission and the head/hidden diagnostics are all recorded in the artifact.

## 4. Frozen representation source (section 5)

`evaluation/task6d1_hidden_extract_manifest.json`. Source: the **Task 6C `P_C` checkpoint before any
Task 6D grounding training**, hash-verified against `evaluation/task6c_checkpoint_manifest.json`
(`e2f55087…`, matches). Extraction is `torch.no_grad`, Qwen frozen, teacher-forced assistant text.

| split | vectors | shape | dtype | image ids | templates |
|---|---|---|---|---|---|
| train (`P`) | 480 | [480, 2048] | float32 | 240 | 21 |
| val (fixed 120) | 120 | [120, 2048] | float32 | 120 | 21 |
| paired (20 images × 2) | 40 | [40, 2048] | float32 | 20 | 21 |

An implementation-audit finding: the 20 paired images are a **different subset of the val split** than the
fixed 120-record validation set (overlap 0), so they are extracted separately. Without this, the paired
probe silently matched zero pairs.

Tensors stay local/gitignored (`artifacts/task6d1_hidden/`); the committed manifest records shape, dtype,
sample-id SHA256 and each file's SHA256.

## 5. Four frozen probes (sections 6-8)

Protocol for all probes: Qwen frozen (features extracted once), only the readout trained, AdamW
full-batch `weight_decay=0`, fixed seed, 6,000 steps, 20-sample overfit first, then the 480-sample fit,
then validation and the paired geometry probe. GT boxes are **probe supervision only**.

**Two implementation defects were found and fixed before any conclusion was drawn** — this is exactly the
Case D protocol doing its job:

* raw hidden vectors have **norm ≈ 112**, so an unscaled readout saturates immediately: the loss
  *increased* and predicted boxes collapsed to zero width. Fixed with **one fixed scalar divisor**
  (`1/112.03`, computed from the training split, no learnable parameters, no per-sample mean subtraction —
  this is not LayerNorm).
* the LayerNorm probe did not converge at the first learning rate (loss rose 0.0443 → 0.1115, constant
  output). Three learning rates were then attempted and the best-converging one provides the metrics; an
  unconverged readout says nothing about a representation.

| Probe | structure | overfit 20 box IoU | train 480 box IoU | val box IoU | paired | val box std |
|---|---|---|---|---|---|---|
| A | `Linear(2048,4)`, no LayerNorm | 0.903 | 0.027 | 0.0042 | 0/20 | 0.14–0.17 |
| B | raw MLP, **no LayerNorm** | 0.645 | 0.036 | 0.0075 | 0/20 | 0.19–0.22 |
| C | **Task 6D head** (LayerNorm MLP) | 0.778 | 0.051 | 0.0072 | 0/20 | 0.21–0.22 |

**Label-shuffled control (the reason the 20-sample overfit cannot be used as evidence):**

| Probe | real train-480 IoU | shuffled train-480 IoU | real − shuffled | shuffled overfit-20 final loss |
|---|---|---|---|---|
| A | 0.027 | 0.020 | **+0.007** | 0.0000 |
| B | 0.036 | 0.020 | **+0.015** | 0.0000 |
| C | 0.051 | 0.010 | +0.041 | 0.113 (did not fit) |

The shuffled readout fits **20 randomly permuted boxes to zero loss**, so "a 1M MLP can overfit 20 frozen
vectors" proves nothing about the representation. At 480 samples the real fit is within a few IoU points
of the shuffled fit for the raw probes, i.e. **no measurable decodable signal**.

## 6. Probe D + expanded control + SVD (sections 9-10)

**The statistics LayerNorm discards carry no box information.** Across the 480 training hiddens:
mean `0.0637 ± 0.0025`, std `2.4747 ± 0.0041`, norm `112.027 ± 0.185` (range 111.53–112.43); Pearson
correlations with box centre-x are 0.019 / 0.052 / 0.052 and with box area 0.013 / 0.056 / 0.057;
an in-sample linear map from those three statistics to the box reaches **R² = 0.0044** (train) and
0.0179 (val). So §9's question resolves to *no*: whatever LayerNorm removes is not where the location is.

**Expanded control on the paired split (19 same-image and 19 different-image pairs):**

| Condition | cosine | **centered cosine** | LayerNorm cosine | L2 |
|---|---|---|---|---|
| same image, different template | 0.99677 | **0.0770** | 0.99677 | 8.97 |
| different image, same template | 0.99814 | **0.5311** | 0.99814 | 6.81 |

* LayerNorm cosine ≈ raw cosine, confirming LayerNorm does **not** remove the shared component (it removes
  the per-sample mean, not the global mean) — consistent with §9's finding that there was nothing to lose.
* Once the shared component is removed, the instruction-dependent residual is nearly **orthogonal**
  (0.077) between the two instructions of one image, while two different images with the *same* template
  are much more similar (0.531). The residual therefore tracks the **instruction wording**, not the target.
* Within-image variance 0.00999 vs a between-target box L1 of **0.306** — the token moves ~30× less than
  the targets differ.
* **SVD (480 × 2048):** effective rank (participation ratio) **8.33**; top-1 explains **31.6 %**, top-5
  52.5 %, top-10 61.5 %, top-50 80.9 %, and 111 components are needed for 90 %. The representation is a
  low-dimensional, largely shared subspace.

## 7. Interpretation matrix (section 11)

| Case | Resolution |
|---|---|
| **A** corrected G0-R passes | **no** — G0-R fails the same gate |
| **B** `LAYER_NORM_READOUT_CONFOUND` | **no** — with a fair LR the LayerNorm probe matches the raw MLP (val 0.0072 vs 0.0075; train 0.051 vs 0.036). The apparent confound was my probe's learning rate |
| **C** memorizable but not generalizable | **no** — the raw MLP does not fit the 480 training samples either (0.036 ≈ shuffled 0.020) |
| **D** readout/feature-identity bug | **cleared** — the audit found and fixed two real implementation defects (input scale, probe LR), after which every readout converges and the linear probe fits 20 samples to IoU 0.90 |
| **E** overfits 20 and 480, val collapses | **no** — the 480 fit never happens, and the 20-sample fit is achievable with shuffled labels |
| **F** cannot fit even 480 after verification | **YES** → `PRACTICALLY_NOT_DECODABLE_GEOMETRY` |

**Accepted causal statement (and the wording rule):** after the implementation was audited and every
readout converged, the frozen `[SEG]` representation contains **no practically decodable target geometry**
under this training setup — a statement about *practical decodability*, not an information-theoretic
absence claim. The Task 6D wording "`[SEG]` hidden carries no target location" is replaced by this, and by
"Task 6D's first grounding readout collapsed; the original result was confounded by a scheduler-horizon
defect and by an unverified LayerNorm hypothesis, both now resolved" (§16).

## 8. Recommended next architecture direction (section 12)

**One option only: explicit coordinate/grid tokens (Option 1).** Make target location a first-class
supervised *sequence* target — quantized spatial tokens emitted by the language model — instead of
expecting the generic `[SEG]` hidden state to encode geometry. Evidence:

* the frozen hidden is norm-112, low-effective-rank (8.33), and its instruction-dependent residual is
  nearly orthogonal between instructions (centered cosine 0.077) while carrying **no** decodable box
  (real ≈ shuffled at 480 samples, paired 0/20) — it encodes the template, not the target;
* the LM objective saturates at CE 0.0000 on 21 reasoning templates, so nothing in the current objective
  ever pressures that residual to encode *which* building;
* the oracle box shows the downstream machinery is already good (0.7506 mIoU, paired 20/20), so the fix
  belongs at the representation/supervision level, not in SAM.

Option 2 (a dedicated target-aware query token) remains a fallback; Option 3 (`[REF]`) is not indicated by
any evidence collected so far. 4B is not a response to this result. Nothing here implements any of them.

## 9. Dataset policy (section 13)

Unchanged: no dataset change in this task. The oracle box result (0.7506 / paired 20/20) shows the current
validation material is sufficient to test this architecture pathway. WHU limitations are recorded
separately and remain open: pseudo-instance ambiguity, touching/merged buildings, border truncation and
tiny-target prevalence were quantified in `evaluation/task6d_error_analysis.json` (tiny target 108/120,
border truncation 35/120 records) with the caveat that Task 6D's failure population was model-caused, not
dataset-caused.

## 10. Artifacts (section 14)

```text
evaluation/task6d1_scheduler_audit.json           before/after horizons + observed LRs
evaluation/task6d1_g0_corrected.json              G0-R report (LR traces, losses, IoU, gate)
evaluation/task6d1_hidden_extract_manifest.json   frozen feature provenance (3 splits)
evaluation/task6d1_probe_linear.json              Probe A
evaluation/task6d1_probe_raw_mlp.json             Probe B (+ shuffled control)
evaluation/task6d1_probe_layernorm_mlp.json       Probe C (+ LR attempts)
evaluation/task6d1_representation_stats.json      Probe D + expanded control + SVD
evaluation/task6d1_decodability_summary.json      interpretation matrix resolution
docs/task6d1_corrective_grounding_audit.md        this report
```

No `task6d1_g1.json` and no `task6d1_paired_probe.json`: G1 was not run because the corrected G0 gate
failed. Hidden tensors stay local/gitignored; only their manifest is committed.

## 11. Tests (section 15)

`python -m pytest tests/ -q` → see `handoff/FROM_DSH.md` §16.

`tests/test_task6d1_corrective_audit.py` covers the 15 required items: the scheduler horizon equals the
actual optimizer-step budget, epoch-2 initial LR is non-zero for the standard 2×480 run, the terminal
cosine factor occurs only at the final full-run step, smoke/max-step mode uses its actual budget, G0-R
differs from Task 6D only by the scheduler correction, extraction does not update Qwen, Probe A/B use raw
hidden with no LayerNorm, Probe C includes LayerNorm, probe labels are GT boxes used only in probe
supervision, no test split, strict determinism, no 4B/`[REF]`/SRE/SCL, no dataset change, and no G1
without a passing corrected gate.

## 12. Reproduce

```bash
python scripts/task6d1_scheduler_audit.py                    # the horizon defect, before/after
python scripts/task6d_train.py --stage G0 --epochs 2 \
    --output evaluation/task6d1_g0_corrected.json             # G0-R (corrected scheduler only)
python scripts/task6d1_extract_hidden.py                     # frozen P_C features (3 splits)
python scripts/task6d1_probes.py --overfit-steps 6000 --train-steps 6000
python scripts/task6d1_representation_stats.py               # Probe D, control, SVD, verdict
```
