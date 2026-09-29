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

# FROM_DSH — Task 6O Report: Geometric Relation Field Causal Decomposition

_This file holds the Task 6O report. The Task 6N report is preserved in git history at commit
`90f3735`; the Task 6M.1 report at `5e52d95`; the Task 6M report at `b9f49f8`._

Full design notes: `docs/task6o_field_causal_decomposition.md`.

## 1. Verdict

**`FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`** — section 15 priority order applied literally:

1. `INVALID_EXPERIMENT` — not applicable: no leakage, no frozen-artifact mutation, no target-input
   leakage, no test access, no protocol violation, and the frozen B2 baseline reproduced bit-identically.
2. `FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE` — not applicable: **B3 passed O1** (mIoU 0.918509 ≥
   0.85, Dice 0.933920 ≥ 0.90).
3. `GEOMETRY_ONLY_BENCHMARK_CONFOUND` — not applicable: **section 14.3 fails**.
4. `DIRECT_REFERENCE_CHANNEL_MATTERS` — not applicable: `B3 − B2 = −0.023159`, which is not below
   `−0.05`.
5. **`FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`** — B3 O1 passes, **14.1 passes**, **14.2 passes**,
   **14.3 does not pass**.

## 2. What Task 6O answers

1. **Does the direct `M_ref_down` channel still matter once `P_rel` is supplied?** Measured: removing
   it costs **0.023159 mIoU** (0.453127 → 0.429968), **0.036689 Dice**, **2 paired passes** (16/20 →
   14/20) and **0.043993** own−cross margin. That is inside the predeclared 0.03 slack, so the direct
   reference channel is **not required** for this directional decoder (14.1 passes).
2. **Is the gain field-guided visual segmentation?** Measured: the geometry-only control loses
   **0.383331 mIoU** (0.429968 → 0.046637), **0.465169 Dice**, **12 paired passes** (14/20 → 2/20) and
   **0.348241** own−cross margin, and it cannot even fit 20 samples (O1 mIoU 0.109511). So the
   handcrafted field alone does **not** solve the benchmark (14.3 fails, 14.2 passes).

## 3. Frozen-asset verification (before anything ran)

`evaluation/task6o_b2_reproduction.json`: all four Task 6N packs (`overfit20`, `mini_train_1000`,
`mini_val_240`, `paired_val_20`) match `evaluation/task6n_pack_manifest.json` **byte-for-byte**, and the
frozen Task 6N B2 checkpoint matches its recorded SHA256 exactly. No pack was regenerated.
`buildreasonseg_mvp/geometric_relation_field.py` and `configs/spatial_relations_v1.yaml` are unchanged
relative to the Task 6O base commit. `evaluation/task6n_*.json` and the Task 6N B0/B1/B2 checkpoints are
untouched. Scope stayed at the same 8 directional L2 programs; the test split was never read.

## 4. Part A — B2 re-evaluated exactly once

Frozen Task 6N evaluator, frozen B2 checkpoint, **no retraining**:

| Comparison | Stored Task 6N | Reproduced | Absolute delta |
|---|---|---|---|
| MiniVal240 mIoU | 0.4531265609993713 | 0.4531265609993713 | **0.0** |
| MiniVal240 Dice | 0.5768438150123932 | 0.5768438150123932 | **0.0** |
| PairedVal20 pass | 16 | 16 | exact |
| paired mean own IoU | 0.44334608244093643 | 0.44334608244093643 | **0.0** |
| paired mean cross IoU | 0.002157857978561074 | 0.002157857978561074 | **0.0** |

Verdict **`B2_REPRODUCED`** (tolerance 1e-6; every delta is exactly 0.0). This also proves that adding
B3/B4 changed nothing about the frozen B0/B1/B2 behaviour.

## 5. The two new variants

| Variant | Fusion input | First conv in-ch | Parameters |
|---|---|---|---|
| **N-B3** | `visual_128` + `P_rel` + relation_embed | 145 | 274,625 |
| **N-B4** | `Conv1x1(1→128)+GN+GELU` on `P_rel`, then + relation_embed | 144 | 240,833 |

B3 never receives the direct reference mask and B4 receives neither the visual feature nor the
reference mask (both "not used" properties are asserted by tests). The field, relation embedding
(4 × 16), trunk, loss (`BCEWithLogitsLoss + DiceLoss`) and 512 × 512 bilinear evaluation are identical
to Task 6N; `P_rel` is still generated outside the decoder by the unchanged
`GeometricRelationField v0.1` (alpha 1.2, tau 0.04, `s_axis = s_margin = 0.02`, softness `tau/2`).

## 6. Stage O1 — Overfit20 (exact Task 6N N1 settings)

AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no augmentation / seed 20260929 /
evaluate every 100 steps:

| Variant | best mIoU | best Dice |
|---|---|---|
| **N-B3** | **0.918509** | **0.933920** |
| N-B4 | 0.109511 | 0.162109 |

**O1 gate PASS** (B3 only; B4 has no gate). B4 could not fit 20 samples within the same budget.

## 7. Stage O2 — MiniTrain1000 → MiniVal240 (exact Task 6N N2 settings)

AdamW / lr 3e-4 / wd 1e-4 / batch 8 / ≤ 25 epochs / early stopping patience 5 on val mIoU / seed
20260929 / fresh initialisation / model selection = best MiniVal240 mIoU.

| Metric | B1 (frozen 6N) | **B2 (reproduced)** | **B3** | **B4** |
|---|---|---|---|---|
| MiniVal240 mIoU | 0.241316 | **0.453127** | **0.429968** | **0.046637** |
| MiniVal240 Dice | — | 0.576844 | 0.540155 | 0.074986 |
| Precision@0.5 | — | — | 0.692450 | 0.708565 |
| per relation mIoU (left/right/above/below) | — | — | 0.4219 / 0.3905 / 0.4581 / 0.4493 | 0.0464 / 0.0508 / 0.0455 / 0.0438 |
| largest-ref / smallest-ref | — | — | 0.4101 / 0.4498 | 0.0420 / 0.0512 |
| border-target (n=114) | — | — | 0.4132 | 0.0753 |
| tiny-target (n=4) | — | — | ≈0 | 0.00023 |
| parameters | 274,625 | 275,777 | 274,625 | 240,833 |
| peak VRAM / wall time | — | — | 0.642 GB / 116.2 s | 0.626 GB / 45.0 s |
| selected epoch (epochs run) | 11 (16) | 9 (14) | 14 (19) | 3 (8) |

B4's precision@0.5 of 0.708565 together with a near-zero mIoU follows from predicting almost no
positive pixels; reported as measured, without interpretation.

## 8. PairedVal20 (exact frozen Task 6N pack)

| Variant | pass | mean own IoU | mean cross IoU | own − cross |
|---|---|---|---|---|
| **B2 (reproduced)** | **16/20** | 0.443346 | 0.002158 | **+0.441188** |
| **B3** | **14/20** | 0.398969 | 0.001773 | **+0.397196** |
| **B4** | 2/20 | 0.048955 | 0.000000 | +0.048955 |

## 9. Predeclared comparisons and criteria (section 14)

```text
delta_B3_B2 = -0.023159   (Dice -0.036689, paired -2, margin -0.043993)
delta_B3_B1 = +0.188652
delta_B3_B4 = +0.383331   (Dice +0.465169, paired +12, margin +0.348241)
```

* **14.1 direct reference-channel retention — PASS**: `0.429968 >= 0.423127` (B2 − 0.03) and
  `14/20 >= 14` and `+0.397196 >= 0.10`.
* **14.2 visual contribution — PASS**: `+0.383331 >= 0.10`.
* **14.3 geometry-only confound — FAIL**: `0.046637 < 0.379968` (B3 − 0.05) and `2 < 12`.

No threshold, gate or comparison was altered.

## 10. Tests, storage, git

`python -m pytest tests/ -q` → **627 passed, 1 skipped** (Task 6N ended at 597 passed / 1 skipped; no
prior passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism
check that needs the proposal env). `tests/test_task6o_field_causal_decomposition.py` covers the 30
section-17 checks: Task 6N artifacts unchanged, pack hashes exact, B2 checkpoint hash matches,
reproduction within tolerance, no test access, only the 8 programs, target never an input, B3 receives
visual + field + relation only, B3 never receives the reference mask, B4 receives field + relation only,
B4 never receives the visual feature or the reference mask, B4's projection is 1 → 128 and its trunk
input is 144, the field is unchanged, alpha/tau/softness unchanged, the frozen SAM2 path is unchanged,
the same packs are used across B2/B3/B4, O1 settings equal Task 6N N1, O2 settings equal Task 6N N2,
the seed is exactly 20260929, no new relation, no nearest/L3, no `[REF]`, no GRCL/SCL, no graph
transformer, no proposal training, no 4B, no download/install, no GUI, and the previous suite is
preserved.

Not committed: checkpoints, the frozen feature cache, SAM2 weights, source imagery/vector data,
`.conda`, large caches. Committed: code, small JSON artifacts, docs, tests, handoff.

Watt was **not needed** in Task 6O: this task downloaded nothing (no new weights, packages or datasets)
and installed nothing. The pre-existing Watt instance is transport-only, is not owned by this project,
and was left running, per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 11. Interpretation boundary (section 16)

DSH reports measurements only and decides none of: whether this is paper novelty; whether
predicted-reference grounding should be implemented; whether the field formula should change; whether
GRCL should be added. No architecture decision is taken in this task.

## 12. Recommended next step

等待 ChatGPT 根据 Task 6O 因果分解结果决定后续架构，不自行开始 predicted-reference、nearest、L3 或 GRCL。

## 13. STOP

Task 6O stops here: no predicted-reference grounding, no nearest, no L3, no GRCL, no full-dataset
training, no proposal optimisation, no GUI. Waiting for the ChatGPT audit.
