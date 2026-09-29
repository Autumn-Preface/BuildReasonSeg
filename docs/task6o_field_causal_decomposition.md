# Task 6O — Geometric Relation Field Causal Decomposition

> Task: `handoff/TO_DSH.md` (Task 6O) · Base commit: `90f3735` · Predecessor: Task 6N →
> `GEOMETRIC_RELATION_FIELD_FEASIBLE`
> **Verdict: `FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`**
> Tests: `tests/test_task6o_field_causal_decomposition.py` · Evidence: `evaluation/task6o_*.json`

Narrow causal-decomposition experiment on the Task 6N result. Everything is **oracle-reference**
(`reference_source = oracle_native_gt`), the GT target is label/evaluation only, the **test split was
never read**, and the frozen Task 6N packs were reused **byte-for-byte**.

## 1. The two questions Task 6O answers

1. Does the direct `M_ref_down` channel still matter once `P_rel` is supplied?
2. Is the gain really field-guided **visual** segmentation, or does the handcrafted geometric field
   alone solve most of the directional benchmark?

## 2. Task 6N N0 clarification (not a bug, nothing rewritten)

In `evaluation/task6n_field_sanity.json`:

* `mean_other_building_score` **includes** the GT target, because section 13 of Task 6N asked for the
  mean over *every other native building*;
* `mean_best_distractor_score` **excludes** both the reference and the GT target.

No Task 6N artifact was rewritten.

## 3. Frozen assets verified before anything ran

`evaluation/task6o_b2_reproduction.json` records the verification: the four Task 6N packs
(`overfit20`, `mini_train_1000`, `mini_val_240`, `paired_val_20`) all match
`evaluation/task6n_pack_manifest.json` byte-for-byte, and the frozen Task 6N B2 checkpoint matches its
recorded SHA256 **exactly**. `buildreasonseg_mvp/geometric_relation_field.py` and
`configs/spatial_relations_v1.yaml` are unchanged (`git diff` against the base commit is empty).
Scope is the same 8 directional L2 programs; no nearest, L1, L3, new relation, predicted reference,
GRCL or new data.

## 4. Part A — B2 reproduced bit-identically, once

The frozen Task 6N evaluator was run **exactly once** on the frozen B2 checkpoint (no retraining):

| Comparison | Stored Task 6N | Reproduced | Absolute delta |
|---|---|---|---|
| MiniVal240 mIoU | 0.4531265609993713 | 0.4531265609993713 | **0.0** |
| MiniVal240 Dice | 0.5768438150123932 | 0.5768438150123932 | **0.0** |
| PairedVal20 pass | 16 | 16 | exact |
| paired mean own IoU | 0.44334608244093643 | 0.44334608244093643 | **0.0** |
| paired mean cross IoU | 0.002157857978561074 | 0.002157857978561074 | **0.0** |

Verdict **`B2_REPRODUCED`** (tolerance 1e-6, all deltas exactly 0.0). This also confirms that adding
B3/B4 to the shared decoder left the frozen B0/B1/B2 behaviour bit-identical.

## 5. The two new variants

| Variant | Fusion input | First conv in-ch | Parameters | Purpose |
|---|---|---|---|---|
| **N-B3** | `visual_128` + `P_rel` + relation_embed | **145** | **274,625** | field-guided visual model **without** the direct reference channel |
| **N-B4** | `field_128` (`Conv1x1(1→128)` + GN + GELU) + relation_embed | **144** | **240,833** | geometry-only capacity control (**no** visual, **no** reference mask) |

Both use the Task 6N visual projection (B3), trunk, relation embedding (4 × 16), loss
(`BCEWithLogitsLoss + DiceLoss`) and bilinear upsample to 512 × 512. `P_rel` is still generated
outside the decoder from the oracle reference mask by the unchanged `GeometricRelationField v0.1`
(`alpha 1.2`, `tau 0.04`, `s_axis = s_margin = 0.02`, softness `tau/2`). B4 is called with
`visual=None` and `mask_ref_down=None`, so the RGB-derived feature is never handed to it at all.

## 6. Stage O1 — Overfit20 (exact Task 6N N1 settings)

`evaluation/task6o_overfit20.json`, AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler /
no augmentation / seed 20260929 / evaluate every 100 steps:

| Variant | best mIoU | best Dice | final mIoU | final Dice | paired own−cross (4 pairs) |
|---|---|---|---|---|---|
| **N-B3** | **0.918509** | **0.933920** | 0.918509 | 0.933920 | see artifact |
| N-B4 | 0.109511 | 0.162109 | 0.109511 | 0.162109 | see artifact |

**O1 gate (applies to B3 only): PASS** (mIoU 0.9185 ≥ 0.85, Dice 0.9339 ≥ 0.90). B4 has no gate; it
could not even fit 20 samples (loss stayed ≈ 1.0), which is reported as measured.

## 7. Stage O2 — MiniTrain1000 → MiniVal240 (exact Task 6N N2 settings)

`evaluation/task6o_o2_training.json`, AdamW / lr 3e-4 / wd 1e-4 / batch 8 / ≤ 25 epochs / early
stopping patience 5 on val mIoU / seed 20260929 / fresh initialisation / model selection = highest
MiniVal240 mIoU.

## 8. MiniVal240 metrics (section 12)

| Metric | B1 (frozen 6N) | **B2 (reproduced)** | **B3** | **B4** |
|---|---|---|---|---|
| mIoU | 0.241316 | **0.453127** | **0.429968** | **0.046637** |
| Dice | — | 0.576844 | 0.540155 | 0.074986 |
| Precision@0.5 | — | — | 0.692450 | 0.708565 |
| per relation mIoU (left/right/above/below) | — | — | 0.4219 / 0.3905 / 0.4581 / 0.4493 | 0.0464 / 0.0508 / 0.0455 / 0.0438 |
| largest-ref / smallest-ref mIoU | — | — | 0.4101 / 0.4498 | 0.0420 / 0.0512 |
| border-target mIoU (n=114) | — | — | 0.4132 | 0.0753 |
| tiny-target mIoU (n=4) | — | — | ≈0 | 0.00023 |
| first conv in-ch / parameters | 145 / 274,625 | 146 / 275,777 | **145 / 274,625** | **144 / 240,833** |
| peak VRAM / wall time | — | — | 0.642 GB / 116.2 s | 0.626 GB / 45.0 s |
| selected epoch (epochs run) | 11 (16) | 9 (14) | 14 (19) | 3 (8) |

B4's high precision (0.7086) together with a near-zero mIoU is a direct consequence of predicting
almost no positive pixels; it is reported as measured, without interpretation.

## 9. PairedVal20 (section 13)

| Variant | pass | mean own IoU | mean cross IoU | own − cross |
|---|---|---|---|---|
| **B2 (reproduced)** | **16/20** | 0.443346 | 0.002158 | **+0.441188** |
| **B3** | **14/20** | 0.398969 | 0.001773 | **+0.397196** |
| **B4** | 2/20 | 0.048955 | 0.000000 | +0.048955 |

## 10. Predeclared causal comparisons (section 14)

```text
delta_B3_B2 = 0.429968 - 0.453127 = -0.023159   (Dice -0.036689, paired -2, margin -0.043993)
delta_B3_B1 = 0.429968 - 0.241316 = +0.188652
delta_B3_B4 = 0.429968 - 0.046637 = +0.383331   (Dice +0.465169, paired +12, margin +0.348241)
```

| Criterion | Condition | Measured | Result |
|---|---|---|---|
| **14.1** direct reference-channel retention | `B3 ≥ B2 − 0.03` **and** B3 paired ≥ 14/20 **and** margin ≥ 0.10 | 0.429968 ≥ 0.423127 ✓ · 14/20 ✓ · +0.397196 ✓ | **PASS** |
| **14.2** visual contribution | `B3 − B4 ≥ 0.10` | **+0.383331** | **PASS** |
| **14.3** geometry-only confound | `B4 ≥ B3 − 0.05` **and** B4 paired ≥ 12/20 | 0.046637 < 0.379968 ✗ · 2/20 ✗ | **FAIL** |

## 11. Verdict (section 15, fixed priority order)

`evaluation/task6o_verdict.json` — **`FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`**:

1. `INVALID_EXPERIMENT` — not applicable (no leakage, no frozen-artifact mutation, no target-input
   leakage, no test access, no protocol violation; B2 reproduced bit-identically).
2. `FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE` — not applicable (B3 passed O1).
3. `GEOMETRY_ONLY_BENCHMARK_CONFOUND` — not applicable (14.3 fails).
4. `DIRECT_REFERENCE_CHANNEL_MATTERS` — not applicable (`B3 − B2 = −0.023159`, which is **not** below
   `−0.05`).
5. **`FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`** — B3 O1 passes, 14.1 passes, 14.2 passes and 14.3
   does not pass.

Measured context, reported without interpretation: dropping the direct reference channel costs
`0.023159` mIoU (still within the predeclared `0.03` slack) and 2 paired passes; removing the visual
feature costs `0.383331` mIoU and 12 paired passes, and B4 never becomes competitive on MiniVal240.

## 12. Interpretation boundary (section 16)

DSH reports measurements only. This document does **not** decide whether anything here is paper
novelty, whether predicted-reference grounding should be implemented, whether the field formula should
change, or whether GRCL should be added. No threshold, gate or comparison was altered.

## 13. Reproduce

```text
python scripts/task6o_evaluate.py --stage b2-reproduction   # Part A (frozen B2, once)
python scripts/task6o_train.py --stage o1                   # O1 (Overfit20, B3 gate)
python scripts/task6o_train.py --stage o2                   # O2 (MiniTrain1000 -> MiniVal240)
python scripts/task6o_evaluate.py --stage variants          # MiniVal240 + PairedVal20
python scripts/task6o_report.py                             # causal summary + verdict
```

Run in `.conda/buildreasonseg-proposal`. Checkpoints live under the gitignored
`artifacts/task6o/checkpoints/`; packs, features and caches stay under the gitignored
`artifacts/task6n/` tree.
