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

# FROM_DSH — Task 6Z Report: Oracle-Reference L3 Direction × Nearest Composition Feasibility

_This file holds the Task 6Z report. The Task 6Y report is preserved in git history at commit `24be954`;
Task 6X at `9318890`; Task 6W at `3de2142`; Task 6V at `3e185fb`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6z_oracle_l3_composition.md`.

## 1. Verdict

**`L3_COMPOSITION_NO_MEANINGFUL_GAIN`** — section 31 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, exactly four L3 programs, six variants, no L1/L2/test
   record, no predicted reference, constants untuned.
2. `L3_PAIRED_SET_INSUFFICIENT` — no: **N_pair = 20** (≥ 12).
3. `L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH` — **no: the composition sanity gate passed** (direction-valid
   top-1 1.0000, top-3 1.0000, mean Spearman 0.9983).
4. `L3_LEARNED_COMPOSITION_NOT_LEARNABLE` — no: Z-B3 Overfit20 mIoU **0.9619** / Dice **0.9804**.
5. `L3_GEOMETRY_ONLY_CONFOUND` — no: B3−B5 = **+0.2355** ≥ 0.10 while B5's paired pass rate is 0.30 < 0.60.
6. `L3_DETERMINISTIC_COMPOSITION_ONLY` — no: the deterministic product comparator Z-B4 does **not** pass
   its section-29 criteria (B4 mIoU 0.2788 < 0.35, B4−B1 +0.0231 < 0.05, paired 0.60 < 0.70).
7. **`L3_COMPOSITION_NO_MEANINGFUL_GAIN`** — Z-B3 mask criteria 1-5 fail (B3 mIoU **0.3242 < 0.35** and
   B3−B1 **+0.0231 < +0.05**) and B4 does not pass. ← **verdict**
8. `L3_COMPOSITION_COUNTERFACTUAL_WEAK` — not reached (mask criteria already fail).
9. `L3_LEARNED_COMPOSITION_FEASIBLE` — no (5 of 7 section-28 criteria pass).

## 2. Recorded Task 6Y result (Task 6Y artifacts not mutated)

`NEAREST_FIELD_NO_MEANINGFUL_GAIN`: nearest-field sanity top-1 1.0000 / top-3 1.0000 / Spearman 1.0000;
Y-B2 Overfit20 0.9663 / 0.9827; MiniVal240 B0 0.1837, B1 0.2640, B2 0.2864, B3 0.0468 with deltas
B2−B0 +0.1027, B2−B1 +0.0224, B2−B3 +0.2396; paired B0 0/20, B1 8/20, B2 6/20, B3 6/20 and B2 margin
+0.1828. Carried forward: the nearest field carries valid spatial information, standalone nearest target
segmentation is not sufficiently strong, and Task 6Z asks whether directional gating reduces the ambiguity
enough for L3 composition.

## 3. Frozen scope, semantics and the composed field

Only `largest_to_{left_of,right_of,above,below}_to_nearest` (v0.2 availability reproduced exactly: train
338/336/323/347, val 224/213/250/249). Canonical order preserved and not regenerated:
`argmax_area` → `filter_relation` (frozen directional predicate, `alpha 1.2`, `tau 0.04`) →
`argmin_boundary_distance` (frozen nearest eligibility/margin: `boundary_distance`, `margin_px_floor 2.0`,
`margin_diag_fraction 0.005`). The composed field is the exact clamped product with **no** renormalization,
learned scalar, temperature, exponent or threshold:

```text
P_prod_512 = clamp(P_dir_512 * P_near_512, 0, 1)      P_dir from frozen v0.2, P_near from frozen 6Y
P_prod_64  = clamp(P_dir_64  * P_near_64,  0, 1)      (sigma_diag 0.05, both untouched)
```

## 4. Parameter-free composition sanity (Part D)

`evaluation/task6z_composition_sanity.json` on the frozen Z-MiniVal240:

| Candidate set | Records | top-1 | top-3 | mean target | mean best distractor | Δ | mean candidates | mean Spearman |
|---|---|---|---|---|---|---|---|---|
| all eligible non-reference (diagnostic) | 240 | 0.9042 | 1.0000 | 0.2985 | 0.1051 | +0.2070 | 7.21 | — |
| **exact direction-valid subset (gate)** | 240 | **1.0000** | **1.0000** | 0.2985 | 0.1058 | +0.2155 | 2.97 | **0.9983** |
| canonical step-2 list (diagnostic) | 240 | 0.9958 | 1.0000 | 0.2985 | 0.1006 | +0.2207 | 4.18 | 0.9988 |

Gate passed (≥ 0.95 / ≥ 0.99 / ≥ 0.90), `constants_tuned = false`. The direction-valid subset comes from the
frozen `relations.evaluate_direction`; that recomputation is a strict **subset** of the record's canonical
step-2 list in 125/240 records and identical in 115/240 (never a superset), so the gate is measured on the
stricter set and the generator-consistent canonical-list numbers are reported alongside.

## 5. Frozen packs (seed 20260930)

`evaluation/task6z_pack_manifest.json`: Z-Overfit20 **20** (5 above / 5 below / 5 left / 5 right, 20 tiles),
Z-MiniTrain1200 **1200** (300 each, 826 tiles), Z-MiniVal240 **240** (60 each, 219 tiles), Z-PairedVal20
**N_pair 20** (same tile, same largest reference id, two different directions, two different targets).
Only the four L3 programs appear; no L1/L2/test record and no smallest-L3 program.

## 6. Six variants and training

| Variant | Inputs | Fusion | Params | Overfit20 | MiniVal mIoU | Dice | Pr@0.5 | Paired |
|---|---|---|---|---|---|---|---|---|
| Z-B0 | visual_128 + dir_embed16 | 144 | 273,473 | 0.9572/0.9780 | 0.1510 | 0.2338 | 0.1984 | 9/20 (+0.0879) |
| Z-B1 | + P_dir_64 | 145 | 274,625 | 0.9629/0.9810 | 0.3011 | 0.4067 | 0.4785 | **16/20** (+0.2630) |
| Z-B2 | + P_near_64 | 145 | 274,625 | 0.9670/0.9832 | 0.1724 | 0.2461 | 0.3127 | 9/20 (+0.1583) |
| **Z-B3** | + P_dir_64 + P_near_64 (learned) | 146 | **275,777** | **0.9619/0.9804** | **0.3242** | **0.4390** | **0.5331** | **15/20 (+0.3004)** |
| Z-B4 | + P_prod_64 (deterministic) | 145 | 274,625 | 0.9631/0.9811 | 0.2788 | 0.3845 | 0.4604 | 12/20 (+0.2702) |
| Z-B5 | field_128 (P_prod 1→128), no visual | 144 | 240,833 | 0.2860/0.3686 | 0.0888 | 0.1429 | 0.1123 | 6/20 (+0.0654) |

Per-direction MiniVal mIoU — B3 above 0.2806 / below 0.3061 / left 0.3777 / right 0.3325; B1 0.2870 /
0.2562 / 0.3452 / 0.3161; B4 0.2500 / 0.2630 / 0.2493 / 0.3530; B5 0.0955 / 0.0889 / 0.0673 / 0.1033.
Deltas: **B1−B0 +0.1501**, B2−B0 +0.0214, **B3−B0 +0.1732**, **B3−B1 +0.0231**, **B3−B2 +0.1519**,
**B3−B4 +0.0454**, **B3−B5 +0.2355**, B4−B5 +0.1901. Z1 lr 1e-3 / batch 4 / 1200 steps / eval every 100;
Z2 lr 3e-4 / wd 1e-4 / batch 8 / ≤25 epochs / patience 5 / selection by MiniVal240 mIoU; seed 20260930;
bfloat16 AMP; wall 47.7–124.5 s and peak VRAM 0.666–0.894 GB per variant. Loss exactly
`BCEWithLogitsLoss + DiceLoss`; no attention/Transformer/GNN, no GRCL, no auxiliary loss.

Border-target and tiny-target mIoU are **not applicable** (0 records): the L3 pipeline's nearest eligibility
excludes border-truncated and tiny targets. B3 target-area quartiles (edges 1010.5 / 1449.5 / 2380.5 px):
0.3600 / 0.3725 / 0.3704 / 0.1940; boundary-distance quartiles (edges 20.91 / 51.15 / 99.87 px):
0.3392 / 0.2743 / 0.3177 / 0.3656.

## 7. Criteria

Section 28 (Z-B3): mIoU 0.3242 < 0.35 ✗ · B3−B0 +0.1732 ≥ 0.10 ✓ · B3−B1 **+0.0231 < 0.05 ✗** ·
B3−B2 +0.1519 ≥ 0.05 ✓ · B3−B5 +0.2355 ≥ 0.10 ✓ · paired 0.75 ≥ 0.70 ✓ · margin +0.3004 ≥ 0.15 ✓ →
**5 of 7**.

Section 29 (Z-B4): `deterministic_product_pass = false` (four of seven conditions pass).
Section 30: `delta_learned_vs_product = +0.0454` → `learned_not_worse = true`,
`product_materially_stronger = **false**`.

Measured reading (reported, not prescribed): composition clearly helps over the visual baseline (+0.1732),
the nearest-only variant (+0.1519) and geometry-only (+0.2355); the learned two-field composition is the
best variant overall (mIoU 0.3242, paired 15/20, margin +0.3004) and beats the deterministic product
comparator by +0.0454 without ever being worse. The two predeclared bars that fail are the absolute-quality
bar (0.35) and the incremental-value bar for nearest *given* the directional field (+0.0231 < +0.05) — once
the directional field is present, the nearest field adds little on this split.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **1062 passed, 1 skipped** (Task 6Y ended at 1011 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6z_l3_composition.py` adds the 51 section-O checks.

Not committed: L3 pack JSONs, the twelve checkpoints, feature caches, source imagery/vectors, `.conda`.
Committed: composition/decoder code, small manifests and evaluation JSON, scripts, tests, docs, handoff.

Task 6Z downloaded nothing and installed nothing. Watt was **not needed** in Task 6Z: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no
proxy, host, certificate or TLS setting was read or modified.

## 9. Interpretation boundary

DSH reports measurements only. No global novelty claim; no final L3 end-to-end capability claim; the
reference subsystem was not modified; field constants not tuned; no attention/global competition; no
predicted reference; ProgramHead not retrained; target loss unchanged; DSH does not decide whether learned
or deterministic composition enters the final model.

## 10. Recommended next step (exact wording required by Part M)

等待 ChatGPT 根据 Task 6Z 的 L3 direction×nearest composition 因果结果决定下一步，不自行进行 predicted-reference L3 集成、attention/global competition 改造或正式全量训练。

## 11. STOP

Task 6Z stops here: no predicted-reference integration, no attention/global context, no field-formula
change, no full-dataset training, no test access, no GUI. Waiting for the ChatGPT audit.
