# Task 6Z — Oracle-Reference L3 Direction × Nearest Composition Feasibility

> Task: `handoff/TO_DSH.md` (Task 6Z) · Base commit: `24be954` · Predecessor: Task 6Y →
> `NEAREST_FIELD_NO_MEANINGFUL_GAIN`
> **Verdict: `L3_COMPOSITION_NO_MEANINGFUL_GAIN`** · Composition sanity **passed** · Z-B3 overfit **passed**
> Tests: `tests/test_task6z_l3_composition.py` · Evidence: `evaluation/task6z_*.json`

Task 6Z asks whether the already-validated directional field and the partial nearest boundary field can be
**composed** to solve the canonical L3 program `largest → direction → nearest`. It is an oracle-reference
causal experiment (the canonical GT largest reference is used so reference errors do not contaminate the
composition question), and no predicted-reference L3 or end-to-end claim is made. Field constants, both
frozen field modules, SAM2, the reference subsystem and all historical artifacts are untouched.

## 1. Recorded Task 6Y result (Task 6Y artifacts not mutated)

Verdict `NEAREST_FIELD_NO_MEANINGFUL_GAIN`. Nearest-field semantic sanity: target top-1 **1.0000**, target
top-3 **1.0000**, mean Spearman vs `−boundary_distance` **1.0000**. Y-B2 Overfit20 mIoU **0.9663** / Dice
**0.9827**. MiniVal240: B0 visual 0.1837, B1 visual + direct reference mask 0.2640, B2 visual + nearest
field 0.2864, B3 nearest field only 0.0468; deltas B2−B0 **+0.1027**, B2−B1 **+0.0224**, B2−B3
**+0.2396**; paired B0 0/20, B1 8/20, B2 6/20, B3 6/20 with B2 own-cross margin **+0.1828**.

Interpretation boundary carried into Task 6Z: the nearest field carries valid spatial information;
standalone nearest target segmentation is not demonstrated as sufficiently strong; Task 6Z asks whether
**directional gating reduces the ambiguity enough for L3 composition**.

## 2. Exact L3 scope and frozen semantics

Only the four canonical programs `largest_to_{left_of,right_of,above,below}_to_nearest` (v0.2 availability
matches the task file exactly: train above 338 / below 336 / left 323 / right 347; val above 224 / below
213 / left 250 / right 249). No L1, no L2, no `smallest_to_*_to_nearest` (it does not exist in the frozen
vocabulary), no test.

Canonical semantics, not regenerated: step 1 `argmax_area` (largest eligible building = oracle reference) →
step 2 `filter_relation` with the frozen directional predicate (`alpha = 1.2`, `tau = 0.04`) → step 3
`argmin_boundary_distance` subject to the frozen nearest eligibility/margin policy
(`boundary_distance`, `margin_px_floor 2.0`, `margin_diag_fraction 0.005`).

## 3. Deterministic composed field

`buildreasonseg_mvp/task6z_field_composition.py` — the **exact clamped product**, with no renormalization,
no learned scalar, no temperature, no exponent and no threshold:

```text
P_prod_512 = clamp(P_dir_512 * P_near_512, 0, 1)
P_prod_64  = clamp(P_dir_64  * P_near_64,  0, 1)
```

`P_dir` comes from the frozen `geometric_relation_field_v02` and `P_near` from the frozen Task 6Y
`nearest_boundary_field` (`sigma_diag = 0.05`); neither formula is recomputed or rewritten.

## 4. Parameter-free composition sanity (Part D)

`evaluation/task6z_composition_sanity.json` on the frozen Z-MiniVal240 (240 records, 60 per direction):

| Candidate set | Records | top-1 | top-3 | mean target | mean best distractor | mean target−distractor | mean candidates | mean Spearman |
|---|---|---|---|---|---|---|---|---|
| all eligible non-reference (diagnostic) | 240 | 0.9042 | 1.0000 | 0.2985 | 0.1051 | +0.2070 | 7.21 | — |
| **exact direction-valid subset (gate)** | 240 | **1.0000** | **1.0000** | 0.2985 | 0.1058 | +0.2155 | 2.97 | **0.9983** |
| canonical step-2 list (diagnostic) | 240 | 0.9958 | 1.0000 | 0.2985 | 0.1006 | +0.2207 | 4.18 | 0.9988 |

Gate (direction-valid subset): top-1 ≥ 0.95 ✓, top-3 ≥ 0.99 ✓, mean Spearman ≥ 0.90 ✓ → **`sanity_passed`**,
constants not tuned. The exact direction-valid subset is produced by the frozen Task 3B predicate
(`relations.evaluate_direction`); the engine recomputation is a **strict subset** of the record's canonical
step-2 list in 125/240 records and agrees exactly in 115/240 (never a superset), so the gate is measured on
the stricter set, and the generator-consistent canonical list is reported alongside (top-1 0.9958).

## 5. Frozen L3 packs (seed 20260930)

`evaluation/task6z_pack_manifest.json`:

| Pack | Records | Split | Unique tiles |
|---|---|---|---|
| Z-Overfit20 | 20 (**5 above + 5 below + 5 left + 5 right**) | train | **20** |
| Z-MiniTrain1200 | 1200 (**300 per direction**) | train | 826 |
| Z-MiniVal240 | 240 (**60 per direction**) | val | 219 |
| Z-PairedVal20 | **N_pair 20** (40 records; same tile, same largest reference id, different directions, different targets) | val | 20 |

Only the four L3 programs appear; no L1/L2/test record and no smallest-L3 program.

## 6. Six variants, training and parameters

`buildreasonseg_mvp/task6z_l3_decoder.py` — frozen SAM2 `V = 256×64×64`; visual projection
`Conv1x1(256→128) + GroupNorm(8,128) + GELU`; direction embedding vocab exactly 4
(`left_of, right_of, above, below`) dim 16 broadcast 64×64; **no** separate trainable nearest embedding;
trunk `Conv3x3(in→128)+GN+GELU → Conv3x3(128→64)+GN+GELU → Conv1x1(64→1)`; loss exactly
`BCEWithLogitsLoss + DiceLoss`; no attention/Transformer/GNN, no GRCL, no auxiliary loss.

| Variant | Inputs | Fusion | Params | Overfit20 mIoU/Dice |
|---|---|---|---|---|
| Z-B0 | visual_128 + direction_embed_16 | 144 | 273,473 | 0.9572 / 0.9780 |
| Z-B1 | visual_128 + P_dir_64 + embed16 | 145 | 274,625 | 0.9629 / 0.9810 |
| Z-B2 | visual_128 + P_near_64 + embed16 | 145 | 274,625 | 0.9670 / 0.9832 |
| **Z-B3** | visual_128 + P_dir_64 + P_near_64 + embed16 | 146 | **275,777** | **0.9619 / 0.9804** |
| Z-B4 | visual_128 + P_prod_64 + embed16 | 145 | 274,625 | 0.9631 / 0.9811 |
| Z-B5 | field_128 (P_prod projected 1→128) + embed16, **no visual** | 144 | 240,833 | 0.2860 / 0.3686 |

Z1 (lr 1e-3, batch 4, 1200 steps, eval every 100) → **Z-B3 gate passed** (`L3_OVERFIT_PASS`), so Z2 ran
(lr 3e-4, wd 1e-4, batch 8, ≤25 epochs, patience 5, selection by MiniVal240 mIoU, seed 20260930, bf16 AMP).

## 7. MiniVal240 and PairedVal

| Variant | mIoU | Dice | Pr@0.5 | above | below | left | right | best epoch | wall | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| Z-B0 | 0.1510 | 0.2338 | 0.1984 | 0.1502 | 0.1311 | 0.1587 | 0.1639 | 4 | 124.5 s | 0.894 GB |
| Z-B1 | 0.3011 | 0.4067 | 0.4785 | 0.2870 | 0.2562 | **0.3452** | 0.3161 | 10 | 52.0 s | 0.681 GB |
| Z-B2 | 0.1724 | 0.2461 | 0.3127 | 0.1374 | 0.1477 | 0.2010 | 0.2033 | 7 | 47.7 s | 0.681 GB |
| **Z-B3** | **0.3242** | **0.4390** | **0.5331** | 0.2806 | **0.3061** | **0.3777** | 0.3325 | 12 | 66.6 s | 0.682 GB |
| Z-B4 | 0.2788 | 0.3845 | 0.4604 | 0.2500 | 0.2630 | 0.2493 | **0.3530** | 9 | 53.7 s | 0.681 GB |
| Z-B5 | 0.0888 | 0.1429 | 0.1123 | 0.0955 | 0.0889 | 0.0673 | 0.1033 | 10 | 58.1 s | 0.666 GB |

Deltas: **B1−B0 +0.1501**, B2−B0 +0.0214, **B3−B0 +0.1732**, **B3−B1 +0.0231**, **B3−B2 +0.1519**,
**B3−B4 +0.0454**, **B3−B5 +0.2355**, B4−B5 +0.1901.

PairedVal (N_pair 20): Z-B0 9/20 (+0.0879), Z-B1 **16/20** (+0.2630), Z-B2 9/20 (+0.1583),
**Z-B3 15/20 (+0.3004)**, Z-B4 12/20 (+0.2702), Z-B5 6/20 (+0.0654).

B3 quartiles — target area (edges 1010.5 / 1449.5 / 2380.5 px): 0.3600 / 0.3725 / 0.3704 / 0.1940;
boundary distance (edges 20.91 / 51.15 / 99.87 px): 0.3392 / 0.2743 / 0.3177 / 0.3656. Border-target and
tiny-target mIoU are **not applicable** (0 records): the L3 pipeline's nearest eligibility excludes
border-truncated and tiny targets.

## 8. Criteria and verdict (Parts K-L)

Section 28 — learned two-field composition (Z-B3):

| # | Criterion | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | B3 MiniVal mIoU | ≥ 0.35 | **0.3242** | ✗ |
| 2 | B3 − B0 | ≥ +0.10 | **+0.1732** | ✓ |
| 3 | B3 − B1 (nearest adds beyond directional) | ≥ +0.05 | **+0.0231** | ✗ |
| 4 | B3 − B2 (directional adds beyond nearest) | ≥ +0.05 | **+0.1519** | ✓ |
| 5 | B3 − B5 (visual still necessary) | ≥ +0.10 | **+0.2355** | ✓ |
| 6 | B3 paired pass rate | ≥ 0.70 | **0.75** (15/20) | ✓ |
| 7 | B3 own-cross margin | ≥ 0.15 | **+0.3004** | ✓ |

Section 29 — deterministic product comparator (Z-B4): `deterministic_product_pass = **false**`
(B4 mIoU 0.2788 < 0.35, B4−B1 +0.0231 < 0.05, paired rate 0.60 < 0.70; the other four conditions pass).

Section 30 — `delta_learned_vs_product = B3 − B4 = +0.0454`; labels: **`learned_not_worse` = true**,
`product_materially_stronger` = **false**.

Section 31 priority: protocol clean, `N_pair 20 ≥ 12`, sanity passed, Z-B3 learnable; the geometry-only
confound does **not** apply (B3−B5 = +0.2355 ≥ 0.10 with B5 paired 0.30 < 0.60), and the deterministic
comparator does not pass → **`L3_COMPOSITION_NO_MEANINGFUL_GAIN`**.

Measured reading (reported, not prescribed): composition clearly helps relative to the visual baseline
(+0.1732), to the nearest-only variant (+0.1519) and to geometry-only (+0.2355), and the learned two-field
composition is the best variant overall (mIoU 0.3242, paired 15/20, margin +0.3004) and beats the
deterministic product comparator by +0.0454 while never being worse; but the two predeclared bars that fail
are precisely the absolute-quality bar (0.35) and the incremental-value bar for nearest *given* the
directional field (+0.0231 < +0.05) — i.e. once the directional field is present, the nearest field adds
little on this split.

## 9. Interpretation boundary (Part M)

DSH reports measurements only. No global novelty claim; no final L3 end-to-end capability claim; the
reference subsystem was not modified; field constants not tuned; no attention/global competition; no
predicted reference; ProgramHead not retrained; target loss unchanged; and DSH does not decide whether
learned or deterministic composition enters the final model. Final recommendation exactly:

`等待 ChatGPT 根据 Task 6Z 的 L3 direction×nearest composition 因果结果决定下一步，不自行进行 predicted-reference L3 集成、attention/global competition 改造或正式全量训练。`

## 10. Reproduce

```text
python scripts/task6z_freeze_packs.py         # Part E packs + manifest
python scripts/task6z_composition_sanity.py   # Part D sanity gate
python scripts/task6z_train.py --stage z1     # Part H Overfit20 (+ Z-B3 gate)
python scripts/task6z_train.py --stage z2     # Part I MiniTrain1200 -> MiniVal240
python scripts/task6z_evaluate.py             # Part J MiniVal240 + PairedVal
python scripts/task6z_report.py               # Parts K-L criteria + verdict
```

Packs and checkpoints live under the gitignored `artifacts/task6z/` and `artifacts/checkpoints/task6z/`; the
frozen SAM2 feature cache is reused from `artifacts/task6n/features`. Run the training/evaluation steps in
`.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the pack/sanity/report steps also run in
`.conda/buildreasonseg-mvp`.
