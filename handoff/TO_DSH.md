# TO_DSH — Task 6Z: Oracle-Reference L3 Direction × Nearest Composition Feasibility

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `24be95494a36e1a21d1f3566590c06ef118c7f28`
>
> Predecessor: Task 6Y → `NEAREST_FIELD_NO_MEANINGFUL_GAIN`
>
> Research decision already made by ChatGPT:
>
> 1. Accept Task 6Y as a valid negative/partial causal result: the nearest boundary field is semantically exact and adds clear signal over visual-only and geometry-only baselines, but nearest-only dense segmentation did not meet the predeclared absolute/paired gates.
> 2. Do NOT claim standalone nearest capability from Task 6Y.
> 3. Do NOT tune `sigma_diag`, change the nearest field, or redesign the decoder in Task 6Z.
> 4. Task 6Z tests the project-relevant next question directly: **Can the already-validated directional field and the partial nearest boundary field be composed to solve the canonical L3 program `largest → direction → nearest`?**
> 5. This is an **oracle-reference causal experiment**. It uses the canonical GT largest reference so reference proposal errors do not contaminate the composition question.
> 6. No predicted-reference L3/end-to-end claim is allowed after Task 6Z alone.
> 7. DSH is an executor. Do not redesign fields, packs, architecture, loss, gates, or the next task.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- reuse the frozen GeometricRelationField v0.2;
- reuse the frozen NearestBoundaryField v0.1;
- freeze the exact L3 train/validation packs specified here;
- implement the exact six ablations Z-B0 ... Z-B5;
- train/evaluate those variants;
- solve ordinary implementation/runtime bugs without changing the experiment.

DSH MUST NOT:
- alter directional field formulas/constants;
- alter nearest field formulas/constants;
- tune `sigma_diag`;
- change the Task 3B relation semantics;
- use predicted/proposal references in the main experiment;
- retrain ProgramHead;
- change SAM2;
- introduce attention/Transformer/GNN;
- add a new loss;
- add GRCL;
- use candidate masks as model inputs;
- use test split;
- start predicted-reference L3 integration;
- choose the next task.

If a prohibited change is required, STOP and report.

---

# PART A — Record Task 6Y findings exactly

## 1. Task 6Y status

Copy into `docs/task6z_oracle_l3_composition.md`:

Task 6Y verdict: `NEAREST_FIELD_NO_MEANINGFUL_GAIN`

Key measurements:
- nearest-field semantic sanity: target top-1 = 1.0000, target top-3 = 1.0000, mean Spearman vs `-boundary_distance` = 1.0000;
- Y-B2 Overfit20: mIoU = 0.9663, Dice = 0.9827;
- MiniVal240: B0 visual = 0.1837, B1 visual + direct reference mask = 0.2640, B2 visual + nearest field = 0.2864, B3 nearest field only = 0.0468;
- deltas: B2-B0 = +0.1027, B2-B1 = +0.0224, B2-B3 = +0.2396;
- paired: B0 0/20, B1 8/20, B2 6/20, B3 6/20; B2 own-cross margin = +0.1828.

Interpretation boundary:
- nearest field carries valid spatial information;
- standalone nearest target segmentation is not demonstrated as sufficiently strong;
- Task 6Z asks whether **directional gating reduces the ambiguity enough for L3 composition**.

Do not modify Task 6Y artifacts.

---

# PART B — Exact L3 scope

## 2. Exactly four canonical L3 programs

Task 6Z handles only:
- `largest_to_left_of_to_nearest`
- `largest_to_right_of_to_nearest`
- `largest_to_above_to_nearest`
- `largest_to_below_to_nearest`

No L1. No L2. No `smallest_to_*_to_nearest` because those programs are not part of the frozen v0.2 vocabulary. No test.

BuildSpatialReason v0.2 availability:

Train:
- above = 338
- below = 336
- left = 323
- right = 347

Val:
- above = 224
- below = 213
- left = 250
- right = 249

## 3. Canonical program semantics

For every L3 record:

```text
Step 1:
largest eligible building → oracle reference

Step 2:
filter all buildings by one frozen directional relation
with respect to the reference

Step 3:
among the direction-filtered candidates,
select the one with minimum canonical boundary_distance
subject to the frozen nearest eligibility/margin policy
```

Do not regenerate labels.

Use the frozen:
- directional `alpha = 1.2`
- directional `tau = 0.04`
- nearest metric = `boundary_distance`
- nearest margin config from `configs/spatial_relations_v1.yaml`.

---

# PART C — Frozen field modules

## 4. Direction field

Read-only: `buildreasonseg_mvp/geometric_relation_field_v02.py`

For each record:
- oracle largest reference mask;
- canonical direction relation.

Generate `P_dir_512` and `P_dir_64` using the existing v0.2 implementation. Do not copy/rewrite the formula.

## 5. Nearest field

Read-only: `buildreasonseg_mvp/nearest_boundary_field.py`

Use exactly:
- `sigma_diag = 0.05`
- EDT boundary-proximity semantics
- zero inside reference
- bilinear 512→64.

Generate `P_near_512` and `P_near_64`.

## 6. Deterministic composed field

At source resolution:

```text
P_prod_512 = clamp(P_dir_512 * P_near_512, 0, 1)
```

At decoder resolution:

```text
P_prod_64 = clamp(P_dir_64 * P_near_64, 0, 1)
```

No renormalization. No learned scalar. No temperature. No exponent. No threshold.

---

# PART D — Parameter-free composition sanity

## 7. Candidate sets

Before training, run on frozen Z-MiniVal240.

For every record:

### 7.1 All eligible non-reference candidates

Candidate building must:
- be a canonical native instance;
- not be the reference;
- satisfy frozen nearest target eligibility.

Score:

```text
score_all(C) = max(P_prod_512[p] for p in C)
```

Report target top-1, target top-3, mean target score, mean best distractor score, and target-minus-distractor.

### 7.2 Exact direction-valid subset

Use the frozen Task 3B directional predicate with the record's direction. Keep only candidates that are direction-valid with respect to the oracle reference.

Score with the same `P_prod_512`.

Report:
- target top-1
- target top-3
- mean Spearman between product-field score and `-boundary_distance` inside the direction-valid set.

Write `evaluation/task6z_composition_sanity.json`.

## 8. Sanity gate

Require on the exact direction-valid subset:
- target top-1 >= `0.95`
- target top-3 >= `0.99`
- mean Spearman >= `0.90`

If any fails, STOP with `L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH`.

The all-candidate metrics are diagnostic only and do not stop the task.

---

# PART E — Freeze L3 packs

## 9. General

Seed `20260930`.

Source: BuildSpatialReason v0.2 train/val only.

Write `evaluation/task6z_pack_manifest.json`.

Local packs: `artifacts/task6z/packs/` (gitignored). Record SHA256 for every pack.

## 10. Z-Overfit20

Train split. Exactly:
- 5 above
- 5 below
- 5 left
- 5 right

Requirements:
- unique record ids;
- at least 15 unique tiles if possible;
- canonical oracle reference/target available.

## 11. Z-MiniTrain1200

Train split. Exactly:
- 300 above
- 300 below
- 300 left
- 300 right

Stable-hash deterministic selection.

## 12. Z-MiniVal240

Val split. Exactly:
- 60 above
- 60 below
- 60 left
- 60 right

## 13. Z-PairedVal20

Build same-reference directional counterfactual pairs.

Each pair must satisfy:
- same tile;
- same largest reference source feature id;
- two different L3 direction programs;
- two different target source feature ids;
- both canonical valid records.

Sort deterministically by stable hash.

Use first 20 if >=20 exist. If fewer than 20 but >=12, use all. If fewer than 12, STOP with `L3_PAIRED_SET_INSUFFICIENT`.

Record actual `N_pair`. No test.

---

# PART F — Frozen visual representation

## 14. Visual features

Reuse exactly:
- frozen SAM2.1 Hiera Base+;
- `V = 256×64×64`;
- no image-backbone training;
- no GT visual input.

Reuse existing cache where possible.

Common visual projection for Z-B0...Z-B4:

```text
Conv1x1(256 → 128)
GroupNorm(8,128)
GELU
```

## 15. Direction embedding

Trainable:
- vocabulary size = 4
- ids exactly: left_of, right_of, above, below
- embedding dim = 16

Broadcast 64×64.

No separate trainable `nearest` embedding because nearest is constant for every Task 6Z program.

## 16. Common target trunk

```text
Conv3x3(in_channels → 128, padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128 → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Upsample logits bilinearly to 512×512.

Loss EXACTLY: `BCEWithLogitsLoss + DiceLoss`

No auxiliary loss. No ranking loss. No GRCL. No field supervision.

---

# PART G — Six exact variants

## 17. Z-B0 — semantic/visual baseline

Inputs:
```text
visual_128
direction_embed_16
```
Fusion = 144. No field.

## 18. Z-B1 — directional field only

Inputs:
```text
visual_128
P_dir_64
direction_embed_16
```
Fusion = 145.

## 19. Z-B2 — nearest field only

Inputs:
```text
visual_128
P_near_64
direction_embed_16
```
Fusion = 145.

Direction embedding remains present because the program differs by direction, but there is no explicit directional field.

## 20. Z-B3 — two-field learnable composition

PRIMARY HYPOTHESIS.

Inputs:
```text
visual_128
P_dir_64
P_near_64
direction_embed_16
```
Fusion = 146.

There is no deterministic multiplication before the decoder. The common convolutional target trunk must learn how to combine the two explicit fields.

## 21. Z-B4 — deterministic product-field comparator

Inputs:
```text
visual_128
P_prod_64
direction_embed_16
```
Fusion = 145.

## 22. Z-B5 — deterministic composition, geometry-only control

No SAM2 visual input.

Project `P_prod_64`:
```text
Conv1x1(1 → 128)
GroupNorm(8,128)
GELU
```

Then:
```text
field_128
direction_embed_16
```
Fusion = 144.

Purpose: test whether the composed field itself solves the target mask rather than guides visual segmentation.

Report exact parameter counts. No seventh variant.

---

# PART H — Stage Z1: Overfit20

## 23. Training

Train all six variants separately from fresh initialization.

Exact settings:
- AdamW
- lr = `1e-3`
- weight_decay = `1e-4`
- batch = `4`
- max steps = `1200`
- no scheduler
- no augmentation
- seed = `20260930`
- same bfloat16 AMP policy as Tasks 6N/6Y
- evaluate every 100 steps.

Record best/final mIoU, Dice, Pr@0.5, by direction, params, wall time, peak VRAM.

Write `evaluation/task6z_overfit20.json`.

## 24. Primary learnability gate

Z-B3 must reach:
- mIoU >= `0.85`
- Dice >= `0.90`

If Z-B3 fails, STOP with `L3_LEARNED_COMPOSITION_NOT_LEARNABLE`.

Z-B4/B5 have no stop gate.

---

# PART I — Stage Z2: MiniTrain1200 → MiniVal240

## 25. Training

Only if Z-B3 overfit passes.

Train all six from fresh initialization.

Exact settings:
- AdamW
- lr = `3e-4`
- weight_decay = `1e-4`
- batch = `8`
- max epochs = `25`
- early stopping patience = `5`
- model selection = MiniVal240 mIoU
- seed = `20260930`
- no scheduler
- no augmentation
- same AMP.

No test.

Write `evaluation/task6z_training.json`.

---

# PART J — Evaluation

## 26. MiniVal240

For Z-B0...Z-B5 report:
- mIoU
- Dice
- Pr@0.5
- per-direction mIoU
- target area quartiles
- canonical target boundary-distance quartiles
- parameter count
- best epoch
- wall time
- peak VRAM.

Write `evaluation/task6z_mini_val.json`.

## 27. PairedVal

For every pair:
- run both direction programs independently;
- pair passes iff both predictions prefer their own GT target over the other member's target.

For every variant report:
- pass / N_pair
- pass rate
- mean own IoU
- mean cross IoU
- own-cross margin.

Write `evaluation/task6z_paired_val.json`.

---

# PART K — Predeclared causal criteria

## 28. Learned two-field composition criteria

Z-B3 mask criteria pass only if ALL:
1. MiniVal B3 mIoU >= `0.35`
2. B3-B0 >= `+0.10`
3. B3-B1 >= `+0.05` — nearest information adds value beyond the directional field
4. B3-B2 >= `+0.05` — directional information adds value beyond the nearest field
5. B3-B5 >= `+0.10` — visual evidence remains materially necessary

Counterfactual criteria:
6. B3 paired pass rate >= `0.70`
7. B3 own-cross margin >= `0.15`

## 29. Deterministic product comparator criteria

For Z-B4 define the analogous pass:
- B4 mIoU >= `0.35`
- B4-B0 >= `+0.10`
- B4-B1 >= `+0.05`
- B4-B2 >= `+0.05`
- B4-B5 >= `+0.10`
- B4 paired pass rate >= `0.70`
- B4 own-cross margin >= `0.15`

Report `deterministic_product_pass = true/false`.

## 30. Learned-vs-product comparison

Report:
```text
delta_learned_vs_product = B3_mIoU - B4_mIoU
```

Interpretation labels only:
- `learned_not_worse` iff B3 >= B4 - 0.03
- `product_materially_stronger` iff B4 >= B3 + 0.05

These labels do not independently change the scientific validity of the experiment.

---

# PART L — Verdict

## 31. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `L3_PAIRED_SET_INSUFFICIENT`
3. `L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH`
4. `L3_LEARNED_COMPOSITION_NOT_LEARNABLE`
5. `L3_GEOMETRY_ONLY_CONFOUND`
   - B3-B5 < 0.10 AND B5 paired pass rate >= 0.60.
6. `L3_DETERMINISTIC_COMPOSITION_ONLY`
   - learned B3 section-28 criteria fail,
   - but deterministic product B4 section-29 criteria all pass.
7. `L3_COMPOSITION_NO_MEANINGFUL_GAIN`
   - B3 mask criteria 1-5 fail,
   - and B4 section-29 does not pass.
8. `L3_COMPOSITION_COUNTERFACTUAL_WEAK`
   - B3 mask criteria 1-5 pass,
   - but B3 paired criterion 6 or 7 fails.
9. `L3_LEARNED_COMPOSITION_FEASIBLE`
   - all B3 section-28 criteria pass.

No other verdict.

If B3 passes but deterministic product is materially stronger, verdict remains `L3_LEARNED_COMPOSITION_FEASIBLE` and `product_materially_stronger` must be reported for ChatGPT.

---

# PART M — Interpretation boundary

DSH may report measurements only.

Do NOT:
- claim global novelty;
- claim final L3 end-to-end capability;
- modify the reference subsystem;
- tune field constants;
- add attention/global competition;
- add predicted reference;
- retrain ProgramHead;
- change target loss;
- choose whether learned or deterministic composition enters the final model.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 6Z 的 L3 direction×nearest composition 因果结果决定下一步，不自行进行 predicted-reference L3 集成、attention/global competition 改造或正式全量训练。`

---

# PART N — Required artifacts

Create at minimum:

```text
buildreasonseg_mvp/task6z_l3_decoder.py
buildreasonseg_mvp/task6z_field_composition.py

evaluation/task6z_pack_manifest.json
evaluation/task6z_composition_sanity.json
evaluation/task6z_overfit20.json
evaluation/task6z_training.json
evaluation/task6z_mini_val.json
evaluation/task6z_paired_val.json
evaluation/task6z_verdict.json

docs/task6z_oracle_l3_composition.md

scripts/task6z_freeze_packs.py
scripts/task6z_composition_sanity.py
scripts/task6z_train.py
scripts/task6z_evaluate.py
scripts/task6z_report.py
```

If a STOP gate fires, create only applicable completed artifacts plus verdict/docs/handoff.

Caches/checkpoints:
```text
artifacts/task6z/
artifacts/checkpoints/task6z/
```
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

---

# PART O — Required tests

Task 6Y ended at `1011 passed, 1 skipped`.

Add tests for at least:
1. Task 6Y artifacts unchanged
2. exactly four L3 program ids
3. no L1/L2/test record in Task 6Z packs
4. no nonexistent smallest-L3 program introduced
5. canonical L3 operation order remains extreme → direction filter → nearest
6. direction alpha/tau unchanged
7. nearest boundary-distance semantics unchanged
8. nearest sigma unchanged
9. GeometricRelationField v0.2 unchanged
10. NearestBoundaryField v0.1 unchanged
11. P_prod source formula exact multiplication
12. P_prod decoder formula exact multiplication
13. no renormalization/temperature/power
14. composition sanity score is max field in candidate mask
15. sanity exact direction-valid subset uses frozen relation engine
16. Overfit20 exact 5/5/5/5
17. MiniTrain1200 exact 300/300/300/300
18. MiniVal240 exact 60/60/60/60
19. PairedVal same tile
20. PairedVal same reference id
21. PairedVal different direction
22. PairedVal different target id
23. pack hashes recorded
24. no test split
25. frozen SAM2 visual path unchanged
26. Z-B0 exact inputs
27. Z-B1 exact inputs
28. Z-B2 exact inputs
29. Z-B3 exact inputs
30. Z-B4 exact inputs
31. Z-B5 no visual input
32. Z-B5 projection exactly 1→128
33. exactly six variants
34. direction embedding vocab exactly 4
35. direction embedding dim 16
36. no separate learned nearest embedding
37. common target trunk exact
38. BCE+Dice only
39. no GRCL
40. no auxiliary/ranking loss
41. same schedule across variants
42. oracle reference explicitly recorded
43. GT target only label/evaluation
44. no candidate mask model input
45. no predicted/proposal reference in main experiment
46. no ProgramHead training
47. no attention/Transformer/GNN
48. no new dataset/download/install/GUI
49. previous passing suite preserved.

Run `python -m pytest tests/ -q`.
Do not reduce previous passing tests.

---

# PART P — Git/storage

Do not commit:
- model checkpoints;
- SAM2/YOLO/parser/B3 weights;
- feature caches;
- source imagery/vectors;
- large local pack JSONs;
- `.conda`.

Commit:
- L3 composition/decoder code;
- small manifests/evaluation JSON;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: add L3 direction-nearest field composition`
2. `eval: test multi-hop field composition causally`
3. optional docs/handoff commit

---

# PART Q — DSH model policy

Default: **DeepSeek V4.1 Flash + High**

Use Max only for a genuine runtime/data/cross-module bug.
No installs or downloads.

---

# PART R — STOP

After Task 6Z:
- commit;
- push;
- handoff;
- STOP.

Do NOT:
- integrate predicted reference;
- add attention/global context;
- change field formulas;
- start full-dataset training;
- access test;
- build GUI.

Wait for ChatGPT audit.
