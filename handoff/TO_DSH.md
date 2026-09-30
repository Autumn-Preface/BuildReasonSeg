# TO_DSH — Task 6Y: Oracle-Reference Nearest Boundary Field Feasibility

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `93188905df246c2669c730ad1251a651ec22f106`
>
> Predecessor: Task 6X → `SAM2_REFINEMENT_NOT_HELPFUL`
>
> Research decision already made by ChatGPT:
> 1. Stop reference hardening here. Tasks 6P/6Q/6U/6V/6W/6X have characterized the support-module bottleneck sufficiently.
> 2. Freeze the current practical reference resolver for later end-to-end work as **U-C1 proposals + Task 6Q deterministic largest/smallest selector** (`YOLO26m-seg, imgsz=640, conf=0.05, max_det=300, no TTA, no tiling`).
> 3. Do NOT use ProposalSetRanker, ProposalQualityEstimator, family routing, or SAM2 proposal refinement in the primary reference resolver.
> 4. Task 6Y returns to the core spatial-reasoning method and tests the next relation family: **nearest**, whose canonical dataset semantics are based on `boundary_distance`.
> 5. Task 6Y is an **oracle-reference causal experiment**. It intentionally uses GT reference masks so reference errors do not contaminate nearest-field feasibility.
> 6. No nearest end-to-end claim is allowed after Task 6Y alone.
>
> DSH is an executor. Do not redesign the field, decoder, packs, loss, gates, or next task.

All user-facing DSH output must be Chinese.

---

## 0. DSH role

DSH MAY:
- implement the exact `NearestBoundaryField v0.1` below;
- reuse frozen SAM2 image features;
- freeze the exact nearest-only train/validation packs;
- train/evaluate Y-B0/Y-B1/Y-B2/Y-B3;
- solve ordinary implementation/runtime bugs without altering the experiment.

DSH MUST NOT:
- modify GeometricRelationField v0.2;
- modify Task 3B nearest semantics;
- use centroid distance instead of boundary distance;
- change `sigma_diag`;
- tune field parameters;
- change SAM2 or decoder architecture;
- use proposal/predicted references in the main experiment;
- add nearest proposal execution, L3, GRCL, ProgramHead training, test access, or Task 6Z.

If a prohibited change is required, STOP and report.

---

# PART A — Freeze reference subsystem status

## 1. Record the reference-hardening conclusion

Copy into `docs/task6y_oracle_nearest_boundary_field.md`:

Task 6X verdict: `SAM2_REFINEMENT_NOT_HELPFUL`.

On train-only U-Calib200:
- X-C0 U-C1 baseline mIoU = `0.4789276`
- X-C1 exact-box SAM2 single-mask = `0.4744`
- X-C2 exact-box multimask = `0.4284`
- X-C3 expanded-box multimask = `0.3741`

X-C0 won calibration, so 6X correctly stopped before RefVal/MiniVal/Paired evaluation.

Frozen practical reference resolver for future integration:
`U-C1 + deterministic Task 6Q area semantics`.

Known development metrics from Task 6U:
- RefVal reference mIoU ≈ `0.4289355`
- MiniVal answered target mIoU ≈ `0.3141364`
- strict target mIoU ≈ `0.3089008`
- PairedVal `11/20`
- own-cross margin ≈ `0.3209`

Known limitations remain: proposal coverage/extreme selection, especially smallest; tiny buildings; scene-disjoint support-module generalization.

Do not modify old artifacts.

---

# PART B — Canonical nearest semantics

## 2. Exact query scope

Task 6Y handles exactly:
- `largest_to_nearest`
- `smallest_to_nearest`

No directional programs in the main nearest experiment. No L3.

BuildSpatialReason v0.2 counts:
- train: largest_to_nearest 672; smallest_to_nearest 358
- val: largest_to_nearest 458; smallest_to_nearest 249

No test.

## 3. Frozen nearest definition

Read-only: `configs/spatial_relations_v1.yaml`.

Nearest semantics MUST remain:
- metric = `boundary_distance`
- NOT centroid distance
- `margin_px_floor = 2.0`
- `margin_diag_fraction = 0.005`
- `margin_mode = normalized_with_absolute_floor`
- nearest anchor and target eligibility unchanged: reject border-truncated, suspected-large-merge, tiny; require at least two valid components.

Canonical labels are frozen. Do not regenerate them.

---

# PART C — NearestBoundaryField v0.1

## 4. Purpose

The field means: pixels closer to the grounded reference-building boundary receive larger prior value.
It must not directly choose a building candidate.

Create: `buildreasonseg_mvp/nearest_boundary_field.py`.

Constants:
```text
sigma_diag = 0.05
eps = 1e-6
source_size = 512x512
decoder_field_size = 64x64
```

No learned parameters. No sweep.

## 5. Exact source-resolution field

Input: binary oracle reference mask `M_ref`, 512×512, non-empty.

Use `scipy.ndimage.distance_transform_edt`.
If SciPy is unavailable in the existing environment, STOP with `NEAREST_FIELD_DEPENDENCY_UNAVAILABLE`; do not install.

```text
R = M_ref > 0
D_px = distance_transform_edt(~R)
diag_px = sqrt(H^2 + W^2)
D_norm = D_px / diag_px
P_near_512 = exp(-D_norm / sigma_diag)
P_near_512[R] = 0
P_near_512 = clamp(P_near_512, 0, 1)
```

Do NOT use reference centroid, reference bbox distance, target mask/centroid, candidate proposals, or GT candidate ids.

## 6. Decoder-resolution field

```text
P_near_64 = interpolate(
  P_near_512[None,None], size=(64,64), mode="bilinear", align_corners=False
)
```
Clamp `[0,1]`.

For Y-B1 only:
```text
M_ref_64 = interpolate(M_ref_float[None,None], size=(64,64), mode="area")
```
Clamp `[0,1]`.

---

# PART D — Field semantic sanity

## 7. Parameter-free nearest ranking diagnostic

Before training, on frozen Y-MiniVal240:

For each record:
1. compute `P_near_512` from oracle reference;
2. consider every canonical native building other than the reference that satisfies frozen nearest target eligibility;
3. candidate score:
   `field_score(C) = max(P_near_512[p] for p in C)`.

Report:
- target top-1 / top-3 rate
- mean target score
- mean best non-target distractor score
- mean target-minus-distractor score
- by largest/smallest reference family
- mean eligible candidate count
- per-record Spearman correlation between `field_score(C)` and `-canonical boundary_distance(reference,C)`, then mean.

Write: `evaluation/task6y_field_sanity.json`.

Sanity gate:
- top-1 >= `0.85`
- top-3 >= `0.97`
- mean Spearman >= `0.90`

If any fail: STOP `NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH`.
Do not tune sigma.

---

# PART E — Freeze nearest packs

## 8. Common rules

Create deterministic packs from v0.2 train/val only. Seed `20260930`.
Write `evaluation/task6y_pack_manifest.json` with record ids and SHA256.
Local packs: `artifacts/task6y/packs/` (gitignored).

## 9. Y-Overfit20

Train only:
- 10 largest_to_nearest
- 10 smallest_to_nearest
- unique record ids
- at least 15 unique tiles if possible.

## 10. Y-MiniTrain1000

Train only, exactly:
- 650 largest_to_nearest
- 350 smallest_to_nearest

Deterministic stable record-id hash selection after eligibility checks.

## 11. Y-MiniVal240

Val only, exactly:
- 120 largest_to_nearest
- 120 smallest_to_nearest

## 12. Y-PairedVal

Pair requirements:
- same tile has both largest_to_nearest and smallest_to_nearest;
- different reference source feature ids;
- different target source feature ids;
- both canonical valid records.

Sort by stable hash.
Use first 20 if >=20 exist; otherwise all if >=12.
If fewer than 12: STOP `NEAREST_PAIRED_SET_INSUFFICIENT`.
Record `N_pair`.

No test.

---

# PART F — Frozen visual representation and decoder

## 13. Frozen SAM2 feature

Reuse exact Task 6N/6O path:
- SAM2.1 Hiera Base+
- image embedding V = 256×64×64
- frozen checkpoint/config
- no GT visual input
- no retraining.

## 14. Common visual projection

For Y-B0/Y-B1/Y-B2:
```text
Conv1x1(256→128)
GroupNorm(8,128)
GELU
```

## 15. Nearest embedding

One trainable embedding:
- vocab size 1
- dim 16
- semantic id `nearest`

Broadcast 64×64.

## 16. Common target trunk

```text
Conv3x3(in→128,padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128→64,padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64→1)
```

Upsample logits bilinearly to 512×512.
Loss exactly `BCEWithLogitsLoss + DiceLoss`.
No GRCL, ranking loss, auxiliary field loss, or candidate loss.

---

# PART G — Four variants

## 17. Y-B0 — visual baseline

Input:
`visual_128 + nearest_embed_16`
Fusion channels 144.
No ref mask, no field.

## 18. Y-B1 — direct reference-mask control

Input:
`visual_128 + M_ref_64 + nearest_embed_16`
Fusion channels 145.
No nearest field.

## 19. Y-B2 — nearest-boundary-field model

Primary hypothesis.
Input:
`visual_128 + P_near_64 + nearest_embed_16`
Fusion channels 145.
No direct reference-mask channel.

## 20. Y-B3 — geometry-only control

No SAM2 visual feature.

```text
field projection:
Conv1x1(1→128)
GroupNorm(8,128)
GELU
```

Then:
`field_128 + nearest_embed_16`
Fusion channels 144 → common trunk.

Report exact params. No extra parameter padding.

---

# PART H — Stage Y1: Overfit20

## 21. Training

Train all four variants separately from fresh init, same pack.

- AdamW
- lr `1e-3`
- weight_decay `1e-4`
- batch `4`
- max steps `1200`
- no scheduler
- no augmentation
- seed `20260930`
- same AMP/bfloat16 policy as 6N/6O
- evaluate every 100 steps.

Write `evaluation/task6y_overfit20.json`.

B2 gate:
- mIoU >= `0.85`
- Dice >= `0.90`

If fail: STOP `NEAREST_FIELD_NOT_LEARNABLE`.

---

# PART I — Stage Y2: MiniTrain1000 → MiniVal240

## 22. Training

Only if B2 overfit passes.
Fresh initialization for all four.

- AdamW
- lr `3e-4`
- weight_decay `1e-4`
- batch `8`
- max epochs `25`
- early stopping patience `5`
- selection metric = MiniVal240 mIoU
- seed `20260930`
- no scheduler
- no augmentation
- same AMP.

No test.
Write `evaluation/task6y_training.json`.

---

# PART J — Evaluation

## 23. MiniVal240

For all variants report:
- mIoU
- Dice
- Pr@0.5
- largest-reference mIoU
- smallest-reference mIoU
- border-target mIoU
- tiny-target mIoU if present
- target-area quartiles
- canonical target boundary-distance quartiles
- params
- best epoch
- wall time
- peak VRAM.

Write `evaluation/task6y_mini_val.json`.

## 24. PairedVal

For each pair, run largest-reference and smallest-reference query independently.
Pair passes iff both predictions have higher IoU with their own target than with the other pair target.

Report:
- pass / N_pair
- pass rate
- mean own IoU
- mean cross IoU
- own-cross margin.

Write `evaluation/task6y_paired_val.json`.

---

# PART K — Causal criteria

## 25. Feasibility criteria

`NEAREST_BOUNDARY_FIELD_FEASIBLE` requires ALL:

1. field semantic sanity passes;
2. B2 overfit passes;
3. B2-B0 MiniVal mIoU >= `+0.08`;
4. B2-B1 MiniVal mIoU >= `+0.05`;
5. B2-B3 MiniVal mIoU >= `+0.10`;
6. B2 MiniVal mIoU >= `0.35`;
7. B2 paired pass rate >= `0.70`;
8. B2 own-cross margin >= `0.15`;
9. no target GT model input.

Interpretation:
- B0→B1 = value of raw reference localization;
- B1→B2 = value of explicit boundary-proximity geometry;
- B3 vs B2 = whether visual evidence is materially necessary.

Do not invent another interpretation.

---

# PART L — Verdict

## 26. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `NEAREST_FIELD_DEPENDENCY_UNAVAILABLE`
3. `NEAREST_PAIRED_SET_INSUFFICIENT`
4. `NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH`
5. `NEAREST_FIELD_NOT_LEARNABLE`
6. `NEAREST_FIELD_GEOMETRY_ONLY_CONFOUND`
   - B2 learns/generalizes, but B2-B3 <0.10 and B3 paired pass rate >=0.60.
7. `NEAREST_FIELD_NO_MEANINGFUL_GAIN`
   - learnability passes but B2-B0 <0.08 OR B2-B1 <0.05 OR B2 mIoU <0.35.
8. `NEAREST_FIELD_COUNTERFACTUAL_WEAK`
   - mask gains pass but paired rate <0.70 OR margin <0.15.
9. `NEAREST_BOUNDARY_FIELD_FEASIBLE`
   - all section-25 criteria pass.

No other verdict.

---

# PART M — Interpretation boundary

DSH reports measurements only.

Do NOT:
- claim global novelty;
- claim end-to-end nearest capability;
- change sigma;
- add learned distance transform;
- add predicted reference;
- combine directional and nearest fields;
- start L3;
- change reference resolver.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 6Y 的 nearest boundary field 因果结果决定下一步，不自行进行 predicted-reference nearest 集成、direction+nearest 场组合或 L3 训练。`

---

# PART N — Required artifacts

Create at minimum:

```text
buildreasonseg_mvp/nearest_boundary_field.py
buildreasonseg_mvp/task6y_nearest_decoder.py

evaluation/task6y_pack_manifest.json
evaluation/task6y_field_sanity.json
evaluation/task6y_overfit20.json
evaluation/task6y_training.json
evaluation/task6y_mini_val.json
evaluation/task6y_paired_val.json
evaluation/task6y_verdict.json

docs/task6y_oracle_nearest_boundary_field.md

scripts/task6y_freeze_packs.py
scripts/task6y_field_sanity.py
scripts/task6y_train.py
scripts/task6y_evaluate.py
scripts/task6y_report.py
```

If a STOP gate fires, create only applicable completed artifacts + verdict/docs/handoff.

Caches/checkpoints under `artifacts/task6y/` and `artifacts/checkpoints/task6y/`, gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

---

# PART O — Required tests

Task 6X ended at `965 passed, 1 skipped`.

Add tests covering at least:

1. Task 6X artifacts unchanged
2. reference subsystem status freezes U-C1 deterministic
3. exactly two nearest program ids
4. no directional/L3 records in packs
5. no test split
6. nearest metric remains boundary_distance
7. nearest margin config unchanged
8. nearest eligibility unchanged
9. sigma_diag exactly 0.05
10. scipy EDT uses inverse binary reference mask
11. field zero inside reference
12. field in [0,1]
13. synthetic monotonic decay with distance
14. no reference centroid
15. no bbox distance
16. no target input to field
17. bilinear field 512→64
18. direct ref area resize exact
19. field-sanity candidate score = max field inside candidate
20. field sanity only nearest-eligible non-reference candidates
21. Overfit20 exact family counts
22. MiniTrain1000 exact 650/350
23. MiniVal240 exact 120/120
24. paired same tile/different refs/different targets
25. pack hashes recorded
26. frozen SAM2 path unchanged
27. Y-B0 inputs exact
28. Y-B1 inputs exact
29. Y-B2 inputs exact
30. Y-B3 has no visual
31. B3 field projection exact 1→128
32. nearest embedding vocab1/dim16
33. common trunk exact
34. BCE+Dice only
35. no GRCL
36. no ranking/aux loss
37. same training schedule across variants
38. GT target only label/eval
39. oracle reference explicitly recorded
40. no proposal/predicted reference in main experiment
41. no ProgramHead training
42. no deterministic nearest executor replacing dense target segmentation
43. no L3
44. no new dataset/download/install/GUI
45. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART P — Git/storage

Do not commit model weights/checkpoints, feature caches, source imagery/vectors, large local packs, or `.conda`.

Commit nearest field/decoder code, small manifests/evaluation JSON, scripts, tests, docs and handoff.

Suggested commits:
1. `feat: add nearest boundary relation field`
2. `eval: test nearest field causal segmentation gains`
3. optional docs/handoff commit

---

# PART Q — DSH model policy

Default: **DeepSeek V4.1 Flash + High**.
Use Max only for a genuine data/runtime/cross-module bug.
No installs or downloads.

---

# PART R — STOP

After Task 6Y:
- commit
- push
- handoff
- STOP

Do NOT alter the frozen reference resolver, add predicted-reference nearest integration, combine directional+nearest fields, start L3, add GRCL, access test, or build GUI.

Wait for ChatGPT audit.
