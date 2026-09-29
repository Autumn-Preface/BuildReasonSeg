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

# FROM_DSH — Task 6P Report: Differentiable Relation Field + Predicted-Reference Substitution

_This file holds the Task 6P report. The Task 6O report is preserved in git history at commit
`595e7bb`; Task 6N at `90f3735`; Task 6M.1 at `5e52d95`; Task 6M at `b9f49f8`._

Full design notes: `docs/task6p_differentiable_field_predicted_reference.md`.

## 1. Verdict

**`REFERENCE_HEAD_INSUFFICIENT`** — priority order of section 19 applied literally:

1. `INVALID_EXPERIMENT` — no.
2. `DIFFERENTIABLE_FIELD_INVALID` — no: the v0.2 equivalence/gradient audit passes.
3. `TASK6O_B3_REPRODUCTION_FAIL` — no: B3 reproduced with **absolute delta 0.0** on mIoU and Dice.
4. `REFERENCE_HEAD_NOT_LEARNABLE` — no: P1 Overfit20 gate passes (mIoU 0.969504, Dice 0.984388).
5. **`REFERENCE_HEAD_INSUFFICIENT`** — section 17 fails (mIoU 0.220176 < 0.35; median centroid error
   0.124060 > 0.05; p90 0.319618 > 0.12). ← **verdict**
6. `REFERENCE_ERROR_PROPAGATION_SEVERE` — not reached (section 18 is only evaluated behind section 17).
7. `PREDICTED_REFERENCE_CHAIN_FEASIBLE` — no.

## 2. Implementation correction recorded (v0.1 is not differentiable)

`buildreasonseg_mvp/geometric_relation_field.py` is **frozen and unchanged**. Recorded: v0.1 calls
`.detach()` on tensor masks and converts the centroid to a Python `float`, so it is numerically smooth
as a spatial function but **not differentiable with respect to the input reference mask**. That did not
invalidate Task 6N/6O (the reference mask was oracle and frozen and no gradient to the reference source
was required); predicted-reference training does require that gradient.

## 3. GeometricRelationField v0.2 (Part B)

`buildreasonseg_mvp/geometric_relation_field_v02.py`: identical formula (`alpha 1.2`, `tau 0.04`,
`s_axis = s_margin = 0.02`, `P_rel = clamp(sign·axis·margin, 0, 1)`, no learned parameter, soft
centroid with `eps = 1e-6`, pixel-centre normalized coordinates), accepts `(H,W)` / `(B,H,W)` /
`(B,1,H,W)`, and has **no** `.detach()`, Python-float centroid or NumPy in the forward path.

`evaluation/task6p_field_v02_audit.json` = **`FIELD_V02_VALID`**:

| Gate | Required | Measured |
|---|---|---|
| binary equivalence masks (16 per direction) | 64 | 16 / 16 / 16 / 16 |
| max abs error vs v0.1 | ≤ 1e-6 | **0.0** |
| mean abs error vs v0.1 | ≤ 1e-7 | **0.0** |
| gradient masks, non-binary, `requires_grad` | ≥ 8, all four directions | 8, four directions |
| gradient L1 to soft `M_ref` | > 1e-8, finite | **11,833.33**, finite |

## 4. Frozen B3 reproduction (Part C)

`evaluation/task6p_b3_reproduction.json` = **`B3_REPRODUCED`**: Task 6O B3 checkpoint verified by
SHA256 and **not retrained**, re-evaluated once on MiniVal240 with the oracle mask processed by v0.2 —
mIoU 0.4299680351479113 (stored 0.4299680351479113, delta **0.0**) and Dice 0.5401551056992948
(stored 0.5401551056992948, delta **0.0**); tolerance 1e-6.

## 5. Deduplicated reference packs (Part D)

`evaluation/task6p_reference_pack_manifest.json`, unique key
`(split, tile_id, reference_source_feature_id, reference_family)`:

* **RefTrainUnique 825** (from 1,000 MiniTrain1000 records; dedup 1.21×), 491 largest / 334 smallest;
* **RefValUnique 219** (from 240 MiniVal240 records), 110 largest / 109 smallest;
* **Reference Overfit20 20** unique train references, 10 largest + 10 smallest, distinct
  `(tile, source_feature_id)`, no duplicate mask.

The packs contain **no target identity and no relation id** (asserted by tests); the test split was
never read.

## 6. ReferenceMaskHead v0.1 and training (Parts E-F)

Architecture exactly as specified: frozen SAM2 feature (256×64×64) + family id (`largest`/`smallest`,
2×16 embedding) → `Conv1x1(256→128)+GN(8,128)+GELU` → concat 144 → `Conv3x3(144→128)+GN+GELU` →
`Conv3x3(128→64)+GN+GELU` → `Conv1x1(64→1)` → bilinear 512×512, loss `BCEWithLogitsLoss + DiceLoss`.
**273,441 parameters** (project 33,152 + family embedding 32 + trunk 240,257).

* **P1 (Overfit20)** — AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no augmentation
  / seed 20260929 / evaluate every 100 steps: best **mIoU 0.969504, Dice 0.984388** → gate **PASS**.
* **P2 (RefTrainUnique → RefValUnique)** — fresh init, AdamW / lr 3e-4 / wd 1e-4 / batch 8 / ≤ 30 epochs
  / patience 6 / seed 20260929; selected epoch **12**:

| Metric | Value |
|---|---|
| RefValUnique mIoU / Dice / Pr@0.5 | **0.220176** / 0.302794 / 0.359514 |
| largest-reference mIoU / Dice (n=110) | 0.279714 / 0.376918 |
| smallest-reference mIoU / Dice (n=109) | **0.160091** / 0.227989 |
| centroid error normalized mean / median / p90 | 0.150767 / **0.124060** / **0.319618** |
| median `pred_area / gt_area` | 1.019795 |
| border / tiny references in RefValUnique | 0 records / 0 records (all 219 non-border, non-tiny) |
| best epoch / wall time / peak VRAM | 12 / 47.8 s / 0.65 GB |

The head fits 20 references almost perfectly but generalizes to only 0.220176 on 219 unique validation
references, with median normalized centroid error 0.124060.

## 7. Predicted-reference chain (Parts G-H)

`evaluation/task6p_predicted_reference_target_val.json` (frozen head → soft reference → **v0.2** field →
frozen B3; **no oracle reference mask** on this path; GT reference only for reference metrics, GT target
only for scoring):

| Metric | Predicted reference | Oracle reference (frozen B3) |
|---|---|---|
| target mIoU | **0.240968** | 0.429968 |
| Dice | 0.316164 | 0.540155 |
| Precision@0.5 | 0.546548 | — |
| border target (n=114) | 0.231872 | 0.413228 |
| tiny target (n=4) | ≈0 | ≈0 |

`evaluation/task6p_predicted_reference_paired_val.json` (exact frozen PairedVal20; a pair sharing the
image and the reference source reuses the same predicted reference mask):

| Metric | Predicted reference | Oracle reference (frozen B3) |
|---|---|---|
| pass | **0/20** | 14/20 |
| mean own IoU | 0.064774 | 0.398969 |
| mean cross IoU | 0.061155 | 0.001773 |
| own − cross margin | **+0.003619** | +0.397196 |

`evaluation/task6p_field_propagation_diagnostics.json`: predicted vs oracle v0.2 field on MiniVal240 —
**MAE 0.120730**, **RMSE 0.274298**, **mean per-record Pearson 0.653476**, predicted-vs-oracle
reference centroid error mean 0.150767 / median 0.124060 / p90 0.319618, with per-family and
per-relation breakdowns. Diagnostic only; no threshold was tuned.

Net propagation, reported as measurement: target mIoU **−0.189000**, paired pass **−14**, own−cross
margin **−0.393577** relative to the frozen oracle-reference B3.

## 8. Gates (Parts I)

* **Section 17 reference-head adequacy — FAIL**: mIoU 0.220176 < 0.35; median centroid error 0.124060
  > 0.05; p90 0.319618 > 0.12.
* **Section 18 predicted-reference chain retention — FAIL**: section 17 does not pass; predicted target
  mIoU 0.240968 < 0.3009776246035379; PairedVal 0/20 < 10; margin +0.003619 < 0.20.

No gate or threshold was altered.

## 9. Tests, storage, git

`python -m pytest tests/ -q` → **658 passed, 1 skipped** (Task 6O ended at 627 passed / 1 skipped; no
prior passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism
check that needs the proposal env). `tests/test_task6p_differentiable_field_predicted_reference.py`
covers the 31 section-20 checks: Task 6N artifacts unchanged, Task 6O artifacts unchanged, v0.1 field
file unchanged, v0.2 binary numerical equivalence, v0.2 finite nonzero gradient to a soft `M_ref`,
v0.2 has no detach call in the field forward path, v0.2 has no Python-float centroid, no test split
access, only the 8 directional L2 programs, reference packs deduplicated by exact key, target source id
never a reference-head input, relation id never a reference-head input, reference head input is visual +
family only, family vocab exactly largest/smallest, frozen SAM2 unchanged, B3 checkpoint hash verified,
B3 oracle reproduction tolerance, B3 not retrained, predicted chain uses v0.2, predicted chain does not
use the oracle reference mask, GT reference only for reference evaluation, GT target only for target
scoring, same predicted reference reused for a pair, no `[REF]`, no GRCL/SCL, no nearest/L3, no graph
transformer, no proposal training, no 4B, no download/install/GUI, previous suite preserved.

Not committed: checkpoints, SAM2 weights, the frozen feature cache, source imagery/vector data,
`.conda`, large caches. Committed: code, small JSON artifacts, docs, tests, handoff.

Watt was **not needed** in Task 6P: this task downloaded nothing (no new weights, packages or datasets)
and installed nothing. The pre-existing Watt instance is transport-only, is not owned by this project,
and was left running, per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 10. Interpretation boundary (section 21)

DSH reports measurements only and does not claim final novelty or choose joint training, `[REF]`, GRCL,
nearest/L3, another reference architecture or a different field. No architecture decision is taken in
this task.

## 11. Recommended next step

等待 ChatGPT 根据 Task 6P 的 predicted-reference 误差传播结果决定 Task 6Q，不自行进行联合训练、MLLM 隐状态融合、nearest/L3 或 GRCL。

## 12. STOP

Task 6P stops here: no joint reference-target training, no ProgramHead/MLLM hidden-state fusion, no
nearest, no L3, no GRCL, no full-dataset training, no proposal optimisation, no GUI. Waiting for the
ChatGPT audit.
