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

# FROM_DSH — Task 6N Report: Oracle-Reference Geometric Relation Field Feasibility

_This file holds the Task 6N report. The Task 6M.1 report is preserved in git history at commit
`5e52d95`; the Task 6M report at `b9f49f8`._

Full design notes, the fixed literature-overlap statement and the Task 6M.1 metric erratum:
`docs/task6n_oracle_reference_geometric_relation_field.md`.

## 1. Verdict

**`GEOMETRIC_RELATION_FIELD_FEASIBLE`** — all five Task 6N section-17 criteria pass:

| Criterion | Required | Measured |
|---|---|---|
| B2 passes N1 (Overfit20) | mIoU ≥ 0.85, Dice ≥ 0.90 | 0.9280 / 0.9520 |
| MiniVal `B2 − B0` mIoU | ≥ 0.05 | **+0.2383** |
| MiniVal `B2 − B1` mIoU | ≥ 0.02 | **+0.2118** |
| B2 PairedVal | ≥ 14/20 | **16/20** |
| B2 own − cross IoU | ≥ 0.10 | **+0.4412** |
| no GT target in input | required | satisfied by construction (tested) |

Every measurement below is an **oracle-reference** measurement
(`reference_source = oracle_native_gt`); nothing in Task 6N is end-to-end inference and the test split
was never read.

## 2. Scope and protocol

The 8 directional L2 programs only (`largest|smallest_to_{left_of,right_of,above,below}`): `nearest`,
L1 extremes, L3 compositions and new relations are out of scope. For each sample `M_ref` is the
canonical native-vector GT mask of the reference building and `M_target` the canonical GT target used
**only** as label/evaluation GT; the relation id is the v0.2 canonical direction; the visual feature is
the frozen SAM2 image embedding. The GT target is never an input.

## 3. Frozen visual representation

The Task 6C.7 / 6I frozen path: `Sam2Encoder` over frozen **SAM2.1 Hiera Base+**
(`local_cache/models/sam2.1_hiera_base_plus.pt`, SHA256
`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`, config
`configs/sam2.1/sam2.1_hiera_b+.yaml`), giving **C × h × w = 256 × 64 × 64**. SAM2 was not retrained, its
checkpoint was not changed, no YOLO feature map is used and no GT building-union mask is an input. A
derived float32 feature memo lives in the gitignored `artifacts/task6n/features/` so all three variants
consume bit-identical inputs.

## 4. GeometricRelationField v0.1

Parameter-free and differentiable: `s_axis = s_margin = 0.02`, `alpha = 1.2`, `tau = 0.04`, softness
`= tau/2 = 0.02`, `P_rel = clamp(sign_score * axis_score * margin_score, 0, 1)` with the frozen Task 3B
sign convention (`left_of`: `cx_T < cx_R`; `right_of`: `cx_T > cx_R`; `above`: `cy_T < cy_R`; `below`:
`cy_T > cy_R`; image `y` increases downward). No distance transform, bounding box, candidate mask or
extra geometry channel was added, and no learned parameter exists in field generation.

## 5. Decoder and controlled variants

One common decoder family (1×1 → 128 + GroupNorm + GELU; 4 × 16 relation embedding; 3×3 → 128; 3×3 →
64; 1×1 → 1; bilinear upsample to 512 × 512 for loss/evaluation). No attention, transformer, graph
block or extra MLP. Loss exactly `BCEWithLogitsLoss + DiceLoss`.

| Variant | Inputs | First conv in-ch | Params | MiniVal240 mIoU | Dice | Pr@0.5 |
|---|---|---|---|---|---|---|
| N-B0 | visual + relation | 144 | 273,473 | 0.2148 | 0.3043 | 0.4191 |
| N-B1 | + `M_ref_down` | 145 | 274,625 | 0.2413 | 0.3276 | 0.4496 |
| **N-B2** | + `M_ref_down` + `P_rel` | **146** | **275,777** | **0.4531** | **0.5768** | **0.6640** |

Channels were **not** padded to equalise parameters; the counts above are exact.

## 6. Frozen packs

`evaluation/task6n_pack_manifest.json`, frozen before training, test untouched: **Overfit20** (20 train,
4 directions × both families, 4 same-image counterfactual pairs), **MiniTrain1000** (seeded
stratification over direction × family), **MiniVal240** (30 per program, all 8 programs),
**PairedVal20** (20 val pairs: same tile, same reference, different direction, different target). All
1,395 in-scope train and 1,008 in-scope val records were eligible — no tiny/border/visibility filter and
no difficulty deletion.

## 7. Stage N0 — field sanity (no training, diagnostic only)

`evaluation/task6n_field_sanity.json` on MiniVal240: **top-1 rate 1.0000**, **top-3 rate 1.0000**, mean
target score **0.8668**, mean other-building score 0.3520, mean best **true** distractor score
**0.1003**, mean margin **+0.7698**, mean 6.97 native instances per tile. The parameter-free field ranks
the true target first among all non-reference buildings in every one of the 240 records. No threshold
was tuned.

## 8. Stage N1 — Overfit20

`evaluation/task6n_overfit20.json`, AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no
augmentation / seed 20260929 / evaluation every 100 steps, identical for all variants:

| Variant | best mIoU | best Dice |
|---|---|---|
| N-B0 | 0.9431 | 0.9643 |
| N-B1 | 0.9332 | 0.9576 |
| **N-B2** | **0.9280** | **0.9520** |

**N1 gate PASS** → the task proceeded to N2. On 20 samples the field variant fits marginally behind the
baselines; that is reported as measured, without interpretation.

## 9. Stage N2 — MiniTrain1000 → MiniVal240

`evaluation/task6n_mini_val.json`, `evaluation/task6n_ablation_summary.json`: AdamW / lr 3e-4 / wd 1e-4
/ batch 8 / ≤ 25 epochs / early stopping patience 5 on val mIoU / seed 20260929 / fresh initialisation /
model selection = best MiniVal240 mIoU.

* per relation mIoU (left / right / above / below): B0 0.186 / 0.209 / 0.220 / 0.244 · B1 0.228 / 0.210
  / 0.266 / 0.262 · **B2 0.454 / 0.458 / 0.468 / 0.433**;
* largest-ref vs smallest-ref mIoU: B0 0.174 / 0.256 · B1 0.206 / 0.276 · **B2 0.440 / 0.467**;
* border-target mIoU (n = 114): B0 0.1935 · B1 0.2171 · **B2 0.4253**;
* tiny-target mIoU (n = 4): B0 ≈ 0 · B1 ≈ 0 · **B2 0.0216**;
* peak VRAM: 0.894 / 0.674 / 0.674 GB; wall time 174.8 / 50.9 / 44.8 s (B0 includes building the frozen
  feature cache); selected epoch 10 / 11 / **9** (epochs run 15 / 16 / 14).

## 10. PairedVal20

`evaluation/task6n_paired_val.json` — both relations of a pair run with the same image and the same
oracle reference; a pair passes only if **both** members prefer their own GT target over the paired
alternative by IoU:

| Variant | pass | mean own IoU | mean cross IoU | own − cross |
|---|---|---|---|---|
| N-B0 | 9/20 | 0.1456 | 0.0398 | +0.1058 |
| N-B1 | 7/20 | 0.1871 | 0.0506 | +0.1365 |
| **N-B2** | **16/20** | **0.4433** | **0.0022** | **+0.4412** |

## 11. Frozen assets, tests, storage

Task 6M.1 artifacts, `datasets/whu_native_vector/v1.0/`, `datasets/build_spatial_reason/v0.2/` and
`configs/spatial_relations_v1.yaml` are unchanged; Task 6N added only new files. `python -m pytest
tests/ -q` → **597 passed, 1 skipped** (Task 6M.1 ended at 569 passed / 1 skipped; no previously
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check
that needs the proposal env). `tests/test_task6n_oracle_relation_field.py` covers the 30
section-21 checks: Task 6M.1 artifacts unchanged, v0.2 unchanged, spatial config unchanged, no test
access, only the 8 allowed programs, target never an input, oracle reference explicitly marked,
direction signs, y-axis convention, `alpha = 1.2`, `tau = 0.04`, softness `= tau/2`, field bounded in
[0, 1], left/right and above/below mirror sanity, B0 has neither reference nor field, B1 has the
reference but not the field, B2 has both, same visual backbone/cache, same packs, same N1/N2 optimiser
and budgets, deterministic pack hashes, PairedVal same tile/reference with different targets, no `[REF]`
token, no GRCL/SCL, no graph transformer, no proposal training, no 4B, no GUI/download/install.

Not committed: SAM2 weights, model checkpoints, feature caches, `.conda`, source imagery/vectors, large
caches. Committed: code, config, small JSON artifacts, docs, tests, handoff.

Watt was **not needed** in Task 6N: this task downloaded nothing (no new weights, packages or datasets)
and installed nothing — the frozen SAM2 checkpoint was already present in `local_cache/models/`. The
pre-existing Watt instance is transport-only, is not owned by this project, and was left running, per
the ownership rule; no proxy, host, certificate or TLS setting was read or modified.

## 12. What is **not** claimed

No "first-ever" claim and no novelty claim for any of the seven listed overlap areas; no end-to-end
claim (the reference mask is an oracle throughout, so this is an upper-bound feasibility measurement);
no research interpretation by DSH.

## 13. Recommended next step

等待 ChatGPT 根据 Task 6N 测量结果决定 Task 6O，不自行选择后续算法。

## 14. STOP

Task 6N stops here: no predicted-reference integration, no `nearest`, no L3, no GRCL, no counterfactual
loss, no full-dataset training, no proposal optimisation, no GUI. Waiting for the ChatGPT audit.
