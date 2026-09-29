# Task 6N — Oracle-Reference Geometric Relation Field Feasibility

> Task: `handoff/TO_DSH.md` (Task 6N) · Base commit: `5e52d95` · Predecessor: Task 6M.1 →
> `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`
> **Verdict: `GEOMETRIC_RELATION_FIELD_FEASIBLE`** (all five section-17 criteria pass)
> Tests: `tests/test_task6n_oracle_relation_field.py` · Evidence: `evaluation/task6n_*.json`

This is the first execution task of the final innovation-architecture phase. It is an
**oracle-reference ablation**: every artifact carries `reference_source = oracle_native_gt`, and
nothing here is end-to-end inference. The research direction, architecture, losses, stages and gates
were fixed by ChatGPT; DSH implemented and measured them without redesigning anything.

## 1. The question actually tested

> With an oracle reference mask, does an explicit reference-conditioned geometric relation field
> improve dense target segmentation over equally controlled relation-aware baselines that do not
> receive that field?

The long-term method hypothesis under investigation (not tested end to end here) is that a
predicted/grounded reference building mask is converted into an explicit, differentiable,
relation-conditioned geometric prior and fused into a dense segmentation decoder, later constrained by
relation-level geometry supervision.

## 2. Literature-overlap statement (fixed research note)

1. SegLLM (ICLR 2025) already re-injects previous/reference masks and uses `[REF]`/`[SEG]` mask-aware
   decoding. Reference-mask conditioning or `[REF]` is **not** our novelty.
2. R²S (ICCV 2025) already uses a two-stage relevant-element → reasoning-prior paradigm in 3D.
   Generic two-stage reasoning priors are **not** our novelty.
3. Think2Seg-RS (ISPRS JPRS 2026) decouples LVLM reasoning from SAM geometry execution using
   structured geometric prompts. Semantic/geometry decoupling is **not** our novelty.
4. SegEarth-R2 (CVPR 2026) uses spatial-attention supervision for remote-sensing language-guided
   segmentation. Generic spatial supervision is **not** our novelty.
5. SRGFormer (Sensors 2026) decomposes target/relation/position semantics and performs relation-aware
   graph reasoning. Relation decomposition or graph reasoning alone is **not** our novelty.
6. GeoSelect (TGRS 2026) executes typed spatial programs and uses continuous geometric fields plus
   discrete operators over candidate sets. "Geometric field + spatial program" alone is **not** our
   novelty.
7. GeoRefer-Bench (2026) already provides executable geospatial relation queries and counterfactual
   pairs. Executable relation datasets/counterfactual evaluation alone are **not** our novelty.

The possible novelty under investigation is narrower: **a differentiable reference-conditioned
geometric relation field fused into a dense segmentation decoder and later supervised by explicit
reference–target geometry consistency.** No "first-ever" claim is made in Task 6N, and none of the
listed overlap areas is claimed as novelty.

## 3. Task 6M.1 metric erratum

`evaluation/task6m1_verdict.json` records `combined_best_mask_mAP50_95 = 0.44891`. That value is
actually the **box mAP50-95 / Ultralytics fitness proxy**. The authoritative Task 6M.1 values from
`evaluation/task6m1_training_summary.json` are:

| Metric | Value |
|---|---|
| best box mAP50-95 | **0.44891** |
| best **mask** mAP50-95 | **0.40475** |
| best **mask** mAP50 | **0.73742** |
| best epoch | **40** |

The frozen Task 6M.1 JSON was **not** edited; future reporting uses the training-summary values.

## 4. Scope and protocol

Only the 8 directional L2 programs are handled: `largest_to_{left_of,right_of,above,below}` and
`smallest_to_{left_of,right_of,above,below}`. `nearest`, L1 extremes, L3 compositions and any new
relation are out of scope. **Train is used for training, val for development, and the test split is
never read in Task 6N.**

Per sample: `M_ref` = canonical native-vector GT mask of the reference building; `M_target` =
canonical native-vector GT target mask used only as the training label / evaluation GT; the relation id
is the canonical direction encoded by v0.2; the visual feature is the frozen SAM2 image embedding of
the source RGB tile. The GT target is **never** an input.

## 5. Frozen visual representation

`buildreasonseg_mvp.sam2_bridge.Sam2Encoder` over the **frozen SAM2.1 Hiera Base+** checkpoint
(`local_cache/models/sam2.1_hiera_base_plus.pt`, SHA256
`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`, config
`configs/sam2.1/sam2.1_hiera_b+.yaml`), i.e. exactly the Task 6C.7 / 6I frozen visual path:
**256 × 64 × 64** image embeddings. SAM2 is never retrained, replaced by YOLO features, or fed GT
building-union masks. Features are memoised per tile under the gitignored
`artifacts/task6n/features/` (float32) so all three variants see bit-identical inputs; RBG tiles are
read with PIL because the WHU source root contains non-ASCII characters that `cv2.imread` cannot open.

## 6. GeometricRelationField v0.1

Parameter-free, differentiable, relation-conditioned; no learned parameters, no distance transform, no
bounding box, no candidate masks and no extra geometry channels.

```text
dx = x - cx_ref   dy = y - cy_ref   ax = |dx|   ay = |dy|
s_axis = 0.02   s_margin = 0.02   alpha = 1.2   tau = 0.04   softness = tau / 2 = 0.02

horizontal:  axis = sigmoid((ax - alpha*ay)/s_axis)   margin = sigmoid((ax - tau)/s_margin)
vertical:    axis = sigmoid((ay - alpha*ax)/s_axis)   margin = sigmoid((ay - tau)/s_margin)
left_of: sign = sigmoid(-dx/s_margin)   right_of: sign = sigmoid(dx/s_margin)
above:   sign = sigmoid(-dy/s_margin)   below:    sign = sigmoid(dy/s_margin)

P_rel = clamp(sign * axis * margin, 0, 1)
```

`M_ref` is downsampled to `(h, w)` by coverage-fraction (average) pooling and clamped to `[0, 1]`;
coordinates use pixel centres (`x = (col+0.5)/w`, `y = (row+0.5)/h`). Sign convention follows frozen
Task 3B: `relation(subject, object)` means the subject satisfies the relation **with respect to** the
object, and image `y` increases downward.

## 7. Decoder and the three controlled variants

One common decoder family: `Conv1x1(C→128) + GroupNorm(8,128) + GELU`; a trainable 4 × 16 relation
embedding broadcast over `(h,w)`; then `Conv3x3(in→128) + GN + GELU`, `Conv3x3(128→64) + GN + GELU`,
`Conv1x1(64→1)`; logits bilinearly upsampled to the canonical target-mask resolution (512 × 512) before
loss/evaluation. No attention, transformer, graph block or extra MLP.

| Variant | Fusion input | First conv in-ch | Reference | Field | Parameters |
|---|---|---|---|---|---|
| **N-B0** | visual_128 + relation_embed | 144 | no | no | **273,473** |
| **N-B1** | visual_128 + M_ref_down + relation_embed | 145 | yes | no | **274,625** |
| **N-B2** | visual_128 + M_ref_down + P_rel + relation_embed | 146 | yes | yes | **275,777** |

Missing channels are **not** padded to equalise parameters. Loss is exactly
`BCEWithLogitsLoss + DiceLoss` (the project's canonical `mask_soft_dice`); no GRCL, relation loss,
counterfactual loss, auxiliary field loss, focal loss or class weighting.

## 8. Frozen packs

`evaluation/task6n_pack_manifest.json` (all four packs frozen before training, test untouched):

| Pack | Size | Content |
|---|---|---|
| Overfit20 | 20 train | all 4 directions × both families, **4** same-image counterfactual pairs |
| MiniTrain1000 | 1000 train | stable seeded stratification over direction × reference family |
| MiniVal240 | 240 val | 30 per program id, all 8 ids |
| PairedVal20 | 20 val pairs | same tile, same reference, different direction, different target |

Eligibility: one of the 8 programs, oracle reference and target resolve to canonical native instances,
RGB exists, frozen feature available — **all 1,395 in-scope train and all 1,008 in-scope val records
qualified** (0 exclusions). No tiny/border/visibility filter and no difficulty deletion is applied.

## 9. Stage N0 — field sanity (no training)

`evaluation/task6n_field_sanity.json`, MiniVal240, oracle reference:

| Diagnostic | Value |
|---|---|
| top-1 rate (target ranked first among non-reference instances) | **1.0000** |
| top-3 rate | **1.0000** |
| mean target score | **0.8668** |
| mean score over other buildings | 0.3520 |
| mean best **true** distractor score | **0.1003** |
| mean target − best true distractor margin | **+0.7698** |
| mean native instances per tile | 6.97 |

The parameter-free field already separates the true target from every other building in **100 %** of
the 240 records. This is a diagnostic only; no threshold was tuned.

## 10. Stage N1 — Overfit20

`evaluation/task6n_overfit20.json`, AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler /
no augmentation / seed 20260929 / evaluated every 100 steps, identical for all three variants:

| Variant | best mIoU | best Dice | final mIoU | final Dice |
|---|---|---|---|---|
| N-B0 | 0.9431 | 0.9643 | 0.9408 | 0.9631 |
| N-B1 | 0.9332 | 0.9576 | 0.9332 | 0.9576 |
| **N-B2** | **0.9280** | **0.9520** | **0.9280** | **0.9520** |

**N1 gate: PASS** (B2 mIoU 0.9280 ≥ 0.85 and Dice 0.9520 ≥ 0.90), so the task proceeded to N2. On 20
samples all three variants fit; the field variant is marginally *behind* the baselines here, which is
reported as measured.

## 11. Stage N2 — MiniTrain1000 → MiniVal240

`evaluation/task6n_mini_val.json` + `evaluation/task6n_ablation_summary.json`, AdamW / lr 3e-4 /
wd 1e-4 / batch 8 / ≤ 25 epochs / early stopping patience 5 on val mIoU / seed 20260929 / fresh
initialisation / model selection = highest MiniVal240 mIoU.

| Metric | N-B0 | N-B1 | **N-B2** |
|---|---|---|---|
| MiniVal240 mIoU | 0.2148 | 0.2413 | **0.4531** |
| MiniVal240 Dice | 0.3043 | 0.3276 | **0.5768** |
| Precision@0.5 | 0.4191 | 0.4496 | **0.6640** |
| per relation mIoU (left/right/above/below) | 0.186 / 0.209 / 0.220 / 0.244 | 0.228 / 0.210 / 0.266 / 0.262 | **0.454 / 0.458 / 0.468 / 0.433** |
| largest-ref / smallest-ref mIoU | 0.174 / 0.256 | 0.206 / 0.276 | **0.440 / 0.467** |
| border-target mIoU (n=114) | 0.1935 | 0.2171 | **0.4253** |
| tiny-target mIoU (n=4) | ≈0 | ≈0 | **0.0216** |
| total / trainable params | 273,473 | 274,625 | 275,777 |
| peak VRAM (GB) | 0.894 | 0.674 | 0.674 |
| wall time (s) | 174.8 | 50.9 | 44.8 |
| selected epoch (epochs run) | 10 (15) | 11 (16) | **9 (14)** |

B0's wall time includes generating the frozen feature cache for both packs.

**PairedVal20** (`evaluation/task6n_paired_val.json`; both relations of a pair are run with the same
image and the same oracle reference; a pair passes only if **both** members prefer their own GT target
over the paired alternative by IoU):

| Variant | pass | mean own IoU | mean cross IoU | own − cross |
|---|---|---|---|---|
| N-B0 | 9/20 | 0.1456 | 0.0398 | +0.1058 |
| N-B1 | 7/20 | 0.1871 | 0.0506 | +0.1365 |
| **N-B2** | **16/20** | **0.4433** | **0.0022** | **+0.4412** |

## 12. Causal success criteria and verdict

`evaluation/task6n_verdict.json` — **`GEOMETRIC_RELATION_FIELD_FEASIBLE`**, because all five
section-17 criteria pass:

| Criterion | Required | Measured | Pass |
|---|---|---|---|
| 1. B2 passes N1 | mIoU ≥ 0.85, Dice ≥ 0.90 | 0.9280 / 0.9520 | ✓ |
| 2. MiniVal `B2 − B0` | ≥ 0.05 | **+0.2383** | ✓ |
| 2b. MiniVal `B2 − B1` | ≥ 0.02 | **+0.2118** | ✓ |
| 3. B2 PairedVal | ≥ 14/20 | **16/20** | ✓ |
| 4. B2 own − cross IoU | ≥ 0.10 | **+0.4412** | ✓ |
| 5. no GT target in input | required | satisfied by construction (tested) | ✓ |

Measured context, reported without interpretation: the reference mask alone moves MiniVal mIoU from
0.2148 to 0.2413 (`B1 − B0 = +0.0265`), while adding the field moves it to 0.4531. The field is a
strictly additive input channel in an otherwise identical trunk, and no gate was altered.

## 13. What Task 6N does **not** claim

No "first-ever" claim. No claim that any listed overlap area is novel. No end-to-end claim: the
reference mask is an **oracle** throughout, so this measures an upper-bound feasibility question, not a
deployable pipeline. No research conclusion is drawn by DSH — see `handoff/FROM_DSH.md`.

## 14. Reproduce

```text
python scripts/task6n_freeze_packs.py      # packs + manifest (before any training)
python scripts/task6n_field_sanity.py      # N0
python scripts/task6n_train.py --stage n1  # N1 (Overfit20, gate)
python scripts/task6n_train.py --stage n2  # N2 (MiniTrain1000 -> MiniVal240)
python scripts/task6n_evaluate.py          # MiniVal breakdowns + PairedVal20
python scripts/task6n_report.py            # ablation summary + verdict
```

Run the training/evaluation scripts in `.conda/buildreasonseg-proposal` (frozen SAM2 + torch). Feature
caches, checkpoints and packs live under the gitignored `artifacts/task6n/` tree.
