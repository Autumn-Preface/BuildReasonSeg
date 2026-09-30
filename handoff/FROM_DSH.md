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

# FROM_DSH — Task 7A Report: L3 Predicted-Reference + Natural-Language Integration Audit

_This file holds the Task 7A report. The Task 6Z report is preserved in git history at commit `7006d7d`;
Task 6Y at `24be954`; Task 6X at `9318890`; Task 6W at `3de2142`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7a_l3_predicted_reference_integration.md`. **No model was trained in Task 7A.**

## 1. Verdict

**`L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`** — section 23 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, no test use, no training, both checkpoint hashes exact.
2. `L3_CHECKPOINT_UNAVAILABLE` — no: Z-B3 SHA256 `74f308e1…9fc0f0bc` recomputed and matching, variant `Z-B3`.
3. `PARSER_CHECKPOINT_UNAVAILABLE` — no: hardened ProgramHead `4cbba36b…d44a5e` matching.
4. `TASK6Z_REPRODUCTION_FAIL` — **no: reproduction is exact** (mIoU Δ 0.0, Dice Δ 0.0, paired 15/20,
   margin Δ 0.0).
5. **`L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`** — the canonical predicted-reference section-20 gate fails
   on four of six conditions (strict mIoU 0.2170 < 0.22; answered-only 0.2198 < 0.24; retention 0.6693 <
   0.68; paired 8/20 < 10/20; margin +0.1995 ✓; abstention 0.0125 ✓). ← **verdict**
6. `L3_LANGUAGE_HARDENING_REQUIRED` — not reached (the reference gate already fails).
7. `L3_END_TO_END_DEVELOPMENT_CHAIN_READY` — no.

No threshold was changed after seeing results.

## 2. Recorded Task 6Z result (Task 6Z artifacts not mutated)

`L3_COMPOSITION_NO_MEANINGFUL_GAIN`: composition sanity direction-valid top-1 1.0000 / top-3 1.0000 /
Spearman 0.9983; Z-B3 Overfit20 0.9619 / 0.9804; MiniVal240 Z-B0 0.1510, Z-B1 0.3011, Z-B2 0.1724,
**Z-B3 0.3242**, Z-B4 0.2788, Z-B5 0.0888 with Z-B3 deltas vs B0 +0.1732, vs B1 +0.0231, vs B2 +0.1519,
vs B4 +0.0454, vs B5 +0.2355; paired Z-B3 **15/20**, margin **+0.300354**; tests `1062 passed, 1 skipped`.
Z-B3 stays the current best L3 research decoder; Task 7A does **not** convert the 6Z verdict into a success
claim and only measures practical reference/parser error propagation.

## 3. A0 oracle reproduction (exact)

`evaluation/task7a_oracle_reproduction.json` — the frozen 6Z inference path re-run on the exact
Z-MiniVal240 / Z-PairedVal20 packs:

| Quantity | Recomputed | Frozen 6Z | |Δ| | Tolerance |
|---|---|---|---|---|
| MiniVal mIoU | 0.3242128982543474 | 0.3242128982543474 | **0.0** | 1e-6 |
| MiniVal Dice | 0.4389840055529761 | 0.4389840055529761 | **0.0** | 1e-6 |
| Paired | **15/20** | 15/20 | 0 | exact |
| own−cross margin | +0.30035408969722216 | +0.30035408969722216 | **0.0** | 1e-6 |

`TASK6Z_REPRODUCTION_PASS` (own mean 0.30316, cross mean 0.00280).

## 4. A1 canonical predicted-reference chain

Reference diagnostics (240 records; mean 19.35 proposals / 14.25 eligible): abstentions **3** (rate
**0.0125**), selected-reference mIoU **0.4700**, Dice 0.5430, Pr@0.5 **0.5527**, centroid error mean/median/
p90 0.1056/**0.0234**/0.3240, best eligible coverage@0.50 **0.8583**; buckets `NO_PROPOSALS` 1 ·
`NO_ELIGIBLE_PROPOSALS` 2 · `REFERENCE_NOT_COVERED_IOU50` **31** · `REFERENCE_SELECTION_WRONG` **75** ·
`REFERENCE_GEOMETRY_POOR` 0 · `REFERENCE_OK` **131**.

Target metrics: strict mIoU **0.2170**, Dice 0.2931, Pr@0.5 0.4252, answered-only **0.2198**, abstentions 3,
**retention 0.6693** (oracle Z-B3 0.3242), reference-OK subset **0.3434** vs reference-fail subset
**0.0651**, per direction above 0.1913 / below 0.2356 / left 0.2518 / right 0.1893. Paired: **8/20**,
own 0.2248 / cross 0.0253 / margin **+0.1995**, 0 reference-abstention pairs (the predicted reference is
computed once per tile and reused for both member programs).

## 5. A2 hardened ProgramHead integration

Canonical-query parser audit: **240/240 exact (1.0000)**, per-class recall 1.0 for all four L3 classes,
**0** out-of-scope predictions. Natural-language end-to-end: strict mIoU **0.2170**, Dice 0.2931,
answered-only **0.2198**, abstentions 3, out-of-scope 0, per direction identical to A1 — i.e. **the language
stage adds no measurable loss on the dataset's own queries**. Paired: parser-correct members **40/40**,
**8/20** pairs, margin **+0.1995**, 0 parser-error pairs, 0 reference-abstention pairs.

## 6. Fixed 24-prompt paraphrase audit

The pack (`evaluation/task7a_l3_paraphrase_pack.json`) was frozen **before** any parser run: 24 prompts, 6 per
L3 program (3 Chinese + 3 English), all eight required compact/contrast forms present. Result:
**3/24 exact (0.125)**; LEFT 1/6, RIGHT 0/6, ABOVE 1/6, BELOW 1/6; Chinese 3/12, English **0/12**;
compact **0/8** → `l3_paraphrase_ready = false`.

Failure modes: **19 of 21 failures drop the terminal `to_nearest`** and return the L2 program
`largest_to_<direction>` (e.g. `分割面积最大的建筑物右侧最近的建筑物。` → `largest_to_right_of`,
`find the closest building to the left of the largest building` → `largest_to_left_of`), plus 2 collapses to
`smallest_to_<direction>`. No parser retraining was performed.

## 7. CMD entry point

`predict_buildreasonseg_l3.py` (no GT/annotation argument; three such arguments refused). CLI audit
`all_checks_passed = true`: success exit 0 with all six outputs and every required `result.json` key
(`ground_truth_used = false`); out-of-scope canonical program exit **5** with only `result.json` and
`stopped_before = [proposals, reference, fields, sam2, z_b3]`.

## 8. Failure attribution

| Bucket | Count | Share |
|---|---|---|
| `PARSER_WRONG` | **0** | 0.0000 |
| `REFERENCE_NO_PROPOSALS` | 1 | 0.0042 |
| `REFERENCE_NO_ELIGIBLE` | 2 | 0.0083 |
| `REFERENCE_NOT_COVERED_IOU50` | 31 | 0.1292 |
| `REFERENCE_SELECTION_WRONG` | **75** | **0.3125** |
| `REFERENCE_GEOMETRY_POOR` | 0 | 0.0000 |
| `TARGET_FAIL_WITH_REFERENCE_OK` | 90 | 0.3750 |
| `TARGET_OK` | 41 | 0.1708 |

parser_fail 0 ≤ 0.05 → not PARSER; reference_fail **109** > target_fail **90** →
**dominant bottleneck `REFERENCE`**. DSH proposes no repair.

## 9. Tests, storage, git

`python -m pytest tests/ -q` → **1107 passed, 1 skipped** (Task 6Z ended at 1062 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task7a_l3_integration.py` adds the 45 section-N checks.

Not committed: parser/YOLO/SAM2/Z-B3 weights, feature/proposal caches, generated CLI masks and overlays
(`artifacts/task7a/`), source imagery/vectors, `.conda`. Committed: integration code, small evaluation JSON,
scripts, tests, docs, handoff. **No new model checkpoint.**

Task 7A downloaded nothing and installed nothing. Watt was **not needed** in Task 7A: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no
proxy, host, certificate or TLS setting was read or modified.

## 10. Interpretation boundary

DSH reports measurements only. Task 6Z is not turned into a success claim; the parser was not retrained;
reference hardening was not reopened; no global attention; Z-B3 and the fields are unchanged; no full
training and no test evaluation; no repair proposed.

## 11. Recommended next step (exact wording required by Part L)

等待 ChatGPT 根据 Task 7A 的 L3 predicted-reference、ProgramHead 与 failure attribution 结果决定下一步，不自行进行 parser 再训练、reference 再硬化、attention/global competition 或正式全量训练。

## 12. STOP

Task 7A stops here: no parser retraining, no reference re-hardening, no attention/global competition, no
field/decoder change, no formal full training, no test access, no GUI. Waiting for the ChatGPT audit.
