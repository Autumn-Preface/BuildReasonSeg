# Task 7A — L3 Predicted-Reference + Natural-Language Integration Audit

> Task: `handoff/TO_DSH.md` (Task 7A) · Base commit: `7006d7d` · Predecessor: Task 6Z →
> `L3_COMPOSITION_NO_MEANINGFUL_GAIN`
> **Verdict: `L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`** · A0 reproduction **exact** · parser on canonical
> queries **240/240** · dominant bottleneck **REFERENCE**
> Tests: `tests/test_task7a_l3_integration.py` · Evidence: `evaluation/task7a_*.json`
> **No model was trained in Task 7A.**

Task 7A is an integration/attribution audit: it measures how much of the oracle-reference Z-B3 L3 capability
survives when the oracle largest reference is replaced by the frozen practical U-C1 deterministic resolver
and when the hardened ProgramHead supplies the natural-language program.

## 1. Recorded Task 6Z result (Task 6Z artifacts not mutated)

Verdict `L3_COMPOSITION_NO_MEANINGFUL_GAIN`. Parameter-free composition sanity: direction-valid top-1
`1.0000`, top-3 `1.0000`, Spearman `0.9983`. Z-B3 Overfit20 mIoU `0.9619` / Dice `0.9804`. MiniVal240:
Z-B0 `0.1510`, Z-B1 `0.3011`, Z-B2 `0.1724`, **Z-B3 `0.3242`**, Z-B4 `0.2788`, Z-B5 `0.0888`; Z-B3 deltas vs
B0 `+0.1732`, vs B1 `+0.0231`, vs B2 `+0.1519`, vs B4 `+0.0454`, vs B5 `+0.2355`; paired Z-B3 **15/20** with
own-cross margin **+0.300354**; tests `1062 passed, 1 skipped`.

Interpretation boundary: Z-B3 is the current best L3 research decoder; Task 7A does **not** convert the
formal 6Z verdict into a success claim, and only measures practical reference/parser error propagation.

## 2. Frozen assets

All Task 6Z artifacts, the Z-MiniVal240/Z-PairedVal20 packs, the Z-B3 checkpoint
(`artifacts/checkpoints/task6z/zb3_minitrain1200.pt`, SHA256
`74f308e1…9fc0f0bc`, variant metadata `Z-B3`, 275,777 params), the Task 6T hardened ProgramHead
(`4cbba36b…d44a5e`), the Task 6M.1 YOLO26m-seg checkpoint, U-C1 (`imgsz 640 / conf 0.05 / max_det 300 /
default NMS / no TTA / no tiling`), the Task 6Q deterministic `largest` resolver, GeometricRelationField
v0.2, NearestBoundaryField v0.1 and the frozen SAM2.1 feature path. No test split; no training.

## 3. Scope and the A0 oracle reproduction

Exactly four L3 programs, decomposed without any learned step: `largest_to_left_of_to_nearest` /
`..._right_of...` / `..._above...` / `..._below...` → `family=largest`, the corresponding direction,
`terminal=nearest`.

`evaluation/task7a_oracle_reproduction.json` — the frozen 6Z inference path re-run on the exact packs:

| Quantity | Recomputed | Frozen Task 6Z | |Δ| | Tolerance |
|---|---|---|---|---|
| MiniVal mIoU | 0.3242128982543474 | 0.3242128982543474 | **0.0** | 1e-6 |
| MiniVal Dice | 0.4389840055529761 | 0.4389840055529761 | **0.0** | 1e-6 |
| Paired | **15/20** | 15/20 | 0 | exact |
| own−cross margin | +0.30035408969722216 | +0.30035408969722216 | **0.0** | 1e-6 |

**`TASK6Z_REPRODUCTION_PASS`** (own mean 0.30316, cross mean 0.00280).

## 4. A1 canonical predicted-reference chain

`evaluation/task7a_predicted_reference_quality.json` — 240 records, mean 19.35 proposals / 14.25 eligible:

| Metric | Value |
|---|---|
| reference abstentions | **3** (rate **0.0125**) |
| selected-reference mIoU / Dice | **0.4700** / 0.5430 |
| Pr@0.5 | **0.5527** |
| centroid error mean / median / p90 | 0.1056 / 0.0234 / 0.3240 |
| best eligible proposal coverage@0.50 | **0.8583** |
| buckets | `NO_PROPOSALS` 1 · `NO_ELIGIBLE_PROPOSALS` 2 · `REFERENCE_NOT_COVERED_IOU50` **31** · `REFERENCE_SELECTION_WRONG` **75** · `REFERENCE_GEOMETRY_POOR` 0 · `REFERENCE_OK` **131** |

`evaluation/task7a_canonical_predicted_reference_val.json` (strict over all 240 records):
strict mIoU **0.2170**, Dice 0.2931, Pr@0.5 0.4252, answered-only mIoU **0.2198**, abstentions 3,
**retention 0.6693** against the oracle Z-B3 0.3242; reference-OK subset **0.3434** vs reference-fail subset
**0.0651**; per direction above 0.1913 / below 0.2356 / left 0.2518 / right 0.1893.

`evaluation/task7a_canonical_predicted_reference_paired.json` — the predicted largest reference is computed
once per tile and reused for both member programs: **8/20**, mean own 0.2248, cross 0.0253, margin
**+0.1995**, 0 reference-abstention pairs.

Section 20 gate: strict ≥ 0.22 ✗ (0.2170), answered ≥ 0.24 ✗ (0.2198), retention ≥ 0.68 ✗ (0.6693),
paired ≥ 10/20 ✗ (8/20), margin ≥ 0.15 ✓ (+0.1995), abstention ≤ 0.10 ✓ (0.0125) → **gate fails**.

## 5. A2 hardened ProgramHead integration

`evaluation/task7a_parser_l3_val.json` — the frozen ProgramHead on the actual natural-language queries of the
exact Z-MiniVal240 records (`instruction_en`, joined by `sample_id` against the frozen v0.2 val split):
**240/240 exact (accuracy 1.0000)**, per-class recall 1.0 for all four L3 classes, **0** out-of-scope
predictions.

`evaluation/task7a_natural_language_val.json` — end-to-end with only the image and the natural-language
query: parser 240/240, strict mIoU **0.2170**, Dice 0.2931, answered-only **0.2198**, abstentions 3,
out-of-scope 0, per direction above 0.1913 / below 0.2356 / left 0.2518 / right 0.1893 — i.e. **identical to
A1**, so the language stage adds no measurable loss on the dataset's own queries.

`evaluation/task7a_natural_language_paired.json`: parser-correct members **40/40**, **8/20** pairs,
margin **+0.1995**, 0 parser-error pairs, 0 reference-abstention pairs.

Section 21 gate: parser ≥ 0.98 ✓ (1.0000), strict ≥ 0.21 ✓ (0.2170), answered ≥ 0.23 ✗ (0.2198),
paired ≥ 9/20 ✗ (8/20), margin ≥ 0.14 ✓ (+0.1995) → **gate fails**.

## 6. Fixed 24-prompt paraphrase audit

`evaluation/task7a_l3_paraphrase_pack.json` was frozen **before** any parser run: 24 prompts, 6 per L3
program (3 Chinese + 3 English), including all eight exact compact/contrast forms required by section 15
(`required_compact_present = true`).

`evaluation/task7a_l3_paraphrase_result.json`: **3/24 exact (0.125)**; per program LEFT 1/6, RIGHT 0/6,
ABOVE 1/6, BELOW 1/6; per language Chinese 3/12, English **0/12**; exact compact prompts **0/8** →
`l3_paraphrase_ready = **false**`.

Measured failure modes (reported, no repair proposed): **19 of 21 failures drop the terminal `to_nearest`**
and return the level-2 directional program `largest_to_<direction>` (e.g. the required forms
`分割面积最大的建筑物右侧最近的建筑物。` → `largest_to_right_of` and
`find the closest building to the left of the largest building` → `largest_to_left_of`), plus 2 collapses to
`smallest_to_<direction>`. The parser is perfect on the dataset's own templates but not paraphrase-robust.

## 7. CMD entry point

`predict_buildreasonseg_l3.py` — required example form, **no** ground-truth/annotation argument (three
such arguments are refused with a non-zero exit code). `evaluation/task7a_cli_audit.json` →
`all_checks_passed = true`:

* success path: exit **0**, all six outputs written (`result.json`, `reference_mask.png`,
  `direction_field.png`, `nearest_field.png`, `target_mask.png`, `overlay.png`), every required
  `result.json` key present, `ground_truth_used = false`;
* out-of-scope canonical program: exit **5**, only `result.json` written, `stopped_before` lists
  `proposals/reference/fields/sam2/z_b3`.

## 8. Failure attribution

`evaluation/task7a_failure_attribution.json` (exclusive buckets, fixed section-19 priority):

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

parser_fail 0 / 240 = 0 ≤ 0.05 → not PARSER; reference_fail **109** > target_fail **90** →
**dominant bottleneck `REFERENCE`**. DSH proposes no repair.

## 9. Verdict (section 23)

Protocol clean, both checkpoint hashes exact, A0 reproduction exact; the canonical predicted-reference
section-20 gate fails on four of six conditions → **`L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`** (item 5,
which precedes the language item). No threshold was changed after seeing results.

Measured reading (reported only): the L3 decoder itself is not the limiting factor in this integration — the
reference stage is. Selected-reference mIoU 0.4700 with Pr@0.5 0.5527 and only 3 abstentions, but
**106 of 240 records (44.2%) land in the two "wrong or uncovered reference" buckets**, and the
reference-OK subset reaches 0.3434 while the reference-fail subset collapses to 0.0651 — i.e. the predicted
reference, not the two-field decoder, carries most of the practical loss (retention 0.6693). On the
dataset's own queries the parser contributes nothing to that loss (240/240); on unseen paraphrases it fails
21/24, mostly by dropping the terminal `to_nearest`.

## 10. Interpretation boundary

DSH reports measurements only. Task 6Z is not turned into a novelty/success claim; the parser was not
retrained; reference hardening was not reopened; no global attention; Z-B3 and the fields are unchanged; no
full training and no test evaluation was started; no repair is proposed. Final recommendation exactly:

`等待 ChatGPT 根据 Task 7A 的 L3 predicted-reference、ProgramHead 与 failure attribution 结果决定下一步，不自行进行 parser 再训练、reference 再硬化、attention/global competition 或正式全量训练。`

## 11. Reproduce

```text
python scripts/task7a_evaluate_reference.py       # A0 reproduction + A1 predicted reference
python scripts/task7a_parser_audit.py             # paraphrase pack + canonical parser audit + paraphrases
python scripts/task7a_evaluate_pipeline.py        # A2 natural-language end-to-end
python scripts/task7a_failure_attribution.py      # exclusive buckets + dominant bottleneck
python scripts/task7a_cli_audit.py                # CMD entry point audit
python scripts/task7a_report.py                   # gates + verdict
python predict_buildreasonseg_l3.py --image <tile.tif> --prompt "<query>" \
  --parser-checkpoint artifacts/checkpoints/task6t/program_parser_hardened_v1.pt \
  --proposal-checkpoint artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt \
  --target-checkpoint artifacts/checkpoints/task6z/zb3_minitrain1200.pt --out-dir outputs/buildreasonseg_l3
```

CLI outputs and caches live under the gitignored `artifacts/task7a/`. Run in
`.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the attribution/report steps also run in
`.conda/buildreasonseg-mvp`.
