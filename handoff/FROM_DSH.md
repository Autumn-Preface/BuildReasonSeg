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

# FROM_DSH — Task 6S Report: Directional End-to-End Integration + Hardening Checkpoint

_This file holds the Task 6S report. The Task 6R report is preserved in git history at commit
`a3d59da`; Task 6Q at `7c19bec`; Task 6P at `b80f3cc`; Task 6O at `595e7bb`; Task 6N at `90f3735`._

Full design notes: `docs/task6s_directional_end_to_end_integration.md`.

## 1. Verdict

**`DIRECTIONAL_PARSER_HARDENING_REQUIRED`** — section 24 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen artifacts unchanged, no test access, no GT/annotation dependency in
   the CLI, no GRCL in the primary chain, no oracle reference in inference.
2. `PARSER_CHECKPOINT_UNAVAILABLE` — no: the frozen ProgramHead SHA256 matches the expected hash exactly.
3. `END_TO_END_INTEGRATION_REGRESSION` — no: answered-only MiniVal240 mIoU delta versus frozen Task 6Q is
   **exactly 0.0** (0.3045812554881724) on the identical 234-record answered set.
4. **`DIRECTIONAL_PARSER_HARDENING_REQUIRED`** — canonical parser accuracy 1.0000 ≥ 0.95, but the fixed
   24-prompt paraphrase pack reaches only **21/24 < 22**. ← **verdict**
5. `DIRECTIONAL_CHAIN_BELOW_GATE` — not reached.
6. `DIRECTIONAL_END_TO_END_CHAIN_READY_FOR_HARDENING` — no.

**Dominant bottleneck (fixed rule, sections 20-21): `REFERENCE`** — parser_fail 0 (0.0000 ≤ 0.05),
reference_fail **117**, target_fail 67, so `reference_fail > target_fail`.

## 2. Frozen-asset audit

`evaluation/task6s_frozen_asset_audit.json` = `ASSETS_FROZEN`.

| Asset | Result |
|---|---|
| ProgramHead | SHA256 `eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3` **exact**, text-only (no image input), frozen 20-program vocabulary, not retrained |
| Proposal resolver | SHA256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` **exact**; YOLO26m-seg, imgsz 640, conf 0.10, max_det 100, default NMS, no TTA/tiling; eligibility/ranking untouched |
| B3 target decoder | SHA256 matched exactly against the frozen Task 6O artifact; 274,625 params; not retrained |
| Relation field | v0.2 unmodified (`alpha 1.2`, `tau 0.04`, `s_axis 0.02`, `s_margin 0.02`) |

**Recorded path erratum (no improvisation):** section 5's default path
`artifacts/checkpoints/task6j/j2_best.pt` hashes to
`224f70c97f60d9eb21284cfefdd806eabf300b42375f10d45a25d19547097457` and does **not** match the section-5
expected hash; the checkpoint whose SHA256 equals the expected frozen hash is
`artifacts/checkpoints/task6m/program_parser_v02_best.pt`, which is exactly *"the existing text-only
Qwen3-VL-2B ProgramHead used by Task 6M structured CLI"* as section 5 describes. The **expected SHA256
was treated as authoritative**; both candidates and hashes are recorded. Nothing was retrained or
replaced.

## 3. Primary chain and its exclusions

`natural-language instruction → frozen Qwen3-VL-2B ProgramHead → canonical program → deterministic
decomposition → frozen Task 6Q proposal reference resolver → predicted reference mask →
GeometricRelationField v0.2 → frozen SAM2.1 visual feature + P_rel + relation embedding → frozen Task 6O
B3 → target mask`. **No GRCL**, no oracle reference, no deterministic target-proposal selection, no
`[REF]`, no nearest, no L3, no MLLM hidden-state fusion. Exit codes: 0 answered, 3 reference abstention,
4 out-of-domain, 5 valid-but-unsupported canonical program.

Task 6R evidence recorded for excluding GRCL v0.1: R0/B3 `0.4299680351479113` vs R1 `0.41185668634454997`
mIoU; relation accuracy `0.95` vs `0.9541666666666667`; PairedVal `14/20` vs `12/20`; proposal transfer
`0.3045812554881724` / `10/20` vs `0.283782` / `2/20`. **Task 6R gate erratum recorded, not mutated:**
the predeclared `R1 relation_accuracy >= R0 relation_accuracy + 0.08` was ill-posed because the frozen
baseline already measured `R0 = 0.95`, leaving only 0.05 of headroom below the ceiling 1.0 — a ChatGPT
experiment-design error, not a DSH implementation error.

## 4. Parser audit (MiniVal240, 240 records, both languages)

| Language | exact program accuracy | family / relation accuracy | unsupported classifications |
|---|---|---|---|
| English (primary) | **1.0000** | 1.0000 / 1.0000 | **0** |
| Chinese | **1.0000** | 1.0000 / 1.0000 | 0 |

Gate 1 (≥ 0.95) PASS; every PairedVal20 member also parsed correctly (40/40).

## 5. End-to-end MiniVal240

| Metric | Value |
|---|---|
| strict all-240 mIoU / Dice / Pr@0.5 | **0.2969667241009681** / 0.3753212788292584 / 0.2916666666666667 (gate ≥ 0.28 ✓) |
| answered-only mIoU / Dice / Pr@0.5 (234 records) | **0.3045812554881724** / 0.38494490136334186 / 0.29914529914529914 (gate ≥ 0.2893521927137638 ✓) |
| abstention rate | **0.025000000000000022** (6 records, all `reference_abstention`) |
| per direction (left/right/above/below) mIoU | 0.335333 / 0.247051 / 0.329150 / 0.303450 |
| per family (largest/smallest) mIoU | 0.335005 / 0.273099 |
| border target (n=110) mIoU | 0.30496797891142324 |
| tiny target (n=4) mIoU | 7.42794373054799e-08 (≈ 0) |

Delta versus the frozen Task 6Q chain (oracle program ids): answered-only mIoU **0.0**, Dice
`0.384945 → 0.38494490136334186`, abstentions 6 → 6. **No incremental cost of natural-language parsing**
is measurable on this pack, because the frozen parser is exact on these queries.

*Investigated implementation difference (§23, recorded):* the first paired run gave 14/20. Cause was the
metric epsilon only — the Task 6S aggregate-IoU helper used the project-wide
`(intersection + 1e-6)/(union + 1e-6)`, which lets an **empty** predicted mask score `1e-6/(union+1e-6)`
and win the own-vs-cross comparison whenever the counterpart target is larger; the frozen Task 6Q/6R
paired convention is `intersection/union` (empty → 0.0). Task 6S now uses a dedicated strict paired
helper and reproduces Task 6Q to the last digit. No model, threshold or algorithm changed.

## 6. PairedVal20

| Metric | Task 6S | Frozen Task 6Q chain |
|---|---|---|
| parser-correct members | 40/40 | 40/40 (oracle ids) |
| pass | **10/20** (gate ≥ 9 ✓) | 10/20 |
| mean own / cross / margin | 0.2779456770946653 / 0.00424534769465611 / **0.27370032940000916** (gate ≥ 0.22 ✓) | identical |
| reference-abstention pairs / parser-error pairs | 0 / 0 | 0 / — |

## 7. Fixed 24-prompt paraphrase/CLI audit

| Item | Result |
|---|---|
| supported paraphrases parsed exactly | **21/24** (gate ≥ 22 **FAIL**) |
| failing paraphrases | #2 `找出最大建筑左边的建筑物。` → `smallest_to_left_of`; #5 `找出最大建筑右边的建筑物。` → `smallest_to_right_of`; #11 `找出最大建筑下面的建筑物。` → `smallest_to_below` |
| OOD controls (4, incl. empty string) | **all exit 4** before parser/proposal/SAM2/B3 ✓ |
| out-of-scope controls (4) | 2 exit 5 ✓ (`分割面积最大的建筑物。` → `largest`; `分割最左侧的建筑物。` → `leftmost`); **2 exit 3** ✗ (both "nearest" prompts are parsed as `largest_to_right_of`, a *supported* program) |
| GT available to the CLI | no (`ground_truth_required: false` in every result JSON) |
| on-the-fly SAM2 | **proven** on tile `2_0` (not in MiniVal240, no feature-cache key): exit 0, `sam2_feature = 0.206 s` versus 0.002 s from cache; all four visuals written |

Two measured parser/scope facts, reported without repair as required: (a) the frozen parser is exact on
all 240 canonical queries in both languages but confuses **largest vs smallest** on 3 of the 16 short
Chinese paraphrases; (b) the frozen 20-program vocabulary has **no "nearest" program**, so nearest
semantics are silently mapped onto a supported directional program and the chain stops later at the
resolver (exit 3) instead of being rejected as out-of-scope (exit 5). The 6S scope check can only reject
canonical programs outside the eight; it cannot detect semantics the vocabulary cannot express.

## 8. Failure attribution and dominant bottleneck

| Bucket | Count | Share |
|---|---|---|
| `PARSER_WRONG` | **0** | 0.0000 |
| `REFERENCE_NO_PROPOSALS` | 3 | 0.0125 |
| `REFERENCE_NO_ELIGIBLE` | 3 | 0.0125 |
| `REFERENCE_NOT_COVERED_IOU50` | **68** | 0.2833 |
| `REFERENCE_SELECTION_WRONG` | **42** | 0.1750 |
| `REFERENCE_GEOMETRY_POOR` | 1 | 0.0042 |
| `TARGET_FAIL_WITH_REFERENCE_OK` | 67 | 0.2792 |
| `TARGET_OK` | 56 | 0.2333 |

`parser_fail = 0` (0.0000 ≤ 0.05), `reference_fail = 117`, `target_fail = 67`
→ **dominant bottleneck `REFERENCE`**. Tiny targets (n=4) fall into
`REFERENCE_NOT_COVERED_IOU50` (1) and `TARGET_FAIL_WITH_REFERENCE_OK` (3). This label is for ChatGPT's
next research decision only; DSH does not choose a repair.

## 9. Predeclared gates

| # | Gate | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | parser exact-program accuracy | ≥ 0.95 | 1.0000 | ✓ |
| 2 | strict all-240 mIoU | ≥ 0.28 | 0.2969667 | ✓ |
| 3 | answered-only mIoU | ≥ 0.2893521927137638 | 0.3045813 | ✓ |
| 4 | PairedVal | ≥ 9/20 | 10/20 | ✓ |
| 5 | own-cross margin | ≥ 0.22 | +0.2737003 | ✓ |
| 6 | paraphrase pack | ≥ 22/24 | **21/24** | ✗ |
| 7 | all OOD controls exit 4 | yes | yes | ✓ |
| 8 | all out-of-scope controls exit 5 | yes | 2/4 | ✗ |
| 9 | no GT/annotation dependency in CLI | yes | yes | ✓ |
| 10 | no test split | yes | yes | ✓ |

No gate or threshold was changed.

## 10. Tests, storage, git

`python -m pytest tests/ -q` → **773 passed, 1 skipped** (Task 6R ended at 733 passed / 1 skipped; no
prior passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism
check that needs the proposal env). `tests/test_task6s_directional_end_to_end.py` covers the 42
section-27 checks (10 test-group items merged where the spec lists several assertions for one artifact
write).

Not committed: parser/proposal/B3 checkpoints, SAM2 weights, proposal/feature caches, source
imagery/vectors, generated masks/overlays, `.conda`, large caches (the CLI working directory and all
generated visuals live under the gitignored `artifacts/task6s/`). Committed: integration code, the CLI,
small JSON evaluation artifacts, docs, tests, handoff.

The frozen parser was loaded with `HF_HUB_OFFLINE=1` from the existing local cache, so Task 6S
downloaded nothing (no weights, packages or datasets) and installed nothing. Watt was **not needed** in
Task 6S: the pre-existing Watt instance is transport-only, is not owned by this project and was left
running per the ownership rule; no proxy, host, certificate or TLS setting was read or modified.

## 11. Interpretation boundary

DSH reports measurements only: it does not repair the dominant bottleneck, does not retrain the parser or
the reference resolver, does not add GRCL/nearest/L3, does not start full-dataset training, does not
access the test split and does not build a GUI. A `READY_FOR_HARDENING`-style outcome would not mean the
final model or paper is ready; this run did not reach that outcome.

## 12. Recommended next step

等待 ChatGPT 根据 Task 6S 的全链路审计与 dominant bottleneck（REFERENCE）决定 Task 6T 的 hardening 方向，不自行修复 parser、reference 或 target decoder。

## 13. STOP

Task 6S stops here: no bottleneck repair, no reference retraining, no parser retraining, no GRCL, no
nearest, no L3, no full-dataset training, no test access, no GUI. Waiting for the ChatGPT audit.
