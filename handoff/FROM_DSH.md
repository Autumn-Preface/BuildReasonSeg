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

# FROM_DSH — Task 6V Report: Family-Conditioned Reference Resolver Policy

_This file holds the Task 6V report. The Task 6U report is preserved in git history at commit `b8b5224`;
Task 6T at `e63f8c4`; Task 6S at `dc8544f`; Task 6R at `a3d59da`; Task 6Q at `7c19bec`._

**Note on the legacy block above:** those `ARTIFACT-FACTS` numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6v_family_conditioned_reference_resolver.md`.

## 1. Verdict

**`FAMILY_POLICY_NOT_BETTER`** — section 15 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: no test use, no GT in inference, no post-RefVal policy change, no frozen
   module mutation, no fourth option, no training.
2. `FROZEN_RESOLVER_ASSET_UNAVAILABLE` — no: YOLO26m-seg, ranker, hardened ProgramHead and B3 hashes all
   verified exactly.
3. **`FAMILY_POLICY_NOT_BETTER`** — the section 13 improvement flag fails (RefVal mIoU Δ **+0.0094** <
   +0.015; `REFERENCE_OK` 114 < 117) **and** the section 14 hardening gate fails on 7 of 10 conditions.
   ← **verdict**
4. `FAMILY_POLICY_PARTIAL` — not applicable (the improvement flag does not hold).
5. `FAMILY_CONDITIONED_REFERENCE_HARDENING_PASS` — no.

**Frozen policy: `{"largest": "V-P2", "smallest": "V-P0"}`. No model was trained in Task 6V.**

## 2. Recorded Task 6U findings (Task 6U artifacts not mutated)

Candidate coverage, U-C1 vs U-C0 on untouched RefValUnique: overall eligible@0.50 0.68950 → 0.75342,
largest 0.83636 → 0.87273, smallest 0.54128 → 0.63303 (candidate recall improved). Selector behaviour on
RefValUnique — largest: U-S0 0.54284/0.65138, U-S1 0.51830/0.60000, U-S2 0.56804/0.66364 (mIoU/Pr@0.5);
smallest: U-S0 0.30109/0.39423, U-S1 0.33532/0.43810, U-S2 0.04105/0.03810. The shared v0.1 ranker is not
accepted globally; family-specific usefulness must be chosen on train-only calibration, not RefVal.

## 3. Frozen assets and exactly three options

Verified exactly: YOLO26m-seg `ef852b5801…61f474` (not retrained), ProposalSetRanker v0.1
`c738fcf77419626f…1a46c0` (not retrained), Task 6T hardened ProgramHead `4cbba36b1364b0a8…d44a5e` (not
retrained), Task 6O B3 `7556e4a4862b75d4…c7d6ab` (not retrained), frozen pack hashes byte-identical. Only
three options exist:

* **V-P0** = U-C0 + deterministic Task 6Q area selector;
* **V-P1** = U-C1 + deterministic Task 6Q area selector;
* **V-P2** = U-C1 + frozen ProposalSetRanker v0.1.

## 4. Train-only policy selection on U-Calib200 (Parts C-D)

| family | option | Pr@0.5 | mIoU | SELECTION_WRONG | REFERENCE_OK | NOT_COVERED | centroid med | centroid p90 |
|---|---|---|---|---|---|---|---|---|
| largest | V-P0 | 0.7600 | 0.6338 | 22 | 76 | 2 | 0.0047 | 0.2508 |
| largest | V-P1 | 0.7100 | 0.6034 | 27 | 71 | 2 | 0.0059 | 0.2806 |
| **largest** | **V-P2** | **0.8000** | **0.6650** | **18** | **80** | 2 | 0.0044 | 0.1929 |
| **smallest** | **V-P0** | **0.5300** | **0.4154** | **38** | **53** | 9 | 0.0044 | 0.3146 |
| smallest | V-P1 | 0.4500 | 0.3545 | 49 | 45 | 6 | 0.0506 | 0.3033 |
| smallest | V-P2 | 0.0300 | 0.0546 | 91 | 3 | 6 | 0.1599 | 0.3801 |

Section 7 priority → `largest: [V-P2, V-P0, V-P1]`, `smallest: [V-P0, V-P1, V-P2]` →
`task6v_frozen_family_policy.json` = **`{"largest": "V-P2", "smallest": "V-P0"}`**, frozen before any
RefValUnique evaluation and never revisited.

**Recorded inversion:** the train-only calibration prefers **U-C0** for `smallest`, while Task 6U's
RefValUnique evaluation preferred U-C1 (0.33532 vs 0.30109). The predeclared rule selects on calibration
only, so the frozen policy uses U-C0 for `smallest` and gives back the C1 coverage gain on that family.

## 5. RefValUnique evaluation (Part E)

| Metric | Frozen family policy | U-S0 | U-S1 | U-S2 | U-C1 oracle ceiling |
|---|---|---|---|---|---|
| selected-reference mIoU | **0.4383** | 0.4248 | 0.4289 | 0.3107 | 0.6203 |
| Dice | 0.4995 | 0.4832 | 0.4933 | 0.3564 | 0.7092 |
| Pr@0.5 | **0.5327** | 0.5258 | 0.5209 | 0.3581 | 0.7674 |
| centroid mean / median / p90 | 0.1132 / **0.0107** / 0.3845 | — / 0.0156 / 0.3935 | — / 0.0159 / 0.3803 | — / 0.0890 / 0.3771 | — / 0.0046 / 0.108 |
| area-ratio median | 1.0757 | 1.0813 | 1.0719 | 1.4695 | — |
| abstention rate | **0.0228** | 0.0274 | 0.0183 | 0.0183 | — |
| largest mIoU / Pr@0.5 | **0.5680 / 0.6636** | 0.5428 / 0.6514 | 0.5183 / 0.6000 | 0.5680 / 0.6636 | — |
| smallest mIoU / Pr@0.5 | 0.3011 / 0.3942 | 0.3011 / 0.3942 | **0.3353 / 0.4381** | 0.0411 / 0.0381 | — |

| Bucket | Frozen policy | U-S0 | U-S1 | U-S2 |
|---|---|---|---|---|
| `NO_PROPOSALS` | 2 | 3 | 1 | 1 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 3 | 3 | 3 |
| `REFERENCE_NOT_COVERED_IOU50` | 59 | 62 | **50** | **50** |
| `REFERENCE_SELECTION_WRONG` | **41** | 39 | 53 | 88 |
| `SELECTED_MASK_GEOMETRY_POOR` | 0 | 1 | 0 | 0 |
| `REFERENCE_OK` | 114 | 111 | **112** | 77 |

Routing works as intended (largest gets the ranker's better mIoU, smallest keeps the deterministic
selector's competence instead of the ranker's collapse) but the C0 choice for `smallest` costs coverage
(`NOT_COVERED` 50 → 59).

## 6. Downstream causal evaluation and integration (Parts F-G)

MiniVal240 with canonical program ids (no parser):

| Metric | Frozen family policy | U-S0 | U-S1 | U-S2 |
|---|---|---|---|---|
| strict all-240 mIoU | 0.3005 | 0.2970 | **0.3089** | 0.2425 |
| answered-only mIoU | 0.3069 | 0.3046 | **0.3141** | 0.2466 |
| abstentions | 5 | 6 | **4** | 4 |
| reference-fail count | 116 | 117 | **115** | 158 |
| target-fail-with-reference-ok | 68 | 67 | — | — |
| largest / smallest target mIoU | 0.3393 / 0.2617 | — | — | — |
| per direction (left/right/above/below) | 0.3189 / 0.2365 / 0.3441 / 0.3025 | — | — | — |
| border target (n=114) | 0.2985 | 0.2943 | 0.2896 | 0.2666 |
| tiny target (n=4) | ≈0 | ≈0 | ≈0 | ≈0 |

PairedVal20: pass **10/20**, own 0.291106, cross 0.003251, margin **+0.2879**, 0 abstention pairs
(U-S0 10/20 +0.2737; U-S1 11/20 +0.3209; U-S2 11/20 +0.3332).

`task6v_hardened_parser_integration.json` — Task 6T hardened ProgramHead → family-conditioned resolver →
field v0.2 → SAM2 → B3: parser **240/240**, strict 0.3005, answered-only 0.3069, abstentions 5,
reference-fail 116 (identical to the causal run). No nearest/L3 execution was evaluated.

## 7. Predeclared flags and gates

Section 13 `family_policy_improved` = **false** (versus frozen U-S1):

| Criterion | Required | Measured | Pass |
|---|---|---|---|
| RefVal mIoU | ≥ U-S1 + 0.015 = 0.4439 | 0.4383 (Δ **+0.0094**) | ✗ |
| RefVal `REFERENCE_OK` | ≥ U-S1 + 5 = 117 | 114 | ✗ |
| RefVal `REFERENCE_SELECTION_WRONG` | ≤ U-S1 − 5 = 48 | **41** | ✓ |
| abstention rate | ≤ 0.05 | 0.0228 | ✓ |

Section 14 `DIRECTIONAL_REFERENCE_HARDENING_PASS`: **3 of 10** pass — centroid median 0.0107 ≤ 0.03 ✓,
parser 240/240 ✓, no test/no GT ✓; failing: RefVal mIoU 0.4383 < 0.45, centroid p90 0.3845 > 0.32,
MiniVal answered 0.3069 < 0.325, strict 0.3005 < 0.305, paired 10 < 12, margin 0.2879 < 0.30,
reference-fail 116 > 105.

No threshold was altered after seeing results.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **881 passed, 1 skipped** (Task 6U ended at 851 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6v_family_reference_resolver.py` adds the 30 section-L checks.

Not committed: ranker/parser/YOLO/SAM2/B3 checkpoints, proposal caches (`artifacts/task6u/proposals/`),
feature caches, source imagery/vectors, `.conda`. Committed: family resolver code, small JSON artifacts,
scripts, tests, docs, handoff. **No new checkpoint was created in Task 6V.**

Task 6V downloaded nothing and installed nothing; every frozen asset was loaded from local disk. Watt was
**not needed** in Task 6V: the pre-existing Watt instance is transport-only, is not owned by this project
and was left running per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 9. Interpretation boundary

DSH reports measurements only. Family routing is **not** claimed as a project novelty; the ranker was not
redesigned; no smallest-only ranker, proposal-quality classifier, candidate-config change, additional
confidence threshold, TTA/tiling, YOLO retraining, nearest/L3 work or Task 6W selection was performed.

## 10. Recommended next step

等待 ChatGPT 根据 Task 6V 的 family-conditioned resolver 结果决定是否需要新的 proposal-quality / selection 机制，不自行继续修改 reference 架构。

## 11. STOP

Task 6V stops here: no new reference models, no proposal threshold/config changes, no YOLO/ranker/parser/B3
retraining, no proposal-quality model, no GRCL, no nearest/L3, no test access, no GUI. Waiting for the
ChatGPT audit.

---

_The Task 6U report is preserved below in git history at commit `b8b5224`._
