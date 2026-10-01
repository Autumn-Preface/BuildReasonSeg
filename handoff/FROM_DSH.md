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

# FROM_DSH — Task 7E Report: Deterministic Prototype Holdout + Predicted-Reference Audit

_This file holds the Task 7E report. The Task 7D report is preserved in git history at commit `86e4f4c`;
Task 7C at `632c9c0`; Task 7B at `b325585`; Task 7A at `6ee3d0d`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7e_deterministic_prototype_holdout.md`. **No model was trained in Task 7E.**

## 1. Verdict

**`DB1_PREDICTED_REFERENCE_BELOW_GATE`** — section 22 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, no training, no test use, no ranker/quality/refinement.
2. `DB1_CHECKPOINT_UNAVAILABLE` — no: D-B1 SHA256 `6df31909…21a89c0` exact (variant D-B1, selected epoch 7).
3. `L3_HOLDOUT_REMAINDER_INSUFFICIENT` — no: 669 records (≥600), every L3 program ≥120, zero overlap.
4. `L3_HOLDOUT_PAIRED_INSUFFICIENT` — no: 20 pairs chosen from 175 candidates.
5. `TASK7D_REPRODUCTION_FAIL` — no: Z-B3 0.3242128982543474 and D-B1 0.3978996298363562 with Δ **0.0** each,
   paired 15/20 and 19/20 as frozen.
6. `DB1_HOLDOUT_GENERALIZATION_FAIL` — **no: the section-13 gate passed 6/6.**
7. **`DB1_PREDICTED_REFERENCE_BELOW_GATE`** — the D-B1 gain survives the untouched oracle holdout, but the
   section-19 predicted-reference gate fails 3 of 7 conditions. ← **verdict**
8. `DB1_DEVELOPMENT_L3_DECODER_READY` — no.

No threshold was changed after seeing results.

## 2. Frozen assets

| Asset | Value |
|---|---|
| D-B1 `artifacts/checkpoints/task7d/db1_minitrain1200.pt` | SHA256 `6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0` exact · variant **D-B1** · selected epoch **7** · never retrained |
| Z-B3 `artifacts/checkpoints/task6z/zb3_minitrain1200.pt` | SHA256 `74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc` exact · never retrained |

Read-only: GeometricRelationField v0.2 (`alpha 1.2`, `tau 0.04`, `s_axis 0.02`, `s_margin 0.02`),
NearestBoundaryField v0.1 (`sigma_diag 0.05`), frozen SAM2.1 Hiera Base+ features, U-C1 proposal config,
Task 6Q deterministic largest resolver, BuildSpatialReason v0.2, WHU-EA-NativeVector v1.0. No test split.
D-B1 semantics are the frozen Task 7D implementation (no learned score head, no threshold, no proposal).

## 3. `E-HoldoutL3` and `E-PairedHoldout`

Source: the full v0.2 **val** L3 population — **936** records (above 224, below 213, left 250, right 249),
exactly as the task file states. Exclusion set: Z-MiniVal240 (240) ∪ Z-PairedVal20 members (40) = **267** ids.
Remainder **`E-HoldoutL3` = 669** records (left 184, right 181, above 154, below 150), never subsampled: zero
overlap with Z-MiniVal240, zero with Z-PairedVal20, zero test records, ≥600 total, every program ≥120.
`E-PairedHoldout` = **20** pairs from **175** candidates (same tile, same oracle largest reference, different
direction programs, different targets, both members in the holdout), sorted by SHA256 stable key, zero
overlap with the Task 6Z PairedVal20 members.

## 4. E0 reproduction, E1 oracle holdout and the section-13 gate

E0: Z-B3 mIoU `0.3242128982543474` / D-B1 `0.3978996298363562` (both Δ **0.0**), paired **15/20** and
**19/20** → `TASK7D_REPRODUCTION_PASS`.

E1 oracle holdout (669 records): Z-B3 **0.314111** mIoU / 0.424719 Dice / 0.516799 Pr@0.5; D-B1 **0.385496** /
**0.510648** / **0.547313**. Delta **+0.071384** mIoU (+0.085929 Dice, +0.030514 Pr@0.5). Per direction
(Z-B3 → D-B1): above 0.3493 → 0.4075 (**+0.0582**), below 0.2738 → 0.3506 (**+0.0768**), left 0.3314 →
0.3880 (**+0.0566**), right 0.3000 → 0.3931 (**+0.0931**) → **4/4 improve**. Bootstrap (seed 20261001, 2000
paired resamples by record id, 95 % percentile CI): mean **+0.071384**, CI **[+0.058623, +0.084263]**.
`E-PairedHoldout20`: Z-B3 15/20 (margin +0.2592), **D-B1 18/20 (margin +0.3193)**.

**`DB1_HOLDOUT_GENERALIZES = true (6/6)`**: mIoU 0.3855 ≥ 0.36 ✓ · Δ +0.0714 ≥ +0.05 ✓ · 4/4 directions ✓ ·
CI lower +0.0586 > 0.00 ✓ · paired 18/20 ≥ 16/20 ✓ · margin +0.3193 ≥ 0.30 ✓.

## 5. E2 predicted-reference audit and the section-19 gate

Frozen U-C1 as section 14 mandates (the Task 6U `CONFIGS['U-C1']` entry actually used): YOLO26m-seg Task 6M.1
`ef852b58…61f474`, imgsz 640, **conf 0.05**, **max_det 300**, default NMS, no TTA, no tiling; eligibility =
not border-touching and bbox extent ratio ≤ 0.20; selection = max area → tie higher confidence → lower index.
No ranker, no quality filter, no SAM2 refinement. (Task 6Q's read-only `config_report()` prints the resolver
module defaults conf 0.10 / max_det 100; the pipeline uses the frozen U-C1 entry this task specifies.)

Reference diagnostics (GT offline): mIoU **0.440647**, Dice **0.508456**, Pr@0.5 **0.501556**, abstentions
**2** (0.299 %), REFERENCE_OK **346**, SELECTION_WRONG **220**, NOT_COVERED **101**, ABSTENTION **2**,
GEOMETRY_POOR **0**.

| Pipeline | Strict mIoU | Dice | Pr@0.5 | Answered | Answered-only | Reference-OK subset | Abstentions |
|---|---:|---:|---:|---:|---:|---:|---:|
| E-P0 Z-B3 | 0.206911 | 0.279792 | 0.427600 | 667/669 | 0.207542 | 0.328808 | 2 |
| **E-P1 D-B1** | **0.245405** | **0.324796** | **0.471116** | 667/669 | 0.246114 | **0.396365** | 2 |

Retention = **0.636596**; E-P1 − E-P0 strict = **+0.038535**. Predicted-reference `E-PairedHoldout20`
(reference resolved once per pair): Z-B3 **6/20** (margin +0.1251), D-B1 **6/20** (margin +0.1251),
reference-abstention pairs 0.

**`DB1_PREDICTED_REFERENCE_USABLE = false (4/7)`**: strict 0.2454 ≥ 0.24 ✓ · answered-only **0.2461 < 0.25** ✗ ·
Δ vs E-P0 +0.0385 ≥ +0.03 ✓ · retention 0.6366 ≥ 0.62 ✓ · paired **6/20 < 11/20** ✗ · margin **+0.1251 < 0.20**
✗ · abstention 0.299 % ≤ 10 % ✓.

Because section 19 failed, **section 20 was not executed**: `evaluation/task7e_canonical_parser_integration.json`
records `executed: false` with the reason; the Task 7C parser was neither trained nor used to choose the
architecture.

## 6. Decision

| | Value |
|---|---|
| `DB1_HOLDOUT_GENERALIZES` | **true** |
| `DB1_PREDICTED_REFERENCE_USABLE` | **false** |
| `DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER` | **false** |
| Development baseline | **Z-B3 retained** |
| D-B1 role | **positive oracle-reference ablation only** · neither checkpoint deleted or overwritten |

Measured reading (reported only, no repair proposed): D-B1's advantage is real and not a selection artifact —
it survives 669 untouched records with +0.0714 mIoU, CI [+0.0586, +0.0843], 4/4 directions and 18/20 held-out
pairs. Under the frozen U-C1 predicted reference the *ordering* survives (E-P1 beats E-P0 by +0.0385, retention
0.637, reference-OK subset 0.3964 vs 0.3288) but the absolute level and the counterfactual audit miss the
section-19 bars: answered-only 0.2461 < 0.25, predicted paired 6/20 < 11/20, margin +0.1251 < 0.20. The
reference stage dominates the loss — only 346/669 references are REFERENCE_OK, 220 are SELECTION_WRONG and 101
NOT_COVERED — and sharing one reference within a pair makes the own/cross audit far harder than in the oracle
case (18/20 → 6/20).

## 7. Tests, storage, git

`python -m pytest tests/ -q` → **1285 passed, 1 skipped** (Task 7D ended at 1243 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7e_holdout_audit.py` adds the 42 Part-P checks.

Not committed: checkpoints, model weights, proposal/feature caches, the large expanded holdout rows
(`artifacts/task7e/holdout/`), source imagery/vectors, `.conda`. Committed: the small holdout manifest,
evaluation JSON, scripts/helpers, tests, docs, handoff. No new checkpoint was created.

Task 7E downloaded nothing and installed nothing. Watt was **not needed** in Task 7E: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no proxy,
host, certificate or TLS setting was read or modified.

## 8. Interpretation boundary

DSH reports measurements only. D-B1 is **not** claimed globally novel and is **not** called the final model; no
full training was run; the test split was not evaluated; parser/reference were not retrained; no learned
competition, graph or attention module was added; D-B1 was **not** modified after the holdout results; Task 7F
was not chosen.

## 9. Recommended next step (exact wording required by Part N)

等待 ChatGPT 根据 Task 7E 的 untouched L3 holdout 与 predicted-reference 结果决定是否正式冻结 D-B1 为开发版 L3 decoder，不自行进行全量训练、test 评估或新的架构改动。

## 10. STOP

Task 7E stops here: no training, no test run, no parser/reference/D-B1 change, no additional architecture, no
formal full-data training, no GUI. Waiting for the ChatGPT audit.
