# Task 7E — Deterministic Field-Weighted Prototype Holdout + Predicted-Reference Audit

> Task: `handoff/TO_DSH.md` (Task 7E) · Base commit: `86e4f4c` · Predecessor: Task 7D →
> `GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`
> **Verdict: `DB1_PREDICTED_REFERENCE_BELOW_GATE`** · oracle holdout gate **passed 6/6** · predicted-reference
> gate **4/7**
> Evaluation/integration only — **no model was trained**, nothing was tuned
> Tests: `tests/test_task7e_holdout_audit.py` · Evidence: `evaluation/task7e_*.json`

Task 7E asks whether the frozen Task 7D **D-B1 deterministic field-weighted visual prototype** keeps its gain
on an untouched validation remainder that played no part in its checkpoint selection, and then how much of that
gain survives predicted-reference error propagation from the frozen **U-C1** resolver.

## 1. Recorded Task 7D result (frozen, not modified)

Verdict `GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`.

| Variant | MiniVal mIoU | Dice | Paired | Margin |
|---|---:|---:|---:|---:|
| D-B0 frozen Z-B3 | 0.3242129 | 0.4389840 | 15/20 | +0.3003541 |
| D-B1 deterministic field-weighted prototype | **0.3978996** | **0.5272593** | **19/20** | **+0.3888** |
| D-B2 learned relation-guided competition | 0.3264764 | 0.4372995 | 17/20 | +0.2956 |
| D-B3 learned visual-only competition | 0.1595259 | 0.2310875 | 8/20 | +0.1629 |
| D-B4 learned competition map, no prototype | 0.3291152 | 0.4487840 | 19/20 | +0.3056 |

Important diagnostics: D-B2 target mass 0.0067, argmax-in-target 0.0000, entropy 0.8848; learned competition
is **not** accepted; D-B1 is the only variant with a clear mask/counterfactual gain over Z-B3.

## 2. Frozen assets

| Asset | SHA256 / setting |
|---|---|
| D-B1 checkpoint `artifacts/checkpoints/task7d/db1_minitrain1200.pt` | `6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0` (exact), variant D-B1, selected epoch **7**, never retrained |
| Z-B3 checkpoint `artifacts/checkpoints/task6z/zb3_minitrain1200.pt` | `74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc` (exact), never retrained |
| Read-only | GeometricRelationField v0.2, NearestBoundaryField v0.1, frozen SAM2.1 Hiera Base+ `V ∈ R^(256×64×64)`, U-C1 proposal configuration, Task 6Q deterministic largest resolver, BuildSpatialReason v0.2, WHU-EA-NativeVector v1.0 · no test split |

D-B1 semantics are the **frozen Task 7D implementation** (unchanged): `W = clamp(P_dir * P_near)`,
`A_fixed = W / (ΣW + eps)`, `A_fixed_vis = A_fixed * 4096`, `q = Σ_i A_fixed_i F_i`, `C_i = cos(F_i, q)`,
decoder inputs `F + P_dir + P_near + direction embedding + A_fixed_vis + C` — no learned score head, no
threshold, no target proposal.

## 3. `E-HoldoutL3` (untouched L3 validation remainder)

`evaluation/task7e_holdout_manifest.json`:

| | Value |
|---|---|
| Source | full v0.2 **val** L3 population: **936** records (above 224, below 213, left 250, right 249) — matches the expected totals exactly |
| Exclusion | Z-MiniVal240 (240 ids) ∪ Z-PairedVal20 members (40 ids) = **267** unique ids |
| `E-HoldoutL3` | **669** records — left 184, right 181, above 154, below 150 (no subsampling) |
| Overlap checks | Z-MiniVal240 **0**, Z-PairedVal20 **0**, test records **0**; ≥600 ✓, every program ≥120 ✓ |
| `E-PairedHoldout` | **20** pairs chosen from **175** candidates by SHA256 stable key; same tile, same oracle largest reference, different direction programs, different targets; overlap with Task 6Z Z-PairedVal20 members **0** |

## 4. E0 — exact Task 7D reproduction (before any holdout evaluation)

| Decoder | MiniVal mIoU | Expected | Δ | Paired | Expected |
|---|---:|---:|---:|---:|---:|
| Z-B3 | 0.3242128982543474 | 0.3242128982543474 | **0.0** | 15/20 | 15/20 |
| D-B1 | 0.3978996298363562 | 0.3978996298363562 | **0.0** | 19/20 | 19/20 |

→ `TASK7D_REPRODUCTION_PASS` (tolerance 1e-6).

## 5. E1 — oracle-reference held-out audit (E-O0 vs E-O1)

| Decoder | Records | mIoU | Dice | Pr@0.5 | Params |
|---|---:|---:|---:|---:|---:|
| E-O0 frozen Z-B3 | 669 | 0.314111 | 0.424719 | 0.516799 | 275,777 |
| **E-O1 frozen D-B1** | 669 | **0.385496** | **0.510648** | **0.547313** | 278,081 |

Delta: **+0.071384** mIoU, +0.085929 Dice, +0.030514 Pr@0.5. Per direction (records / Z-B3 / D-B1 / delta):
above 154 / 0.3493 / 0.4075 / **+0.0582**; below 150 / 0.2738 / 0.3506 / **+0.0768**; left 184 / 0.3314 /
0.3880 / **+0.0566**; right 181 / 0.3000 / 0.3931 / **+0.0931** → **4/4 directions improve**.

D-B1 per program: `largest_to_above_to_nearest` 0.4075, `largest_to_below_to_nearest` 0.3506,
`largest_to_left_of_to_nearest` 0.3880, `largest_to_right_of_to_nearest` 0.3931. Target-area quartiles
(q1→q4): 0.3368 / 0.4672 / 0.4275 / 0.3109; boundary-distance quartiles: 0.3326 / 0.4132 / 0.4158 / 0.3805.
Targets that touch the border or are tiny: **0** records (U-C1 eligibility excludes border-touching references
and the L3 targets are never tiny).

**Bootstrap** (deterministic seed 20261001, 2000 paired resamples by record id, 95 % percentile CI):
mean delta **+0.071384**, CI **[+0.058623, +0.084263]** — the lower bound is well above zero.

**E-PairedHoldout20** (oracle reference):

| Decoder | Pass | own IoU | cross IoU | margin |
|---|---:|---:|---:|---:|
| Z-B3 | 15/20 | 0.2592 | 0.0000 | +0.2592 |
| **D-B1** | **18/20** | 0.3193 | 0.0000 | **+0.3193** |

**Section 13 gate `DB1_HOLDOUT_GENERALIZES` = true (6/6)**: holdout mIoU 0.3855 ≥ 0.36 ✓; Δ +0.0714 ≥ +0.05 ✓;
4/4 directions ✓; CI lower +0.0586 > 0.00 ✓; paired 18/20 ≥ 16/20 ✓; margin +0.3193 ≥ 0.30 ✓.

## 6. E2 — predicted-reference holdout audit (E-P0 vs E-P1)

Frozen U-C1 as mandated by section 14 (the Task 6U `CONFIGS['U-C1']` entry actually used): YOLO26m-seg Task
6M.1 checkpoint `ef852b58…61f474`, **imgsz 640, conf 0.05, max_det 300**, default NMS, no TTA, no tiling;
largest eligibility = not border-touching and bbox extent ratio ≤ 0.20; selection = max predicted mask area →
tie higher confidence → lower original index. No ranker, no quality filter, no SAM2 refinement. *(Task 6Q's
read-only `config_report()` prints the resolver module defaults conf 0.10 / max_det 100; the pipeline uses the
frozen U-C1 entry, which is what this task specifies.)*

Reference diagnostics (GT offline only): reference mIoU **0.440647**, Dice **0.508456**, Pr@0.5 **0.501556**,
abstentions **2** (0.299 %), mean proposals 19.75 / mean eligible 14.70, best-eligible coverage@0.5 0.8460;
buckets **REFERENCE_OK 346**, **SELECTION_WRONG 220**, **NOT_COVERED 101**, **ABSTENTION 2**, GEOMETRY_POOR 0.

| Pipeline | Strict mIoU | Dice | Pr@0.5 | Answered | Answered-only mIoU | Reference-OK subset mIoU | Abstentions |
|---|---:|---:|---:|---:|---:|---:|---:|
| E-P0 Z-B3 | 0.206911 | 0.279792 | 0.427600 | 667/669 | 0.207542 | 0.328808 | 2 |
| **E-P1 D-B1** | **0.245405** | **0.324796** | **0.471116** | 667/669 | 0.246114 | **0.396365** | 2 |

Retention `pred_db1_strict / oracle_db1` = **0.636596** (oracle D-B1 0.385496 → predicted 0.245405);
E-P1 − E-P0 strict = **+0.038535** (answered-only +0.038651). Per direction E-P1: above 0.2356, below 0.2304,
left 0.2590, right 0.2523.

**E-PairedHoldout20** (reference resolved once per pair and reused for both members and both decoders):
Z-B3 **6/20** (margin +0.1251), D-B1 **6/20** (own 0.2592, cross 0.1340, margin **+0.1251**),
reference-abstention pairs 0.

**Section 19 gate `DB1_PREDICTED_REFERENCE_USABLE` = false (4/7)**: strict mIoU 0.2454 ≥ 0.24 ✓; answered-only
0.2461 < 0.25 ✗; Δ vs E-P0 +0.0385 ≥ +0.03 ✓; retention 0.6366 ≥ 0.62 ✓; paired 6/20 < 11/20 ✗; margin
+0.1251 < 0.20 ✗; abstention 0.299 % ≤ 10 % ✓.

Because section 19 did not pass, **section 20 (canonical parser integration) was not executed**:
`evaluation/task7e_canonical_parser_integration.json` records `executed: false` with the reason, and the Task 7C
parser was neither trained nor used to select the architecture.

## 7. Decision and verdict

| | Value |
|---|---|
| `DB1_HOLDOUT_GENERALIZES` | **true** |
| `DB1_PREDICTED_REFERENCE_USABLE` | **false** |
| `DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER` | **false** |
| Development baseline | **Z-B3 retained** (`z_b3_role` = frozen baseline / ablation) |
| D-B1 role | **positive oracle-reference ablation only**; neither checkpoint deleted or overwritten |

Section 22 priority: protocol clean, D-B1 hash exact, remainder sufficient, 20 held-out pairs, Task 7D
reproduced exactly, oracle generalization **passed** — and the predicted-reference gate fails →
**`DB1_PREDICTED_REFERENCE_BELOW_GATE`**.

Measured reading (reported only, no repair proposed): D-B1's advantage is **real and not a selection artifact**
— it survives on 669 untouched records with +0.0714 mIoU, a bootstrap CI of [+0.0586, +0.0843], improvement in
4/4 directions and 18/20 held-out pairs. Under the frozen U-C1 predicted reference the *ordering* survives
(E-P1 beats E-P0 by +0.0385, retention 0.637, reference-OK subset 0.3964 vs 0.3288) but the absolute level and
the counterfactual audit do not meet the section-19 development bars: answered-only mIoU 0.2461 (< 0.25),
predicted-reference paired 6/20 (< 11/20) and margin +0.1251 (< 0.20). The reference stage itself is the
dominant loss: only 346/669 references are REFERENCE_OK, 220 are SELECTION_WRONG and 101 NOT_COVERED, and the
same-reference pairing makes the own/cross audit much harder than in the oracle case (18/20 → 6/20).

## 8. Interpretation boundary

DSH reports measurements only. D-B1 is **not** claimed globally novel and is **not** called the final model; no
full training was run; the test split was not evaluated; parser/reference were not retrained; no learned
competition, graph or attention module was added; D-B1 was **not** modified after seeing holdout results; Task
7F was not chosen. Final recommendation exactly:

`等待 ChatGPT 根据 Task 7E 的 untouched L3 holdout 与 predicted-reference 结果决定是否正式冻结 D-B1 为开发版 L3 decoder，不自行进行全量训练、test 评估或新的架构改动。`

## 9. Reproduce

```text
python scripts/task7e_build_holdout.py                # E-HoldoutL3 + E-PairedHoldout manifest
python scripts/task7e_evaluate_oracle.py              # E0 reproduction + E1 oracle holdout + gate
python scripts/task7e_evaluate_predicted_reference.py # E2 predicted-reference audit (only if the gate passes)
python scripts/task7e_parser_integration.py           # section 20, conditional on section 19
python scripts/task7e_report.py                       # decision + verdict
```

Local expanded holdout rows and caches live in the gitignored `artifacts/task7e/`; checkpoints and SAM2
features are reused read-only. Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the report step
also runs in `.conda/buildreasonseg-mvp`.
