# Task 7F — Reference Bottleneck Ceiling Decomposition for Frozen D-B1

> Task: `handoff/TO_DSH.md` (Task 7F) · Base commit: `f9c6e87` · Predecessor: Task 7E →
> `DB1_PREDICTED_REFERENCE_BELOW_GATE`
> **Verdict: `REFERENCE_SELECTION_DOMINANT`** · selection +0.1195 / coverage +0.0543 / geometry +0.0043
> Ceiling decomposition only — **no model was trained**, nothing was tuned, no repair was implemented
> Tests: `tests/test_task7f_reference_ceiling.py` · Evidence: `evaluation/task7f_*.json`

Task 7F quantifies how much of the remaining end-to-end loss of the frozen D-B1 L3 decoder comes from **wrong
proposal selection**, **proposal-mask geometry** and **proposal coverage**, using four exact reference modes over
the frozen Task 7E populations. All GT-assisted modes are diagnostic upper bounds and are never production
inference.

## 1. Recorded Task 7E evidence (frozen, not modified)

Verdict `DB1_PREDICTED_REFERENCE_BELOW_GATE`. Untouched E-HoldoutL3: 669 records, four L3 programs only, zero
overlap with the Task 6Z MiniVal/Paired packs, zero test.

Oracle reference: Z-B3 mIoU `0.3141113773`, D-B1 `0.3854957053`, delta `+0.0713843280`, bootstrap 95 % CI
`[+0.0586233648, +0.0842626436]`, D-B1 held-out paired `18/20`, margin `+0.3193402994`,
`DB1_HOLDOUT_GENERALIZES = true`.

Predicted U-C1 reference: reference mIoU `0.4406471173`, Pr@0.5 `0.5015555235`, `REFERENCE_OK = 346`,
`SELECTION_WRONG = 220`, `NOT_COVERED = 101`, abstentions `2`, D-B1 strict mIoU `0.2454050104`, answered-only
`0.2461408575`, D-B1 − Z-B3 strict `+0.0385350024`, retention `0.6365959646`, paired `6/20`, margin
`+0.1250628819`.

Interpretation: the D-B1 architecture gain is real; the practical chain is reference-limited. Task 7F
decomposes that limitation without training.

## 2. Frozen assets and population

D-B1 `artifacts/checkpoints/task7d/db1_minitrain1200.pt` (SHA256 `6df31909…21a89c0`), frozen U-C1
(YOLO26m-seg Task 6M.1 `ef852b58…61f474`, imgsz 640, conf 0.05, max_det 300, default NMS, no TTA, no tiling,
source image 512×512; eligibility = not border-touching and bbox extent ratio ≤ 0.20, no other filter),
read-only GeometricRelationField v0.2 / NearestBoundaryField v0.1 / frozen SAM2.1 Hiera Base+ features.

Population reused byte-for-byte from `evaluation/task7e_holdout_manifest.json`: `E-HoldoutL3` **669** records
and `E-PairedHoldout` **20** pairs, with the record-id hash and pair-id hash re-verified (plus zero test and zero
Task 6Z MiniVal/Paired overlap). No new sampling.

## 3. The four exact reference modes

One frozen U-C1 proposal inference per tile is reused by every mode.

| Mode | Construction |
|---|---|
| **F-R0** `CURRENT_SELECTED_PRED_MASK` | eligible proposals → max predicted mask area → tie higher YOLO confidence → lower original index → that predicted mask (abstain if no eligible proposal) |
| **F-R1** `ORACLE_SELECTED_PRED_MASK` | among the same eligible proposals → max IoU to the canonical GT reference → tie higher confidence → lower index → that **predicted** proposal mask |
| **F-R2** `COVERAGE_CONDITIONAL_GT_MASK` | best eligible IoU ≥ 0.50 → canonical GT mask; no eligible proposal or best IoU < 0.50 → abstain |
| **F-R3** `FULL_ORACLE_GT_MASK` | canonical GT mask for every record |

GT therefore enters only as the declared diagnostic reference of F-R2/F-R3 and to *choose* the F-R1 proposal
(which still returns a predicted mask).

## 4. Reference-mode audit (669 records)

| Mode | Answered | Abstentions | Ref mIoU | Dice | Pr@0.5 | Centroid err median / p90 |
|---|---:|---:|---:|---:|---:|---:|
| F-R0 | 667 | 2 | 0.4406 | 0.5085 | 0.5016 | 19.114 / 247.748 |
| **F-R1** | 667 | 2 | **0.7119** | **0.8111** | **0.8710** | 4.195 / 25.916 |
| F-R2 | 566 | 103 | 1.0000 | 1.0000 | 1.0000 | 0.000 / 0.000 |
| F-R3 | 669 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.000 / 0.000 |

F-R0 extras: mean selected confidence `0.3577`, mean selected area `5350.76` px, mean selected-vs-best proposal
IoU `0.5591`, mean reference-IoU gap to the best eligible proposal `0.2713`. F-R1 best-eligible coverage at
IoU ≥ 0.25 / 0.50 / 0.75 is `641/669 = 0.9581`, `566/669 = 0.8460`, `392/669 = 0.5860`. F-R2: covered **566**,
uncovered **103**, coverage rate **0.8460** at the frozen 0.50 threshold.

## 5. Frozen D-B1 downstream per mode (strict all-record mIoU)

| Mode | Strict mIoU | Dice | Pr@0.5 | Answered | Answered-only mIoU |
|---|---:|---:|---:|---:|---:|
| F-R0 | **0.245405** | 0.324762 | 0.471130 | 667/669 | 0.246141 |
| F-R1 | **0.364909** | 0.484050 | 0.538483 | 667/669 | 0.366004 |
| F-R2 | **0.331180** | 0.436579 | 0.462272 | 566/669 | 0.391447 |
| F-R3 | **0.385496** | 0.510649 | 0.547313 | 669/669 | 0.385496 |

Reproduction: **F-R0 Δ 0.0** against Task 7E predicted D-B1 (`0.24540501038500215`, answered-only
`0.24614085749260334`, abstentions `2`, all exact) and **F-R3 Δ 2.27e-07** against Task 7E oracle D-B1
(`0.38549570532647004`) — both inside the 1e-6 tolerance → `TASK7E_NUMERIC_REPRODUCTION_PASS`.

## 6. Covered-subset geometry decomposition (COVERED50 = 566 records)

COVERED50 = at least one eligible U-C1 proposal **and** best eligible IoU ≥ 0.50 — exactly the non-abstaining
population of F-R2. On that same subset: **G-PRED** (F-R1 predicted proposal mask) mIoU `0.3872` / Dice
`0.5103` / Pr@0.5 `0.5432`; **G-GT** (canonical GT mask) mIoU `0.3914` / Dice `0.5160` / Pr@0.5 `0.5464`.
`geometry_gain_covered = G-GT − G-PRED = **+0.0043**`, i.e. proposal-mask geometry is **not** the bottleneck
wherever the proposal set already covers the reference. Paired diagnostics restricted to the 17 pairs whose
shared reference is COVERED50: F-R1 `16/17` (margin +0.3315), F-R3 `16/17` (margin +0.3306).

## 7. Exact gap decomposition and diagnostic labels

```text
M0 = F-R0 = 0.245405    M1 = F-R1 = 0.364909
M2 = F-R2 = 0.331180    M3 = F-R3 = 0.385496

selection_gain       = M1 - M0 = +0.119504     selection_fraction = 0.8530
coverage_gain        = M3 - M2 = +0.054316     coverage_fraction  = 0.3877
geometry_gain_covered = G-GT - G-PRED = +0.004253   (different subset/denominator, reported separately)
total_reference_gap  = M3 - M0 = +0.140091
```

Components are deliberately not forced to sum to 100 %.

| Predeclared label | Conditions (measured) | Value |
|---|---|---|
| `SELECTION_IS_ACTIONABLE` | selection_gain +0.1195 ≥ 0.05 ✓ · F-R1 mIoU 0.3649 ≥ 0.29 ✓ · F-R1 paired 18/20 ≥ 10 ✓ · F-R1 margin +0.3005 ≥ 0.18 ✓ | **true** |
| `PROPOSAL_GEOMETRY_IS_MAJOR` | geometry_gain_covered +0.0043 ≥ 0.05 ✗ | **false** |
| `PROPOSAL_COVERAGE_IS_MAJOR` | coverage_gain +0.0543 ≥ 0.04 ✓ **or** F-R2 coverage rate 0.8460 < 0.85 ✓ | **true** |
| `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING` | F-R2 strict 0.3312 ≥ 0.30 ✓ · F-R2 paired 16/20 ≥ 12 ✓ · F-R2 margin +0.3306 ≥ 0.22 ✓ | **true** |

Paired counterfactual ceilings (E-PairedHoldout20, reference built once per pair and reused):

| Mode | Passed | answered pairs | abstention pairs | own IoU | cross IoU | margin |
|---|---:|---:|---:|---:|---:|---:|
| F-R0 | 6/20 | 20 | 0 | 0.1560 | 0.0309 | +0.1251 |
| **F-R1** | **18/20** | 20 | 0 | 0.3005 | 0.0000 | **+0.3005** |
| F-R2 | 16/20 | 17 | 3 | 0.3306 | 0.0000 | +0.3306 |
| F-R3 | 18/20 | 20 | 0 | 0.3193 | 0.0000 | +0.3193 |

## 8. Verdict

Section 20 priority: protocol clean, holdout hashes exact, both reproductions pass, the usable-ceiling label is
**true**; `SELECTION_IS_ACTIONABLE` is **true** and `selection_gain (+0.1195) ≥ coverage_gain (+0.0543)` →
**`REFERENCE_SELECTION_DOMINANT`**.

The verdict authorizes no automatic repair. Measured reading (reported only): with the frozen U-C1 proposal set
held fixed, repairing *selection* is worth about **2.2×** more strict mIoU than repairing *coverage*
(+0.1195 vs +0.0543 out of a +0.1401 total gap), and mask-geometry refinement is worth almost nothing on records
that are already covered (+0.0043). The existing candidate coverage is theoretically usable (F-R2 0.3312 strict
with 16/20 paired and +0.3306 margin), so the reference bottleneck is dominated by *which* eligible proposal is
chosen — F-R0 picks a mask that agrees with the best eligible candidate at only 0.5591 IoU and leaves a mean
0.2713 reference-IoU on the table — while coverage adds a secondary contribution (103/669 records have no
eligible proposal at IoU ≥ 0.50).

## 9. D-B1 architecture status

All Task 7E oracle holdout facts reproduce, so D-B1 is recorded as the
**`preferred oracle-reference L3 target decoder candidate`** (untouched oracle-reference gain over Z-B3:
+0.0714 mIoU, CI [+0.0586, +0.0843], 4/4 directions, 18/20 pairs, margin +0.3193). It is explicitly **not**
end-to-end ready, **not** the final model and **not** paper-final; Z-B3 remains the frozen baseline/ablation and
the practical development chain stays blocked until ChatGPT decides what to do with the reference bottleneck.

## 10. Interpretation boundary

DSH reports measurements only. No selector was trained; YOLO was not retrained; U-C1 was not changed; D-B1 was
not retrained; no threshold change is proposed; no full training or test was started; no repair was chosen from
the diagnostic labels; Task 7G was not chosen. Final recommendation exactly:

`等待 ChatGPT 根据 Task 7F 的 selection / proposal-geometry / coverage ceiling 分解决定是否值得进行最后一次 reference 干预；不自行训练 selector、重训 YOLO 或开始正式 test。`

## 11. Reproduce

```text
python scripts/task7f_reference_modes.py            # holdout verification, four modes, paired ceilings, cache
python scripts/task7f_evaluate_downstream.py        # frozen D-B1 per mode + Task 7E reproductions
python scripts/task7f_gap_decomposition.py          # COVERED50 geometry, gaps and labels
python scripts/task7f_report.py                     # verdict + D-B1 status
```

Per-record measurement caches live in the gitignored `artifacts/task7f/`; checkpoints, proposal caches and SAM2
features are reused read-only. Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the downstream,
gap and report steps also run in `.conda/buildreasonseg-mvp`.
