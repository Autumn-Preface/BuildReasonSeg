# Task 6V — Family-Conditioned Reference Resolver Policy

> Task: `handoff/TO_DSH.md` (Task 6V) · Base commit: `b8b5224` · Predecessor: Task 6U →
> `REFERENCE_RANKER_NOT_HELPFUL`
> **Verdict: `FAMILY_POLICY_NOT_BETTER`**
> Frozen policy: **`{"largest": "V-P2", "smallest": "V-P0"}`** · **No model was trained in Task 6V**
> Tests: `tests/test_task6v_family_reference_resolver.py` · Evidence: `evaluation/task6v_*.json`

Task 6V tests whether a **family-conditioned policy over already-frozen resolver options** can recover
the useful part of Task 6U at zero new model cost. The proposal configurations, the Task 6Q deterministic
selector, the ProposalSetRanker v0.1 checkpoint, the hardened ProgramHead, field v0.2, SAM2 and B3 are all
frozen read-only assets.

## 1. Recorded Task 6U findings (copied for the record; Task 6U artifacts not mutated)

**Candidate coverage** — U-C1 vs U-C0 on untouched RefValUnique:

| eligible coverage@0.50 | U-C0 | U-C1 |
|---|---|---|
| overall | 0.68950 | 0.75342 |
| largest | 0.83636 | 0.87273 |
| smallest | 0.54128 | 0.63303 |

Candidate recall improved.

**Selector behaviour by family** — U-S0 = U-C0 + deterministic area selector; U-S1 = U-C1 + deterministic;
U-S2 = U-C1 + ProposalSetRanker v0.1, on RefValUnique:

| largest | mIoU | Pr@0.5 |
|---|---|---|
| U-S0 | 0.54284 | 0.65138 |
| U-S1 | 0.51830 | 0.60000 |
| U-S2 | 0.56804 | 0.66364 |

| smallest | mIoU | Pr@0.5 |
|---|---|---|
| U-S0 | 0.30109 | 0.39423 |
| U-S1 | 0.33532 | 0.43810 |
| U-S2 | 0.04105 | 0.03810 |

Interpretation boundary: the shared v0.1 ranker is not accepted globally; its family-specific usefulness
must be selected using **train-only calibration**, not RefVal.

## 2. Frozen assets (Part B)

Verified exactly before selection: YOLO26m-seg Task 6M.1 checkpoint
`ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` (not retrained), ProposalSetRanker
v0.1 `c738fcf77419626f5f9bde21ff8cd477547418c095537b4bf50f1d6c1b1a46c0` (not retrained), Task 6T hardened
ProgramHead `4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e` (not retrained), Task 6O B3
`7556e4a4862b75d47e61b5c6391c2a05689d5bde3e74586aed1d94a1c3c7d6ab` (not retrained), plus the
byte-identical frozen packs (MiniVal240 / PairedVal20 manifest hashes). No test split.

## 3. Exactly three options and the train-only selection (Parts C-D)

`buildreasonseg_mvp/task6v_family_reference_resolver.py` + `evaluation/task6v_calibration_family_policy.json`.
Only the three predeclared options exist; no fourth option and no new learned logic.

Per-family metrics on the frozen **train-only** U-Calib200 (100 largest + 100 smallest):

| family | option | Pr@0.5 | mIoU | SELECTION_WRONG | REFERENCE_OK | NOT_COVERED | centroid median | centroid p90 | abstain |
|---|---|---|---|---|---|---|---|---|---|
| largest | V-P0 | 0.7600 | 0.6338 | 22 | 76 | 2 | 0.0047 | 0.2508 | 0.0000 |
| largest | V-P1 | 0.7100 | 0.6034 | 27 | 71 | 2 | 0.0059 | 0.2806 | 0.0000 |
| **largest** | **V-P2** | **0.8000** | **0.6650** | **18** | **80** | 2 | 0.0044 | 0.1929 | 0.0000 |
| **smallest** | **V-P0** | **0.5300** | **0.4154** | **38** | **53** | 9 | 0.0044 | 0.3146 | 0.0000 |
| smallest | V-P1 | 0.4500 | 0.3545 | 49 | 45 | 6 | 0.0506 | 0.3033 | 0.0000 |
| smallest | V-P2 | 0.0300 | 0.0546 | 91 | 3 | 6 | 0.1599 | 0.3801 | 0.0000 |

Section 7 priority (Pr@0.5 → mIoU → fewer `SELECTION_WRONG` → lower centroid median → lower abstention →
simpler option) gives ranking `largest: [V-P2, V-P0, V-P1]`, `smallest: [V-P0, V-P1, V-P2]` →
`evaluation/task6v_frozen_family_policy.json` freezes **`{"largest": "V-P2", "smallest": "V-P0"}`**.

Note the calibration evidence **contradicts** Task 6U's RefVal observation for `smallest`: on train-only
U-Calib200 the deterministic selector prefers **U-C0** (V-P0) over U-C1 (V-P1), while on RefValUnique U-C1
was better (0.33532 vs 0.30109). Per the predeclared rule the train-only calibration choice is frozen and
was not revisited after seeing RefVal.

## 4. RefValUnique evaluation (Part E)

`evaluation/task6v_refval_family_policy.json` — frozen policy on the untouched 219-reference RefValUnique:

| Metric | Frozen family policy (V-P2/V-P0) | U-S0 | U-S1 | U-S2 | U-C1 oracle ceiling |
|---|---|---|---|---|---|
| selected-reference mIoU | **0.4383** | 0.4248 | 0.4289 | 0.3107 | 0.6203 |
| Dice | 0.4995 | 0.4832 | 0.4933 | 0.3564 | 0.7092 |
| Pr@0.5 | **0.5327** | 0.5258 | 0.5209 | 0.3581 | 0.7674 |
| centroid mean / median / p90 | 0.1132 / **0.0107** / 0.3845 | 0.0156 / 0.3935 | 0.0159 / 0.3803 | 0.0890 / 0.3771 | 0.0046 / 0.108 |
| area-ratio median | 1.0757 | 1.0813 | 1.0719 | 1.4695 | — |
| abstention rate | **0.0228** | 0.0274 | 0.0183 | 0.0183 | — |
| largest mIoU / Pr@0.5 | **0.5680 / 0.6636** | 0.5428 / 0.6514 | 0.5183 / 0.6000 | 0.5680 / 0.6636 | — |
| smallest mIoU / Pr@0.5 | 0.3011 / 0.3942 | 0.3011 / 0.3942 | **0.3353 / 0.4381** | 0.0411 / 0.0381 | — |

Failure buckets:

| Bucket | Frozen policy | U-S0 | U-S1 | U-S2 |
|---|---|---|---|---|
| `NO_PROPOSALS` | 2 | 3 | 1 | 1 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 3 | 3 | 3 |
| `REFERENCE_NOT_COVERED_IOU50` | 59 | 62 | **50** | **50** |
| `REFERENCE_SELECTION_WRONG` | **41** | 39 | 53 | 88 |
| `SELECTED_MASK_GEOMETRY_POOR` | 0 | 1 | 0 | 0 |
| `REFERENCE_OK` | 114 | 111 | **112** | 77 |

The routing works as intended (largest gets the ranker's better mIoU, smallest keeps the deterministic
selector's competence instead of the ranker's collapse), but because `smallest` is routed to **U-C0** rather
than U-C1 it gives back the candidate-coverage gain on that family: `NOT_COVERED` 50 → 59.

## 5. Downstream causal evaluation and integration (Parts F-G)

`evaluation/task6v_downstream_minival240.json` (canonical program ids, no parser):

| Metric | Frozen family policy | U-S0 | U-S1 | U-S2 |
|---|---|---|---|---|
| strict all-240 mIoU | 0.3005 | 0.2970 | **0.3089** | 0.2425 |
| answered-only mIoU | 0.3069 | 0.3046 | **0.3141** | 0.2466 |
| Dice | — (in artifact) | — | — | — |
| Pr@0.5 | in artifact | — | — | — |
| abstentions | 5 | 6 | **4** | 4 |
| reference-fail count | 116 | 117 | **115** | 158 |
| target-fail-with-reference-ok | 68 | 67 | — | — |
| largest / smallest target mIoU | 0.3393 / 0.2617 | — | — | — |
| per direction (left/right/above/below) | 0.3189 / 0.2365 / 0.3441 / 0.3025 | — | — | — |
| border target (n=114) | 0.2985 | 0.2943 | 0.2896 | 0.2666 |
| tiny target (n=4) | ≈0 | ≈0 | ≈0 | ≈0 |

`evaluation/task6v_downstream_pairedval20.json`: pass **10/20**, own 0.291106, cross 0.003251, margin
**+0.2879**, 0 abstention pairs (U-S0 10/20 +0.2737; U-S1 11/20 +0.3209; U-S2 11/20 +0.3332).

`evaluation/task6v_hardened_parser_integration.json` — Task 6T hardened ProgramHead → family-conditioned
resolver → field v0.2 → SAM2 → B3: parser **240/240**, strict 0.3005, answered-only 0.3069, abstentions 5,
reference-fail 116 (identical to the causal run; no nearest/L3 execution).

## 6. Predeclared flags, gates and verdict

Section 13 `family_policy_improved` versus the frozen Task 6U U-S1 (**false**):

| Criterion | Required | Measured | Pass |
|---|---|---|---|
| overall RefVal mIoU | ≥ U-S1 + 0.015 = 0.4439 | **0.4383** (Δ +0.0094) | ✗ |
| RefVal `REFERENCE_OK` | ≥ U-S1 + 5 = 117 | **114** | ✗ |
| RefVal `REFERENCE_SELECTION_WRONG` | ≤ U-S1 − 5 = 48 | **41** | ✓ |
| abstention rate | ≤ 0.05 | **0.0228** | ✓ |

Section 14 directional hardening gates: **3 of 10 pass**.

| # | Gate | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | RefVal mIoU | ≥ 0.45 | 0.4383 | ✗ |
| 2 | RefVal centroid median | ≤ 0.03 | **0.0107** | ✓ |
| 3 | RefVal centroid p90 | ≤ 0.32 | 0.3845 | ✗ |
| 4 | MiniVal answered-only mIoU | ≥ 0.325 | 0.3069 | ✗ |
| 5 | MiniVal strict mIoU | ≥ 0.305 | 0.3005 | ✗ |
| 6 | PairedVal | ≥ 12/20 | 10 | ✗ |
| 7 | own-cross margin | ≥ 0.30 | 0.2879 | ✗ |
| 8 | MiniVal reference-fail | ≤ 105 | 116 | ✗ |
| 9 | parser integration | 240/240 | **240/240** | ✓ |
| 10 | no test / no GT inference | yes | yes | ✓ |

Section 15 priority order: protocol clean, frozen assets available, the improvement flag fails **and** the
hardening gate fails → **`FAMILY_POLICY_NOT_BETTER`**.

## 7. Interpretation boundary (Part J)

DSH reports measurements only. Family routing is **not** claimed as a project novelty; the ranker was not
redesigned; no smallest-only ranker, proposal-quality classifier, candidate-config change, confidence
threshold, TTA/tiling, YOLO retraining or nearest/L3 work was started; Task 6W was not chosen. No model was
trained or fine-tuned in Task 6V.

## 8. Reproduce

```text
python scripts/task6v_select_family_policy.py --device 0     # Parts C-D (U-Calib200 only)
python scripts/task6v_evaluate_reference.py --device 0       # Part E
python scripts/task6v_evaluate_downstream.py --stage causal  # Parts F
python scripts/task6v_evaluate_downstream.py --stage parser  # Part G
python scripts/task6v_report.py                              # Parts H-I
```

Proposal caches are reused from `artifacts/task6u/proposals/<config>/` (gitignored); the ranker checkpoint
stays under the gitignored `artifacts/checkpoints/task6u/`. Run in `.conda/buildreasonseg-proposal` with
`HF_HUB_OFFLINE=1`.
