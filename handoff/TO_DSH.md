# TO_DSH — Task 7F: Reference Bottleneck Ceiling Decomposition for Frozen D-B1

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `f9c6e876f7ba2be33f5dff04a182a06813edf580`
>
> Predecessor: Task 7E → `DB1_PREDICTED_REFERENCE_BELOW_GATE`
>
> Research decision already made by ChatGPT:
>
> 1. Accept Task 7E as a clean result:
>    - frozen D-B1 generalizes on untouched oracle-reference L3 holdout;
>    - D-B1 is genuinely stronger than Z-B3 under the same correct reference;
>    - practical end-to-end performance remains below gate because the frozen U-C1 reference stage corrupts the relation state.
> 2. Do NOT retrain or modify D-B1.
> 3. Do NOT reopen reference hardening blindly.
> 4. Task 7F is a **ceiling decomposition only**. It quantifies how much of the remaining end-to-end loss comes from:
>    - wrong proposal selection;
>    - proposal-mask geometry;
>    - proposal coverage.
> 5. All GT-assisted reference modes in Task 7F are diagnostic upper bounds only and are never production inference.
> 6. No model training is permitted.
> 7. DSH is an executor. Do not invent a repair or choose Task 7G.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- reuse exact Task 7E E-HoldoutL3 and E-PairedHoldout20;
- reuse frozen U-C1 proposal inference;
- reuse frozen D-B1;
- construct the exact four reference modes below;
- compute reference/downstream ceilings and causal gap decomposition;
- solve ordinary runtime/integration bugs without changing the protocol.

DSH MUST NOT:
- train any model;
- change D-B1;
- change Z-B3;
- change U-C1;
- change YOLO weights;
- change proposal eligibility;
- change relation fields;
- retrain parser/reference;
- add ranker/filter/refinement;
- add attention/graph/Transformer;
- access test;
- use Task 7F diagnostics as hidden tuning data for a new model;
- choose the next algorithm.

If a prohibited change is required, STOP and report.

---

# PART A — Freeze Task 7E evidence

## 1. Record Task 7E result

Copy into `docs/task7f_reference_ceiling_decomposition.md`:

Task 7E verdict:
`DB1_PREDICTED_REFERENCE_BELOW_GATE`

Untouched E-HoldoutL3:
- records = 669
- four L3 programs only
- zero overlap with Task 6Z MiniVal/Paired
- zero test.

Oracle-reference:
- Z-B3 mIoU = `0.3141113773`
- D-B1 mIoU = `0.3854957053`
- delta = `+0.0713843280`
- bootstrap 95% CI = `[+0.0586233648, +0.0842626436]`
- D-B1 held-out paired = `18/20`
- D-B1 margin = `+0.3193402994`
- `DB1_HOLDOUT_GENERALIZES = true`.

Predicted U-C1 reference:
- reference mIoU = `0.4406471173`
- reference Pr@0.5 = `0.5015555235`
- `REFERENCE_OK = 346`
- `SELECTION_WRONG = 220`
- `NOT_COVERED = 101`
- abstentions = 2
- D-B1 strict mIoU = `0.2454050104`
- answered-only = `0.2461408575`
- D-B1 - Z-B3 strict = `+0.0385350024`
- retention = `0.6365959646`
- paired = `6/20`
- margin = `+0.1250628819`.

Interpretation:
- D-B1 architecture gain is real;
- practical chain is reference-limited;
- Task 7F decomposes that reference limitation without training.

Do not modify Task 7E artifacts.

---

# PART B — Frozen assets

## 2. D-B1

Use exact frozen checkpoint:

`artifacts/checkpoints/task7d/db1_minitrain1200.pt`

SHA256:
`6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0`

No retraining.

## 3. U-C1 proposal source

Use exactly:

```text
YOLO26m-seg Task 6M.1
checkpoint SHA256 =
ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474

imgsz = 640
conf = 0.05
max_det = 300
default NMS
no TTA
no tiling
source image = 512×512
```

Largest eligibility exactly:
- not border-touching;
- bbox extent ratio <= 0.20.

No other candidate filter.

## 4. Frozen fields/visual path

Read-only:
- GeometricRelationField v0.2
- NearestBoundaryField v0.1
- SAM2.1 Hiera Base+ frozen visual path/cache.

No test.

---

# PART C — Exact evaluation population

## 5. Reuse Task 7E packs byte-for-byte

Use:
- `E-HoldoutL3` exactly 669 records;
- `E-PairedHoldout` exactly 20 pairs.

Read IDs/hashes from:
`evaluation/task7e_holdout_manifest.json`

Verify:
- record-id hash exact;
- pair-id hash exact;
- no test;
- no overlap with Task 6Z MiniVal/Paired.

If mismatch:
STOP:
`TASK7E_HOLDOUT_MISMATCH`

No new sampling.

---

# PART D — Four exact reference modes

For every E-HoldoutL3 record, generate the same frozen U-C1 proposal set once and reuse it for F-R0/F-R1/F-R2.

The canonical GT reference is available only for offline diagnostic construction/evaluation.

## 6. F-R0 — CURRENT_SELECTED_PRED_MASK

Production-like frozen baseline:

```text
U-C1 eligible proposals
→ select maximum predicted mask area
→ tie higher YOLO confidence
→ lower original proposal index
→ use selected predicted mask as reference
```

If no eligible proposal:
- abstain.

This must reproduce Task 7E predicted-reference D-B1 numerics within tolerance.

## 7. F-R1 — ORACLE_SELECTED_PRED_MASK

Diagnostic selection ceiling.

Among the exact same U-C1 eligible proposals:

```text
choose proposal with maximum IoU to canonical GT reference mask
```

Tie-break:
1. higher IoU;
2. higher YOLO confidence;
3. lower original proposal index.

Use the **predicted proposal mask itself** as reference.

If there are no eligible proposals:
- abstain.

GT is used only to choose the proposal for this diagnostic ceiling.

This mode isolates:
> how much could improve if candidate selection were perfect while proposal coverage and proposal mask geometry stayed unchanged.

## 8. F-R2 — COVERAGE_CONDITIONAL_GT_MASK

Diagnostic geometry/selection-removed ceiling under existing proposal coverage.

Find the best eligible proposal exactly as F-R1.

If:
`best eligible proposal IoU with GT reference >= 0.50`

then use:
`canonical GT reference mask`

as the reference mask.

If:
- no eligible proposal, OR
- best eligible IoU < 0.50

then:
- abstain.

This mode intentionally uses GT mask only on records where U-C1 already covers the true reference at IoU>=0.50.

It isolates:
> what the downstream decoder could do if selection and reference-mask geometry were perfect, while the current U-C1 coverage limitation remained.

## 9. F-R3 — FULL_ORACLE_GT_MASK

Use canonical GT reference mask for every record.

This must reproduce the Task 7E oracle-reference D-B1 holdout result within tolerance.

This is the full reference ceiling.

---

# PART E — Reference-mode audit

## 10. Reference metrics

For F-R0/F-R1/F-R2/F-R3 report:

- answered records;
- abstentions;
- reference mIoU;
- Dice;
- Pr@0.5;
- centroid median/p90;
- per direction.

For F-R0/F-R1 also:
- selected proposal confidence;
- selected predicted area;
- selected-vs-best proposal IoU gap.

For F-R1 report:
- best eligible coverage@0.25 / 0.50 / 0.75.

For F-R2:
- covered records;
- uncovered records;
- coverage rate.

Write:
`evaluation/task7f_reference_modes.json`

---

# PART F — Frozen D-B1 downstream evaluation

## 11. Holdout all-record evaluation

For each mode F-R0...F-R3:

```text
canonical L3 program
→ mode-specific reference mask
→ P_dir v0.2
→ P_near v0.1
→ frozen SAM2 feature
→ frozen D-B1
→ target mask
```

If mode abstains:
- strict IoU/Dice = 0.

Report:
- strict mIoU/Dice;
- answered-only mIoU/Dice;
- Pr@0.5;
- per direction;
- reference-OK-style subset where applicable.

Write:
`evaluation/task7f_downstream_reference_modes.json`

## 12. Required reproduction checks

F-R0 must reproduce Task 7E predicted D-B1:
- strict mIoU `0.24540501038500215`
- answered-only `0.24614085749260334`
- abstentions `2`

Tolerance:
- floats <= `1e-6`
- counts exact.

F-R3 must reproduce Task 7E oracle D-B1:
- mIoU `0.38549570532647004`

Tolerance <= `1e-6`.

If either fails:
STOP:
`TASK7E_NUMERIC_REPRODUCTION_FAIL`

---

# PART G — Covered-subset geometry decomposition

## 13. Define COVERED50 subset

Records where:
- at least one U-C1 eligible proposal exists;
- best eligible proposal IoU with GT reference >= 0.50.

This is exactly the non-abstaining population of F-R2.

On this SAME subset compare:

### G-PRED
D-B1 using F-R1 best predicted proposal mask.

### G-GT
D-B1 using canonical GT reference mask.

Report:
- records;
- G-PRED mIoU/Dice;
- G-GT mIoU/Dice;
- delta `G-GT - G-PRED`;
- paired diagnostics restricted where both pair members' shared reference is COVERED50.

Write inside:
`evaluation/task7f_gap_decomposition.json`

This is the primary **proposal-mask geometry gap**.

---

# PART H — Paired counterfactual ceilings

## 14. E-PairedHoldout20

For each pair and each F-R0...F-R3:
- same tile/reference;
- resolve/build reference once;
- reuse for both direction programs.

Report:
- pass /20;
- own IoU;
- cross IoU;
- margin;
- abstention pairs.

Write:
`evaluation/task7f_paired_reference_modes.json`

No new pairs.

---

# PART I — Quantitative gap decomposition

## 15. Compute exact gaps

Let all be strict all-record D-B1 mIoU unless specified.

```text
M0 = F-R0
M1 = F-R1
M2 = F-R2
M3 = F-R3
```

Report:

### Selection ceiling gain
```text
selection_gain = M1 - M0
```

### Geometry gain on COVERED50
```text
geometry_gain_covered = G_GT - G_PRED
```

### Coverage ceiling gain
Because F-R2 and F-R3 both use the same GT reference mask whenever covered:

```text
coverage_gain = M3 - M2
```

### Remaining current-to-oracle gap
```text
total_reference_gap = M3 - M0
```

Also report percentages:

```text
selection_fraction = max(selection_gain,0) / total_reference_gap
coverage_fraction = max(coverage_gain,0) / total_reference_gap
```

`geometry_gain_covered` is reported separately because its denominator/subset differs.

Do not force the components to sum to 100%.

Write:
`evaluation/task7f_gap_decomposition.json`

---

# PART J — Predeclared diagnostic classifications

These are diagnostic labels, not new architecture verdicts.

## 16. `SELECTION_IS_ACTIONABLE`

true iff ALL:
- selection_gain >= `+0.05`
- F-R1 strict mIoU >= `0.29`
- F-R1 paired >= `10/20`
- F-R1 margin >= `0.18`.

## 17. `PROPOSAL_GEOMETRY_IS_MAJOR`

true iff:
- geometry_gain_covered >= `+0.05`.

## 18. `PROPOSAL_COVERAGE_IS_MAJOR`

true iff:
- coverage_gain >= `+0.04`
OR
- F-R2 coverage rate < `0.85`.

## 19. `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING`

true iff ALL:
- F-R2 strict mIoU >= `0.30`
- F-R2 paired >= `12/20`
- F-R2 margin >= `0.22`.

Interpretation:
- if true, existing U-C1 candidate coverage is theoretically sufficient for a materially better practical chain if selection/mask quality can be solved;
- if false, proposal coverage itself prevents a strong chain even under perfect selection/geometry on covered cases.

No thresholds may be changed after results.

---

# PART K — Verdict

## 20. Exactly one formal verdict

Priority:

1. `INVALID_EXPERIMENT`
   - training, test use, module mutation, GT used outside declared diagnostic modes, protocol violation.

2. `TASK7E_HOLDOUT_MISMATCH`

3. `TASK7E_NUMERIC_REPRODUCTION_FAIL`

4. `REFERENCE_PROPOSAL_CEILING_INSUFFICIENT`
   - `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING=false`

5. `REFERENCE_SELECTION_DOMINANT`
   - usable ceiling true
   - `SELECTION_IS_ACTIONABLE=true`
   - and selection_gain >= coverage_gain.

6. `REFERENCE_COVERAGE_OR_GEOMETRY_DOMINANT`
   - usable ceiling true
   - selection dominant condition false
   - and (`PROPOSAL_COVERAGE_IS_MAJOR=true` OR `PROPOSAL_GEOMETRY_IS_MAJOR=true`).

7. `REFERENCE_MIXED_BOTTLENECK`
   - usable ceiling true
   - none of verdicts 5/6 uniquely applies.

No other verdict.

The verdict does NOT authorize an automatic repair.

---

# PART L — Architecture status

## 21. D-B1 status after Task 7F

Regardless of reference verdict:

If all Task 7E oracle holdout facts reproduce, record:

> `D-B1 = preferred oracle-reference L3 target decoder candidate`

because its untouched oracle-reference gain over Z-B3 is already established.

Do NOT call it:
- end-to-end ready;
- final model;
- paper-final architecture.

Practical development chain remains blocked until ChatGPT decides what to do with the reference bottleneck.

---

# PART M — Interpretation boundary

DSH reports measurements only.

Do NOT:
- train a new selector;
- retrain YOLO;
- change U-C1;
- retrain D-B1;
- propose threshold changes;
- start full training/test;
- choose a repair based on the diagnostic labels.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7F 的 selection / proposal-geometry / coverage ceiling 分解决定是否值得进行最后一次 reference 干预；不自行训练 selector、重训 YOLO 或开始正式 test。`

---

# PART N — Required artifacts

Create:

```text
evaluation/task7f_reference_modes.json
evaluation/task7f_downstream_reference_modes.json
evaluation/task7f_gap_decomposition.json
evaluation/task7f_paired_reference_modes.json
evaluation/task7f_verdict.json

docs/task7f_reference_ceiling_decomposition.md

scripts/task7f_reference_modes.py
scripts/task7f_evaluate_downstream.py
scripts/task7f_gap_decomposition.py
scripts/task7f_report.py
```

Optional helper:
`buildreasonseg_mvp/task7f_reference_ceiling.py`

Local caches:
`artifacts/task7f/`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No checkpoint creation.

---

# PART O — Tests

Task 7E ended at:
`1285 passed, 1 skipped`

Add tests for at least:

1. Task 7E artifacts unchanged
2. E-Holdout record-id hash exact
3. E-Paired pair-id hash exact
4. D-B1 checkpoint hash exact
5. U-C1 exact config
6. no training
7. no test split
8. same proposal set reused F-R0/F-R1/F-R2
9. F-R0 deterministic max-area exact
10. F-R1 maximum GT-IoU proposal exact
11. F-R1 tie-break exact
12. F-R1 uses predicted mask, not GT mask
13. F-R2 coverage threshold exactly IoU>=0.50
14. F-R2 uses GT mask only when covered
15. F-R2 abstains when uncovered
16. F-R3 always uses GT reference
17. GT never enters D-B1 except as declared reference in R2/R3
18. GT target never enters inference
19. F-R0 numeric reproduction exact
20. F-R3 numeric reproduction exact
21. COVERED50 definition exact
22. G-PRED/G-GT same subset exact
23. selection_gain formula exact
24. geometry_gain_covered formula exact
25. coverage_gain formula exact
26. total_reference_gap exact
27. pair reference reused within pair
28. no ranker/quality/SAM refinement
29. fields unchanged
30. SAM2 unchanged
31. D-B1 unchanged
32. no parser training/use for architecture decision
33. no attention/graph/Transformer
34. no GRCL
35. no new dataset/download/install/GUI
36. diagnostic thresholds exact
37. verdict priority exact
38. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# PART P — Git/storage

Do not commit:
- model checkpoints;
- proposal/feature caches;
- source imagery/vectors;
- large per-record local cache;
- `.conda`.

Commit:
- small evaluation JSON;
- diagnostic code/scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `eval: decompose D-B1 reference bottleneck ceilings`
2. `docs: record reference ceiling attribution`

---

# PART Q — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for genuine runtime/integration bugs.

No installs/downloads.

---

# PART R — STOP

After Task 7F:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- train anything;
- choose or implement a reference repair;
- run test;
- start formal full training.

Wait for ChatGPT audit.
