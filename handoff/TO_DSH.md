# TO_DSH — Task 6V: Family-Conditioned Reference Resolver Policy

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `b8b5224ec88d32af7d086e337016f6e7d2ada1a2`
>
> Predecessor: Task 6U → `REFERENCE_RANKER_NOT_HELPFUL`
>
> Research decision already made by ChatGPT:
>
> 1. Accept Task 6U's **U-C1** proposal configuration as a useful candidate-recall improvement:
>    - imgsz 640
>    - conf 0.05
>    - max_det 300
>    - no TTA / no tiling
> 2. Do **not** use ProposalSetRanker v0.1 globally: it catastrophically fails on `smallest`.
> 3. Task 6U revealed a strong family asymmetry:
>    - ranker helps `largest`;
>    - deterministic area selection is much better for `smallest`.
> 4. Before designing another reference model, Task 6V tests whether a **family-conditioned policy over already-frozen resolver options** can recover the useful part of Task 6U at essentially zero new model cost.
> 5. No new training is permitted in Task 6V.

DSH is an executor. Do not redesign the policy search.

---

# 0. DSH role

All user-facing DSH output must be Chinese.

DSH MAY:
- load existing frozen Task 6U proposal caches/checkpoints;
- evaluate the three predeclared resolver options on the train-only U-Calib200 set;
- freeze exactly one resolver option for `largest` and one for `smallest`;
- evaluate the frozen family-conditioned resolver on RefValUnique, MiniVal240 and PairedVal20;
- integrate the already-frozen Task 6T hardened ProgramHead for one final directional-chain check;
- solve ordinary runtime/integration bugs without changing the experiment.

DSH MUST NOT:
- train or fine-tune any model;
- retrain the ProposalSetRanker;
- retrain YOLO;
- change YOLO weights;
- introduce a fourth resolver option;
- change U-C0/U-C1 configuration;
- change Task 6Q eligibility thresholds;
- change the hardened ProgramHead;
- change GeometricRelationField v0.2;
- change SAM2;
- retrain B3;
- add GRCL;
- add nearest/L3 execution;
- access test split;
- choose Task 6W.

If a prohibited change is required, STOP and report.

---

# PART A — Frozen evidence and Task 6U interpretation

## 1. Record the Task 6U findings

Copy into `docs/task6v_family_conditioned_reference_resolver.md`:

Task 6U established:

### Candidate coverage
U-C1 vs U-C0 on untouched RefValUnique:
- overall eligible coverage@0.50:
  - 0.68950 → 0.75342
- largest:
  - 0.83636 → 0.87273
- smallest:
  - 0.54128 → 0.63303

Therefore candidate recall improved.

### Selector behavior by family

U-S0 = U-C0 + deterministic area selector

U-S1 = U-C1 + deterministic area selector

U-S2 = U-C1 + ProposalSetRanker v0.1

RefValUnique:

Largest:
- U-S0 mIoU 0.54284 / Pr@0.5 0.65138
- U-S1 mIoU 0.51830 / Pr@0.5 0.60000
- U-S2 mIoU 0.56804 / Pr@0.5 0.66364

Smallest:
- U-S0 mIoU 0.30109 / Pr@0.5 0.39423
- U-S1 mIoU 0.33532 / Pr@0.5 0.43810
- U-S2 mIoU 0.04105 / Pr@0.5 0.03810

Interpretation boundary:
- the shared v0.1 ranker is not accepted globally;
- its family-specific usefulness must be selected using **train-only calibration**, not RefVal.

Do not modify Task 6U artifacts or verdict.

---

# PART B — Frozen assets

## 2. Read-only

Freeze:

- all Task 6U artifacts;
- U-C0 proposal cache/config;
- U-C1 proposal cache/config;
- ProposalSetRanker v0.1 checkpoint:
  `artifacts/checkpoints/task6u/reference_ranker_v01.pt`
- Task 6T hardened ProgramHead checkpoint;
- Task 6Q deterministic selector rules;
- Task 6O B3 checkpoint;
- GeometricRelationField v0.2;
- SAM2 feature path/cache;
- U-Calib200;
- RefValUnique;
- MiniVal240;
- PairedVal20.

No test split.

## 3. Verify hashes

Require exact hashes recorded in previous artifacts for:

- YOLO26m-seg Task 6M.1 checkpoint;
- ProposalSetRanker v0.1 checkpoint;
- Task 6T hardened ProgramHead checkpoint;
- Task 6O B3 checkpoint where recorded;
- frozen pack hashes.

If ranker checkpoint is missing/mismatched:
STOP with `FROZEN_RESOLVER_ASSET_UNAVAILABLE`.

---

# PART C — Exactly three family-policy options

## 4. Resolver options

For each family independently, compare EXACTLY:

### V-P0
`U-C0 + deterministic Task 6Q area selector`

### V-P1
`U-C1 + deterministic Task 6Q area selector`

### V-P2
`U-C1 + frozen ProposalSetRanker v0.1`

No new option.

Do not retrain or recalibrate the ranker.

---

# PART D — Train-only policy selection

## 5. Use exact U-Calib200 only

Use the frozen Task 6U U-Calib200:
- 100 largest
- 100 smallest

This is the **only** set used to choose the family policies.

RefValUnique / MiniVal240 / PairedVal20 must not influence selection.

## 6. Per-family metrics on U-Calib200

For each family and each V-P0/P1/P2 report:

- selected-reference mIoU;
- Dice;
- Pr@0.5;
- centroid median;
- centroid p90;
- abstention rate;
- `REFERENCE_NOT_COVERED_IOU50`;
- `REFERENCE_SELECTION_WRONG`;
- `REFERENCE_OK`.

Write:
`evaluation/task6v_calibration_family_policy.json`

## 7. Freeze one option per family

For `largest` and `smallest` independently, rank options by this exact priority:

1. higher `Pr@0.5`
2. higher selected-reference mIoU
3. lower `REFERENCE_SELECTION_WRONG`
4. lower centroid median
5. lower abstention rate
6. simpler option priority:
   - V-P0
   - V-P1
   - V-P2

The simplicity tie-break means a learned ranker is used only when calibration evidence is actually better.

Write:

`evaluation/task6v_frozen_family_policy.json`

Example schema:

```json
{
  "largest": "V-P2",
  "smallest": "V-P1"
}
```

Do not change it after any RefVal result is observed.

---

# PART E — Untouched RefValUnique evaluation

## 8. Family-conditioned resolver

Create:

`buildreasonseg_mvp/task6v_family_reference_resolver.py`

It only dispatches:

```text
if family == largest:
    use frozen selected largest policy
elif family == smallest:
    use frozen selected smallest policy
```

No additional learned logic.

No GT at inference.

## 9. RefValUnique metrics

Evaluate frozen family policy on exact RefValUnique.

Report:

- overall selected-reference mIoU;
- Dice;
- Pr@0.5;
- centroid mean/median/p90;
- area ratio median;
- abstention rate;
- largest metrics;
- smallest metrics.

Failure buckets exactly:

1. `NO_PROPOSALS`
2. `NO_ELIGIBLE_PROPOSALS`
3. `REFERENCE_NOT_COVERED_IOU50`
4. `REFERENCE_SELECTION_WRONG`
5. `SELECTED_MASK_GEOMETRY_POOR`
6. `REFERENCE_OK`

Compare against:
- Task 6U U-S0
- Task 6U U-S1
- Task 6U U-S2
- Task 6U U-C1 oracle-selection ceiling (diagnostic only)

Write:
`evaluation/task6v_refval_family_policy.json`

---

# PART F — Downstream causal evaluation

## 10. Canonical-program MiniVal240

Use exact frozen MiniVal240.

Use canonical program ids, not natural-language parsing, for the main causal comparison.

Pipeline:

```text
canonical program
→ family + direction
→ frozen family-conditioned reference resolver
→ GeometricRelationField v0.2
→ frozen SAM2 visual feature
→ frozen B3
→ target mask
```

GT reference/target are evaluation only.

Report:

- strict all-240 mIoU;
- answered-only mIoU;
- Dice;
- Pr@0.5;
- abstentions;
- reference-fail count;
- target-fail-with-reference-ok count;
- largest/smallest target mIoU;
- per direction;
- border target;
- tiny target if present.

Write:
`evaluation/task6v_downstream_minival240.json`

## 11. PairedVal20

Use exact PairedVal20.

Report:
- pass /20;
- mean own IoU;
- mean cross IoU;
- own-cross margin;
- reference-abstention pairs.

Write:
`evaluation/task6v_downstream_pairedval20.json`

---

# PART G — Natural-language integration regression

## 12. Hardened ProgramHead integration

Run exact MiniVal240 natural-language queries through:

```text
Task 6T hardened ProgramHead
→ family-conditioned resolver
→ field v0.2
→ SAM2
→ B3
```

Require parser:
- 240/240 exact.

Report:
- strict mIoU;
- answered-only mIoU;
- abstentions;
- paired pass;
- own-cross margin.

Do not run nearest/L3 execution.

Write:
`evaluation/task6v_hardened_parser_integration.json`

---

# PART H — Predeclared gates

## 13. Family-policy improvement

Compared with Task 6U U-S1, require all for `family_policy_improved=true`:

- overall RefVal selected-reference mIoU >= U-S1 + `0.015`
- RefVal `REFERENCE_OK` >= U-S1 + `5`
- RefVal `REFERENCE_SELECTION_WRONG` <= U-S1 - `5`
- abstention rate <= `0.05`

Use exact U-S1 values from Task 6U.

## 14. Directional downstream hardening

`DIRECTIONAL_REFERENCE_HARDENING_PASS` requires ALL:

1. RefVal selected-reference mIoU >= `0.45`
2. RefVal centroid median <= `0.03`
3. RefVal centroid p90 <= `0.32`
4. MiniVal answered-only target mIoU >= `0.325`
5. MiniVal strict target mIoU >= `0.305`
6. PairedVal >= `12/20`
7. own-cross margin >= `0.30`
8. MiniVal reference-fail count <= `105`
9. parser integration remains 240/240
10. no test split / no GT inference.

These are development hardening gates, not final paper/test gates.

Do not alter thresholds.

---

# PART I — Verdict

## 15. Exactly one, priority order

1. `INVALID_EXPERIMENT`
   - test use, GT inference, post-RefVal policy change, frozen-module mutation, protocol violation.

2. `FROZEN_RESOLVER_ASSET_UNAVAILABLE`

3. `FAMILY_POLICY_NOT_BETTER`
   - section 13 improvement flag fails and section 14 hardening gate fails.

4. `FAMILY_POLICY_PARTIAL`
   - measurable improvement, but one or more section 14 gates fail.

5. `FAMILY_CONDITIONED_REFERENCE_HARDENING_PASS`
   - all section 14 gates pass.

No other verdict.

---

# PART J — Interpretation boundary

DSH reports measurements only.

Do NOT:
- claim family routing as novelty;
- redesign the ranker;
- train a smallest-only ranker;
- train a proposal-quality classifier;
- change candidate config;
- add confidence thresholds;
- add TTA/tiling;
- retrain YOLO;
- start nearest/L3;
- choose Task 6W.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 6V 的 family-conditioned resolver 结果决定是否需要新的 proposal-quality / selection 机制，不自行继续修改 reference 架构。`

---

# PART K — Required artifacts

Create at minimum:

```text
buildreasonseg_mvp/task6v_family_reference_resolver.py

evaluation/task6v_calibration_family_policy.json
evaluation/task6v_frozen_family_policy.json
evaluation/task6v_refval_family_policy.json
evaluation/task6v_downstream_minival240.json
evaluation/task6v_downstream_pairedval20.json
evaluation/task6v_hardened_parser_integration.json
evaluation/task6v_verdict.json

docs/task6v_family_conditioned_reference_resolver.md

scripts/task6v_select_family_policy.py
scripts/task6v_evaluate_reference.py
scripts/task6v_evaluate_downstream.py
scripts/task6v_report.py
```

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No new checkpoint.

---

# PART L — Tests

Task 6U ended at:
`851 passed, 1 skipped`

Add tests for at least:

1. Task 6U artifacts unchanged
2. U-Calib200 exact reuse
3. RefValUnique exact reuse
4. MiniVal240 exact reuse
5. PairedVal20 exact reuse
6. exactly three policy options
7. V-P0 exact
8. V-P1 exact
9. V-P2 exact
10. ranker not retrained
11. YOLO not retrained
12. no fourth option
13. per-family selection uses U-Calib200 only
14. no RefVal in policy selection
15. priority rule exact
16. policy frozen before RefVal eval
17. resolver dispatch is family-only
18. no relation input to resolver policy
19. no GT in resolver inference
20. canonical-program causal evaluation has no parser
21. natural-language integration uses hardened ProgramHead
22. parser remains 240/240
23. field v0.2 unchanged
24. B3 unchanged
25. SAM2 unchanged
26. no GRCL
27. no nearest/L3 execution
28. no test split
29. no new dataset/download/install/GUI
30. previous test suite preserved.

Run:

`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# PART M — Git/storage

Do not commit:
- existing ranker/parser/YOLO/SAM2/B3 checkpoints;
- proposal caches;
- feature caches;
- source imagery/vectors;
- `.conda`.

Commit:
- family resolver code;
- small JSON;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:

1. `feat: add family-conditioned reference resolver`
2. `eval: measure family-specific resolver policy`
3. optional docs commit

---

# PART N — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for a genuine integration/runtime bug.

No downloads or installs.

---

# PART O — STOP

After Task 6V:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- train new reference models;
- change proposal thresholds/configs;
- retrain YOLO/ranker/parser/B3;
- add proposal-quality model;
- add GRCL;
- add nearest/L3;
- access test;
- build GUI.

Wait for ChatGPT audit.
