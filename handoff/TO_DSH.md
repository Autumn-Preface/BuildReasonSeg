# TO_DSH — Task 7E: Deterministic Field-Weighted Prototype Holdout + Predicted-Reference Audit

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `86e4f4cd4bd5aa6858264a7d5709092b32678ebc`
>
> Predecessor: Task 7D → `GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`
>
> Research decision already made by ChatGPT:
>
> 1. Reject learned global competition D-B2 as a replacement for Z-B3.
> 2. Preserve the negative evidence that D-B2's learned competition map did not localize the target.
> 3. Treat **D-B1 deterministic field-weighted visual prototype** as a serious architecture candidate because it achieved:
>    - oracle-reference MiniVal mIoU 0.3978996;
>    - +0.0736867 over frozen Z-B3;
>    - Paired 19/20;
>    - own-cross margin +0.3888.
> 4. Do NOT retrain or tune D-B1 in Task 7E.
> 5. Before adopting D-B1, test whether its gain survives on a validation remainder that was not used for D-B1 checkpoint selection, and then measure predicted-reference error propagation.
> 6. Task 7E is evaluation/integration only. No model training is permitted.
> 7. Parser hardening remains stopped. Reference hardening remains stopped.
>
> DSH is an executor. Do not redesign D-B1, fields, resolver, packs, gates, or the next task.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- load frozen Z-B3 and D-B1 checkpoints;
- construct the exact held-out L3 validation remainder specified below;
- evaluate oracle-reference Z-B3 vs D-B1;
- evaluate predicted-reference Z-B3 vs D-B1 using the frozen U-C1 deterministic largest resolver;
- run paired/counterfactual audits on newly frozen held-out pairs;
- build/update a D-B1 L3 inference helper only if the final gates pass;
- solve ordinary runtime/integration bugs without altering algorithms.

DSH MUST NOT:
- train any model;
- retrain D-B1;
- retrain Z-B3;
- train parser/reference/YOLO/SAM2;
- change U-C1;
- change fields;
- change D-B1 formula;
- change thresholds;
- add learned competition;
- add attention/graph/Transformer;
- add target proposal/candidate inputs;
- use test split;
- use free-form parser performance to choose the architecture;
- choose Task 7F.

If a prohibited change is required, STOP and report.

---

# PART A — Freeze Task 7D interpretation

## 1. Record Task 7D result

Copy into `docs/task7e_deterministic_prototype_holdout.md`:

Task 7D verdict:
`GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`

Frozen Task 7D results:

| Variant | MiniVal mIoU | Dice | Paired | Margin |
|---|---:|---:|---:|---:|
| D-B0 frozen Z-B3 | 0.3242129 | 0.4389840 | 15/20 | +0.3003541 |
| D-B1 deterministic field-weighted prototype | **0.3978996** | **0.5272593** | **19/20** | **+0.3888** |
| D-B2 learned relation-guided competition | 0.3264764 | 0.4372995 | 17/20 | +0.2956 |
| D-B3 learned visual-only competition | 0.1595259 | 0.2310875 | 8/20 | +0.1629 |
| D-B4 learned competition map, no prototype | 0.3291152 | 0.4487840 | 19/20 | +0.3056 |

Important diagnostics:
- D-B2 target mass 0.0067
- D-B2 argmax-in-target 0.0000
- D-B2 entropy 0.8848
- learned competition is not accepted
- D-B1 is the only variant with a clear mask/counterfactual gain over Z-B3.

Do not modify Task 7D artifacts/verdict.

---

# PART B — Frozen checkpoints and modules

## 2. D-B1 checkpoint

Use exactly:

`artifacts/checkpoints/task7d/db1_minitrain1200.pt`

Expected SHA256:
`6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0`

Require:
- exists;
- SHA exact;
- variant metadata D-B1;
- checkpoint selected at Task 7D epoch 7;
- no retraining.

If unavailable/mismatch:
STOP:
`DB1_CHECKPOINT_UNAVAILABLE`

## 3. Z-B3 checkpoint

Use frozen Task 6Z Z-B3:

Expected SHA256:
`74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc`

No retraining.

## 4. Frozen architecture components

Read-only:
- GeometricRelationField v0.2
- NearestBoundaryField v0.1
- SAM2.1 Hiera Base+ frozen visual feature path
- U-C1 proposal configuration
- Task 6Q deterministic largest resolver
- BuildSpatialReason v0.2
- WHU-EA-NativeVector v1.0.

No test split.

---

# PART C — Exact D-B1 semantics

## 5. D-B1 formula

Do not rewrite or alter the Task 7D implementation.

For each oracle/predicted reference:

```text
P_dir
P_near
W = clamp(P_dir * P_near, 0, 1)
A_fixed = W / (sum(W) + eps)
A_fixed_vis = A_fixed * 4096

F = frozen SAM2 visual feature → trainable frozen-checkpoint projection
q = Σ_i A_fixed_i * F_i
C_i = cosine(F_i, q)
```

Frozen D-B1 decoder input:

```text
F
P_dir
P_near
direction embedding
A_fixed_vis
C
```

No learned competition score head.
No threshold.
No target proposal.

---

# PART D — Construct untouched L3 validation remainder

## 6. Source population

Use the full BuildSpatialReason v0.2 **val** split and select exactly the four L3 programs:

- largest_to_left_of_to_nearest
- largest_to_right_of_to_nearest
- largest_to_above_to_nearest
- largest_to_below_to_nearest

Expected total L3 val population:
- above 224
- below 213
- left 250
- right 249
- total 936

## 7. Exclusion set

Build exclusion record-id set from:
- every Z-MiniVal240 record;
- every record used as either member of Z-PairedVal20.

Remove the union from the 936-record population.

Call the remainder:
`E-HoldoutL3`

Requirements:
- no record-id overlap with Z-MiniVal240;
- no record-id overlap with Z-PairedVal20;
- no test records;
- at least 600 records total;
- every four L3 program has at least 120 records.

If any requirement fails:
STOP:
`L3_HOLDOUT_REMAINDER_INSUFFICIENT`

Do not subsample E-HoldoutL3. Use the complete remainder.

Write:
`evaluation/task7e_holdout_manifest.json`

Include:
- source counts;
- exclusion counts;
- final per-program counts;
- record-id hash;
- zero-overlap checks.

Local expanded row file may be gitignored under:
`artifacts/task7e/holdout/`.

---

# PART E — New held-out paired set

## 8. E-PairedHoldout

Construct only from E-HoldoutL3.

Pair requirements:
- same tile;
- same oracle largest reference source-feature id;
- different L3 direction programs;
- different target source-feature ids;
- both members belong to E-HoldoutL3.

Sort pairs by SHA256 stable key.

Require at least 20.
Use exactly first 20.

If fewer than 20:
STOP:
`L3_HOLDOUT_PAIRED_INSUFFICIENT`

Write pair IDs/hashes into:
`evaluation/task7e_holdout_manifest.json`

No overlap with Task 6Z Z-PairedVal20 members.

---

# PART F — E0 frozen MiniVal reproduction

## 9. Reproduce Task 7D first

Before holdout evaluation, reproduce on exact Z-MiniVal240:

Z-B3:
- mIoU 0.3242128982543474

D-B1:
- mIoU 0.3978996298363562

Require both absolute deltas <= 1e-6.

Reproduce paired:
- Z-B3 = 15/20
- D-B1 = 19/20.

Write:
`evaluation/task7e_minival_reproduction.json`

If fail:
STOP:
`TASK7D_REPRODUCTION_FAIL`

---

# PART G — E1 oracle-reference held-out architecture audit

## 10. Oracle inference

For every E-HoldoutL3 record:

Input:
- image
- canonical program id
- canonical oracle largest reference mask

Compare exactly:

### E-O0
Frozen Z-B3.

### E-O1
Frozen D-B1.

No retraining.
No parser.
No predicted reference.

## 11. Metrics

Report for both:
- record count
- mIoU
- Dice
- Pr@0.5
- per program/direction mIoU
- target area quartiles
- boundary-distance quartiles.

Report D-B1 - Z-B3 delta:
- overall
- each direction.

Bootstrap diagnostic:
- deterministic seed 20261001
- 2000 paired bootstrap resamples by record id
- report 95% percentile CI for:
  `mean IoU(D-B1) - mean IoU(Z-B3)`

Write:
`evaluation/task7e_oracle_holdout.json`

## 12. Oracle held-out paired

On E-PairedHoldout20 report for Z-B3 and D-B1:
- pass /20
- own IoU
- cross IoU
- own-cross margin.

Write:
`evaluation/task7e_oracle_holdout_paired.json`

---

# PART H — Oracle generalization gate

## 13. `DB1_HOLDOUT_GENERALIZES`

True iff ALL:

1. E-HoldoutL3 D-B1 mIoU >= `0.36`
2. D-B1 - Z-B3 overall mIoU >= `+0.05`
3. D-B1 improves or ties Z-B3 in at least 3/4 directions
4. 95% bootstrap CI lower bound for delta > `0.00`
5. D-B1 E-PairedHoldout >= `16/20`
6. D-B1 own-cross margin >= `0.30`

If false:
do NOT run predicted-reference adoption path.
Proceed only to verdict/report and STOP.

---

# PART I — E2 predicted-reference holdout audit

Run only if section 13 passes.

## 14. Frozen practical resolver

Use exactly U-C1:

```text
YOLO26m-seg Task 6M.1 checkpoint
imgsz 640
conf 0.05
max_det 300
default NMS
no TTA
no tiling
```

Largest eligibility:
- not border-touching
- bbox extent ratio <=0.20

Select:
- max predicted mask area
- tie higher confidence
- then lower original index.

No ranker/filter/SAM2 refinement.

## 15. Compare predicted-reference pipelines

For every E-HoldoutL3 record:

### E-P0
canonical program
→ predicted U-C1 largest reference
→ frozen Z-B3

### E-P1
canonical program
→ predicted U-C1 largest reference
→ frozen D-B1

No GT reference/target inference input.

Reference abstention:
- strict target IoU = 0.

## 16. Reference diagnostics

GT reference only offline.

Report:
- reference mIoU
- Dice
- Pr@0.5
- abstentions
- NOT_COVERED
- SELECTION_WRONG
- GEOMETRY_POOR
- REFERENCE_OK.

Write:
`evaluation/task7e_predicted_reference_quality.json`

## 17. Target metrics

For E-P0 and E-P1:
- strict mIoU/Dice
- answered-only mIoU/Dice
- Pr@0.5
- reference-OK subset target mIoU
- per direction
- abstentions.

Define:
```text
oracle_db1 = E-O1 mIoU
pred_db1_strict = E-P1 strict mIoU
retention = pred_db1_strict / oracle_db1
```

Write:
`evaluation/task7e_predicted_reference_holdout.json`

## 18. Predicted-reference held-out paired

Use E-PairedHoldout20.

Because pair members share same tile/reference:
- resolve reference once and reuse for both.

Report:
- Z-B3 pass /20
- D-B1 pass /20
- own/cross/margin
- reference-abstention pairs.

Write:
`evaluation/task7e_predicted_reference_paired.json`

---

# PART J — Predicted-reference development gate

## 19. `DB1_PREDICTED_REFERENCE_USABLE`

Requires section 13 pass AND ALL:

1. E-P1 strict mIoU >= `0.24`
2. E-P1 answered-only mIoU >= `0.25`
3. E-P1 - E-P0 strict mIoU >= `+0.03`
4. oracle→predicted retention >= `0.62`
5. predicted-reference paired >= `11/20`
6. own-cross margin >= `0.20`
7. reference abstention <= `10%`.

These are development gates, not final-paper gates.

---

# PART K — Optional canonical parser integration

## 20. Interface regression only

If section 19 passes, run the Task 7C parser on the existing canonical query strings associated with E-HoldoutL3.

Do NOT evaluate free-form paraphrase here.

Require canonical accuracy >= `0.995`.

Then run:
```text
Task 7C canonical parser
→ U-C1 reference
→ D-B1
```

Report strict/answered metrics.

Write:
`evaluation/task7e_canonical_parser_integration.json`

Parser does not decide architecture adoption except an accuracy <0.995 is recorded as an interface regression.

No parser training.

---

# PART L — Development architecture decision

## 21. Architecture status

Define:

### `DB1_ADOPT_AS_DEVELOPMENT_L3_DECODER`
true iff:
- `DB1_HOLDOUT_GENERALIZES=true`
- `DB1_PREDICTED_REFERENCE_USABLE=true`
- no protocol violation.

If true:
- D-B1 becomes the **development L3 decoder candidate** replacing Z-B3 for subsequent integration/formalization work.
- Z-B3 remains frozen baseline/ablation.
- Do NOT delete/overwrite either checkpoint.

If false:
- retain Z-B3 as current development baseline;
- D-B1 remains a positive ablation only.

No full training/test yet.

---

# PART M — Verdict

## 22. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `DB1_CHECKPOINT_UNAVAILABLE`
3. `L3_HOLDOUT_REMAINDER_INSUFFICIENT`
4. `L3_HOLDOUT_PAIRED_INSUFFICIENT`
5. `TASK7D_REPRODUCTION_FAIL`
6. `DB1_HOLDOUT_GENERALIZATION_FAIL`
7. `DB1_PREDICTED_REFERENCE_BELOW_GATE`
8. `DB1_DEVELOPMENT_L3_DECODER_READY`

No other verdict.

---

# PART N — Interpretation boundary

DSH reports measurements only.

Do NOT:
- claim D-B1 is globally novel;
- call D-B1 final model;
- run full training;
- evaluate test;
- retrain parser/reference;
- add learned competition;
- add graph/attention;
- modify D-B1 after holdout results;
- choose Task 7F.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7E 的 untouched L3 holdout 与 predicted-reference 结果决定是否正式冻结 D-B1 为开发版 L3 decoder，不自行进行全量训练、test 评估或新的架构改动。`

---

# PART O — Required artifacts

Create:

```text
evaluation/task7e_holdout_manifest.json
evaluation/task7e_minival_reproduction.json
evaluation/task7e_oracle_holdout.json
evaluation/task7e_oracle_holdout_paired.json

# only if oracle holdout gate passes:
evaluation/task7e_predicted_reference_quality.json
evaluation/task7e_predicted_reference_holdout.json
evaluation/task7e_predicted_reference_paired.json
evaluation/task7e_canonical_parser_integration.json

evaluation/task7e_verdict.json

docs/task7e_deterministic_prototype_holdout.md

scripts/task7e_build_holdout.py
scripts/task7e_evaluate_oracle.py
scripts/task7e_evaluate_predicted_reference.py
scripts/task7e_parser_integration.py
scripts/task7e_report.py
```

Optional helper:
`buildreasonseg_mvp/task7e_l3_decoder_adapter.py`
only if needed to cleanly load D-B1 without changing Task 7D code.

Local caches:
`artifacts/task7e/`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No new checkpoint.

---

# PART P — Tests

Task 7D ended at:
`1243 passed, 1 skipped`

Add tests for at least:

1. Task 7D artifacts unchanged
2. D-B1 checkpoint hash exact
3. Z-B3 checkpoint hash exact
4. no training in Task 7E
5. D-B1 architecture unchanged
6. no learned score head in D-B1
7. exact product field weighting
8. exact prototype formula
9. exact cosine similarity
10. no target proposal input
11. full L3 val source total checked
12. MiniVal record IDs excluded from holdout
13. PairedVal member IDs excluded from holdout
14. no test split
15. holdout >=600
16. each L3 class >=120
17. held-out paired same tile/reference
18. held-out paired different direction/target
19. held-out paired has no old paired member
20. MiniVal reproduction exact
21. oracle comparison has no parser
22. oracle comparison uses GT reference only as declared
23. bootstrap seed/count exact
24. bootstrap samples paired by record
25. U-C1 exact if predicted stage runs
26. shared reference resolver between P0/P1
27. no GT reference in predicted inference
28. no GT target in inference
29. reference reused within predicted pair
30. Task 7C parser not trained
31. no free-form paraphrase used for architecture selection
32. fields unchanged
33. SAM2 unchanged
34. no ranker/quality/refinement
35. no learned competition
36. no attention/Transformer/GNN
37. no GRCL
38. no full training
39. no test
40. no new dataset/download/install/GUI
41. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART Q — Git/storage

Do not commit:
- checkpoints;
- model weights;
- proposal/feature caches;
- large local expanded holdout rows;
- source imagery/vectors;
- `.conda`.

Commit:
- small holdout manifest;
- evaluation JSON;
- scripts/helpers;
- tests;
- docs;
- handoff.

Suggested commits:
1. `eval: validate deterministic prototype on untouched L3 holdout`
2. `eval: audit D-B1 predicted-reference retention`
3. optional docs/handoff commit

---

# PART R — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for genuine integration/runtime bugs.

No installs or downloads.

---

# PART S — STOP

After Task 7E:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- train anything;
- run test;
- change parser/reference/D-B1;
- add another architecture;
- start formal full-data training.

Wait for ChatGPT audit.
