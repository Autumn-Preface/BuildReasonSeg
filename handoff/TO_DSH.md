# TO_DSH — Task 7I: Formal L3 Three-Seed Train/Validation — Z-B3 vs D-B1

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `511a7bd0d7aca2f5f52637d875f0b3bc84751671`
>
> Predecessors:
> - Task 7G → `LARGEST_SELECTOR_NOT_LEARNABLE`
> - Task 7H → `DEVELOPMENT_ARCHITECTURE_FROZEN`
>
> ChatGPT audit decision:
>
> 1. Task 7G is accepted. Reference intervention stops for the current project version.
> 2. Task 7H is accepted only as a **development-architecture / limitation / test-lock documentation freeze**.
> 3. Task 7H formal-training protocol is corrected in Task 7I in two places:
>    - formal checkpoint selection MUST use **all 936 valid v0.2 L3 validation records**, NOT the historical
>      `Z-MiniVal240`;
>    - the formal comparison MUST retrain **both Z-B3 and D-B1** from fresh trainable weights under the same
>      three seeds and the same full train/val population.
> 4. Do NOT modify the frozen Task 7H artifacts. Record these as Task 7I protocol corrections.
> 5. The test lock remains **LOCKED** throughout Task 7I. No test image, mask, annotation, record id, metric or
>    test-derived statistic may be read or produced.
> 6. No architecture, loss, threshold, field, parser or reference-policy tuning is permitted.
> 7. Task 7I is the formal train/validation confirmation stage for the frozen L3 target-decoder architecture.
> 8. DSH is executor only. Do not choose whether to unlock test; ChatGPT decides after audit.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- audit frozen Task 7H assets;
- freeze the full formal train/val record manifests;
- freeze a deterministic full-validation counterfactual-pair manifest;
- train Z-B3 and D-B1 from fresh trainable weights for exactly three seeds each;
- select each run's checkpoint using all 936 L3 val records;
- evaluate selected checkpoints on full val using oracle-reference and predicted-reference protocols;
- record all metrics, efficiency, checkpoints and hashes;
- solve implementation/runtime bugs that do not change the protocol.

DSH MUST NOT:
- read or evaluate test split;
- train YOLO, SAM2 or ProgramHead;
- train a reference selector;
- change U-C1;
- change deterministic largest selection;
- change GeometricRelationField v0.2;
- change NearestBoundaryField v0.1;
- change Z-B3 or D-B1 architecture;
- add GRCL or any new loss;
- change optimizer/LR/batch/max epochs/patience;
- use MiniVal240 as formal checkpoint selection;
- choose one seed by external/test performance;
- tune architecture based on formal val results;
- unlock test;
- choose Task 7J.

If a prohibited change is required, STOP and report.

---

# PART A — Accept Task 7G and Task 7H correctly

## 1. Record Task 7G result

Copy into `docs/task7i_formal_l3_trainval.md`:

Task 7G:
`LARGEST_SELECTOR_NOT_LEARNABLE`

Internal tile-disjoint holdout:

```text
G-I0 deterministic mean selected ref IoU = 0.5199913587
G-I1 learned mean selected ref IoU       = 0.6198877726
gain                                      = +0.0998964138

G-I1 oracle-best exact top1              = 0.6591928251
G-I1 best-minus-selected gap             = 0.1790755783
```

Gate:
- gain >= +0.08: PASS
- mean selected IoU >= 0.62: FAIL
- top1 >= 0.55: PASS
- gap <= 0.14: FAIL

Therefore:
- external E-Holdout stage was not executed;
- no scene-disjoint claim exists for Task 7G selector;
- selector is not adopted;
- reference intervention stops.

## 2. Freeze development architecture roles

Preserve Task 7H roles:

- dataset:
  `WHU-EA-NativeVector v1.0 + BuildSpatialReason v0.2 + scene_disjoint_v1`
- parser:
  Task 7C Qwen3-VL-2B text-only ProgramHead;
  controlled/canonical interface only;
  NOT used for checkpoint selection in Task 7I.
- proposal:
  Task 6M.1 YOLO26m-seg U-C1.
- practical reference:
  deterministic Task 6Q largest selector.
- directional field:
  GeometricRelationField v0.2.
- nearest field:
  NearestBoundaryField v0.1.
- L3 baseline:
  Z-B3.
- preferred L3 candidate:
  D-B1.
- rejected modules remain excluded.

## 3. Task 7I protocol corrections

Record explicitly:

### Correction I-01
Task 7H JSON contained:
`formal_d_b1_training.schedule.checkpoint_selection = "MiniVal240 mIoU"`

For formal training this is superseded by:

`all 936 valid v0.2 L3 validation records, oracle-reference mIoU`

### Correction I-02
Formal architecture comparison retrains BOTH:
- Z-B3
- D-B1

with:
- identical formal train/val population;
- identical seeds;
- identical optimizer schedule;
- fresh trainable initialization.

Historical Z-B3/D-B1 checkpoints remain development evidence only.

Do not edit Task 7H files.

---

# PART B — Test lock

## 4. Test remains locked

Read:
`evaluation/task7h_test_lock.json`

Require:
- status = `LOCKED`
- test_execution_authorized = false.

Task 7I MUST NOT:
- open test JSONL;
- enumerate test sample ids;
- hash test records/files for Task 7I;
- generate test cache;
- run test inference;
- output any Task 7I test metric.

If test lock is not locked:
STOP:
`TEST_LOCK_INCONSISTENT`

---

# PART C — Formal populations

## 5. Exact programs

Formal L3 covers only:

- `largest_to_left_of_to_nearest`
- `largest_to_right_of_to_nearest`
- `largest_to_above_to_nearest`
- `largest_to_below_to_nearest`

No L1/L2 enters Task 7I training.

## 6. Formal train

Use all valid BuildSpatialReason v0.2 TRAIN records of those four programs.

Expected:

```text
left   323
right  347
above  338
below  336
total  1344
```

Every training record uses:
- source image;
- canonical GT largest reference mask;
- canonical direction;
- canonical GT target mask as training label.

GT target is never a model feature.

## 7. Formal validation

Use all valid BuildSpatialReason v0.2 VAL records of those four programs.

Expected:

```text
left   250
right  249
above  224
below  213
total  936
```

For checkpoint selection:
- use oracle GT reference;
- use all 936 records;
- metric = mean target mIoU over all 936.

No MiniVal-only checkpoint selection.

## 8. Freeze manifests before training

Create:
`evaluation/task7i_formal_population_manifest.json`

Include:
- exact train count and per-program counts;
- exact val count and per-program counts;
- train sample-id SHA256;
- val sample-id SHA256;
- train/val sample-id overlap = 0;
- active dataset/version/split identity;
- no test material read.

Large expanded row files:
`artifacts/task7i/packs/`
gitignored.

---

# PART D — Full-validation counterfactual set

## 9. Freeze before training

Construct from the 936 formal VAL records only.

Create all unique unordered record pairs satisfying:

- same `tile_id`;
- same `reference_source_feature_id`;
- different canonical direction;
- different `target_source_feature_id`;
- both are one of the four Task 7I L3 programs.

Pair key:

```text
min(sample_id_a, sample_id_b) + "||" + max(sample_id_a, sample_id_b)
```

Deduplicate exact pair key.
Sort lexicographically.
Do NOT subsample.

Call:
`I-FormalValPairsAll`

Write:
`evaluation/task7i_formal_val_pair_manifest.json`

This pair set is reporting-only.
It is NOT used for checkpoint selection or early stopping.

---

# PART E — Frozen visual / geometric inputs

## 10. Frozen SAM2

Use exact project SAM2.1 Hiera Base+ frozen image embeddings:

`V ∈ R^(256×64×64)`

No SAM2 gradient.
No checkpoint change.

Generating missing TRAIN/VAL feature caches is allowed.
No test cache.

## 11. Oracle reference training/selection

For BOTH Z-B3 and D-B1 formal training and checkpoint selection:

`reference_source = oracle_native_gt`

This deliberately compares target-decoder architectures independently of the known reference bottleneck.

Practical predicted-reference validation is run only AFTER each best checkpoint is frozen.

## 12. Fields

Exact frozen:
- GeometricRelationField v0.2 → `P_dir`
- NearestBoundaryField v0.1 → `P_near`

No formula/constant change.

---

# PART F — Model F0: Z-B3 formal baseline

## 13. Architecture

Use exact Task 6Z Z-B3 architecture.

All trainable Z-B3 projection/embedding/decoder parameters are freshly initialized for each formal seed.

Do NOT load historical Z-B3 trained weights as initialization.

---

# PART G — Model F1: D-B1 formal candidate

## 14. Architecture

Use exact Task 7D D-B1:

```text
P_dir
P_near
W = clamp(P_dir * P_near, 0, 1)
A_fixed = W / (sum(W) + eps)
A_fixed_vis = A_fixed * 4096

F = trainable 256→128 visual projection applied to frozen SAM2 feature
q = Σ_i A_fixed_i F_i
C_i = cosine(F_i, q)

decoder input =
F
+ P_dir
+ P_near
+ direction embedding
+ A_fixed_vis
+ C
→ exact D-B1 decoder
→ target mask
```

No learned global competition head.

All trainable D-B1 parameters are freshly initialized per seed.
Do NOT initialize from Task 7D D-B1 checkpoint.

---

# PART H — Exact training protocol

## 15. Seeds

Run BOTH F0 and F1 for exactly:

- `20261001`
- `20261002`
- `20261003`

Total formal training runs:
`6`

Do not add/replace seeds.

## 16. Common optimization

For every F0/F1 seed:

- optimizer: AdamW
- lr: `3e-4`
- weight_decay: `1e-4`
- batch size: `8`
- max epochs: `25`
- early stopping patience: `5`
- no scheduler
- no augmentation
- bfloat16 autocast / same established AMP implementation
- loss: `BCEWithLogitsLoss + DiceLoss`
- gradient updates: formal TRAIN 1344 only.

No hyperparameter sweep.

## 17. Formal checkpoint selection

At the end of every epoch evaluate ALL 936 oracle-reference VAL records.

Primary:
- highest val mean mIoU.

Tie-break:
1. higher val Dice;
2. higher val Pr@0.5;
3. earlier epoch.

Early stopping:
- based ONLY on full-val mIoU;
- patience 5.

Do not use:
- MiniVal240 as selection metric;
- pair result as selection metric;
- practical predicted-reference result as selection metric;
- test.

## 18. Checkpoints

For every model/seed store locally:

```text
artifacts/checkpoints/task7i/<model>/<seed>/best.pt
artifacts/checkpoints/task7i/<model>/<seed>/last.pt
```

gitignored.

Record:
- SHA256
- bytes
- selected epoch
- final epoch
- optimizer settings
- trainable params
- wall time
- peak VRAM
- NaN/Inf/OOM status.

Resume after process interruption only from that SAME run's `last.pt`.
Do not replace a failed seed with another seed.

---

# PART I — Oracle-reference full-val evaluation

## 19. Selected checkpoints only

After all six best checkpoints are frozen, run one final full-val oracle-reference evaluation per checkpoint.

Report for each model/seed:

- mIoU
- Dice
- Pr@0.5
- per direction;
- target area quartiles;
- boundary-distance quartiles;
- FormalValPairsAll:
  - pair pass count/rate;
  - own IoU;
  - cross IoU;
  - own-cross margin.

Write:
`evaluation/task7i_oracle_val_results.json`

## 20. Aggregate

For Z-B3 and D-B1 separately report:
- three seed values;
- mean;
- sample std (`ddof=1`) for mIoU/Dice/Pr@0.5;
- mean/std paired pass rate;
- mean/std paired margin.

Also report matched-seed D-B1 minus Z-B3 mIoU deltas.

---

# PART J — Practical predicted-reference full-val evaluation

## 21. Frozen practical reference

After checkpoint selection is complete, evaluate all six selected checkpoints with:

```text
U-C1 proposals:
imgsz=640
conf=0.05
max_det=300
default NMS
no TTA
no tiling

largest selector:
max predicted mask area
tie higher confidence
then lower index
```

No Task 7G selector.

Proposal inference/reference selection must be cached/reused across all six model runs.

## 22. Reference metrics

Compute ONCE over all 936 val records:

- reference mIoU
- Dice
- Pr@0.5
- abstention rate
- NO_PROPOSALS
- NO_ELIGIBLE
- NOT_COVERED
- SELECTION_WRONG
- GEOMETRY_POOR
- REFERENCE_OK.

Write in:
`evaluation/task7i_predicted_reference_val_results.json`

## 23. Target metrics

For each model/seed:

- strict all-record mIoU/Dice/Pr@0.5;
- answered-only mIoU/Dice;
- reference-OK subset target mIoU;
- per direction;
- FormalValPairsAll pass rate/own/cross/margin using shared predicted reference per pair.

Aggregate mean/std across seeds.

Predicted-reference results do NOT affect checkpoint selection.

---

# PART K — Formal architecture confirmation

## 24. Required run validity

Require:
- all 3 Z-B3 runs valid;
- all 3 D-B1 runs valid;
- no test access;
- all checkpoint selection from full 936 oracle val;
- same formal population and schedule.

## 25. `DB1_FORMAL_VAL_CONFIRMED`

True iff ALL:

1. D-B1 mean oracle-reference full-val mIoU >= `0.35`
2. D-B1 mean oracle mIoU >= Z-B3 mean oracle mIoU + `0.04`
3. D-B1 oracle mIoU > Z-B3 oracle mIoU in at least `2/3` matched seeds
4. no D-B1 seed oracle mIoU < `0.32`
5. D-B1 mean predicted-reference strict mIoU >= Z-B3 mean predicted strict mIoU + `0.02`
6. D-B1 mean predicted-reference strict mIoU >= `0.22`
7. D-B1 mean FormalValPairsAll own-cross margin is not lower than Z-B3 by more than `0.02`
8. no protocol violation.

These gates confirm the architecture on formal validation only.
They do NOT unlock test automatically.

## 26. Diagnostics, not gates

Report:
- oracle-to-predicted retention per seed/model;
- seed std;
- D-B1 vs Z-B3 parameter difference;
- training time difference;
- inference-time difference;
- reference failure attribution.

---

# PART L — Test-lock decision artifact

## 27. Do not unlock test

Create:
`evaluation/task7i_test_lock_status.json`

It MUST say:

```text
status = LOCKED
test_execution_authorized = false
task7i_completed_train_val = true
db1_formal_val_confirmed = true_or_false
unlock_requires = ChatGPT audit of Task 7I
```

No test action is authorized by DSH.

---

# PART M — Verdict

## 28. Exactly one formal verdict

Priority:

1. `INVALID_EXPERIMENT`
2. `TEST_LOCK_INCONSISTENT`
3. `FORMAL_POPULATION_MISMATCH`
4. `FORMAL_TRAINING_INCOMPLETE`
5. `DB1_FORMAL_VAL_NOT_CONFIRMED`
6. `DB1_FORMAL_VAL_CONFIRMED`

No other verdict.

---

# PART N — Interpretation boundary

DSH may report measurements only.

Do NOT:
- call validation results final test results;
- call test untouched;
- choose best seed by test;
- average only successful/favorable seeds;
- change architecture after seeing formal val;
- unlock test;
- start final test;
- retrain parser/reference/YOLO;
- run another selector;
- claim unrestricted natural-language capability;
- claim unseen-city generalization;
- claim novelty/first.

Final recommendation exactly:

`等待 ChatGPT 审核 Task 7I 的三种子正式 train/val 结果；在 ChatGPT 明确解锁前，test 保持 LOCKED，不运行任何 final-test inference。`

---

# PART O — Required artifacts

Create:

```text
evaluation/task7i_formal_population_manifest.json
evaluation/task7i_formal_val_pair_manifest.json

evaluation/task7i_training_zb3.json
evaluation/task7i_training_db1.json

evaluation/task7i_oracle_val_results.json
evaluation/task7i_predicted_reference_val_results.json
evaluation/task7i_formal_comparison.json

evaluation/task7i_test_lock_status.json
evaluation/task7i_verdict.json

docs/task7i_formal_l3_trainval.md

scripts/task7i_freeze_formal_population.py
scripts/task7i_train.py
scripts/task7i_evaluate_oracle_val.py
scripts/task7i_evaluate_predicted_val.py
scripts/task7i_compare.py
scripts/task7i_report.py
```

Local:
```text
artifacts/task7i/
artifacts/checkpoints/task7i/
```
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

Do not change production CLI defaults.

---

# PART P — Required tests

Task 7H ended at:
`1420 passed, 1 skipped`

Add tests for at least:

1. Task 7H artifacts unchanged
2. Task 7G negative result preserved
3. Task 7I correction I-01 documented
4. Task 7I correction I-02 documented
5. test lock initially LOCKED
6. no test path opened by Task 7I scripts
7. exact four L3 programs
8. train count 1344
9. train per-program counts exact
10. val count 936
11. val per-program counts exact
12. train/val ids disjoint
13. pair pack uses val only
14. pair same tile
15. pair same reference
16. pair different direction
17. pair different target
18. pair pack no subsampling
19. frozen SAM2 unchanged
20. oracle reference used for formal gradient train
21. GT target is label only
22. directional field unchanged
23. nearest field unchanged
24. Z-B3 exact architecture reused
25. D-B1 exact architecture reused
26. Z-B3 fresh initialization per seed
27. D-B1 fresh initialization per seed
28. exactly three fixed seeds
29. exactly six formal runs
30. optimizer exact
31. lr exact
32. weight decay exact
33. batch exact
34. max epochs exact
35. patience exact
36. BCE+Dice only
37. no scheduler/augmentation
38. checkpoint selection uses FULL 936 val
39. MiniVal240 not used for checkpoint selection
40. predicted reference not used for checkpoint selection
41. no test checkpoint selection
42. same-run resume only
43. predicted reference U-C1 exact
44. Task 7G selector absent from formal chain
45. reference cache shared across model/seeds
46. formal comparison gates exact
47. test lock remains locked after Task 7I
48. no parser/YOLO/reference training
49. no new loss/GRCL/attention/graph
50. no new dataset/download/install/GUI
51. previous passing suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART Q — Git/storage

Do not commit:
- Task 7I checkpoints;
- YOLO/SAM2/parser weights;
- feature/proposal caches;
- source imagery/vectors;
- expanded pack rows if large;
- `.conda`.

Commit:
- population/pair manifests;
- small training/evaluation JSON;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:

1. `eval: freeze formal L3 train-val populations`
2. `train: run three-seed Z-B3 and D-B1 formal validation`
3. `docs: record formal L3 validation comparison`

---

# PART R — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for genuine training/resume/CUDA/cross-module bugs.

No downloads or installs.

---

# PART S — STOP

After Task 7I:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- unlock test;
- run final test;
- start Task 7J;
- change architecture;
- train another selector/reference/parser.

Wait for ChatGPT audit.
