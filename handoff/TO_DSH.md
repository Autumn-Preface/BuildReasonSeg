# TO_DSH — Task 6U: Reference Candidate Coverage + Proposal-Set Ranker Hardening

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `e63f8c40761637040ac6169ad603b6f7c7001274`
>
> Predecessor: Task 6T → `PARSER_SEMANTIC_CONTRAST_FAIL`
>
> Research decision already made by ChatGPT:
>
> 1. Accept the Task 6T hardened ProgramHead as the **current directional-chain parser** because it preserves 240/240 MiniVal, 40/40 paired members, improves fixed24 from 21/24 to 24/24, and improves the held-out stress pack to 0.99375 accuracy/macro-F1.
> 2. Do **not** claim Task 6T fully solves nearest semantics. Compact `direction + nearest` phrasing remains technical debt and must be revisited before enabling nearest/L3 execution.
> 3. The dominant scientific bottleneck from Task 6S remains **REFERENCE**: 117 reference-side failures vs 67 target-decoder failures.
> 4. Task 6U therefore isolates and hardens two reference subproblems:
>    - candidate proposal **coverage**;
>    - extreme-instance **selection/ranking**.
> 5. Do not change the core GeometricRelationField/B3 method.

DSH is an executor. Do not redesign the experiment.

---

# 0. DSH role

All user-facing DSH output must be Chinese.

DSH MAY:
- run the exact frozen YOLO26m-seg checkpoint under the four declared inference configurations;
- build the exact train-side calibration split defined here;
- implement the exact ProposalSetRanker v0.1 specified here;
- train only that small ranker;
- evaluate reference and downstream target chains;
- solve ordinary implementation/runtime bugs without changing the experiment.

DSH MUST NOT:
- retrain YOLO;
- change YOLO weights;
- use another detector/segmenter;
- change the native dataset/splits;
- alter the hardened ProgramHead;
- retrain ProgramHead;
- change SAM2;
- change GeometricRelationField v0.2;
- retrain B3;
- add GRCL;
- add nearest/L3 target execution;
- use test split;
- use oracle target/reference at inference;
- invent an additional inference configuration;
- choose Task 6V.

If an unexpected issue requires a prohibited change, STOP and report.

---

# PART A — Record Task 6T audit notes

## 1. Parser status

Record in `docs/task6u_reference_hardening.md`:

The Task 6T hardened parser is accepted for the current eight-program directional chain, but Task 6T's formal verdict remains `PARSER_SEMANTIC_CONTRAST_FAIL`.

Measured hardened parser:
- full v0.2 val = 1.0000;
- MiniVal240 = 240/240;
- PairedVal members = 40/40;
- fixed24 = 24/24;
- stress v1 accuracy/macro-F1 = 0.99375/0.99375;
- residual failure: one `largest_to_nearest` stress/minimal-pair case and the two compact direction+nearest controls.

Do not modify Task 6T verdict/artifacts.

## 2. Task 6T reporting erratum

Record without mutating Task 6T artifacts:

`handoff/FROM_DSH.md` states Task 6T peak VRAM as `0.67 GB`, while the authoritative `evaluation/task6t_training_summary.json` records C1 peak VRAM as **7.29 GB**.

For future reporting use the training-summary value.

This has no effect on parser metrics.

---

# PART B — Frozen assets

## 3. Freeze all prior evidence

Read-only:
- all Task 6S artifacts;
- all Task 6T artifacts;
- hardened ProgramHead checkpoint;
- Task 6M.1 YOLO26m-seg checkpoint;
- Task 6Q resolver implementation;
- Task 6O B3 checkpoint;
- GeometricRelationField v0.2;
- frozen SAM2 feature cache/path;
- WHU-EA-NativeVector v1.0;
- BuildSpatialReason v0.2;
- Task 3B spatial config.

No test split.

## 4. Proposal checkpoint

Use exactly:

`artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt`

SHA256:
`ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`

No retraining.

## 5. Main evaluation packs

Main reference evaluation:
- exact Task 6P `RefValUnique` (219 references).

Main downstream evaluation:
- exact Task 6N MiniVal240;
- exact Task 6N PairedVal20.

Do not regenerate.

---

# PART C — Train-side calibration split

## 6. Use RefTrainUnique only

Start from the frozen Task 6P `RefTrainUnique`.

Create two disjoint groups by unique reference key:

```text
(split, tile_id, reference_source_feature_id, reference_family)
```

Deterministic seed:
`20260930`

### U-Calib200
Exactly 200 references if available:
- 100 largest;
- 100 smallest;
- unique tile/reference key;
- selected deterministically by SHA256 of `seed:key`.

If one family has <100, use all of that family and fill from the other.

### U-RankerTrain
All remaining RefTrainUnique references not in U-Calib200.

No RefValUnique record may enter calibration or ranker training.

Write:
`evaluation/task6u_reference_train_split.json`

Require:
- zero key overlap U-Calib200 ↔ U-RankerTrain;
- zero tile/reference key overlap with RefValUnique;
- no test split.

---

# PART D — Fixed proposal inference configurations

## 7. Exactly four configurations

Use the same frozen YOLO weights.

Do not add any others.

### U-C0 — frozen baseline
- imgsz = 640
- conf = 0.10
- max_det = 100
- default NMS
- no TTA
- no tiling

### U-C1 — lower confidence / larger candidate cap
- imgsz = 640
- conf = 0.05
- max_det = 300
- default NMS
- no TTA
- no tiling

### U-C2 — higher network input resolution
- imgsz = 1024
- conf = 0.10
- max_det = 300
- default NMS
- no TTA
- no tiling

### U-C3 — higher resolution + recall-oriented threshold
- imgsz = 1024
- conf = 0.05
- max_det = 300
- default NMS
- no TTA
- no tiling

Important:
- source image remains the original 512×512 tile;
- Ultralytics performs its normal input resize/letterbox;
- predicted masks must be restored to exact source 512×512 coordinates before reference logic;
- no artificial source-image super-resolution;
- no TTA;
- no sliding-window/tiling.

## 8. Eligibility policy

For every config, use exactly the Task 6Q family eligibility rules:

Largest:
- not border-touching;
- bbox extent ratio <= 0.20.

Smallest:
- not border-touching;
- bbox extent ratio <= 0.20;
- area >= 150 px.

Do not change thresholds.

---

# PART E — U0 candidate-coverage selection

## 9. Run U-Calib200 under all four configs

For each config report:

- all-proposal coverage@0.25 / 0.50 / 0.75;
- eligible coverage@0.25 / 0.50 / 0.75;
- overall;
- largest;
- smallest;
- proposals/tile mean/median/p90;
- eligible proposals/tile;
- no-proposal count;
- no-eligible count;
- inference wall time/tile;
- peak VRAM.

GT reference is evaluation only.

Write:
`evaluation/task6u_calibration_candidate_coverage.json`

## 10. Freeze one selected proposal configuration

Selection rule, in this exact priority order:

1. highest `smallest eligible coverage@0.50`;
2. then highest `overall eligible coverage@0.50`;
3. then highest `largest eligible coverage@0.50`;
4. then lower mean eligible proposals/tile;
5. then lower imgsz;
6. then higher conf;
7. then lower config id.

Do not use RefValUnique or MiniVal240 to choose config.

Freeze:
`evaluation/task6u_selected_proposal_config.json`

After this file exists, the selected config may not change.

---

# PART F — Reference candidate audit on untouched RefValUnique

## 11. Compare baseline and selected config

Run:
- U-C0 baseline;
- selected U-C* config.

On exact RefValUnique report:

- eligible coverage@0.25 / 0.50 / 0.75;
- overall/largest/smallest;
- proposals/tile;
- eligible proposals/tile;
- no proposal;
- no eligible.

Also compute **oracle-selection ceiling**:
if GT were allowed only to choose the best eligible proposal, report:
- oracle selected-reference mIoU;
- Dice;
- centroid median/p90;
- Pr@0.5.

This ceiling is diagnostic only and never enters inference.

Write:
`evaluation/task6u_refval_candidate_audit.json`

---

# PART G — ProposalSetRanker v0.1

## 12. Purpose

Task 6S showed:
- `REFERENCE_NOT_COVERED_IOU50 = 68`;
- `REFERENCE_SELECTION_WRONG = 42`.

The ranker addresses only the second category.

It cannot create missing proposals.

It is support infrastructure, NOT a claimed project novelty.

## 13. Training examples

Generate proposals for U-RankerTrain using the **selected frozen config**.

For each reference record:
- keep only eligible proposals for its family;
- find best eligible proposal IoU vs GT reference;
- if best eligible IoU < 0.50: exclude record from ranker-loss training and count it as `untrainable_not_covered`;
- otherwise target class = proposal index with highest IoU to GT reference;
- ties in GT IoU → higher YOLO confidence → lower original proposal index.

GT is used only to make training labels.

At inference, ranker receives no GT.

## 14. Exact proposal feature vector

For each eligible proposal compute exactly these 12 scalar features:

1. `log_area = log1p(area_px) / log1p(512*512)`
2. `area_ratio = area_px / (512*512)`
3. `area_over_set_max = area_px / max(area_px_in_set)`
4. `set_min_over_area = min(area_px_in_set) / area_px`
5. `area_rank_desc = rank by descending area / max(1,N-1)` where largest rank=0
6. `area_rank_asc = rank by ascending area / max(1,N-1)` where smallest rank=0
7. `confidence`
8. `bbox_extent_ratio`
9. `width_ratio = bbox_width / 512`
10. `height_ratio = bbox_height / 512`
11. `fill_ratio = area_px / max(1,bbox_area)`
12. `proposal_count_norm = min(N,300)/300`

Append a 2-d one-hot family indicator:
- largest = `[1,0]`
- smallest = `[0,1]`

Final per-proposal vector = 14 dims.

Do NOT use:
- GT-derived features;
- target relation;
- centroid x/y;
- image location;
- SAM2 features;
- target mask;
- source feature id.

## 15. Exact ranker architecture

Create:
`buildreasonseg_mvp/task6u_reference_ranker.py`

Shared per-proposal MLP:

```text
Linear(14 → 32)
GELU
Linear(32 → 16)
GELU
Linear(16 → 1)
```

For each candidate set:
- score all eligible proposals;
- softmax over proposal scores;
- cross-entropy against the labelled positive proposal.

No attention/Transformer/GNN.

## 16. Training

Create deterministic 90/10 internal split of ranker-trainable U-RankerTrain records:
- seed `20260930`;
- stratify by family;
- split by reference key hash.

Train:
- AdamW
- lr = `1e-3`
- weight_decay = `1e-4`
- batch = 64 reference sets, using padded/masked candidate tensors as needed
- max epochs = 50
- early stopping patience = 6
- selection metric:
  1. internal holdout top-1 reference selection accuracy;
  2. tie-break mean selected-reference IoU
- seed `20260930`
- no hyperparameter sweep.

Checkpoint:
`artifacts/checkpoints/task6u/reference_ranker_v01.pt`
local/gitignored.

Write:
`evaluation/task6u_ranker_training.json`

---

# PART H — Three selectors on untouched RefValUnique

## 17. Compare exactly three systems

### U-S0 Baseline
- U-C0 proposals
- frozen Task 6Q deterministic area selector

### U-S1 Candidate-hardening only
- selected U-C* proposals
- same Task 6Q deterministic area selector

### U-S2 Candidate-hardening + learned ranker
- selected U-C* proposals
- ProposalSetRanker v0.1

No fourth selector.

## 18. Reference metrics

For U-S0/U-S1/U-S2 report on exact RefValUnique:

- selected-reference mIoU;
- Dice;
- Pr@0.5;
- centroid error mean/median/p90;
- area-ratio median;
- abstention rate;
- largest mIoU / Pr@0.5;
- smallest mIoU / Pr@0.5.

Failure attribution exactly:

1. `NO_PROPOSALS`
2. `NO_ELIGIBLE_PROPOSALS`
3. `REFERENCE_NOT_COVERED_IOU50`
4. `REFERENCE_SELECTION_WRONG`
5. `SELECTED_MASK_GEOMETRY_POOR`
6. `REFERENCE_OK`

Use the same definitions as Task 6Q.

Write:
`evaluation/task6u_refval_selector_comparison.json`

---

# PART I — Downstream causal evaluation

## 19. Isolate reference module

Use canonical/oracle program IDs from the frozen records.

Do NOT use natural-language parser in the main U-S0/U-S1/U-S2 comparison.

This keeps Task 6U causal:
proposal/reference change only.

For MiniVal240:

```text
canonical program id
→ family/relation
→ U-S0/U-S1/U-S2 reference resolver
→ GeometricRelationField v0.2
→ frozen SAM2 visual feature
→ frozen B3
→ target mask
```

GT target/reference only for evaluation.

Report for U-S0/U-S1/U-S2:

- strict all-240 target mIoU/Dice;
- answered-only mIoU/Dice;
- Pr@0.5;
- abstentions;
- largest/smallest;
- per direction;
- tiny target if present;
- border target.

Failure attribution using Task 6S categories except parser bucket must be zero/not applicable.

Write:
`evaluation/task6u_downstream_minival240.json`

## 20. PairedVal20

For all three systems:

- pass / 20;
- mean own IoU;
- mean cross IoU;
- own-cross margin;
- reference-abstention pairs.

Write:
`evaluation/task6u_downstream_pairedval20.json`

---

# PART J — Full natural-language regression with hardened parser

## 21. One final integration check

If U-S2 exists normally, run the current Task 6T hardened ProgramHead + U-S2 reference path + frozen field/B3 on MiniVal240.

This is not used to train or choose U-S2.

Require parser:
- 240/240 exact.

Report:
- strict mIoU;
- answered-only mIoU;
- PairedVal;
- margin;
- abstentions.

Write:
`evaluation/task6u_hardened_parser_integration.json`

Do not evaluate nearest/L3 execution.

---

# PART K — Predeclared success criteria

## 22. Candidate-coverage improvement flag

`candidate_coverage_improved = true` iff selected config on RefValUnique satisfies BOTH:

- overall eligible coverage@0.50 >= baseline + `0.04`;
- smallest eligible coverage@0.50 >= baseline + `0.06`.

Baseline Task 6Q values for context:
- overall ≈ 0.6895;
- largest ≈ 0.8364;
- smallest ≈ 0.5413.

Use exact recomputed U-S0 values in the comparison.

## 23. Ranker improvement flag

`ranker_improved_selection = true` iff on RefValUnique:

- U-S2 selected-reference mIoU >= U-S1 + `0.05`;
- U-S2 `REFERENCE_SELECTION_WRONG` count <= `0.70 * U-S1` count;
- U-S2 abstention rate <= `0.10`.

## 24. Downstream hardening gate

Task 6U is considered a useful reference hardening only if U-S2 satisfies ALL:

1. RefVal selected-reference mIoU >= `0.48`
2. RefVal centroid median <= `0.03`
3. RefVal centroid p90 <= `0.25`
4. MiniVal240 answered-only target mIoU >= `0.33`
5. MiniVal240 strict target mIoU >= `0.31`
6. PairedVal >= `12/20`
7. own-cross margin >= `0.30`
8. MiniVal240 reference-fail count <= `93`
   - at least 20% reduction from Task 6S `117`
9. no test split
10. no GT in inference.

Do not alter thresholds after results.

---

# PART L — Verdict

## 25. Exactly one, priority order

1. `INVALID_EXPERIMENT`
   - leakage/test/frozen-module mutation/GT inference/protocol violation.

2. `PROPOSAL_CHECKPOINT_UNAVAILABLE`

3. `REFERENCE_CANDIDATE_COVERAGE_STILL_LIMITING`
   - selected config does not improve candidate coverage and U-S2 fails downstream hardening.

4. `REFERENCE_RANKER_NOT_HELPFUL`
   - coverage improves or is usable, but `ranker_improved_selection=false` and U-S2 fails downstream hardening.

5. `REFERENCE_HARDENING_PARTIAL`
   - U-S2 improves reference/downstream metrics materially but misses one or more section-24 gates.

6. `REFERENCE_HARDENING_FEASIBLE`
   - all section-24 gates pass.

No other verdict.

---

# PART M — Interpretation boundary

DSH may report measurements only.

Do NOT:
- claim ProposalSetRanker is a project novelty;
- choose to retrain YOLO;
- choose a different detector;
- decide to keep 1024 permanently beyond the frozen selected config;
- add tiling/TTA;
- alter smallest threshold;
- start nearest/L3;
- retrain B3;
- add GRCL;
- choose the next architecture.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 6U 的 candidate coverage、ranker 与 downstream 因果结果决定下一步，不自行修改 detector、reference 语义或 target 架构。`

---

# PART N — Required artifacts

Create at minimum:

```text
buildreasonseg_mvp/task6u_reference_ranker.py

evaluation/task6u_reference_train_split.json
evaluation/task6u_calibration_candidate_coverage.json
evaluation/task6u_selected_proposal_config.json
evaluation/task6u_refval_candidate_audit.json
evaluation/task6u_ranker_training.json
evaluation/task6u_refval_selector_comparison.json
evaluation/task6u_downstream_minival240.json
evaluation/task6u_downstream_pairedval20.json
evaluation/task6u_hardened_parser_integration.json
evaluation/task6u_verdict.json

docs/task6u_reference_hardening.md

scripts/task6u_freeze_reference_split.py
scripts/task6u_candidate_coverage.py
scripts/task6u_train_ranker.py
scripts/task6u_evaluate_reference.py
scripts/task6u_evaluate_downstream.py
scripts/task6u_report.py
```

Proposal caches:
`artifacts/task6u/proposals/<config>/`
gitignored.

Ranker checkpoint:
`artifacts/checkpoints/task6u/reference_ranker_v01.pt`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

---

# PART O — Required tests

Task 6T ended at:
`807 passed, 1 skipped`

Add tests covering at least:

1. Task 6T artifacts unchanged
2. hardened parser checkpoint not retrained
3. YOLO checkpoint SHA exact
4. exactly four candidate configs
5. C0 exact baseline
6. C1 exact values
7. C2 exact values
8. C3 exact values
9. no TTA
10. no tiling
11. source masks restored to 512×512
12. Task 6Q eligibility unchanged
13. U-Calib200 train-only
14. U-RankerTrain train-only
15. U-Calib/U-RankerTrain disjoint
16. RefVal untouched by config selection
17. selected-config priority rule exact
18. ranker feature dimension exactly 14
19. no centroid/location feature in ranker
20. no GT feature in ranker
21. no relation input to ranker
22. ranker architecture 14→32→16→1
23. family one-hot exact
24. ranker train labels use GT only offline
25. uncovered refs excluded from ranker loss and counted
26. U-S0 exact Task 6Q behavior
27. U-S1 deterministic area selector
28. U-S2 learned ranker
29. only three selectors
30. MiniVal240 exact reuse
31. PairedVal20 exact reuse
32. main comparison uses canonical program ids
33. final integration uses hardened ProgramHead
34. no oracle reference in inference
35. no GT target in inference
36. field v0.2 unchanged
37. B3 unchanged
38. no YOLO training
39. no parser training
40. no GRCL
41. no nearest/L3 execution
42. no test split
43. no new dataset/download/install/GUI
44. previous passing suite preserved.

Run:

`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# PART P — Git/storage

Do not commit:
- YOLO/SAM2/B3/parser/ranker weights;
- proposal caches;
- feature caches;
- source imagery/vectors;
- `.conda`;
- large generated local data.

Commit:
- ranker code;
- small JSON artifacts;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: add proposal-set reference ranker`
2. `eval: isolate reference coverage and ranking gains`
3. optional `docs: record Task 6U reference hardening`

---

# PART Q — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Flash + Max only for genuine CUDA/runtime/cross-file implementation bugs.

Do not use V4 Pro by default.

No installs/downloads. The required model/assets already exist locally.

---

# PART R — STOP

After Task 6U:
- commit;
- push;
- handoff;
- STOP.

Do NOT:
- retrain YOLO;
- add tiling/TTA;
- change thresholds outside the four declared configs;
- change the ranker;
- train another reference model;
- change B3/field/SAM2;
- revisit GRCL;
- add nearest/L3 target execution;
- access test;
- build GUI.

Wait for ChatGPT audit.
