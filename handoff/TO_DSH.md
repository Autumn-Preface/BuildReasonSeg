# TO_DSH — Task 7G: Largest-Reference Set-Context Selector — Final Reference Intervention

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `c59c6d310849c99b3d8f6b1b2e0116c74dc8e932`
>
> Predecessor: Task 7F → `REFERENCE_SELECTION_DOMINANT`
>
> Research decision already made by ChatGPT:
>
> 1. Task 7F established that the remaining practical L3 bottleneck is dominated by **proposal selection**, not
>    proposal-mask geometry:
>    - current D-B1 + F-R0 strict mIoU = 0.245405;
>    - oracle selection over the SAME U-C1 proposal set = 0.364909;
>    - selection gain = +0.119504;
>    - coverage gain = +0.054316;
>    - covered-subset proposal-mask geometry gain = only +0.004253.
> 2. The current U-C1 proposal set has a usable ceiling:
>    - coverage@0.50 = 0.8460;
>    - F-R2 strict mIoU = 0.3312;
>    - F-R2 paired = 16/20.
> 3. Therefore one **final reference intervention** is justified: learn to select the canonical `largest`
>    reference from the existing frozen U-C1 proposal set.
> 4. Do NOT retrain YOLO, refine masks, alter proposal thresholds, or reopen generic reference hardening.
> 5. Task 7G trains a **largest-only permutation-invariant proposal-set selector** using train-split GT only as
>    ranking supervision.
> 6. The selector uses proposal geometry/confidence and proposal-set overlap structure only. It does NOT use
>    SAM2 visual features, image position, relation/direction, target data, parser features, or GT at inference.
>    This deliberately avoids repeating Task 6W's scene-disjoint visual-quality transfer failure.
> 7. If Task 7G fails the external scene-disjoint gate, reference hardening stops permanently for the current
>    project version. DSH must not invent Task 7G.1.
>
> DSH is an executor. Do not redesign the features, model, training protocol, gates, or next task.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- generate frozen U-C1 proposals on train-split unique largest-reference records;
- construct the exact 18-dimensional proposal feature vector below;
- train the exact set-context selector below;
- evaluate the frozen selector on Task 7E E-HoldoutL3 and E-PairedHoldout20;
- integrate it with the frozen D-B1 decoder for evaluation;
- solve ordinary implementation/runtime bugs without changing the experiment.

DSH MUST NOT:
- retrain YOLO;
- change U-C1 `imgsz/conf/max_det/NMS`;
- change largest eligibility;
- use SAM2 features in the selector;
- use image RGB features in the selector;
- use centroid x/y or absolute proposal location;
- use direction/relation/program id in the selector;
- use target mask/id/features;
- train on val/test;
- use E-HoldoutL3 for checkpoint selection or early stopping;
- modify D-B1;
- modify relation fields;
- add attention/Transformer/GNN;
- add ProposalSetRanker v0.1 into the primary selector;
- add ProposalQualityEstimator;
- use test split;
- choose the next task.

If a prohibited change is required, STOP and report.

---

# PART A — Freeze Task 7F evidence

## 1. Task 7F formal result

Copy into `docs/task7g_largest_reference_set_context_selector.md`:

Task 7F verdict:
`REFERENCE_SELECTION_DOMINANT`

Authoritative gap decomposition:

```text
M0 F-R0 current selected predicted mask = 0.2454050104
M1 F-R1 oracle-selected predicted mask = 0.3649093893
M2 F-R2 coverage-conditional GT mask    = 0.3311796697
M3 F-R3 full oracle GT mask             = 0.3854959324

selection_gain        = +0.1195043789
coverage_gain         = +0.0543162627
geometry_gain_covered = +0.0042531670
total_reference_gap   = +0.1400909220
```

Task 7F labels:
- `SELECTION_IS_ACTIONABLE = true`
- `PROPOSAL_GEOMETRY_IS_MAJOR = false`
- `PROPOSAL_COVERAGE_IS_MAJOR = true`
- `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING = true`.

Reference ceiling:
- F-R0 reference mIoU = 0.440647
- F-R1 reference mIoU = 0.711942
- best eligible coverage@0.50 = 0.846039.

Paired:
- F-R0 = 6/20, margin +0.125063
- F-R1 = 18/20, margin +0.300548
- F-R2 = 16/20, margin +0.330636
- F-R3 = 18/20, margin +0.319340.

## 2. Task 7F reporting erratum

Record this without mutating Task 7F artifacts:

`buildreasonseg_mvp/task7f_reference_ceiling.py` correctly stores both:
- `selected_confidence/selected_area` for the F-R0 current selector;
- `oracle_selected_confidence/oracle_selected_area` for F-R1.

However, `scripts/task7f_reference_modes.py::summarise_references` reports the generic
`selected_confidence_mean/selected_area_mean` fields for both F-R0 and F-R1, so the F-R1 values shown in the
report are not F-R1-specific proposal statistics.

This reporting-only issue does NOT affect:
- the F-R1 mask itself;
- F-R1 reference mIoU;
- F-R1 downstream metrics;
- paired metrics;
- gap decomposition;
- Task 7F verdict.

Do not rewrite frozen Task 7F artifacts.

---

# PART B — Frozen assets

## 3. Proposal generator

Use exactly:

```text
YOLO26m-seg Task 6M.1 checkpoint
SHA256 = ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474

imgsz = 640
conf = 0.05
max_det = 300
default NMS
no TTA
no tiling
source tile = original 512×512
```

## 4. Largest eligibility

Use exact Task 6Q/U-C1 `largest` eligibility:

- proposal mask non-empty;
- not border-touching;
- bbox extent ratio <= 0.20.

No `area >= 150` rule for largest.

No additional quality threshold.

## 5. Frozen downstream

Read-only:
- D-B1 checkpoint
  `artifacts/checkpoints/task7d/db1_minitrain1200.pt`
- D-B1 SHA256
  `6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0`
- GeometricRelationField v0.2
- NearestBoundaryField v0.1
- frozen SAM2.1 Hiera Base+ visual features
- Task 7E E-HoldoutL3 / E-PairedHoldout20.

No D-B1 training.

---

# PART C — Build the train-only largest-reference selection dataset

## 6. Source records

Use **BuildSpatialReason v0.2 train split only**.

Select every train record whose canonical program contains a `largest` reference and whose metadata provides the
canonical `reference_source_feature_id`.

This includes valid largest-reference records from:
- L2 directional programs;
- `largest_to_nearest`;
- L3 largest→direction→nearest programs;
- any other frozen v0.2 program whose reference is explicitly the canonical largest building.

Do NOT include:
- `smallest` reference records;
- L1 extreme-only records with no explicit reference field if their record schema does not provide the same
  canonical reference id;
- val/test records.

## 7. Deduplicate

Deduplicate by exact key:

```text
(split, tile_id, reference_source_feature_id)
```

Call the result:
`G-RefTrainUniqueLargest`

The same reference appearing in multiple programs contributes exactly one proposal-set example.

Record:
- source rows;
- unique references;
- unique tiles;
- per source-program counts;
- duplicate rows removed.

## 8. Proposal generation and ranking label

For each unique training reference:

1. generate/reuse frozen U-C1 proposals;
2. apply exact largest eligibility;
3. compute IoU of every eligible proposal mask against canonical GT reference mask;
4. if no eligible proposal:
   - count `no_eligible`;
   - exclude from selector loss;
5. if best eligible IoU < 0.50:
   - count `untrainable_not_covered`;
   - exclude from selector loss;
6. otherwise:
   - target candidate index = maximum GT IoU;
   - tie higher YOLO confidence;
   - then lower original proposal index.

GT is label construction only.

At selector inference GT is unavailable.

## 9. Minimum data gate

Require:
- selector-trainable unique references >= `300`;
- at least `50` unique train tiles in the eventual internal holdout.

If not:
STOP:
`SELECTOR_TRAIN_DATA_INSUFFICIENT`

Write:
`evaluation/task7g_training_dataset_manifest.json`

Large proposal rows/caches:
`artifacts/task7g/selector_dataset/`
gitignored.

---

# PART D — Exact 18-dimensional proposal feature vector

## 10. General rules

Feature extraction uses only:
- the candidate's predicted binary mask;
- its predicted bbox;
- YOLO confidence;
- the other eligible predicted proposals in the SAME tile.

No GT-derived value is an input feature.

No absolute centroid x/y.
No direction/program/relation.
No image pixel/RGB/SAM2 feature.
No target information.

For one eligible proposal `i`, let:
- source area `S = 512*512`;
- mask area `a_i`;
- bbox width/height `w_i,h_i`;
- bbox area `b_i`;
- number of eligible proposals `N`;
- `eps = 1e-6`.

## 11. Features 1–10: candidate geometry/confidence

Exactly:

1. `log_area = log1p(a_i) / log1p(S)`
2. `area_ratio = a_i / S`
3. `area_over_set_max = a_i / max_j(a_j)`
4. `area_rank_desc = rank_desc(i) / max(1,N-1)`
   - largest area rank = 0
   - stable tie by higher confidence then lower original proposal index
5. `confidence`
6. `bbox_extent_ratio = max(w_i/512, h_i/512)`
7. `width_ratio = w_i / 512`
8. `height_ratio = h_i / 512`
9. `fill_ratio = a_i / max(1,b_i)`
10. `abs_log_aspect = abs(log((w_i+1)/(h_i+1)))`

## 12. Feature 11: mask boundary complexity

Use `scipy.ndimage.binary_erosion` with:
- 3×3 all-ones structuring element;
- one iteration;
- border_value=0.

```text
boundary = mask_i AND NOT eroded(mask_i)
boundary_px = sum(boundary)

boundary_over_sqrt_area =
    boundary_px / sqrt(max(1,a_i))
```

If SciPy unexpectedly unavailable:
STOP `SELECTOR_DEPENDENCY_UNAVAILABLE`.
Do not install.

## 13. Features 12–17: proposal-set overlap structure

For each `j != i`:

```text
inter_ij = |mask_i ∩ mask_j|
iou_ij = inter_ij / |mask_i ∪ mask_j|
self_in_other_ij = inter_ij / max(1,a_i)
other_in_self_ij = inter_ij / max(1,a_j)
```

If `N=1`, all overlap features below are exactly 0.

12. `max_iou_other = max_j iou_ij`
13. `mean_iou_other = mean_j iou_ij`
14. `max_self_contained_by_other = max_j self_in_other_ij`
15. `max_other_contained_in_self = max_j other_in_self_ij`
16. `self_contained_count_norm =
      count_j(self_in_other_ij >= 0.50) / max(1,N-1)`
17. `contains_other_count_norm =
      count_j(other_in_self_ij >= 0.50) / max(1,N-1)`

## 14. Feature 18: set size

18. `proposal_count_norm = min(N,300)/300`

Final feature dim:
`18`

No other feature.

---

# PART E — SetContextLargestSelector v1

## 15. Create module

Create:

`buildreasonseg_mvp/task7g_largest_reference_selector.py`

## 16. Per-proposal encoder

For proposal feature vector `x_i ∈ R^18`:

```text
Linear(18 → 64)
LayerNorm(64)
GELU
Linear(64 → 32)
GELU
```

Output:
`h_i ∈ R^32`

Shared weights for all proposals.

## 17. Permutation-invariant set context

For the candidate set:

```text
h_mean = mean_i(h_i)  # 32
h_max  = max_i(h_i)   # 32
context = concat(h_mean,h_max)  # 64
```

Broadcast the same context to every candidate.

No positional encoding.
No proposal-order feature.

## 18. Candidate score head

For candidate i:

```text
z_i = concat(h_i, context)  # 96

Linear(96 → 32)
GELU
Linear(32 → 1)
```

One scalar score per eligible proposal.

Softmax over valid candidates only.

No Transformer.
No attention.
No GNN.
No pairwise learned interaction beyond the frozen overlap features and pooled set context.

Total parameter count must be reported.

---

# PART F — Loss and training

## 19. Listwise loss

For each trainable proposal set:
- target = section-8 oracle-best candidate index.

Loss:
`CrossEntropyLoss(candidate_scores, target_index)`

Exactly one loss.

No:
- IoU regression;
- pairwise margin loss;
- quality loss;
- focal loss;
- auxiliary loss.

## 20. Internal split

Split `G-RefTrainUniqueLargest` at the **tile id** level before proposal-set rows enter train/holdout.

Deterministic:
- seed `20261001`;
- 80% train tiles;
- 20% internal holdout tiles;
- SHA256-based assignment.

Require:
- zero tile overlap;
- zero `(tile_id,reference_source_feature_id)` overlap;
- >=50 holdout tiles.

Uncovered/no-eligible records remain counted in their assigned partition but are excluded from loss/selector
accuracy denominators.

## 21. Optimization

Train from fresh selector initialization.

Exact:
- AdamW
- lr = `1e-3`
- weight_decay = `1e-4`
- batch = `32 proposal sets`
- max epochs = `40`
- early stopping patience = `6`
- seed = `20261001`
- no scheduler
- no augmentation
- FP32 is acceptable; AMP optional but unnecessary
- no hyperparameter sweep.

Checkpoint selection on internal holdout trainable sets ONLY:

1. highest mean selected-reference GT IoU;
2. tie higher oracle-best exact top-1 accuracy;
3. tie lower mean `best_iou - selected_iou`;
4. tie earlier epoch.

Do not use E-HoldoutL3 for checkpoint selection.

Checkpoint:

`artifacts/checkpoints/task7g/largest_set_context_selector_v1.pt`

gitignored.

Write:
`evaluation/task7g_training.json`

---

# PART G — Internal holdout learnability gate

## 22. Compare deterministic baseline vs new selector

On internal holdout trainable proposal sets compare:

### G-I0
Current deterministic max-area selector.

### G-I1
Frozen selected Task 7G checkpoint.

Report:
- mean selected-reference IoU;
- median;
- Pr(selected IoU>=0.50);
- oracle-best exact top1 match rate;
- mean best-minus-selected IoU gap.

## 23. Internal gate

Continue to external scene-disjoint E-Holdout only if ALL:

- G-I1 mean selected IoU >= G-I0 + `0.08`
- G-I1 mean selected IoU >= `0.62`
- G-I1 oracle-best exact top1 match >= `0.55`
- G-I1 mean best-minus-selected gap <= `0.14`

If fail:
STOP:
`LARGEST_SELECTOR_NOT_LEARNABLE`

Do not alter features/model/loss.

---

# PART H — External E-HoldoutL3 selection audit

Only if section 23 passes.

## 24. Exact evaluation set

Reuse byte-for-byte:
- Task 7E E-HoldoutL3 = 669 records;
- Task 7E E-PairedHoldout = 20 pairs.

Verify hashes from Task 7E/7F.

No new sampling.

## 25. Compare exact modes

### G-S0
Frozen current production-like selector:
- U-C1 proposals
- deterministic max predicted mask area.

Must reproduce Task 7F F-R0.

### G-S1
Task 7G learned selector:
- exact same eligible U-C1 proposals;
- compute 18D features;
- SetContextLargestSelector;
- highest score;
- score tie within exact float equality:
  - higher YOLO confidence;
  - lower original proposal index.

Returns the selected **predicted proposal mask**.

No GT.

### G-ORACLE
Task 7F F-R1 oracle-selected predicted proposal.
Diagnostic ceiling only; reuse frozen Task 7F value, do not use GT to influence G-S1.

## 26. Reference metrics

For G-S0/G-S1 report:

- reference mIoU;
- Dice;
- Pr@0.5;
- centroid median/p90;
- abstentions;
- oracle-best exact top1 match rate;
- mean best-eligible GT IoU;
- mean selected GT IoU;
- mean `best_iou - selected_iou`;
- number where G-S1 improves selected-reference IoU over G-S0 by >=0.10;
- number where G-S1 worsens by >=0.10;
- per direction.

Write:
`evaluation/task7g_external_reference.json`

GT only for offline metrics.

---

# PART I — Frozen D-B1 external downstream audit

## 27. Main pipeline

For every E-HoldoutL3 record:

```text
canonical L3 program
→ U-C1 proposals
→ G-S0 or G-S1 largest reference selector
→ selected predicted reference mask
→ P_dir v0.2
→ P_near v0.1
→ frozen SAM2 feature
→ frozen D-B1
→ target mask
```

No parser.

No GT inference input for G-S0/G-S1.

## 28. Metrics

For G-S0/G-S1:

- strict mIoU/Dice;
- answered-only mIoU/Dice;
- Pr@0.5;
- reference-IoU>=0.50 subset target mIoU;
- per direction;
- abstentions.

G-S0 must reproduce Task 7F:
- strict `0.24540501038500215`
- answered `0.24614085749260334`
- abstentions 2.

Tolerance:
- floats `1e-6`
- counts exact.

Write:
`evaluation/task7g_external_downstream.json`

If reproduction fails:
STOP:
`TASK7F_BASELINE_REPRODUCTION_FAIL`

---

# PART J — External paired counterfactual

## 29. E-PairedHoldout20

Use exact Task 7E pairs.

For each pair:
- one shared tile/reference;
- run selector once for that shared reference;
- reuse reference for both directions.

Report G-S0/G-S1:
- pass /20;
- mean own IoU;
- mean cross IoU;
- own-cross margin;
- abstention pairs.

G-S0 must reproduce:
- 6/20
- margin `0.1250628820` within 1e-6.

Write:
`evaluation/task7g_external_paired.json`

---

# PART K — Predeclared external success gate

## 30. `LARGEST_SELECTOR_EXTERNAL_PASS`

Requires ALL:

### Reference
1. G-S1 reference mIoU >= `0.55`
2. G-S1 reference mIoU >= G-S0 + `0.08`
3. G-S1 Pr@0.5 >= `0.65`
4. G-S1 mean best-minus-selected IoU gap <= `0.15`

### Downstream D-B1
5. G-S1 strict mIoU >= `0.30`
6. G-S1 strict mIoU >= G-S0 + `0.05`
7. G-S1 answered-only mIoU >= `0.30`

### Counterfactual
8. G-S1 paired >= `12/20`
9. G-S1 own-cross margin >= `0.20`

### Safety
10. abstention rate <= `0.01`
11. no GT enters selector inference
12. no test split.

Do not change gates after results.

## 31. Oracle-ceiling capture diagnostics

Report, but not separately gate:

```text
selection_gain_ceiling = Task7F_F-R1_mIoU - Task7F_F-R0_mIoU
learned_selection_gain = G-S1_strict_mIoU - G-S0_strict_mIoU

ceiling_capture_fraction =
    learned_selection_gain / selection_gain_ceiling
```

Also at reference level:

```text
reference_ceiling_capture =
    (G-S1_ref_mIoU - G-S0_ref_mIoU)
    / (F-R1_ref_mIoU - F-R0_ref_mIoU)
```

---

# PART L — Development-chain decision

## 32. If external gate passes

Set:

`TASK7G_SELECTOR_ADOPTED = true`

Development L3 chain becomes:

```text
canonical program
→ U-C1 proposals
→ Task 7G largest set-context selector
→ predicted reference
→ P_dir + P_near
→ frozen D-B1
→ target mask
```

Roles:
- D-B1 = development L3 decoder candidate
- Task 7G selector = development largest-reference selector
- Z-B3 = frozen decoder baseline/ablation
- deterministic max-area selector = frozen reference baseline/ablation
- F-R1/F-R2/F-R3 remain diagnostic only.

No full training/test in Task 7G.

## 33. If external gate fails

Set:

`TASK7G_SELECTOR_ADOPTED = false`

Then:
- do NOT invent another selector;
- do NOT retrain YOLO;
- preserve D-B1 as preferred oracle-reference decoder candidate;
- preserve U-C1 + deterministic selector as practical baseline;
- mark practical Reference selection as an unresolved limitation;
- stop reference intervention for the current project version.

---

# PART M — Verdict

## 34. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `SELECTOR_DEPENDENCY_UNAVAILABLE`
3. `SELECTOR_TRAIN_DATA_INSUFFICIENT`
4. `LARGEST_SELECTOR_NOT_LEARNABLE`
5. `TASK7F_BASELINE_REPRODUCTION_FAIL`
6. `LARGEST_SELECTOR_SCENE_DISJOINT_FAIL`
   - internal gate passes but section-30 external gate fails.
7. `LARGEST_SELECTOR_DEVELOPMENT_READY`
   - all section-30 gates pass.

No other verdict.

---

# PART N — Interpretation boundary

DSH may report measurements only.

Do NOT:
- call the selector a project novelty;
- claim final end-to-end readiness;
- train another selector after failure;
- retrain YOLO;
- modify D-B1;
- access test;
- start formal full-data training;
- choose the next architecture/task.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7G 的 scene-disjoint selector 与 frozen D-B1 结果决定是否冻结开发版完整 L3 链；若 gate 失败，不自行继续 reference 干预。`

---

# PART O — Required artifacts

Create:

```text
buildreasonseg_mvp/task7g_largest_reference_selector.py

evaluation/task7g_training_dataset_manifest.json
evaluation/task7g_training.json
evaluation/task7g_internal_holdout.json
evaluation/task7g_external_reference.json
evaluation/task7g_external_downstream.json
evaluation/task7g_external_paired.json
evaluation/task7g_verdict.json

docs/task7g_largest_reference_set_context_selector.md

scripts/task7g_build_selector_dataset.py
scripts/task7g_train_selector.py
scripts/task7g_evaluate_selector.py
scripts/task7g_evaluate_downstream.py
scripts/task7g_report.py
```

If a STOP gate fires early:
- create completed-stage artifacts + verdict/docs/handoff;
- do not fabricate later artifacts.

Local:
```text
artifacts/task7g/
artifacts/checkpoints/task7g/
```
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

Do not change production CLI defaults yet.

---

# PART P — Required tests

Task 7F ended at:
`1325 passed, 1 skipped`

Add tests for at least:

1. Task 7F artifacts unchanged
2. Task 7F reporting erratum documented, not artifact-mutated
3. U-C1 exact config
4. YOLO checkpoint hash exact
5. largest eligibility exact
6. train source is BuildSpatialReason v0.2 train only
7. only explicit largest-reference records used
8. dedup key exact
9. no val/test selector training
10. uncovered/no-eligible excluded from loss and counted
11. label = best eligible GT-IoU proposal
12. label tie-break exact
13. feature dim exactly 18
14. no GT-derived input feature
15. no centroid x/y
16. no relation/direction/program feature
17. no target feature
18. no RGB/SAM2 feature
19. boundary feature formula exact
20. overlap feature formulas exact
21. proposal-count feature exact
22. shared proposal encoder exact
23. set mean pool exact
24. set max pool exact
25. candidate head exact 96→32→1
26. no attention/Transformer/GNN
27. listwise CE only
28. tile-level 80/20 internal split
29. train/holdout tile overlap zero
30. internal checkpoint selection uses no E-Holdout
31. internal gate exact
32. E-Holdout hashes exact
33. G-S0 deterministic baseline exact
34. G-S1 no GT inference
35. G-S1 same proposal set as G-S0
36. D-B1 checkpoint unchanged
37. fields unchanged
38. SAM2 unchanged
39. G-S0 downstream reproduction exact
40. paired reference reused within pair
41. external gate exact
42. no ProposalSetRanker v0.1 in primary G-S1
43. no ProposalQualityEstimator
44. no mask refinement
45. no YOLO retraining
46. no D-B1 training
47. no test split
48. no new dataset/download/install/GUI
49. previous passing suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# PART Q — Git/storage

Do not commit:
- selector checkpoint;
- YOLO/SAM2/D-B1/Z-B3 weights;
- proposal caches;
- large generated selector rows;
- source imagery/vectors;
- `.conda`.

Commit:
- selector architecture code;
- small manifests/evaluation JSON;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: add largest-reference set-context selector`
2. `eval: test final reference selection intervention`
3. optional docs/handoff commit

---

# PART R — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for genuine selector-data/runtime bugs.

No installs or downloads.

---

# PART S — STOP

After Task 7G:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- run Task 7G.1;
- train another selector;
- retrain YOLO;
- modify D-B1;
- access test;
- start formal full training.

Wait for ChatGPT audit.
