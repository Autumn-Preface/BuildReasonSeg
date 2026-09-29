# TO_DSH — Task 6R: Directional Geometry-Relation Consistency Loss (GRCL) Feasibility

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `7c19bec577282029d0eb0eac7187940262c1a63a`
>
> Predecessor: Task 6Q → `REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`
>
> Research decision: ChatGPT accepts the Task 6Q proposal resolver as a **usable but imperfect support path** because its downstream target chain already reaches the predeclared target mIoU / paired / own-cross thresholds. Do not spend this task optimizing the resolver.
>
> Task 6R returns to the project's core algorithm contribution and tests a precisely defined relation-level training regularizer. DSH is an executor; do not redesign the loss or architecture.

## 0. DSH role

All user-facing DSH output must be Chinese.

DSH MAY:
- implement the exact GRCL formula below;
- reproduce frozen Task 6O B3 relation metrics;
- train exactly the specified R1/R2 variants;
- evaluate on the frozen Task 6N validation packs;
- run a diagnostic transfer evaluation with the frozen Task 6Q proposal reference resolver;
- solve ordinary code/runtime bugs that do not alter the experiment.

DSH MUST NOT:
- change GeometricRelationField v0.2;
- change SAM2;
- change B3 architecture;
- tune lambda;
- add another relation loss;
- change alpha/tau;
- add `[REF]`;
- add nearest/L3;
- add MLLM hidden-state fusion;
- retrain YOLO;
- improve the proposal resolver;
- use test split;
- add a dataset;
- choose Task 6S.

If any such change is required, STOP and report.

# PART A — Research question

## 1. Question

Tasks 6N/6O established that an explicit reference-conditioned GeometricRelationField improves dense target segmentation.

Task 6R tests the second candidate algorithm contribution:

> Does **explicit supervision on the geometric relation between the predicted target mask and the reference mask** improve relation correctness and counterfactual target discrimination beyond field guidance alone, without materially degrading mask quality?

This task handles only four directional relations:
- left_of
- right_of
- above
- below

Only the same 8 directional L2 programs are in scope.

No nearest, no L3.

## 2. Literature-position note

Copy this fixed note into Task 6R docs; do not perform a new literature search:

- Recent RRSIS work already contains “consistency” losses/regularizers, including text–vision structural consistency and cross-modal alignment consistency.
- Therefore the word “consistency” itself is not novel.
- The candidate contribution tested here is narrower: **a mask-level reference–target geometric relation constraint whose variables are the predicted target mask centroid, the grounded reference mask centroid, and the instruction-specified spatial relation**.
- Do not claim “first-ever” or final novelty from Task 6R alone.

# PART B — Frozen evidence and scope

## 3. Freeze previous artifacts

Read-only:

- all Task 6N artifacts;
- all Task 6O artifacts;
- all Task 6P artifacts;
- all Task 6Q artifacts;
- `buildreasonseg_mvp/geometric_relation_field_v02.py`;
- Task 6O B3 checkpoint;
- frozen SAM2 feature cache;
- Task 6N packs:
  - Overfit20
  - MiniTrain1000
  - MiniVal240
  - PairedVal20
- BuildSpatialReason v0.2;
- WHU native-vector v1.0;
- Task 3B spatial relation config.

No test split.

## 4. Exact program scope

Only:

- `largest_to_left_of`
- `largest_to_right_of`
- `largest_to_above`
- `largest_to_below`
- `smallest_to_left_of`
- `smallest_to_right_of`
- `smallest_to_above`
- `smallest_to_below`

Reference source for the main causal training experiment:
`oracle_native_gt`

This is intentional to isolate the loss.

GT target remains mask label/evaluation only.

# PART C — GRCL v0.1 exact mathematical definition

## 5. Create loss module

Create:

`buildreasonseg_mvp/grcl_directional.py`

Input:
- target logits `Z_t`, shape `(B,1,H,W)` at any resolution;
- oracle reference mask `M_ref`, shape compatible with batch;
- relation id in `{left_of,right_of,above,below}`.

Constants:
- `alpha = 1.2`
- `tau = 0.04`
- `eps = 1e-6`
- `lambda_grcl = 0.5`

`alpha` and `tau` must match the frozen Task 3B relation semantics.

No sweep.

## 6. Soft target centroid

```text
P_t = sigmoid(Z_t)
mass_t = sum(P_t) + eps

cx_t = sum(P_t * x_grid) / mass_t
cy_t = sum(P_t * y_grid) / mass_t
```

Use pixel-centre normalized coordinates `[0,1]`.

Reference centroid:

```text
mass_r = sum(M_ref) + eps
cx_r = sum(M_ref * x_grid) / mass_r
cy_r = sum(M_ref * y_grid) / mass_r
```

Do not detach target logits.
Do not threshold target probabilities inside the loss.
Oracle reference can be treated as fixed.

## 7. Signed primary displacement and orthogonal displacement

For each relation:

### left_of
```text
signed = cx_r - cx_t
orth = abs(cy_t - cy_r)
```

### right_of
```text
signed = cx_t - cx_r
orth = abs(cy_t - cy_r)
```

### above
```text
signed = cy_r - cy_t
orth = abs(cx_t - cx_r)
```

### below
```text
signed = cy_t - cy_r
orth = abs(cx_t - cx_r)
```

## 8. Loss terms

Use exactly:

```text
L_margin = relu(tau - signed)
L_axis   = relu(alpha * orth - signed)

L_GRCL = L_margin + L_axis
```

Batch reduction = mean.

Total training loss for R1:

```text
L_total = L_BCE + L_Dice + 0.5 * L_GRCL
```

No normalization by tau.
No softplus.
No additional sign loss.
No field loss.
No contrastive loss.
No pair loss.

## 9. Gradient audit before training

Create a deterministic synthetic audit:

- at least 8 soft target-logit tensors;
- all 4 relations represented;
- non-symmetric reference masks;
- backprop `L_GRCL`.

Require:
- finite loss;
- target-logit grad exists;
- grad finite;
- grad L1 > `1e-8`.

Also numerical directional sanity:
- moving a synthetic target centroid farther into the correct relation must not increase GRCL;
- moving it across the reference to the wrong side must increase GRCL.

Write:
`evaluation/task6r_grcl_audit.json`

If fail:
STOP with `GRCL_IMPLEMENTATION_INVALID`.

# PART D — Frozen baseline B3 relation evaluation

## 10. Reproduce frozen B3

Use Task 6O B3 checkpoint, verify hash from Task 6O artifact.

Reproduce MiniVal240:
- mIoU
- Dice

Require absolute delta <= `1e-6`.

Do not retrain B3.

## 11. Define hard relation-correctness metric

This is evaluation only.

For each predicted target mask:
1. threshold sigmoid target output at `0.5`;
2. if mask is empty → relation incorrect;
3. compute predicted-mask centroid;
4. compute oracle-reference centroid;
5. compute `signed` and `orth` using section 7;
6. relation is correct iff BOTH:

```text
signed >= tau
signed >= alpha * orth
```

Report:
- relation accuracy overall;
- per direction;
- mean positive signed margin `signed - tau`;
- axis violation rate.

Compute this for frozen B3 on MiniVal240 and PairedVal20.

Write:
`evaluation/task6r_b3_relation_baseline.json`

# PART E — Variants

## 12. R0 = frozen B3 baseline

R0 is not trained in Task 6R.

Architecture:
`visual + P_rel + relation embedding → target mask`

Loss history:
BCE + Dice only.

Use frozen Task 6O B3 metrics/checkpoint.

## 13. R1 = B3 + GRCL

Architecture EXACTLY B3:
- frozen SAM2 visual features;
- GeometricRelationField v0.2 from oracle reference;
- relation embedding;
- B3 decoder.

Only change:
training loss includes GRCL v0.1.

No direct reference-mask decoder channel.

## 14. R2 = no-field + GRCL control

Architecture EXACTLY Task 6N B0:
- visual feature;
- relation embedding;
- no reference mask input;
- no geometric relation field input.

Training loss:
BCE + Dice + 0.5 * GRCL.

Important:
- oracle reference mask may be used by GRCL during training and by relation evaluation;
- it is NOT an R2 model input.

Purpose:
test whether GRCL alone can replace the explicit field.

# PART F — Stage R1: Overfit20

## 15. Train R1 and R2

Use exact frozen Task 6N Overfit20.

For both variants:

- AdamW
- lr `1e-3`
- weight_decay `1e-4`
- max steps `1200`
- batch `4`
- no scheduler
- no augmentation
- seed `20260929`
- same bfloat16 AMP policy as Tasks 6N/6O
- evaluate every 100 steps

Record:
- mIoU
- Dice
- hard relation accuracy
- mean GRCL
- paired own/cross on available pairs
- peak VRAM
- wall time.

### R1 overfit gate

R1 must reach:
- mIoU >= `0.85`
- Dice >= `0.90`
- hard relation accuracy >= `0.95`

If fail:
STOP with `GRCL_OVERFIT_FAIL`.

R2 has no stop gate.

# PART G — Stage R2: MiniTrain1000 → MiniVal240

## 16. Train from fresh initialization

Only if R1 overfit gate passes.

Train R1 and R2 separately from fresh initialization.

Exact training schedule:

- AdamW
- lr `3e-4`
- weight_decay `1e-4`
- batch `8`
- max epochs `25`
- early stopping patience `5`
- model selection metric = MiniVal240 mIoU
- seed `20260929`
- no scheduler
- no augmentation
- same AMP.

Do not access test.

Write:
`evaluation/task6r_training.json`

# PART H — Main evaluation

## 17. MiniVal240

For R0 / R1 / R2 report:

- mIoU
- Dice
- Pr@0.5
- hard relation accuracy
- per-relation mIoU
- per-relation relation accuracy
- largest/smallest family mIoU
- mean GRCL
- signed-margin statistics
- axis violation rate
- border-target mIoU
- tiny-target mIoU if present
- params
- selected epoch
- wall time
- peak VRAM.

Write:
`evaluation/task6r_mini_val.json`

## 18. PairedVal20

For R0/R1/R2 use exact frozen PairedVal20.

Report:
- pass / 20
- mean own IoU
- mean cross IoU
- own-cross margin
- hard relation accuracy of the 40 member predictions.

Write:
`evaluation/task6r_paired_val.json`

# PART I — Proposal-reference transfer diagnostic

## 19. R1 under frozen Task 6Q reference resolver

Diagnostic only; no training and no gate tuning.

Use frozen Task 6Q proposal resolver and exact frozen proposal configuration.

For MiniVal240:
- resolve reference with Task 6Q;
- create field v0.2;
- run trained R1 target decoder;
- abstain exactly when Task 6Q resolver abstains.

Report:
- target mIoU
- Dice
- paired pass
- own-cross margin
- hard relation accuracy
- abstentions.

Compare against frozen Task 6Q B3 proposal-reference chain:
- mIoU `0.3045812554881724`
- paired `10/20`
- own-cross margin `0.27370032940000916`

Write:
`evaluation/task6r_proposal_reference_transfer.json`

Do NOT use this diagnostic to select lambda or retrain anything.

# PART J — Predeclared success criteria

## 20. Main GRCL feasibility criteria

First compute frozen R0 hard relation accuracy using section 11.

R1 is considered a positive GRCL signal only if ALL:

1. R1 overfit gate passes.
2. MiniVal mask retention:
   - `R1 mIoU >= R0 mIoU - 0.02`
   - numerical threshold = `0.4099680351479113`
3. Relation correctness gain:
   - `R1 relation_accuracy >= R0 relation_accuracy + 0.08`
4. Paired discrimination:
   - R1 PairedVal >= `16/20`
5. R1 own-cross margin >= `0.38`
6. Field remains materially useful:
   - `R1 mIoU - R2 mIoU >= 0.10`

Do not alter these criteria.

## 21. Strong signal flag

In addition to the verdict, report:

`strong_mask_gain = true`

only if:
- R1 mIoU >= R0 mIoU + `0.02`

This flag is NOT required for feasibility.

# PART K — Verdict

## 22. Exactly one, priority order

1. `INVALID_EXPERIMENT`
   - leakage, test access, frozen mutation, target-as-input or protocol violation.

2. `GRCL_IMPLEMENTATION_INVALID`

3. `GRCL_OVERFIT_FAIL`

4. `GRCL_MASK_RELATION_TRADEOFF`
   - R1 relation accuracy improves by >= 0.08, but mask-retention criterion fails.

5. `GRCL_NO_MEANINGFUL_RELATION_GAIN`
   - mask retention passes, but relation accuracy gain < 0.08 OR paired < 16/20.

6. `GRCL_FIELD_REDUNDANCY_RISK`
   - preceding criteria pass but `R1-R2 mIoU < 0.10`.

7. `GRCL_DIRECTIONAL_FEASIBLE`
   - all section 20 criteria pass.

No other verdict.

# PART L — Interpretation boundary

DSH may report measured results only.

Do NOT:
- claim final novelty;
- change lambda;
- design another relation loss;
- start nearest/L3;
- start MLLM integration;
- redesign reference grounding.

Final handoff recommendation exactly:

`等待 ChatGPT 根据 Task 6R 的 GRCL 因果实验结果决定 Task 6S，不自行修改 loss、reference 架构或开始 MLLM/nearest/L3。`

# PART M — Required artifacts

Create:

```text
buildreasonseg_mvp/grcl_directional.py

evaluation/task6r_grcl_audit.json
evaluation/task6r_b3_relation_baseline.json
evaluation/task6r_overfit20.json
evaluation/task6r_training.json
evaluation/task6r_mini_val.json
evaluation/task6r_paired_val.json
evaluation/task6r_proposal_reference_transfer.json
evaluation/task6r_verdict.json

docs/task6r_grcl_directional_feasibility.md

scripts/task6r_grcl_audit.py
scripts/task6r_train.py
scripts/task6r_evaluate.py
scripts/task6r_proposal_transfer.py
scripts/task6r_report.py
```

Checkpoints:
`artifacts/task6r/checkpoints/`
gitignored.

Reuse all frozen feature/proposal caches.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

# PART N — Tests

## 23. Required tests

At least:

1. Task 6Q artifacts unchanged
2. Task 6O B3 hash exact
3. GeometricRelationField v0.2 unchanged
4. alpha exactly 1.2
5. tau exactly 0.04
6. lambda_grcl exactly 0.5
7. target centroid is differentiable
8. target logits receive finite nonzero GRCL gradient
9. no thresholding inside GRCL
10. hard relation metric threshold = 0.5 only for evaluation
11. left sign exact
12. right sign exact
13. above sign exact
14. below sign exact
15. axis term exact
16. margin term exact
17. R1 architecture equals B3
18. R1 has field
19. R1 no direct ref channel
20. R2 architecture equals B0
21. R2 has no field input
22. R2 has no ref-mask model input
23. oracle ref used by R2 only in loss/eval
24. exact frozen packs reused
25. same training settings R1/R2
26. no test split
27. GT target never model input
28. proposal transfer uses frozen Task 6Q resolver/config
29. proposal transfer does not tune
30. no YOLO retraining
31. no `[REF]`
32. no nearest/L3
33. no MLLM hidden-state fusion
34. no graph transformer
35. no 4B
36. no new dataset/download/install/GUI
37. previous suite preserved

Run:
`python -m pytest tests/ -q`

Task 6Q ended at **696 passed, 1 skipped**.
Do not reduce previous passing tests.

# PART O — Git/storage

Do not commit:
- checkpoints
- proposal caches
- feature caches
- SAM2/YOLO weights
- source imagery/vector data
- `.conda`
- large caches

Commit code/small JSON/docs/tests/handoff only.

Recommended:
1. `feat: add directional geometry-relation consistency loss`
2. `eval: measure GRCL relation and mask effects`
3. optional docs/handoff commit

# PART P — Model policy

Default:
- DeepSeek V4.1 Flash + High

Use Flash + Max only for genuine implementation/runtime bugs.

Do not use V4 Pro by default.

# PART Q — STOP

After Task 6R:
- commit
- push
- handoff
- STOP

Do not start:
- MLLM integration
- nearest
- L3
- joint reference-target training
- new reference architecture
- full-dataset training
- GUI

Wait for ChatGPT audit.
