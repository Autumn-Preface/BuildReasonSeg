# TO_DSH — Task 7H: Development Architecture Freeze + Formal Experiment Protocol

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `22401feedd42e349de425849415fc2fd4caa66d2`
>
> Predecessor: Task 7G → `LARGEST_SELECTOR_NOT_LEARNABLE`
>
> Research decision already made by ChatGPT:
>
> 1. **Task 7G ends reference intervention for the current project version.**
> 2. Do NOT run Task 7G.1, train another selector, retrain YOLO, or redesign the reference stage.
> 3. The final development reference path remains the frozen deterministic U-C1 path.
> 4. **D-B1 is frozen as the preferred L3 target-decoder architecture candidate** because its gain over Z-B3
>    was independently verified on Task 7E untouched oracle-reference holdout.
> 5. D-B1 is NOT claimed end-to-end ready; the practical reference stage remains a measured limitation.
> 6. Parser hardening also remains stopped. The Task 7C parser is usable for canonical/program-template
>    development input, while free-form L3 paraphrase robustness is a known limitation.
> 7. Task 7H performs **no training and no test evaluation**. Its purpose is to freeze:
>    - the development architecture;
>    - module/checkpoint roles;
>    - supported/unsupported capability claims;
>    - the exact formal-training / validation / final-test protocol to be executed only in later tasks.
> 8. This task is a research-governance freeze. DSH is an executor and must not choose another architecture.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- audit existing tracked artifacts and local frozen checkpoints;
- compute hashes and metadata from already-authorized local assets;
- assemble an evidence registry;
- write the exact architecture freeze and future experiment protocol;
- add regression tests that guarantee future tasks cannot silently substitute modules.

DSH MUST NOT:
- train any model;
- run inference on the test split;
- read test images/masks/annotations to compute a new metric;
- retrain YOLO/ProgramHead/reference/D-B1;
- run the Task 7G selector on E-Holdout after its STOP;
- change thresholds, fields, losses, model structures or data splits;
- download/install anything;
- choose Task 7I implementation details beyond the protocol explicitly frozen here.

If a frozen asset is missing or hash-inconsistent, STOP and report.

---

# PART A — Accept Task 7G result exactly

## 1. Record formal Task 7G result

Copy into `docs/task7h_development_architecture_freeze.md`:

Task 7G verdict:
`LARGEST_SELECTOR_NOT_LEARNABLE`

Internal tile-disjoint selector holdout:

```text
G-I0 deterministic:
mean selected reference IoU      0.5199913587
oracle-best exact top-1          0.4887892377
mean best-minus-selected gap     0.2789719922

G-I1 learned:
mean selected reference IoU      0.6198877726
median selected IoU              0.7895902547
Pr(selected IoU >=0.50)          0.7309417040
oracle-best exact top-1          0.6591928251
mean best-minus-selected gap     0.1790755783
mean IoU gain over G-I0         +0.0998964138
```

Predeclared gate:
- gain >= +0.08 → PASS
- mean selected IoU >= 0.62 → FAIL (`0.6198878`)
- oracle top-1 >= 0.55 → PASS
- mean gap <= 0.14 → FAIL (`0.1790756`)

Therefore:
- external E-HoldoutL3 stage did not run;
- Task 7G selector is not adopted;
- no scene-disjoint result may be claimed for Task 7G selector;
- reference intervention stops.

## 2. Preserve the correct interpretation

Record explicitly:

The Task 7G selector showed meaningful **internal** improvement, but did not clear the predeclared internal
learnability gate. It is scientifically incorrect to call it a scene-disjoint failure because the external
scene-disjoint stage was never executed.

The development system therefore keeps the deterministic selector, not because it is best in principle, but
because no learned replacement has passed the frozen adoption protocol.

---

# PART B — Evidence hierarchy

## 3. Create evidence registry

Create:

`evaluation/task7h_evidence_registry.json`

For every module below record:
- role;
- task of origin;
- artifact/checkpoint path;
- SHA256 if checkpoint/file is local;
- whether frozen;
- evidence population;
- verified result;
- claim boundary.

Required entries:

### Data
- `WHU-EA-NativeVector v1.0`
- `BuildSpatialReason v0.2`
- `scene_disjoint_v1`

### Proposal model
- Task 6M.1 YOLO26m-seg
- checkpoint SHA256:
  `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`

### Program parser
- Task 7C Qwen3-VL-2B text-only ProgramHead checkpoint
- expected SHA256:
  `c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a`
- canonical 20-class behaviour: verified
- free-form L3 paraphrase robustness: NOT solved

### Direction field
- `GeometricRelationField v0.2`

### Nearest field
- `NearestBoundaryField v0.1`

### Directional L2 decoder
- frozen Task 6O N-B3 checkpoint/artifact
- role: verified directional-field visual segmentation baseline/development component
- do not substitute N-B2/B4.

### L3 baseline decoder
- Task 6Z Z-B3
- expected checkpoint SHA256:
  `74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc`
- role: frozen L3 baseline/ablation

### Preferred L3 target decoder
- Task 7D D-B1
- expected checkpoint SHA256:
  `6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0`
- role: preferred L3 target-decoder architecture candidate
- evidence:
  Task 7E untouched oracle-reference holdout 669 records:
  - Z-B3 mIoU 0.3141113773
  - D-B1 mIoU 0.3854957053
  - delta +0.0713843280
  - bootstrap CI [+0.0586233648,+0.0842626436]
  - D-B1 paired 18/20
- practical predicted-reference chain:
  - D-B1 strict 0.2454050104
  - limitation attributed mainly to reference selection.

### Rejected/not-adopted support modules
Record as frozen negative evidence only:
- Task 6P ReferenceMaskHead
- Task 6U ProposalSetRanker v0.1
- Task 6W ProposalQualityEstimator v0.1
- Task 6X SAM2 proposal refinement
- Task 7G SetContextLargestSelector v1
- Task 7D D-B2 learned global competition.

---

# PART C — Freeze the development architecture

## 4. Development architecture name

Use the descriptive internal name:

`BuildReasonSeg-DevFreeze-2026-10`

This is NOT a new model checkpoint.

## 5. Development natural-language front end

Freeze:

```text
user instruction
→ Task 7C Qwen3-VL-2B text-only ProgramHead
→ one of the 20 canonical spatial programs
```

Claim boundary:

- Verified: canonical/template-like 20-class program classification.
- NOT verified: robust unrestricted free-form language.
- Therefore documentation must say:
  `canonical-program / controlled-language development interface`
  rather than:
  `arbitrary natural-language understanding`.

Do not hide the Task 7C fixed24/stress weakness.

## 6. Proposal/reference stage

Freeze proposal generator:

```text
YOLO26m-seg Task 6M.1
imgsz=640
conf=0.05
max_det=300
default NMS
no TTA
no tiling
```

Freeze deterministic reference semantics:

### largest
- family eligibility as Task 6Q/6U;
- maximum predicted mask area;
- tie: higher confidence;
- then lower original index.

### smallest
- preserve exact Task 6Q smallest eligibility/ranking if a supported L2 path requires it.

Task 7G selector:
- NOT part of development chain.

Claim boundary:
- reference proposal selection is a known practical bottleneck;
- Task 7F proves selection has a large oracle ceiling;
- current project version does not contain a learned selector that passed adoption gates.

## 7. Relation representation

Freeze:

```text
Reference mask
→ GeometricRelationField v0.2 for directional relations
→ NearestBoundaryField v0.1 for nearest relation
```

For L3:
- use both fields separately;
- deterministic product is used only for D-B1 prototype weighting as defined by Task 7D;
- do not replace the two decoder-visible fields with only the product.

## 8. Directional L2 target path

Freeze the Task 6O N-B3 role:

```text
reference
+ directional field
+ frozen SAM2 dense visual feature
+ relation embedding
→ N-B3 directional target decoder
```

Evidence boundary:
- field-guided visual segmentation supported;
- do not claim universal relation reasoning.

## 9. Nearest-only L2 status

Record:

`nearest-only = experimental/limited`

Task 6Y B2:
- field semantics verified;
- clear gain over visual baseline;
- failed absolute/generalization/counterfactual feasibility gate.

Nearest-only is NOT a headline final capability.

Do not silently promote it to a validated final module.

## 10. L3 target path — preferred architecture

Freeze D-B1:

```text
predicted/oracle reference
→ P_dir
→ P_near
→ W = clamp(P_dir * P_near,0,1)
→ A_fixed = W / (sum(W)+eps)
→ frozen SAM2 dense feature F
→ q = Σ A_fixed_i F_i
→ cosine similarity C(F_i,q)
→ decoder input:
   F + P_dir + P_near + direction embedding + A_fixed_vis + C
→ target mask
```

No learned competition head.

D-B1 role:
`preferred L3 target-decoder architecture candidate`

Practical development chain:

```text
controlled instruction
→ Task 7C ProgramHead
→ U-C1 proposals
→ deterministic reference selector
→ P_dir + P_near
→ D-B1
→ target mask
```

Do NOT call this:
- final paper model;
- unrestricted natural-language model;
- end-to-end solved system.

---

# PART D — Freeze current limitations

## 11. Create limitation registry

Create:

`evaluation/task7h_limitation_registry.json`

It must contain at least:

### L-01 Reference selection
Evidence:
- Task 7F F-R0 0.245405
- oracle candidate selection F-R1 0.364909
- selection gain +0.119504
- Task 7G not adopted.

Status:
`unresolved practical bottleneck`

### L-02 Proposal coverage
Evidence:
- U-C1 best eligible coverage@0.50 = 0.846039
- coverage gain +0.054316.

Status:
`secondary unresolved bottleneck`

### L-03 Proposal-mask geometry
Evidence:
- Task 7F covered-subset geometry gain +0.004253.

Status:
`not a major current bottleneck`

### L-04 Free-form L3 language
Evidence:
- Task 7C canonical full-val 1.0
- fixed24 5/24
- compact 0/8
- stress 0.6667.

Status:
`controlled-language interface only`

### L-05 nearest-only
Evidence:
- Task 6Y B2 MiniVal 0.2864
- paired 6/20
- verdict `NEAREST_FIELD_NO_MEANINGFUL_GAIN`.

Status:
`not validated as standalone final capability`

### L-06 unseen-city/domain generalization
Status:
`not established`

The native split demonstrates raster/scene separation, not unseen-city generalization.

---

# PART E — Formal experimental protocol freeze

## 12. Purpose

Create:

`evaluation/task7h_formal_experiment_protocol.json`

and describe it in:

`docs/task7h_formal_experiment_protocol.md`

Task 7H does NOT execute the protocol.

## 13. Formal data policy

Use only active canonical:
- WHU-EA-NativeVector v1.0
- BuildSpatialReason v0.2
- `scene_disjoint_v1`.

Future formal training:
- train split only for gradient updates;
- val split only for checkpoint selection / early stopping / frozen threshold-free model selection;
- test split only after all architecture and training choices are frozen.

Important disclosure:

The scene-disjoint test split was historically accessed in Task 6M for an earlier proposal-baseline J4-v2 audit.
Therefore future reporting must NOT call the project test split "never previously viewed" or "completely
untouched". It may be called:

`final frozen-architecture test evaluation`

because post-6M architecture selection did not use test metrics.

Do not access test records in Task 7H.

## 14. Formal L3 training population

Future formal L3 target-decoder training shall use ALL valid train records from:

- largest_to_left_of_to_nearest
- largest_to_right_of_to_nearest
- largest_to_above_to_nearest
- largest_to_below_to_nearest

Expected historical counts from Task 6Z:
- left 323
- right 347
- above 338
- below 336
- total 1344.

Task 7H may verify these counts only from existing tracked manifests/statistics.
Do not read test samples.

Validation:
- use all valid v0.2 val L3 records;
- historical total 936.

## 15. Formal D-B1 retraining plan

Freeze the architecture/hyperparameters to the current D-B1 design.

Future formal training must:
- initialize D-B1 from fresh random trainable decoder/projection weights;
- keep SAM2 frozen;
- use exact field formulas;
- use exact D-B1 architecture;
- use BCE+Dice only;
- no new loss;
- no architecture tuning.

Training schedule SHALL be frozen in Task 7H by reading the authoritative Task 7D D-B1 training artifact and
copying its exact:
- optimizer;
- learning rate;
- weight decay;
- batch size;
- maximum epochs;
- early stopping patience;
- AMP;
- checkpoint selection metric.

Do not invent values if an authoritative tracked artifact exists.

Formal seeds:
- `20261001`
- `20261002`
- `20261003`

Run all three only in a later training task.

Checkpoint selection:
- val mIoU only;
- each seed selected independently;
- no seed chosen by test.

## 16. Reference policy in formal L3 evaluation

Future formal evaluation must report BOTH:

### Practical predicted-reference chain
```text
U-C1
→ deterministic largest selector
→ D-B1
```

### Oracle-reference target-decoder diagnostic
```text
GT reference
→ D-B1
```

Do not mix these into one headline number.

Do not use Task 7F F-R1 oracle-selected proposal as a production result.

## 17. Formal baselines/ablations

Freeze L3 comparison roles:

### B-L3-0
Z-B3:
`visual + P_dir + P_near + direction embedding`
Role: baseline/ablation.

### B-L3-1
D-B1:
`deterministic relation-conditioned prototype`
Role: main target-decoder architecture candidate.

### Historical causal controls
Use existing frozen evidence where appropriate:
- Task 6Z visual-only / directional-only / nearest-only / geometry-only;
- Task 7D learned competition D-B2;
- Task 7D map-only D-B4.

Do not invent a new retrospective ablation architecture in Task 7H.

Formal documentation must distinguish:
- historical development ablations;
- future formally retrained seeds;
- diagnostic oracle-reference values.

## 18. Required formal metrics

Freeze:

### Mask
- mIoU
- Dice
- Pr@0.5

### Counterfactual
- pair pass rate
- own IoU
- cross IoU
- own-cross margin

### Reference
- selected-reference mIoU
- reference Pr@0.5
- abstention rate
- NO_PROPOSALS
- NO_ELIGIBLE
- NOT_COVERED
- SELECTION_WRONG
- GEOMETRY_POOR
- REFERENCE_OK

### Per relation
- left
- right
- above
- below

### Stratification
- target area quartiles
- reference quality bins
- boundary-distance quartiles

### Efficiency
- train wall time
- peak VRAM
- inference time/tile
- trainable parameter count.

## 19. Three-seed reporting

For formal D-B1:
- report each seed separately;
- report mean ± standard deviation on val and final test;
- do not report only the best test seed.

If a run crashes:
- document it;
- resume only from that run's own checkpoint/state;
- do not replace the seed.

## 20. Formal test lock

Create:

`evaluation/task7h_test_lock.json`

Exact required state:

```text
status = "LOCKED"
architecture_head = "D-B1"
reference_policy = "U-C1 deterministic largest"
parser_role = "controlled-language/canonical interface"
formal_seeds = [20261001,20261002,20261003]
test_execution_authorized = false
unlock_condition = "ChatGPT audit after formal train/val completion"
```

Future DSH task must check this file before test execution.

Task 7H itself must not unlock it.

---

# PART F — Paper/Challenge Cup claim matrix

## 21. Create claim registry

Create:

`evaluation/task7h_claim_registry.json`

Every claim must be tagged:
- `SUPPORTED`
- `SUPPORTED_WITH_LIMITATION`
- `NOT_SUPPORTED`
- `DIAGNOSTIC_ONLY`.

At minimum:

### C1
`Explicit reference-conditioned directional geometric fields improve same-class building target segmentation.`
Status:
`SUPPORTED`
Evidence:
Tasks 6N/6O.

### C2
`Directional and nearest geometric priors can be composed for L3 direction→nearest reasoning.`
Status:
`SUPPORTED_WITH_LIMITATION`
Evidence:
Task 6Z paired strength and D-B1 holdout gain; practical performance remains reference-limited.

### C3
`Relation-conditioned deterministic spatial weighting can extract a useful target visual prototype.`
Status:
`SUPPORTED`
Evidence:
Task 7D + untouched Task 7E holdout.

### C4
`Learned global competition is superior.`
Status:
`NOT_SUPPORTED`
Evidence:
Task 7D D-B2.

### C5
`The practical end-to-end system solves unrestricted natural-language L3 segmentation.`
Status:
`NOT_SUPPORTED`
Evidence:
Task 7C language robustness + Task 7E practical chain.

### C6
`Reference selection is the dominant practical bottleneck.`
Status:
`SUPPORTED_WITH_LIMITATION`
Evidence:
Task 7F ceilings; no learned replacement passed Task 7G adoption gate.

### C7
`Proposal-mask geometry is the main reference problem.`
Status:
`NOT_SUPPORTED`
Evidence:
Task 7F geometry gain +0.004253.

### C8
`Performance generalizes to unseen cities.`
Status:
`NOT_SUPPORTED`

### C9
`D-B1 is final paper-ready architecture.`
Status:
`NOT_SUPPORTED`
It is a frozen development architecture candidate pending formal retraining/test.

No novelty/"first" claim in this registry.

---

# PART G — Freeze verdict

## 22. Allowed verdicts

Priority:

1. `INVALID_EXPERIMENT`
2. `FROZEN_ASSET_MISSING`
3. `FROZEN_ASSET_HASH_MISMATCH`
4. `FREEZE_PROTOCOL_INCONSISTENT`
5. `DEVELOPMENT_ARCHITECTURE_FROZEN`

Return `DEVELOPMENT_ARCHITECTURE_FROZEN` only if ALL:

1. required frozen checkpoint hashes match;
2. active dataset identity is v0.2/native-vector;
3. no rejected selector/ranker is in the development chain;
4. D-B1 role is correct;
5. parser limitation is recorded;
6. reference limitation is recorded;
7. formal protocol artifact is complete;
8. test lock is LOCKED;
9. no training occurred;
10. no test was accessed.

No other verdict.

---

# PART H — Required artifacts

Create:

```text
evaluation/task7h_evidence_registry.json
evaluation/task7h_limitation_registry.json
evaluation/task7h_formal_experiment_protocol.json
evaluation/task7h_test_lock.json
evaluation/task7h_claim_registry.json
evaluation/task7h_verdict.json

docs/task7h_development_architecture_freeze.md
docs/task7h_formal_experiment_protocol.md

scripts/task7h_audit_frozen_assets.py
scripts/task7h_build_protocol.py
scripts/task7h_report.py
```

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No checkpoint creation.

---

# PART I — Tests

Task 7G ended at:
`1377 passed, 1 skipped`

Add tests for at least:

1. Task 7G artifacts unchanged
2. Task 7G selector not adopted
3. external Task 7G scene-disjoint result not claimed
4. active dataset is BuildSpatialReason v0.2
5. native-vector v1.0 recorded
6. YOLO hash exact
7. Task 7C parser hash exact
8. Z-B3 hash exact
9. D-B1 hash exact
10. directional field source unchanged
11. nearest field source unchanged
12. D-B1 role exact
13. Z-B3 baseline role exact
14. Task 7G selector excluded
15. ProposalSetRanker excluded
16. ProposalQualityEstimator excluded
17. SAM2 refinement excluded
18. learned global competition D-B2 excluded
19. deterministic reference policy exact
20. controlled-language parser limitation recorded
21. free-form L3 robustness not claimed
22. nearest-only limitation recorded
23. unseen-city generalization not claimed
24. D-B1 oracle holdout evidence exact
25. Task 7F bottleneck evidence exact
26. formal L3 train programs exact four
27. formal seeds exact
28. formal metrics complete
29. practical vs oracle results separated
30. future val is model-selection only
31. test not authorized
32. test lock status LOCKED
33. historical Task 6M test access disclosed
34. no "untouched test" claim
35. claim registry statuses exact
36. no novelty/"first" claim
37. no training in Task 7H
38. no test inference
39. no new dataset/download/install/GUI
40. prior passing suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# PART J — Git/storage

Do not commit:
- model checkpoints;
- proposal/feature caches;
- imagery/vectors;
- local generated model data;
- `.conda`.

Commit:
- freeze/protocol JSON;
- docs;
- audit scripts;
- tests;
- handoff.

Suggested commits:
1. `docs: freeze BuildReasonSeg development architecture`
2. `eval: freeze formal experiment and test-lock protocol`

---

# PART K — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

This task is mostly audit/documentation.
Use Max only for a genuine cross-artifact inconsistency.

No installs/downloads.

---

# PART L — STOP

After Task 7H:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- start Task 7I formal training;
- unlock test;
- run test;
- train another selector;
- retrain YOLO/parser/D-B1;
- modify the architecture.

Final recommendation exactly:

`等待 ChatGPT 审核 Task 7H 的开发版架构冻结与正式实验协议；在审核通过前不启动正式三种子训练，不解锁 test。`
