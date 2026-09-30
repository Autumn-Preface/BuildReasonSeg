# TO_DSH — Task 7C: 20-Class Rehearsal + L3 ProgramHead Hardening

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `b325585c18ddef1ee80cf05d4fc971c5c19b477a`
>
> Predecessor: Task 7B → `L3_PARSER_CANONICAL_REGRESSION`
>
> Research decision already made by ChatGPT:
>
> 1. Task 7B is accepted as a valid negative diagnostic: it improved L3 paraphrase robustness but catastrophically regressed canonical 20-class behaviour because only 14/20 classes were represented in training.
> 2. Task 7C MUST restart from the stable Task 6T checkpoint, not the failed Task 7B checkpoint.
> 3. Task 7C uses all-20-class synthetic rehearsal while preserving L3 compositional emphasis.
> 4. The Task 7A/7B downstream chain remains frozen.
> 5. No keyword/regex correction is allowed.
> 6. Task 7C is the final parser-hardening attempt before ChatGPT freezes parser status.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- reuse the already-frozen Task 7A/7B evaluation packs;
- generate the exact 8,000-prompt rehearsal set below;
- train the same Qwen3-VL-2B text-only LoRA + ProgramHead from Task 6T;
- run the fixed final evaluation once;
- update the L3 CLI default parser only after canonical gates pass;
- fix ordinary runtime bugs without altering the experiment.

DSH MUST NOT:
- initialize from Task 7B;
- change model family/size;
- add image input or a second classifier;
- change the 20 classes;
- use original v0.2 train instruction strings for optimization;
- use any val/test/eval prompt for optimization;
- use Task 7A fixed24 / Task 7B minimal96 / stress192 strings in training;
- add keyword/regex semantic remapping;
- modify YOLO/reference/SAM2/fields/Z-B3;
- use test split;
- run a hyperparameter sweep;
- choose Task 7D.

If a prohibited change is required, STOP and report.

---

# PART A — Preserve Task 7B evidence and record the design correction

## 1. Record Task 7B result

Copy into `docs/task7c_20class_rehearsal_l3_hardening.md`:

- verdict: `L3_PARSER_CANONICAL_REGRESSION`
- full v0.2 val accuracy `0.7129`
- full-val macro F1 `0.7939`
- minimum class recall `0.0000`
- Z-MiniVal240 `240/240`
- Z-Paired members `40/40`
- Task 7A fixed24 `21/24`
- compact `7/8`
- minimal pairs `73/96`
- stress `0.8281`

Task 7B training used 4,800 synthetic rows but only 14/20 classes. The six zero-support classes were:

- `leftmost`
- `rightmost`
- `topmost`
- `bottommost`
- `largest`
- `smallest`

The internal Task 7B holdout also lacked those six classes, so 1.0 holdout accuracy could not detect forgetting.

This is a **ChatGPT protocol-design error**, not a DSH implementation error.

Do not mutate Task 7B artifacts or verdict.

---

# PART B — Baseline and architecture freeze

## 2. Initialization checkpoint

Start ONLY from:

`artifacts/checkpoints/task6t/program_parser_hardened_v1.pt`

Expected SHA256:

`4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e`

Do not initialize from Task 7B.

If missing/hash mismatch:
`BASELINE_PARSER_UNAVAILABLE`

## 3. Parser architecture

Preserve exactly:
- Qwen3-VL-2B;
- text-only;
- same tokenizer/prompt format;
- existing Task 6M/6T LoRA parameters;
- same ProgramHead;
- exactly 20 canonical classes;
- base backbone frozen outside existing LoRA trainables.

No image tokens. No new class. No new head.

## 4. Downstream freeze

Read-only:
- Task 7A U-C1 reference resolver;
- YOLO checkpoint/config;
- GeometricRelationField v0.2;
- NearestBoundaryField v0.1;
- frozen SAM2 visual path;
- Task 6Z Z-B3 checkpoint.

No downstream training.

---

# PART C — Reuse frozen evaluation packs

## 5. Exact reuse

Do NOT regenerate/edit/extend:
- full BuildSpatialReason v0.2 val parser set;
- Task 6Z Z-MiniVal240;
- Task 6Z Z-PairedVal20 member queries;
- Task 7A fixed24 paraphrases;
- Task 7B `task7b_compositional_minimal_pairs.json` (96);
- Task 7B `task7b_l3_stress_v1.json` (192);
- Task 7B scope controls.

No test.

Write SHA256 reuse manifest:
`evaluation/task7c_reused_eval_manifest.json`

---

# PART D — No original template training

## 6. Original v0.2 train text

Do NOT use original BuildSpatialReason v0.2 train instruction strings for optimization.

Reason: the generator uses the same finite instruction-template vocabulary across train and val, so original train text would leak into full-val parser evaluation.

Original train may be read only to verify class ids/schema.

---

# PART E — Exact 8,000-prompt all-class rehearsal

## 7. Canonical classes

Exactly these 20:

```text
leftmost
rightmost
topmost
bottommost
largest
smallest
largest_to_nearest
smallest_to_nearest
largest_to_above
largest_to_below
largest_to_left_of
largest_to_right_of
smallest_to_above
smallest_to_below
smallest_to_left_of
smallest_to_right_of
largest_to_above_to_nearest
largest_to_below_to_nearest
largest_to_left_of_to_nearest
largest_to_right_of_to_nearest
```

## 8. Exact allocation

Generate exactly **8,000** unique normalized prompts.

### Four L3 classes
Each L3 class:
- 800 total
- 400 Chinese
- 400 English

Total = 3,200.

### Remaining 16 classes
Each:
- 300 total
- 150 Chinese
- 150 English

Total = 4,800.

Grand total = 8,000.

## 9. Semantic contrast construction

Training must explicitly cover:
- L1 extreme contrasts;
- largest vs smallest;
- nearest-only;
- largest L2 direction;
- smallest L2 direction;
- L3 direction+nearest.

L2 and L3 templates must share some structural stems so the classifier must learn the terminal nearest relation rather than a style cue.

---

# PART F — Train lexical pools

## 10. Chinese TRAIN vocabulary

Use combinations from pools such as:

Extremes:
- `最靠左的`
- `最靠右的`
- `位置最高的`
- `位置最低的`
- `占地面积最大的`
- `占地面积最小的`

Reference family:
- `面积最大的建筑`
- `占地最大的建筑`
- `面积最小的建筑`
- `占地最小的建筑`

Direction:
- `位于其左侧的` / `处在其左方的`
- `位于其右侧的` / `处在其右方的`
- `位于其上方的` / `处在其上侧的`
- `位于其下方的` / `处在其下侧的`

Nearest:
- `其中距离最近的`
- `其中与参考建筑间距最小的`
- `其中最接近参考建筑的`
- nearest-only: `与其距离最近的` / `与其间距最小的`

## 11. English TRAIN vocabulary

Extremes:
- `the farthest-left building`
- `the farthest-right building`
- `the uppermost building`
- `the lowermost building`
- `the building with the greatest footprint`
- `the building with the least footprint`

Reference:
- `the building with the largest footprint`
- `the greatest-area building`
- `the building with the smallest footprint`
- `the least-area building`

Direction:
- `located on its left side` / `situated to its left`
- `located on its right side` / `situated to its right`
- `located above it` / `situated on its upper side`
- `located below it` / `situated on its lower side`

Nearest:
- `the one with the minimum separation`
- `the one nearest to the reference`
- `the closest one among them`
- nearest-only: `the building with the minimum separation from it`

Do NOT reproduce any exact/normalized evaluation prompt.

---

# PART G — Leakage audit

## 12. Zero-overlap requirement

Before training compare all 8,000 prompts against the union of section-5 evaluation prompts.

Normalization:
- Unicode strip;
- lowercase English;
- collapse whitespace;
- normalize common punctuation;
- strip terminal punctuation only for normalized comparison.

Require:
- exact train/eval overlap = 0;
- normalized train/eval overlap = 0;
- normalized duplicates inside training = 0;
- exact per-class counts;
- exact zh/en counts.

Write:
`evaluation/task7c_training_data_audit.json`

If fail:
`INVALID_EXPERIMENT`

Store large rows locally under:
`artifacts/task7c/parser_rehearsal/`

Tracked generation spec:
`evaluation/task7c_rehearsal_spec.json`

---

# PART H — Internal holdout must contain all 20 classes

## 13. Split

Seed `20261001`.

For EACH class independently:
- 90% train
- 10% internal holdout

Split by normalized prompt hash.

Expected holdout support:
- L3: 80/class
- non-L3: 30/class

Every class MUST have support >0.

Require train↔holdout normalized overlap = 0.

---

# PART I — Reduced-adaptation optimization

## 14. Parameter groups

Same trainable LoRA + ProgramHead as Task 6T, but fixed differential LR:

ProgramHead:
`lr = 1e-4`

Existing LoRA parameters:
`lr = 2e-5`

Other settings:
- AdamW
- weight_decay `1e-4`
- effective batch `32`
- seed `20261001`
- max epochs `5`
- early stopping patience `2`
- bfloat16 AMP
- grad clip `1.0`
- no scheduler
- no sweep.

Do not unfreeze base backbone.

## 15. Checkpoint selection

Internal holdout only.

Compute across ALL 20 classes:
- `macro_f1_20`
- accuracy
- minimum class recall
- `l3_macro_recall`

Define:
`selection_primary = min(macro_f1_20, l3_macro_recall)`

Select by:
1. highest selection_primary
2. higher min class recall
3. higher macro F1
4. higher accuracy
5. earlier epoch.

External val/fixed24/stress may NOT influence checkpoint selection.

Checkpoint:
`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt`

Write:
`evaluation/task7c_training_summary.json`

If no finite checkpoint:
`PARSER_TRAINING_FAILED`

---

# PART J — Final evaluation once

After checkpoint selection, freeze SHA256 and run each external set once.

## 16. Full v0.2 val

Require:
- accuracy >= `0.995`
- macro F1 >= `0.995`
- every class recall >= `0.98`

Write:
`evaluation/task7c_full_val.json`

## 17. Z-MiniVal240

Require `240/240`.

Write:
`evaluation/task7c_z_minival240.json`

## 18. Z-Paired member queries

Require `40/40`.

Write:
`evaluation/task7c_z_paired_parser.json`

## 19. Exact Task 7A fixed24

Require:
- >= `22/24`
- all 8 compact prompts correct
- each L3 class >= `5/6`

Write:
`evaluation/task7c_task7a_fixed24.json`

## 20. Reused Task 7B minimal96

Require `96/96`.

Write:
`evaluation/task7c_minimal_pairs_result.json`

## 21. Reused Task 7B stress192

Require:
- accuracy >= `0.97`
- macro F1 >= `0.97`
- every represented class recall >= `0.90`
- L3-four-class macro recall >= `0.95`

Write:
`evaluation/task7c_stress_result.json`

---

# PART K — Scope safety

## 22. Reuse Task 7B controls

Require:
- L2 directional controls parse as L2 and exit 5 before proposal;
- nearest-only controls parse as nearest-only and exit 5 before proposal;
- OOD controls retain existing exit 4 behaviour;
- no keyword/regex gate.

Write:
`evaluation/task7c_scope_safety.json`

---

# PART L — Frozen end-to-end regression

## 23. Regression

Run exact Task 7A Z-MiniVal240 natural-language queries:

```text
Task 7C parser
→ frozen U-C1 reference resolver
→ frozen P_dir + P_near
→ frozen Z-B3
```

If parser remains 240/240, require exact Task 7A reproduction:

- strict mIoU `0.21700369907681483`
- answered mIoU `0.21975058134361`
- abstentions `3`
- Paired `8/20`
- margin `0.19949275176250805`

Tolerance:
- floating metrics abs delta <= `1e-6`
- counts exact.

Write:
`evaluation/task7c_end_to_end_regression.json`

---

# PART M — CLI default update

## 24. Parser default

Only if these canonical gates pass:
- full-val accuracy/F1/min recall;
- Z-MiniVal240;
- Z-Paired.

Then update L3 CLI default parser path to Task 7C.

If canonical gates fail:
- keep Task 6T as default.

No other CLI change.

---

# PART N — Verdict

## 25. PASS criteria

`L3_20CLASS_REHEARSAL_PASS` requires ALL:

1. Task 6T baseline hash exact
2. initialized from Task 6T, not Task 7B
3. same Qwen3-VL-2B text-only 20-class architecture
4. exactly 8,000 rows
5. all 20 classes represented
6. exact zh/en allocation
7. exact overlap zero
8. normalized overlap zero
9. internal holdout all 20 classes
10. full-val accuracy >=0.995
11. full-val macro F1 >=0.995
12. every full-val class recall >=0.98
13. Z-MiniVal 240/240
14. Z-Paired 40/40
15. fixed24 >=22/24
16. compact 8/8
17. every L3 fixed24 class >=5/6
18. minimal96 = 96/96
19. stress accuracy >=0.97
20. stress macro F1 >=0.97
21. stress every class recall >=0.90
22. stress L3 macro recall >=0.95
23. scope safety passes
24. no keyword/regex override
25. downstream reproduction exact
26. no test.

## 26. Exactly one verdict

Priority:

1. `INVALID_EXPERIMENT`
2. `BASELINE_PARSER_UNAVAILABLE`
3. `PARSER_TRAINING_FAILED`
4. `L3_20CLASS_CANONICAL_REGRESSION`
5. `L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`
6. `END_TO_END_REGRESSION`
7. `L3_20CLASS_REHEARSAL_PASS`

No other verdict.

---

# PART O — Interpretation boundary

DSH reports measurements only.

Do NOT:
- make parser a novelty claim;
- automatically run another parser repair;
- reopen reference hardening;
- modify Z-B3/fields/reference subsystem;
- add attention/global competition;
- start formal full training/test;
- choose Task 7D.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7C 的 20-class rehearsal 与 L3 组合语义结果决定 parser 是否正式冻结，不自行继续 parser 调参、reference hardening 或下游架构改造。`

---

# PART P — Required artifacts

Create:

```text
evaluation/task7c_reused_eval_manifest.json
evaluation/task7c_rehearsal_spec.json
evaluation/task7c_training_data_audit.json
evaluation/task7c_training_summary.json
evaluation/task7c_full_val.json
evaluation/task7c_z_minival240.json
evaluation/task7c_z_paired_parser.json
evaluation/task7c_task7a_fixed24.json
evaluation/task7c_minimal_pairs_result.json
evaluation/task7c_stress_result.json
evaluation/task7c_scope_safety.json
evaluation/task7c_end_to_end_regression.json
evaluation/task7c_verdict.json

docs/task7c_20class_rehearsal_l3_hardening.md

scripts/task7c_build_rehearsal.py
scripts/task7c_train_parser.py
scripts/task7c_eval_parser.py
scripts/task7c_scope_audit.py
scripts/task7c_report.py
```

Update only if section 24 permits:
- L3 parser default checkpoint path.

Always update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No downstream checkpoint changes.

---

# PART Q — Tests

Task 7B ended at:
`1148 passed, 1 skipped`

Add tests for at least:

1. Task 7B artifacts unchanged
2. Task 7B checkpoint not used as initialization
3. Task 6T baseline hash exact
4. same Qwen3-VL-2B
5. text-only
6. 20 classes exact
7. no original v0.2 template row used in optimization
8. exactly 8000 rehearsal prompts
9. all 20 classes represented
10. L3 800/class
11. non-L3 300/class
12. exact Chinese/English balance
13. no normalized training duplicate
14. exact reused eval hashes
15. train/eval exact overlap zero
16. train/eval normalized overlap zero
17. internal holdout all 20 classes
18. train/holdout normalized-disjoint
19. LoRA lr `2e-5`
20. ProgramHead lr `1e-4`
21. no base-backbone unfreeze
22. no hyperparameter sweep
23. external eval not used for selection
24. full val all 20 classes evaluated
25. Z-MiniVal exact reuse
26. Z-Paired exact reuse
27. fixed24 exact reuse
28. minimal96 exact reuse
29. stress192 exact reuse
30. scope controls exact reuse
31. no keyword/regex correction
32. U-C1 unchanged
33. field modules unchanged
34. Z-B3 unchanged
35. no YOLO/downstream training
36. no GRCL
37. no attention/GNN/Transformer downstream change
38. no test
39. no new dataset/download/install/GUI
40. end-to-end tolerance exact
41. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART R — Git/storage

Do not commit:
- Task 7C parser checkpoint;
- model/tokenizer caches;
- large generated rehearsal rows;
- YOLO/SAM2/Z-B3 weights;
- source imagery/vectors;
- `.conda`.

Commit:
- small specs/audits/evaluations;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: add 20-class parser rehearsal hardening`
2. `eval: audit L3 semantics without canonical forgetting`
3. optional docs/handoff commit

---

# PART S — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for genuine training/runtime bugs.

No installs or downloads.

---

# PART T — STOP

After Task 7C:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- run another parser experiment;
- reopen reference hardening;
- change downstream architecture;
- add attention/global competition;
- start formal full training/test;
- build GUI.

Wait for ChatGPT audit.
