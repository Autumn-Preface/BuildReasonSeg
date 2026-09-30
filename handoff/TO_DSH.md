# TO_DSH — Task 7B: L3 ProgramHead Compositional-Semantic Hardening

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `6ee3d0d90a56fb11c19112ab51e341b9f36c1392`
>
> Predecessor: Task 7A → `L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`
>
> Research decision already made by ChatGPT:
>
> 1. Keep the Task 7A L3 pipeline, U-C1 reference resolver, fields, SAM2 and Z-B3 frozen.
> 2. Do NOT reopen reference hardening in this task. Tasks 6P–6X already characterized that support-module bottleneck.
> 3. Task 7A exposed an independent language-interface defect:
>    - canonical Z-MiniVal240 queries: 240/240 correct;
>    - fixed L3 paraphrases: only 3/24 correct;
>    - all 8 required compact `direction + nearest` prompts failed;
>    - most errors drop terminal `to_nearest` and collapse L3 → L2.
> 4. Task 7B therefore hardens the **same Qwen3-VL-2B text-only 20-class ProgramHead** for compositional L2/L3 semantic contrast.
> 5. No keyword/rule remapping is allowed. The ProgramHead itself must learn the distinction.
> 6. Task 7B is parser-only. No segmentation/reference metric improvement is expected except preserving the frozen downstream chain.
>
> DSH is an executor. Do not redesign the parser architecture, downstream model, labels, or next task.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- create the exact train-only compositional augmentation specified below;
- train the same Qwen3-VL-2B + existing LoRA/ProgramHead implementation;
- evaluate on frozen parser packs and exact Task 7A paraphrases;
- update the L3 CLI default parser checkpoint after a successful hardening result;
- solve ordinary training/runtime bugs without changing the protocol.

DSH MUST NOT:
- change model family or size;
- move to 4B;
- add image tokens;
- add a second classifier;
- add keyword/regex semantic overrides;
- change the 20 canonical program ids;
- use val/test prompts for optimization;
- use Task 7A fixed24 strings in training;
- use Task 7B held-out stress/minimal-pair strings in training;
- modify YOLO/reference resolver/SAM2/fields/Z-B3;
- use test split;
- choose Task 7C.

If a prohibited change is required, STOP and report.

---

# PART A — Record Task 7A status

## 1. Task 7A findings to preserve

Copy into `docs/task7b_l3_programhead_hardening.md`:

Task 7A verdict:
`L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`

Task 7A A0:
- oracle Z-B3 reproduced exactly:
  - mIoU `0.3242128982543474`
  - Dice `0.4389840055529761`
  - Paired `15/20`
  - margin `0.30035408969722216`

Task 7A A1 predicted-reference:
- strict mIoU `0.21700369907681483`
- answered mIoU `0.21975058134361`
- retention `0.6693246944992726`
- Paired `8/20`
- margin `0.19949275176250805`
- reference abstention `3/240`
- dominant bottleneck `REFERENCE`

Task 7A language:
- canonical Z-MiniVal240 parser = `240/240`
- paired members = `40/40`
- fixed L3 paraphrase = `3/24`
- compact required prompts = `0/8`
- dominant paraphrase failure = terminal `to_nearest` dropped to L2.

Task 7B must not reinterpret Task 7A as a success.

---

# PART B — Frozen baseline parser and vocabulary

## 2. Initialization checkpoint

Initialize from the authoritative Task 6T hardened parser used by Task 7A.

Expected SHA256:
`4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e`

Require exact match.

If missing/mismatch:
STOP with `BASELINE_PARSER_UNAVAILABLE`.

## 3. Parser architecture

Preserve exactly:
- Qwen3-VL-2B backbone;
- text-only input;
- same tokenizer/prompt formatting;
- same LoRA/trainable policy as Task 6T;
- same ProgramHead;
- exactly 20 canonical classes.

No image input.
No new class such as `unsupported`.
No architecture change.

---

# PART C — Freeze all evaluation material BEFORE training

## 4. Reuse exact frozen evaluation sets

Read-only, byte-for-byte:

- BuildSpatialReason v0.2 full val records;
- Task 6Z Z-MiniVal240 queries;
- Task 6Z Z-PairedVal20 member queries;
- Task 7A fixed 24 L3 paraphrases;
- Task 7A four supported L3 ids;
- Task 7A CLI/domain behavior.

No test.

## 5. New Task 7B minimal-pair pack

Create before training:

`evaluation/task7b_compositional_minimal_pairs.json`

Exactly **96 prompts**.

Four contrast families:

### M1 — L2 direction vs L3 direction+nearest
For each of four directions, include 3 Chinese + 3 English matched semantic pairs:
- `largest_to_<direction>`
- `largest_to_<direction>_to_nearest`

Total = 48 prompts.

### M2 — L1 largest vs L2 direction
For four directions:
- `largest`
- `largest_to_<direction>`

Total = 16 prompts.

### M3 — nearest-only vs direction+nearest
For four directions:
- `largest_to_nearest`
- `largest_to_<direction>_to_nearest`

Total = 16 prompts.

### M4 — largest vs smallest family in L2
For four directions:
- `largest_to_<direction>`
- `smallest_to_<direction>`

Total = 16 prompts.

Requirements:
- bilingual;
- semantically unambiguous;
- lexical forms not present in Task 7A fixed24;
- lexical forms reserved from Task 7B training grammar.

Store:
- id
- contrast_group
- language
- prompt
- expected_program.

## 6. New held-out stress pack

Create before training:

`evaluation/task7b_l3_stress_v1.json`

Exactly **192 prompts**.

Coverage:
- 4 L3 classes × 24 prompts each = 96
- corresponding 4 L2 direction classes × 12 each = 48
- `largest_to_nearest` = 16
- `smallest_to_nearest` = 8
- `largest` = 8
- `smallest` = 8
- `leftmost/rightmost/topmost/bottommost` total = 8

Total must equal 192.

For each L3 class:
- 12 Chinese
- 12 English
- short, medium, long phrasing;
- nearest/closest variants in English;
- 最近/距离最近/最靠近 variants in Chinese;
- word-order variants.

No exact Task 7A fixed24 string.
Do not generate new evaluation prompts after observing results.

Write pre-training hash manifest:
`evaluation/task7b_eval_prompt_manifest.json`

---

# PART D — Training data

## 7. Allowed sources

Optimization may use only:
1. BuildSpatialReason v0.2 train split instruction text + canonical program id;
2. deterministic Task 7B train-only compositional paraphrase augmentation.

Do not use any val/test/eval prompt for optimization.

## 8. Remove train/eval text overlap first

Build evaluation exclusion set from:
- full v0.2 val;
- Z-MiniVal240;
- Z-PairedVal members;
- Task 7A fixed24;
- Task 7B minimal pairs;
- Task 7B stress v1.

Normalize:
- Unicode strip;
- lowercase English;
- collapse whitespace;
- normalize common punctuation;
- strip terminal punctuation for normalized comparison only.

Remove original train record if exact OR normalized text overlaps evaluation.

Record counts.

## 9. Controlled train-only augmentation

Create exactly **4,800 synthetic training prompts**.

### L3 target classes — 2,400 total
Each of four L3 classes: exactly 600
- 300 Chinese
- 300 English.

### Matched L2 direction controls — 1,200 total
Each `largest_to_<direction>`: exactly 300
- 150 Chinese
- 150 English.

### Nearest-only controls — 600 total
- `largest_to_nearest`: 400
- `smallest_to_nearest`: 200
balanced Chinese/English.

### Family controls — 600 total
Each `smallest_to_<direction>`: exactly 150
approximately balanced Chinese/English.

## 10. Train grammar constraints

L3 templates must contain:
- largest reference;
- one direction;
- terminal nearest.

L2 controls share similar stems but omit nearest.
Nearest-only controls express nearest without a direction.

Chinese TRAIN lexical pool:
- largest: `面积最大的`, `规模最大的`, `占地最大的`
- left/right/above/below: `左侧/位于左侧`, `右侧/位于右侧`, `上方/位于上方`, `下方/位于下方`
- nearest: `距离最近的`, `与其距离最近的`, `最靠近该方向区域的`

English TRAIN lexical pool:
- largest: `the largest`, `the greatest-area`, `the building with the greatest footprint`
- direction: `to the left of`, `to the right of`, `above`, `below`
- nearest: `the nearest`, `the closest`, `with the minimum distance`

Reserve for evaluation only:
- Chinese `最大建筑左边最近`, `最大建筑右边最近`, `最大建筑上方最近`, `最大建筑下方最近`
- English `closest building to the left/right`, `nearest building above`, `closest building below`

Do not reproduce exact Task 7A fixed24 strings.

## 11. Leakage audit

Require before training:
- exact overlap train ↔ every evaluation source = 0;
- normalized overlap = 0.

Write:
`evaluation/task7b_parser_leakage_audit.json`

If any overlap:
STOP `INVALID_EXPERIMENT`.

Tracked spec:
`evaluation/task7b_train_augmentation_spec.json`

Generated large training data:
`artifacts/task7b/parser_train_augmented/`
gitignored.

---

# PART E — Training protocol

## 12. Internal selection split

Use only cleaned train + augmentation data.

Create deterministic 90/10 internal split:
- seed `20261001`;
- group by normalized prompt hash;
- stratify by canonical class.

No normalized prompt in both sides.

Only this holdout may select checkpoint.

## 13. Optimization

Start from Task 6T hardened checkpoint.

Frozen settings:
- AdamW
- learning rate `2e-4`
- weight_decay `1e-4`
- effective batch size `32`
- seed `20261001`
- max epochs `8`
- early stopping patience `2`
- AMP/bfloat16 as supported
- no hyperparameter sweep.

Selection:
1. internal holdout macro F1
2. exact accuracy
3. L3-four-class macro recall.

Record trainable/total params, LoRA config, actual microbatch/accumulation, epochs, wall time, peak VRAM.

Checkpoint:
`artifacts/checkpoints/task7b/program_parser_l3_hardened_v1.pt`
gitignored.

Write:
`evaluation/task7b_training_summary.json`

If no valid checkpoint:
STOP `PARSER_TRAINING_FAILED`.

---

# PART F — Final frozen parser evaluation

After selection, freeze checkpoint SHA and evaluate once.

## 14. Full BuildSpatialReason v0.2 val

Report accuracy, macro F1, per-class recall, confusion matrix.

Gate:
- accuracy >= `0.995`
- macro F1 >= `0.995`
- every class recall >= `0.98`.

Write:
`evaluation/task7b_full_val.json`

## 15. Z-MiniVal240

Require:
- `240/240`.

Write:
`evaluation/task7b_z_minival240.json`

## 16. Z-PairedVal member queries

Require:
- `40/40`.

Write:
`evaluation/task7b_z_paired_parser.json`

## 17. Exact Task 7A fixed24

Gate:
- >= `22/24`;
- all exact 8 compact prompts correct;
- every L3 class >= `5/6`.

Write:
`evaluation/task7b_task7a_fixed24.json`

## 18. Task 7B minimal pairs

Gate:
- `96/96`.

Write:
`evaluation/task7b_minimal_pairs_result.json`

## 19. Task 7B stress v1

Gate:
- overall accuracy >= `0.97`
- macro F1 >= `0.97`
- every represented class recall >= `0.90`
- four L3-class macro recall >= `0.95`.

Write:
`evaluation/task7b_stress_result.json`

---

# PART G — Scope safety / CLI regression

## 20. Update parser checkpoint only

Update `predict_buildreasonseg_l3.py` default parser checkpoint to Task 7B checkpoint only if canonical gates pass.

Do not change domain guard, supported L3 list, U-C1 resolver, fields, Z-B3.

## 21. Out-of-scope semantic controls

Evaluate:

- `分割面积最大的建筑物右侧的建筑物。`
  → `largest_to_right_of`
  → exit 5 before proposal.

- `segment the building to the left of the largest building`
  → `largest_to_left_of`
  → exit 5 before proposal.

- `分割距离面积最大的建筑物最近的建筑物。`
  → `largest_to_nearest`
  → exit 5 before proposal.

- `segment the nearest building to the largest building`
  → `largest_to_nearest`
  → exit 5 before proposal.

OOD controls preserve existing exit 4 behavior.

No keyword nearest gate.

Write:
`evaluation/task7b_scope_safety.json`

---

# PART H — End-to-end frozen regression

## 22. Canonical-query end-to-end

Run exact Z-MiniVal240 natural-language queries:

```text
Task 7B parser
→ frozen Task 7A U-C1 reference resolver
→ frozen P_dir + P_near
→ frozen Z-B3
```

If parser = 240/240, require reproduction of Task 7A:

- strict mIoU `0.21700369907681483`
- answered mIoU `0.21975058134361`
- abstentions `3`
- Paired `8/20`
- margin `0.19949275176250805`

Tolerance:
- metric absolute delta <= `1e-6`
- counts exact.

Write:
`evaluation/task7b_end_to_end_regression.json`

---

# PART I — Verdict

## 23. PASS criteria

`L3_PROGRAMHEAD_HARDENING_PASS` requires ALL:

1. baseline parser hash exact;
2. same Qwen3-VL-2B text-only 20-class architecture;
3. exact/normalized leakage zero;
4. full val accuracy >=0.995;
5. full val macro F1 >=0.995;
6. every full-val class recall >=0.98;
7. Z-MiniVal240 = 240/240;
8. Z-Paired members = 40/40;
9. fixed24 >=22/24;
10. all 8 compact prompts correct;
11. every fixed24 L3 class >=5/6;
12. minimal pairs = 96/96;
13. stress accuracy >=0.97;
14. stress macro F1 >=0.97;
15. stress every class recall >=0.90;
16. stress L3 macro recall >=0.95;
17. scope controls semantically correct before exit5;
18. no keyword/regex override;
19. frozen downstream reproduces Task7A;
20. no test split.

## 24. Exactly one verdict

Priority:
1. `INVALID_EXPERIMENT`
2. `BASELINE_PARSER_UNAVAILABLE`
3. `PARSER_TRAINING_FAILED`
4. `L3_PARSER_CANONICAL_REGRESSION`
5. `L3_PARSER_COMPOSITIONAL_ROBUSTNESS_FAIL`
6. `END_TO_END_REGRESSION`
7. `L3_PROGRAMHEAD_HARDENING_PASS`

No other verdict.

---

# PART J — Interpretation boundary

DSH reports measurements only.

Do NOT:
- claim parser as project novelty;
- change downstream architecture;
- reopen reference hardening;
- add attention/global competition;
- start full training/test;
- decide Task 7C.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7B 的 L3 ProgramHead 组合语义泛化结果决定下一步，不自行重新开启 reference hardening、修改 Z-B3 或进行 attention/global competition 改造。`

---

# PART K — Required artifacts

Create:

```text
evaluation/task7b_compositional_minimal_pairs.json
evaluation/task7b_l3_stress_v1.json
evaluation/task7b_eval_prompt_manifest.json
evaluation/task7b_train_augmentation_spec.json
evaluation/task7b_parser_leakage_audit.json
evaluation/task7b_training_summary.json
evaluation/task7b_full_val.json
evaluation/task7b_z_minival240.json
evaluation/task7b_z_paired_parser.json
evaluation/task7b_task7a_fixed24.json
evaluation/task7b_minimal_pairs_result.json
evaluation/task7b_stress_result.json
evaluation/task7b_scope_safety.json
evaluation/task7b_end_to_end_regression.json
evaluation/task7b_verdict.json

docs/task7b_l3_programhead_hardening.md

scripts/task7b_build_parser_data.py
scripts/task7b_train_parser.py
scripts/task7b_eval_parser.py
scripts/task7b_scope_audit.py
scripts/task7b_report.py
```

Update only as needed:
- parser default path/helper;
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No downstream checkpoint changes.

---

# PART L — Tests

Task 7A ended at:
`1107 passed, 1 skipped`

Add tests for at least:

1. Task 7A artifacts unchanged
2. baseline parser hash exact
3. same Qwen3-VL-2B
4. text-only parser
5. exactly 20 classes
6. no keyword/regex remap
7. minimal-pair pack frozen before training
8. stress pack frozen before training
9. exactly 96 minimal-pair prompts
10. exactly 192 stress prompts
11. bilingual L3 stress coverage
12. exact Task7A fixed24 reused
13. all 8 compact prompts present
14. original train/eval overlap removed
15. augmentation/eval exact overlap zero
16. augmentation/eval normalized overlap zero
17. exactly 4800 augmentations
18. exact class allocation
19. train/internal-holdout hash-disjoint
20. val/test not used for checkpoint selection
21. new checkpoint SHA recorded
22. full-val all 20 classes evaluated
23. Z-MiniVal exact reuse
24. Z-Paired exact reuse
25. scope L2 controls exit5
26. scope nearest-only controls exit5
27. no proposal call on exit5
28. no reference/SAM2/Z-B3 call on exit5
29. U-C1 unchanged
30. fields unchanged
31. Z-B3 unchanged
32. no YOLO training
33. no downstream training
34. no GRCL
35. no attention/Transformer/GNN downstream change
36. no test split
37. no new dataset/download/install/GUI
38. end-to-end regression tolerance exact
39. previous passing suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART M — Git/storage

Do not commit:
- Task 7B parser checkpoint;
- tokenizer/model caches;
- YOLO/SAM2/Z-B3 weights;
- large generated training JSONL;
- source imagery/vectors;
- `.conda`.

Commit:
- small eval packs/spec/manifests;
- parser scripts/code changes;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: harden L3 ProgramHead compositional semantics`
2. `eval: audit L2-L3 language contrast robustness`
3. optional docs/handoff commit

---

# PART N — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for a genuine parser-training/runtime bug.

No installs or downloads.

---

# PART O — STOP

After Task 7B:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- retrain downstream models;
- reopen reference hardening;
- add attention/global competition;
- start full training;
- use test;
- build GUI.

Wait for ChatGPT audit.
