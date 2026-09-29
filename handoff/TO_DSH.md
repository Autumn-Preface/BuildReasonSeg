# TO_DSH — Task 6T: ProgramHead Semantic Hardening + Scope-Safety Regression

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `dc8544f161535d7321ee4c874184348c10192105`
>
> Predecessor: Task 6S
>
> Task 6S formal verdict: `DIRECTIONAL_PARSER_HARDENING_REQUIRED`
>
> Task 6S research bottleneck: `REFERENCE`
>
> Important distinction:
> - Task 6T is a **small parser/scope-safety hardening task** required to close the 6S interface gate.
> - Task 6T does **not** attempt to solve the main scientific bottleneck (`REFERENCE`).
> - After Task 6T, STOP. ChatGPT will audit it and then design the reference-hardening task separately.

All user-facing DSH output must be Chinese.

---

# 0. Research decision already made by ChatGPT

Keep the Task 6S primary directional architecture frozen:

```text
natural-language instruction
→ Qwen3-VL-2B ProgramHead
→ canonical program id
→ deterministic Reference family + Direction relation
→ frozen Task 6Q proposal reference resolver
→ predicted reference mask
→ GeometricRelationField v0.2
→ frozen SAM2 visual feature + relation field + relation embedding
→ frozen Task 6O B3 target decoder
→ target mask
```

Task 6T may change **only the ProgramHead checkpoint/training data and parser-evaluation code**.

DO NOT change:
- proposal model/checkpoint/config;
- Task 6Q reference resolver;
- B3 target decoder;
- SAM2;
- GeometricRelationField v0.2;
- GRCL;
- directional target architecture;
- MiniVal240/PairedVal20;
- native-vector dataset/splits;
- any test split.

---

# 1. Mandatory Task 6S audit erratum

Before training, record the following factual correction in:

`docs/task6t_programhead_semantic_hardening.md`

and in the new `handoff/FROM_DSH.md`.

Task 6S `FROM_DSH.md` incorrectly stated that the frozen 20-program vocabulary has no `"nearest"` program.

This is false.

The canonical `EXPECTED_QUERY_TYPES` in:

`buildreasonseg_mvp/structured_grounding.py`

explicitly contains:

```text
largest_to_nearest
smallest_to_nearest
largest_to_above_to_nearest
largest_to_below_to_nearest
largest_to_left_of_to_nearest
largest_to_right_of_to_nearest
```

Therefore the two Task 6S nearest controls failed because the frozen ProgramHead **misclassified nearest-containing instructions as direction-only programs**, not because the vocabulary lacked nearest semantics.

Do NOT mutate Task 6S frozen JSON artifacts or historical verdict. Record this as a documentation erratum only.

---

# 2. Frozen baseline parser

The authoritative Task 6S parser checkpoint is:

`artifacts/checkpoints/task6m/program_parser_v02_best.pt`

Expected SHA256:

`eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3`

The old `artifacts/checkpoints/task6j/j2_best.pt` is NOT the authoritative Task 6S parser because its hash does not match.

Before training:
1. verify the Task 6M checkpoint exists;
2. recompute SHA256;
3. require exact match;
4. verify vocabulary is exactly the frozen 20 ids;
5. verify parser remains text-only.

If unavailable or mismatched:
STOP with `BASELINE_PARSER_UNAVAILABLE`.

---

# 3. What Task 6T must solve

Task 6S measured:

- MiniVal240 exact program accuracy: `240/240 = 1.0000`
- PairedVal20 parser members: `40/40`
- fixed 24 paraphrases: `21/24`
- failing short Chinese paraphrases:
  - `找出最大建筑左边的建筑物。`
    - wrongly → `smallest_to_left_of`
  - `找出最大建筑右边的建筑物。`
    - wrongly → `smallest_to_right_of`
  - `找出最大建筑下面的建筑物。`
    - wrongly → `smallest_to_below`
- out-of-scope nearest controls:
  - `分割面积最大的建筑物右侧最近的建筑物。`
  - `segment the building nearest to the right of the largest building`
  - both wrongly → `largest_to_right_of`
  - then entered downstream instead of being rejected by Task 6S directional scope.

Task 6T must improve **actual ProgramHead semantic classification**.

Forbidden shortcut:
- no keyword remapping such as `if "最近" in prompt: ...`;
- no regex/class-id override after ProgramHead;
- no hard-coded special handling for the five failed strings;
- no deterministic largest/smallest post-correction;
- no bypassing ProgramHead for supported domain prompts.

The existing pre-parser OOD domain guard may remain unchanged.

The post-parser Task 6S scope check may remain unchanged.

---

# 4. Model family and trainable scope

Use the **same Qwen3-VL-2B text-only ProgramHead implementation and the same 20-class vocabulary**.

Initialize from the authoritative Task 6S checkpoint.

Do NOT:
- move to 4B;
- change to another LLM/MLLM;
- add image tokens;
- add a separate external classifier;
- change the 20 canonical ids;
- collapse classes;
- add an `"unsupported"` class.

Reuse the existing Task 6M parser training implementation where technically possible.

Prefer the same trainable-parameter policy used by Task 6M. If the prior implementation freezes the backbone and trains only the ProgramHead / existing lightweight trainable parts, preserve that policy.

Do not invent a larger fine-tuning scope unless required by the existing implementation contract.

Record exact trainable parameter count.

---

# 5. No-leakage parser-hardening dataset

## 5.1 Sources allowed for training

Training may use only:
- BuildSpatialReason v0.2 **train split** instruction text + canonical program id;
- deterministic controlled paraphrase augmentation derived from the train split / program semantics.

Do not use:
- BuildSpatialReason val/test labels for optimization;
- MiniVal240 queries for training;
- PairedVal20 queries for training;
- the exact 24 fixed Task 6S paraphrase strings for training;
- Task 6T stress-evaluation strings for training.

## 5.2 Training augmentation goal

Create a tracked grammar specification, for example:

`evaluation/task6t_parser_train_augmentation_spec.json`

and a generated local dataset under:

`artifacts/task6t/parser_train_augmented/`

The training augmentation must explicitly cover semantic contrasts:

### reference-family contrasts
- largest vs smallest

### level contrasts
- largest
- largest_to_direction
- largest_to_nearest
- largest_to_direction_to_nearest

### direction contrasts
- left_of / right_of / above / below

### lexical variation
Chinese examples may use variants such as:
- 面积最大 / 最大 / 最大的
- 面积最小 / 最小 / 最小的
- 左侧 / 左边 / 位于左边
- 右侧 / 右边 / 位于右边
- 上方 / 上面
- 下方 / 下面
- 最近 / 距离最近 / 最靠近

English examples may vary:
- largest / greatest-area
- smallest / least-area
- to the left of / left of
- to the right of / right of
- above / over
- below / under
- nearest / closest

Training variants must not reproduce the exact Task 6S 24-prompt evaluation strings.

## 5.3 Exact-string and normalized-string leakage audit

Before training, build frozen evaluation prompt hashes.

For every training prompt:
- compare exact UTF-8 string;
- compare normalized string (strip whitespace, lowercase English, normalize common punctuation).

Require zero overlap against:
- MiniVal240 query strings;
- PairedVal20 query strings;
- the 24 fixed Task 6S paraphrases;
- all Task 6T stress prompts.

Write:
`evaluation/task6t_parser_leakage_audit.json`

If any overlap:
STOP with `INVALID_EXPERIMENT`.

---

# 6. Freeze Task 6T evaluation packs BEFORE training

## 6.1 Reuse frozen packs byte-for-byte

Reuse:
- Task 6N MiniVal240
- Task 6N PairedVal20
- Task 6S exact 24 paraphrase prompt list
- Task 6S exact 8 unsupported controls

Do not regenerate or edit them.

## 6.2 Create a new semantic minimal-pair pack

Create:

`evaluation/task6t_parser_minimal_pairs.json`

Freeze it before training.

At minimum 48 prompts.

It must contain paired contrasts for both Chinese and English:

### A. largest vs smallest
For each direction:
- largest_to_direction
- smallest_to_direction

### B. direction-only vs direction+nearest
For all four directions:
- `largest_to_<direction>`
- `largest_to_<direction>_to_nearest`

### C. L1 vs L2
- largest vs largest_to_left/right/above/below
- smallest vs smallest_to_left/right/above/below

### D. simple-nearest vs direction-nearest
- largest_to_nearest vs largest_to_direction_to_nearest
- smallest_to_nearest must also appear in held-out stress evaluation.

Prompts must be semantically unambiguous.

Reserve lexical forms not used in the training-augmentation grammar.

Every prompt stores:
- language
- expected_program
- contrast_group
- prompt

## 6.3 Create a broader held-out stress pack

Create:

`evaluation/task6t_parser_stress_v1.json`

Freeze before training.

Requirements:
- all 20 canonical classes represented;
- at least 6 prompts per class;
- at least 3 Chinese + 3 English per class;
- total >= 120 prompts;
- lexical forms disjoint from the training augmentation templates where practical;
- no exact/normalized overlap with training prompts.

Must include:
- short Chinese prompts;
- longer Chinese prompts;
- short English prompts;
- longer English prompts;
- largest/smallest contrasts;
- nearest vs direction-only contrasts;
- L1/L2/L3 contrasts.

Do not generate additional stress prompts after observing model failures.

---

# 7. Training protocol

## 7.1 Baseline-first

Evaluate the old frozen parser on:
- MiniVal240;
- PairedVal20 members;
- Task 6S fixed 24;
- Task 6T minimal pairs;
- Task 6T stress v1.

Write:
`evaluation/task6t_parser_baseline.json`

## 7.2 Training split

Use only the parser-hardening train dataset.

Make an internal train-only holdout:
- deterministic fixed seed `20260930`;
- e.g. 90/10 split by normalized prompt hash;
- stratified by canonical program id.

This internal holdout is the only set allowed for checkpoint selection / early stopping.

Do not select checkpoint using MiniVal240, fixed24, stress, or test.

## 7.3 Optimization

Reuse the Task 6M parser training recipe as the default.

Record:
- initialization checkpoint hash;
- trainable parameters;
- total parameters;
- optimizer;
- LR;
- weight decay;
- batch size;
- grad accumulation;
- AMP;
- seed;
- max epochs/steps;
- early stopping;
- selection metric;
- wall time;
- peak VRAM.

Recommended selection metric:
1. internal-holdout macro F1;
2. tie-break exact accuracy.

No hyperparameter sweep larger than 3 declared configurations.

If a sweep is needed:
- declare all candidates before training;
- use only train-internal holdout for selection.

No val/test tuning.

## 7.4 Output checkpoint

Save local-only:

`artifacts/checkpoints/task6t/program_parser_hardened_v1.pt`

Record SHA256.

Do not commit weights.

---

# 8. Final frozen evaluation

After selecting the new parser, freeze the checkpoint hash and then run all final evaluations exactly once.

## 8.1 Full BuildSpatialReason v0.2 val parser audit

Evaluate all v0.2 val records.

Report:
- exact accuracy;
- macro F1;
- per-class recall;
- confusion matrix;
- language if bilingual records permit.

Write:
`evaluation/task6t_parser_full_val.json`

No test split.

## 8.2 MiniVal240 regression

Require:
- exact accuracy `240/240`.

Write:
`evaluation/task6t_parser_minival240.json`

## 8.3 PairedVal20 parser regression

Require:
- 40/40 members correct.

Write:
`evaluation/task6t_parser_pairedval20.json`

## 8.4 Fixed 24 paraphrase regression

Run the exact Task 6S fixed 24 prompts.

Target:
- `24/24`.

At minimum for PASS:
- `>= 23/24`

BUT additionally the three previously failing prompts MUST all be correct.

Write:
`evaluation/task6t_parser_fixed24.json`

## 8.5 Minimal pairs

Require:
- overall accuracy = `1.000`
- every contrast group = `1.000`

Write:
`evaluation/task6t_parser_minimal_pairs_result.json`

## 8.6 Stress v1

Require:
- exact accuracy >= `0.95`
- macro F1 >= `0.95`
- every class recall >= `0.90`

Write:
`evaluation/task6t_parser_stress_result.json`

---

# 9. Directional CLI scope-safety rerun

Update the Task 6S directional CLI default parser reference to the new Task 6T checkpoint.

Do not change:
- domain guard;
- supported 8 directional program list;
- proposal/SAM2/field/B3 logic.

Run exact Task 6S controls.

## 9.1 OOD controls

Must remain:
- 4/4 exit code `4`
- before ProgramHead/downstream where Task 6S specifies.

## 9.2 Valid but outside Task 6S directional scope

Exact controls:
- `分割面积最大的建筑物。`
- `分割最左侧的建筑物。`
- `分割面积最大的建筑物右侧最近的建筑物。`
- `segment the building nearest to the right of the largest building`

Expected parser programs:
- largest
- leftmost
- largest_to_right_of_to_nearest
- largest_to_right_of_to_nearest

All four must:
- classify semantically correctly;
- exit code `5`;
- stop before proposal/SAM2/B3.

This is a parser + existing scope-check result.

Do NOT add a nearest keyword gate.

Write:
`evaluation/task6t_cli_scope_safety.json`

---

# 10. Directional end-to-end regression

Because only the parser changes, rerun MiniVal240 through the full Task 6S directional chain.

If parser is still 240/240 exact, the downstream result must reproduce Task 6S within numerical tolerance.

Expected Task 6S:
- answered records = 234
- answered-only mIoU = `0.3045812554881724`
- strict all-240 mIoU = `0.2969667241009681`
- PairedVal = `10/20`
- own-cross margin = `0.27370032940000916`
- abstentions = 6

Tolerance:
- mIoU absolute delta <= `1e-6`
- paired count exact
- abstention count exact

Write:
`evaluation/task6t_end_to_end_regression.json`

If downstream differs despite identical parsed programs and frozen modules:
verdict `END_TO_END_REGRESSION`.

---

# 11. Task 6S documentation/state cleanup

## 11.1 Vocabulary erratum

As section 1: record that nearest classes do exist.

Do not rewrite Task 6S JSON results.

## 11.2 PROJECT_STATE canonical dataset wording

`handoff/PROJECT_STATE.md` currently still contains a legacy machine-checked v0.1.1 ARTIFACT-FACTS block and an identity row that can be read as if v0.1.1 were the active canonical reasoning dataset.

Do NOT hand-edit or falsify the machine-checked legacy ARTIFACT-FACTS numbers.

Instead:
- clearly label that block as the legacy v0.1.1 artifact-consistency block if the existing checker requires it;
- update the human-readable Identity/current-state section to state that the **active canonical dataset for current Task 6L+ work is BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**;
- keep v0.1.1 explicitly historical/frozen.

Do not change dataset files or splits.

---

# 12. Predeclared PASS gates

`PARSER_HARDENING_PASS` requires ALL:

1. authoritative old parser hash verified before training;
2. zero train/evaluation prompt leakage;
3. same Qwen3-VL-2B text-only 20-class architecture;
4. full v0.2 val exact accuracy >= `0.995`;
5. MiniVal240 = `240/240`;
6. PairedVal members = `40/40`;
7. fixed24 >= `23/24`;
8. all three previously failed Chinese short prompts correct;
9. minimal-pair pack = `100%`;
10. stress v1 exact accuracy >= `0.95`;
11. stress v1 macro F1 >= `0.95`;
12. every stress class recall >= `0.90`;
13. OOD controls 4/4 preserve exit 4;
14. all 4 valid-but-out-of-scope controls classify semantically correctly and exit 5 before downstream;
15. end-to-end MiniVal240 mIoU reproduces Task 6S within 1e-6;
16. PairedVal remains exactly 10/20;
17. no test split;
18. no proposal/B3/field/SAM2 changes;
19. no keyword/regex semantic overrides.

---

# 13. Verdict — exactly one

Priority order:

1. `INVALID_EXPERIMENT`
   - leakage, test use, frozen non-parser mutation, keyword semantic override, protocol violation.

2. `BASELINE_PARSER_UNAVAILABLE`

3. `PARSER_TRAINING_FAILED`
   - cannot obtain a valid finite trained checkpoint / internal training failure.

4. `PARSER_CANONICAL_REGRESSION`
   - MiniVal240 < 240/240 OR full v0.2 val < 0.995 OR PairedVal members < 40/40.

5. `PARSER_SEMANTIC_CONTRAST_FAIL`
   - canonical gates pass but fixed24/minimal-pair/stress/scope-safety gates fail.

6. `END_TO_END_REGRESSION`
   - parsed programs are correct but frozen downstream no longer reproduces Task 6S.

7. `PARSER_HARDENING_PASS`
   - all section 12 gates pass.

No other verdict.

---

# 14. Required artifacts

Create at minimum:

```text
evaluation/task6t_parser_train_augmentation_spec.json
evaluation/task6t_parser_leakage_audit.json
evaluation/task6t_parser_minimal_pairs.json
evaluation/task6t_parser_stress_v1.json
evaluation/task6t_parser_baseline.json
evaluation/task6t_training_summary.json
evaluation/task6t_parser_full_val.json
evaluation/task6t_parser_minival240.json
evaluation/task6t_parser_pairedval20.json
evaluation/task6t_parser_fixed24.json
evaluation/task6t_parser_minimal_pairs_result.json
evaluation/task6t_parser_stress_result.json
evaluation/task6t_cli_scope_safety.json
evaluation/task6t_end_to_end_regression.json
evaluation/task6t_verdict.json

docs/task6t_programhead_semantic_hardening.md

scripts/task6t_build_parser_data.py
scripts/task6t_train_parser.py
scripts/task6t_eval_parser.py
scripts/task6t_scope_audit.py
scripts/task6t_report.py
```

Update:
- `predict_buildreasonseg_directional.py` only as needed to use the new parser checkpoint by default;
- parser-loading config/helper only as needed;
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

Do not alter proposal/field/B3 algorithms.

---

# 15. Tests

Task 6S ended at:

`773 passed, 1 skipped`

Add tests for at least:

1. Task 6S artifacts unchanged
2. baseline parser SHA exact
3. same 20-class vocabulary
4. nearest classes actually present in vocabulary
5. same Qwen3-VL-2B text-only architecture
6. no image input to parser
7. no keyword/regex semantic remap
8. exact training/eval prompt leakage = 0
9. normalized training/eval prompt leakage = 0
10. stress pack frozen before training
11. minimal-pair pack frozen before training
12. all 20 classes represented in stress
13. >= 120 stress prompts
14. bilingual stress coverage
15. full-val no test access
16. MiniVal240 byte-identical reuse
17. PairedVal20 byte-identical reuse
18. fixed24 exact reuse
19. eight controls exact reuse
20. new checkpoint exists locally
21. new checkpoint SHA recorded
22. three previously failed short Chinese prompts correct
23. both nearest controls classify `largest_to_right_of_to_nearest`
24. both nearest controls exit 5 before proposal
25. no proposal config change
26. no B3 change
27. field v0.2 hash unchanged
28. no GRCL
29. no 4B
30. no new dataset
31. no test split
32. end-to-end regression exact paired 10/20
33. end-to-end mIoU tolerance
34. previous passing suite preserved.

Run:

`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# 16. Git/storage

Do NOT commit:
- new parser checkpoint;
- model weights;
- tokenizer/model cache;
- generated large augmented training JSONL if large;
- source imagery;
- SAM2/YOLO/B3 assets;
- `.conda`.

Commit:
- augmentation specification;
- small frozen eval packs;
- small evaluation JSON;
- scripts/code;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: harden ProgramHead semantic classification`
2. `eval: audit parser contrast and scope safety`
3. optional `docs: record Task 6T parser hardening`

---

# 17. Model / execution policy

Use DSH:
- **DeepSeek V4.1 Flash + High**

Use Max only for a genuine parser-training/runtime/cross-file bug.

Do not use V4 Pro by default.

No downloads or installs unless the existing local environment is unexpectedly missing a dependency required by the already-established parser stack. If that occurs, STOP and report before installing.

---

# 18. STOP

After Task 6T:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- repair the REFERENCE bottleneck;
- tune YOLO;
- change proposal conf/imgsz/max_det;
- add tiling/TTA;
- train a reference ranker;
- retrain B3;
- change relation field;
- add GRCL;
- add nearest execution to the target model;
- add L3 to the target model;
- access test;
- build GUI.

Wait for ChatGPT to audit the parser hardening and then decide the separate reference-hardening task.
