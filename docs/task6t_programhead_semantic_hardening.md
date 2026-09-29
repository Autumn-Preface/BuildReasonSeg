# Task 6T — ProgramHead Semantic Hardening + Scope-Safety Regression

> Task: `handoff/TO_DSH.md` (Task 6T) · Base commit: `dc8544f` · Predecessor: Task 6S →
> `DIRECTIONAL_PARSER_HARDENING_REQUIRED` (bottleneck `REFERENCE`)
> **Verdict: `PARSER_SEMANTIC_CONTRAST_FAIL`**
> Checkpoint: `artifacts/checkpoints/task6t/program_parser_hardened_v1.pt` (local only) · Tests:
> `tests/test_task6t_programhead_hardening.py`

Only the **ProgramHead checkpoint/training data and the parser-evaluation code** changed. The proposal
model, the Task 6Q reference resolver, the geometric relation field v0.2, SAM2, B3, the directional
target architecture, MiniVal240/PairedVal20, the native-vector dataset/splits and the test split were
untouched (verified by `git diff` against the base commit for every frozen path).

## 1. Mandatory Task 6S erratum (recorded, Task 6S artifacts not mutated)

Task 6S `FROM_DSH.md` stated that *"the frozen 20-program vocabulary has no nearest program"*. **This is
false.** `buildreasonseg_mvp/structured_grounding.py::EXPECTED_QUERY_TYPES` contains six nearest classes:

```text
largest_to_nearest
smallest_to_nearest
largest_to_above_to_nearest
largest_to_below_to_nearest
largest_to_left_of_to_nearest
largest_to_right_of_to_nearest
```

The two Task 6S nearest controls therefore failed because the frozen ProgramHead **misclassified
nearest-containing instructions as direction-only programs** (`largest_to_right_of`), not because the
vocabulary lacked nearest semantics. The Task 6S JSON artifacts and its historical verdict were **not**
modified; this is a documentation erratum only.

## 2. Baseline parser (authoritative)

`artifacts/checkpoints/task6m/program_parser_v02_best.pt`, SHA256
`eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3` — verified exactly before training.
`artifacts/checkpoints/task6j/j2_best.pt` is **not** the Task 6S parser (its hash does not match).
Text-only, 20-class vocabulary preserved.

## 3. No-leakage hardening data

`evaluation/task6t_parser_train_augmentation_spec.json` +
`artifacts/task6t/parser_train_augmented/` (gitignored).

| Item | Value |
|---|---|
| BuildSpatialReason v0.2 **train** records | 12,778 (both languages → 25,556 examples) |
| Examples kept after the deterministic collision filter | 24,161 (1,395 dropped) |
| Deterministic paraphrase augmentation kept | **1,800** (90 per program per language cap) |
| Combined training prompts | **25,961** |
| Grammar coverage | largest vs smallest · L1 / direction / nearest / direction+nearest · all four directions · zh 面积最大/最大/最大(的)、左侧/左边、上方/上面、下方/下面、最近/距离最近/最靠近 · en largest/greatest-area、smallest/least-area、to the left of/left of、to the right of/right of、above/over、below/under、nearest/closest |

`evaluation/task6t_parser_leakage_audit.json`: **exact overlap 0, normalized overlap 0** over all 25,961
training prompts against MiniVal240 queries, PairedVal20 queries, the exact Task 6S fixed-24 list and the
Task 6T minimal-pair and stress packs (`NO_LEAKAGE`).

*Recorded dataset fact:* the v0.2 **train** split itself contains 1,395 instruction strings that are
exact/normalized duplicates of v0.2 **val** queries (the dataset generator reuses templates across
splits). Those examples are removed from the hardening training set by a deterministic pre-training
filter so that the frozen leakage audit is exactly zero. This is a property of the frozen dataset, not a
Task 6T construction; no dataset file or split was changed.

## 4. Frozen evaluation packs (created before training)

`evaluation/task6t_parser_minimal_pairs.json` — **76 prompts** ≥ 48, contrast groups
A largest-vs-smallest (16), B direction-vs-direction+nearest (16), C L1-vs-L2 (32),
D simple-nearest-vs-direction-nearest (12); both languages; each prompt stores language,
expected_program, contrast_group.

`evaluation/task6t_parser_stress_v1.json` — **160 prompts**, all 20 canonical classes,
4 Chinese + 4 English per class, held-out template and lexical forms, including a minority of held-out
synonyms (占地面积最大/占地面积最小/楼房区域; most extensive/least extensive/structure). Never extended
after observing failures.

Both packs are validated by construction: no prompt maps to two different programs, and every direction
program's prompt provably contains its direction word. **Two pack-construction defects were found and
fixed by that validation before any training** (a Chinese direction template that omitted the direction
word, and an English template producing "on the on the left side of side of"): the packs were regenerated,
the leakage audit re-run (0/0) and the whole baseline → training → evaluation pipeline re-executed, so the
packs remain frozen *before* the final training run.

## 5. Training

`evaluation/task6t_training_summary.json` — same Qwen3-VL-2B text-only 20-class ProgramHead and the same
Task 6M trainable-parameter policy (LoRA adapters + ProgramHead, backbone frozen), initialised from the
authoritative Task 6S checkpoint.

| Item | Value |
|---|---|
| Declared configurations | C1 = Task 6M recipe (lora 1e-4 / head 3e-4), C2 = half LR, C3 = quarter LR |
| Configurations run | **C1 only** — the sweep terminated at the selection ceiling |
| Epoch 1 (C1) | train loss **0.0067**, internal-holdout **macro F1 1.0000**, accuracy 1.0000 |
| Optimizer / schedule | AdamW (β 0.9/0.999), wd 0.01, grad clip 1.0, bf16 autocast, seed 20260930, constant LRs, no scheduler |
| Batch size / epochs | 16 / 1 (early stop patience 1) |
| Internal holdout | 90/10 by `sha256(seed:normalized_prompt)`, stratified by program → 23,195 train / 2,578 holdout |
| Selection metric | internal-holdout macro F1, tie-break exact accuracy (never MiniVal240 / fixed24 / stress / test) |
| Trainable / total parameters | **17,479,700** / 2,144,466,964 |
| Wall time / peak VRAM | 321.1 s / 0.67 GB |
| Checkpoint | `artifacts/checkpoints/task6t/program_parser_hardened_v1.pt`, SHA256 `4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e` (local only, not committed) |

## 6. Baseline vs hardened

| Metric | Baseline (Task 6S parser) | **Hardened (Task 6T)** | Gate |
|---|---|---|---|
| Full v0.2 val, exact accuracy (en / zh / combined) | 1.0000 / 1.0000 / — | **1.0000 / 1.0000 / 1.0000** | ≥ 0.995 ✓ |
| MiniVal240 (240 en queries) | 240/240 | **240/240** (480/480 both languages) | 240/240 ✓ |
| PairedVal20 members (40) | 40/40 | **40/40** (80/80 both languages) | 40/40 ✓ |
| Task 6S fixed-24 paraphrases | 21/24 | **24/24** | ≥ 23/24 ✓ |
| previously failing short zh prompts | all 3 wrong (`smallest_to_*`) | **all 3 correct** | required ✓ |
| Minimal-pair pack | 0.9474 (72/76) | **0.9868 (75/76)** | 1.000 ✗ |
| Stress v1 exact accuracy | 0.9375 (150/160) | **0.99375 (159/160)** | ≥ 0.95 ✓ |
| Stress v1 macro F1 | 0.9375 | **0.99375** | ≥ 0.95 ✓ |
| Stress v1 minimum class recall | — | **0.875** (`largest_to_nearest`) | ≥ 0.90 ✗ |
| Stress v1 by language | — | en **1.0000** (80/80), zh **0.9875** (79/80) | — |

Minimal-pair groups for the hardened parser: A 16/16 ✓, B 16/16 ✓, C 32/32 ✓, D **11/12** ✗.

## 7. Residual failures (reported, not repaired)

1. **Minimal pairs (1/76)**: the Chinese L2-nearest prompt
   `以面积最大的建筑区域为参照，请标注边界距离最小的另一建筑区域。` is misclassified (group D).
2. **Stress (1/160)**: the held-out-noun variant of the same phrasing
   `以面积最大的楼房区域为参照，请标注边界距离最小的另一楼房区域。` — dropping `largest_to_nearest`
   class recall to 0.875, below the 0.90 class floor.
3. **Scope safety (2/4 out-of-scope controls)**: the compact nearest controls
   `分割面积最大的建筑物右侧最近的建筑物。` and
   `segment the building nearest to the right of the largest building` are still classified as
   `largest_to_right_of` (direction-only) rather than `largest_to_right_of_to_nearest`, so the chain
   proceeds to the resolver and exits 3 instead of 5. The two other controls classify correctly
   (`largest`, `leftmost`) and exit 5. The augmentation grammar contains the *split* nearest pattern
   ("先确定 … 再从它右侧的 … 中选出最近的 …") but never the *compact* pattern ("面积最大的建筑物右侧最近的
   建筑物"), which is the residual gap. No keyword/regex gate was added (forbidden by the task), so this
   remains a genuine ProgramHead limitation.

## 8. Scope safety and end-to-end regression

`evaluation/task6t_cli_scope_safety.json` — the real CLI with the hardened ProgramHead as its default
parser (the CLI default reference was updated per section 9; the domain guard, the supported-8 list and
all downstream logic are unchanged):

| Control | Exit code | Parsed program | Downstream |
|---|---|---|---|
| `Write a poem about the sea.` | 4 ✓ | — (domain guard) | not called ✓ |
| `今天天气怎么样？` | 4 ✓ | — | not called ✓ |
| `检测道路。` | 4 ✓ | — | not called ✓ |
| `` (empty) | 4 ✓ | — | not called ✓ |
| `分割面积最大的建筑物。` | 5 ✓ | `largest` ✓ | not called ✓ |
| `分割最左侧的建筑物。` | 5 ✓ | `leftmost` ✓ | not called ✓ |
| `分割面积最大的建筑物右侧最近的建筑物。` | **3 ✗** | `largest_to_right_of` ✗ | reached resolver ✗ |
| `segment the building nearest to the right of the largest building` | **3 ✗** | `largest_to_right_of` ✗ | reached resolver ✗ |

The exact 24 fixed Task 6S paraphrases through the CLI: **24/24** parsed correctly, each reaching the
correct downstream branch.

`evaluation/task6t_end_to_end_regression.json` — MiniVal240 rerun through the frozen Task 6S chain with
only the ProgramHead replaced:

| Metric | Task 6S | Task 6T | Δ |
|---|---|---|---|
| parser exact accuracy | 1.0 | **1.0** | 0 |
| answered-only mIoU | 0.3045812554881724 | **0.3045812554881724** | **0.0** |
| strict all-240 mIoU | 0.2969667241009681 | **0.2969667241009681** | **0.0** |
| PairedVal20 | 10/20 | **10/20** | 0 |
| own−cross margin | 0.27370032940000916 | **0.27370032940000916** | 0.0 |
| abstentions | 6 | **6** | 0 |

→ `END_TO_END_REG_REPRODUCED`, i.e. exactly the expected result when only the parser changes and its
predictions stay identical.

*Implementation note recorded for transparency:* the first regression comparison reported
`END_TO_END_REGRESSION` although every delta was 0.0 — the abstention check compared
`abstention_rate * 240 == 6` in floating point (`0.025000000000000022 * 240 = 6.000000000000005`). The
check now rounds to the integer count; no model, threshold or metric definition changed.

## 9. Predeclared gates (section 12) and verdict (section 13)

| # | Gate | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | baseline parser hash verified | exact | exact | ✓ |
| 2 | zero train/eval prompt leakage | 0 | 0 / 0 | ✓ |
| 3 | same Qwen3-VL-2B text-only 20-class architecture | yes | yes (17.48 M trainable) | ✓ |
| 4 | full v0.2 val exact accuracy | ≥ 0.995 | **1.0000** | ✓ |
| 5 | MiniVal240 | 240/240 | **240/240** | ✓ |
| 6 | PairedVal20 members | 40/40 | **40/40** | ✓ |
| 7 | fixed24 | ≥ 23/24 | **24/24** | ✓ |
| 8 | the three previously failing prompts | all correct | **all correct** | ✓ |
| 9 | minimal-pair pack | 1.000 | **0.9868** | **✗** |
| 10 | stress v1 exact accuracy | ≥ 0.95 | **0.99375** | ✓ |
| 11 | stress v1 macro F1 | ≥ 0.95 | **0.99375** | ✓ |
| 12 | every stress class recall | ≥ 0.90 | **0.875** | **✗** |
| 13 | OOD controls exit 4 | 4/4 | **4/4** | ✓ |
| 14 | out-of-scope controls classify + exit 5 | 4/4 | **2/4** | **✗** |
| 15 | end-to-end mIoU within 1e-6 | ≤ 1e-6 | **0.0** | ✓ |
| 16 | PairedVal remains 10/20 | 10 | **10** | ✓ |
| 17 | no test split | yes | yes | ✓ |
| 18 | no proposal/B3/field/SAM2 change | yes | yes | ✓ |
| 19 | no keyword/regex semantic override | yes | yes (0 findings) | ✓ |

Section 13 priority order applied literally: not `INVALID_EXPERIMENT` (protocol clean),
not `BASELINE_PARSER_UNAVAILABLE` (hash verified), not `PARSER_TRAINING_FAILED` (valid finite checkpoint),
not `PARSER_CANONICAL_REGRESSION` (all canonical gates pass), therefore
**`PARSER_SEMANTIC_CONTRAST_FAIL`** (fixed24/minimal-pair/stress/scope-safety gates 9, 12 and 14 fail).
`END_TO_END_REGRESSION` does not apply (the downstream reproduces Task 6S exactly).

## 10. Interpretation boundary

Task 6T is a small parser/scope-safety hardening step required to close the Task 6S interface gate. It
does **not** address the main scientific bottleneck, which remains **`REFERENCE`** (Task 6S: 117 of 240
records), and it does not extend the parser to nearest/L3 execution in the target model. The hardening
measurably improved paraphrase robustness (fixed24 21 → 24/24, minimal pairs 0.947 → 0.987, stress
0.938 → 0.994, all three previously failing short Chinese prompts fixed) while leaving every canonical
metric bit-identical, but three predeclared semantic/scope gates still fail, all traced to the
Chinese/compact L2-nearest phrasing ("边界距离最小的", "[X]右侧最近的[Y]"). DSH does not repair this and
does not choose the next direction.

## 11. Reproduce

```text
python scripts/task6t_build_parser_data.py --stage packs     # frozen packs (before training)
python scripts/task6t_build_parser_data.py --stage augment   # augmentation spec + dataset
python scripts/task6t_build_parser_data.py --stage leakage   # leakage audit (must be 0)
python scripts/task6t_eval_parser.py --stage baseline        # section 7.1 baseline
python scripts/task6t_train_parser.py --configs C1,C2,C3 --max-epochs 1
python scripts/task6t_eval_parser.py --stage final           # section 8 final evaluations
python scripts/task6t_scope_audit.py --stage scope           # section 9 scope safety
python scripts/task6t_scope_audit.py --stage e2e             # section 10 end-to-end regression
python scripts/task6t_report.py                              # sections 12-13 gates + verdict
```

Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1` so the frozen parser loads from the local
cache and nothing is downloaded.
