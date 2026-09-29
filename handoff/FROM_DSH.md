<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 6T Report: ProgramHead Semantic Hardening + Scope-Safety Regression

_This file holds the Task 6T report. The Task 6S report is preserved in git history at commit `dc8544f`;
Task 6R at `a3d59da`; Task 6Q at `7c19bec`; Task 6P at `b80f3cc`; Task 6O at `595e7bb`._

**Note on the legacy block above:** those `ARTIFACT-FACTS` numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6t_programhead_semantic_hardening.md`.

## 1. Verdict

**`PARSER_SEMANTIC_CONTRAST_FAIL`** — section 13 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: zero leakage, no test use, no frozen non-parser mutation, no keyword/regex
   override (0 findings), baseline hash verified.
2. `BASELINE_PARSER_UNAVAILABLE` — no: SHA256 `eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3` verified exactly.
3. `PARSER_TRAINING_FAILED` — no: a valid finite checkpoint was produced (holdout macro F1 1.0000).
4. `PARSER_CANONICAL_REGRESSION` — no: full v0.2 val **1.0000**, MiniVal240 **240/240**, PairedVal **40/40**.
5. **`PARSER_SEMANTIC_CONTRAST_FAIL`** — canonical gates pass but gates 9 (minimal-pair pack 0.9868 < 1.0),
   12 (`largest_to_nearest` stress class recall 0.875 < 0.90) and 14 (2 of 4 nearest scope controls still
   classify `largest_to_right_of` and exit 3) fail. ← **verdict**
6. `END_TO_END_REGRESSION` — no: the frozen downstream reproduces Task 6S exactly.
7. `PARSER_HARDENING_PASS` — no.

## 2. Mandatory Task 6S erratum (recorded; Task 6S artifacts and verdict untouched)

The Task 6S statement *"the frozen 20-program vocabulary has no nearest program"* is **false**.
`buildreasonseg_mvp/structured_grounding.py::EXPECTED_QUERY_TYPES` contains six nearest classes:
`largest_to_nearest`, `smallest_to_nearest`, `largest_to_above_to_nearest`, `largest_to_below_to_nearest`,
`largest_to_left_of_to_nearest`, `largest_to_right_of_to_nearest`. The two Task 6S nearest controls
therefore failed because the frozen ProgramHead **misclassified nearest-containing instructions as
direction-only programs**, not because nearest semantics were missing from the vocabulary.

## 3. Frozen scope

Only the ProgramHead checkpoint/training data and the parser-evaluation code changed. `git diff` against
the base commit `dc8544f` shows **no** change to `task6q_reference_resolver.py`,
`geometric_relation_field_v02.py`, `task6n_relation_decoder.py`, `program_parser.py`,
`structured_grounding.py`, `evaluation/task6q_*`, `evaluation/task6s_*` and `evaluation/task6o_*`. The CLI
default parser reference was updated to the hardened checkpoint (the only `predict_buildreasonseg_directional.py`
change); the domain guard, the supported-8 list and every downstream module are unchanged. No GRCL, no 4B,
no new dataset, no download/install, no test split.

## 4. No-leakage hardening data and frozen packs

| Item | Value |
|---|---|
| v0.2 **train** examples (both languages) | 25,556 → **24,161 kept** (1,395 exact/normalized duplicates of v0.2 val queries removed) |
| Deterministic paraphrase augmentation | **1,800** kept (8-program × 2-language grammar; 0 collisions) |
| Combined training prompts | **25,961** |
| Leakage audit | **exact 0 / normalized 0** vs MiniVal240, PairedVal20, the Task 6S fixed-24 and both Task 6T packs → `NO_LEAKAGE` |
| Minimal-pair pack (frozen pre-training) | **76** prompts: A 16, B 16, C 32, D 12 |
| Stress v1 (frozen pre-training) | **160** prompts: all 20 classes × 4 zh + 4 en, held-out template/lexical forms |

*Recorded dataset fact:* the frozen v0.2 **train** split itself contains 1,395 instruction strings that
are exact/normalized duplicates of v0.2 **val** queries; the deterministic pre-training filter removes
them from the hardening training set. No dataset file or split was changed.

*Pack-construction integrity:* both packs are validated by construction (no prompt maps to two programs;
every direction prompt contains its direction word). This validation caught and fixed two template defects
(a Chinese direction template missing its direction word, and an English template producing
"on the on the left side of side of") **before the final training run**; the packs were regenerated, the
leakage audit re-run (0/0) and the whole baseline → training → evaluation pipeline re-executed.

## 5. Training

| Item | Value |
|---|---|
| Initialisation | authoritative Task 6S checkpoint (hash verified), same Qwen3-VL-2B text-only 20-class head |
| Trainable policy | Task 6M policy — LoRA adapters + ProgramHead, backbone frozen |
| Trainable / total parameters | **17,479,700** / 2,144,466,964 |
| Declared configs / run | C1 (6M recipe), C2 (half LR), C3 (quarter LR) / **C1 only** |
| C1 epoch 1 | train loss **0.0067**, internal-holdout **macro F1 1.0000**, accuracy 1.0000 → selection ceiling, sweep terminated |
| Optimizer / schedule | AdamW β 0.9/0.999, wd 0.01, grad clip 1.0, bf16 autocast, seed 20260930, constant LRs |
| Internal holdout | 90/10 by `sha256(seed:normalized_prompt)`, stratified by program (23,195 / 2,578) |
| Selection metric | internal-holdout macro F1, tie-break exact accuracy (never MiniVal240/fixed24/stress/test) |
| Wall time / peak VRAM | **321.1 s** / 0.67 GB |
| Checkpoint (local only) | `artifacts/checkpoints/task6t/program_parser_hardened_v1.pt`, SHA256 `4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e` |

## 6. Baseline vs hardened parser

| Metric | Baseline | **Hardened** | Gate |
|---|---|---|---|
| full v0.2 val exact accuracy (en / zh) | 1.0000 / 1.0000 | **1.0000 / 1.0000** | ≥ 0.995 ✓ |
| MiniVal240 (240 en queries) | 240/240 | **240/240** | 240/240 ✓ |
| PairedVal20 members (40) | 40/40 | **40/40** | 40/40 ✓ |
| Task 6S fixed-24 paraphrases | 21/24 | **24/24** | ≥ 23/24 ✓ |
| previously failing short zh prompts | all 3 wrong (`smallest_to_*`) | **all 3 correct** (`largest_to_*`) | required ✓ |
| minimal-pair pack | 0.9474 (72/76) | **0.9868 (75/76)** | 1.000 ✗ |
| stress v1 accuracy / macro F1 | 0.9375 / 0.9375 | **0.99375 / 0.99375** | ≥ 0.95 / ≥ 0.95 ✓ |
| stress v1 minimum class recall | — | **0.875** (`largest_to_nearest`) | ≥ 0.90 ✗ |
| stress v1 by language | — | en **1.0000**, zh **0.9875** | — |

Minimal-pair groups (hardened): A 16/16 ✓, B 16/16 ✓, C 32/32 ✓, D **11/12** ✗.

## 7. Residual failures (reported, not repaired)

1. Minimal pairs 1/76 and stress 1/160: the Chinese L2-nearest phrasing with the boundary-distance
   superlative (`以面积最大的建筑区域为参照，请标注边界距离最小的另一建筑区域。` and its held-out-noun
   variant with 楼房区域) is misclassified, dropping `largest_to_nearest` class recall to 0.875.
2. Scope safety 2/4: the compact nearest controls
   `分割面积最大的建筑物右侧最近的建筑物。` and
   `segment the building nearest to the right of the largest building` are still classified as
   `largest_to_right_of`, so the chain reaches the resolver and exits **3** instead of **5**. The
   augmentation grammar covers the *split* nearest pattern but never the *compact* one. No keyword/regex
   gate was added (forbidden), so this remains a genuine ProgramHead limitation.

## 8. Scope safety and end-to-end regression

`evaluation/task6t_cli_scope_safety.json` — OOD controls **4/4 exit 4** before ProgramHead/downstream;
out-of-scope controls: `分割面积最大的建筑物。` → `largest` exit 5 ✓, `分割最左侧的建筑物。` → `leftmost`
exit 5 ✓, the two nearest controls → `largest_to_right_of` exit 3 ✗; the exact 24 fixed paraphrases through
the CLI: **24/24** correct, each reaching the correct downstream branch.

`evaluation/task6t_end_to_end_regression.json` — MiniVal240 through the frozen Task 6S chain with only the
ProgramHead replaced: parser **1.0**; answered-only mIoU **0.3045812554881724 (Δ 0.0)**, strict all-240
mIoU **0.2969667241009681 (Δ 0.0)**, PairedVal **10/20**, own−cross margin **0.27370032940000916 (Δ 0.0)**,
abstentions **6** → `END_TO_END_REG_REPRODUCED`.

*Recorded implementation note:* the first regression comparison reported `END_TO_END_REGRESSION` despite
all deltas being 0.0 — the abstention check compared `abstention_rate * 240 == 6` in floating point
(`0.025000000000000022 * 240 = 6.000000000000005`). Fixed by rounding to the integer count; no model,
threshold or metric definition changed.

## 9. Predeclared gates

| # | Gate | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | baseline parser hash verified | exact | exact | ✓ |
| 2 | zero prompt leakage | 0 | 0 / 0 | ✓ |
| 3 | same 2B text-only 20-class architecture | yes | yes | ✓ |
| 4 | full v0.2 val exact accuracy | ≥ 0.995 | 1.0000 | ✓ |
| 5 | MiniVal240 | 240/240 | 240/240 | ✓ |
| 6 | PairedVal20 members | 40/40 | 40/40 | ✓ |
| 7 | fixed24 | ≥ 23/24 | 24/24 | ✓ |
| 8 | three previously failing prompts | all correct | all correct | ✓ |
| 9 | minimal-pair pack | 1.000 | **0.9868** | ✗ |
| 10 | stress v1 accuracy | ≥ 0.95 | 0.99375 | ✓ |
| 11 | stress v1 macro F1 | ≥ 0.95 | 0.99375 | ✓ |
| 12 | every stress class recall | ≥ 0.90 | **0.875** | ✗ |
| 13 | OOD controls exit 4 | 4/4 | 4/4 | ✓ |
| 14 | out-of-scope controls classify + exit 5 | 4/4 | **2/4** | ✗ |
| 15 | end-to-end mIoU within 1e-6 | ≤ 1e-6 | 0.0 | ✓ |
| 16 | PairedVal remains 10/20 | 10 | 10 | ✓ |
| 17 | no test split | yes | yes | ✓ |
| 18 | no proposal/B3/field/SAM2 change | yes | yes | ✓ |
| 19 | no keyword/regex override | yes | yes (0 findings) | ✓ |

No gate or threshold was changed.

## 10. Tests, storage, git

`python -m pytest tests/ -q` → **807 passed, 1 skipped** (Task 6S ended at 773 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6t_programhead_hardening.py` adds the 34 section-15 checks.

Not committed: the new parser checkpoint, model weights, tokenizer/model cache, the generated augmented
JSONL (large, under the gitignored `artifacts/task6t/`), source imagery, SAM2/YOLO/B3 assets, `.conda`.
Committed: augmentation specification, small frozen eval packs, small evaluation JSON, scripts, tests,
docs, handoff.

The frozen parser was loaded with `HF_HUB_OFFLINE=1` from the existing local cache; Task 6T downloaded
nothing (no weights, packages or datasets) and installed nothing. Watt was **not needed** in Task 6T: the
pre-existing Watt instance is transport-only, is not owned by this project and was left running per the
ownership rule; no proxy, host, certificate or TLS setting was read or modified.

## 11. Interpretation boundary

Task 6T is a small parser/scope-safety hardening step that closes part of the Task 6S interface gate. It
does not address the main scientific bottleneck, which remains **`REFERENCE`** (Task 6S: 117 of 240
records), and it does not add nearest/L3 execution to the target model. DSH does not repair the residual
nearest-phrasing failures and does not choose the next direction.

## 12. Recommended next step

等待 ChatGPT 根据 Task 6T 的 parser hardening 结果决定下一步（继续 parser/scope 修复或转向 reference-hardening），不自行修复或扩展范围。

## 13. STOP

Task 6T stops here: no REFERENCE-bottleneck repair, no YOLO tuning, no proposal conf/imgsz/max_det change,
no tiling/TTA, no reference ranker, no B3 retraining, no relation-field change, no GRCL, no nearest/L3
execution in the target model, no test access, no GUI. Waiting for the ChatGPT audit.
