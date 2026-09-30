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

# FROM_DSH — Task 7C Report: 20-Class Rehearsal + L3 ProgramHead Hardening

_This file holds the Task 7C report. The Task 7B report is preserved in git history at commit `b325585`;
Task 7A at `6ee3d0d`; Task 6Z at `7006d7d`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7c_20class_rehearsal_l3_hardening.md`.

## 1. Verdict

**`L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`** — section 26 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, leakage zero, no keyword/regex override, no sweep,
   no test use, initialized from Task 6T only.
2. `BASELINE_PARSER_UNAVAILABLE` — no: Task 6T SHA256 `4cbba36b…d44a5e` exact.
3. `PARSER_TRAINING_FAILED` — no: valid checkpoint `c1505736…d58d9a`.
4. `L3_20CLASS_CANONICAL_REGRESSION` — **no: all fourteen canonical gates pass** (full v0.2 val
   18,222/18,222 = 1.0000 accuracy, macro F1 1.0000, every class recall 1.0000; Z-MiniVal240 240/240;
   Z-PairedVal 40/40).
5. **`L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`** — the canonical gates pass but the compositional
   robustness gates fail: fixed24 **5/24** (< 22), compact **0/8**, per L3 class 1/0/2/2 (< 5/6),
   minimal96 **60/96** (< 96), stress accuracy **0.6667** (< 0.97), macro F1 **0.5924** (< 0.97),
   minimum class recall **0.4167** (< 0.90), L3 macro recall **0.4479** (< 0.95). ← **verdict**
6. `END_TO_END_REGRESSION` — not reached (item 5 applies first); the frozen downstream does reproduce
   Task 7A exactly.
7. `L3_20CLASS_REHEARSAL_PASS` — no.

No threshold was changed after seeing results.

## 2. Recorded Task 7B result (Task 7B artifacts not mutated)

`L3_PARSER_CANONICAL_REGRESSION`: full v0.2 val accuracy `0.7129`, macro F1 `0.7939`, minimum class recall
`0.0000`, Z-MiniVal240 `240/240`, Z-Paired members `40/40`, fixed24 `21/24`, compact `7/8`, minimal pairs
`73/96`, stress `0.8281`. Task 7B trained 4,800 synthetic rows covering only 14/20 classes; the six
zero-support classes had no internal-holdout support either, so 1.0 holdout accuracy could not detect the
forgetting. This was a **task-file protocol-design error, not a DSH implementation error**.

## 3. Baseline, architecture and reused evaluation material

Initialization is **only** the stable Task 6T checkpoint (`4cbba36b…d44a5e` exact); the Task 7B checkpoint is
never used and the new checkpoint hash differs from it. Architecture preserved: Qwen3-VL-2B, text-only, same
tokenizer/prompt format, same LoRA + ProgramHead policy (17,434,624 trainable / 2,144,421,888 total, backbone
frozen), exactly 20 classes. `task7c_reused_eval_manifest.json` hashes every frozen evaluation source without
regenerating anything; original v0.2 train instruction text is never used for optimization (the train file is
read only for the class-id schema, because v0.2 emits the same 120 templates in train and val).

## 4. The 8,000-prompt all-class rehearsal

| Group | Classes | Per class | zh / en | Subtotal |
|---|---|---|---|---|
| L3 direction+nearest | 4 | **800** | 400 / 400 | 3,200 |
| Remaining classes | 16 | **300** | 150 / 150 | 4,800 |
| **Total** | 20 | — | — | **8,000** |

All 8,000 normalized prompts are unique; exact overlap with the evaluation union **0**, normalized overlap
**0**, normalized duplicates **0**, per-class and zh/en counts exact → `REHEARSAL_CLEAN`. L2 and L3 templates
share the `verb + reference + direction + noun` stems (the L3 form inserts one nearest phrase), only the
section-10/11 TRAIN pools are used, and the Task 7A/7B/full-val phrasings are reserved.

## 5. Internal split and optimization

Per-class 90/10 by normalized-prompt group (seed 20261001): holdout support **80/class for L3** and
**30/class for the other 16 classes**, every class supported, train↔holdout normalized overlap **0**. Frozen
protocol: AdamW, ProgramHead lr **1e-4**, LoRA lr **2e-5**, weight decay 1e-4, effective batch 32
(microbatch 16 × accumulation 2), max 5 epochs, patience 2, bfloat16 AMP, grad clip 1.0, no scheduler, no
sweep. Selection by `min(macro_f1_20, l3_macro_recall)` → minimum class recall → macro F1 → accuracy →
earlier epoch, internal holdout only.

Selected epoch **2**: holdout primary 1.0000, macro F1 1.0000, L3 macro recall 1.0000, minimum class recall
1.0000, accuracy 1.0000. Wall **419.3 s**, peak VRAM **7.41 GB**, checkpoint
`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt`, SHA256
`c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a` (gitignored).

## 6. Final evaluation

| Set | Result | Gate | Pass |
|---|---|---|---|
| Full v0.2 val (18,222) | **1.0000** / **1.0000** / min recall **1.0000** | ≥0.995 / ≥0.995 / ≥0.98 | ✓ |
| Z-MiniVal240 | **240/240** | 240/240 | ✓ |
| Z-PairedVal members | **40/40** | 40/40 | ✓ |
| Task 7A fixed24 | **5/24** · compact **0/8** · per class 1/0/2/2 | ≥22/24 · 8/8 · ≥5/6 | ✗ |
| Task 7B minimal96 | **60/96** | 96/96 | ✗ |
| Task 7B stress192 | **0.6667** · F1 **0.5924** · min recall **0.4167** · L3 **0.4479** | ≥0.97 / ≥0.97 / ≥0.90 / ≥0.95 | ✗ |

Canonical behaviour is fully restored (no forgetting at all), while L3 paraphrase robustness is lower than
Task 7B's (fixed24 5/24 vs 21/24, minimal 60 vs 73, stress 0.6667 vs 0.8281).

## 7. Scope safety, frozen regression and CLI default

The reused Task 7B scope controls all pass: L2 directional controls parse as `largest_to_left_of` /
`largest_to_right_of` and exit **5** before proposals/reference/fields/SAM2/Z-B3, the nearest-only controls
parse as `largest_to_nearest` and exit 5, OOD prompts keep their guard behaviour, no keyword/regex gate.

The frozen downstream reproduces Task 7A **exactly**: parser 240/240, strict `0.21700369907681483` (Δ 0.0),
answered `0.21975058134361` (Δ 0.0), abstentions **3**, Paired **8/20**, margin `0.19949275176250805`
(Δ 0.0). Because the canonical gates pass, section 24 moves the L3 CLI default parser to the Task 7C
checkpoint (`default_l3_parser_checkpoint()` consults the newest verdict whose canonical gates passed); the
Task 7A helper and every frozen Task 7A/7B artifact are untouched.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **1189 passed, 1 skipped** (Task 7B ended at 1148 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task7c_rehearsal_hardening.py` adds the 41 section-Q checks, and the
Task 7B CLI-default test was generalised to the "latest passing verdict wins" invariant.

Not committed: the Task 7C checkpoint, generated rehearsal rows
(`artifacts/task7c/parser_rehearsal/`), model/tokenizer caches, YOLO/SAM2/Z-B3 weights, source imagery/vectors,
`.conda`. Committed: small specs/audits/evaluations, scripts, tests, docs, handoff.

Task 7C downloaded nothing and installed nothing. Watt was **not needed** in Task 7C: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no
proxy, host, certificate or TLS setting was read or modified.

## 9. Interpretation boundary

DSH reports measurements only. The parser is not claimed as a novelty; no further parser repair was run; the
reference subsystem was not reopened; Z-B3/fields/reference were not modified; no attention/global
competition was added; no formal full training or test evaluation started; Task 7D was not chosen.

## 10. Recommended next step (exact wording required by Part O)

等待 ChatGPT 根据 Task 7C 的 20-class rehearsal 与 L3 组合语义结果决定 parser 是否正式冻结，不自行继续 parser 调参、reference hardening 或下游架构改造。

## 11. STOP

Task 7C stops here: no further parser experiment, no reference re-hardening, no downstream architecture
change, no attention/global competition, no formal full training or test, no GUI. Waiting for the ChatGPT
audit.
