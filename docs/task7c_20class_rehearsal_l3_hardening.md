# Task 7C — 20-Class Rehearsal + L3 ProgramHead Hardening

> Task: `handoff/TO_DSH.md` (Task 7C) · Base commit: `b325585` · Predecessor: Task 7B →
> `L3_PARSER_CANONICAL_REGRESSION`
> **Verdict: `L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`** · canonical gates **all pass** (full val 1.0000,
> min class recall 1.0000) · compositional gates fail
> Tests: `tests/test_task7c_rehearsal_hardening.py` · Evidence: `evaluation/task7c_*.json`

Task 7C is the final parser-hardening attempt: restart from the stable Task 6T checkpoint, train once on an
8,000-prompt **all-20-class** rehearsal set with fixed differential learning rates, then evaluate once on the
already-frozen Task 7A/7B sets. Everything downstream (YOLO, U-C1 resolver, direction/nearest fields, SAM2,
Z-B3) stays frozen and no keyword/regex remapping exists.

## 1. Recorded Task 7B result (Task 7B artifacts not mutated)

`L3_PARSER_CANONICAL_REGRESSION`: full v0.2 val accuracy `0.7129`, macro F1 `0.7939`, minimum class recall
`0.0000`; Z-MiniVal240 `240/240`; Z-Paired members `40/40`; Task 7A fixed24 `21/24`; compact `7/8`;
minimal pairs `73/96`; stress `0.8281`. Task 7B trained 4,800 synthetic rows covering only 14/20 classes;
the six zero-support classes (`leftmost`, `rightmost`, `topmost`, `bottommost`, `largest`, `smallest`) also
had no internal-holdout support, so 1.0 holdout accuracy could not detect the forgetting. This was a
**protocol-design error in the task file, not a DSH implementation error**. Task 7B artifacts and verdict
are untouched.

## 2. Baseline, architecture and frozen evaluation material

Initialization is **only** `artifacts/checkpoints/task6t/program_parser_hardened_v1.pt`
(SHA256 `4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e` exact); the Task 7B checkpoint is
never used (`initialized_from_task7b = false`, and the new checkpoint hash differs from it). Architecture
preserved: Qwen3-VL-2B, text-only, same tokenizer/prompt format, same LoRA parameters + ProgramHead
(17,434,624 trainable / 2,144,421,888 total), base backbone frozen, exactly 20 classes.

`evaluation/task7c_reused_eval_manifest.json` hashes every evaluation source without regenerating it
(full v0.2 val, Z-MiniVal240, Z-PairedVal20 members, Task 7A fixed24, Task 7B minimal96, stress192, scope
controls). Original BuildSpatialReason v0.2 train instruction **text** is never used for optimization (the
train file is read only for the class-id schema — v0.2 emits the same 120 templates in train and val, so
train text would leak into full-val evaluation).

## 3. The 8,000-prompt all-class rehearsal

`evaluation/task7c_rehearsal_spec.json` + `task7c_training_data_audit.json`:

| Group | Classes | Per class | zh / en | Subtotal |
|---|---|---|---|---|
| L3 direction+nearest | 4 | **800** | 400 / 400 | 3,200 |
| Remaining classes | 16 | **300** | 150 / 150 | 4,800 |
| **Total** | 20 | — | — | **8,000** |

All 8,000 normalized prompts are **unique**; exact overlap with the evaluation union **0**, normalized
overlap **0**, normalized duplicates **0**, per-class and zh/en counts exact → `REHEARSAL_CLEAN`. L2 and L3
templates share the `verb + reference + direction + noun` stems and the L3 forms insert one nearest phrase,
so a style cue cannot separate them. Only the section-10/11 TRAIN lexical pools were used; the Task 7A
fixed24, Task 7B minimal96/stress192 and full-val phrasings are reserved and never emitted.

## 4. Internal split and optimization

Per class 90/10 by normalized-prompt group (seed 20261001): holdout support **80/class for L3** and
**30/class for the other 16** — every class supported, train↔holdout normalized overlap **0**. Frozen
protocol: AdamW, ProgramHead lr **1e-4**, LoRA lr **2e-5**, weight decay 1e-4, effective batch 32
(microbatch 16 × accumulation 2), max 5 epochs, patience 2, bfloat16 AMP, grad clip 1.0, no scheduler, no
sweep. Selection by `min(macro_f1_20, l3_macro_recall)` → minimum class recall → macro F1 → accuracy →
earlier epoch, using the internal holdout only.

Selected epoch **2** with holdout primary 1.0000, macro F1 1.0000, L3 macro recall 1.0000, minimum class
recall 1.0000, accuracy 1.0000. Wall **419.3 s**, peak VRAM **7.41 GB**, checkpoint
`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt`, SHA256
`c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a` (gitignored).

## 5. Final evaluation (six reused frozen sets)

| Set | Result | Gate | Pass |
|---|---|---|---|
| Full v0.2 val (18,222 prompts) | **1.0000** accuracy · **1.0000** macro F1 · min class recall **1.0000** | ≥0.995 / ≥0.995 / ≥0.98 | ✓ |
| Z-MiniVal240 | **240/240** | 240/240 | ✓ |
| Z-PairedVal members | **40/40** | 40/40 | ✓ |
| Task 7A fixed24 | **5/24** · compact **0/8** · per L3 class 1/0/2/2 | ≥22/24 · 8/8 · ≥5/6 | ✗ |
| Task 7B minimal96 | **60/96** | 96/96 | ✗ |
| Task 7B stress192 | **0.6667** · macro F1 **0.5924** · min class recall **0.4167** · L3 macro **0.4479** | ≥0.97 / ≥0.97 / ≥0.90 / ≥0.95 | ✗ |

**Canonical behaviour is fully restored** — the 20-class rehearsal removed the Task 7B forgetting entirely
(full val 18,222/18,222 with every class recall 1.0000, Z-MiniVal240 and Z-PairedVal perfect).

## 6. Scope safety and frozen end-to-end regression

The reused Task 7B scope controls all pass (`scope.passed = true`): L2 directional controls parse as
`largest_to_left_of` / `largest_to_right_of` and exit **5** before proposals/reference/fields/SAM2/Z-B3, the
nearest-only controls parse as `largest_to_nearest` and exit 5, OOD prompts keep their guard behaviour, and
no keyword/regex gate exists.

The frozen downstream reproduces Task 7A **exactly** with the Task 7C parser: parser 240/240, strict
`0.21700369907681483` (Δ 0.0), answered `0.21975058134361` (Δ 0.0), abstentions **3**, Paired **8/20**,
margin `0.19949275176250805` (Δ 0.0).

Because the canonical gates **pass**, Task 7C section 24 moves the L3 CLI default parser to the Task 7C
checkpoint: `default_l3_parser_checkpoint()` now returns
`artifacts/checkpoints/task7c/program_parser_l3_rehearsal_v1.pt` (the Task 7A helper and all frozen
Task 7A/7B artifacts are untouched; the helper consults the newest verdict whose canonical gates passed).

## 7. Verdict

Section 26 priority: protocol clean, baseline hash exact, a valid checkpoint exists, and **the canonical
gates all pass** — but the compositional robustness gates fail (fixed24 5/24 with 0/8 compact, minimal96
60/96, and all four stress gates) → **`L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL`**.

Measured reading (reported only, no repair proposed): the rehearsal protocol solved the canonical-forgetting
failure it was designed for, but the rehearsal phrasing does not transfer to the frozen paraphrase packs —
the all-class rehearsal deliberately reserves the evaluation lexical forms, so the model learned the
rehearsal L3 phrasings rather than the held-out ones; compared with Task 7B (fixed24 21/24, minimal 73/96,
stress 0.8281) the L3 paraphrase robustness is *lower*, while canonical accuracy went from 0.7129 to 1.0000.

## 8. Interpretation boundary

DSH reports measurements only. The parser is not claimed as a novelty; no further parser repair was run; the
reference subsystem was not reopened; Z-B3/fields/reference were not modified; no attention/global
competition was added; no formal full training or test evaluation started; Task 7D was not chosen. Final
recommendation exactly:

`等待 ChatGPT 根据 Task 7C 的 20-class rehearsal 与 L3 组合语义结果决定 parser 是否正式冻结，不自行继续 parser 调参、reference hardening 或下游架构改造。`

## 9. Reproduce

```text
python scripts/task7c_build_rehearsal.py         # Parts C-G reused manifest, 8,000-prompt rehearsal, audit
python scripts/task7c_train_parser.py            # Parts H-I split + reduced-adaptation training
python scripts/task7c_eval_parser.py             # Part J six reused evaluation sets
python scripts/task7c_scope_audit.py --canonical-gates-passed   # Part K scope safety
python scripts/task7c_e2e_regression.py          # Part L frozen downstream regression
python scripts/task7c_report.py                  # Parts M-N gates + verdict
```

Generated rehearsal rows live in the gitignored `artifacts/task7c/parser_rehearsal/`; the Task 7C checkpoint
is gitignored too. Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the data, scope and report
steps also run in `.conda/buildreasonseg-mvp`.
