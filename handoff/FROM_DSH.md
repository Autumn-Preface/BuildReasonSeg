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

# FROM_DSH — Task 7B Report: L3 ProgramHead Compositional-Semantic Hardening

_This file holds the Task 7B report. The Task 7A report is preserved in git history at commit `6ee3d0d`;
Task 6Z at `7006d7d`; Task 6Y at `24be954`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7b_l3_programhead_hardening.md`.

## 1. Verdict

**`L3_PARSER_CANONICAL_REGRESSION`** — section 24 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, leakage zero, no keyword/regex override, no sweep,
   no test use.
2. `BASELINE_PARSER_UNAVAILABLE` — no: Task 6T baseline SHA256 `4cbba36b…d44a5e` exact.
3. `PARSER_TRAINING_FAILED` — no: a valid checkpoint was produced (`2a285e4a…cf48eb4`).
4. **`L3_PARSER_CANONICAL_REGRESSION`** — the canonical evaluation regressed: full v0.2 val accuracy
   **0.7129 < 0.995**, macro F1 **0.7939 < 0.995**, minimum class recall **0.0000 < 0.98**
   (`leftmost` collapsed to 0.0000), while Z-MiniVal240 **240/240** and Z-PairedVal **40/40** stayed
   perfect. ← **verdict**
5. `L3_PARSER_COMPOSITIONAL_ROBUSTNESS_FAIL` — not reached (canonical gates already fail) although the
   compositional gates also fail: fixed24 21/24 (compact 7/8), minimal pairs 73/96, stress 0.8281.
6. `END_TO_END_REGRESSION` — no: the frozen downstream reproduces Task 7A exactly.
7. `L3_PROGRAMHEAD_HARDENING_PASS` — no.

No threshold was changed after seeing results.

## 2. Recorded Task 7A findings (Task 7A artifacts not mutated)

`L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`; A0 oracle Z-B3 mIoU 0.3242128982543474 / Dice
0.4389840055529761 / Paired 15/20 / margin 0.30035408969722216; A1 strict 0.21700369907681483, answered
0.21975058134361, retention 0.6693246944992726, Paired 8/20, margin 0.19949275176250805, reference
abstention 3/240, dominant bottleneck `REFERENCE`; language: canonical 240/240, paired members 40/40,
fixed paraphrase 3/24, compact 0/8, dominant failure = terminal `to_nearest` dropped to L2. Task 7B does not
reinterpret Task 7A as a success.

## 3. Frozen baseline, architecture and evaluation packs

Baseline = Task 6T hardened parser, SHA256 `4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e`
exact. Architecture preserved: Qwen3-VL-2B, text-only (no image input), same tokenizer/prompt formatting,
same Task 6M trainable policy (LoRA + ProgramHead, backbone frozen; 17,434,624 trainable /
2,144,421,888 total), exactly 20 classes, no `unsupported` class, no keyword/regex remap.

Frozen **before** training (`task7b_eval_prompt_manifest.json`): full v0.2 val (9,111 records → 18,222
bilingual prompts), Z-MiniVal240 (240), Z-PairedVal20 members (40), Task 7A fixed24 (24), Task 7B minimal
pairs (**96**: M1 48 / M2 16 / M3 16 / M4 16; 48 zh + 48 en) and Task 7B stress v1 (**192**: L3 4×24 = 96
with 12 zh + 12 en each, L2 direction 4×12 = 48, `largest_to_nearest` 16, `smallest_to_nearest` 8,
`largest` 8, `smallest` 8, extremes 8; 96 zh + 96 en). No Task 7A fixed24 string is reused; the minimal-pair
and stress lexical pools are reserved from the training grammar.

## 4. Training data and leakage audit

**All 12,778 original v0.2 train records were removed** by the section-8 rule: BuildSpatialReason v0.2
emits the same 120 instruction templates in train and val (120 distinct normalized texts each, **100 %
intersection**), so every train prompt overlaps the evaluation set exactly. Cleaned train rows = **0**.

Exactly **4,800** deterministic augmentations were generated with the section-9 allocation — L3 600 each
(300 zh + 300 en), `largest_to_<direction>` 300 each, `largest_to_nearest` 400, `smallest_to_nearest` 200,
`smallest_to_<direction>` 150 each — using only the section-10 TRAIN lexical pools. Leakage audit:
augmentation ↔ evaluation **exact 0**, **normalized 0**, cleaned-train 0/0 → **`LEAKAGE_FREE`**.

## 5. Training

Internal group-disjoint, class-stratified split (seed 20261001): 4,399 train / 401 holdout, normalized
overlap **0**. Frozen protocol: AdamW lr 2e-4, weight decay 1e-4, effective batch 32 (microbatch 16 ×
accumulation 2), seed 20261001, max 8 epochs, patience 2, bfloat16 AMP, no sweep. Selected epoch **1**
(holdout accuracy 1.0000, L3 macro recall 1.0000, macro F1 0.7000 — only 14 of 20 classes have any support).
Wall **417.9 s**, peak VRAM **6.922 GB**, checkpoint `artifacts/checkpoints/task7b/program_parser_l3_hardened_v1.pt`,
SHA256 `2a285e4ac91051cb2cfcea370d384fb50f29927832ad30d3baf09c672cf48eb4` (gitignored).

## 6. Evaluation

| Set | Result | Gate | Pass |
|---|---|---|---|
| Full v0.2 val | **0.7129** accuracy · macro F1 **0.7939** · min class recall **0.0000** | ≥0.995 / ≥0.995 / ≥0.98 | ✗ |
| Z-MiniVal240 | **240/240** | 240/240 | ✓ |
| Z-PairedVal members | **40/40** | 40/40 | ✓ |
| Task 7A fixed24 | **21/24** · compact **7/8** · per class 5/4/6/6 | ≥22/24 · 8/8 · ≥5/6 | ✗ |
| Minimal pairs | **73/96** (M1 29/48, M2 16/16, M3 12/16, M4 16/16) | 96/96 | ✗ |
| Stress v1 | **0.8281** · macro F1 **0.6030** · min recall **0.0000** · L3 macro **0.8333** | ≥0.97 / ≥0.97 / ≥0.90 / ≥0.95 | ✗ |

Full-val per-class recall: the six classes with **zero** Task 7B training rows collapsed — `leftmost`
**0.0000**, `rightmost` 0.5000, `topmost` 0.8248, `bottommost` 0.8304, `smallest` 0.8514,
`largest_to_nearest` 0.8144 — while the 14 trained classes are 1.0000. Stress is language-balanced
(zh 83/96, en 76/96).

## 7. Scope safety and frozen end-to-end regression

All four section-21 out-of-scope controls parse to the listed canonical program (`largest_to_right_of`,
`largest_to_left_of`, `largest_to_nearest`, `largest_to_nearest`), exit **5** before
`proposals/reference/fields/sam2/z_b3`, and write only `result.json`; OOD controls keep their guard
behaviour; `keyword_or_regex_override = false`.

The frozen downstream reproduces Task 7A **exactly** with the Task 7B parser: parser 240/240, strict
`0.21700369907681483` (Δ 0.0), answered `0.21975058134361` (Δ 0.0), abstentions **3**, Paired **8/20**,
margin `0.19949275176250805` (Δ 0.0). Because the canonical gates failed, section 20 forbids moving the CLI
default: `default_l3_parser_checkpoint()` still returns the Task 6T parser, and the Task 7A helper is
untouched.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **1148 passed, 1 skipped** (Task 7A ended at 1107 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task7b_parser_hardening.py` adds the 41 section-L checks.

Not committed: the Task 7B parser checkpoint, the generated training JSONL
(`artifacts/task7b/parser_train_augmented/`), model/tokenizer caches, YOLO/SAM2/Z-B3 weights, source
imagery/vectors, `.conda`. Committed: small packs/spec/manifests/evaluation JSON, parser scripts, tests,
docs, handoff.

Task 7B downloaded nothing and installed nothing. Watt was **not needed** in Task 7B: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no
proxy, host, certificate or TLS setting was read or modified.

## 9. Interpretation boundary

DSH reports measurements only. The parser is not claimed as a project novelty; the downstream architecture
was not changed; reference hardening was not reopened; no attention/global competition was added; no full
training or test evaluation was started; Task 7C was not chosen; no repair is proposed.

## 10. Recommended next step (exact wording required by Part J)

等待 ChatGPT 根据 Task 7B 的 L3 ProgramHead 组合语义泛化结果决定下一步，不自行重新开启 reference hardening、修改 Z-B3 或进行 attention/global competition 改造。

## 11. STOP

Task 7B stops here: no downstream retraining, no reference re-hardening, no attention/global competition, no
full training, no test access, no GUI. Waiting for the ChatGPT audit.
