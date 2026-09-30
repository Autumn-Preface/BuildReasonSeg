# Task 7B — L3 ProgramHead Compositional-Semantic Hardening

> Task: `handoff/TO_DSH.md` (Task 7B) · Base commit: `6ee3d0d` · Predecessor: Task 7A →
> `L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`
> **Verdict: `L3_PARSER_CANONICAL_REGRESSION`** · baseline hash exact · leakage **zero** · checkpoint
> `2a285e4a…cf48eb4`
> Tests: `tests/test_task7b_parser_hardening.py` · Evidence: `evaluation/task7b_*.json`

Task 7B hardens the **same** Qwen3-VL-2B text-only 20-class ProgramHead for compositional L2/L3 semantic
contrast. It is parser-only: the Task 7A L3 pipeline, U-C1 reference resolver, fields, SAM2 and Z-B3 stay
frozen, and no keyword/regex remapping was added — the ProgramHead itself had to learn the distinction.

## 1. Recorded Task 7A findings (Task 7A artifacts not mutated)

Verdict `L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`. A0 oracle reproduction: Z-B3 mIoU
`0.3242128982543474`, Dice `0.4389840055529761`, Paired `15/20`, margin `0.30035408969722216`. A1
predicted-reference: strict mIoU `0.21700369907681483`, answered `0.21975058134361`, retention
`0.6693246944992726`, Paired `8/20`, margin `0.19949275176250805`, reference abstention `3/240`, dominant
bottleneck `REFERENCE`. Language: canonical Z-MiniVal240 `240/240`, paired members `40/40`, fixed L3
paraphrase `3/24`, required compact prompts `0/8`, dominant failure = terminal `to_nearest` dropped to L2.
Task 7B does **not** reinterpret Task 7A as a success.

## 2. Frozen baseline, architecture and evaluation material

Baseline = the authoritative Task 6T hardened parser, SHA256
`4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e` (**exact**). Architecture preserved
exactly: Qwen3-VL-2B, text-only (no image input), same tokenizer/prompt formatting, same Task 6M trainable
policy (LoRA adapters + ProgramHead, backbone frozen: 17,434,624 trainable / 2,144,421,888 total), exactly
20 canonical classes, no `unsupported` class.

Frozen before training (`evaluation/task7b_eval_prompt_manifest.json`):

| Source | Prompts |
|---|---|
| BuildSpatialReason v0.2 full val | 9,111 records (18,222 bilingual prompts) |
| Z-MiniVal240 queries | 240 |
| Z-PairedVal20 member queries | 40 |
| Task 7A fixed 24 L3 paraphrases | 24 |
| **Task 7B compositional minimal pairs** | **96** |
| **Task 7B L3 stress v1** | **192** |

**Minimal pairs (96)** — M1 L2-direction vs L3 direction+nearest **48**, M2 L1-largest vs L2-direction
**16**, M3 nearest-only vs direction+nearest **16**, M4 largest vs smallest in L2 **16**; bilingual, matched
semantic pairs, reserved lexical pool, no Task 7A fixed24 string.

**Stress v1 (192)** — 4 L3 classes × 24 = **96** (12 Chinese + 12 English each, short/medium/long phrasing,
nearest/closest and 最近/距离最近/最靠近 variants, word-order variants), 4 L2 direction classes × 12 = **48**,
`largest_to_nearest` **16**, `smallest_to_nearest` **8**, `largest` **8**, `smallest` **8**, extreme classes
**8** (=192).

## 3. Training data, overlap removal and leakage audit

Allowed sources only: v0.2 train instruction text + deterministic Task 7B train-only augmentation.

Section 8 removed **all 12,778 original train records**: BuildSpatialReason v0.2 emits the same 120
instruction templates in train and val (120 distinct normalized texts each, **100 % intersection**), so every
train prompt overlaps the evaluation set exactly. Cleaned v0.2 train rows = **0**.

Exactly **4,800** augmentations were generated with the section-9 allocation — each L3 class **600**
(300 zh + 300 en), each `largest_to_<direction>` **300**, `largest_to_nearest` **400**,
`smallest_to_nearest` **200**, each `smallest_to_<direction>` **150** — using only the section-10 TRAIN
lexical pools (evaluation forms reserved: `最大建筑左边最近…`, `closest building to the left…`, etc.).

`evaluation/task7b_parser_leakage_audit.json`: augmentation ↔ evaluation **exact overlap 0**, **normalized
overlap 0**, cleaned-train overlap 0/0 → **`LEAKAGE_FREE`**.

## 4. Training protocol and checkpoint

Internal split (seed 20261001, groups keyed by normalized prompt, class-stratified): 4,399 train / 401
holdout with **0** normalized overlap. Frozen protocol: AdamW, lr **2e-4**, weight decay **1e-4**,
effective batch **32** (microbatch 16 × accumulation 2), seed 20261001, max 8 epochs, early-stopping
patience 2, bfloat16 AMP, no sweep. Selection: internal-holdout macro F1 → exact accuracy → L3-four-class
macro recall.

Selected epoch **1** (holdout accuracy 1.0000, L3 macro recall 1.0000, macro F1 0.7000 — the macro average
covers all 20 classes while only 14 have support, because the six classes with no training rows also have no
holdout rows). Wall **417.9 s**, peak VRAM **6.922 GB**, checkpoint
`artifacts/checkpoints/task7b/program_parser_l3_hardened_v1.pt`, SHA256
`2a285e4ac91051cb2cfcea370d384fb50f29927832ad30d3baf09c672cf48eb4`.

## 5. Final frozen-parser evaluation

| Set | Result | Gate | Pass |
|---|---|---|---|
| Full v0.2 val (18,222 prompts) | **0.7129** accuracy, macro F1 **0.7939**, min class recall **0.0000** | ≥0.995 / ≥0.995 / every class ≥0.98 | ✗ |
| Z-MiniVal240 | **240/240** | 240/240 | ✓ |
| Z-PairedVal members | **40/40** | 40/40 | ✓ |
| Task 7A fixed24 | **21/24**, compact **7/8**, per class 5/4/6/6 | ≥22/24, 8/8 compact, every class ≥5/6 | ✗ |
| Minimal pairs | **73/96** (M1 29/48, M2 16/16, M3 12/16, M4 16/16) | 96/96 | ✗ |
| Stress v1 | **0.8281** accuracy, macro F1 **0.6030**, min class recall **0.0000**, L3 macro recall **0.8333** | ≥0.97 / ≥0.97 / ≥0.90 / ≥0.95 | ✗ |

Full-val per-class recall — the six classes with **zero** Task 7B training rows collapsed: `leftmost`
**0.0000**, `rightmost` 0.5000, `topmost` 0.8248, `bottommost` 0.8304, `smallest` 0.8514,
`largest_to_nearest` 0.8144; the 14 trained classes are 1.0000 except `smallest_to_nearest` 1.0000 and the
six above. Stress is language-balanced (Chinese 83/96, English 76/96), so the residual L3 contrast errors
are not a single-language artefact.

## 6. Scope safety and frozen end-to-end regression

`evaluation/task7b_scope_safety.json` — all four section-21 out-of-scope controls parse to the listed
canonical program (`largest_to_right_of`, `largest_to_left_of`, `largest_to_nearest`, `largest_to_nearest`),
exit **5** before `proposals/reference/fields/sam2/z_b3`, and write only `result.json`; OOD controls keep
their guard behaviour; `keyword_or_regex_override = false`. **No keyword nearest gate exists.**

`evaluation/task7b_end_to_end_regression.json` — with the Task 7B parser the frozen downstream reproduces
Task 7A **exactly**: parser 240/240, strict mIoU `0.21700369907681483` (Δ **0.0**), answered
`0.21975058134361` (Δ **0.0**), abstentions **3**, Paired **8/20**, margin `0.19949275176250805`
(Δ **0.0**).

Because the canonical gates failed, section 20 forbids moving the CLI default to the Task 7B checkpoint:
`default_l3_parser_checkpoint()` keeps returning the Task 6T parser, and the Task 7A helper is untouched.

## 7. Verdict

Section 24 priority: protocol clean, baseline hash exact, a valid checkpoint exists → the canonical
evaluation regressed (`full_val` accuracy 0.7129 < 0.995, macro F1 0.7939 < 0.995, `leftmost` recall 0.0
< 0.98) while Z-MiniVal240 and Z-PairedVal stayed perfect →
**`L3_PARSER_CANONICAL_REGRESSION`** (item 4, which precedes the compositional-robustness item).

Measured reading (reported only, no repair proposed): the section-8 overlap rule, applied literally to a
dataset whose train and val splits share all 120 instruction templates, left the fine-tune with **only** the
4,800 augmentations covering 14 of 20 classes; the three-epoch LoRA update therefore destroyed the six
never-trained canonical classes (`leftmost` recall 0.0) while still not solving the L3 contrast generally
(minimal pairs 73/96, stress L3 macro recall 0.8333). The canonical Z-MiniVal240 queries and the paired
members — the queries the Task 7A chain actually uses — remain perfect, and the frozen downstream chain
reproduces Task 7A to 1e-6, so no downstream regression was introduced.

## 8. Interpretation boundary

DSH reports measurements only. The parser is not claimed as a project novelty; the downstream architecture
was not changed; reference hardening was not reopened; no attention/global competition was added; no full
training or test evaluation was started; Task 7C was not chosen. Final recommendation exactly:

`等待 ChatGPT 根据 Task 7B 的 L3 ProgramHead 组合语义泛化结果决定下一步，不自行重新开启 reference hardening、修改 Z-B3 或进行 attention/global competition 改造。`

## 9. Reproduce

```text
python scripts/task7b_build_parser_data.py    # Part C-E packs, manifest, augmentation, leakage audit
python scripts/task7b_train_parser.py         # Part E training
python scripts/task7b_eval_parser.py          # Part F six evaluation sets
python scripts/task7b_scope_audit.py          # Part G scope safety / CLI regression
python scripts/task7b_e2e_regression.py       # Part H frozen downstream regression
python scripts/task7b_report.py               # Parts I gates + verdict
```

Generated training data lives in the gitignored `artifacts/task7b/parser_train_augmented/`; the Task 7B
checkpoint is gitignored too. Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the data,
scope and report steps also run in `.conda/buildreasonseg-mvp`.
