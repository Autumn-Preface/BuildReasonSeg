# TO_DSH — Task 6E: Explicit Spatial Token Grounding v0.1

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: replace the failed implicit `[SEG] hidden → geometry` readout with an **explicit, sample-specific spatial token target**.
>
> Accepted evidence before this task:
> - Oracle BOX → SAM2: strict mIoU **0.7506**, paired **20/20**.
> - Corrected Task 6D G0 still fails: geometry paired **0/20**, box IoU **0.0088**.
> - Frozen Task 6C `[SEG]` hidden is **practically not decodable for target geometry** under the audited readouts.
>
> Therefore Task 6E makes the box coordinates a first-class autoregressive language target instead of expecting a generic `[SEG]` hidden state to spontaneously contain location.
>
> This is a **capability-enabling architecture experiment**, not yet the project's final novelty claim.

## 0. User priorities

The user's project priorities remain:
1. correct model capability;
2. training / prediction / evaluation that can eventually run from CMD;
3. architecture that can support a paper;
4. UI/GUI only at the very end.

Do **not** build GUI/Web UI.
Do **not** reopen performance tuning in this task.

## 1. Fixed model/data/runtime

Use:
- Qwen3-VL-2B-Instruct;
- SAM2.1 Hiera Base+;
- text-only LoRA;
- existing `[SEG]`;
- strict deterministic mode;
- Task 6C paired `P` train subset (480 records / 240 images × 2 targets);
- fixed 120 validation records;
- fixed 20 paired validation images;
- Task 6C.7 accepted visual-feature cache;
- corrected multi-epoch scheduler logic from Task 6D.1.

Do not use:
- Qwen 4B;
- `[REF]`;
- Spatial Relation Encoder;
- Spatial Consistency Loss;
- new dataset;
- full training;
- true batch > 1;
- GUI.

WHU remains provisional, not permanent.

# PART A — Documentation hygiene before new experiments

## 2. Correct two stale/inconsistent statements

Do not alter historical numeric results.

### 2.1 Task 6D wording

Where current project-state text still says:

> "`[SEG]` hidden state does not carry the target's location"

replace/supersede it with:

> "Under the audited Task 6C training setup, the frozen `[SEG]` representation contains no **practically decodable** target geometry with the tested readouts. This is not an information-theoretic absence claim."

### 2.2 Decodability-summary boolean

`evaluation/task6d1_decodability_summary.json` currently contains:
`twenty_sample_overfit_is_diagnostic: true`

while the human report concludes that the 20-sample overfit is **not evidence of spatial decodability**, because shuffled labels can also be overfit.

Correct the field semantics/name/value so the machine artifact and human report agree.
Do not change measured probe numbers.

# PART B — Quantized box representation

## 3. Quantization candidates

Before changing the tokenizer, determine how many spatial bins are needed.

Test:
`B ∈ {32, 64, 128, 256}`

Use one shared location-token family:
`<loc_000> ... <loc_{B-1}>`

A box is four ordered tokens:
`<loc_x1> <loc_y1> <loc_x2> <loc_y2>`

Token position determines x1/y1/x2/y2. Do not create separate X/Y vocabularies unless a tokenizer correctness issue requires it.

## 4. Deterministic enclosing-box quantization

For normalized GT box `(x1,y1,x2,y2)` in `[0,1]`, use:

- min edges: `q1 = floor(c1 * (B - 1))`
- max edges: `q2 = ceil(c2 * (B - 1))`

Clip to `[0, B-1]`.

For a non-empty target, ensure dequantized width/height remain non-zero. If equality occurs, expand minimally and deterministically within bounds.

Dequantize with:
`c_hat = q / (B - 1)`.

The same quantizer must be used by target generation, oracle audit, parser/evaluator, and tests.

# PART C — Quantized-oracle ceiling

## 5. Measure quantization loss before training

Using the frozen Task 6D Oracle BOX path, replace continuous GT boxes with quantized/dequantized GT boxes for every B.

Evaluate on:
- same 120 validation records;
- same 20 paired validation images.

Report:
- quantized-box IoU vs continuous GT box;
- SAM strict mask mIoU;
- Dice;
- paired mask pass /20;
- own/cross margin.

Create:
`evaluation/task6e_quantized_oracle.json`

### Bin selection rule

Let unquantized oracle mIoU = `0.7506`.

Choose the **smallest B** satisfying both:
- paired mask = **20/20**;
- strict mIoU >= **0.7206** (`oracle - 0.03`).

If no candidate satisfies both:
`QUANTIZED_BOX_REPRESENTATION_INADEQUATE`
and STOP before tokenizer/model modification.

# PART D — Explicit spatial vocabulary

## 6. New target tokens

After B is selected, add exactly:
- `[BOX]`;
- B tokens `<loc_000>` ... `<loc_{B-1}>`;
- retain existing `[SEG]`.

Every new token must:
- encode to exactly one id;
- decode round-trip with `skip_special_tokens=False`;
- have a unique id;
- be available to free generation.

Target assistant sequence:

`{reasoning_zh} [BOX] <loc_x1> <loc_y1> <loc_x2> <loc_y2> [SEG]`

Freeze and test the exact whitespace/token convention.

There must be:
- exactly one `[BOX]`;
- exactly four loc tokens after it;
- exactly one `[SEG]`;
- `[SEG]` after the four loc tokens.

## 7. Generalize trainable-token support safely

Current code was built around one trainable `[SEG]` row.

Generalize to the set:
`{[SEG], [BOX], all selected <loc_*> tokens}`

Preferred:
- PEFT `trainable_token_indices=[...]` with all ids, if supported and verified.

Fallback:
- generalize the project-owned token-row adapter to multiple rows.

Requirements:
- base embedding table frozen;
- ordinary rows unchanged;
- tied input/output behavior valid;
- every new output row receives gradient and can be emitted;
- never unfreeze the full vocabulary.

One-step smoke must prove:
- `[BOX]` row changes;
- sampled loc rows change;
- `[SEG]` row changes where expected;
- at least 16 ordinary rows change exactly 0;
- visual tower has zero LoRA.

Create:
`evaluation/task6e_token_setup.json`

# PART E — Token-aware teacher forcing and loss

## 8. Build explicit token-position masks

Do not infer coordinate positions from decoded strings.

Batch construction must expose masks/positions for:
- reasoning tokens;
- `[BOX]`;
- four loc tokens;
- `[SEG]`;
- EOS.

Preserve the project's causal-label rule:
`logits at position i predict token i+1`.

Add tests for off-by-one correctness.

## 9. Training objective

Use:

`L_total = 1.0 * L_assistant + 5.0 * L_location`

where:
- `L_assistant` = normal assistant-span causal CE;
- `L_location` = causal CE only at the four location-token prediction positions.

Thus location tokens receive explicit additional supervision while normal sequence behavior is retained.

Do not add mask loss.
Do not add SmoothL1 on hidden states.
Do not sweep loss weights.

Record raw assistant CE, location CE, total.

# PART F — Stage E0: implementation sanity

## 10. E0 20-sample overfit

Use deterministic 20 records, including same-image/different-target pairs where possible.

Train:
- text LoRA;
- `[SEG]`;
- `[BOX]`;
- loc-token rows.

Freeze:
- Qwen base;
- Qwen visual tower;
- all SAM2;
- old Projection MLP;
- Task 6D SpatialGroundingHead.

Maximum 1,500 optimizer steps.
Use corrected scheduler horizon for the actual budget.

### Free-generation E0 gate

From image + instruction only:
- structural validity >= **19/20**;
- exact four-loc-token sequence >= **18/20**;
- mean dequantized box IoU >= **0.70**.

If unresolved after implementation audit:
`SPATIAL_TOKEN_IMPLEMENTATION_FAILED`.

# PART G — Stage E1: paired mini-train

## 11. E1 training

Only if E0 passes.

Train on 480 paired P records, maximum 3 epochs.

Trainables remain:
- text LoRA;
- `[SEG]`;
- `[BOX]`;
- loc rows.

SAM remains fully frozen.
No mask loss.

Use corrected scheduler over the actual total optimizer-step count.

Evaluate after each epoch.

Model selection priority:
1. geometry paired pass;
2. validation mean box IoU;
3. structural validity.

# PART H — Free-generation geometry evaluation

## 12. Parse token ids, not text

Free generation:

`image + instruction → reasoning → [BOX] → 4 loc tokens → [SEG]`

Parser must operate on token ids.

Valid sample requires:
- exactly one `[BOX]`;
- exactly four consecutive loc tokens in the required span;
- canonical non-empty dequantized box;
- exactly one `[SEG]` after the box.

Invalid structure = geometry failure.

Do not repair invalid output using GT or prose heuristics.

## 13. Geometry metrics

On 120 val + 20 paired:
- structural-valid rate;
- exact 4-token sequence accuracy;
- per-coordinate token accuracy;
- mean absolute bin error;
- normalized coordinate error;
- mean predicted-box IoU;
- center-inside-target rate;
- geometry paired /20;
- same-image predicted-box L1 distance;
- L1/L2/L3 breakdown;
- query-family breakdown.

Teacher-forced token accuracy is diagnostic only.
Free generation is primary.

### E1 geometry gate

Require:
- structural valid >= **90%**;
- geometry paired >= **14/20**;
- mean predicted-box IoU >= **0.35**.

If this fails:
`EXPLICIT_SPATIAL_TOKENS_FAILED`

Do not train SAM to hide the failure.

# PART I — Stage E2: end-to-end segmentation

## 14. E2 is inference-only

If E1 geometry gate passes:

`generated loc tokens → dequantized box → official frozen SAM2 box prompt → frozen current mask decoder → mask`

No GT geometry may enter the prompt.
No SAM weights may update.

## 15. E2 metrics

On fixed 120 val + 20 paired:
- strict e2e mIoU;
- Dice;
- conditional mIoU for valid spatial-token generations;
- mask paired /20;
- own-target IoU;
- cross-target IoU;
- own-minus-cross margin;
- IoU(pred_A,pred_B);
- L1/L2/L3 breakdown;
- query-family breakdown.

Invalid coordinate sequence => strict mask IoU 0.

Compare against:
- Task 6C P_C: mIoU `0.10604`, paired `0/20`;
- selected quantized-oracle ceiling;
- continuous Oracle BOX: mIoU `0.7506`, paired `20/20`.

# PART J — Verdicts

## 16. Use exactly one

### `EXPLICIT_SPATIAL_TOKENS_FIX_FOUND`
Require:
- E0 pass;
- E1 geometry gate pass;
- E2 mask paired >= **14/20**;
- strict e2e mIoU >= **0.20**;
- own-minus-cross margin > **0.05**;
- no GT leakage.

### `EXPLICIT_SPATIAL_TOKENS_PARTIAL`
Generated boxes become clearly instruction-conditioned, but full mask gate is not met.

### `EXPLICIT_SPATIAL_TOKENS_FAILED`
Generation is structurally valid but geometry does not generalize.

### `SPATIAL_TOKEN_IMPLEMENTATION_FAILED`
E0 cannot pass after correctness audit.

### `QUANTIZED_BOX_REPRESENTATION_INADEQUATE`
Quantization itself destroys the oracle pathway.

### `INVALID_EXPERIMENT`
Leakage, split mismatch, tokenizer/checkpoint corruption or other correctness failure.

# PART K — Interpretation rule

## 17. Coordinate tokens are not the final novelty claim

If they work, conclude only:

> We repaired the functional spatial-grounding bottleneck and established a usable `reasoning → explicit geometry → segmentation` pathway.

Do not claim coordinate tokens themselves as the final paper innovation.

Later novelty may still come from:
- geometry-verifiable BuildSpatialReason supervision;
- reference/relation mechanisms;
- Spatial Relation Encoder;
- Spatial Consistency Loss;
- relation-level evaluation.

# PART L — Data observation

## 18. Keep WHU, but classify errors

Do not change dataset.

Classify coordinate-token errors:
- tiny target / quantization sensitivity;
- border truncation;
- touching/merged pseudo-instance;
- visually ambiguous buildings;
- complex L3 relation;
- valid token sequence but wrong target.

If data quality dominates, flag it for ChatGPT.

# PART M — Reusable inference path

## 19. Reusable API only if E1 passes

If geometry works, create reusable core functions:

```python
generate_spatial_tokens(image, instruction)
parse_box_tokens(token_ids)
predict_box(image, instruction)
predict_mask_from_generated_box(image, instruction)
```

No GUI.

A polished top-level `predict.py` can wait for ChatGPT review.

# PART N — Required artifacts

## 20. Create

```text
evaluation/task6e_quantized_oracle.json
evaluation/task6e_token_setup.json
evaluation/task6e_e0_overfit.json
evaluation/task6e_e1_training.json
evaluation/task6e_geometry_eval.json
evaluation/task6e_segmentation_eval.json
evaluation/task6e_paired_probe.json
evaluation/task6e_error_analysis.json
evaluation/task6e_checkpoint_manifest.json
docs/task6e_explicit_spatial_tokens.md
```

`task6e_segmentation_eval.json` only if E1 gate passes.

Checkpoints/tokenizer local state stay gitignored; commit hashes/manifests only.

# PART O — Tests

## 21. Required tests

Cover at least:
1. enclosing quantizer deterministic;
2. quantizer/dequantizer canonical;
3. non-empty GT remains non-empty after quantization;
4. selected B obeys the oracle rule;
5. `[BOX]` single-token roundtrip;
6. every loc token single-token roundtrip;
7. token ids unique;
8. base embedding frozen;
9. all new token rows receive gradients;
10. ordinary rows unchanged after one step;
11. tied output side can emit all new tokens;
12. causal location-label positions off-by-one correct;
13. target format has one `[BOX]`, four loc tokens, one `[SEG]` in order;
14. parser uses token ids and rejects malformed output;
15. no GT geometry enters free generation or SAM prompt;
16. E0/E1 scheduler uses true total steps;
17. SAM weights stay frozen;
18. old SpatialGroundingHead unused;
19. no test split;
20. no 4B / `[REF]` / SRE / SCL;
21. strict determinism;
22. Task 6D.1 stale wording/boolean correction is consistent.

Run:
`python -m pytest tests/ -q`

# PART P — Git / Watt

## 22. Git hygiene

Do not stage:
- weights;
- checkpoints;
- local tokenizer/model cache snapshots;
- `.conda`;
- visual/SAM feature caches;
- dataset JSONL edits.

Recommended commit:
`feat: add explicit spatial grounding tokens`

Ignore UU completely.
Use established Watt ownership rules for final push only.

# PART Q — Handoff

## 23. FROM_DSH

Include:
1. Verdict
2. Frozen Evidence
3. Quantized Oracle by Bin Count
4. Selected Bin Count
5. Vocabulary / Trainable Token Setup
6. Target Sequence and Loss
7. E0 Overfit
8. E1 Training
9. Free-Generation Geometry
10. Paired Geometry Probe
11. E2 End-to-End Segmentation
12. Paired Mask Probe
13. L1/L2/L3 + Query Breakdown
14. Error / Data Adequacy Analysis
15. Reusable Inference Plumbing
16. Runtime / VRAM
17. Tests
18. Git / Watt
19. Recommended next architecture task

## 24. Final DSH UI — Chinese only

Report:
- verdict;
- unquantized oracle and each quantized-oracle result;
- selected B;
- added-token count and trainable-token mechanism;
- E0 exact-token / box-IoU;
- E1 structural validity;
- E1 val box IoU;
- geometry paired /20;
- if E2 ran: strict e2e mIoU, mask paired /20, own-cross margin;
- comparison to Task 6C P_C and Oracle BOX;
- whether same-image different instructions now emit different spatial tokens/boxes;
- dominant remaining error;
- whether WHU appears limiting;
- tests;
- commit/push;
- Watt handling.

# 25. STOP

After Task 6E:

**STOP.**

Do not automatically:
- add `[REF]`;
- add Spatial Relation Encoder;
- add Spatial Consistency Loss;
- scale to 4B;
- migrate dataset;
- run full training;
- build GUI.

Wait for ChatGPT review.
