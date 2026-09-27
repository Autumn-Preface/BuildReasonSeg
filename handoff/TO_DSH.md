# TO_DSH — Task 6J: Structured Proposal Grounding Feasibility v0.1

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: stop trying to make one global query directly localize pixels and test a more structured route that matches the project's original spatial-reasoning idea:
>
> **instruction → relation program → building candidates → explicit geometry execution → selected building mask**
>
> This task is a feasibility / causal audit, not the final paper method.
>
> Accepted evidence before this task:
>
> - frozen SAM2 point/box prompts are strong when target geometry is known;
> - direct pixel-grounding attempts 6D–6I repeatedly fail to generalize, although Task 6I improves 10-pair memorization to inside 13/20 and paired point 5/10;
> - Task 6I final heatmap already places much more probability on the correct region (own mass ~0.417 vs cross ~0.024), so language semantics are not absent, but converting them into a reliable single pixel is still unstable;
> - the project already owns an audited spatial relation engine and a frozen YOLOv8m-seg building baseline.
>
> Task 6J therefore decomposes the problem:
>
> 1. Can the existing relation semantics select the correct target from **oracle building candidates**?
> 2. Can the frozen YOLO baseline provide a sufficiently complete **inference-time candidate set**?
> 3. Can Qwen parse the natural-language instruction into the correct **canonical relation program**?
> 4. If all three work, what is the end-to-end result using predicted program + predicted candidates?
>
> No new pixel-grounding head. No 4B. No `[REF]`. No SRE. No SCL. No new dataset. No GUI.

## 0. User-facing language

All DSH narrative/UI output must be **Chinese**.

Code, paths, raw logs, enum names and metric keys may remain English.

# PART A — Strategic freeze

## 1. Freeze Tasks 6D–6I as evidence

Do not rerun their training.

Preserve:
- Task 6C P_C: `0.10604 / paired 0/20`;
- Point Oracle: about `0.488 / 18/20`;
- Box Oracle: `0.7506 / 20/20`;
- Task 6I I0:
  - inside `13/20`;
  - paired point `5/10`;
  - bounded ranking `10/10`;
  - normalized error `0.0838`.

Do not claim Task 6I proved attention has zero target information. Its target attention mass rose above initialization, but was insufficient.

Task 6J is a deliberate architecture-path pivot, not a deletion of prior work.

# PART B — Canonical relation programs

## 2. Build a canonical program vocabulary from actual `query_type`

Do not invent new semantics.

Map existing BuildSpatialReason v0.1.1 query types to deterministic canonical programs, preserving the frozen Task 3B relation convention.

Use the actual existing query types in the dataset/artifacts.

Create:

`evaluation/task6j_program_spec.json`

For each query type record:
- canonical program id;
- ordered operations;
- required reference role(s);
- expected output role;
- sample counts by split.

No test split used for training.

# PART C — J0: Oracle program + oracle candidates

## 3. Oracle candidate set

Use frozen component maps / pseudo-instance representation.

For each fixed validation sample:
- all visible building components are candidates;
- geometry comes only from the component map:
  - mask;
  - centroid;
  - bbox;
  - area;
  - border flag only for diagnostics.

Diagnostic only; not inference.

## 4. Independent program executor

Implement a compact executor consuming:

```text
canonical program + candidate geometry
```

and returning one candidate id.

Reuse frozen Task 3B semantics/thresholds.

The executor must not read:
- target component id;
- GT reasoning;
- target mask except for post-selection scoring.

## 5. J0 gate

Run on:
- fixed 120 validation records;
- fixed 20 paired validation images;
- optionally full v0.1.1 val as secondary consistency audit if cheap.

Report:
- exact target candidate accuracy;
- paired target-selection /20;
- L1/L2/L3;
- query-type breakdown;
- abstentions/ambiguities.

Gate:

```text
exact target accuracy >= 0.98
paired selection >= 19/20
```

If J0 fails:
`STRUCTURED_EXECUTOR_SEMANTICS_MISMATCH`
and STOP.

# PART D — J1: Oracle program + predicted YOLO proposals

## 6. Existing YOLO baseline is read-only

Only if J0 passes.

Use the already frozen YOLOv8m-seg-WHU baseline from the legacy project.

Rules:
- legacy repo/directory read-only;
- weights read-only;
- verify expected baseline provenance/hash before inference;
- do not retrain YOLO;
- do not modify the old Ultralytics fork;
- do not install packages into existing environments.

If the existing YOLO inference environment is available, it may be invoked **read-only** with its existing Python / `conda run`.

Do not create or modify another environment solely for J1.

If model/environment cannot be safely resolved:
`YOLO_BASELINE_INFERENCE_UNAVAILABLE`
and stop J1/J4 without installing anything.

## 7. Proposal material

Run frozen YOLO segmentation inference on unique images needed for:
- fixed 120 validation records;
- fixed 20 paired validation images.

Cache predicted metadata/masks only under gitignored artifacts.

For each proposal record:
- mask;
- bbox;
- centroid;
- area;
- confidence.

Do not use GT to alter proposals.

## 8. Proposal recall audit

For each GT target candidate, measure best predicted-proposal IoU.

Report:
- recall @ mask IoU 0.25 / 0.50 / 0.75;
- mean/median best IoU;
- proposal-count distribution;
- missing-target rate;
- duplicate/merged proposal diagnostics;
- border/tiny breakdown.

Also report all-building candidate recall per image if practical.

## 9. Oracle-program execution on YOLO proposals

Execute same canonical program over predicted proposal geometry.

No GT target id may influence execution.

Score selected predicted mask against GT target mask.

Report:
- strict target mask mIoU;
- Dice;
- paired target selection /20 using own-vs-cross mask IoU;
- selected proposal best-match identity;
- missing-proposal vs wrong-relation failures.

J1 viability gate:

```text
target proposal recall@0.50 >= 0.75
oracle-program selected-mask mIoU >= 0.30
paired mask selection >= 12/20
```

If proposal recall fails, do not blame the language model.

# PART E — J2: Instruction → canonical program

## 10. Program parser objective

This branch answers only:

> What spatial program does the instruction request?

It does not localize pixels and sees no GT geometry.

Use Qwen3-VL-2B as language backbone, but make this branch **instruction-semantic first**.

Preferred input:
- instruction text only;
- normal tokenizer/chat formatting;
- no image tokens.

This deliberately avoids the image-dominated `[BOX]` problem.

## 11. ProgramHead

Use a minimal classifier over a clearly documented text representation, e.g. final instruction-token / assistant-prefix representation:

```text
Qwen text representation
→ LayerNorm
→ Linear(hidden_dim, num_programs)
```

Train:
- text-only LoRA;
- ProgramHead.

Freeze:
- Qwen base;
- Qwen visual tower;
- all SAM2;
- all prior grounding heads.

Do not use free-form program generation in Task 6J.

## 12. Training subset

Use deterministic query-type-stratified train-only data.

Preferred:
- up to ~100 records per canonical program;
- total capped around 1,500–2,000 records;
- no test split.

If a program has fewer records, use all and report imbalance.

Do not train all 15,592 unless the capped subset is clearly insufficient and documented.

## 13. J2 evaluation

Evaluate on:
- fixed 120 validation records;
- fixed 20 paired validation images;
- optionally full val classification if cheap.

Report:
- exact program accuracy;
- macro F1;
- confusion matrix;
- L1/L2/L3 accuracy;
- query-type accuracy;
- paired program correctness /20.

J2 gate:

```text
program accuracy >= 0.90
macro F1 >= 0.85
paired program correctness >= 18/20
```

If J2 fails:
`PROGRAM_PARSER_NOT_READY`.

Do not compensate with GT program later.

# PART F — J3: Predicted program + oracle candidates

## 14. Parser ceiling

Only if J2 passes.

Pipeline:

```text
instruction
→ predicted canonical program
→ GT/oracle candidate set
→ program executor
→ selected candidate
```

Oracle candidates remain diagnostic only.

Report:
- exact selected-target accuracy;
- paired selection /20;
- program-error vs executor-error attribution.

Gate:

```text
selected-target accuracy >= 0.85
paired >= 17/20
```

# PART G — J4: Predicted program + predicted YOLO candidates

## 15. First structured inference-time end-to-end test

Only if:
- J1 viability gate passes;
- J2 passes;
- J3 passes.

Pipeline:

```text
image → frozen YOLO building proposals
instruction → Qwen ProgramHead → canonical program
program + proposal geometry → explicit relation executor
→ selected proposal mask
```

No GT geometry/mask/id enters inference.

No SAM2 training.

Optional diagnostic:
- derive predicted point/bbox from selected predicted proposal;
- feed it to frozen SAM2;
- report proposal mask and SAM-refined mask separately.

Do not silently replace one with the other.

## 16. J4 metrics

On fixed 120 val + 20 paired:
- strict selected-mask mIoU;
- Dice;
- paired mask selection /20;
- own-target vs cross-target IoU;
- L1/L2/L3;
- query-type breakdown.

Failure attribution:
- wrong predicted program;
- target absent from proposal set;
- proposal geometry changes relation outcome;
- correct selected proposal but poor mask quality.

J4 success gate:

```text
strict mIoU >= 0.20
paired mask selection >= 12/20
```

This is a feasibility gate, not final paper performance.

# PART H — Verdicts

## 17. Use one primary verdict

### `STRUCTURED_PROPOSAL_GROUNDING_PROMISING`
J0/J1/J2/J3 pass and J4 reaches success gate.

### `PROPOSAL_QUALITY_LIMIT`
J0/J2/J3 healthy but proposal recall or J1/J4 is the binding failure.

### `PROGRAM_PARSER_NOT_READY`
J0 passes but J2 misses semantic parser gate.

### `STRUCTURED_EXECUTOR_SEMANTICS_MISMATCH`
J0 fails with oracle program + oracle candidates.

### `YOLO_BASELINE_INFERENCE_UNAVAILABLE`
Legacy model/env cannot be safely used read-only.

### `STRUCTURED_ROUTE_PARTIAL`
Some major gates pass and route materially improves target selection but J4 misses full gate.

### `INVALID_EXPERIMENT`
Leakage, split misuse, GT used in inference, provenance mismatch or correctness failure.

# PART I — Interpretation discipline

## 18. Do not claim final novelty

If route works, freeze only:

> Explicit program parsing + proposal-level geometry execution is a viable functional target-selection architecture for current BuildSpatialReason tasks.

Do not yet claim:
- deterministic relation execution is final innovation;
- YOLO proposals are final segmentation backbone;
- current templates prove open-vocabulary reasoning.

Later work may replace/augment executor with:
- learned Spatial Relation Encoder;
- `[REF]`-style reference grounding;
- relation-level Spatial Consistency Loss;
- richer natural-language paraphrases.

# PART J — Dataset/proposal adequacy

## 19. Separate failure sources

Report:
- merged/touching proposal failures;
- tiny-target misses;
- border truncation;
- pseudo-instance ambiguity;
- relation instability from missing/merged proposals.

If parser/executor works but proposal/data quality dominates, recommend a proposal-backbone/dataset task next.

Do not migrate data automatically.

# PART K — Reusable API

## 20. If J4 runs

Create/refactor:

```python
parse_program(instruction)
extract_building_candidates(image)
execute_program(program, candidates)
predict_structured_mask(image, instruction)
```

No GUI.

# PART L — Required artifacts

## 21. Create as applicable

```text
evaluation/task6j_program_spec.json
evaluation/task6j_j0_oracle_executor.json
evaluation/task6j_yolo_proposal_recall.json
evaluation/task6j_j1_oracle_program_yolo.json
evaluation/task6j_parser_setup.json
evaluation/task6j_j2_program_parser.json
evaluation/task6j_j3_predicted_program_oracle_candidates.json
evaluation/task6j_j4_structured_end_to_end.json
evaluation/task6j_error_attribution.json
evaluation/task6j_checkpoint_manifest.json
docs/task6j_structured_proposal_grounding.md
```

Large YOLO prediction caches stay local/gitignored.

# PART M — Tests

## 22. Required tests

Cover at least:

1. program vocabulary comes from actual frozen query types;
2. executor uses frozen relation convention;
3. oracle executor cannot read target id;
4. oracle candidates are diagnostic-only;
5. J0 exact reproduction on synthetic/unit cases;
6. legacy YOLO path/hash provenance checked;
7. legacy project/env never modified;
8. no package installation into existing environments;
9. proposal geometry computed without GT;
10. proposal recall matching is evaluation-only;
11. ProgramHead input contains instruction text only;
12. ProgramHead has no image tokens;
13. query_type appears only as CE target, never inference input;
14. train/val/test split hygiene;
15. J3 never falls back to GT program;
16. J4 uses predicted program + predicted proposals only;
17. optional SAM refinement uses predicted proposal geometry only;
18. no old direct-pixel grounding head in J4;
19. no 4B / `[REF]` / SRE / SCL;
20. strict determinism where supported;
21. no GUI;
22. failure attribution auditable.

Run:
`python -m pytest tests/ -q`

If YOLO inference runs through a separate existing conda env, record command/version output and keep that env read-only.

# PART N — Git / Watt

## 23. Git hygiene

Do not stage:
- YOLO weights;
- YOLO prediction caches;
- checkpoints;
- `.conda`;
- legacy files;
- dataset JSONL edits.

Recommended commit:
`feat: audit structured proposal grounding`

Use established Watt ownership rules for final push only.

# PART O — Handoff

## 24. FROM_DSH

Include:

1. Verdict
2. Strategic Pivot
3. Program Vocabulary
4. J0 Oracle Executor
5. YOLO Baseline Provenance
6. Proposal Recall
7. J1 Oracle Program + YOLO
8. ProgramHead Architecture
9. J2 Program Parsing
10. J3 Predicted Program + Oracle Candidates
11. J4 Structured End-to-End
12. Optional SAM Refinement
13. L1/L2/L3 + Query Breakdown
14. Failure Attribution
15. Dataset / Proposal Adequacy
16. Reusable Core API
17. Runtime / VRAM
18. Tests
19. Git / Watt
20. Recommended next architecture decision

## 25. Final DSH UI — Chinese only

Report:
- primary verdict;
- J0 oracle target accuracy / paired;
- YOLO target recall@0.5 and mean best IoU;
- J1 oracle-program+YOLO mIoU / paired;
- J2 program accuracy / macro F1 / paired;
- J3 selected-target accuracy / paired;
- if J4 ran: strict mIoU / paired;
- largest remaining error source: parser, proposal, executor or mask quality;
- whether route is materially better than direct pixel grounding;
- whether WHU/proposal quality is now a practical limitation;
- tests;
- commit/push;
- Watt handling.

# 26. STOP

After Task 6J STOP.

Do not automatically:
- add learned SRE;
- add `[REF]`;
- add Spatial Consistency Loss;
- retrain YOLO;
- scale to 4B;
- migrate dataset;
- run full training;
- build GUI.

Wait for ChatGPT review.
