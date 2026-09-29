# TO_DSH — Task 6S: Directional End-to-End Integration + Architecture Hardening Checkpoint

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `a3d59da8c8698f793357887762dbc95a08a254aa`
>
> Predecessor: Task 6R → `GRCL_NO_MEANINGFUL_RELATION_GAIN`
>
> Research decision already made by ChatGPT:
> 1. Keep **GeometricRelationField v0.2**.
> 2. Use frozen **Task 6O B3** as the target decoder.
> 3. Use frozen **Task 6Q proposal reference resolver** as current support infrastructure.
> 4. **Do NOT use GRCL v0.1 in the primary chain.** Preserve Task 6R only as a negative ablation.
> 5. Integrate the frozen Qwen3-VL-2B ProgramHead to obtain the first natural-language directional end-to-end chain.
> 6. Task 6S is a mandatory architecture-hardening checkpoint. It does NOT start nearest/L3/full training.

DSH is an executor. Do not redesign any module or choose the next research direction.

---

## 0. DSH role and STOP rule

All user-facing DSH output must be Chinese.

DSH MAY:
- integrate the exact frozen modules listed below;
- implement glue code and a clean CMD inference entry point;
- run the exact validation/CLI/failure-attribution audits below;
- fix ordinary integration/runtime bugs that do not change algorithms or thresholds.

DSH MUST NOT:
- retrain ProgramHead;
- retrain YOLO;
- retrain B3;
- train R1/GRCL;
- change GeometricRelationField v0.2;
- modify proposal-reference eligibility/ranking;
- change prompt/parser model family;
- add `[REF]`;
- add GRCL/SCL to the primary chain;
- add nearest;
- add L3;
- add a new dataset;
- use test split;
- tune inference thresholds;
- choose the hardening solution after attribution;
- start Task 6T.

If an unexpected issue requires any prohibited change, STOP and report it.

---

# PART A — Freeze the architecture being integrated

## 1. Primary Task 6S chain

The exact primary chain is:

```text
natural-language instruction
        ↓
Task 6M/6J frozen Qwen3-VL-2B ProgramHead
        ↓
canonical program id
        ↓
program decomposition:
reference_family ∈ {largest, smallest}
relation ∈ {left_of, right_of, above, below}
        ↓
Task 6Q frozen YOLO26m-seg proposal reference resolver
        ↓
predicted reference mask
        ↓
GeometricRelationField v0.2
        ↓
P_rel
        ↓
frozen SAM2.1 Hiera Base+ visual feature
        +
P_rel
        +
relation embedding
        ↓
Task 6O frozen B3 target decoder
        ↓
target mask
```

No GRCL in this chain.

No oracle reference mask in inference.

No deterministic final-target proposal selection.

The proposal system is used **only to ground the reference building**. The final target is produced by the dense B3 visual decoder.

## 2. Why GRCL v0.1 is excluded

Record in the Task 6S design doc:

Task 6R found:
- R0/B3 MiniVal mIoU = `0.4299680351479113`
- R1/B3+GRCL mIoU = `0.4118569...` (use exact stored value)
- R0 hard relation accuracy = `0.95`
- R1 relation accuracy ≈ `0.954167`
- PairedVal: R0 `14/20`, R1 `12/20`
- proposal-reference transfer: frozen 6Q B3 chain mIoU `0.3045812554881724`, R1 transfer ≈ `0.283782`; paired `10/20 → 2/20`

Therefore Task 6S uses B3, not R1.

Do not delete or modify Task 6R artifacts.

## 3. Task 6R gate erratum to record, not mutate

Record in the Task 6S design doc:

The predeclared Task 6R criterion
`R1 relation_accuracy >= R0 relation_accuracy + 0.08`
was ill-posed because the frozen baseline measured `R0 = 0.95`, leaving only 0.05 headroom before the metric ceiling 1.0.

This was a **ChatGPT experiment-design error**, not a DSH implementation error.

Do NOT edit frozen Task 6R artifacts or verdict.

The independent measurements still support excluding GRCL v0.1 because:
- relation gain was only about +0.0042;
- mIoU decreased;
- PairedVal decreased;
- proposal-reference transfer degraded substantially.

---

# PART B — Frozen assets and integrity verification

## 4. Read-only assets

Treat as frozen/read-only:

- all Task 6M / 6M.1 artifacts;
- all Task 6N / 6O / 6P / 6Q / 6R artifacts;
- BuildSpatialReason v0.2;
- WHU native-vector v1.0;
- Task 3B relation config;
- GeometricRelationField v0.2;
- frozen SAM2 feature path/cache;
- frozen Task 6Q resolver code/config;
- frozen Task 6O B3 checkpoint;
- frozen ProgramHead checkpoint.

No test split.

## 5. Frozen ProgramHead

Use the existing text-only Qwen3-VL-2B ProgramHead used by Task 6M structured CLI.

Default local path:
`artifacts/checkpoints/task6j/j2_best.pt`

Expected historical SHA256:
`eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3`

Before any evaluation:

1. require the checkpoint exists;
2. recompute SHA256;
3. require exact match;
4. verify the image is NOT an input to ProgramHead;
5. verify ProgramHead output vocabulary remains the frozen 20 canonical program ids;
6. do not retrain.

If checkpoint missing or hash mismatched:
STOP with `PARSER_CHECKPOINT_UNAVAILABLE`.

Write:
`evaluation/task6s_frozen_asset_audit.json`

## 6. Frozen proposal reference resolver

Use exactly Task 6Q:

- checkpoint SHA256:
  `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`
- YOLO26m-seg
- imgsz = 640
- conf = 0.10
- max_det = 100
- default NMS
- no TTA
- no tiling
- largest/smallest eligibility/ranking rules unchanged.

Do not tune.

## 7. Frozen B3 target decoder

Read B3 checkpoint path and SHA256 from the frozen Task 6O evaluation artifact.

Require:
- local checkpoint exists;
- hash exact.

Do not retrain.

## 8. Frozen relation field

Use:
`buildreasonseg_mvp/geometric_relation_field_v02.py`

Do not modify formula/constants:
- alpha = 1.2
- tau = 0.04
- s_axis = 0.02
- s_margin = 0.02

---

# PART C — Supported language scope

## 9. Exactly eight supported programs

Task 6S supports only:

- `largest_to_left_of`
- `largest_to_right_of`
- `largest_to_above`
- `largest_to_below`
- `smallest_to_left_of`
- `smallest_to_right_of`
- `smallest_to_above`
- `smallest_to_below`

After ProgramHead classification:

- if program is one of these 8 → continue;
- if ProgramHead returns another valid canonical program (L1, nearest, L3, etc.) → explicit:
  - status = `unsupported_directional_program`
  - exit code = `5`
  - do not run proposal/SAM2/B3;
- if the deterministic domain guard rejects the prompt → preserve Task 6M.1 behavior:
  - status = `unsupported_instruction`
  - exit code = `4`
  - no parser/proposal/SAM2/B3.

Do NOT silently remap unsupported programs.

## 10. Program decomposition

Exact mapping:

```text
largest_to_left_of   → family=largest, relation=left_of
largest_to_right_of  → family=largest, relation=right_of
largest_to_above     → family=largest, relation=above
largest_to_below     → family=largest, relation=below

smallest_to_left_of  → family=smallest, relation=left_of
smallest_to_right_of → family=smallest, relation=right_of
smallest_to_above    → family=smallest, relation=above
smallest_to_below    → family=smallest, relation=below
```

No learned decomposition.

---

# PART D — Clean end-to-end CLI

## 11. Create CLI

Create:

`predict_buildreasonseg_directional.py`

Required example:

```cmd
python predict_buildreasonseg_directional.py ^
  --image path\to\image.tif ^
  --prompt "分割面积最大的建筑物右侧的建筑物" ^
  --parser-checkpoint artifacts\checkpoints\task6j\j2_best.pt ^
  --proposal-checkpoint artifacts\checkpoints\task6m1\runs\m1_yolo26m_seg_continued\weights\best.pt ^
  --target-checkpoint <Task6O_B3_checkpoint> ^
  --out-dir outputs\buildreasonseg_directional
```

The CLI must work from:
- image;
- natural-language prompt;
- model checkpoints/configs.

It must NOT require:
- GT masks;
- annotation JSON;
- BuildSpatialReason record;
- source feature ids;
- oracle reference;
- target id.

## 12. CLI outputs

For a successful supported prompt, write:

```text
result.json
target_mask.png
overlay.png
reference_mask.png
relation_field.png
```

`result.json` must include at least:

- status
- prompt
- parsed_program
- reference_family
- relation
- domain_gate result
- parser checkpoint hash
- proposal checkpoint hash
- B3 checkpoint hash
- proposal count
- eligible reference proposal count
- selected reference proposal index/confidence/area/bbox
- explicit reference abstention status/reason
- relation-field min/max/mean
- target mask positive-pixel count
- runtime breakdown:
  - parser
  - proposal/reference
  - SAM2 feature
  - field
  - target decoder
  - total
- `ground_truth_used = false`

If reference resolver abstains:
- status = `reference_abstention`
- do not fabricate a target mask;
- JSON must state reason.

## 13. On-the-fly SAM2

The CLI must be able to compute the frozen SAM2 image feature directly from the supplied image.

It may use the existing cache during batch evaluation for speed, but the CLI audit must prove that a source image not addressed through a precomputed Task 6N record/cache key can execute through the normal frozen SAM2 encoder path.

Do not download a new SAM2 model.

---

# PART E — Canonical validation integration audit

## 14. Reuse exact frozen validation packs

Use byte-for-byte:

- Task 6N MiniVal240
- Task 6N PairedVal20

Do not regenerate.

No test split.

Each MiniVal240 record already has:
- natural-language query;
- expected canonical program;
- GT reference/target for evaluation only.

The actual inference chain receives only:
- image
- natural-language query
- frozen checkpoints/configs.

Expected program and GT are held outside the inference path.

## 15. Parser audit on MiniVal240

Run ProgramHead on each record's actual natural-language query.

Report:
- exact program accuracy / 240
- confusion matrix over 8 programs
- reference-family accuracy
- relation accuracy
- number classified into unsupported canonical programs.

Write:
`evaluation/task6s_parser_val.json`

## 16. End-to-end MiniVal240

For every record:

```text
query
→ ProgramHead
→ program decomposition
→ proposal reference resolver
→ field v0.2
→ B3
→ target mask
```

If parser gives wrong/unsupported program or reference abstains, score target IoU as **0** in the strict end-to-end aggregate.

Report BOTH:
- strict all-240 mIoU/Dice;
- answered-only mIoU/Dice;
- Pr@0.5;
- abstention rate;
- per program;
- per direction;
- largest/smallest family;
- border target;
- tiny target if present.

Compare to frozen Task 6Q proposal-reference+B3 baseline:
- answered target mIoU = `0.3045812554881724`
- paired = `10/20`
- own-cross margin = `0.27370032940000916`

Because Task 6Q used oracle program ids, Task 6S measures the incremental cost of natural-language parsing/integration.

Write:
`evaluation/task6s_end_to_end_val.json`

## 17. PairedVal20

Use natural-language query separately for each pair member.

Report:
- parser-correct members / 40
- pass / 20
- mean own IoU
- mean cross IoU
- own-cross margin
- reference abstention pairs
- parser-error pairs.

Write:
`evaluation/task6s_end_to_end_paired_val.json`

---

# PART F — Fixed paraphrase/CLI prompt pack

## 18. Exact 24-prompt supported pack

Evaluate these exact prompts. Expected program is fixed.

### largest_to_left_of
1. `分割面积最大的建筑物左侧的建筑物。`
2. `找出最大建筑左边的建筑物。`
3. `segment the building to the left of the largest building`

### largest_to_right_of
4. `分割面积最大的建筑物右侧的建筑物。`
5. `找出最大建筑右边的建筑物。`
6. `segment the building to the right of the largest building`

### largest_to_above
7. `分割面积最大的建筑物上方的建筑物。`
8. `找出最大建筑上面的建筑物。`
9. `segment the building above the largest building`

### largest_to_below
10. `分割面积最大的建筑物下方的建筑物。`
11. `找出最大建筑下面的建筑物。`
12. `segment the building below the largest building`

### smallest_to_left_of
13. `分割面积最小的建筑物左侧的建筑物。`
14. `找出最小建筑左边的建筑物。`
15. `segment the building to the left of the smallest building`

### smallest_to_right_of
16. `分割面积最小的建筑物右侧的建筑物。`
17. `找出最小建筑右边的建筑物。`
18. `segment the building to the right of the smallest building`

### smallest_to_above
19. `分割面积最小的建筑物上方的建筑物。`
20. `找出最小建筑上面的建筑物。`
21. `segment the building above the smallest building`

### smallest_to_below
22. `分割面积最小的建筑物下方的建筑物。`
23. `找出最小建筑下面的建筑物。`
24. `segment the building below the smallest building`

Use validation images deterministically selected from the corresponding frozen MiniVal240 program groups.

For each prompt:
- expected program exact;
- parser result;
- supported/unsupported;
- CLI reaches correct downstream branch;
- GT unavailable to the CLI.

No retraining if a paraphrase fails.

## 19. Unsupported controls

Run at least:

OOD / domain-gate rejection:
- `Write a poem about the sea.`
- `今天天气怎么样？`
- `检测道路。`
- empty string

In-domain but out-of-6S-scope canonical tasks:
- `分割面积最大的建筑物。`
- `分割最左侧的建筑物。`
- `分割面积最大的建筑物右侧最近的建筑物。`
- `segment the building nearest to the right of the largest building`

Expected:
- OOD → exit 4 before ProgramHead/downstream.
- Valid but unsupported directional scope → parser may classify normally, then exit 5 before proposal/SAM2/B3.

Write:
`evaluation/task6s_cli_prompt_audit.json`

---

# PART G — Failure attribution: mandatory architecture hardening checkpoint

## 20. One exclusive bucket per MiniVal240 record

Assign in this exact priority order:

1. `PARSER_WRONG`
   - parsed canonical program != expected program, including parsed unsupported program.

2. `REFERENCE_NO_PROPOSALS`
   - parser correct, resolver reports no proposals.

3. `REFERENCE_NO_ELIGIBLE`
   - parser correct, proposals exist but no eligible reference proposal.

4. `REFERENCE_NOT_COVERED_IOU50`
   - parser correct; eligible proposals exist; best eligible proposal vs GT reference IoU < 0.50.

5. `REFERENCE_SELECTION_WRONG`
   - best eligible ref proposal IoU >= 0.50, but selected ref IoU < 0.50.

6. `REFERENCE_GEOMETRY_POOR`
   - selected ref IoU >= 0.50 but selected-reference normalized centroid error > 0.05.

7. `TARGET_FAIL_WITH_REFERENCE_OK`
   - selected reference passes:
     - ref IoU >= 0.50
     - centroid error <= 0.05
   - but predicted target IoU < 0.50.

8. `TARGET_OK`
   - selected reference adequate and target IoU >= 0.50.

GT is allowed ONLY in this offline attribution evaluator, never in inference.

Report:
- counts and percentages;
- by largest/smallest;
- by direction;
- by border/non-border target;
- tiny-target count if present.

Write:
`evaluation/task6s_failure_attribution.json`

## 21. Dominant-bottleneck label

DSH must compute, not interpret beyond this fixed rule:

```text
parser_fail =
    PARSER_WRONG

reference_fail =
    REFERENCE_NO_PROPOSALS
  + REFERENCE_NO_ELIGIBLE
  + REFERENCE_NOT_COVERED_IOU50
  + REFERENCE_SELECTION_WRONG
  + REFERENCE_GEOMETRY_POOR

target_fail =
    TARGET_FAIL_WITH_REFERENCE_OK
```

Dominant bottleneck:
- if parser_fail / 240 > `0.05` → `PARSER`
- else if reference_fail > target_fail → `REFERENCE`
- else → `TARGET_DECODER_FIELD`

Tie `reference_fail == target_fail` → `REFERENCE`

This label is for ChatGPT's next research decision only.

DSH MUST NOT choose how to repair the dominant bottleneck.

---

# PART H — Predeclared integration gates

## 22. Directional full-chain gate

Pass only if ALL:

1. MiniVal240 parser exact-program accuracy >= `0.95`
2. strict all-240 end-to-end mIoU >= `0.28`
3. answered-only end-to-end mIoU >= `0.2893521927137638`
   - 95% of frozen Task 6Q `0.3045812554881724`
4. PairedVal >= `9/20`
5. own-cross margin >= `0.22`
6. supported 24-prompt paraphrase pack parser accuracy >= `22/24`
7. all OOD controls follow exit-code-4 behavior
8. all valid-but-out-of-scope controls exit 5 before proposal/SAM2/B3
9. no GT/annotation dependency in CLI
10. no test split.

Do not change these thresholds.

## 23. Regression sanity

If parser accuracy on the canonical MiniVal240 is 1.0, then the downstream answered-only metrics should reproduce Task 6Q within numerical/integration tolerance except for records affected only by CLI execution mechanics.

If:
- same expected program,
- same image,
- same frozen resolver,
- same B3,
- same field,

but target output differs materially from Task 6Q, classify as integration regression and investigate only implementation differences.

Do not tune model thresholds.

---

# PART I — Verdict

## 24. Exactly one, priority order

1. `INVALID_EXPERIMENT`
   - leakage, test access, frozen mutation, GT inference dependency, protocol violation.

2. `PARSER_CHECKPOINT_UNAVAILABLE`

3. `END_TO_END_INTEGRATION_REGRESSION`
   - frozen modules receive semantically identical inputs but output differs materially from Task 6Q.

4. `DIRECTIONAL_PARSER_HARDENING_REQUIRED`
   - canonical parser accuracy < 0.95 OR paraphrase accuracy < 22/24.

5. `DIRECTIONAL_CHAIN_BELOW_GATE`
   - parser gates pass, but any numeric end-to-end/paired gate fails.

6. `DIRECTIONAL_END_TO_END_CHAIN_READY_FOR_HARDENING`
   - all section 22 gates pass.

No other verdict.

`READY_FOR_HARDENING` does NOT mean final model/paper ready.

---

# PART J — Architecture-freeze decision artifact

## 25. Create hardening checkpoint summary

Write:
`evaluation/task6s_hardening_checkpoint.json`

Must include:

```text
primary_chain:
  parser: frozen Qwen3-VL-2B ProgramHead
  reference: frozen Task6Q proposal resolver
  relation_field: v0.2
  target_decoder: frozen Task6O B3
  grcl_primary: false

proven_positive_modules:
  - GeometricRelationField v0.2 + dense visual target decoder

negative_or_non-primary_evidence:
  - Task6P dense ReferenceMaskHead
  - Task6R GRCL v0.1

known_technical_debt:
  - reference proposal coverage / extreme ranking, especially smallest
  - tiny buildings
  - directional-only current scope
  - no nearest
  - no L3 multi-hop
  - no cross-dataset generalization

dominant_bottleneck:
  one of PARSER / REFERENCE / TARGET_DECODER_FIELD

next_research_decision:
  "WAIT_FOR_CHATGPT"
```

Do not prescribe the implementation solution.

---

# PART K — Required files

## 26. Required artifacts

Create at minimum:

```text
predict_buildreasonseg_directional.py

buildreasonseg_mvp/task6s_directional_pipeline.py

evaluation/task6s_frozen_asset_audit.json
evaluation/task6s_parser_val.json
evaluation/task6s_end_to_end_val.json
evaluation/task6s_end_to_end_paired_val.json
evaluation/task6s_cli_prompt_audit.json
evaluation/task6s_failure_attribution.json
evaluation/task6s_hardening_checkpoint.json
evaluation/task6s_verdict.json

docs/task6s_directional_end_to_end_integration.md

scripts/task6s_evaluate.py
scripts/task6s_cli_audit.py
scripts/task6s_failure_attribution.py
scripts/task6s_report.py
```

Use gitignored:
`artifacts/task6s/`

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

---

# PART L — Tests

## 27. Required tests

At minimum:

1. Task 6R artifacts unchanged
2. Task 6Q artifacts unchanged
3. ProgramHead checkpoint SHA exact
4. ProgramHead receives text only
5. ProgramHead not retrained
6. exactly 8 supported directional programs
7. exact program→family/relation mapping
8. OOD exit 4 before parser/downstream
9. out-of-scope canonical program exit 5 before proposal/SAM2/B3
10. proposal checkpoint SHA exact
11. proposal conf 0.10
12. proposal imgsz 640
13. proposal max_det 100
14. Task 6Q eligibility/ranking unchanged
15. GeometricRelationField v0.2 unchanged
16. B3 checkpoint SHA exact
17. B3 not retrained
18. primary chain does not instantiate/use GRCL
19. R1 checkpoint not used
20. no oracle reference in inference
21. no GT target in inference
22. no BuildSpatialReason record required by CLI
23. SAM2 can run on source image on-the-fly
24. MiniVal240 byte-for-byte reused
25. PairedVal20 byte-for-byte reused
26. no test split
27. strict errors score zero in all-240 aggregate
28. same reference reuse in paired same-reference case
29. exactly 24 fixed supported paraphrases audited
30. exact 8 unsupported controls audited
31. result JSON contains `ground_truth_used=false`
32. successful CLI writes target mask
33. successful CLI writes reference mask
34. successful CLI writes relation field
35. failure attribution bucket is exclusive
36. dominant bottleneck rule exact
37. no GRCL primary-chain training
38. no nearest/L3
39. no `[REF]`
40. no 4B
41. no new dataset/download/install/GUI
42. previous suite preserved.

Run:

`python -m pytest tests/ -q`

Task 6R ended at:
`733 passed, 1 skipped`

Do not reduce previous passing tests.

---

# PART M — Git/storage

## 28. Do not commit

- parser checkpoint
- proposal checkpoint
- B3 checkpoint
- SAM2 weights
- proposal/feature caches
- source imagery/vectors
- `.conda`
- generated masks/overlays
- large caches

Commit:
- integration code
- small JSON evaluation artifacts
- tests
- docs
- handoff.

Recommended commits:

1. `feat: integrate directional BuildReasonSeg inference chain`
2. `eval: audit directional end-to-end failure cascade`
3. optional `docs:` handoff commit

---

# PART N — Model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use:
- **V4.1 Flash + Max**

only for genuine cross-module/runtime integration bugs.

Do not use V4 Pro by default.

---

# PART O — STOP

After Task 6S:

- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- fix the dominant bottleneck;
- retrain reference;
- retrain parser;
- add GRCL;
- add nearest;
- add L3;
- start full-dataset training;
- access test;
- build GUI.

Wait for ChatGPT to audit the first full directional chain and choose the hardening task.
