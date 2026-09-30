# TO_DSH — Task 7A: L3 Predicted-Reference + Natural-Language Integration Audit

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `7006d7d416382f775112541a448425da1535cfd1`
>
> Predecessor: Task 6Z → `L3_COMPOSITION_NO_MEANINGFUL_GAIN`
>
> Research decision already made by ChatGPT:
>
> 1. Task 6Z is accepted as a **partial positive L3 composition result**, not a final success:
>    - Z-B3 is the best overall L3 decoder;
>    - it beats visual-only, nearest-only, deterministic product, and geometry-only controls;
>    - it achieves strong paired discrimination (15/20, margin ≈ +0.3004);
>    - but it misses the predeclared absolute mIoU bar and the incremental-nearest-over-direction bar.
> 2. Do NOT redesign the L3 decoder yet.
> 3. Do NOT add attention/global competition yet.
> 4. The next question is practical error propagation:
>
>    **How much of the oracle-reference Z-B3 L3 capability survives when the oracle largest reference is
>    replaced by the frozen practical U-C1 deterministic resolver, and when the hardened ProgramHead
>    supplies the natural-language program?**
>
> 5. Task 7A is an **integration / attribution task**. No model training is permitted.
> 6. DSH is an executor. Do not redesign any module or choose the next research direction.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- load all frozen checkpoints/modules;
- reproduce the oracle-reference Z-B3 result;
- run the exact predicted-reference and natural-language chains below;
- build a CMD inference entry point for the four L3 programs;
- perform failure attribution;
- solve ordinary runtime/integration bugs without changing algorithms or thresholds.

DSH MUST NOT:
- train or fine-tune any model;
- retrain ProgramHead;
- retrain YOLO;
- retrain Z-B3;
- alter GeometricRelationField v0.2;
- alter NearestBoundaryField v0.1;
- alter U-C1 proposal configuration;
- alter Task 6Q deterministic largest resolver;
- add ProposalSetRanker / ProposalQualityEstimator / SAM2 refinement;
- add attention / Transformer / GNN;
- add GRCL or another loss;
- change parser classes;
- use test split;
- tune any threshold after observing results;
- choose Task 7B.

If any prohibited change is required, STOP and report.

---

# PART A — Record Task 6Z status

## 1. Task 6Z measured result

Copy into `docs/task7a_l3_predicted_reference_integration.md`:

Task 6Z verdict:
`L3_COMPOSITION_NO_MEANINGFUL_GAIN`

Important measured evidence:

- parameter-free composition sanity:
  - direction-valid top-1 = 1.0000
  - top-3 = 1.0000
  - Spearman = 0.9983
- Z-B3 Overfit20:
  - mIoU = 0.9619
  - Dice = 0.9804
- MiniVal240:
  - Z-B0 visual = 0.1510
  - Z-B1 directional field = 0.3011
  - Z-B2 nearest field = 0.1724
  - Z-B3 learned two-field composition = **0.3242**
  - Z-B4 deterministic product = 0.2788
  - Z-B5 geometry-only = 0.0888
- Z-B3 deltas:
  - vs B0 = +0.1732
  - vs B1 = +0.0231
  - vs B2 = +0.1519
  - vs B4 = +0.0454
  - vs B5 = +0.2355
- paired:
  - Z-B3 = **15/20**
  - own-cross margin = **+0.300354**
- tests:
  - `1062 passed, 1 skipped`

Interpretation boundary:
- Z-B3 is the current best L3 research decoder;
- Task 7A does not convert the formal 6Z verdict into a success claim;
- Task 7A only measures practical reference/parser error propagation.

---

# PART B — Frozen assets

## 2. Freeze all prior artifacts

Read-only:

- all Task 6Z artifacts;
- exact Task 6Z Z-MiniVal240 and Z-PairedVal20 packs;
- Task 6Z Z-B3 checkpoint;
- Task 6T hardened ProgramHead checkpoint;
- Task 6M.1 YOLO26m-seg checkpoint;
- U-C1 proposal configuration from Task 6U;
- Task 6Q deterministic largest resolver semantics;
- GeometricRelationField v0.2;
- NearestBoundaryField v0.1;
- frozen SAM2.1 Hiera Base+ image feature path/cache;
- BuildSpatialReason v0.2;
- WHU-EA-NativeVector v1.0.

No test split.

## 3. Verify Z-B3 checkpoint

Read the selected Z-B3 checkpoint path and SHA256 from the frozen Task 6Z evaluation/training artifacts.

Require:
- local checkpoint exists;
- recomputed SHA256 matches exactly;
- variant metadata is Z-B3;
- architecture unchanged.

If unavailable/mismatch:
STOP with `L3_CHECKPOINT_UNAVAILABLE`.

## 4. Verify hardened ProgramHead

Use the frozen Task 6T hardened ProgramHead.

Read path/hash from Task 6T authoritative training/evaluation artifact.

Require:
- same Qwen3-VL-2B text-only parser;
- same 20 canonical program vocabulary;
- no image input;
- checkpoint hash exact.

If unavailable/mismatch:
STOP with `PARSER_CHECKPOINT_UNAVAILABLE`.

## 5. Practical reference resolver

Freeze exactly:

```text
YOLO26m-seg Task 6M.1 weights
imgsz = 640
conf = 0.05
max_det = 300
default NMS
no TTA
no tiling
```

Reference family for all Task 7A L3 programs is:
`largest`

Apply exact Task 6Q largest eligibility:
- not border touching;
- bbox extent ratio <= 0.20.

Select:
- maximum predicted mask area;
- tie break higher YOLO confidence;
- then lower original proposal index.

No learned ranker/filter/refinement.

---

# PART C — Four supported L3 programs

## 6. Exact scope

Support only:

- `largest_to_left_of_to_nearest`
- `largest_to_right_of_to_nearest`
- `largest_to_above_to_nearest`
- `largest_to_below_to_nearest`

Exact decomposition:

```text
largest_to_left_of_to_nearest  → family=largest, direction=left_of,  terminal=nearest
largest_to_right_of_to_nearest → family=largest, direction=right_of, terminal=nearest
largest_to_above_to_nearest    → family=largest, direction=above,    terminal=nearest
largest_to_below_to_nearest    → family=largest, direction=below,    terminal=nearest
```

No learned decomposition.

Any other canonical program is outside Task 7A scope.

---

# PART D — A0 oracle reproduction

## 7. Reproduce frozen Z-B3

Use exact Task 6Z Z-MiniVal240 and Z-PairedVal20.

For every record:
- canonical program id from frozen pack;
- oracle largest reference mask;
- frozen directional field v0.2;
- frozen nearest boundary field v0.1;
- frozen SAM2 feature;
- frozen Z-B3 checkpoint.

Require:

MiniVal:
- mIoU absolute delta <= `1e-6`
- Dice absolute delta <= `1e-6`

Paired:
- exactly `15/20`
- own-cross margin absolute delta <= `1e-6`

against Task 6Z frozen metrics.

Write:
`evaluation/task7a_oracle_reproduction.json`

If reproduction fails:
STOP with `TASK6Z_REPRODUCTION_FAIL`.

---

# PART E — A1 predicted-reference / canonical-program chain

## 8. Chain

Use canonical program ids from Z-MiniVal240.

Inference receives:
- source image;
- canonical program id;
- frozen checkpoints/configs.

It must NOT receive:
- oracle reference mask;
- GT target;
- source feature ids;
- GT candidate ids.

Pipeline:

```text
canonical L3 program
→ largest family + direction
→ U-C1 YOLO proposals
→ deterministic largest reference resolver
→ predicted reference mask
→ P_dir v0.2
→ P_near v0.1
→ frozen SAM2 image feature
→ frozen Z-B3
→ target mask
```

If reference resolver abstains:
- explicit abstention;
- strict target IoU = 0.

## 9. Reference diagnostics

GT reference is evaluation only.

Report on Z-MiniVal240:
- proposal count;
- eligible proposal count;
- reference abstentions;
- selected-reference mIoU;
- Dice;
- Pr@0.5;
- centroid error mean/median/p90;
- best eligible proposal coverage@0.50;
- failure buckets:
  1. `NO_PROPOSALS`
  2. `NO_ELIGIBLE_PROPOSALS`
  3. `REFERENCE_NOT_COVERED_IOU50`
  4. `REFERENCE_SELECTION_WRONG`
  5. `REFERENCE_GEOMETRY_POOR`
  6. `REFERENCE_OK`

Write:
`evaluation/task7a_predicted_reference_quality.json`

## 10. Target metrics

Report:
- strict all-240 mIoU/Dice;
- answered-only mIoU/Dice;
- Pr@0.5;
- abstention count/rate;
- per direction;
- reference-OK subset target mIoU;
- reference-fail subset target mIoU.

Write:
`evaluation/task7a_canonical_predicted_reference_val.json`

## 11. PairedVal20

Use exact Task 6Z Z-PairedVal20.

Because each pair has same tile and same largest reference id:
- run the predicted largest reference once per pair/tile and reuse it for both member programs;
- only direction changes.

Report:
- pass /20;
- mean own IoU;
- mean cross IoU;
- own-cross margin;
- reference-abstention pairs.

Write:
`evaluation/task7a_canonical_predicted_reference_paired.json`

---

# PART F — A2 hardened ProgramHead integration

## 12. Canonical-query parser audit

Run the Task 6T hardened ProgramHead on the actual natural-language queries stored in exact Z-MiniVal240.

Report:
- exact canonical program accuracy /240;
- per-class recall;
- confusion matrix;
- number predicted into non-Task7A canonical classes.

Write:
`evaluation/task7a_parser_l3_val.json`

## 13. Natural-language end-to-end chain

For every Z-MiniVal240 record, inference receives only:

- source image;
- natural-language query;
- frozen parser/proposal/SAM2/Z-B3 checkpoints/configs.

Pipeline:

```text
natural language
→ hardened ProgramHead
→ canonical program
→ Task 7A scope check
→ predicted largest reference
→ P_dir + P_near
→ frozen Z-B3
→ target mask
```

Rules:
- parser wrong relative to expected program → strict target IoU = 0;
- parser outputs valid but out-of-scope program → status `unsupported_l3_program`, no downstream, strict IoU=0;
- reference abstention → strict IoU=0.

Report:
- parser accuracy;
- strict all-240 target mIoU/Dice;
- answered-only target mIoU/Dice;
- Pr@0.5;
- abstentions;
- parser failures;
- per direction.

Write:
`evaluation/task7a_natural_language_val.json`

## 14. Natural-language PairedVal20

Run each pair member's natural-language query separately.

Report:
- parser-correct members /40;
- pass /20;
- own IoU;
- cross IoU;
- margin;
- parser-error pairs;
- reference-abstention pairs.

Write:
`evaluation/task7a_natural_language_paired.json`

---

# PART G — Fixed L3 paraphrase audit

## 15. Freeze exactly 24 prompts

Use 6 prompts per L3 class:
- 3 Chinese
- 3 English

Create before running the parser:
`evaluation/task7a_l3_paraphrase_pack.json`

The pack must include these exact known compact/contrast forms among the 24:

### right + nearest
- `分割面积最大的建筑物右侧最近的建筑物。`
- `segment the building nearest to the right of the largest building`

### left + nearest
- `找出最大建筑左边最近的建筑。`
- `find the closest building to the left of the largest building`

### above + nearest
- `找出最大建筑上方最近的建筑。`
- `find the nearest building above the largest building`

### below + nearest
- `找出最大建筑下方最近的建筑。`
- `find the closest building below the largest building`

The other 16 prompts must be semantically unambiguous paraphrases of the same four programs.

Do not train on them.

## 16. Parser-only paraphrase result

Report:
- total exact accuracy /24;
- per program;
- Chinese / English;
- each failure with predicted canonical id.

Write:
`evaluation/task7a_l3_paraphrase_result.json`

No parser retraining in Task 7A.

---

# PART H — CMD inference entry point

## 17. Create CLI

Create:

`predict_buildreasonseg_l3.py`

Required example:

```cmd
python predict_buildreasonseg_l3.py ^
  --image path\to\image.tif ^
  --prompt "分割面积最大的建筑物右侧最近的建筑物。" ^
  --parser-checkpoint <Task6T hardened checkpoint> ^
  --proposal-checkpoint artifacts\checkpoints\task6m1\runs\m1_yolo26m_seg_continued\weights\best.pt ^
  --target-checkpoint <Task6Z Z-B3 checkpoint> ^
  --out-dir outputs\buildreasonseg_l3
```

No GT/annotation arguments.

## 18. CLI outputs

Successful inference writes:
- `result.json`
- `reference_mask.png`
- `direction_field.png`
- `nearest_field.png`
- `target_mask.png`
- `overlay.png`

`result.json` must include:
- status;
- prompt;
- parsed_program;
- direction;
- reference_family=`largest`;
- parser checkpoint hash;
- proposal checkpoint hash;
- Z-B3 checkpoint hash;
- proposal count;
- eligible proposal count;
- selected reference index/conf/area/bbox;
- reference abstention reason if any;
- directional field min/max/mean;
- nearest field min/max/mean;
- target positive-pixel count;
- runtime breakdown;
- `ground_truth_used=false`.

Valid but non-Task7A program:
- exit code `5`;
- stop before proposal/SAM2/Z-B3.

OOD behavior:
preserve existing Task 6S/6T domain-guard behavior where applicable.

---

# PART I — Failure attribution

## 19. Exclusive bucket per Z-MiniVal240 natural-language record

Assign exact priority:

1. `PARSER_WRONG`
2. `REFERENCE_NO_PROPOSALS`
3. `REFERENCE_NO_ELIGIBLE`
4. `REFERENCE_NOT_COVERED_IOU50`
5. `REFERENCE_SELECTION_WRONG`
6. `REFERENCE_GEOMETRY_POOR`
   - selected reference IoU >= 0.50 but normalized centroid error > 0.05.
7. `TARGET_FAIL_WITH_REFERENCE_OK`
   - reference IoU >=0.50 and centroid error <=0.05;
   - target IoU <0.50.
8. `TARGET_OK`
   - adequate reference and target IoU >=0.50.

Report:
- counts / percentages;
- by direction;
- parser_fail;
- reference_fail;
- target_fail.

Dominant bottleneck:
- if parser_fail / 240 > 0.05 → `PARSER`
- else if reference_fail > target_fail → `REFERENCE`
- else → `L3_TARGET_DECODER`

Tie reference_fail == target_fail → `REFERENCE`.

Write:
`evaluation/task7a_failure_attribution.json`

DSH must not propose a repair.

---

# PART J — Predeclared gates

## 20. Canonical predicted-reference retention

Define:

```text
oracle_ZB3_mIoU = frozen Task6Z Z-B3 mIoU
retention = A1_strict_mIoU / oracle_ZB3_mIoU
```

Canonical predicted-reference chain is considered usable if ALL:

- strict mIoU >= `0.22`
- answered-only mIoU >= `0.24`
- retention >= `0.68`
- PairedVal >= `10/20`
- own-cross margin >= `0.15`
- reference abstention rate <= `0.10`

## 21. Natural-language integration

Natural-language integration is considered usable if ALL:

- canonical-query parser accuracy >= `0.98`
- strict end-to-end mIoU >= `0.21`
- answered-only mIoU >= `0.23`
- PairedVal >= `9/20`
- own-cross margin >= `0.14`

## 22. Paraphrase robustness flag

`l3_paraphrase_ready = true` iff:
- fixed24 accuracy >= `22/24`
- every one of the 8 exact compact prompts listed in section 15 is correct.

This flag is NOT required for the canonical predicted-reference gate, but it affects the final verdict priority below.

Do not change thresholds.

---

# PART K — Verdict

## 23. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `L3_CHECKPOINT_UNAVAILABLE`
3. `PARSER_CHECKPOINT_UNAVAILABLE`
4. `TASK6Z_REPRODUCTION_FAIL`
5. `L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE`
   - canonical predicted-reference section-20 gate fails.
6. `L3_LANGUAGE_HARDENING_REQUIRED`
   - canonical predicted-reference gate passes,
   - but section-21 natural-language gate fails OR `l3_paraphrase_ready=false`.
7. `L3_END_TO_END_DEVELOPMENT_CHAIN_READY`
   - canonical predicted-reference gate passes,
   - natural-language section-21 gate passes,
   - `l3_paraphrase_ready=true`.

No other verdict.

`READY` means development-chain ready, NOT paper/test/final-model ready.

---

# PART L — Interpretation boundary

DSH may report measurements only.

Do NOT:
- turn Task 6Z into a novelty/success claim;
- retrain parser because paraphrases fail;
- reopen reference hardening;
- add global attention;
- change Z-B3;
- change fields;
- start full training or test evaluation.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 7A 的 L3 predicted-reference、ProgramHead 与 failure attribution 结果决定下一步，不自行进行 parser 再训练、reference 再硬化、attention/global competition 或正式全量训练。`

---

# PART M — Required artifacts

Create at minimum:

```text
predict_buildreasonseg_l3.py
buildreasonseg_mvp/task7a_l3_pipeline.py

evaluation/task7a_oracle_reproduction.json
evaluation/task7a_predicted_reference_quality.json
evaluation/task7a_canonical_predicted_reference_val.json
evaluation/task7a_canonical_predicted_reference_paired.json
evaluation/task7a_parser_l3_val.json
evaluation/task7a_natural_language_val.json
evaluation/task7a_natural_language_paired.json
evaluation/task7a_l3_paraphrase_pack.json
evaluation/task7a_l3_paraphrase_result.json
evaluation/task7a_failure_attribution.json
evaluation/task7a_verdict.json

docs/task7a_l3_predicted_reference_integration.md

scripts/task7a_evaluate_reference.py
scripts/task7a_evaluate_pipeline.py
scripts/task7a_parser_audit.py
scripts/task7a_cli_audit.py
scripts/task7a_failure_attribution.py
scripts/task7a_report.py
```

Generated CLI outputs/caches:
`artifacts/task7a/`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No new model checkpoint.

---

# PART N — Tests

Task 6Z ended at:
`1062 passed, 1 skipped`

Add tests for at least:

1. Task 6Z artifacts unchanged
2. Z-B3 checkpoint hash exact
3. hardened ProgramHead checkpoint hash exact
4. parser remains text-only
5. same 20-class vocabulary
6. exactly four Task7A L3 supported programs
7. exact program decomposition
8. U-C1 config exact
9. Task 6Q largest eligibility exact
10. deterministic largest selection exact
11. no ProposalSetRanker
12. no ProposalQualityEstimator
13. no SAM2 proposal refinement
14. oracle reproduction exact tolerance
15. A1 uses canonical program ids
16. A1 has no parser
17. A1 has no oracle reference in inference
18. natural-language A2 uses hardened ProgramHead
19. parser wrong scores zero in strict aggregate
20. out-of-scope program stops before downstream
21. same predicted reference reused within paired same-reference pair
22. directional field v0.2 unchanged
23. nearest field v0.1 unchanged
24. Z-B3 unchanged
25. frozen SAM2 visual path unchanged
26. no training
27. no GT target inference
28. no GT reference inference
29. GT only offline reference/target scoring
30. paraphrase pack frozen before parser evaluation
31. exactly 24 paraphrases
32. 4 programs × 6 prompts
33. bilingual paraphrase coverage
34. exact 8 compact prompts present
35. CLI has no annotation/GT argument
36. CLI writes required successful outputs
37. result.json ground_truth_used=false
38. failure buckets exclusive
39. dominant bottleneck rule exact
40. no attention/Transformer/GNN
41. no GRCL
42. no test split
43. no new dataset/download/install/GUI
44. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART O — Git/storage

Do not commit:
- parser/YOLO/SAM2/Z-B3 weights;
- feature/proposal caches;
- source imagery/vectors;
- generated overlays/masks;
- `.conda`.

Commit:
- integration code;
- small evaluation JSON;
- tests;
- scripts;
- docs;
- handoff.

Suggested commits:
1. `feat: integrate predicted-reference L3 chain`
2. `eval: audit L3 language and reference error propagation`
3. optional docs/handoff commit

---

# PART P — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for a genuine cross-module/runtime bug.

No installs or downloads.

---

# PART Q — STOP

After Task 7A:
- commit;
- push;
- update handoff;
- STOP.

Do NOT:
- retrain parser;
- reopen reference hardening;
- add attention/global competition;
- change fields/decoder;
- start formal full training;
- use test;
- build GUI.

Wait for ChatGPT audit.
