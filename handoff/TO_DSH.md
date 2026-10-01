# TO_DSH — Task 7J: Final Frozen-Architecture Test Evaluation

> Status: ACTIVE — FINAL TEST AUTHORIZED FOR TASK 7J ONLY
>
> Repository: `BuildReasonSeg`
>
> Base commit: `cdd6a6801b8b1437847c22a6f24e4c33442b8df0`
>
> Predecessor: Task 7I → `DB1_FORMAL_VAL_NOT_CONFIRMED`

## 0. ChatGPT audit decision

1. Accept Task 7I as protocol-clean formal train/validation evidence.
2. Preserve the formal verdict exactly: `DB1_FORMAL_VAL_NOT_CONFIRMED`.
3. The only failed predeclared gate was practical predicted-reference gain:
   D-B1 strict mean `0.233519` vs Z-B3 `0.226984`, delta `+0.006535 < +0.02`.
4. Preserve the decoder-level result separately:
   oracle-reference D-B1 mean `0.383424` vs Z-B3 `0.332443`, delta `+0.050982`, D-B1 wins `3/3` matched seeds.
5. Therefore:
   - D-B1 is formally supported as the stronger target decoder **when reference is correct**;
   - practical end-to-end advantage is **not confirmed** because frozen reference selection/coverage dilutes the gain.
6. No more architecture/reference/parser tuning is authorized.
7. ChatGPT now authorizes ONE final test evaluation because model architecture, training protocol, seeds,
   checkpoint selection and reference policy are frozen.
8. This test is measurement only, not rescue/model selection.
9. Historical Task 6M test access must be disclosed; use only the wording
   `final frozen-architecture test evaluation`, never `untouched test`.
10. No training in Task 7J. Evaluate all six Task 7I `best.pt` checkpoints. After results, no model changes.

All user-facing DSH output must be Chinese.

---

# 1. Frozen Task 7I facts

Record in `docs/task7j_final_test.md`:

Oracle validation:
- Z-B3 seeds: `0.319367 / 0.339382 / 0.338579`, mean `0.332443 ± 0.011331`
- D-B1 seeds: `0.379934 / 0.382651 / 0.387688`, mean `0.383424 ± 0.003934`
- matched D-B1−Z-B3: `+0.060567 / +0.043269 / +0.049109`, wins `3/3`

Practical predicted-reference validation:
- Z-B3 strict mean `0.226984`
- D-B1 strict mean `0.233519`
- delta `+0.006535`

Reference on 936 val:
- mIoU `0.446949`
- REFERENCE_OK `491`
- SELECTION_WRONG `305`
- NOT_COVERED `135`
- abstentions `5`
- coverage@0.50 `0.850427`

Counterfactual validation:
- oracle pair pass: Z-B3 `0.869121`, D-B1 `0.844581`
- oracle margin: Z-B3 `+0.316684`, D-B1 `+0.377485`
- predicted pair pass: Z-B3 `0.515337`, D-B1 `0.417178`
- predicted margin: Z-B3 `+0.195162`, D-B1 `+0.200931`

Do NOT claim D-B1 universally improves pair pass rate.

---

# 2. One-time test authorization

Before reading test, create `evaluation/task7j_test_authorization.json` with:

```text
authorized_by = ChatGPT audit after Task 7I
scope = Task 7J final frozen-architecture evaluation only
training_allowed = false
checkpoint_selection_allowed = false
architecture_change_allowed = false
test_access_authorized = true
historical_test_access_disclosed = true
permitted_test_description = final frozen-architecture test evaluation
```

Verify Task 7I test lock was LOCKED and all six formal `best.pt` checkpoints exist.

---

# 3. Exact six checkpoints

Evaluate exactly:

```text
Z-B3 / 20261001
Z-B3 / 20261002
Z-B3 / 20261003
D-B1 / 20261001
D-B1 / 20261002
D-B1 / 20261003
```

Use only:
`artifacts/checkpoints/task7i/<model>/<seed>/best.pt`

Read authoritative hashes/bytes/epochs from:
- `evaluation/task7i_training_zb3.json`
- `evaluation/task7i_training_db1.json`

Create `evaluation/task7j_checkpoint_manifest.json`.

If any mismatch/missing:
STOP `FORMAL_CHECKPOINT_MISMATCH`.

No retraining; never use `last.pt`.

---

# 4. Final L3 test population

After authorization, read BuildSpatialReason v0.2 TEST and select ALL valid records of exactly:

- `largest_to_left_of_to_nearest`
- `largest_to_right_of_to_nearest`
- `largest_to_above_to_nearest`
- `largest_to_below_to_nearest`

No filtering by model output, target size, reference quality or success.

Freeze before inference:
`evaluation/task7j_test_population_manifest.json`

Record:
- total and per-program counts
- unique tiles
- sample-id SHA256
- dataset/version/split identity
- historical test-access disclosure
- no post-inference exclusion.

Large local rows:
`artifacts/task7j/packs/final_l3_test.jsonl` (gitignored).

---

# 5. Final test counterfactual pairs

From the frozen final L3 test population, generate ALL unique unordered pairs:

- same tile
- same reference_source_feature_id
- different direction
- different target_source_feature_id

Pair key:
`min(sample_id_a,sample_id_b) + "||" + max(...)`

Deduplicate, sort lexicographically, no subsampling.

Create:
`evaluation/task7j_test_pair_manifest.json`

Record pair count, direction-pair counts, unique tiles and pair-id SHA256.

Reporting only.

---

# 6. Frozen common components

- Frozen SAM2.1 Hiera Base+; test feature caching now allowed; no gradient.
- Exact GeometricRelationField v0.2.
- Exact NearestBoundaryField v0.1.
- Canonical dataset program ids/directions; ProgramHead is NOT inserted into Task 7J.
- GT target is metric label only.

No architecture/field/threshold changes.

---

# 7. Oracle-reference final test

For every test record:

```text
image + canonical program/direction + GT largest reference
→ frozen fields
→ frozen selected Z-B3 or D-B1 checkpoint
→ target mask
```

For every one of six checkpoints report:

Mask:
- mIoU
- Dice
- Pr@0.5

Per relation:
- left/right/above/below

Stratification:
- target area quartiles
- boundary-distance quartiles

All frozen counterfactual pairs:
- pair pass count/rate
- mean own IoU
- mean cross IoU
- own-cross margin

Efficiency:
- trainable parameter count
- total inference wall time
- ms/record
- peak inference VRAM if available without changing execution.

Create:
`evaluation/task7j_oracle_test_results.json`

Aggregate each model over all 3 seeds with mean and sample std (`ddof=1`), plus matched-seed D-B1−Z-B3 deltas.
Do not select a seed.

---

# 8. Practical predicted-reference final test

Reference is computed once and reused across all six runs:

```text
U-C1 YOLO26m-seg
imgsz 640
conf 0.05
max_det 300
default NMS
no TTA
no tiling

largest eligibility = exact frozen Task 6Q/6U policy
selector = max predicted mask area
tie higher confidence
then lower original index
```

No Task 7G selector, ranker, quality filter or SAM refinement.

Reference metrics once:
- mIoU
- Dice
- Pr@0.5
- abstentions/rate
- NO_PROPOSALS
- NO_ELIGIBLE
- NOT_COVERED
- SELECTION_WRONG
- GEOMETRY_POOR
- REFERENCE_OK
- best eligible coverage@0.50

For each six checkpoints:
- strict mIoU/Dice/Pr@0.5
- answered-only mIoU/Dice
- REFERENCE_OK-subset target mIoU
- per direction
- target area quartiles
- reference quality bins
- boundary-distance quartiles
- all test pair pass/own/cross/margin

For a pair, resolve reference once and reuse for both members.

Create:
`evaluation/task7j_predicted_reference_test_results.json`

---

# 9. Final comparison

Create:
`evaluation/task7j_final_comparison.json`

Include:

Oracle:
- Z-B3 mean±std
- D-B1 mean±std
- matched seed deltas
- pair pass/margin

Practical:
- Z-B3 mean±std
- D-B1 mean±std
- matched seed deltas
- pair pass/margin
- oracle→predicted retention

Reference:
- all reference metrics/buckets

Validation→test shift:
- Task 7I val mean
- Task 7J test mean
- test−val delta
for each model/reference mode.

This is descriptive only.

---

# 10. No performance pass/fail gate

Task 7J is a final measurement stage, not model selection.

No result may trigger:
- retraining
- threshold tuning
- architecture change
- seed selection
- reference changes
- a second test run with changed settings.

Safe interpretation examples:
- If oracle D-B1 remains better:
  `D-B1 retains a target-decoder advantage under correct reference on the final frozen-architecture test.`
- If practical gain is small/negative:
  `The decoder-level gain does not translate proportionally to the practical chain because reference selection/coverage remains limiting.`

Never claim:
- unrestricted natural language solved
- unseen-city generalization
- reference solved
- untouched test
- novelty/first from test results.

---

# 11. Test consumption lock

After evaluation create:
`evaluation/task7j_test_consumption_status.json`

Required:

```text
status = FINAL_TEST_CONSUMED
test_execution_authorized = false
task7j_final_test_completed = true
architecture_changes_after_test_authorized = false
retest_for_model_selection_authorized = false
```

Test may later be reproduced/reported for the same frozen setup, but not used for iterative development.

---

# 12. Verdict

Exactly one procedural verdict:

1. `INVALID_EXPERIMENT`
2. `FORMAL_CHECKPOINT_MISMATCH`
3. `FINAL_TEST_POPULATION_INVALID`
4. `FINAL_TEST_INCOMPLETE`
5. `FINAL_FROZEN_TEST_COMPLETE`

No performance-based PASS/FAIL verdict.

---

# 13. Required artifacts

Create:

```text
evaluation/task7j_test_authorization.json
evaluation/task7j_checkpoint_manifest.json
evaluation/task7j_test_population_manifest.json
evaluation/task7j_test_pair_manifest.json
evaluation/task7j_oracle_test_results.json
evaluation/task7j_predicted_reference_test_results.json
evaluation/task7j_final_comparison.json
evaluation/task7j_test_consumption_status.json
evaluation/task7j_verdict.json

docs/task7j_final_test.md

scripts/task7j_freeze_test_population.py
scripts/task7j_evaluate_oracle_test.py
scripts/task7j_evaluate_predicted_test.py
scripts/task7j_compare.py
scripts/task7j_report.py
```

Local caches only:
`artifacts/task7j/`

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No checkpoint creation. No CLI change.

---

# 14. Required tests

Task 7I ended at `1467 passed, 1 skipped`.

Add checks for at least:

1. Task 7I artifacts unchanged
2. Task 7I verdict preserved
3. one-time ChatGPT authorization
4. historical test access disclosure
5. untouched-test wording forbidden
6. all six best checkpoint hashes exact
7. no `last.pt`
8. no training code path
9. exact four programs
10. no output-based filtering
11. all valid test records used
12. test manifest frozen before evaluation
13. all pairs used, no subsampling
14. pair same tile/reference
15. pair different direction/target
16. canonical program only; ProgramHead absent
17. SAM2 frozen
18. fields frozen
19. Z-B3/D-B1 frozen
20. U-C1 exact
21. deterministic selector exact
22. Task 7G selector absent
23. reference shared across all six runs
24. pair reference reused
25. GT reference only in declared oracle mode
26. GT target label only
27. all three seeds for both models
28. no best-seed selection
29. `ddof=1`
30. oracle metrics complete
31. predicted metrics complete
32. reference buckets complete
33. per-direction metrics complete
34. stratification metrics complete
35. efficiency metrics present
36. val→test shift reported
37. no performance architecture gate
38. test consumption lock final
39. no post-test architecture changes
40. no parser/reference/YOLO training
41. no new loss/module/threshold
42. no new dataset/download/install/GUI
43. previous suite preserved.

Run:
`python -m pytest tests/ -q`

Do not reduce prior passing tests.

---

# 15. Git/storage

Do not commit:
- weights/checkpoints
- feature/proposal caches
- imagery/vectors
- expanded local test rows
- `.conda`

Commit:
- small manifests/results
- scripts/tests/docs/handoff.

Suggested commits:
1. `eval: authorize and freeze final L3 test population`
2. `eval: run final frozen Z-B3 and D-B1 test`
3. `docs: record final frozen-architecture test`

---

# 16. DSH model policy

Default: **DeepSeek V4.1 Flash + High**.
Use Max only for genuine evaluation/runtime/cross-file bugs.
No installs/downloads.

---

# 17. STOP

After Task 7J:
- commit
- push
- update handoff
- STOP

Do NOT:
- retrain
- rerun with changed settings
- choose best seed
- alter architecture/reference/parser
- start Demo packaging automatically

Final recommendation exactly:

`等待 ChatGPT 审核 Task 7J 的 final frozen-architecture test 结果；test 已消费，不自行据此修改模型、选择 seed、调整阈值或重新运行开发实验。`
