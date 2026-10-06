# Task 8C Final Demo V1

Status: **READY_FOR_SUPERVISOR_AUDIT**. One formal runner invocation; four candidates attempted once each; inference retries **0**. Post-hoc GT audit and five review PNGs complete. Runtime SUCCESS **4/4**; CHAIN_IDENTITY_MATCH **0/4**. This is procedural completion; the scientific verdict remains with ChatGPT Supervisor.

## 1. Scope

Evaluation-only on `eval/task8c-final-demo-v1`, starting from accepted Task 8B.4 predecessor `92f28133931231d16b6053aca253b9e5073955ae`. Latest continuation accepted `7704fa520a85fe25457c5950e4d0062695859099`: **THIRD_STOP_VALID / CONTINUE_EVALUATOR_ONLY**. Only the explicit saved-overlay writable-copy repair and one fake regression test were added; product, governance, runner and frozen formal runtime remain unchanged. Frozen supported domain: WHU East Asia RGB optical overhead 512x512 tiles, BuildSpatialReason v0.2 `test`, `scene_disjoint_v1`, tile-relative right/left/above/below. No A2 or historical six-case suite was run.

## 2. Immutable candidates

| Order | Relation | Sample ID | Prompt | Expected program | GT ref / target |
| --- | --- | --- | --- | --- | --- |
| 1 | right | buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91 | 最大建筑右侧最近的建筑 | largest_to_right_of_to_nearest | 4 / 3 |
| 2 | left | buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3 | 最大建筑左侧最近的建筑 | largest_to_left_of_to_nearest | 26 / 24 |
| 3 | above | buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314 | 最大建筑上方最近的建筑 | largest_to_above_to_nearest | 4 / 6 |
| 4 | below | buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450 | 最大建筑下方最近的建筑 | largest_to_below_to_nearest | 3 / 2 |

All four path/512x512/RGB/TIFF/SHA256 identity gates passed. Exact input identities:

| Relation | Frozen TIFF path | SHA256 |
| --- | --- | --- |
| right | C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif | 1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2 |
| left | C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif | eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38 |
| above | C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif | 0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd |
| below | C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif | c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7 |

## 3. Qualitative reuse disclosure

The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

这些定性 Demo 候选是在 Task 7J 最终冻结架构测试指标已经被消耗之后，从冻结的 BuildSpatialReason v0.2 测试划分中确定性选取的。其定性复用不会改变、替换或重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。

## 4. Formal freeze, historical STOPs and authorized evaluator repair

The original **FORMAL_FINAL_DEMO_FREEZE_V1** payload is preserved byte-for-byte in its Git record and equal as a nested JSON payload. Its pushed commit is `20615080264774eb7b93b7d6a3c1e547093c4876`. Harness Ready checkpoint: `3fac0e668c2af30b35bfae1ab8a3f36cee329e26`. Local/remote equality, frozen harness identities, caches, source/config, input/history/protected assets and grandfathered settings were verified before the single formal invocation. Raw runtime checkpoint `f41f32a05b83141312b078e6ab5975fbf70ec889` was pushed before any GT evaluator access.

Frozen external `logs/task8c_final_demo_v1/runtime_results.json`: **102173 bytes**, SHA256 **2fcbb705acf1ceeb28fc9609bfda869150e7b800b2bea0d9fba9a723c12ff8ca**. This raw file and every formal artifact/transcript remained unchanged through evaluator invocation #2. Raw cases retain `RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED`; post-hoc conclusions are recorded separately in `gt_audit`.

| Historical checkpoint / gate | Observed result | Disposition / preservation |
| --- | --- | --- |
| Earlier fake-test revision | 40 passed / exit 0 | Historical only; did not replace current 42-test gate |
| 44a7dd1c3c3c8afe7ef1a0a569619017a7d4e4ea | 27 passed / 15 fixture setup errors / 0 assertion failures | STOP_VALID; missing nested basetemp parent corrected without harness edits |
| Corrected current fake-test invocation | 42 passed in 0.74s / exit 0 | Accepted original pre-inference gate; unchanged harness |
| ce408ac73baa5c3e5fcd0b128c545b1717b456ee | check_setup READY; external Ultralytics settings side effect | SECOND_STOP_VALID; explicitly grandfathered unchanged |
| 7704fa520a85fe25457c5950e4d0062695859099 | Evaluator #1 exit 1; 0 model / 0 detector calls; no completed review | THIRD_STOP_VALID; evaluator-only fix authorized |

Accepted preflight gates: external source/config **135/135**, external manifest **MATCH**, `check_setup.py` **READY**, all four TIFF identity gates **PASS**. The accepted setup result was reused; no setup rerun was needed. Under frozen GOV-D007, complete-delivery full-suite acceptance belongs to external RC1; no new canonical complete-suite gate was imposed. Task 8B.4 external full-suite result remains its accepted historical **150 passed** result.

Evaluator invocation #1 failed during the first SUCCESS review target-boundary assignment. It accessed right native truth, but persisted no complete per-case GT audit and produced zero review PNGs. The original error, full traceback and STOP record remain unchanged in JSON and in the prior pushed checkpoint:

```text
Traceback (most recent call last):
  File "C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\scripts\task8c_final_demo_evaluate.py", line 302, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\scripts\task8c_final_demo_evaluate.py", line 275, in main
    reviewed = review_image(row, audit, reference, masks)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\scripts\task8c_final_demo_evaluate.py", line 226, in review_image
    target_panel[boundary(masks[audit["canonical_target_id"]])] = (0, 255, 255)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: assignment destination is read-only
```

Supervisor authorized **POST_RUNTIME_EVALUATOR_RENDER_FIX_V1**, reason `READ_ONLY_OVERLAY_ARRAY_RENDER_FAILURE`. The sole evaluator replacement changes `np.asarray(saved overlay)` to `np.array(saved overlay, copy=True)` so the local review panel can accept a GT boundary. GT reconstruction, context mapping, IoU/Dice, best overlap, identity/enums/aggregate, candidate/GT IDs, review semantics and runtime/artifact verification remain unchanged. Original freeze identities were not rewritten or portrayed as current evaluator bytes.

| File | Identity scope | SHA256 |
| --- | --- | --- |
| scripts/task8c_final_demo_runner.py | Original frozen bytes, unchanged | 15fcb108ca385c96d970e1076f04706e3bb3f398947df37a85ad67bbc59cc666 |
| scripts/task8c_final_demo_evaluate.py | Original formal freeze | d471563773102c7ac53f8c766f26bd7bd3a43e06a5063f3e45f5dfe8990b4b9b |
| scripts/task8c_final_demo_evaluate.py | Authorized post-runtime fix | 5a23d5e2a0801e81a9a6af3c44fa71afdf5b0864bfa6f50cce6aabb71cc5e542 |
| tests/test_task8c_final_demo.py | Original 42-test revision | 051755ce1a3af05bc6a862f31506c1ac7e2705878559a5f14c5ee4f38e6c1e3a |
| tests/test_task8c_final_demo.py | Original 42 retained plus one regression | bfcb7b13dde3bc470b5b703c1bebca83493384308a2c0d0839d08fbf853584b5 |

Current fake-only dedicated suite: **43 passed in 0.80s / exit 0**. The existing 42 tests remain byte-for-byte as the prefix; one SUCCESS + saved-overlay regression proves overlay reads, writable GT-boundary drawing, source-overlay byte identity, deterministic review/PNG and zero model/detector/predict/process calls. The evaluator-only repair checkpoint was committed and pushed at **c18ba1e30e4772c7a6de4ab06410b5692a15eef4** before evaluator #2. Full frozen-runtime/source/history/asset/settings integrity was verified again before the second invocation.

Evaluator invocation count is **2**, not 1. Invocation #1: exit **1**, reason `READ_ONLY_OVERLAY_ARRAY_RENDER_FAILURE`, model calls **0**, detector calls **0**. Invocation #2: exit **0**, current repaired evaluator SHA above, model calls **0**, detector calls **0**, formal output writes **0**, native-vector writes **0**. Invocation #2 completed right, left, above, below audits and all five PNGs. No further evaluator or inference invocation was performed. Scientific computation and formal runtime changed: **false**.

Execution environment: Python `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`; `YOLO_CONFIG_DIR=C:\Users\ROG\AppData\Local\Temp\task8c-yolo-config-v1`; `MPLCONFIGDIR=C:\Users\ROG\AppData\Local\Temp\task8c-mpl-cache-v1`; `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Cache directories were created/probed before use outside repository/external; no subsequent runtime cache was added to external RC1.

## 5. Language results

| Relation | Initial program | Initial confidence (raw) | Suggested program | Decision | Language status |
| --- | --- | --- | --- | --- | --- |
| right | largest_to_right_of | 0.9999436140060425 | largest_to_right_of_to_nearest | Y: exact expected suggestion | SUGGESTION_CORRECT |
| left | largest_to_left_of | 0.9991422891616821 | largest_to_left_of_to_nearest | Y: exact expected suggestion | SUGGESTION_CORRECT |
| above | largest_to_above | 0.9998648166656494 | largest_to_above_to_nearest | Y: exact expected suggestion | SUGGESTION_CORRECT |
| below | largest_to_below | 0.9999477863311768 | largest_to_below_to_nearest | Y: exact expected suggestion | SUGGESTION_CORRECT |

All four expected programs executed through exact matching suggestions. Direct initial parses lacked `to_nearest`; direct-program correctness is not claimed. Runtime fallback was never accepted. Full initial confidence, displayed confidence, raw stdout and driver decisions remain in JSON. No prompt retry or replacement occurred.

## 6. Runtime results

| Relation | Exit | Runtime | Raw / merged proposals | Current automatic proposal ref ID | Mask area | Run root |
| --- | --- | --- | --- | --- | --- | --- |
| right | 0 | SUCCESS | 7 / 6 | 3 | 415 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1010 |
| left | 0 | SUCCESS | 59 / 46 | 11 | 1627 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1003 |
| above | 0 | SUCCESS | 10 / 10 | 5 | 1580 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1008 |
| below | 0 | SUCCESS | 7 / 6 | 1 | 2955 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1009 |

The sole runner invocation executed **right -> left -> above -> below**, each once, and all child processes exited before output hashes froze. Each run root contains 15 artifacts (13 diagnostics, mask, overlay). All 60 artifact identities and six log identities are preserved. Current proposal counts/IDs are observed production output, never a historical P1D12/REF-01 oracle. Automatic proposal IDs and native GT instance IDs belong to separate namespaces; equality of their numbers does not establish identity. Runtime SUCCESS is structural and does not establish correct reference, target or mask quality. Exact argv used automatic mode and `--confirm-command`, with no `--reference-id` or `--inspect-proposals`.

## 7. Reference post-hoc audit

The GT embargo was honored: all formal child processes exited, outputs/hashes froze and the raw checkpoint was pushed before GT access. The evaluator reads saved `reference_context_mask.png`, maps the frozen context origin/size back to the original 512x512 coordinate system, and compares with existing Task 6M WHU native-vector masks. Reconstructing cached native rings in original instance order exactly matched each existing label map; test split/sample/source-feature IDs were verified. No cache regeneration, detector call or model call was used. Identity uses the frozen best-overlap rule; no new IoU pass threshold was introduced.

| Relation | Canonical GT ref | Selected ref best GT instance | Best IoU | IoU with canonical ref | Reference identity match |
| --- | --- | --- | --- | --- | --- |
| right | 4 | 3 | 0.478559 | 0.000000 | false |
| left | 26 | 37 | 0.736328 | 0.000000 | false |
| above | 4 | 4 | 0.909743 | 0.909743 | true |
| below | 3 | 4 | 0.838361 | 0.000000 | false |

Reference identity matches only for above. For right/left/below the frozen selected reference overlaps other native instances most strongly. These are post-hoc identity findings, not a request to replace current selections or reconcile old proposal IDs. Exact truth-input paths/SHA256, native metadata and continuous values remain in JSON.

## 8. Target GT metrics

| Relation | Canonical GT target | Predicted area | GT area | Target IoU | Target Dice | Predicted best GT instance | Best IoU | Target identity match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| right | 3 | 415 | 1401 | 0.000000 | 0.000000 | null (no positive overlap) | 0.000000 | false |
| left | 24 | 1627 | 1934 | 0.000000 | 0.000000 | 34 | 0.176899 | false |
| above | 6 | 1580 | 2571 | 0.077622 | 0.144062 | 7 | 0.481581 | false |
| below | 2 | 2955 | 1210 | 0.215703 | 0.354862 | 3 | 0.603056 | false |

Mean target IoU over the four runtime successes: **0.07333136469693911**. Mean target Dice: **0.1247309041660509**. Every case is evaluable. right has no positive target-mask overlap with any native GT instance, so best instance is null. Continuous metrics are reported without adding a quality acceptance threshold; masks/overlays were never regenerated or altered.

## 9. Semantic-chain classifications and aggregate

| Relation | Frozen-rule post-hoc classification |
| --- | --- |
| right | REFERENCE_AND_TARGET_IDENTITY_MISMATCH |
| left | REFERENCE_AND_TARGET_IDENTITY_MISMATCH |
| above | TARGET_IDENTITY_MISMATCH |
| below | REFERENCE_AND_TARGET_IDENTITY_MISMATCH |

| Aggregate field | Count / value |
| --- | --- |
| cases_attempted | 4 |
| language_expected_program_executed | 4 |
| runtime_success | 4 |
| runtime_failed | 0 |
| chain_identity_match | 0 |
| reference_identity_mismatch | 0 |
| target_identity_mismatch | 1 |
| both_identity_mismatch | 3 |
| language_failed | 0 |
| not_evaluable | 0 |
| mean_target_iou_over_runtime_success | 0.07333136469693911 |
| mean_target_dice_over_runtime_success | 0.1247309041660509 |

There are **0 CHAIN_IDENTITY_MATCH**, **1 TARGET_IDENTITY_MISMATCH** (above), and **3 REFERENCE_AND_TARGET_IDENTITY_MISMATCH** (right/left/below). The remaining frozen enum counts are zero. The evaluator generated these results without manual classification edits; no failure was removed from aggregates. Raw runtime `semantic_status=NOT_EVALUATED` was preserved separately from these post-hoc findings.

## 10. Review pack

All five PNGs were generated by evaluator #2 and visually inspected. Each case shows original RGB, selected-reference magenta boundary with canonical-reference green boundary, and the saved production overlay with canonical-target cyan boundary; program/language/runtime/reference/continuous metrics/classification text is readable. Contact order is right/left, above/below. Inspection recorded no need for redraw or content changes.

| Artifact | Exact repository path | SHA256 |
| --- | --- | --- |
| right_review.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8c_final_demo_v1\right_review.png | a0ae1883a1f32c26e8cafd8096244514fb3cd8d00ddf30b21841e5671314f58d |
| left_review.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8c_final_demo_v1\left_review.png | 0fdfa238f6a2e5561bae39ebb2d07abb4c0712816ccac01ec23e69256aaeb4b3 |
| above_review.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8c_final_demo_v1\above_review.png | 6fc47bf3bf8ed0eacb44139ff1cfedad16ddcaa8fa1900c199f3508263e6fe6b |
| below_review.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8c_final_demo_v1\below_review.png | 34e68f8121da0766fb686cd27052a192eb3005e3568aeef94ef116b051a3ab89 |
| contact_sheet.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8c_final_demo_v1\contact_sheet.png | 759ed3069a069eeee0466ef31da3aea586ac474a202a84da11ee9bd1bbe8c83c |

Review panels use existing saved formal overlays and new in-memory GT boundary drawing; external masks/overlays are untouched. The first evaluator failure and zero completed reviews remain historical evidence; this pack was produced only by the separately authorized second evaluator.

## 11. No retry / no tuning / no rescue

Candidate replacement, prompt replacement/retry, inference retry, reference override, proposal inspect, threshold/ranking/mask-quality tuning, model/checkpoint/architecture change, training, dependency installation and source/product rescue: **NONE**. Formal runner invocations **1**, real candidates attempted **4**, inference retries **0**. Both evaluators together made **0 model calls / 0 detector calls**. The only retry was the explicitly Supervisor-authorized evaluator #2 after the isolated rendering fix; it is not an inference retry. No new algorithm branch, Assisted Mode or failed-case repair was entered.

## 12. Integrity, grandfathered side effect and limitations

Final read-only integrity gate **PASS**: Git-canonical helper checked **135**, match **135**, missing **0**, mismatch **0**. External `source_manifest.json` is byte/hash identical to Git canonical and unchanged (24376 bytes, SHA256 `df9a870d72a25421311698f7a8865e07b4a4e11e9bfe52df18975da17b3bcf7e`). Canonical 136 source/manifest byte identities and three governance identities remain unchanged. Runner is byte-identical to the original formal freeze; current evaluator/test bytes match the pushed authorized repair, and scientific evaluator functions are unchanged.

All **1543 pre-Task8C historical output files** retain bytes/SHA256/mtime. All **6 inference/input files**, **20 protected model/Qwen/SAM2 assets**, all pre-existing logs and all **2118 pre-formal external files** retain bytes/SHA256. Full external inventory contains **2184 files**: only **60 Task8C run files + 6 Task8C logs** were added. Frozen four run-root file sets, every formal artifact, runtime JSON and transcripts are unchanged. No additional external-root runtime config appeared; no external file was deleted, migrated, cleaned, overwritten or modified by either evaluator. Read truth inputs match the evaluator-recorded identities after evaluation.

Grandfathered preflight runtime side effect, explicitly accepted by Supervisor, remains immutable:

```yaml
grandfathered_runtime_side_effect:
  path: C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\Ultralytics\settings.json
  bytes: 606
  sha256: 1dae32b8abfc0f084cdee6c446dcf5420031bfcf005fb358f473a9a34d368cdf
  disposition: GRANDFATHERED_TASK8C_PREFLIGHT_RUNTIME_SIDE_EFFECT
  must_remain_unchanged: true
```

Its bytes/SHA remained unchanged at all subsequent gates. It is outside the 135 manifest-listed source/config entries and protected-model set; its acceptance is the explicit Supervisor L2 disposition. It was never deleted, moved, edited or overwritten.

This four-case qualitative reuse is not a new test-set metric, a generalization claim or high-quality segmentation evidence. Current post-hoc mismatches and low target overlap are preserved as Final Demo observations; no cause beyond recorded artifacts is inferred and no repair is authorized. Prior frozen architecture, thresholds, ranking, seeds, scientific conclusions and Task 7J metrics remain unchanged. Governance staleness is acknowledged without editing governance.

## 13. Supervisor scientific verdict and handoff

The final scientific verdict belongs to **ChatGPT Supervisor**, pending **CHATGPT_TASK8C_FINAL_DEMO_REMOTE_AND_VISUAL_AUDIT**. Executor procedural criteria are complete: four candidates attempted once; all evidence preserved; zero-model-call post-hoc audit completed; review pack generated; integrity passed. Test PASS and runtime SUCCESS do not establish semantic correctness. The observed result includes zero complete chain identity matches. `CURRENT_TASK` and `EXECUTOR_STATE` are **READY_FOR_SUPERVISOR_AUDIT**. Commit/push this review checkpoint, then STOP; no further inference, evaluator, model/algorithm fix or Assisted Mode is authorized.
