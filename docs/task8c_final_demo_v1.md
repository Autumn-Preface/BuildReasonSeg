# Task 8C Final Demo V1

Status: **STOPPED_FOR_SUPERVISOR — GT review rendering error**. Formal runner invocations **1**; candidates attempted **4**, once each; inference retries **0**. No scientific verdict is claimed.

## 1. Scope

Evaluation-only on eval/task8c-final-demo-v1. Initial accepted predecessor: 92f28133931231d16b6053aca253b9e5073955ae. Both earlier procedural STOP checkpoints and subsequent Supervisor dispositions are preserved. Latest accepted continuation: ce408ac73baa5c3e5fcd0b128c545b1717b456ee, SECOND_STOP_VALID / CONTINUE_AUTHORIZED. Product, governance and three harness files remain byte-identical. Frozen domain: WHU East Asia RGB optical overhead tiles, scene_disjoint_v1, tile-relative directions only.

## 2. Immutable candidates

| Order | Relation | Sample | Program | Prompt |
|---|---|---|---|---|
| 1 | right | buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91 | largest_to_right_of_to_nearest | 最大建筑右侧最近的建筑 |
| 2 | left | buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3 | largest_to_left_of_to_nearest | 最大建筑左侧最近的建筑 |
| 3 | above | buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314 | largest_to_above_to_nearest | 最大建筑上方最近的建筑 |
| 4 | below | buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450 | largest_to_below_to_nearest | 最大建筑下方最近的建筑 |

Exact paths, 512x512, RGB, TIFF and frozen SHA256 gates all PASS; identities recorded in JSON. No preview/replacement.

## 3. Qualitative reuse disclosure

The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

这些定性 Demo 候选是在 Task 7J 最终冻结架构测试指标已经被消耗之后，从冻结的 BuildSpatialReason v0.2 测试划分中确定性选取的。其定性复用不会改变、替换或重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。

## 4. Formal freeze and gate history

Earlier 40-test PASS and later 27 PASS / 15 fixture setup errors / 0 assertion failures remain historical evidence. Supervisor-authorized corrected invocation of the same current revision passed **42/42 / exit 0**. No subsequent test rerun or harness change occurred. Accepted external setup **READY**, source **135/135**, manifest **MATCH**, four TIFF gates **PASS** were reused.

Harness Ready checkpoint: 3fac0e668c2af30b35bfae1ab8a3f36cee329e26. FORMAL_FINAL_DEMO_FREEZE_V1 commit: **20615080264774eb7b93b7d6a3c1e547093c4876**. Freeze was pushed and exact local/remote HEAD equality, harness identities, assets/input/history, grandfathered settings and outside-cache writability were verified before inference. Raw runtime and output hashes were committed/pushed at **f41f32a05b83141312b078e6ab5975fbf70ec889** before any GT read.

Exact runtime environment: YOLO_CONFIG_DIR=C:\Users\ROG\AppData\Local\Temp\task8c-yolo-config-v1; MPLCONFIGDIR=C:\Users\ROG\AppData\Local\Temp\task8c-mpl-cache-v1. Both were created/probed before use outside repo/external; no cache fell back into external during formal execution.

## 5. Language results

| Relation | Initial program | Suggested program | Decision | Language status |
|---|---|---|---|---|
| right | largest_to_right_of | largest_to_right_of_to_nearest | Y to exact expected suggestion | SUGGESTION_CORRECT |
| left | largest_to_left_of | largest_to_left_of_to_nearest | Y to exact expected suggestion | SUGGESTION_CORRECT |
| above | largest_to_above | largest_to_above_to_nearest | Y to exact expected suggestion | SUGGESTION_CORRECT |
| below | largest_to_below | largest_to_below_to_nearest | Y to exact expected suggestion | SUGGESTION_CORRECT |

All four expected programs executed via validated exact suggestions. No runtime fallback was accepted. Initial parses, confidence, raw stdout and driver decisions are preserved; direct parse correctness is not claimed.

## 6. Runtime results

| Relation | Exit | Runtime | Raw / merged proposals | Current auto reference ID | Mask area | Run root |
|---|---|---|---|---|---|
| right | 0 | SUCCESS | 7 / 6 | 3 | 415 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1010 |
| left | 0 | SUCCESS | 59 / 46 | 11 | 1627 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1003 |
| above | 0 | SUCCESS | 10 / 10 | 5 | 1580 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1008 |
| below | 0 | SUCCESS | 7 / 6 | 1 | 2955 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\1009 |

One runner invocation, right -> left -> above -> below, each once. Runtime SUCCESS means **RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED**. Current proposal IDs/counts are observed from saved production output and were never compared with or selected using historical REF-01 or P1D12 values. Each case saved 15 artifacts; exact byte/SHA identities, transcripts, context/reference/mask metadata and timings remain frozen in JSON.

## 7. Reference post-hoc audit

The GT embargo was honored: evaluator started only after all four child processes exited, runtime JSON/artifact hashes froze and the raw checkpoint was pushed. Evaluator contains no model/detector/predict calls. Native truth was accessed for the first relation, right; the process reached review rendering and failed. No complete per-case GT result was persisted by the evaluator. No reference identity conclusion is claimed.

## 8. Target GT metrics

No completed IoU/Dice evidence or aggregate is claimed. The failed evaluator did not persist its in-memory first-case audit. Formal masks are unchanged and available for an authorized future post-hoc continuation; they must not be regenerated.

## 9. Semantic-chain classifications

Not completed. All raw runtime cases retain semantic_status NOT_EVALUATED. No case is labeled Demo PASS or semantic correctness.

## 10. Review pack and evaluator failure

Evaluator invocation count **1**, exit **1**, model calls **0**, detector calls **0**. Failure in scripts/task8c_final_demo_evaluate.py:226:

```text
ValueError: assignment destination is read-only
target_panel[boundary(masks[audit["canonical_target_id"]])] = (0, 255, 255)
```

The SUCCESS branch creates target_panel with np.asarray from the saved Pillow overlay; that array is read-only. Drawing the GT boundary raises before the first PNG is saved. The fake review test covered a failed runtime without a saved overlay, so its PASS did not cover this SUCCESS rendering path. The full traceback is retained in JSON. Four review PNGs and contact sheet are **NOT_GENERATED**; no substitute or partial pack is claimed. Frozen harness code was not patched or rerun after this error.

## 11. No retry / no tuning

Candidate replacement, prompt replacement, prompt retry, inference retry, reference override, inspect-before-run, threshold tuning, ranking tuning, mask-quality tuning, architecture change, model change, checkpoint change, training and dependency installation: **NONE**. All four formal attempts are consumed; no model rerun is authorized or needed for this evaluator-side issue.

## 12. Integrity, grandfathered artifact and limitations

Stop-time read-only integrity PASS: external source/config **135/135**; external manifest byte/hash identical to Git; all 1543 historical output files unchanged in bytes/SHA/mtime; all 6 inputs and 20 protected model asset byte/SHA identities unchanged; all pre-existing external byte identities unchanged. Frozen formal artifacts and raw runtime/log hashes are unchanged. Additions are only the four observed Task8C run roots and Task8C logs. No additional external-root runtime config file appeared. Product/governance/harness identities remain frozen.

Grandfathered preflight side effect, explicitly accepted by Supervisor:

- Path: C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\Ultralytics\settings.json
- Bytes: 606
- SHA256: 1dae32b8abfc0f084cdee6c446dcf5420031bfcf005fb358f473a9a34d368cdf
- Disposition: GRANDFATHERED_TASK8C_PREFLIGHT_RUNTIME_SIDE_EFFECT
- must_remain_unchanged: true — verified unchanged; never deleted, moved, edited or overwritten.

Current STOP is an evaluation harness rendering failure, not an adverse model/algorithm Demo result. Fixing it would change the frozen evaluator bytes, which current continuation forbids; no hidden monkey patch, redraw workaround, test weakening or source edit was applied. Supervisor disposition is required before any evaluation-only repair/identity update. Do not repeat formal inference. Domain limitations and prior scientific conclusions remain unchanged; no generalization or high-quality-mask claim is made.

## 13. Supervisor scientific verdict

Final scientific verdict belongs to **ChatGPT Supervisor**. Four structural runtime successes do not establish correct references/targets or mask quality. GT review completion is outstanding; this checkpoint is **not READY_FOR_SUPERVISOR_AUDIT** procedural completion.

Next gate: CHATGPT_TASK8C_FINAL_DEMO_REMOTE_AND_VISUAL_AUDIT.


## Supervisor-authorized post-runtime evaluator repair

Current continuation status: POST_RUNTIME_EVALUATOR_RENDER_FIX_V1 tested, pending commit/push and second evaluator. THIRD_STOP_VALID / CONTINUE_EVALUATOR_ONLY accepted at 7704fa520a85fe25457c5950e4d0062695859099. Earlier failures remain historical evidence. The only evaluator change is np.asarray(saved overlay) -> np.array(saved overlay, copy=True), enabling local review-boundary drawing. GT reconstruction, coordinate mapping, IoU/Dice, best overlap, identity/classification, aggregate semantics, candidate/GT IDs and artifact verification are unchanged. Original formal freeze 20615080264774eb7b93b7d6a3c1e547093c4876 and raw checkpoint f41f32a05b83141312b078e6ab5975fbf70ec889 are preserved. Formal runner must never rerun.

New evaluator SHA256: 5a23d5e2a0801e81a9a6af3c44fa71afdf5b0864bfa6f50cce6aabb71cc5e542. New tests SHA256: bfcb7b13dde3bc470b5b703c1bebca83493384308a2c0d0839d08fbf853584b5. Current fake-only suite: 43/43 PASS, exit 0; original 42 tests preserved, one saved-overlay SUCCESS regression appended. The test verifies exact overlay reads, writable target-boundary rendering, byte-identical input overlay, deterministic review/PNG and no process/model/detector/predict calls. First evaluator invocation remains exit 1 with full traceback and zero model/detector calls; only invocation 2 is authorized after push/integrity checks.
