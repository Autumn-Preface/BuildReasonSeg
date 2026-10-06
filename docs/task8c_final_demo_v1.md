# Task 8C Final Demo V1

Status: **STOPPED_FOR_SUPERVISOR**. Formal runner invocations: **0**. Real candidates attempted: **0**.

## 1. Scope

Evaluation-only continuation on the existing eval/task8c-final-demo-v1 branch. Accepted STOP checkpoint: 44a7dd1c3c3c8afe7ef1a0a569619017a7d4e4ea. Supervisor disposition: STOP_VALID / CONTINUE_AUTHORIZED. Product/governance and runner/evaluator/test bytes are unchanged. Domain is WHU East Asia RGB optical overhead imagery, scene_disjoint_v1, tile-relative directions only.

## 2. Immutable candidates

| Order | Relation | Sample | Expected program | Prompt |
|---|---|---|---|---|
| 1 | right | buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91 | largest_to_right_of_to_nearest | 最大建筑右侧最近的建筑 |
| 2 | left | buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3 | largest_to_left_of_to_nearest | 最大建筑左侧最近的建筑 |
| 3 | above | buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314 | largest_to_above_to_nearest | 最大建筑上方最近的建筑 |
| 4 | below | buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450 | largest_to_below_to_nearest | 最大建筑下方最近的建筑 |

All four paths, 512x512 dimensions, RGB mode, TIFF format and task-locked SHA256 values were verified exactly. No candidate substitution.

## 3. Qualitative reuse disclosure

The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

这些定性 Demo 候选是在 Task 7J 最终冻结架构测试指标已经被消耗之后，从冻结的 BuildSpatialReason v0.2 测试划分中确定性选取的。其定性复用不会改变、替换或重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。

## 4. Test history, static gates and formal freeze

Historical executions remain preserved in JSON and Git: earlier 40-test revision PASS; the later 42-test invocation produced **27 passed / 15 fixture setup errors / 0 assertion failures** because its nested basetemp parent did not exist. Supervisor accepted that STOP as procedural, not scientific.

The authorized continuation reran the same current HEAD revision using a fresh basetemp under the existing TEMP parent: **42 passed in 0.74s / exit 0**. Runner/evaluator/tests before/after SHA256 values match Git exactly. No test or harness code changed.

External helper check: **135/135**, missing=0, mismatch=0; stop-time repeat also **135/135**. External manifest remains byte/hash identical to Git canonical. External check_setup: **READY / exit 0**. All four TIFF gates PASS. Protected/output/input/log file snapshots recorded.

**Formal freeze was not created or pushed.** A new external runtime-config file was detected during check_setup and requires STOP before inference. The captured complete file inventory is explicitly labeled after-check_setup and includes this unauthorized addition; it is not promoted to a clean historical baseline.

## 5. Language decisions

No real language-model call has occurred. Automatic reference mode and the fixed Y/N driver remain unchanged. Supervisor independently confirmed --confirm-command support. No fallback acceptance.

## 6. Runtime outcomes

No formal runner invocation or candidate attempt. No scientific Demo outcome exists.

## 7. Reference post-hoc audit

Not started. No native-vector candidate truth has been read. No detector/model rerun.

## 8. Target GT metrics

Not evaluated; no IoU/Dice claims.

## 9. Semantic-chain classifications

No real case classified. Runtime SUCCESS remains RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED before GT audit.

## 10. Review artifacts

Not generated. No review PNG or contact sheet is claimed.

## 11. No retry / no tuning

Candidate replacement, prompt replacement, prompt retry, inference retry, reference override, inspect-before-run, threshold tuning, ranking tuning, mask-quality tuning, architecture change, model change, checkpoint change, training and dependency installation: **NONE**.

## 12. New procedural STOP and integrity

I created the task TEMP parent but did not create the nested Ultralytics cache directory before check_setup. Existing Ultralytics judged the requested cache location unavailable, fell back to the external working directory, and logged Creating new Ultralytics Settings. It created:

- Path: C:/D/DeepSeekHarness/delivery/BuildReasonSeg_Advisor_RC1/Ultralytics/settings.json
- Bytes: 606
- SHA256: 1dae32b8abfc0f084cdee6c446dcf5420031bfcf005fb358f473a9a34d368cdf
- Creation/mtime UTC: 2026-10-06T19:52:24.509183+00:00

This persistent external config addition is outside task book section 34 allowed run/log roots. Sections 32/34/35/39 require preserving evidence and STOP. The file has **not been deleted, moved, edited or cleaned up**. No retry or cache-path repair was performed after detection.

Read-only comparison against accepted Task8B4 closure confirms: 1543 historical output files unchanged in bytes/SHA256/mtime; 6 inputs unchanged; all 20 protected model assets unchanged in byte/SHA256 identity; all 327 existing logs unchanged. The 135 manifest-listed source/config files and external source_manifest remain unchanged. New settings content is recorded transparently as an unauthorized addition.

Supervisor disposition is needed for this file and cache-directory execution mechanics before continuing toward the formal freeze. This is an executor preflight issue, not a Final Demo model/algorithm failure.

## 13. Supervisor scientific verdict

Final scientific acceptance belongs to **ChatGPT Supervisor**. No Final Demo science verdict can be inferred from this procedural STOP.

Next gate: CHATGPT_TASK8C_FINAL_DEMO_REMOTE_AND_VISUAL_AUDIT.
