# Task 8C Final Demo V1

Status: **STOPPED_FOR_SUPERVISOR**. Formal runner invocations: **0**. Real candidates attempted: **0**.

## 1. Scope

Evaluation-only harness work on `eval/task8c-final-demo-v1`, created from accepted predecessor `92f28133931231d16b6053aca253b9e5073955ae`. The exact predecessor was fetched and verified. Governance and product source remain unchanged. East Asia RGB optical overhead imagery and tile-relative directions only.

## 2. Immutable candidates

| Order | Relation | Sample | Expected program | Prompt |
|---|---|---|---|---|
| 1 | right | buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91 | largest_to_right_of_to_nearest | 最大建筑右侧最近的建筑 |
| 2 | left | buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3 | largest_to_left_of_to_nearest | 最大建筑左侧最近的建筑 |
| 3 | above | buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314 | largest_to_above_to_nearest | 最大建筑上方最近的建筑 |
| 4 | below | buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450 | largest_to_below_to_nearest | 最大建筑下方最近的建筑 |

Raster paths and SHA256 locks are recorded in the JSON, copied from the task book. Actual candidate image identity preflight has **not run**.

## 3. Qualitative reuse disclosure

The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

这些定性 Demo 候选是在 Task 7J 最终冻结架构测试指标已经被消耗之后，从冻结的 BuildSpatialReason v0.2 测试划分中确定性选取的。其定性复用不会改变、替换或重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。

## 4. Harness gate and formal freeze

The first dedicated fake-only run passed 40 tests in 0.72 s. Two additional safety tests and a tighter end-of-pipe timeout check were then added. The current version revalidation exited 1: **27 passed / 15 errors**. All errors arose in pytest temporary-directory fixture setup: the nested `--basetemp` parent did not exist (`FileNotFoundError / WinError 3`). This was an executor invocation mistake, not a demonstrated product regression. The earlier PASS does not validate the current version.

Task book section 39 explicitly requires STOP when dedicated harness tests are not 100% green. The error was preserved without rerun, path repair, test weakening, dependency changes, or product fixes. External 135/135 check, check_setup, locked-TIFF gates, asset/output snapshots and FORMAL_FINAL_DEMO_FREEZE_V1 have **not run / not been completed**.

The runner uses an exclusive persistent invocation journal, records each attempt before process creation, enforces a 900-second per-case deadline and continues after case-level failures. A second invocation is rejected. The only extra predict flag is `--confirm-command`, enabling the task-prescribed direct Y/N interaction; model/reference/threshold defaults are preserved. This implementation remains pending full current-version fake-test validation.

## 5. Language decisions

No real language/model call has occurred. Direct and suggestion Y/N rules and fallback=N are implemented; only fake child processes were used.

## 6. Runtime outcomes

No formal invocation or candidate attempt has been consumed. There are no real runtime outcomes.

## 7. Reference post-hoc audit

Not started. Evaluator is designed to verify four exited child processes, frozen runtime JSON and exact output hashes before reading existing native-vector caches. No candidate GT has been opened.

## 8. Target GT metrics

Not evaluated; no IoU/Dice claims.

## 9. Semantic-chain classifications

No real case has been classified. Runtime SUCCESS would retain RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED before GT audit.

## 10. Review artifacts

Not generated. No review PNG or contact sheet is claimed.

## 11. No retry / no tuning

Candidate replacement, prompt replacement, prompt retry, inference retry, reference override, inspect-before-run, threshold tuning, ranking tuning, mask-quality tuning, architecture change, model change, checkpoint change, training and dependency installation: **NONE**.

## 12. Limitations and STOP disposition

Current dedicated test gate is not green. Its known immediate cause is an uncreated parent directory for an executor-specified temporary test base. No automatic repair was performed because section 39 requires stopping with evidence. External integrity checks remain unperformed; therefore evidence uses null/NOT_RUN values rather than fabricating PASS. No external source/config/assets/input/output/log write operation was reached. Product and governance diff is empty.

Supervisor continuation disposition is needed before correcting the L0 invocation and completing the gates. No real candidate needs or is authorized for retry. No Assisted Mode or failure-repair work was started.

## 13. Supervisor scientific verdict

Final scientific acceptance belongs to **ChatGPT Supervisor**. This STOP checkpoint is an executor record, not a scientific Demo verdict.

Next gate: `CHATGPT_TASK8C_FINAL_DEMO_REMOTE_AND_VISUAL_AUDIT`.
