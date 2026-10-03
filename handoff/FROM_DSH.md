<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 8B.3-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-R2` |
| Status | **PARTIAL / STOP** (dedicated harness-test gate not green → formal suite not started) |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Starting HEAD | `1287d0a2bab50a278452d4cd0ec8d492347afdfe` |
| Main / origin-main | `c45ecbec7fd293c454ccced22310db32c1542be4` / `c45ecbec7fd293c454ccced22310db32c1542be4` |
| Driver | PASS (reader-thread + queue timeout implemented; harness no longer sets HF/Transformers offline flags; frozen constants unchanged) |
| Driver tests | **12 passed, 3 failed** (`test_repeated_prompt_is_answered_only_once`, `test_no_executable_readline_or_communicate_in_driver`, `test_suite_cases_and_prompts_frozen` — all defects of the new harness test file) |
| Repository tests | NOT RUN (gated behind the dedicated tests) |
| check_setup | READY (live `ultralytics==8.4.164`) |
| Inputs | 6/6, hashes unchanged from R1 |
| Formal suite | **NOT RUN** |
| A1 / A2 / A3 / A4 / B1 / B2 | language=NOT RUN runtime=NOT RUN (all six) |
| Review pack | not created |
| Visual verdict | PENDING CHATGPT/USER REVIEW |
| Output-layout proposal | ACCEPTED / DEFERRED TO TASK 8B.4 |
| Report | `docs/task8b3_six_image_demo_suite.md` |
| STOP reason | dedicated harness-test gate failed 3/15 (`test_repeated_prompt_is_answered_only_once, test_no_executable_readline_or_communicate_in_driver, test_suite_cases_and_prompts_frozen`); §9 forbids starting the formal suite |
| Next action | Awaiting ChatGPT audit. |

Harness-only changes: `scripts/task8b3_interactive_suite.py` (reader thread + queue + exact three-variable child
environment + reusable `_run_interactive_process` helper) and `tests/test_task8b3_interactive_suite.py` (rewritten
for the two frozen defects plus the required coverage). No RC1 product source, delivery source/config, `predict.py`,
ProgramHead, Qwen suggestion, detector, Reference, SAM2, D-B1, threshold, config or checkpoint was modified; the six
frozen prompts were never changed or retried; no training, download, package installation or final-test access
occurred.

---

# FROM_DSH — Task 8B.3-R3 addendum (STOP)

- Task: `8B.3-R3`
- Status: **STOP** — execution budget was exhausted immediately after reading the task book, so the four mandated
  harness corrections were **not** applied: (1) repeated-prompt harness test, (2) AST test restricted to
  `_reader_thread` / `_run_interactive_process`, (3) A3 frozen oracle literal `largest_to_above_to_nearest`,
  (4) `changed_outputs(before, after)` detection of overwritten artifacts.
- Driver tests: NOT RUN in R3 (unchanged: 12 passed, 3 failed).
- Repository tests: NOT RUN. check_setup: NOT RUN in R3 (last known READY, live `ultralytics==8.4.164`).
- Inputs: 6/6 (identities unchanged from R1/R2). Formal suite: **NOT RUN**. Review pack: not created.
- Visual verdict: `PENDING CHATGPT/USER REVIEW`. Output-layout proposal: ACCEPTED / DEFERRED TO TASK 8B.4.
- No harness/test/product/runtime change was made in this turn; the frozen state is intact and auditable.
- Next action: Awaiting ChatGPT audit.
