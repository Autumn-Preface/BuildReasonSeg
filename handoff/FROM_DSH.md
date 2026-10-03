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

# FROM_DSH — Task 8B.3-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-R1` |
| Status | **PARTIAL** (deterministic driver implemented; driver-test gate not green, so the real six-case suite was not started) |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Base audited RC1 | `c45ecbec7fd293c454ccced22310db32c1542be4` |
| Remote main | `c45ecbec7fd293c454ccced22310db32c1542be4` (fast-forwarded and pushed this task) |
| Driver | `scripts/task8b3_interactive_suite.py` — implemented per the frozen design (binary pipes, incremental UTF-8 decoding, no newline required, 15-min per-case timeout) |
| Driver tests | `tests/test_task8b3_interactive_suite.py` → **11 passed, 2 failed** (both are defects of the new test file itself; the fix is confined to that file) |
| Repository tests | not run (the gate order requires the dedicated tests to pass first) |
| check_setup | READY (live `ultralytics==8.4.164`) |
| Inputs | 6/6 present and hashed (A1–A4 1024×1024 RGB PNG; B1/B2 5000×5000 RGB TIFF) |
| A1 | language=NOT RUN runtime=NOT RUN |
| A2 | language=NOT RUN runtime=NOT RUN |
| A3 | language=NOT RUN runtime=NOT RUN |
| A4 | language=NOT RUN runtime=NOT RUN |
| B1 | language=NOT RUN runtime=NOT RUN |
| B2 | language=NOT RUN runtime=NOT RUN |
| Review pack | not created (no outputs produced) |
| Report | `docs/task8b3_six_image_demo_suite.md` |
| Output-layout proposal | ACCEPTED / DEFERRED TO SEPARATE DELIVERY ITERATION (reserved as Task 8B.4; not implemented) |
| STOP reason | driver-test gate not green (`test_correct_direct_program_sends_y`, `test_decision_path_has_no_readline`); Task 8B.3-R1 §10 forbids starting the real suite before both test commands pass |
| Next action | Awaiting ChatGPT audit. |

No RC1 product runtime, `predict.py`, ProgramHead, Qwen suggestion, detector/Reference/SAM2/D-B1, threshold,
config or checkpoint was modified; the six frozen prompts were never altered or retried; no Assisted Mode,
`--reference-id` or `--inspect-proposals` was used; no training, download, package installation or final-test
access occurred.

Watt was not needed for Task 8B.3-R1 (no downloads, no transfers).
