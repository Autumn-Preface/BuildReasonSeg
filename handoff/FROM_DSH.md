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

# FROM_DSH — Task 8B.3-R4A Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-R4A` |
| Status | **COMPLETE** |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Starting HEAD | `2562c3bc53b2600098824e93572cca291d334e33` |
| Main / origin-main | `c45ecbec7fd293c454ccced22310db32c1542be4` / `c45ecbec7fd293c454ccced22310db32c1542be4` |
| Harness corrections | PASS (A3 literal oracle; `changed_outputs(before, after)`; repeated-prompt test; AST test on `_reader_thread` / `_run_interactive_process` only) |
| Driver tests | **20 passed** (`pytest tests/test_task8b3_interactive_suite.py -q`) |
| Repository tests | **1554 passed, 1 failed** first run → failure was the `Watt` reference dropped from this handoff by the R3 rewrite; restored in this same allowed file → affected test PASS |
| Formal suite | NOT RUN BY DESIGN (§8 of this task forbids it; a separate ChatGPT task will issue the single formal run) |
| A1–B2 | NOT RUN |
| Review pack | NOT CREATED / unchanged |
| Output-layout proposal | ACCEPTED / DEFERRED TO TASK 8B.4 |
| Report | `docs/task8b3_six_image_demo_suite.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit. |

Watt was not needed for Task 8B.3-R4A (no downloads, no transfers); the pre-existing Watt instance, when present,
remains transport-only and is not owned by this project.

No RC1 product/runtime source, `delivery_src/**`, external delivery source/config, `predict.py`, ProgramHead, Qwen
suggestion, Validator, detector, Reference, SAM2, D-B1, threshold, config, checkpoint or model asset was modified;
the six frozen prompts were never changed or retried; no Assisted Mode, `--reference-id` or `--inspect-proposals`
was used; no real six-image suite was executed; no training, download, package installation or final-test access
occurred.
