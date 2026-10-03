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

# FROM_DSH — Task 8B.2-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.2-R2` |
| Branch | `audit/task8b2-rc1-runtime-closure` |
| Starting HEAD | `7e885e176f23f301156d32f1800c5719165c8322` |
| Ultralytics runtime | 8.4.164 (installed into `.conda/buildreasonseg-mvp`; pinned `==8.4.164` in `environment.yml` and `requirements.txt`) |
| check_setup | `BuildReasonSeg environment: READY` (live runtime checks; runtime vs model provenance explicit) |
| External sync | canonical manifest 135 entries; pre-sync exactly the 4 authorized mismatches; post-sync 135/135 PASS |
| External test suite | 109 passed (external `tests/test_setup_checker.py` 12 passed) |
| Repository test suite | 1535 passed |
| A1 parse | `largest -> right_of -> nearest` (`largest_to_right_of_to_nearest`) |
| A1 Y-path | PASS (`Result : SUCCESS`; detector → Reference 48 → SAM2 → D-B1 executed; mask/overlay/diagnostics written) |
| A1 N-path | PASS (`[E901 USER_ABORTED]`, exit 90; detector/core not executed; no new mask/overlay) |
| Report path | `docs/task8b2_r2_runtime_environment_closure.md` |
| RC1 status | READY |
| Next action | Awaiting ChatGPT audit. Do not start Task 8C or additional Demo images. |

Environment closure only: no training, no checkpoint/threshold/ProgramHead/Qwen-prompt/Reference/SAM2/D-B1/
dataset/final-test change, no package other than `ultralytics==8.4.164` added, and no research conclusion changed.
Protected core versions were verified unchanged (torch 2.13.0+cu132, torchvision 0.28.0+cu132, transformers 5.17.0,
numpy 2.4.6, OpenCV 5.0.0, scipy 1.17.1) and `pip check` reports no broken requirements.

Watt was not needed for Task 8B.2-R2 (no downloads, no transfers); the pre-existing Watt instance, when present,
remains transport-only and is not owned by this project.
