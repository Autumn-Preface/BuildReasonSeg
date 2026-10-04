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

# FROM_DSH — Task 8B.3-M1A.2C-D1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1A.2C-D1` |
| Status | **COMPLETE** (read-only forensics) |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `9bb955eebd9f7351a71631da88113d46672f3beb` |
| pytest / check_setup / predict / model runs | NONE |
| Nodes recovered from `lastfailed` | 23 |
| Class counts | A=7 (missing delivery assets) · B=10 (fixture assumes complete delivery) · C=0 (compact-runtime candidate) · D=6 (other code-regression candidate) · E=0 (insufficient) |
| `tests/test_task8b_runtime.py` failing nodes | none |
| Conclusion | **MIXED_FAILURES_REQUIRE_CODE_AUDIT** |
| Canonical product / tests / manifest / external delivery | NOT modified |
| Real inference / Demo / Task 8B.4 | NOT RUN / NOT RUN / NOT entered |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Implication for the canonical gate: the canonical tree cannot serve as a full delivery-readiness gate, because it
deliberately omits model weights and downloaded assets; the compact-proposal contract suite itself has no failing
node.

Watt was not needed for Task 8B.3-M1A.2C-D1 (no downloads, no transfers).
