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

# FROM_DSH — Task 8B.2-R1.1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.2-R1.1` |
| Branch | `audit/task8b2-rc1-runtime-closure` |
| Starting HEAD | `77d98223c7a91b22980be621d6e6b975a54daefd` |
| Canonical manifest entries | 135 |
| Third-party downloaded assets in canonical source | NONE for the frozen cleanup set (9 Qwen downloaded text assets + 1 SAM2 downloaded config removed) |
| Generated qwen integrity cache in canonical source | ABSENT |
| `qwen_asset_manifest.json` | PRESENT |
| External delivery check | 135/135 PASS (read-only; external runnable delivery unmodified) |
| Dedicated tests | 23 passed (`tests/test_sync_advisor_rc1_delivery.py`) |
| Repository suite | 1534 passed, 1 skipped |
| RC1-ENV-01 | DEFERRED |
| Next action | Awaiting ChatGPT audit. Do not start Task 8B.2-R2. |

Canonical-source policy cleanup only: 135 manifest entries, 11 canonical-only files removed, no RC1 runtime
file edited, no package installed, no inference run, no checkpoint or final-test access.

Repair note: R1's handoff rewrite had dropped the repository-required `ARTIFACT-FACTS` block; it was restored
verbatim from the last committed copy together with the required Watt reference (R1's suite had run before that
write). No research content was changed.

Watt was not needed for Task 8B.2-R1.1 (no downloads, no transfers); the pre-existing Watt instance, when present,
remains transport-only and is not owned by this project.
