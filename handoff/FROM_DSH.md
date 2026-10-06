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

# FROM_DSH — Task MASK01_D1_R1B_CANONICAL_DOCS Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1B_CANONICAL_DOCS` |
| Status | **STOP** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1a-r6-tests` / `385a3ba5d87c22712c75e8a22273d5720bd9cfe8` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1b` |
| Documents updated | `README.md`, `docs/model_card.md`, `docs/runtime_mapping.md`, `inference/README.md` |
| Marker validation | False (three markers × four files) |
| SUCCESS semantics | `RUNTIME_STRUCTURAL_ONLY` · `NOT_EVALUATED` · semantic target correctness is not established |
| product source / tests / manifest | UNCHANGED |
| pytest / model inference / training | NOT run |
| inspect-proposals 30-vs-20 | NOT fixed |
| Evidence | `evaluation\task8b3_mask01_d1_r1b_canonical_docs.json` |
| Report | `docs\task8b3_mask01_d1_r1b_canonical_docs.md` |
| Next action | Awaiting ChatGPT remote review; R1-C not entered |

Watt was not needed for Task MASK01_D1_R1B_CANONICAL_DOCS (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
