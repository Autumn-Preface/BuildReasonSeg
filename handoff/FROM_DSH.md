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

# FROM_DSH — Task 8B.3-D1.1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git history._

| item | value |
|---|---|
| Task | 8B.3-D1.1 |
| Status | COMPLETE (large-image memory-forensics correction; docs and handoff only) |
| Branch | eval/task8b3-six-image-demo-suite |
| Starting HEAD | 4bcd26b5403bcdaf3ee73f9855c1ffca8db5ac63 |
| Inference executed | NO |
| Delivery modified | NO |
| Product/harness/tests modified | NO |
| MEM-01 | CONFIRMED |
| 5000x5000 bool payload | 25,000,000 B = 23.84 MiB per full-frame bool array |
| Retained proposal masks | one full-frame bool mask per retained raw proposal (detector.py:265), stored in the accumulated entries until merge; GlobalProposal.global_mask reuses that object with no constructor copy |
| Pairwise merge temporaries | iou_of() performs np.logical_and and np.logical_or over two full-frame masks, so each comparison also creates 23.84 MiB bool temporaries for 5000x5000, sequentially, up to N(N-1)/2 comparisons |
| Exception signature | EXACT_SIGNATURE_MATCH (shape, dtype and size only) |
| Unique throwing allocation site | NOT CONFIRMED FROM EXISTING ARTIFACTS |
| Failure threshold | NOT ESTABLISHED |
| Output-layout proposal | STILL DEFERRED TO TASK 8B.4 |
| Report | docs/task8b3_d1_demo_failure_forensics.md |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit. |

Accepted D1 findings preserved unchanged: A1/A3/A4 Reference is the largest eligible proposal; user-facing largest building differs from the implementation semantics of largest eligible detected proposal; A1/A4 SUCCESS gates are only non-empty, non-padding and directional-centroid; A2 is a detector/proposal-stage zero-proposal failure with NMS causality NOT CONFIRMED; the frozen six-image baseline remains runtime 3/6 and manual end-to-end semantic 0/6 (qualitative six-sample Demo audit, not a research metric). The unsupported generalization that all inputs above 512 px are blocked has been removed: the failure is a blocker for the demonstrated 5000x5000 B1/B2 path and the threshold is not established.

Watt was not needed for Task 8B.3-D1.1 (no downloads, no transfers).

No inference, no delivery modification and no source, test or harness change occurred in this task.