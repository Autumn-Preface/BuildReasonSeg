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

# FROM_DSH — Task 8B.3-REF01-E3B1-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

```text
Task: 8B.3-REF01-E3B1-R3
Status: COMPLETE
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: 3004cb7e65c73a9cdc005aaa5978876cfed263db
Canonical base: f50404843f5189986f97633cd0edb6140b1d8034
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Manifest changed in R3: NO
Product/test/sync-helper changed in R3: NO
Identity basis: GIT_CANONICAL_BLOB_BYTES
Detector Git-canonical bytes: 21257
Detector Git-canonical SHA256: 934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883
Test Git-canonical bytes: 28128
Test Git-canonical SHA256: 71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b
Manifest file count: 135
Manifest semantic correction: PASS
Manifest identities: PASS 135/135
External pre-sync check: 133 match / 0 missing / 2 mismatch
External mismatch paths: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External check rerun in R3: NO
External write performed: NO
External sync performed: NO
pytest run in R3: NO
py_compile run in R3: NO
Evidence: evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
Report: docs/task8b3_ref01_e3b1_manifest_canonicalization.md
Outcome: REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
Next gate: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects: UNRESOLVED
Next action: Awaiting ChatGPT audit; do not execute E3B2/NEXT.
```

Watt was not needed for Task 8B.3-REF01-E3B1-R3 (no downloads, no transfers).

This handoff normalization changed only the active engineering handoff section of `handoff/FROM_DSH.md`; the
ARTIFACT-FACTS block above is preserved exactly, and no other repository file was modified.
