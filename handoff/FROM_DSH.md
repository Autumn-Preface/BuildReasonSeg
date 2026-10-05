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

# FROM_DSH — Task 8B.3-REF01-E3C0-R2-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3C0-R2-R2` |
| Status | **STOP** (provided corrected script failed its own gate; nothing repaired or re-run) |
| Branch | `audit/task8b3-ref01-locked-replay-artifacts` |
| Starting HEAD | `9faaf37552a39cab241623fb24b911b8706dfc5c` |
| Provided script | `C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py` |
| Prescribed / observed SHA256 | `6edfb02cd97b7204ad9995a5af2dc4566a05d69cdb463a9cb62ca9a7213d3425` / `6edfb02cd97b7204ad9995a5af2dc4566a05d69cdb463a9cb62ca9a7213d3425` (match = True) |
| Script runs / exit | 1 / 1 |
| Readiness / NEXT determination | NONE (the script produced no verdict) |
| Broad search / inference / proposal regeneration / actual replay | NONE |
| External write / product modification | NO / NO |
| Changed paths in this commit | ['handoff/TO_DSH.md'] |
| Next action | Awaiting ChatGPT audit; no rerun and no self-repair was attempted |

Watt was not needed for Task 8B.3-REF01-E3C0-R2-R2 (no downloads, no transfers).

Failure tail reported by the provided script:

```text
  File "C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py", line 747, in <module>
    main()
  File "C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py", line 263, in main
    records = normalize_proposals(proposals_obj)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py", line 84, in normalize_proposals
    raise AssertionError(
AssertionError: AMBIGUOUS_PROPOSALS_TOP_LEVEL: keys=[]
```
