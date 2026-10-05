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

# FROM_DSH — Task 8B.3-REF01-E3C0-R2-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3C0-R2-R1` |
| Status | **STOP** (provided recovery script failed its own gate; nothing repaired or re-run) |
| Branch | `audit/task8b3-ref01-locked-replay-artifacts` |
| Starting HEAD | `9e06b9e231be34d32983bfc12e394d629ed59366` |
| Provided script | `C:\D\DeepSeekHarness\E3C0_R2_R1_audit_exact.py` |
| Prescribed / observed SHA256 | `c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c` / `c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c` (match = True) |
| Script runs | 1 |
| Script exit | 1 |
| Readyness / NEXT determination | NONE (the script did not produce a verdict) |
| Broad search / inference / proposal regeneration / actual replay | NONE |
| External write / product modification | NO / NO |
| Changed paths in this commit | ['handoff/TO_DSH.md'] |
| Next action | Awaiting ChatGPT audit; no rerun and no self-repair was attempted |

Watt was not needed for Task 8B.3-REF01-E3C0-R2-R1 (no downloads, no transfers).

Failure tail reported by the provided script:

```text
Traceback (most recent call last):
  File "C:\D\DeepSeekHarness\E3C0_R2_R1_audit_exact.py", line 744, in <module>
    main()
  File "C:\D\DeepSeekHarness\E3C0_R2_R1_audit_exact.py", line 215, in main
    assert all(line in allowed_status for line in status_lines), (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: UNEXPECTED_WORKTREE: ['M handoff/TO_DSH.md']
```
