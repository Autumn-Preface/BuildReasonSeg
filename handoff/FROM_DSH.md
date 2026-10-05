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

# FROM_DSH — Task 8B.3-REF01-E3C0-R2-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3C0-R2-R3` |
| Status | **STOP** (provided schema probe failed its own gate; nothing repaired or re-run) |
| Branch | `audit/task8b3-ref01-locked-replay-artifacts` |
| Starting HEAD | `fa9ede22f9bee640a6211f0dfd807439d0e8bc57` |
| Provided script | `C:\D\DeepSeekHarness\E3C0_R2_R3_schema_probe.py` |
| Prescribed / observed SHA256 | `49e51459dd9bf218f42fc62021f2c15dd223991a2ede797e42c0519f44934873` / `49e51459dd9bf218f42fc62021f2c15dd223991a2ede797e42c0519f44934873` (match = True) |
| Script runs / exit | 1 / 1 |
| Schema captured | NONE (the probe produced no output file) |
| Broad search / inference / proposal regeneration / selector replay / readiness determination | NONE |
| External write / product modification | NO / NO |
| Changed paths in this commit | ['docs/task8b3_ref01_e3c0_r2r3_proposals_schema_probe.md', 'evaluation/task8b3_ref01_e3c0_r2r3_proposals_schema_probe.json', 'handoff/FROM_DSH.md', 'handoff/TO_DSH.md'] |
| Next action | Awaiting ChatGPT audit; no rerun and no self-repair was attempted |

Watt was not needed for Task 8B.3-REF01-E3C0-R2-R3 (no downloads, no transfers).

Failure tail reported by the provided probe:

```text
Traceback (most recent call last):
  File "C:\D\DeepSeekHarness\E3C0_R2_R3_schema_probe.py", line 254, in <module>
    main()
  File "C:\D\DeepSeekHarness\E3C0_R2_R3_schema_probe.py", line 237, in main
    assert changed == ALLOWED_DIFF, f"BAD_DIFF: {sorted(changed)}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: BAD_DIFF: ['handoff/FROM_DSH.md', 'handoff/TO_DSH.md']
```
