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

# FROM_DSH — Task 8B.3-REF01-F1-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R2` |
| Status | **STOP** (forensic script executed but failed; no classification asserted) |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `141af9cfd943dd13366947ea3a31637a3f8c7bc9` |
| Proposal mask source | RERUN_GLOBALPROPOSAL_MASK_CROP (per the instruction) |
| proposals.json role | METADATA_REPRODUCTION_ONLY |
| Coverage threshold | 0.50 (TASK7F_FROZEN) |
| Forensic script | `scripts/task8b3_ref01_locked_reference_forensics.py` (written, executed, failed) |
| Evidence json | NOT PRODUCED |
| Detector calls consumed | NOT DETERMINISTICALLY ESTABLISHED (0–4 possible); treat the allowance as exhausted and re-run all four in one invocation |
| Failure text | captured on a UTF-8 re-execution and recorded in report §10 |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NOT EXECUTED |
| Manual visual inspection / candidate replacement / repair | NO / NO / NO |
| External delivery / canonical RC1 modified | NO / NO |
| Classification / outcome enum | NOT ASSERTED |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_INCOMPLETE |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| STOP reason | the forensic script failed before producing evidence, and my runner's first error-capture attempt crashed on the GBK console; the underlying failure text is now recorded in the report for the next attempt |
| Next action | Awaiting ChatGPT audit; the next task should re-run the script (allowance treated as exhausted) after the recorded failure is addressed |

Watt was not needed for Task 8B.3-REF01-F1-R2 (no downloads, no transfers).

No delivery file was modified and no Qwen/SAM2/D-B1/relation/target-segmentation stage ran.
