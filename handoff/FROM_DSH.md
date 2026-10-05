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

# FROM_DSH — Task 8B.3-REF01-E3B2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3B2` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-eligibility-repair-sync` |
| Starting HEAD | `300cdad5629b945ffe10480351081f9c8befe7f4` |
| Sync policy audit | 135 entries · weights/runs/logs/inference outputs excluded = True |
| Pre-sync external check | exit 1 · {'checked': 135, 'match': 133, 'missing': 0, 'mismatch': 2} · mismatch paths ['buildreasonseg/runtime/detector.py', 'tests/test_task8b_runtime.py'] |
| Controlled sync (exactly once) | exit 0 · copied 135 · verified 135 · failures 0 |
| Post-sync external check | exit 0 · checked 135 · match 135 · missing 0 · mismatch 0 |
| Canonical manifest Git-object sync | YES (`5c175a9ced591210...`) |
| External `check_setup.py` | exit 0 · READY True |
| Targeted regression | exit 0 · 40 passed in 1.13s |
| External full delivery suite | exit 0 · 124 passed in 112.81s (0:01:52) |
| Canonical product / manifest / helper modified | NO |
| Dependencies installed / model inference | NONE / NONE |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| left/below reference-selection defects | UNRESOLVED |
| Evidence | `evaluation\task8b3_ref01_e3b2_external_sync_validation.json` |
| Report | `docs/task8b3_ref01_eligibility_repair_impl.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-E3B2 (no downloads, no transfers).

The external delivery received exactly the 135 manifest-listed source/config files plus the canonical control manifest;
no model weight, run, log or inference output was copied into it, and the external suites ran without installing
anything or performing model inference.
