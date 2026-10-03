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

# FROM_DSH — Task 8B.3-M1A.1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-M1A.1` |
| Status | **PARTIAL / STOP** (budget-limited; no functional canonical change) |
| Branch | `fix/task8b3-mem01-compact-proposals` @ starting HEAD `8fc71ca49797112cd0a551faadf27ad7e1de29f1` |
| Compact representation | NOT IMPLEMENTED (`global_mask` field still present; full-frame `np.zeros((H,W))` unchanged) |
| Duplicate IoU | still full-frame (`iou_of`) |
| core.py / outputs.py adaptation | NOT IMPLEMENTED |
| Dedicated runtime tests (single gate) | NOT RUN |
| Full canonical suite | NOT RUN (excluded by this task book) |
| `source_manifest.json` | NOT UPDATED (no canonical content change) |
| External delivery modified | NO |
| Real inference executed | NO |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| MEM-01 | implementation still pending |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | execution budget exhausted before the required read of the current function bodies; the coordinated four-file refactor plus its seven test groups was therefore not started, and no file was modified to avoid leaving a broken canonical tree |
| Next action | Awaiting ChatGPT audit. |

The M1A §4 dependency inventory (13 executable `global_mask` sites across `detector.py`, `core.py`, `outputs.py` and
`tests/test_task8b_runtime.py`) remains the complete work list. No detector parameter, tiling, IoU threshold, winner
policy, stable ID, Reference eligibility/selection, reasoning context, language, SAM2, D-B1 or SUCCESS-validity
semantics was changed; no manifest update, canonical suite run, external-delivery sync/edits, real predict, Demo run,
package change, training or final-test access occurred.

Watt was not needed for Task 8B.3-M1A.1 (no downloads, no transfers).
