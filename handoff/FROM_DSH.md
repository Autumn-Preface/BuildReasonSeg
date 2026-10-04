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

# FROM_DSH — Task 8B.3-P1D11A-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11A-R3` |
| Status | **COMPLETE** (candidate-ID and input-domain wording corrections) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `e0e0a74760d619db3cdd6be62c4ea42520f802ec` |
| Left candidate ID typo | **CORRECTED** → `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` |
| Locked right candidate | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` |
| Locked left candidate | `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` |
| Locked above candidate | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` |
| Locked below candidate | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` |
| Other candidate identities | UNCHANGED |
| README input wording | `正式输入域` removed; now `软件接受的输入模态：RGB 光学影像` |
| Model-card input wording | **SOFTWARE_INPUT_MODALITY** (`- 软件输入模态：**RGB 光学遥感影像** …`) |
| A2 wording | unchanged and evidence-bounded (NOT ESTABLISHED · documented persistent non-detection · not claimed out-of-domain) |
| Pre-edit canonical manifest | **135/135 PASS** |
| Post-edit canonical manifest | **135/135 PASS** (135 entries, path list identical) |
| Manifest entries updated | exactly two: `README.md`, `docs/model_card.md` (canonical blob identities) |
| Model / runtime / tests / external changes | NONE |
| Frozen metrics / architecture / locked candidates | UNCHANGED |
| PROP-01 status | OPEN (not closed) |
| Report | `docs/task8b3_p1d11a_policy_implementation.md` (also `docs/task8b3_p1d11_supported_domain_policy.md`) |
| Next action | Awaiting ChatGPT audit; do not sync external or run locked candidates |

Watt was not needed for Task 8B.3-P1D11A-R3 (no downloads, no transfers).

Only documentation, the two canonical manifest identities and this handoff changed.
