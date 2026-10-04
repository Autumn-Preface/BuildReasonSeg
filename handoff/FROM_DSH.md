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

# FROM_DSH — Task 8B.3-P1D11S1-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11S1-R3` |
| Status | **PARTIAL / STOP** (four missing edits applied) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `3d491534a9dacb29e1ef51cc126e69013535c7e9` |
| `subprocess` import | PRESENT (edit A) |
| Test `GIT_CANONICAL_BASIS` symbol | PRESENT — `= sync.GIT_CANONICAL_BASIS` (edit B) |
| Static patch gate | PASS |
| Git diff gate | PASS |
| Edits C / D | appended (`..._identity_mismatch_blocks_write`, `..._unsupported_identity_basis_rejected`) |
| Existing real-manifest test | unchanged (already Git-canonical) |
| Dedicated pytest | exit 0 · **28 passed** · failed nodes: none |
| Read-only external check | exit 1 · {'checked': 135, 'match': 97, 'missing': 0, 'mismatch': 38} |
| Check mismatches | ['README.md', 'buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md', 'buildreasonseg/runtime/_frozen/__init__.py', 'buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py', 'buildreasonseg/runtime/_frozen/mvp/grcl_directional.py', 'buildreasonseg/runtime/_frozen/mvp/native_vector_adapter.py', 'buildreasonseg/runtime/_frozen/mvp/task6m_eval.py', 'buildreasonseg/runtime/_frozen/mvp/task6m_structured.py', 'buildreasonseg/runtime/_frozen/mvp/task6n_relation_decoder.py', 'buildreasonseg/runtime/_frozen/mvp/task6p_reference_head.py', 'buildreasonseg/runtime/_frozen/mvp/task6q_reference_resolver.py', 'buildreasonseg/runtime/_frozen/mvp/task6s_directional_pipeline.py', 'buildreasonseg/runtime/_frozen/mvp/task6u_common.py', 'buildreasonseg/runtime/_frozen/mvp/task6v_family_reference_resolver.py', 'buildreasonseg/runtime/_frozen/mvp/task6w_quality_reference_resolver.py', 'buildreasonseg/runtime/_frozen/mvp/task6x_sam2_reference_refiner.py', 'buildreasonseg/runtime/_frozen/mvp/task6z_field_composition.py', 'buildreasonseg/runtime/_frozen/mvp/task6z_l3_decoder.py', 'buildreasonseg/runtime/_frozen/mvp/task7a_l3_pipeline.py', 'buildreasonseg/runtime/_frozen/mvp/task7d_data.py', 'buildreasonseg/runtime/_frozen/mvp/task7e_l3_decoder_adapter.py', 'buildreasonseg/runtime/_frozen/mvp/whu_vector_audit.py', 'buildreasonseg/runtime/pipeline.py', 'check_setup.py', 'docs/command_grammar.md', 'docs/model_card.md', 'docs/runtime_mapping.md', 'environment.yml', 'model/buildreasonseg_advisor/metadata.json', 'model/buildreasonseg_advisor/model.yaml', 'model/components/program_head/qwen_asset_manifest.json', 'predict.py', 'requirements.txt', 'tests/test_cli_contract.py', 'tests/test_language_contract.py', 'tests/test_model_package.py', 'tests/test_setup_checker.py', 'tests/test_task8b1_fallback_ux.py'] (expected exactly `README.md`, `docs/model_card.md`) |
| Canonical RC1 / external RC1 written | NO / NO |
| Model inference / locked candidate runs | NONE / NONE |
| Frozen metrics / architecture / locked candidates | UNCHANGED |
| PROP-01 status | OPEN (not closed) |
| Outcome | **RC1_SYNC_HELPER_ALIGNMENT_BLOCKED** |
| Next gate (recommended, not executed) | `NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY` |
| Report | `docs/task8b3_p1d11s1_git_canonical_sync_helper.md` |
| Next action | Awaiting ChatGPT audit; the retry sync needs its own task book |

Watt was not needed for Task 8B.3-P1D11S1-R3 (no downloads, no transfers).

Only the helper, the dedicated test file, this report and the handoff changed; no canonical or external delivery file
was written and no model or locked candidate was run.
