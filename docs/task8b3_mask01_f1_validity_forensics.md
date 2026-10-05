# Task MASK01_F1_VALIDITY_FORENSICS — SUCCESS Validity Source Audit & Historical Artifact Forensics

## 1. Git identity

```text
repository      = C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
starting branch = audit/task8b3-ref01-locked-replay-artifacts
starting head   = 43c24de59625dfb97dcc605435fa83b88be46035
task branch     = audit/task8b3-mask01-validity-forensics
initial status  = ['M handoff/TO_DSH.md']
```

## 2. Scope and prohibitions

Read-only audit of the current SUCCESS/mask-validity logic and of historical artifacts. No product source was modified,
no model inference was executed, and no validity repair was implemented. No threshold or implementation plan is
proposed here.

## 3. Exact current SUCCESS contract

```text
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\pipeline.py:93: return self.status == "SUCCESS"
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\pipeline.py:149: "status": "SUCCESS", "mode": "inspect",
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\pipeline.py:164: return PipelineResult(status="SUCCESS", image=request.image, result_payload=payload,
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\pipeline.py:312: payload.update({"mask_area": int(mask_full.sum()), "status": "SUCCESS",
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\pipeline.py:334: return PipelineResult(status="SUCCESS", image=request.image, result_payload=payload,
delivery_src\BuildReasonSeg_Advisor_RC1\predict.py:267: print(f"[{index}/{len(files)}] {path.name} ... SUCCESS")
```

## 4. Exact current failure-return paths

```text
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\cli\common.py:46: raise BuildReasonSegError(code, detail=f"{what}不存在: {path}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\cli\common.py:52: raise NotInTask8AError(detail=f"{what}：{NOT_IMPLEMENTED_STAGE}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\inference\contract.py:50: raise BuildReasonSegError("E201", detail=f"影像不存在: {self.image}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\inference\contract.py:52: raise BuildReasonSegError("E203", detail=f"不支持的影像类型: {self.image.suffix}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\inference\contract.py:54: raise BuildReasonSegError("E201", detail=f"输入目录不存在: {self.input_dir}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:87: raise BuildReasonSegError("E101", detail="指令为空。")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:90: raise BuildReasonSegError("E302", detail=f"Qwen 前端不可用: {reason}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:91: raise NotInTask8AError(
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:112: raise ValueError(f"fallback not permitted for reason {reason!r}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:120: raise BuildReasonSegError("E101", detail="指令为空。")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\registry.py:51: raise ValueError(f"unknown parse source: {self.source!r}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:32: raise BuildReasonSegError("E502", detail=f"无法导入 PyYAML: {error}") from error
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:51: raise BuildReasonSegError("E301", detail=f"模型包目录不存在: {root}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:56: raise BuildReasonSegError("E302", detail=f"模型包缺少文件: {', '.join(missing)}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:103: raise HashMismatchError(detail=f"校验失败的权重: {', '.join(broken)}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:189: raise HashMismatchError(detail=f"模型 {requested} 权重校验失败")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\_frozen\mvp\box_query.py:83: raise RuntimeError(f"{token} did not become a single token: {ids}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\_frozen\mvp\box_query.py:184: raise RuntimeError(f"expected exactly one [SEG] token, found {seg_positions.numel()}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\_frozen\mvp\box_query.py:187: raise RuntimeError("the [BOX] query token must appear exactly once")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\_frozen\mvp\box_query.py:189: raise RuntimeError("the [BOX] query must precede the reasoning target")
```

## 5. Existing mask-validity gates

```text
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py:435: if proposal.mask_area <= 0 or not proposal.mask_crop.any():
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py:467: and proposal.mask_crop.any()
```

## 6. Padding/context forensic result

```text
helper                    = ['delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\box_query.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\program_parser.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\prompt_diagnostics.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\qwen_seg.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6n_relation_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6p_reference_head.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6u_reference_ranker.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6w_proposal_quality.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6y_nearest_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6z_l3_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task7d_global_competition_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\context.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\detector.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\imageio.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\outputs.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\pipeline.py']
context padding semantics = see audited lines below
test evidence             = ['delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\conftest.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_cli_contract.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_language_contract.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_model_package.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_paths_and_package.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_setup_checker.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_task8b1_fallback_ux.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\tests\\test_task8b_runtime.py']
conclusion                = padding/context handling exists in the audited sources; this task asserted nothing about
                            its mask-validity interaction and changed nothing
```

```text
pad_shape = list(value.shape)
pad_shape[-1] = total_length - prompt_length
pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
extended_extra[key] = torch.cat([value, pad], dim=-1)
pad_shape = list(value.shape)
pad_shape[-1] = total_length - prompt_length
pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
extended_extra[key] = torch.cat([value, pad], dim=-1)
pad_shape = list(value.shape)
pad_shape[-1] = 1
pad = torch.zeros(pad_shape, dtype=value.dtype)
inputs[key] = torch.cat([value, pad], dim=-1)
padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id))
padded_mask = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
return ProgramBatch(input_ids=padded_ids, attention_mask=padded_mask, labels=labels)
```

## 7. Existing test coverage

```text
delivery_src\BuildReasonSeg_Advisor_RC1\tests\conftest.py: 
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_cli_contract.py: test_each_cli_help_works, test_predict_help_lists_frozen_arguments, test_predict_image_and_input_dir_are_exclusive, test_predict_prompt_required_for_normal_inference, test_predict_inspect_proposals_does_not_require_prompt, test_predict_defaults_exact
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_language_contract.py: test_normal_path_is_qwen_first, test_supported_registry_exactly_four_l3_programs, test_smallest_programs_are_not_supported, test_l2_program_is_unsupported_but_suggested, test_supported_program_passes_validator, test_suggestion_must_pass_validator
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_model_package.py: test_decoder_hash_and_bytes_exact, test_detector_hash_and_bytes_exact, test_package_verification_matches, test_model_yaml_parses, test_metadata_fields_complete, test_metadata_hashes_match_copied_assets
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_paths_and_package.py: test_delivery_root_resolution, test_required_structure, test_runtime_paths_are_project_relative, test_no_workspace_reference_in_code, test_no_symlink_dependency, test_moving_root_keeps_config_and_model_paths
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_setup_checker.py: test_good_fixture_is_ready, test_missing_decoder_nonzero, test_hash_mismatch_nonzero, test_missing_qwen_component_nonzero, test_missing_sam2_nonzero, test_invalid_model_yaml_nonzero
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b1_fallback_ux.py: test_valid_suggestion_parsed, test_no_safe_suggestion_parsed, test_program_outside_four_is_rejected, test_free_text_is_not_keyword_routed, test_fenced_json_tolerated, test_generation_config_frozen
delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py: test_uint8_rgb, test_uint16_rgb_fixed_linear_scaling, test_rgba_drops_alpha, test_grayscale_rejected, test_two_and_six_channel_rejected, test_extension_and_missing_file
```

## 8. Bounded historical-artifact search

```text
bounded search inputs = ['docs', 'evaluation', 'handoff']
explicit paths found  = 799
availability          = PARTIAL
```

## 9. Historical case evidence

| referenced path | referenced from | exists |
|---|---|---|
| `artifacts/task6k/` | `docs\architecture_decisions.md` | True |
| `artifacts/whu_native_vector/` | `docs\architecture_decisions.md` | True |
| `logs/windows/08-2024/README.md` | `docs\research\_raw\environment.md` | False |
| `logs/windows/08-2024/README.md].` | `docs\research\_raw\environment.md` | False |
| `logs/windows/08-2024/README.md` | `docs\research\task5_5_sources.md` | False |
| `artifacts/checkpoints/task6b/best_joint.pt` | `docs\task6b_minitrain.md` | True |
| `artifacts/checkpoints/task6b/` | `docs\task6b_minitrain.md` | True |
| `artifacts/task6c5_contaminated_variants.json` | `docs\task6c5_training_optimization.md` | True |
| `artifacts/task6d1_hidden/` | `docs\task6d1_corrective_grounding_audit.md` | True |
| `artifacts/checkpoints/task6c/task6d_G0/G0_epoch2.pt` | `docs\task6d_spatial_grounding_bridge.md` | False |

## 10. Failure taxonomy

```text
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\cli\common.py:46: raise BuildReasonSegError(code, detail=f"{what}不存在: {path}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\cli\common.py:52: raise NotInTask8AError(detail=f"{what}：{NOT_IMPLEMENTED_STAGE}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\inference\contract.py:50: raise BuildReasonSegError("E201", detail=f"影像不存在: {self.image}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\inference\contract.py:52: raise BuildReasonSegError("E203", detail=f"不支持的影像类型: {self.image.suffix}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\inference\contract.py:54: raise BuildReasonSegError("E201", detail=f"输入目录不存在: {self.input_dir}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:87: raise BuildReasonSegError("E101", detail="指令为空。")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:90: raise BuildReasonSegError("E302", detail=f"Qwen 前端不可用: {reason}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:91: raise NotInTask8AError(
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:112: raise ValueError(f"fallback not permitted for reason {reason!r}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\frontend.py:120: raise BuildReasonSegError("E101", detail="指令为空。")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\language\registry.py:51: raise ValueError(f"unknown parse source: {self.source!r}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:32: raise BuildReasonSegError("E502", detail=f"无法导入 PyYAML: {error}") from error
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:51: raise BuildReasonSegError("E301", detail=f"模型包目录不存在: {root}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:56: raise BuildReasonSegError("E302", detail=f"模型包缺少文件: {', '.join(missing)}")
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\models\package.py:103: raise HashMismatchError(detail=f"校验失败的权重: {', '.join(broken)}")
```

## 11. Observable-signal inventory

```text
signals = ['delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\detector.py']
segmenter config = min_mask_pixels: NOT_FOUND_IN_AUDITED_CONFIG · min_mask_frac: NOT_FOUND_IN_AUDITED_CONFIG · provenance ['delivery_src\\BuildReasonSeg_Advisor_RC1\\configs\\inference.yaml']
```

## 12. Runtime validity vs semantic correctness

```text
The audited gates establish runtime validity only: a proposal is admitted when its mask is non-empty and its extent
ratio is within the frozen cap. Nothing in the audited code asserts that an admitted mask is semantically the queried
reference, so runtime validity must not be read as semantic correctness.
```

## 13. Evidence gaps

```text
historical referenced artifacts availability = PARTIAL
```

## 14. What F1 does NOT establish

```text
F1 does not establish the root cause of any mask-validity defect, does not measure semantic correctness, does not
evaluate any threshold, and does not propose or implement a validity repair.
```

## 15. Return-to-ChatGPT disposition

```text
repair_decision = DEFER_TO_CHATGPT
next_gate       = CHATGPT_MASK01_DESIGN_REVIEW
task_status     = COMPLETE_WITH_EVIDENCE_GAPS
```
