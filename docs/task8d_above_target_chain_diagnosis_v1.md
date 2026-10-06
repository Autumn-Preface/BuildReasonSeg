# Task 8D ABOVE Target Chain Diagnosis V1

Status: READY_FOR_SUPERVISOR_AUDIT. Diagnosis only; no repair. Next gate: CHATGPT_TASK8D_ABOVE_CHAIN_DIAGNOSIS_REMOTE_AUDIT.

## 1. Question being diagnosed

At what earliest observable stage does native GT target 6 lose numerically to wrong instance 7? Fixed order: W -> A -> C -> logits -> probability -> final mask. No pass threshold or root-cause claim is introduced.

## 2. Why ABOVE is the clean case

Sample buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314; program largest_to_above_to_nearest. Reference best-overlap identity is GT 4, IoU 0.9097432024169184. Frozen target best overlap is GT 7, IoU 0.48158096699923253; canonical target 6 IoU 0.07762201453790239 / Dice 0.14406167188629246. This reduces the reference-identity confounder; reference shape remains imperfect.

## 3. Frozen evidence / no-rerun guarantee

Task8C accepted HEAD: 764a805ef7d9651bfffd27dfaeb5808e8c21df1f. Frozen runtime SHA256: 2fcbb705acf1ceeb28fc9609bfda869150e7b800b2bea0d9fba9a723c12ff8ca. All ABOVE 15 artifacts/transcript and Task8C 10 repository evidence/harness files are SHA-locked. Saved maps are read-only primary evidence and none of the seven automatic maps is recomputed. Model/detector/predict/checkpoint calls and external writes are zero. Original Task8C formal runner remains one invocation / four attempts / zero retries. Pre/post identities and read-only audit are in JSON.

Dedicated offline tests: 41/41 PASS, exit0. Git-canonical source/config helper: checked135/match135/missing0/mismatch0; external source_manifest byte-identical. All2184 external files match preflight bytes/SHA256/mtime/file set; additions/deletions0. Input, all historical output, protected assets, native truth, all four Task8C TIFFs and case artifacts, and Task8C repository evidence are unchanged. The grandfathered Ultralytics/settings.json (606bytes; SHA256 1dae32b8abfc0f084cdee6c446dcf5420031bfcf005fb358f473a9a34d368cdf) remains unchanged. Nine diagnostic PNGs were individually reviewed; values/statistics remain frozen. Final interpretation/acceptance is reserved for ChatGPT Supervisor.

| Map | Shape | Finite | Min | Max | Sum |
| --- | --- | --- | --- | --- | --- |
| P_dir | [64, 64] | True | 1.34083876e-14 | 1 | 1623.23154 |
| P_near | [64, 64] | True | 0 | 0.959506035 | 314.416462 |
| W | [64, 64] | True | 0 | 0.940226555 | 88.8972307 |
| A | [64, 64] | True | 0 | 0.0105765564 | 0.999999983 |
| C | [64, 64] | True | -0.105205163 | 0.861314476 | 2070.56395 |
| logits | [512, 512] | True | -18.4960938 | 16.5617676 | -2638935.72 |
| probability | [512, 512] | True | 9.27360411e-09 | 0.999999881 | 3321.60266 |

## 4. Actual target mechanism in RC1

**W**: clamp(P_dir_64 * P_near_64, 0, 1)

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:165` (field_competition, end line 173).

**A**: deterministic W/(spatial sum W+EPS); not learned; no score head for D-B1

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:165` (field_competition, end line 173); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:263` (GlobalCompetitionDecoder.input_spec, end line 273).

**q**: sum_i A_i F_i using projected SAM2 features F (256->128 Conv1x1/GroupNorm/GELU)

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:176` (target_prototype, end line 182); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:91` (VisualProjection.__init__, end line 94); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:96` (VisualProjection.forward, end line 97); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:84` (_group_norm, end line 85); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:263` (GlobalCompetitionDecoder.input_spec, end line 273).

**C**: plain normalized feature/prototype cosine with EPS; no learned scale, sigmoid or threshold

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:185` (prototype_similarity, end line 193); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:215` (GlobalCompetitionDecoder.competition, end line 239).

**decoder_inputs**: 148 channels: projected F 128, P_dir 1, P_near 1, direction embedding 16, A_vis=A*4096 1, C 1

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:243` (GlobalCompetitionDecoder.forward, end line 257); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:103` (DirectionEmbedding.__init__, end line 106); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:263` (GlobalCompetitionDecoder.input_spec, end line 273).

**decoder**: dense convolution trunk 148->128->64->1, bilinear 64->512 align_corners=False

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:131` (MaskDecoderTrunk.__init__, end line 137); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:139` (MaskDecoderTrunk.forward, end line 142); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:259` (GlobalCompetitionDecoder.upsampled, end line 261).

**output_threshold**: upsampled_logits > 0.0, sigmoid probability recorded separately

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:135` (Db1Runtime.forward, end line 169).

**target_proposal_selection**: False

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:210` (run_core_chain, end line 241); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:135` (Db1Runtime.forward, end line 169); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:243` (GlobalCompetitionDecoder.forward, end line 257).

**graph_construction**: False

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:210` (run_core_chain, end line 241); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:215` (GlobalCompetitionDecoder.competition, end line 239); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:243` (GlobalCompetitionDecoder.forward, end line 257).

**dense_mask_emitted**: True

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:135` (Db1Runtime.forward, end line 169); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py:243` (GlobalCompetitionDecoder.forward, end line 257).

**target_identity_or_nearest_identity_gate**: False

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:157` (direction_satisfied, end line 171).

**saved_C_and_logits_paths**: Saved C is from the separately computed projected-feature competition state; logits use a second model forward under CUDA bfloat16 autocast when device is not cpu. Equality of those internal states is not established by saved maps.

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:135` (Db1Runtime.forward, end line 169).

**scope**: D-B1 selected in core.Db1Runtime.load; other variants in source are not the active runtime.

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py:106` (Db1Runtime.load, end line 129).

Source traces (full exact snippets and SHA identities also in JSON):

| Exact source path | Function | Line / end line |
| --- | --- | --- |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py | Db1Runtime.load | 106 / 129 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py | Db1Runtime.forward | 135 / 169 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py | run_core_chain | 210 / 241 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py | context_to_global | 134 / 154 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py | guard_directional_candidates | 116 / 131 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py | directional_candidates | 89 / 107 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py | direction_satisfied | 157 / 171 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py | predict_one | 186 / 358 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py | _non_padding_mask | 373 / 375 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | field_competition | 165 / 173 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | _group_norm | 84 / 85 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | VisualProjection.__init__ | 91 / 94 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | VisualProjection.forward | 96 / 97 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | DirectionEmbedding.__init__ | 103 / 106 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | MaskDecoderTrunk.__init__ | 131 / 137 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | MaskDecoderTrunk.forward | 139 / 142 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | target_prototype | 176 / 182 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | prototype_similarity | 185 / 193 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | GlobalCompetitionDecoder.competition | 215 / 239 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | GlobalCompetitionDecoder.forward | 243 / 257 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | GlobalCompetitionDecoder.upsampled | 259 / 261 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py | GlobalCompetitionDecoder.input_spec | 263 / 273 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/task6z_field_composition.py | record_fields | 97 / 113 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py | geometric_relation_field_v02 | 128 / 180 |
| delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/_frozen/mvp/nearest_boundary_field.py | nearest_boundary_field_512 | 50 / 64 |
| spatial_reasoning/relations.py | evaluate_direction | 209 / 318 |
| spatial_reasoning/geometry.py | boundary_distance | 345 / 407 |
| buildreasonseg_mvp/native_vector_adapter.py | image_record_for_reasoning | 186 / 246 |

Target-stage candidate rankings below are post-hoc GT aggregations of dense maps; they are not a target-proposal selector. No graph is constructed. A_fixed is deterministic, while projected visual features and the dense decoder are learned components. Exact normalization retains source EPS=1e-6; W/sum(W) is shorthand, not a changed rule.

## 5. Reference geometry audit

Frozen origin [left,top]=[-197,-41], size 512. Context GT is exact integer crop/translation with zero in reflected-image padding; no GT reflection or resizing at 512. Canonical ref4, target6 and wrong7 are nonempty and their exact represented areas are shown below. Every GT instance in tile1008 is reported; rankings include every nonempty context intersection, including ref4 and any canonically invalid direction, so no competitor is hidden. Geometry uses native_vector_adapter.image_record_for_reasoning -> geometry.image_geometry_from_record -> relations.evaluate_direction with unchanged config, and geometry.boundary_distance (existing gap convention). Runtime half-plane checks are audited separately.

| GT id | Full GT area | Context area | Coverage | Canonical above valid | Reason | Boundary gap to GT ref4 (px) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1551 | 1551 | 1 | False | wrong_direction | 43.9221549 |
| 2 | 4202 | 4202 | 1 | False | wrong_direction | 62.5609943 |
| 3 | 2298 | 1273 | 0.553959965 | False | wrong_direction | 221.892351 |
| 4 | 5013 | 5013 | 1 | False | wrong_direction | 0 |
| 5 | 2959 | 2959 | 1 | False | axis_not_dominant | 62.031738 |
| 6 | 2571 | 2571 | 1 | True | N/A | 132.330417 |
| 7 | 2281 | 2281 | 1 | True | N/A | 196.253644 |

## 6. P_dir / P_near analysis

64x64 aggregation uses cv2.INTER_AREA fractional occupancy of exact context masks, never a threshold. mass=sum(X*M), mean=mass/sum(M); target6/wrong7 comparison uses their means. Saved arrays are not altered for quantitative analysis.

| ID | P_dir mass | P_dir mean | P_near mass | P_near mean |
| --- | --- | --- | --- | --- |
| 1 | 0.000623707556 | 2.5736482e-05 | 3.20067406 | 0.132071657 |
| 2 | 0.000287287494 | 4.37563056e-06 | 3.58576719 | 0.0546142552 |
| 3 | 5.98050168e-12 | 3.0066937e-13 | 0.0333223516 | 0.00167527926 |
| 4 | 24.2831254 | 0.310017958 | 5.2818015 | 0.0674317367 |
| 5 | 0.114226716 | 0.00247060149 | 2.49750752 | 0.0540184121 |
| 6 | 26.3984717 | 0.657138151 | 0.427270713 | 0.010636066 |
| 7 | 35.640622 | 0.999999915 | 0.0786626875 | 0.00220710741 |

## 7. W / A analysis

Candidate W mass=sum(W*M); attention mass=sum(A*M). All candidates use the same global A normalization. Mass measures total support and can reflect candidate size; means are separately reported. There is no discrete target-instance decision at either stage.

| ID | W mass | W mean | A attention mass |
| --- | --- | --- | --- |
| 1 | 0.000111237404 | 4.59006695e-06 | 1.25130335e-06 |
| 2 | 4.07835298e-05 | 6.21167517e-07 | 4.58771653e-07 |
| 3 | 1.06313078e-14 | 5.34488374e-16 | 1.19590989e-16 |
| 4 | 1.42360172 | 0.0181748475 | 0.0160140166 |
| 5 | 0.0127999438 | 0.000276849072 | 0.000143985853 |
| 6 | 0.342727985 | 0.00853154066 | 0.003855328 |
| 7 | 0.0786626777 | 0.00220710713 | 0.000884872074 |

## 8. C prototype-similarity analysis

C is the saved cosine field from the A-weighted projected visual prototype. The table reports occupancy-weighted C mean, not an instance-selected prototype. F and q are not saved in this bundle, so contributions to q cannot be disentangled without forbidden model execution.

| ID | C mean |
| --- | --- |
| 1 | 0.130185823 |
| 2 | 0.0473299298 |
| 3 | 0.261742257 |
| 4 | 0.124484478 |
| 5 | 0.112672132 |
| 6 | 0.200210813 |
| 7 | 0.276679558 |

## 9. Decoder logits / probability analysis

Exact context GT pixels aggregate frozen 512x512 upsampled logits and saved sigmoid probabilities. Positive-logit fraction uses the existing runtime logits>0 rule only. Different means can rank differently after nonlinear sigmoid; this does not isolate causal decoder inputs.

| ID | Logit mean | Probability mean | Positive-logit fraction |
| --- | --- | --- | --- |
| 1 | -10.8111124 | 5.67913133e-05 | 0 |
| 2 | -10.19576 | 0.00026645551 | 0 |
| 3 | -10.5461831 | 0.000120872319 | 0 |
| 4 | -8.33187679 | 0.00467185916 | 0 |
| 5 | -9.60286975 | 0.000163759908 | 0 |
| 6 | -5.4717268 | 0.136290266 | 0.116297161 |
| 7 | 1.50096387 | 0.54677527 | 0.550197282 |

## 10. Final mask analysis

All IoUs use the untouched frozen 1008_mask.png and exact full native tile masks. The binary context logits>0 mask maps back to exactly the frozen output. No native GT occupancy rounding, morphology, component picking or repainting is applied.

| ID | Final predicted-mask IoU |
| --- | --- |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0.0776220145 |
| 7 | 0.481580967 |

## 11. Target-6 vs wrong-7 trajectory

| Stage / statistic | Target6 | Wrong7 | Delta 6-7 | Favoured |
| --- | --- | --- | --- | --- |
| P_dir / P_dir_mean | 0.657138151 | 0.999999915 | -0.342861764 | 7 |
| P_near / P_near_mean | 0.010636066 | 0.00220710741 | 0.00842895857 | 6 |
| W / W_mass | 0.342727985 | 0.0786626777 | 0.264065307 | 6 |
| A / A_attention_mass | 0.003855328 | 0.000884872074 | 0.00297045593 | 6 |
| C / C_mean | 0.200210813 | 0.276679558 | -0.0764687454 | 7 |
| logits / logit_mean | -5.4717268 | 1.50096387 | -6.97269066 | 7 |
| probability / probability_mean | 0.136290266 | 0.54677527 | -0.410485004 | 7 |
| positive_logit_fraction / positive_logit_fraction | 0.116297161 | 0.550197282 | -0.433900121 | 7 |
| final_mask / final_mask_iou | 0.0776220145 | 0.481580967 | -0.403958952 | 7 |

| Stage | 6 rank | 7 rank | Top5 IDs | All ordered IDs |
| --- | --- | --- | --- | --- |
| W | 2 | 3 | [4, 6, 7, 5, 1] | [4, 6, 7, 5, 1, 2, 3] |
| A | 2 | 3 | [4, 6, 7, 5, 1] | [4, 6, 7, 5, 1, 2, 3] |
| C | 3 | 1 | [7, 3, 6, 1, 4] | [7, 3, 6, 1, 4, 5, 2] |
| logits | 2 | 1 | [7, 6, 4, 5, 2] | [7, 6, 4, 5, 2, 3, 1] |
| probability | 2 | 1 | [7, 6, 4, 2, 5] | [7, 6, 4, 2, 5, 3, 1] |
| final_mask | 2 | 1 | [7, 6, 1, 2, 3] | [7, 6, 1, 2, 3, 4, 5] |

Descending scores; exact ties break by ascending instance ID. Undefined means are excluded and no numeric pass threshold is used. Predeclared statistics were pushed in the evidence-lock checkpoint before aggregation.

## 12. Canonical-reference deterministic counterfactual

Only P_dir, P_near, W and A are recomputed from exact GT ref4 in the same frozen context using the frozen field code. No SAM2/F/q/C/decoder/checkpoint is evaluated. Normalization calls the frozen pure field_competition function and retains EPS; no neural module is constructed. Interpretation: **MIXED / NOT RESOLVED**; not causal proof. No demonstrated restoration from wrong7. Automatic and GT-reference preferences must both be disclosed; no q/C/decoder counterfactual is available.

| GT-ref stage | 6 mass | 7 mass | Delta 6-7 | Favoured |
| --- | --- | --- | --- | --- |
| W | 0.37446623 | 0.0762215393 | 0.29824469 | 6 |
| A | 0.00408140567 | 0.000830758548 | 0.00325064712 | 6 |

| Stage | Auto 6/7 ranks | GT-ref 6/7 ranks | GT-ref top5 |
| --- | --- | --- | --- |
| W | 2/3 | 2/3 | [4, 6, 7, 5, 1] |
| A | 2/3 | 2/3 | [4, 6, 7, 5, 1] |

| Field | Mean signed diff | Mean abs diff | Max abs diff | RMS diff |
| --- | --- | --- | --- | --- |
| P_dir | 0.00111312052 | 0.00133902635 | 0.0310230255 | 0.00497285819 |
| P_near | 0.00174822131 | 0.00560347887 | 0.96336484 | 0.0379697113 |
| W | 0.000696313869 | 0.00150863914 | 0.953074694 | 0.0214532783 |
| A | 1.34475609e-11 | 1.8923755e-05 | 0.0103878109 | 0.000235105667 |

## 13. First observed divergence

{
  "stage": "C",
  "statistic": "C_mean",
  "kind": "FIRST_OBSERVED_DIVERGENCE",
  "actual_top_instance_id": 7,
  "neither6_nor7_is_top": false,
  "delta_6_minus_7": -0.0764687454352842,
  "interpretation": "Earliest fixed-order numerical preference, not a causal proof."
}

This is the earliest observed preference under the fixed aggregation rule. It is not a proven root cause, and the actual top competitor is reported even when neither instance6 nor instance7 leads.

## 14. Runtime SUCCESS guard explanation

**frozen_directional_guard**: {'direction': 'above', 'candidate_count': 5, 'inside_context': 5, 'candidate_ids': [0, 1, 2, 3, 4]}

**direction_availability**: Pre-inference proposals in requested centroid half-plane; fails only if candidates exist and none has centroid inside context. Zero candidates alone does not fail this guard.

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:89` (directional_candidates, end line 107); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:116` (guard_directional_candidates, end line 131).

**final_direction_check**: above: target_centroid_row < selected_reference_centroid_row

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:157` (direction_satisfied, end line 171); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358).

**observed_target_centroid_yx**: [43.403164556962025, 147.4126582278481]

**observed_reference_centroid_yx**: [316.66790825328366, 59.31170358753185]

**observed_final_direction_pass**: True

**nonempty_mask**: True

**padding_check**: context_to_global already crops padding; _non_padding_mask returns all ones. Not an independent semantic gate.

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:134` (context_to_global, end line 154); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:373` (_non_padding_mask, end line 375); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358).

**nearest_target_identity_checked**: False

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:157` (direction_satisfied, end line 171).

**GT_identity_checked**: False

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358).

**runtime_success_scope**: RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:157` (direction_satisfied, end line 171).

**why_wrong_target_can_succeed**: After other pipeline stages complete, any nonempty mapped mask with centroid above the selected reference can pass; neither nearest building nor native GT identity is verified.

Source: `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py:186` (predict_one, end line 358); `delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py:157` (direction_satisfied, end line 171).

Proposal availability and final mask centroid direction do not check nearest identity or GT identity.

## 15. Evidence-supported diagnosis

FIRST_OBSERVED_DIVERGENCE: C / C_mean. Actual top instance 7; target6 rank 3, wrong7 rank 1. Delta6-minus7=-0.0764687454352842. This is the earliest fixed-order observable preference, not a proven root cause.

PRIMARY_EVIDENCE_SUPPORTS: P_dir mean favours7, P_near mean favours6. Frozen W mass favours6 and A mass favours6 over the other member of the 6-vs7 pair; W/A top instance is 4/4. Thus in this locked case a wrong7 preference is not present in the combined W/A comparison before C, but no discrete correct target selection is claimed.

PRIMARY_EVIDENCE_SUPPORTS: C is where the observed 6-vs7 preference reverses. Logit mean (-5.47172679626631 vs 1.5009638657960804), probability mean (0.13629026622631432 vs 0.5467752699385345), positive fraction and final IoU continue to favour7. This is an observed downstream preference; saved C and decoder-forward states are computed separately. It does not prove C caused the decoder mask or isolate F, embedding, attention, C or learned-trunk contributions.

MIXED / NOT RESOLVED: GT-reference deterministic fields still favour6, as automatic-reference W/A already did. W and A deltas increase numerically, but target6/wrong7 ranks remain2/3; there is no preference restoration. Reference-shape residual does not account for an observed W/A reversal in this comparison; its possible effect on q/C/decoder remains untested, not ruled out.

Canonical geometry: both6 and7 satisfy the frozen above predicate. Existing canonical boundary gap to GT ref4 is 132.33041663476493px for6 versus 196.25364381932212px for7. The frozen nearest/proximity field gives greater mean to6; the eventual dense mask overlaps7 more.

Runtime structural SUCCESS permits the semantic error: saved guard reports 5 directional proposals and 5 inside context, final nonempty mask centroid row 43.403164556962025 < selected-reference row 316.66790825328366. Runtime checks this direction and mapped nonemptiness; it does not verify nearest building or native GT identity.

## 16. What remains unresolved

One locked qualitative test example; not a new population metric or generalization result. Task7J metrics, model, seed, architecture and thresholds remain unchanged.

Instance mass depends on covered area; C/logit/probability means measure different quantities. Changes in their numerical preferences do not isolate causal contributions.

F and q and raw 64x64 decoder logits are not saved. C uses the saved competition state while logits follow the runtime forward path; no equality of mixed-precision internal feature states can be independently established from these artifacts.

Reflected context imagery has padding, but GT is defined only on the original tile. Partially represented instances are disclosed; all nonempty GT context intersections are ranked, with reference included.

The deterministic GT-reference counterfactual changes fields only, cannot establish what a model would output, and is not causal proof. No repair or new acceptance threshold is authorized.

## 17. Possible future research directions - hypotheses only, NO IMPLEMENTATION

Questions for future Supervisor-authorized research: how dense soft direction/proximity priors relate to strict canonical direction filtering and minimum boundary distance; whether global/background token contributions shape the prototype; how visual feature similarity and dense decoder inputs interact. These are hypotheses, not proposed repairs or an architecture prescription. No parameter/threshold/ranking change, ablation, training or inference is performed here.

## 18. Questions worth discussing with the advisor

How should strict L3 relation/nearest semantics be represented in a dense attention distribution? Which observed stage statistic best reflects the intended identity semantics, and how should area dependence be interpreted? What evidence would distinguish prototype-feature effects from decoder behavior without conflating them? What independent population would test such hypotheses without reusing these frozen cases? Final scientific interpretation remains with ChatGPT Supervisor.

Diagnostic figure paths and SHA256:

| Figure | Path | SHA256 |
| --- | --- | --- |
| 01_reference_and_gt.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\01_reference_and_gt.png | f268046762c380c8152ec726abc67f8cc894ead86b3eca89c360a479724b21dd |
| 02_P_dir.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\02_P_dir.png | 38e07cb76a617fa55f7e5521393c11d5dab7f09e78d30a61bf949268b332f538 |
| 03_P_near.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\03_P_near.png | 89bc3fc5e41e6baf2596bf80c4b645a7d2cb785d48c71e54ea51e85f54c293e5 |
| 04_W_and_A.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\04_W_and_A.png | 0c3913b125c97e00efe18f595c6945a75977dff3265e030151009e873857babe |
| 05_C.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\05_C.png | bad9a78051fa67ec53df78a8ca2abadd73d888de3ad49204e471b85eab943d7c |
| 06_logits_probability.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\06_logits_probability.png | 48f28ce291fd7852e97fd5a205f8e396fe85cfdd62c4d7a8c984816e9eea35ba |
| 07_final_mask_vs_gt.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\07_final_mask_vs_gt.png | cd5c16c6adc55bef2e6f3ecae0070659b0834b0e10168298e87d353f4a57d376 |
| 08_stage_trajectory.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\08_stage_trajectory.png | 852c4375ceab43f2a381add2c2360cf32a062885a0926ad091d3472083fe4dec |
| contact_sheet.png | C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\evaluation\task8d_above_target_chain_diagnosis_v1\contact_sheet.png | 0d9649fb0f5b2c62403d17eb93bbd4a78ecacb0da03d02c3cc1336588af743ef |
