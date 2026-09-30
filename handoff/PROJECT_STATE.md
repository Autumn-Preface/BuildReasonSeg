# PROJECT_STATE — BuildReasonSeg

_Last updated by DSH at the end of Task 6X._

**Legacy artifact-consistency block (machine-checked, historical/frozen).** The `ARTIFACT-FACTS` block
below describes the superseded **BuildSpatialReason v0.1.1** dataset; its numbers are read-only legacy
evidence and must not be edited. The **active canonical reasoning dataset** for all Task 6L+ work is
**BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0** (see the Identity/current-state section).

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

The block above is machine-checked against
`evaluation/build_spatial_reason_artifact_index.json` by
`scripts/check_artifact_consistency.py`. Do not hand-edit numbers anywhere else.

## Identity

| Field | Value |
|---|---|
| Project codename | **BuildReasonSeg** |
| Repository | `Autumn-Preface/BuildReasonSeg` (branch `main`) |
| Legacy evidence (read-only) | `../WHU_Building_Segment/` |
| Reasoning dataset (active canonical) | **BuildSpatialReason v0.2** over **WHU-EA-NativeVector v1.0** — the active canonical dataset for all Task 6L+ work (28,108 samples; `scene_disjoint_v1` train1/train2/test splits) |
| Legacy reasoning dataset (historical, frozen) | **BuildSpatialReason v0.1.1** — superseded by the Task 6L native-vector migration; its machine-checked ARTIFACT-FACTS block above is legacy artifact-consistency evidence only and is kept read-only |
| Legacy dataset version | **v0.1** (frozen, superseded — never use for training) |
| MVP environment | **`.conda/buildreasonseg-mvp`** (conda `--prefix`, Python 3.11.16, PyTorch 2.13.0+cu132) |
| MVP stack (measured) | **Qwen3-VL-2B-Instruct + SAM2.1 Hiera Base+ + `[SEG]`** (ADR-013) |
| Design stack (unmeasured) | Qwen3-VL-4B-Instruct + SAM 2.1 hiera-large (ADR-012) |
| Network posture | model/data execution is offline from local assets (`HF_HUB_OFFLINE=1`); Watt is transport-only when needed, and ownership rules determine whether it is left running or closed; no hosts edit, no cert-store edit, no insecure TLS flag |
| Reproducibility | **strict deterministic mode, cross-process bit reproducible** (Task 6C) |

## Completed tasks

| Task | Scope | Result |
|---|---|---|
| 1 | Project foundation + baseline freeze | done |
| 2 | Polygon → building component representation | done (4,038 maps, 36,926 components) |
| 3A | Geometry statistics + relation engine + thresholds | done |
| 3B | Relation semantics correction + freeze | done |
| Naming | `SpatialReasoningSeg` → `BuildReasonSeg` | done |
| 4 | BuildSpatialReason-v0.1 dataset generator | done (32,284 records) |
| 5 | v0.1 validator + semantic quality audit | done → `FAIL_REQUIRES_REVISION` |
| 5B | v0.1.1 corrective regeneration + acceptance audit | done → `PASS` |
| 5C | Acceptance hardening + artifact consistency | done → `PASS` |
| 5.5 | External research, model-stack verification, MVP design freeze | done → ADR-012 |
| 6A | Native-Windows env bootstrap + 2B `[SEG]` MVP smoke/overfit | done → `PASS` |
| 6B | Network cleanup + 2B real mini-train + first generalization audit | done → `FAIL_REQUIRES_DEBUG` |
| Watt | Standalone Watt Toolkit lifecycle test (3 rounds) | done → `FULL_AUTO_OK` |
| 6C | **Paired counterfactual training × neutral SAM prompt 2×2 ablation** | **done → `EXPERIMENT_COMPLETE_PARTIAL_IMPROVEMENT`** |
| 6C.5 | **Batch-1 training-pipeline throughput audit + value-preserving optimization** | **done → `OPTIMIZATION_PARTIAL`** |
| 6C.6 | **Launch-overhead candidates + formal-path integration of the 6C.5 winner** | **done → `OPTIMIZATION_PARTIAL`** |
| 6C.7 | **Frozen Qwen visual-feature cache + remaining batch-1 sync audit** | **done → `OPTIMIZATION_PARTIAL`** |
| 6D | **Spatial Grounding Bridge v0.1 (oracle diagnostic + geometry head)** | **done → `GROUNDING_REPRESENTATION_FAILED`** (`VALID_MEASUREMENT_WITH_SCHEDULER_CONFOUND`) |
| 6D.1 | **Corrective G0 rerun + `[SEG]` spatial-decodability audit** | **done → `PRACTICALLY_NOT_DECODABLE_GEOMETRY`** |
| 6E | **Explicit Spatial Token Grounding v0.1 (`[BOX]` + 256 `<loc_*>` tokens)** | **done → `EXPLICIT_SPATIAL_TOKENS_FAILED`** (E0 passes 20/20; E1 structural 0/120) |
| 6F | **Target-Aware `[BOX]` Query Grounding v0.1 (pre-reasoning query + box head)** | **done → `TARGET_AWARE_QUERY_FAILED`** (F0 passes 0.9046/10-10; F1 paired 0/20) |
| 6G | **Dense Query–Visual Spatial Grounding Map v0.1 (query × frozen SAM2 map → heatmap point)** | **done → `DENSE_GROUNDING_IMPLEMENTATION_FAILED`** (grid-256 oracle 0.4883/18-20; G0 inside 0.10) |
| 6H | **Counterfactual Pair-Aligned Dense Grounding v0.1 (pair step + own-vs-cross ranking)** | **done → `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`** (H0 ranking 10/10 but inside-own 3/20) — **causal wording narrowed in Task 6H.1 §3: the logit ranking was scale-degenerate; query learnability was unresolved** |
| 6H.1 | **Spatial-Softmax Point Supervision + Bounded Counterfactual Grounding** | **done → `BOUNDED_POINT_OBJECTIVE_FAILED`** (H0-R point CE 11.10→4.74, inside-own 7/20) |
| 6I | **Visual Query Refinement Block v0.1 (single query cross-attends frozen 64×64 SAM2 embedding once)** | **done → `VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`** (I0 inside-own 7→13/20, paired point 2→5/10; attention peaks at only ~2.5 % target mass) |
| 6J | **Structured Proposal Grounding Feasibility (instruction → relation program → building candidates → explicit geometry execution)** | **done → `PROPOSAL_QUALITY_LIMIT`** (J0 executor 1.000/20-20; J2 parser 1.000 acc, 1.000 full-val; J3 1.000/20-20; J1 frozen-YOLO chain 5/20 paired → J4 not run) |
| 6K | **WHU source → pseudo-instance data audit (read-only)** | **done → `KEEP_WHU_AS_PRIMARY_FOR_NOW`** (`legacy_baseline_role: keep`; conversion drift 4.33 % weighted, `<50` deletion 3.61 % of contours / 0.058 % of foreground, YOLO re-emulation IoU 1.00000, J1 failures 75.4 % proposal-model) — **superseded as a project-level decision by Task 6K.1** |
| 6K.1 | **Recover and validate the native WHU East-Asia vector ground truth (read-only)** | **done → `MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`** (`keep_whu_imagery: true`, `keep_historical_pseudo_baseline: true`; `EA.shp` = 34,085 manually delineated polygons, vector↔raster mean IoU 0.9505, tile mapping validated, VECTOR-vs-PSEUDO relation change 6.96 % weighted, 5.24 % of native buildings missing from the pseudo view, merges only 1.34 %) |
| 6L | **Native-vector canonical dataset + scene-disjoint split + BuildSpatialReason v0.2** | **done → `VECTOR_DATASET_MIGRATION_PASS`** (all 17,388 tiles indexed; 41,186 clipped instances from 33,788 distinct `EA.shp` features; `scene_disjoint_v1` train1 10,044 / train2 3,618 / test 3,726 with zero tile, feature and RGB leakage; v0.2 = 28,108 samples with all 20 programs supported in val and test; v0.1.1↔v0.2 target change 5.70 % / 5.45 % weighted, reconciling with 6K.1's 6.96 %) |
| 6M | **Native-vector proposal model (YOLO26m-seg) + J1-v2/J4-v2 structured evaluation + CMD Demo CLI** | **done → `PROPOSAL_MODEL_NEEDS_IMPROVEMENT`** (export `EXPORT_VALID` at tile-union IoU 0.9988 with 0 malformed / 0 missing and all 1,235 tiny instances kept; parser `PARSER_READY` at 1.0000 accuracy on full v0.2 val after retraining the same 2B text-only head; Demo CLI gate passed on 12 real images without GT; proposal recall@0.50 **0.6333** with tiny recall **0.0023** → J1-v2 fixed120 mIoU 0.2801 / paired 4-20 and J4-v2 test fixed120 mIoU 0.2575 / paired 5-20 FAIL; M1 trained 18 of 80 epochs, wall-clock bound) |
| 6M.1 | **Proposal convergence continuation + corrected Demo gates** | **done → `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`** (Task 6M epoch-18 state hash-verified and snapshotted; the same configuration resumed safely at epoch 19 from a patched copy of `last.pt` and ran to a **normal early stop** at epoch 55 — 37 epochs / 2.42 h, best epoch 40, mask mAP50 0.6827→**0.7374**, mAP50-95 0.3544→**0.4048**; validation-only freeze at conf 0.10 / max_det 100; recall@0.50 0.6333→**0.6578**, empty-tile false-proposal rate 0.1264→**0.0303**; J1-v2 fixed120 mIoU 0.2801→**0.3506**, paired 4→**7/20**, abstentions 35→**27** — but all five validation gates still FAIL, so per §10 **no test run** and diagnosis `CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`; CLI domain gate now rejects OOD prompts with **exit 4 before parser/proposal inference**, vocabulary completed so **100 %** of frozen v0.2 templates are accepted in zh and en, and the corrected audit passes with **12/12** programs + **6/6** OOD rejections) |
| 6N | **Oracle-reference Geometric Relation Field feasibility (B0/B1/B2 ablation)** | **done → `GEOMETRIC_RELATION_FIELD_FEASIBLE`** (oracle-reference only, val-only, test untouched: parameter-free `GeometricRelationField v0.1` (alpha 1.2 / tau 0.04 / softness tau/2, no learned parameters) ranks the true target **top-1 in 100 %** of MiniVal240 (mean target 0.8668 vs best true distractor 0.1003); one frozen decoder family with 144/145/146-channel first convs and exact params 273,473 / 274,625 / **275,777**; N1 Overfit20 gate **PASS** (B2 mIoU 0.9280, Dice 0.9520); N2 MiniVal240 mIoU **B0 0.2148 → B1 0.2413 → B2 0.4531** (B2−B0 **+0.2383**, B2−B1 **+0.2118**); PairedVal20 **16/20** with own−cross margin **+0.4412**; all five §17 criteria pass) |
| 6O | **Geometric relation field causal decomposition (B3/B4)** | **done → `FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`** (frozen B2 reproduced **bit-identically** — all deltas exactly 0.0, paired 16/20 exact — on byte-identical Task 6N packs; **N-B3** = visual + field + relation, no direct reference channel, 145-ch / 274,625 params, O1 gate **PASS** (0.918509 / 0.933920), MiniVal240 mIoU **0.429968**, Dice 0.540155, **paired 14/20**, own−cross **+0.397196**; **N-B4** = field + relation only, no visual, 144-ch / 240,833 params, MiniVal240 mIoU **0.046637**, paired **2/20**; deltas **B3−B2 −0.023159**, B3−B1 +0.188652, **B3−B4 +0.383331**; criteria **14.1 PASS, 14.2 PASS, 14.3 FAIL**) |
| 6P | **Differentiable field v0.2 + predicted-reference substitution** | **done → `REFERENCE_HEAD_INSUFFICIENT`** (v0.2 `geometric_relation_field_v02.py` is **bit-identical** to frozen v0.1 on 64 binary oracle masks — max/mean abs error **0.0** — while keeping autograd: min gradient L1 **11,833.33** on 8 non-binary soft masks, all four directions → `FIELD_V02_VALID`; frozen Task 6O B3 reproduced with **delta 0.0** on mIoU/Dice (`B3_REPRODUCED`); deduplicated reference packs 825 train / 219 val unique + Overfit20 10+10; `ReferenceMaskHead v0.1` (273,441 params, visual + family only) passes P1 (mIoU **0.969504** / Dice 0.984388) but generalizes to RefValUnique mIoU **0.220176**, Dice 0.302794, centroid error median **0.124060** / p90 **0.319618**; predicted-reference chain through v0.2 into frozen B3 gives target mIoU **0.240968** (−0.189000 vs oracle), paired **0/20** (margin +0.003619), field MAE 0.120730 / RMSE 0.274298 / Pearson 0.653476; section 17 fails → verdict) |
| 6Q | **Frozen proposal reference resolver audit** | **done → `REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`** (no training / no tuning / no test split: frozen Task 6M.1 YOLO26m-seg SHA256 `ef852b58…61f474` verified exactly with imgsz 640 / conf 0.10 / max_det 100; deterministic eligibility (no border, bbox extent ≤ 0.20, smallest also area ≥ 150) and largest/smallest area ranking with confidence→index tie-break, explicit abstention; on frozen RefValUnique 219 refs / 212 tiles eligible coverage@0.50 **0.6895** overall (0.8364 largest, 0.5413 smallest), selected-reference mIoU **0.413164**, Dice 0.469992, centroid median **0.015607** / p90 **0.393517**, abstention 0.027397; failure attribution OK **111** / NOT_COVERED 62 / EXTREME_WRONG 39 / NO_ELIGIBLE 3 / NO_PROPOSALS 3 / GEOMETRY_POOR 1; downstream through v0.2 + frozen B3: target mIoU **0.304581** (oracle 0.429968, dense 6P 0.240968), target abstention 0.025, PairedVal **10/20** margin **+0.273700** (oracle 14/20, dense 0/20); section 11 coverage gate fails → verdict) |
| 6R | **Directional GRCL feasibility (relation-level consistency loss)** | **done → `GRCL_NO_MEANINGFUL_RELATION_GAIN`** (controlled loss ablation, oracle reference, test untouched: `grcl_directional.py` implements `L_GRCL = mean(relu(tau−signed) + relu(alpha·orth−signed))` with alpha 1.2 / tau 0.04 / eps 1e-6 / **lambda 0.5**, soft differentiable target centroid, no thresholding in the loss; audit `GRCL_VALID` (8 soft masks, all relations, hinge-active, gradient L1 min **0.026917**, directional sanity pass); frozen B3 (R0) reproduced with **delta 0.0** and hard relation accuracy **0.950000**; R1 Overfit20 gate **PASS** (mIoU 0.917843 / Dice 0.933565 / rel-acc 0.950000); MiniVal240 mIoU **R0 0.429968 / R1 0.411857 / R2 0.233787**, relation accuracy **0.950000 / 0.954167 / 0.541667**, axis violations **0.012987 / 0.000000 / 0.431034**; PairedVal20 **14 / 12 / 10** with margins +0.397196 / +0.364060 / +0.118147; criteria 1,2,6 pass, **3 (+0.004167 < 0.08), 4 (12 < 16), 5 (+0.364060 < 0.38) fail**; `strong_mask_gain false`; proposal transfer diagnostic R1-under-6Q mIoU 0.283782 vs 0.304581, paired 2/20 vs 10/20) |

| 6S | **Directional natural-language end-to-end integration + hardening checkpoint** | **done → `DIRECTIONAL_PARSER_HARDENING_REQUIRED`** (first full chain, all modules frozen: instruction → frozen Qwen3-VL-2B ProgramHead → deterministic decomposition → frozen Task 6Q proposal resolver → predicted reference → field v0.2 → frozen SAM2 feature + relation embedding → frozen Task 6O B3 → target mask; **no GRCL / no oracle reference / no target-proposal selection**; asset audit `ASSETS_FROZEN` (parser SHA `eb50b021…d028a3` exact, text-only, 20 ids; proposal SHA `ef852b58…61f474` exact with imgsz 640 / conf 0.10 / max_det 100; B3 SHA exact; v0.2 unchanged); parser audit **en 1.0000 / zh 1.0000** with 0 unsupported; MiniVal240 strict all-240 mIoU **0.296967**, Dice 0.375321, answered-only mIoU **0.304581** (delta vs frozen 6Q **0.0**), Dice 0.384945, abstention **0.025**, border (n=110) 0.304968, tiny (n=4) ≈0; PairedVal **10/20** own 0.277946 / cross 0.004245 / margin **+0.273700** (identical to 6Q); CLI audit 24 fixed paraphrases **21/24** (3 short Chinese prompts confuse largest↔smallest), OOD 4/4 exit 4, out-of-scope 2/4 exit 5 (both "nearest" prompts are parsed as a supported program → exit 3), on-the-fly SAM2 proven (0.206 s on an uncached tile); attribution parser 0 / reference **117** / target 67 → dominant **`REFERENCE`**; gates 1-5,7,9,10 pass, **6 (21/24 < 22) and 8 (2/4) fail**) |

| 6T | **ProgramHead semantic hardening + scope-safety regression** | **done → `PARSER_SEMANTIC_CONTRAST_FAIL`** (only the parser checkpoint/data and parser-eval code changed; frozen non-parser paths verified unchanged: authoritative Task 6S checkpoint SHA `eb50b021…d028a3` verified, same Qwen3-VL-2B text-only 20-class head, Task 6M LoRA+head policy, **17,479,700 / 2,144,466,964** trainable/total params; no-leakage hardening data = 24,161 v0.2-**train** examples (1,395 train/val duplicate strings removed) + **1,800** deterministic paraphrases = **25,961** prompts with **exact 0 / normalized 0** leakage; frozen minimal-pair pack (76) + stress v1 (160, all 20 classes, 4 zh + 4 en each) created **before** training; C1 (6M recipe) epoch 1 → internal-holdout macro F1 **1.0000**, sweep terminated at the selection ceiling, checkpoint `4cbba36b…44a5e` (local only, 321 s, 0.67 GB); full v0.2 val **1.0000** (en 1.0000 / zh 1.0000), MiniVal240 **240/240**, PairedVal **40/40**, fixed24 **21/24 → 24/24** with all three previously failing Chinese prompts fixed, minimal pairs 0.9474 → **0.9868**, stress 0.9375 → **0.99375**; **gates 9 (minimal < 1.0), 12 (`largest_to_nearest` recall 0.875 < 0.90), 14 (2/4 nearest controls still parse `largest_to_right_of` and exit 3)** fail while e2e regression reproduces Task 6S **exactly** (answered mIoU Δ 0.0, paired 10/20, abstentions 6) → verdict; main bottleneck `REFERENCE` deliberately untouched) |

| 6U | **Reference candidate coverage + ProposalSetRanker** | **done → `REFERENCE_RANKER_NOT_HELPFUL`** (reference-side hardening only; YOLO26m-seg SHA `ef852b58…61f474` verified and **not retrained**, Task 6Q eligibility/field v0.2/SAM2/B3 frozen: train-only split from RefTrainUnique by unique key = **U-Calib200** 200 (100+100) + **U-RankerTrain** 625 (391+234), zero overlap with each other and with RefValUnique; four declared configs on U-Calib200 → smallest eligible@0.50 U-C0 0.9100 / **U-C1 0.9400** / U-C2 0.8800 / U-C3 0.9000, overall 0.9450/0.9600/0.9200/0.9300 → section-10 priority ranking `U-C1 > U-C0 > U-C3 > U-C2` → **U-C1 frozen** (imgsz 640 / conf 0.05 / max_det 300, no TTA/tiling); RefValUnique coverage overall 0.6895 → **0.7534** (+0.0639) and smallest 0.5413 → **0.6330** (+0.0917) → `candidate_coverage_improved=true`, oracle ceiling 0.6203; ProposalSetRanker v0.1 (14-d features, 14→32→16→1, 1,025 params; 598 trainable / 27 `untrainable_not_covered`; holdout top-1 0.45) **degrades** selection: RefVal mIoU U-S0 0.4248 / U-S1 **0.4289** / U-S2 0.3107, `SELECTION_WRONG` 39 → 53 → **88**; downstream MiniVal240 answered U-S0 0.3046 / U-S1 **0.3141** / U-S2 0.2466, reference-fail 117 → 115 → **158**, PairedVal 10 → **11** → 11, margin +0.2737 → **+0.3209** → +0.3332 (U-S0 reproduces Task 6S exactly); §24 U-S2 gates **3/10 pass** → verdict) |

| 6V | **Family-conditioned reference resolver policy** | **done → `FAMILY_POLICY_NOT_BETTER`** (**no model trained**: only frozen options recombined; YOLO26m-seg `ef852b58…61f474`, ranker `c738fcf7…1a46c0`, hardened parser `4cbba36b…d44a5e`, B3 `7556e4a4…c7d6ab` all hash-verified and untouched; three declared options V-P0 (U-C0+deterministic) / V-P1 (U-C1+deterministic) / V-P2 (U-C1+ranker, no fourth); train-only U-Calib200 per-family selection → **largest V-P2** (Pr@0.5 0.8000 / mIoU 0.6650 / SELECTION_WRONG 18) and **smallest V-P0** (0.5300 / 0.4154 / 38) frozen as `{largest: V-P2, smallest: V-P0}` before RefVal and never revised (calibration prefers U-C0 for smallest, inverting Task 6U's RefVal reading); RefValUnique frozen-policy mIoU **0.4383** / Dice 0.4995 / Pr@0.5 **0.5327** / centroid median **0.0107** / p90 0.3845 / abstain **0.0228**, buckets OK **114** / SELECTION_WRONG **41** / NOT_COVERED 59 (vs U-S0 0.4248/111/39/62, U-S1 0.4289/112/53/50, U-S2 0.3107/77/88/50; oracle ceiling 0.6203); downstream MiniVal240 strict **0.3005** / answered **0.3069** / abstain 5 / ref-fail **116** / target-fail 68 (U-S1 0.3089/0.3141/4/115), PairedVal **10/20** margin **+0.2879**, parser integration **240/240**; §13 flag **false** (mIoU Δ +0.0094 < +0.015, REFERENCE_OK 114 < 117) and §14 gate **3/10** → verdict) |

| 6W | **Proposal-quality filtering + semantic extreme selection** | **done → `QUALITY_FILTER_NOT_HELPFUL`** (support-infrastructure hypothesis test; U-C1 `ef852b58…61f474` / hardened parser / field v0.2 / SAM2 / B3 all frozen and hash-verified; **W0 oracle gate PASSED** on train-only U-Calib200 — oracle `q_gt≥0.50` filter + deterministic area rule: overall mIoU 0.4789 → **0.6575** (+0.1786 ≥ +0.08), smallest 0.3545 → **0.6707** (+0.3162 ≥ +0.10), `SELECTION_WRONG` 76 → **23** (≤45), abstain 0 ✓ → estimator training permitted; dataset from U-RankerTrain tiles only, deduplicated by tile, **36 calib-overlapping and 0 refval tiles excluded** → 548 tiles / **5,246 proposals** (3,337 pos / 1,909 neg), 93 `feature_invalid_small`, tile-disjoint 420/128 split, overlaps 0; `ProposalQualityEstimator v0.1` = 8 geometry scalars + frozen SAM2 256×64×64 inside/ring means (512) → `Linear(512→64)+LN+GELU` ‖ `Linear(8→16)+GELU` → `Linear(80→32→1)`, **35,729 params**, BCE `pos_weight 0.5666` train-only, frozen threshold 0.50, epoch 19 → holdout **AUROC 0.8438 / F1 0.8251** (adequacy ✓); RefValUnique mIoU W-S0 0.4289 → **W-SQ 0.3793** (oracle ceiling **0.5209**), `REFERENCE_OK` 112 → 97 (oracle 144), `SELECTION_WRONG` 53 → **65** (oracle 20), rejected-all 5; downstream MiniVal240 answered 0.3141 → **0.2746**, abstain 4 → 10, ref-fail 115 → **132**, PairedVal 11/20 → **9/20** (margin +0.3209 → +0.2482); parser integration 240/240; §23 gates **4/12** → verdict) |

| 6X | **Frozen SAM2.1 proposal-refinement audit** | **done → `SAM2_REFINEMENT_NOT_HELPFUL`** (**no model trained**; last predeclared reference-hardening audit; SAM2.1 Hiera Base+ `a2345aed…c004c5` + official `SAM2ImagePredictor` (0 trainable params, box-only probe → masks (3,512,512) scores 0.887/0.456/0.863), U-C1 `ef852b58…61f474`, ProgramHead/B3/field v0.2 all frozen and verified → `FROZEN_ASSETS_VERIFIED`; exactly four options X-C0 no-refinement / X-C1 exact box single-mask / X-C2 exact box multimask / X-C3 10%-expanded box multimask, SAM score used only as a tie-break and never thresholded; train-only U-Calib200: **X-C0 mIoU 0.4789 / smallest 0.3545 / largest 0.6034 / Pr@0.5 0.5800** vs X-C1 0.4744/0.3502/0.5986/0.5700 (centroid median better 0.0061 vs 0.0078), X-C2 0.4284, X-C3 0.3741 (+1 abstention) → ranking `['X-C0','X-C1','X-C2','X-C3']` → **frozen option X-C0** (`baseline_is_selected = true`, RefVal never consulted); **STOP rule triggered**: RefValUnique/MiniVal240/PairedVal20 deliberately **not evaluated** and their artifacts absent by design) |

## Task 6X measured results

The last predeclared reference-hardening audit: a **training-free**, foundation-model alternative to Task
6W's learned quality classifier — refine each eligible U-C1 YOLO proposal mask from its geometric box prompt
with the frozen official SAM2.1 image predictor, then run the literal largest/smallest rule on the refined
instance masks. Support infrastructure only. Full detail: `docs/task6x_sam2_proposal_refinement.md`,
`evaluation/task6x_*.json`.

| | Value |
|---|---|
| Frozen assets | SAM2.1 Hiera Base+ SHA `a2345aed…c004c5` exact · official `SAM2ImagePredictor`, 0 trainable params · box-only probe masks `(3,512,512)`, scores `0.8870/0.4559/0.8629` · U-C1 (640 / 0.05 / 300, no TTA/tiling) SHA `ef852b58…61f474` exact · ProgramHead / B3 / field v0.2 unchanged · ranker + quality estimator excluded |
| Four options | **X-C0** no refinement · **X-C1** exact box + single-mask · **X-C2** exact box + multimask (max predicted quality) · **X-C3** 10 %-expanded box (clip [0,511]) + multimask · SAM score = selection tie-break only, never thresholded |
| U-Calib200 (train-only) | **X-C0 mIoU 0.4789** / smallest 0.3545 / largest 0.6034 / Pr@0.5 0.5800 / centroid med 0.0078 / abstain 0 · X-C1 0.4744 / 0.3502 / 0.5986 / 0.5700 / **0.0061** / 0 · X-C2 0.4284 / 0.3277 / 0.5291 / 0.5250 / 0.0075 / 0 · X-C3 0.3741 / 0.3008 / 0.4481 / 0.4523 / 0.0098 / 0.0050 |
| Cost per option | X-C0 0 SAM2 calls/tile · X-C1/X-C2/X-C3 9.67 calls/tile, 0.172 / 0.126 / 0.134 s per tile, 0 empty refined masks |
| Ranking and freeze | `['X-C0', 'X-C1', 'X-C2', 'X-C3']` → **frozen option X-C0**, `baseline_is_selected = true`, `fourth_option = null`, `chosen_without_refval = true`, immutable |
| STOP rule | **triggered** — `refval_minival_paired_evaluated = false`; `task6x_refval_refinement.json` / `task6x_downstream_minival240.json` / `task6x_downstream_pairedval20.json` / `task6x_hardened_parser_integration.json` intentionally absent |
| Verdict | **`SAM2_REFINEMENT_NOT_HELPFUL`** |

1. **Box-prompt SAM2 refinement does not beat the raw U-C1 masks** under the literal largest/smallest rule:
   the best refinement option (X-C1, single-mask) is 0.0045 mIoU below baseline, multimask loses 0.0505 and
   the expanded box loses 0.1048 (plus its first abstention) — while costing 9.67 SAM2 calls per tile.
2. **Centroid quality improves slightly, area semantics do not**: X-C1 has the best median centroid error
   (0.0061 vs 0.0078), so the refined masks are marginally better centred, but their areas shift the
   extreme selection enough to lose overall IoU — consistent with SAM2 preferring object-like regions that
   are not the literal area extremes.
3. Because the baseline won, the **stop rule of section 13.1** ended the audit before any
   RefVal/MiniVal/Paired evaluation, so no untouched-split claim is made for refinement.
4. Next: 等待 ChatGPT 根据 Task 6X 的 U-Calib200 refinement 结果决定 reference 子系统是否冻结并转向 nearest/L3，不自行启动 nearest/L3 或新增 reference 模块。

## Task 6W measured results

Tests the hypothesis that fragmented/merged/incomplete proposals corrupt the literal largest/smallest rule
and that filtering low-quality proposals first would restore reliable extreme selection. Support
infrastructure only; no new algorithmic novelty claim and no change to the detector, field or B3. Full
detail: `docs/task6w_proposal_quality_filter.md`, `evaluation/task6w_*.json`.

| | Value |
|---|---|
| Frozen assets | U-C1 (640 / 0.05 / 300, no TTA/tiling) · YOLO SHA `ef852b58…61f474` exact · field v0.2 / SAM2 / B3 / parser unchanged |
| W0 oracle gate (U-Calib200) | overall mIoU 0.4789 → **0.6575** (**+0.1786** ≥ +0.08 ✓) · smallest 0.3545 → **0.6707** (**+0.3162** ≥ +0.10 ✓) · `SELECTION_WRONG` 76 → **23** (≤45 ✓) · abstain 0 ✓ · `REFERENCE_OK` 116 → 169 |
| Quality dataset | U-RankerTrain only, deduplicated by tile; 36 calib-overlapping + 0 refval tiles **excluded** → **548** tiles · **5,246** proposals (**3,337 pos / 1,909 neg**) · 93 `feature_invalid_small`, 0 empty rings · train/holdout **420 / 128** tiles, overlap 0 |
| Estimator | 8 geometry scalars + SAM2 256×64×64 inside-mean ‖ one-cell-ring-mean = 512 visual · `Linear(512→64)+LayerNorm+GELU` ‖ `Linear(8→16)+GELU` → `Linear(80→32)+GELU+Linear(32→1)` · **35,729 params** · BCE `pos_weight 0.5666` (train-only) · AdamW 5e-4 / wd 1e-4 / batch 256 / patience 5 · epoch **19** → holdout **AUROC 0.8438**, AUPRC 0.8952, **F1 0.8251** (adequacy ✓) |
| RefValUnique | mIoU **0.4289 / 0.3793 / 0.5209** (W-S0 / W-SQ / W-ORACLE) · Pr@0.5 0.5209 / 0.4619 / **0.6872** · centroid median 0.0159 / 0.0490 / **0.0065** · p90 0.3803 / 0.3963 / 0.3251 · abstain 0.0183 / 0.0411 / 0.0365 |
| RefVal buckets | `REFERENCE_OK` 112 → 97 (oracle **144**) · `SELECTION_WRONG` 53 → **65** (oracle 20) · `NOT_COVERED` 50 → 48 · `QUALITY_FILTER_ALL_REJECTED` 0 → **5** · largest/smallest mIoU (oracle) **0.6043 / 0.4366** |
| Downstream MiniVal240 | strict 0.3089 → 0.2631 · answered **0.3141 → 0.2746** · abstain 4 → 10 · ref-fail 115 → **132** · target-fail 69 → 60 · largest/smallest 0.3241/0.2937 → 0.2888/0.2375 · border 0.2896 → 0.2536 · tiny ≈0 |
| PairedVal20 | **11/20 → 9/20** · own 0.3249 → 0.2576 · cross 0.0040 → 0.0095 · margin **+0.3209 → +0.2482** · abstention pairs 0 → 1 |
| Parser integration (Part J) | hardened ProgramHead → W-SQ: parser **240/240**, strict 0.2631, answered 0.2746, abstain 10, ref-fail 132 |
| Flags | W0 ✓ · adequacy ✓ · `quality_reference_improved` **false** (mIoU < W-S0 + 0.04, `REFERENCE_OK` 97 < 120, `SELECTION_WRONG` 65 > 39) |
| §23 gates | **4/12** (W0 ✓, adequacy ✓, parser ✓, no test/GT ✓); ✗ RefVal mIoU < 0.48, centroid median > 0.03, p90 > 0.28, answered < 0.33, strict < 0.31, paired < 12, margin < 0.30, ref-fail > 100 |
| Verdict | **`QUALITY_FILTER_NOT_HELPFUL`** |

1. **The mechanism is sound; the estimator is the bottleneck**: with oracle quality knowledge the same
   deterministic largest/smallest rule jumps to mIoU 0.6575 / smallest 0.6707 on U-Calib200 and to 0.5209
   with `REFERENCE_OK` 144 on RefValUnique, while the learned filter (holdout AUROC 0.844) *reduces* every
   reference and downstream metric.
2. **The learned filter rejects good candidates**: it empties 5 RefVal reference sets, pushes
   `SELECTION_WRONG` 53 → 65 and abstentions 4 → 10 downstream, hurting the smallest family most
   (0.3353 → 0.2982).
3. W-S0 reproduced Task 6U U-S1 exactly (strict 0.3089 / answered 0.3141 / paired 11/20 / margin +0.3209),
   validating the causal isolation; the parser integration stayed 240/240.
4. Next: 等待 ChatGPT 根据 Task 6W 的 oracle-quality ceiling、learned quality filter 与 downstream 结果决定 reference 是否继续硬化，不自行增加 ranker、size estimator 或 detector 改动。

## Task 6V measured results

Tests whether a **family-conditioned policy over already-frozen resolver options** recovers the useful part
of Task 6U at zero new model cost. **No model was trained**; the detector configs, Task 6Q deterministic
selector, ProposalSetRanker v0.1, hardened ProgramHead, field v0.2, SAM2 and B3 are frozen. Full detail:
`docs/task6v_family_conditioned_reference_resolver.md`, `evaluation/task6v_*.json`.

| | Value |
|---|---|
| Frozen assets (all verified exact) | YOLO26m-seg `ef852b58…61f474` · ranker `c738fcf7…1a46c0` · hardened parser `4cbba36b…d44a5e` · B3 `7556e4a4…c7d6ab` — none retrained |
| Options | V-P0 = U-C0 + deterministic · V-P1 = U-C1 + deterministic · V-P2 = U-C1 + ranker (exactly three, no fourth) |
| U-Calib200 largest | V-P0 0.7600 / 0.6338 / 22 · V-P1 0.7100 / 0.6034 / 27 · **V-P2 0.8000 / 0.6650 / 18** (Pr@0.5 / mIoU / SELECTION_WRONG) |
| U-Calib200 smallest | **V-P0 0.5300 / 0.4154 / 38** · V-P1 0.4500 / 0.3545 / 49 · V-P2 0.0300 / 0.0546 / 91 |
| Frozen policy | **`{largest: V-P2, smallest: V-P0}`** (priority: Pr@0.5 → mIoU → fewer SELECTION_WRONG → centroid median → abstention → simpler) |
| RefValUnique (frozen policy) | mIoU **0.4383** · Dice 0.4995 · Pr@0.5 **0.5327** · centroid med **0.0107** / p90 0.3845 · area-ratio 1.0757 · abstain **0.0228** |
| RefVal buckets vs 6U | OK **114** (U-S0 111 / U-S1 112 / U-S2 77) · SELECTION_WRONG **41** (39 / 53 / 88) · NOT_COVERED 59 (62 / 50 / 50) · oracle ceiling 0.6203 |
| RefVal per family | largest mIoU **0.5680** Pr **0.6636** (ranker) · smallest 0.3011 / 0.3942 (deterministic U-C0) |
| Downstream MiniVal240 | strict **0.3005** · answered **0.3069** · abstain **5** · ref-fail **116** · target-fail 68 · largest/smallest 0.3393/0.2617 · border (114) 0.2985 · tiny ≈0 |
| PairedVal20 | **10/20** · own 0.291106 · cross 0.003251 · margin **+0.2879** · 0 abstention pairs |
| Parser integration | **240/240** exact; strict 0.3005, answered 0.3069, abstain 5, ref-fail 116 |
| §13 improvement flag | **false** — mIoU Δ **+0.0094** (needs ≥ +0.015), REFERENCE_OK 114 < 117; SELECTION_WRONG 41 ≤ 48 ✓, abstention 0.0228 ≤ 0.05 ✓ |
| §14 hardening gate | **3/10** ✓ centroid median, parser 240/240, no test/GT; ✗ RefVal mIoU < 0.45, p90 > 0.32, answered < 0.325, strict < 0.305, paired < 12, margin < 0.30, ref-fail > 105 |
| Verdict | **`FAMILY_POLICY_NOT_BETTER`** |

1. **The routing works as designed**: `largest` gets V-P2 (ranker mIoU 0.5680 / Pr 0.6636) and `smallest`
   keeps the deterministic selector (0.3011) instead of the ranker's collapse (0.0411), so the frozen
   policy beats U-S2 and U-S0 on RefVal mIoU/Pr and cuts `SELECTION_WRONG` 53 → 41 versus U-S1.
2. **But the train-only calibration chose U-C0 for `smallest`**, giving back the U-C1 coverage gain on that
   family (`NOT_COVERED` 50 → 59) and holding the overall mIoU gain to +0.0094, below the predeclared
   +0.015 margin; `REFERENCE_OK` also stays below U-S1 + 5.
3. **Recorded calibration/RefVal inversion**: U-Calib200 prefers C0 for smallest while RefValUnique
   preferred C1; the frozen selection rule uses calibration only and was not revised.
4. Downstream the policy sits between U-S0 and U-S1 (strict 0.3005 vs 0.2970/0.3089) and fails 7 of the 10
   §14 hardening conditions.
5. Next: 等待 ChatGPT 根据 Task 6V 的 family-conditioned resolver 结果决定是否需要新的 proposal-quality / selection 机制，不自行继续修改 reference 架构。

## Task 6U measured results

Reference-side hardening that isolates the dominant Task 6S bottleneck (`REFERENCE` 117 vs target 67) into
candidate **coverage** and extreme-instance **selection**. The detector, the Task 6Q eligibility rules,
field v0.2, SAM2 and B3 are frozen; only a new support-infrastructure ranker was trained. Full detail:
`docs/task6u_reference_hardening.md`, `evaluation/task6u_*.json`.

| | Value |
|---|---|
| Frozen assets | YOLO26m-seg SHA `ef852b58…61f474` exact (not retrained) · Task 6O B3 SHA exact · field v0.2 / SAM2 / eligibility unchanged |
| Train-only split | **U-Calib200** 200 (100 largest + 100 smallest) · **U-RankerTrain** 625 (391 + 234) · zero key overlap; RefValUnique 219 untouched |
| Four declared configs (U-Calib200) | smallest eligible@0.50 **U-C0 0.9100 · U-C1 0.9400 · U-C2 0.8800 · U-C3 0.9000**; overall 0.9450 / 0.9600 / 0.9200 / 0.9300; largest 0.9800 / 0.9800 / 0.9600 / 0.9600 |
| Frozen selection | ranking `U-C1 > U-C0 > U-C3 > U-C2` → **U-C1** (640 / 0.05 / 300, default NMS, no TTA, no tiling); higher input resolution *hurt* |
| RefValUnique coverage | overall 0.6895 → **0.7534** (**+0.0639**) · largest 0.8364 → 0.8727 · smallest 0.5413 → **0.6330** (**+0.0917**) → `candidate_coverage_improved` **true** |
| Oracle ceiling (U-C1) | mIoU **0.6203**, Dice 0.7092, Pr@0.5 0.7674, centroid median 0.0046 / p90 0.108 (diagnostic only) |
| ProposalSetRanker v0.1 | 14-d features · 14→32→16→1 MLP · **1,025 params** · 598 trainable / **27 `untrainable_not_covered`** · internal holdout top-1 **0.45**, mean IoU 0.4510 · epoch 2, 28.6 s |
| RefValUnique selectors | mIoU **0.4248 / 0.4289 / 0.3107** · Pr@0.5 0.5258 / 0.5209 / 0.3581 · centroid median 0.0156 / 0.0159 / 0.0890 · abstain 0.0274 / 0.0183 / 0.0183 (U-S0 / U-S1 / U-S2) |
| RefValUnique buckets | `NOT_COVERED` 62 → **50** → 50 · `SELECTION_WRONG` 39 → 53 → **88** · `REFERENCE_OK` 111 → **112** → 77 |
| Downstream MiniVal240 | strict **0.2970 / 0.3089 / 0.2425** · answered **0.3046 / 0.3141 / 0.2466** · abstain 6 / **4** / 4 · reference-fail 117 / **115** / **158** · border 0.2943 / 0.2896 / 0.2666 · tiny ≈0 |
| PairedVal20 | pass **10 / 11 / 11** · own 0.277946 / 0.324870 / 0.336264 · cross 0.004245 / 0.003988 / 0.003089 · margin +0.2737 / **+0.3209** / +0.3332 |
| Final integration (Part J) | Task 6T parser + U-S2 + frozen field/B3: parser **240/240**, strict 0.2425, answered 0.2466, abstain 4 |
| Flags | `candidate_coverage_improved` **true** · `ranker_improved_selection` **false** (Δ mIoU −0.118, `SELECTION_WRONG` ratio 1.66 vs required ≤0.70) |
| §24 gates (U-S2) | **3/10** (margin ✓, no test ✓, no GT ✓); RefVal mIoU / centroid median / p90, MiniVal answered / strict, paired 11 < 12, reference-fail 158 > 93 all fail |
| Verdict | **`REFERENCE_RANKER_NOT_HELPFUL`** |

1. **Candidate hardening works**: lowering conf to 0.05 and raising max_det to 300 (at the same 640
   resolution) adds eligible candidates and improves RefValUnique coverage by +0.064 overall / +0.092
   smallest, cutting `REFERENCE_NOT_COVERED_IOU50` from 62 to 50 and abstentions from 6 to 4, and it lifts
   the downstream chain (strict 0.2970 → 0.3089, answered 0.3046 → 0.3141, paired 10 → 11, margin
   +0.2737 → +0.3209).
2. **Higher resolution did not help** (U-C2/U-C3 rank below U-C0/U-C1 on the calibration split).
3. **The learned ranker with the mandated 14-d feature set is harmful**: it selects the small target's
   reference poorly (smallest mIoU 0.3353 → 0.0411), raising `SELECTION_WRONG` 53 → 88 and downstream
   reference-fail 115 → 158, because the feature set deliberately excludes centroid/location and the
   oracle ceiling (0.6203) shows the information needed to pick well is not in it.
4. U-S0 reproduced Task 6S exactly, validating the causal isolation.
5. Next: 等待 ChatGPT 根据 Task 6U 的 candidate coverage、ranker 与 downstream 因果结果决定下一步，不自行修改 detector、reference 语义或 target 架构。

## Task 6T measured results

Small parser/scope-safety hardening step that closes part of the Task 6S interface gate. **Only the
ProgramHead checkpoint/training data and parser-evaluation code changed** — the proposal model, Task 6Q
resolver, field v0.2, SAM2, B3, MiniVal240/PairedVal20, dataset/splits and the test split are untouched
(git-verified). The main scientific bottleneck (`REFERENCE`) is explicitly out of scope. Full detail:
`docs/task6t_programhead_semantic_hardening.md`, `evaluation/task6t_*.json`.

**Mandatory Task 6S erratum (recorded, Task 6S artifacts not mutated):** the claim that the frozen
20-program vocabulary has no `nearest` program is **false** — `EXPECTED_QUERY_TYPES` contains six nearest
classes. The Task 6S nearest controls failed because the frozen ProgramHead misclassified them as
direction-only programs.

| | Value |
|---|---|
| Baseline parser | `task6m/program_parser_v02_best.pt` SHA `eb50b021…d028a3` verified exactly before training |
| Architecture | same Qwen3-VL-2B text-only 20-class head; Task 6M trainable policy (LoRA + head); 17,479,700 / 2,144,466,964 trainable/total |
| Hardening data | 24,161 v0.2-**train** examples + 1,800 paraphrases = **25,961**; leakage **exact 0 / normalized 0** |
| Frozen packs (pre-training) | minimal pairs **76** (groups A 16 / B 16 / C 32 / D 12) · stress v1 **160** (20 classes × 4 zh + 4 en) |
| Training | declared C1/C2/C3, ran **C1** (holdout macro F1 **1.0000** at epoch 1, sweep stopped at the ceiling); checkpoint `4cbba36b…44a5e` (local only), 321.1 s, 0.67 GB |
| Canonical (baseline → hardened) | full v0.2 val 1.0000 → **1.0000** (en 1.0000 / zh 1.0000); MiniVal240 240/240 → **240/240**; PairedVal 40/40 → **40/40** |
| Paraphrase/contrast (baseline → hardened) | fixed24 21/24 → **24/24** (all 3 previously failing fixed); minimal pairs 0.9474 → **0.9868**; stress v1 0.9375 → **0.99375** (macro F1 0.99375; en 1.0000, zh 0.9875) |
| Residual failures | 1 minimal-pair + 1 stress prompt, both Chinese `largest_to_nearest` with the "边界距离最小的" phrasing (class recall 0.875); the two compact nearest scope controls still parse `largest_to_right_of` |
| Scope safety | OOD **4/4 exit 4** ✓; `largest` / `leftmost` controls **exit 5** ✓; 2 nearest controls **exit 3** ✗; CLI fixed24 **24/24** |
| End-to-end regression | parser 1.0; answered mIoU **Δ 0.0**, strict **Δ 0.0**, paired **10/20**, margin Δ 0.0, abstentions **6** → `END_TO_END_REG_REPRODUCED` |
| Gates | 1-8 ✓ · **9 ✗ 0.9868** · 10 ✓ 0.99375 · 11 ✓ 0.99375 · **12 ✗ 0.875** · 13 ✓ · **14 ✗ 2/4** · 15-19 ✓ |
| Verdict | **`PARSER_SEMANTIC_CONTRAST_FAIL`** |

1. **Canonical behaviour is bit-identical** (full val 1.0000, MiniVal240 240/240, PairedVal 40/40, e2e Δ 0.0)
   while paraphrase robustness improved substantially (fixed24 21 → 24/24, minimal 0.947 → 0.987, stress
   0.938 → 0.994) — the hardening did not trade canonical accuracy for paraphrase robustness.
2. **Three predeclared semantic/scope gates still fail**, all traced to the Chinese/compact L2-nearest
   phrasing: the augmentation grammar taught the split nearest pattern but never the compact
   "[X]右侧最近的[Y]" pattern.
3. The main bottleneck remains **`REFERENCE`**; Task 6T did not touch it.
4. Next: 等待 ChatGPT 根据 Task 6T 的 parser hardening 结果决定是否继续 parser/scope 修复或转向 reference-hardening，不自行修复或扩展范围。

## Task 6S measured results

First natural-language directional end-to-end chain, integrating only frozen modules (no retraining, no
threshold tuning, no test split). Full detail:
`docs/task6s_directional_end_to_end_integration.md`, `evaluation/task6s_*.json`.

| | Value |
|---|---|
| Primary chain | ProgramHead (frozen Qwen3-VL-2B) → decomposition → 6Q proposal reference resolver → field v0.2 → frozen SAM2 feature → frozen 6O B3; **no GRCL, no oracle reference** |
| Frozen assets | parser SHA `eb50b021…d028a3` exact (text-only, 20 ids) · proposal SHA `ef852b58…61f474` exact (imgsz 640 / conf 0.10 / max_det 100) · B3 SHA exact · field v0.2 unchanged → `ASSETS_FROZEN` |
| Parser audit | en **1.0000**, zh **1.0000**, 0 unsupported classifications (240/240 both) |
| End-to-end MiniVal240 | strict all-240 mIoU **0.296967** / Dice 0.375321; answered-only (234) mIoU **0.304581** / Dice 0.384945 / Pr@0.5 0.299145; abstention **0.025** (6 records) |
| Per direction / family | left 0.335333 · right 0.247051 · above 0.329150 · below 0.303450; largest 0.335005 · smallest 0.273099 |
| Border / tiny | border (n=110) **0.304968**; tiny (n=4) **≈0** (7.43e-08) |
| vs frozen Task 6Q | answered mIoU delta **0.0** (identical 234-record answered set), Dice 0.384945 vs 0.3849449 → no integration regression |
| PairedVal20 | pass **10/20** (gate ≥9), own 0.277946 / cross 0.004245 / margin **+0.273700** (gate ≥0.22) — identical to 6Q; parser-correct members 40/40, 0 abstention pairs |
| CLI 24 paraphrases | **21/24** (gate ≥22 ✗): 3 short Chinese prompts (`找出最大建筑{左/右/下}边的建筑物。`) parse as `smallest_to_*` |
| CLI controls | OOD 4/4 **exit 4** ✓; out-of-scope 2/4 **exit 5** ✓ (`largest`, `leftmost`); both "nearest" prompts parse as `largest_to_right_of` → **exit 3** ✗ |
| On-the-fly SAM2 | **proven**: uncached tile `2_0`, `sam2_feature = 0.206 s` (vs 0.002 s cached), exit 0, four visuals written |
| Attribution | `PARSER_WRONG` **0** · `REFERENCE_NO_PROPOSALS` 3 · `REFERENCE_NO_ELIGIBLE` 3 · `REFERENCE_NOT_COVERED_IOU50` **68** · `REFERENCE_SELECTION_WRONG` **42** · `REFERENCE_GEOMETRY_POOR` 1 · `TARGET_FAIL_WITH_REFERENCE_OK` 67 · `TARGET_OK` 56 |
| Dominant bottleneck | parser_fail 0 (0.0000) · reference_fail **117** · target_fail 67 → **`REFERENCE`** |
| Gates | 1 ✓ 1.0000 · 2 ✓ 0.296967 · 3 ✓ 0.304581 · 4 ✓ 10/20 · 5 ✓ +0.273700 · **6 ✗ 21/24** · 7 ✓ · **8 ✗ 2/4** · 9 ✓ · 10 ✓ |
| Verdict | **`DIRECTIONAL_PARSER_HARDENING_REQUIRED`** |

1. **The integration itself is clean**: the whole chain reproduces the frozen Task 6Q answer set and
   metrics exactly (answered mIoU delta 0.0, paired 10/20 and margin +0.273700 to the last digit), with
   the CLI proven to need only an image, a prompt and the frozen checkpoints, and to run SAM2 on the fly.
2. **Parsing is exact on the canonical data but not paraphrase-robust**: 1.0000 on all 240 MiniVal240
   queries in both languages, yet 3 of 16 short Chinese paraphrases flip largest↔smallest.
3. **Scope cannot reject semantics the vocabulary cannot express**: "nearest" prompts are classified as a
   supported directional program and stop at the resolver (exit 3) rather than exiting 5.
4. **The dominant bottleneck is `REFERENCE`** (117 of 240 records), not the parser (0) and not the target
   decoder/field (67), with `REFERENCE_NOT_COVERED_IOU50` (68) and `REFERENCE_SELECTION_WRONG` (42)
   dominating — consistent with Task 6Q's coverage finding.
5. Next: 等待 ChatGPT 根据 Task 6S 的全链路审计与 dominant bottleneck（REFERENCE）决定 Task 6T 的 hardening 方向，不自行修复 parser、reference 或 target decoder。

## Task 6R measured results

Controlled loss ablation for the second candidate algorithm contribution: does explicit supervision on
the geometric relation between the predicted target mask and the reference mask improve relation
correctness beyond field guidance alone, without degrading mask quality? Oracle reference throughout,
four directional relations only, GT target a label/score only, test split never read. Full detail:
`docs/task6r_grcl_directional_feasibility.md`, `evaluation/task6r_*.json`.

| | Value |
|---|---|
| GRCL v0.1 | `alpha 1.2`, `tau 0.04`, `eps 1e-6`, `lambda_grcl 0.5`; soft differentiable target centroid; `L_total = BCE + Dice + 0.5·GRCL`; no thresholding inside the loss |
| Audit | **`GRCL_VALID`** — 8 soft masks, all four relations, non-symmetric references, hinge-active, gradient L1 min **0.026917**, directional sanity pass (correct side 0, wrong side 0.5395) |
| R0 (frozen B3) | reproduction **delta 0.0**; MiniVal240 mIoU 0.429968, Dice 0.540155, Pr@0.5 0.692450, relation accuracy **0.950000**, margin 0.230544, axis violation 0.012987; PairedVal 14/20, margin +0.397196 |
| R1 Overfit20 | mIoU **0.917843**, Dice **0.933565**, rel-acc **0.950000** → gate **PASS** |
| R2 Overfit20 | mIoU 0.932792, Dice 0.957379, rel-acc 1.000000 (no gate) |
| MiniVal240 | mIoU **0.429968 / 0.411857 / 0.233787**; Dice 0.540155 / 0.536478 / 0.329888; Pr@0.5 0.692450 / 0.626946 / 0.417037; rel-acc **0.950000 / 0.954167 / 0.541667**; margin 0.230544 / 0.238583 / 0.179077; axis violation 0.012987 / **0.000000** / 0.431034; mean GRCL 0.018724 / 0.047917 / 0.098439 |
| R1 per relation mIoU / accuracy | left 0.3741 / 0.9333 · right 0.3985 / 0.9333 · above 0.4307 / 1.0000 · below 0.4441 / 0.9500 |
| R1 resources | 274,625 params · epoch 3 · 57.3 s · 0.65 GB (R2: 273,473 · epoch 9 · 51.7 s · 0.65 GB) |
| PairedVal20 | pass **14 / 12 / 10**; own 0.398969 / 0.365532 / 0.157666; cross 0.001773 / 0.001471 / 0.039519; margin **+0.397196 / +0.364060 / +0.118147**; member rel-acc 0.950000 / 0.925000 / 0.400000 |
| Criteria | 1 ✓ overfit · 2 ✓ retention (0.411857 ≥ 0.409968) · 3 ✗ gain **+0.004167** < 0.08 · 4 ✗ paired **12** < 16 · 5 ✗ margin **+0.364060** < 0.38 · 6 ✓ field useful **+0.178069** |
| Strong flag | `strong_mask_gain = false` (R1 is 0.018111 mIoU below R0) |
| Proposal transfer | R1 under the frozen 6Q resolver: mIoU **0.283782** (6Q B3 0.304581), paired **2/20** (10/20), margin +0.079561, rel-acc 0.8291, abstentions 6 — diagnostic only |
| Verdict | **`GRCL_NO_MEANINGFUL_RELATION_GAIN`** |

1. **GRCL is correctly implemented and harmless to mask quality**: the audit passes, retention holds
   (−0.018111 mIoU, inside the 0.02 allowance), and axis violations drop to exactly zero.
2. **It does not deliver the predeclared relation gain**: relation accuracy moves 0.950000 → 0.954167
   (`+0.004167` against a required `+0.08`), and PairedVal falls 14/20 → 12/20 with margin
   0.397196 → 0.364060.
3. **The field remains materially useful**: `R1 − R2 = +0.178069` mIoU, while the no-field control's
   relation accuracy (0.541667) and axis-violation rate (0.431034) show GRCL alone cannot replace it.
4. Next: 等待 ChatGPT 根据 Task 6R 的 GRCL 因果实验结果决定 Task 6S，不自行修改 loss、reference 架构或开始 MLLM/nearest/L3。

## Task 6Q measured results

Diagnostic/selection task on the frozen Task 6M.1 proposal model: can the oracle reference mask be
replaced by a deterministic *proposal-based* reference (eligibility + largest/smallest ranking) and still
drive the field-guided B3 target decoder? Nothing was trained or tuned; the test split was never read.
Full detail: `docs/task6q_frozen_proposal_reference_resolver.md`, `evaluation/task6q_*.json`.

| | Value |
|---|---|
| Frozen proposal config | YOLO26m-seg `best.pt` SHA256 verified exactly; imgsz 640 / conf 0.10 / max_det 100 / default NMS / no TTA / no tiling / no sweep |
| Resolver rules | `largest`: no border + bbox extent ≤ 0.20 → max area · `smallest`: same + area ≥ 150 → min area · tie: higher confidence then lower index · explicit abstention |
| Eligible coverage@0.50 | overall **0.6895** · largest **0.8364** · smallest **0.5413** (all-proposal: 0.6941) |
| Selected-reference quality | mIoU **0.413164**, Dice 0.469992, Pr@0.5 0.511415, centroid median **0.015607**, p90 **0.393517**, area ratio 1.081301, abstentions 6/219 (0.027397) |
| Failure attribution | `REFERENCE_OK` **111** · `REFERENCE_NOT_COVERED_IOU50` 62 · `EXTREME_SELECTION_WRONG` 39 · `NO_ELIGIBLE_PROPOSALS` 3 · `NO_PROPOSALS` 3 · `SELECTED_MASK_GEOMETRY_POOR` 1 |
| Downstream target (MiniVal240) | mIoU **0.304581**, Dice 0.384945, Pr@0.5 0.635327, abstention 6/240 (0.025000); per relation 0.3353 / 0.2471 / 0.3291 / 0.3034; per family 0.3350 / 0.2731; border 0.304968 |
| Comparison | vs oracle B3 0.429968 (**−0.125387**) · vs Task 6P dense predicted 0.240968 (**+0.063614**) |
| PairedVal20 | **10/20** (own 0.277946 / cross 0.004245 / margin **+0.273700**); oracle 14/20 · dense 0/20 |
| Gates | section 11 **FAIL** (0.6895 < 0.70 overall; 0.5413 < 0.60 smallest) · section 12 **FAIL** (p90 0.393517 > 0.12) · section 13 fails only through section 12 |
| Verdict | **`REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`** |

1. **The proposal resolver is a strong support module where it covers**: overall eligible coverage@0.50
   0.6895, largest 0.8364, selected-reference mIoU 0.413164 with a median centroid error of only
   0.015607 — far better centred than the Task 6P dense head (median 0.124060).
2. **It is coverage-limited, not selection-limited**: `REFERENCE_NOT_COVERED_IOU50` dominates the
   smallest family (45 of 109) and the selected-reference p90 tail (0.393517) comes from the cases where
   the reference simply is not among the eligible proposals.
3. **It still propagates better than the dense head**: target mIoU 0.304581 vs 0.240968 and PairedVal
   10/20 vs 0/20, against the oracle-reference 0.429968 / 14-20.
4. Next: 等待 ChatGPT 根据 Task 6Q 的 frozen-proposal reference resolver 结果决定 Task 6R，不自行修改 reference 架构或开始 MLLM/GRCL/nearest/L3。

## Task 6P measured results

Differentiable field v0.2 + predicted-reference substitution: can the oracle reference mask be replaced
by a learned predicted reference mask while preserving the field-guided target-segmentation signal?
Val-only, test untouched, no joint training, no MLLM hidden-state fusion. Full detail:
`docs/task6p_differentiable_field_predicted_reference.md`, `evaluation/task6p_*.json`.

| | Value |
|---|---|
| v0.1 limitation (recorded) | `.detach()` on tensor masks + Python-`float` centroid → **not** differentiable w.r.t. the reference mask; valid for the oracle 6N/6O, insufficient for predicted-reference training |
| v0.2 field | `geometric_relation_field_v02.py`, same formula/constants, soft centroid with eps 1e-6, no detach / no Python float / no NumPy in forward |
| v0.2 audit | equivalence **max 0.0 / mean 0.0** on 64 binary masks (16 per direction, tolerances 1e-6 / 1e-7); gradient L1 min **11,833.33** on 8 soft masks, four directions → **`FIELD_V02_VALID`** |
| B3 reproduction | checkpoint SHA256 verified, not retrained, oracle field via v0.2 → mIoU **0.4299680351479113**, Dice **0.5401551056992948**, both **delta 0.0** → `B3_REPRODUCED` |
| Reference packs | RefTrainUnique **825** (1.21× dedup), RefValUnique **219**, Overfit20 **10 largest + 10 smallest**; key `(split, tile, reference_source_feature_id, family)`; no target identity, no relation id |
| ReferenceMaskHead | 144-ch fusion, **273,441 params**, input = frozen SAM2 feature + family id only |
| P1 Overfit20 | **mIoU 0.969504 / Dice 0.984388** → gate PASS |
| P2 RefValUnique | mIoU **0.220176**, Dice 0.302794, Pr@0.5 0.359514; largest 0.279714 / smallest **0.160091**; centroid mean 0.150767 / **median 0.124060** / **p90 0.319618**; median area ratio 1.019795; epoch 12; 47.8 s; 0.65 GB |
| Chain (predicted ref → v0.2 → frozen B3) | target mIoU **0.240968**, Dice 0.316164, Pr@0.5 0.546548, border 0.231872; **paired 0/20** (own 0.064774 / cross 0.061155 / margin **+0.003619**) |
| Propagation | mIoU **−0.189000**, paired **−14**, margin **−0.393577** vs oracle B3; field MAE 0.120730 / RMSE 0.274298 / Pearson 0.653476 |
| Gates | section 17 **FAIL** (mIoU 0.220176 < 0.35; median 0.124060 > 0.05; p90 0.319618 > 0.12); section 18 **FAIL** |
| Verdict | **`REFERENCE_HEAD_INSUFFICIENT`** |

1. **The differentiable field is exactly right**: v0.2 reproduces frozen v0.1 bit-for-bit on binary masks
   while carrying a strong gradient to a soft reference mask, and frozen B3 reproduces with delta 0.0.
2. **The bottleneck is reference grounding, not the field**: the head fits 20 references (0.9695) but
   reaches only 0.220176 on 219 unique validation references, with centroid error median 0.124060.
3. **The error propagates**: with the predicted reference the target mIoU falls 0.429968 → 0.240968 and
   PairedVal 14/20 → 0/20.
4. Next: 等待 ChatGPT 根据 Task 6P 的 predicted-reference 误差传播结果决定 Task 6Q，不自行进行联合训练、MLLM 隐状态融合、nearest/L3 或 GRCL。

## Task 6O measured results

Narrow causal decomposition of the Task 6N result, oracle-reference only, val-only, test untouched,
frozen Task 6N packs reused byte-for-byte. Full detail:
`docs/task6o_field_causal_decomposition.md`, `evaluation/task6o_*.json`.

| | Value |
|---|---|
| B2 reproduction | **bit-identical** — mIoU 0.4531265609993713, Dice 0.5768438150123932, paired 16/20, own 0.44334608244093643, cross 0.002157857978561074; every delta exactly **0.0** (tolerance 1e-6) |
| N-B3 | visual + `P_rel` + relation, **no direct reference channel**, 145-ch, 274,625 params |
| N-B4 | `Conv1x1(1→128)` on `P_rel` + relation, **no visual, no reference**, 144-ch, 240,833 params |
| O1 Overfit20 | **B3 0.918509 / 0.933920 (gate PASS)** · B4 0.109511 / 0.162109 (no gate; cannot fit 20 samples) |
| MiniVal240 mIoU | B1 0.241316 (frozen) · **B2 0.453127** · **B3 0.429968** · **B4 0.046637** |
| MiniVal240 Dice | B2 0.576844 · B3 0.540155 · B4 0.074986 |
| B3 breakdowns | per relation 0.4219 / 0.3905 / 0.4581 / 0.4493; largest 0.4101 / smallest 0.4498; border 0.4132 (n=114); tiny ≈0 (n=4); Pr@0.5 0.692450; epoch 14; 0.642 GB; 116.2 s |
| PairedVal20 | B2 16/20 (margin +0.441188) · **B3 14/20 (margin +0.397196)** · B4 2/20 (+0.048955) |
| Deltas | **B3−B2 −0.023159** (Dice −0.036689, paired −2) · B3−B1 **+0.188652** · **B3−B4 +0.383331** (Dice +0.465169, paired +12) |
| Criteria | **14.1 PASS** (reference channel not required: within 0.03 slack, 14/20, margin ≥ 0.10) · **14.2 PASS** (+0.383331 ≥ 0.10) · **14.3 FAIL** |
| Verdict | **`FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`** |

1. **The frozen baseline is exactly reproducible**: B2 re-evaluated once from a hash-verified checkpoint
   on byte-identical packs gave bit-identical metrics, and adding B3/B4 changed nothing for B0/B1/B2.
2. **The direct reference channel is not required** for this directional decoder once the field is
   present (measured cost −0.023159 mIoU, inside the predeclared 0.03 slack).
3. **The gain is field-guided visual segmentation**: the geometry-only control loses 0.383331 mIoU and
   12 paired passes and cannot even overfit 20 samples, so the handcrafted field alone does not solve
   the benchmark.
4. Next: 等待 ChatGPT 根据 Task 6O 因果分解结果决定后续架构，不自行开始 predicted-reference、nearest、L3 或 GRCL。

## Task 6N measured results

First task of the final innovation-architecture phase: an **oracle-reference** feasibility ablation of a
differentiable, parameter-free, relation-conditioned geometric prior fused into a dense segmentation
decoder. Val-only; the test split was never read; `reference_source = oracle_native_gt` everywhere.
Full detail: `docs/task6n_oracle_reference_geometric_relation_field.md`, `evaluation/task6n_*.json`.

| | Value |
|---|---|
| Scope | the 8 directional L2 programs only (`largest|smallest_to_{left_of,right_of,above,below}`) |
| Frozen visual | SAM2.1 Hiera Base+ image embedding **256 × 64 × 64** (Task 6C.7 / 6I path), SHA256 recorded, never retrained |
| Field | `GeometricRelationField v0.1`, **zero learned parameters**, alpha 1.2 / tau 0.04 / softness tau/2, `P_rel = sign·axis·margin ∈ [0,1]` |
| N0 field sanity | top-1 **1.0000**, top-3 **1.0000**, mean target 0.8668 vs best true distractor 0.1003, margin **+0.7698** |
| Packs (frozen pre-training) | Overfit20 (4 pairs), MiniTrain1000, MiniVal240 (30×8), PairedVal20; 1,395/1,008 in-scope records all eligible, no difficulty deletion |
| Variants | B0 144-ch / 273,473 params · B1 145-ch / 274,625 · **B2 146-ch / 275,777** (no padding to equalise) |
| N1 Overfit20 | B0 0.9431 · B1 0.9332 · **B2 0.9280 mIoU / 0.9520 Dice** → gate **PASS** |
| N2 MiniVal240 | mIoU B0 0.2148 · B1 0.2413 · **B2 0.4531**; Dice 0.3043 / 0.3276 / **0.5768**; Pr@0.5 0.4191 / 0.4496 / **0.6640** |
| Comparisons | **B2−B0 +0.2383**, **B2−B1 +0.2118**, B1−B0 +0.0265 |
| Border / tiny targets | border mIoU 0.1935 / 0.2171 / **0.4253** (n=114); tiny mIoU ≈0 / ≈0 / **0.0216** (n=4) |
| PairedVal20 | 9/20 · 7/20 · **16/20**; own−cross margin +0.1058 / +0.1365 / **+0.4412** |
| Criteria | all five §17 criteria pass |
| Verdict | **`GEOMETRIC_RELATION_FIELD_FEASIBLE`** |

1. **The field alone is already a strong geometric prior**: with an oracle reference mask it ranks the
   true target first among all non-reference buildings in 100 % of records, with no learned parameters.
2. **The fused field is what moves dense segmentation**: B2 beats both equally controlled baselines by a
   wide margin, and the paired test shows it selects its own target (own IoU 0.4433 vs cross 0.0022).
3. **This is an oracle upper-bound feasibility result, not end-to-end inference**, and no novelty or
   "first-ever" claim is made. Next: 等待 ChatGPT 根据 Task 6N 测量结果决定 Task 6O，不自行选择后续算法。

## Task 6M.1 measured results

Narrow corrective continuation of Task 6M: preserve the 18-epoch evidence, continue the **same**
YOLO26m-seg configuration to convergence, re-evaluate on validation, and fix the Demo CLI's
unsupported-instruction handling. Full detail: `docs/task6m1_proposal_convergence_and_demo_fix.md`,
`evaluation/task6m1_*.json`.

| | Value |
|---|---|
| Evidence preservation | `SOURCE_CHECKPOINT_VERIFIED` — `best.pt` / `last.pt` hashes matched and snapshot copied; still unchanged after the run; frozen packs reused byte-for-byte |
| Safe resume | `SAFE_RESUME_READY` — patched-copy resume at **epoch 19**, optimizer + EMA restored, Task 6M.1-local `save_dir`, frozen config unchanged |
| Continuation | **early stop at epoch 55** (patience 15), 37 epochs, 2.42 h, 235.5 s/epoch, 11.2 GB peak, no NaN/Inf |
| Combined best (epoch 40) | mask mAP50 **0.73742** / mAP50-95 **0.40475** (Task 6M ep 18: 0.68266 / 0.35437) |
| Proposal val | recall@0.25/0.50/0.75 **0.7902 / 0.6578 / 0.3928**; mask mAP50 **0.7367**; 5.39 proposals/tile; empty-tile FP **0.0303**; tiny/border/dense 0.0068 / 0.5753 / 0.6590; small/medium/large 0.1621 / 0.5574 / 0.7505 |
| Frozen config | conf 0.10 / max_det 100, validation-selected from the same declared grid, frozen before any test consideration |
| J1-v2 (val) | fixed120 mIoU **0.3506**, paired **7/20**, abstentions **27**, full-val mIoU 0.3541 → all five gates FAIL |
| Test (J4-v2) | **not run** (§10; no new test evidence, no re-tuning) |
| Demo CLI | OOD prompts **exit 4 before parser/proposal models**; **100 %** of frozen v0.2 templates accepted in zh and en; audit **12/12** programs (4 L1 / 4 L2 / 4 L3, 12 distinct programs), 9 masks + overlays, 3 explicit abstentions, no GT, **6/6** OOD rejected |
| Diagnosis | **`CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`** |
| Verdict | **`PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`** |

1. **The continuation was safe and normal**: the same configuration, resumed from a verified epoch-18
   snapshot, ran to an early stop with no NaN/Inf and no Task 6M mutation.
2. **Convergence helped every metric** (recall, mAP, structured mIoU, paired pass, abstentions,
   empty-tile false-proposal rate) yet **all five validation gates still fail**, with the residual gap
   concentrated in small/tiny buildings.
3. **The Demo path is now correct**: unsupported instructions can no longer be silently mapped — they
   exit 4 before any model runs — and the documented gate vocabulary covers every frozen v0.2
   template in both languages.
4. **Next task decides the small-object strategy** (resolution/tiling, another proposal family, or
   re-scoping the graded target set). No such change was made here.

## Task 6M measured results

Proposal-model task on the migrated annotation source: YOLO26m-seg trained on a derived export of
`WHU-EA-NativeVector v1.0` under `scene_disjoint_v1`, evaluated through the native-vector structured
chain (J1-v2 validation, J4-v2 single graded test run) with a CMD-runnable Demo path. Full detail:
`docs/task6m_native_vector_proposal_demo.md`, `evaluation/task6m_*.json`.

| | Value |
|---|---|
| Eval packs (frozen pre-training) | val/test fixed120 + paired20, SHA256-recorded, all 20 programs represented |
| Model / env | YOLO26m-seg (official weights, SHA256 recorded) on `ultralytics==8.4.164` (PyPI) in `.conda/buildreasonseg-proposal`; AGPL-3.0 note recorded |
| Export | 17,388 tiles (all hardlinked), 41,186 polygons, 11,909 empty labels; tile-union IoU **0.99880**, per-instance (≥9 px) **0.99587**, 0 malformed, 0 missing, tiny 1,235/1,235, 172 documented sub-pixel repairs → **`EXPORT_VALID`** |
| M0 smoke | 6/6 gates pass (checkpoint, 2-image fwd/bwd, 600-tile 2-epoch subset, 512×512 mask decoding, empty tiles, tiny labels) |
| M1 training | **18/80 epochs** (best epoch 14, `mAP50-95(M)` 0.36392), batch 16 at 10.5–11.2 GB VRAM, 313 s/epoch, stopped by session wall-clock budget |
| Proposal metrics (val) | recall@0.25/0.50/0.75 **0.7772 / 0.6333 / 0.3404**; mask mAP50 **0.6819**; 6.38 proposals/tile; empty-tile FP 0.1264; tiny/border/dense recall 0.0023 / 0.5461 / 0.6322; size recall large 0.727 → small 0.125 → tiny 0.002 |
| Frozen inference config | conf 0.10, max_det 100, validation-selected, frozen before test |
| J1-v2 (val) | fixed120 mIoU 0.2801, paired 4/20, abstentions 35/120, full-val mIoU 0.2973, abstention rate 0.2768 → all 5 gates FAIL |
| Parser (v0.2) | frozen J2 head loaded only partially (626 LoRA keys missing → 0.515); retrained same 2B head on v0.2 train → **1.0000 accuracy / 1.0000 macro F1** on 9,111 val records → **`PARSER_READY`** |
| J4-v2 (test, once) | parser 1.0000; fixed120 mIoU **0.2575** (gate 0.40) FAIL; paired **5/20** (gate 14) FAIL; full-test mIoU 0.2914 |
| Demo CLI | **gate passed** — 12 real images, 12 correct programs, 9 selected masks + overlays, 3 explicit abstentions, no GT |
| Verdict | **`PROPOSAL_MODEL_NEEDS_IMPROVEMENT`** |

1. **The pipeline is delivered end-to-end** — export, environment, smoke, training, val tuning
   (frozen), J1-v2, parser, single graded J4-v2, Demo CLI and attribution all exist as artifacts.
2. **The binding constraint is proposal quality on small objects**, amplified by an 18-of-80-epoch
   training budget; the executor, parser, annotation and split are all exonerated by measurement.
3. **Next step: train the same configuration to convergence**, then re-run the frozen protocol; if
   tiny-object recall still limits, address it with an explicit small-object strategy (resolution /
   tiling), not a bigger backbone.

## Task 6L measured results

Migration of the primary annotation source from semantic connected-component pseudo-instances to the
validated native vector polygons (Task 6K.1 verdict), plus a scene-disjoint split and
BuildSpatialReason v0.2. No model training, no downloads, no installations; `datasets/whu/` and
v0.1.1 stay frozen. Full detail: `docs/task6l_vector_dataset_migration.md`, ADR-025,
`evaluation/task6l_*.json`.

| | Value |
|---|---|
| Canonical dataset | `datasets/whu_native_vector/v1.0/` — `WHU-EA-NativeVector` v1.0, UID `(EA.shp SHA256, source_feature_id)` |
| Tile coverage | **17,388 / 17,388** (train 3,135 · train_no 10,527 · test 903 · test_no 2,823); 11,909 empty, 5,479 non-empty |
| Instances | **41,186** clipped instances · **33,788** distinct source features · max 55/tile (uint8-safe) |
| Preservation | no `<50` deletion (1,235 tiny retained and flagged), no component merging, no hole loss (6 instances with holes), no simplification |
| Truncation / visibility | border-truncated 35.4 % · visible_fraction median 1.0, p5 0.069 |
| Split views | `legacy_compat_v1` (historical 2,508 / 627 / 903) and primary `scene_disjoint_v1` (**10,044 / 3,618 / 3,726**) |
| Leakage | tile overlap **0**, source-feature overlap **0**, cross-split RGB duplicates **0** |
| v0.2 samples | **28,108** (train 12,778 · val 9,111 · test 6,219); L1 19,769 · L2 5,323 · L3 3,016; L3 trivial 1,392 / nontrivial 1,624 |
| Program support | all **20 / 20** programs present in train, val and test |
| v0.1.1 ↔ v0.2 | common 23,210 groups · unchanged 21,886 · changed **1,324 (5.70 %)** · weighted **5.45 %** (L1 7.61 %, L2 0.74 %, L3 1.61 %) vs Task 6K.1's 6.96 % → reconciles |
| Acceptance gates | **11 / 11 PASS** (determinism rerun byte-identical; independent overlay audit min IoU **1.0**) |
| Verdict | **`VECTOR_DATASET_MIGRATION_PASS`** |

1. **The migration is complete and verified**, not assumed: every tile has a canonical record, every
   instance keeps its stable source id, and the committed label maps are exactly reproducible from
   `EA.shp`.
2. **The scene-disjoint split is clean** (zero tile/feature/RGB leakage) but only separates scenes —
   it is not a cross-city generalization claim, and no such claim is made.
3. **v0.2 is deliberately not answer-compatible with v0.1.1** (5.45 % weighted change); the two
   versions must never be mixed in one table without stating the version.
4. **Next step is a model task**: train/select an instance-segmentation proposal backbone and run the
   prepared J1/J4 evaluation against native instance masks, reporting deltas against the frozen
   v0.1.1 baseline.

## Task 6K.1 measured results

Read-only recovery and validation of the native WHU East-Asia vector map discovered by Task 6K
(`2. The shape file of the whole images\EA.shp`), parsed with a standard-library ESRI reader because
no shapefile/GDAL/rasterio/tifffile package is installed. No source, converted or legacy file was
modified (`evaluation/task6k1_read_only_proof.json`), no model trained, no dataset regenerated.
Full detail: `docs/task6k1_whu_native_vector_groundtruth.md`, ADR-024, `evaluation/task6k1_*.json`.

| | Value |
|---|---|
| Vector record count | **34,085** polygons (`.shp` == `.shx` == `.dbf`; 0 deleted rows; 0 unclosed rings; 5 with holes; 5 multipart) |
| DBF attributes | **degenerate** — all fields constant over all records (`OBJECTID` 27, `Shape_Area` 211.7518) → identity = `.shp` record order |
| CRS | `WGS_1984_World_Mercator` (map-unit areas, not ground m²) |
| Rasters | BigTIFF RGB 8-bit tiled 128×128 (train1 95,521×27,801; train2 34,772×27,802; test 35,765×27,802), labels LZW/PackBits 1-bit, `.tfw` pixel 0.33956958 m |
| Tile mapping | grid capacity == cropped tiles exactly (10,044 / 3,618 / 3,726); RGB windows **pixel-identical**; labels exact; best offset (0,0) in 240/240 |
| Vector ↔ raster | mean IoU **0.9505**; 4,036/4,038 tiles ≥ 0.90; 0 tiles < 0.50 |
| Instances | 38,824 clipped native instances, 32,590 distinct features, border-truncated 34.1 % |
| Matching vs pseudo | 94.76 % at IoU 0.50; unmatched native **5.24 %**; unmatched pseudo 0.37 % |
| Merges (primary, per semantic component) | **1.32 %** of components contain ≥ 2 native buildings (max 4) |
| Merges (per pseudo-instance, containment ≥ 0.8) | **1.34 %** (2.65 % of native buildings affected) |
| Splits | 0.13 % of native buildings span ≥ 2 pseudo-instances |
| `<50` filter effect | **100 %** of native buildings < 50 px are unmatched (1,039/1,039) |
| **VECTOR vs PSEUDO relation drift** | **6.96 %** weighted (L1 7.87 %, L2 3.38 %, L3 5.03 %); 39.5 % of tiles change ≥ 1 program |
| Task 6J re-attribution | 51/65 = **78.5 %** proposal-model on clean single native buildings; **0 %** merged-target cases |
| Split/geographic | **100 %** of val tiles adjacent to effective-training tiles (nearest distance 1 for all 627); test 0 % adjacency |
| Verdict | **`MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`**, imagery kept, historical pseudo baseline kept |

1. **The archive contains true instance ground truth** that the semantic raster cannot express:
   34,085 manually delineated polygons, aligned to the imagery at IoU 0.95 with zero residual shift.
2. **The old "touching buildings merged" story is small** (1.32–1.34 %). The measurable conversion
   defect is the opposite: **5.24 % of native buildings are missing** from the pseudo view, traced to
   the `<50 contourArea` filter (100 % of sub-50 px buildings are absent) plus hole filling and
   polygon simplification.
3. **Relation answers shift by 6.96 % weighted**, so migrating the instance source is a deliberate,
   budgeted change — the frozen artifacts stay valid and comparable.
4. **Task 6J is settled:** the J1 failure is the proposal chain (78.5 % on clean single buildings), not
   the instance definition. The next task builds the vector-derived canonical dataset and
   BuildSpatialReason-v0.2 on the unchanged imagery.

## Task 6K measured results

Read-only audit of the chain original WHU semantic raster → historical polygon conversion
(`mask_to_yolo.py`) → current BuildReasonSeg component representation. No source, converted or
legacy file was modified; no model trained; nothing installed. Full detail:
`docs/task6k_whu_source_pseudoinstance_audit.md`, ADR-023, `evaluation/task6k_*.json`,
`evaluation/dataset_audit_schema_v1.json`.

| | Value |
|---|---|
| Source inventory | 17,388 tiles: train 3,135 · train_no 10,527 · test 903 · test_no 2,823; labels binary `[0,255]`; empty masks 11,915; 0 missing pairs |
| Historical split | recovered from the folders: train 2,508 + val 627 = source train (exactly) · test 903 = source test; 0 overlaps; `int(3135 × 0.2) = 627` ✓; `set→list→shuffle(42)` caveat recorded |
| Raw semantic components (8-conn) | **38,309** (4-conn: 38,491); 9.49/tile; median area 1,209 px; border-touching 34.2 % |
| Current pseudo-instances | **36,926** = exactly the surviving polygons (`component_id = polygon index + 1`) |
| `<50` filter loss | 1,383 contours removed (**3.61 %**), 37,589 px = **0.058 % of foreground**; 1,099 tiles affected; `len(approx)<3` removed 0 |
| `RETR_EXTERNAL` loss | 6,757 components with holes; 13,399 hole px **added** (0.021 %) |
| Approx loss | IoU **0.99955**, boundary 0.0086 px; 25/4,038 tiles below 0.99 |
| Actual-YOLO fidelity | 36,926/36,926 objects matched, count agreement 4,038/4,038, union IoU **1.00000**, 0 malformed |
| **Relation drift RAW vs CONVERTED** | 26.5 % of tiles change something; **4.33 %** weighted current-query target change; L1 **5.0 %**, L2 0.03 %, L3 0.03 %; flips cancel (constructibility ±0.5 %) |
| Merge risk (heuristic) | raw high **1.40 %**, medium 4.29 %, low 94.31 % |
| Task 6J cross-analysis | 65/120 J1 failures: **49 (75.4 %) proposal-model on clean targets**, 2 (3.1 %) conversion/data, 14 (21.5 %) inseparable |
| Verdict | **`KEEP_WHU_AS_PRIMARY_FOR_NOW`** — no replace gate fires (drift 0.0433 < 0.05, deletion 0.0361 < 0.05, foreground 0.00058 < 0.01, merge high 0.0140 < 0.15) |

1. **The conversion is faithful and light.** All 36,926 objects re-emulate exactly (IoU 1.00000);
the only material loss is the `<50` small-object filter, which deletes 3.6 % of contours but only
0.058 % of foreground area.
2. **Relation targets are stable except at the extremes.** Compositional L2/L3 programs drift
0.03 %; the L1 extremes drift 5 % because a deleted sub-50 px speck can be the extreme component.
Corpus-level answerability is essentially unchanged (per-program constructibility moves ±0.5 %).
3. **WHU's structural limits are recorded, not repaired:** no true instance identity anywhere in
the chain (instances are connected components) and a random train/val split over the same
contiguous regions (region-level correlation certain; tile-level unquantifiable).
4. **Next step unchanged:** the binding constraint is the proposal model, so a proposal-backbone
task (or deterministic proposal post-processing) is next, evaluated with the same J1/J4 gates, plus
the schema-v1 audit of any candidate replacement dataset before a decision.

## Task 6J measured results

The structured route is audited stage by stage with a 20-program canonical vocabulary derived 1:1
from the frozen v0.1.1 query types (verified against all 25,229 records), an independent executor
that reuses the frozen Task 3B relation engine (never reads target id/GT reasoning/target mask),
a text-only Qwen ProgramHead, and the frozen YOLOv8m-seg-WHU baseline invoked read-only with hashed
provenance. Full detail: `docs/task6j_structured_proposal_grounding.md`, ADR-022,
`evaluation/task6j_*.json`.

| | Value |
|---|---|
| J0 oracle executor | exact **1.000 (120/120)**, paired **20/20**, 0 abstentions, **120/120 agreement** with the frozen `recompute_target_from_steps` → gate (0.98/19) **PASS** |
| YOLO provenance | best.pt sha256 `d9a6a65b…`, yolo_sam_env (py 3.10.20, ultralytics 8.4.67), read-only, hash re-verified |
| Proposal recall | @0.25/0.50/0.75 = **0.919/0.869/0.594**, mean best IoU 0.701, missing 13.1 %, ≈9.5 proposals/img, tiny-component recall 0.391 |
| J1 oracle program + YOLO | mIoU **0.3712** (gate 0.30 ✓), **35/120 abstentions**, paired mask **5/20** (gate 12 ✗) → viability gate **FAIL** → J4 not run |
| J2 ProgramHead (text-only) | fixed-120 acc **1.0000**, macro F1 **1.0000**, paired **20/20**, full-val (3,884) **1.0000** → gate **PASS** |
| J3 predicted program + oracle candidates | selected-target **1.0000**, paired **20/20** → gate **PASS** |
| Verdict | **`PROPOSAL_QUALITY_LIMIT`** — parser + executor solved; the frozen YOLO proposal chain (instance semantics mismatch with components) is the binding failure |

1. **Parsing and execution are solved, not the problem.** J0/J2/J3 all pass at ceiling: the 20
   canonical programs reproduce the generator exactly, and a 2B text-only classifier maps
   instructions to programs perfectly (the instructions are template-generated and semantically
   complete, so 100 % is expected and informative: program parsing is NOT the bottleneck).
2. **The proposal backbone is the binding constraint.** Recall and single-mask mIoU pass, but
   YOLO instances (split/merged/border-clipped) change relation outcomes under the frozen
   component-calibrated semantics: 35/120 abstentions and 5/20 paired selection.
3. **J4 is correctly not run** (section 15). Next recommended step: a proposal-backbone/dataset
   task (instance segmentation aligned with component semantics, or deterministic proposal
   post-processing) with the parser/executor frozen. No YOLO retraining, no learned SRE, no
   `[REF]`/SCL, no 4B, no dataset migration, no GUI — waiting for review.

## Task 6I measured results

The frozen Task 6H.1 objective and paired training are reused byte-for-byte; the only addition is
one `VisualQueryRefinementBlock` (q0 → LayerNorm → Linear(2048,256) → one cross-attention over the
frozen 64×64 SAM2 embedding, embed 256 / 4 heads, residual + FFN(256→512→256) → q1 → scores the
frozen 256×256/32-ch feature). Full detail: `docs/task6i_visual_query_refinement.md`, ADR-021,
`evaluation/task6i_*.json`.

| | Value |
|---|---|
| Architecture proof | F64 `[1,256,64,64]`, F256 `[1,32,256,256]` (by size), exactly 1 cross-attention layer, block 1,129,985 params, scorer = dot(q1,K256)/√256 + bias (bit-identical to the combined forward) |
| Freeze proof | gradients reach query/coarse/attention/FFN/fine projections + LoRA + token rows with `frozen_or_other == 0.0`; SAM2 and the frozen Task 6G head bit-identical, zero grad; shared F64/F256 bit-identical |
| Causal proof | `[BOX]` bit-identical under future-text mutation in **fp32** (bf16 sdpa rounds the shared prefix per sequence length: 0.31 delta vs 0.0 fp32) |
| **I0** (10 pairs, 1500 pair steps) | inside-own **13/20**, top-1 9/20, paired point **5/10**, bounded ranking **10/10**, norm-err **0.0838**, own mass **0.4166** vs cross **0.0236**, mean \|logit\| 36.1 → **failed** (gates 18/20, 9/10, 9/10, <0.08) |
| I0 audit | point CE 11.0876 → **3.1833**, CF loss 0.8527 → 0.0967, target-cell prob 1.5e-5 → 0.268, top-1 rate 0 → 0.45, q1 same-image L2 2.2 → **1755.6** vs q0 27.4 → 162.6; **attention target mass 0.0079 → 0.0250** (peaked but not on target) |
| I1 / I2 | **not run** (sections 8/14 stop at the failed I0) |

1. **The refinement step works but under-shoots.** One visual look raises the single-query
   localization roughly 2× (inside 7→13, paired point 2→5, own mass 0.18→0.42, error 0.167→0.084) —
   a real, verified improvement over the frozen 6H.1 baseline — but the gates are far away.
2. **The attention cannot aim itself.** The 1×4096 cross-attention peaks (entropy 8.31→1.46 nats)
   with only ~2.5 % mass inside the target's 64×64 region, and that share barely moves during
   training. A single query slot has no target hypothesis to test against the map, so it locks onto
   a salient but task-irrelevant location and the scorer amplifies it (q1 L2 → 1756, \|logit\| → 36).
3. **Accepted verdict (section 13): `VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`** → clean failure,
   task stops before I1/I2. Recommended for review (section 14): **multiple learned query slots**
   (OMG-Seg/OMG-LLaVA-style object queries), a stronger/larger MLLM, or architecture-level
   reference/relation grounding — none implemented automatically. No SAM2 training, no second
   refinement layer, no multiple queries, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.
4. **Two methodology fixes adopted for all later inference paths**: inference runs the Qwen model
   in **eval mode** (LoRA dropout p=0.05 is training-only noise), and causal bit-identity probes
   run in **fp32**.

## Task 6H.1 measured results

Architecture frozen at Task 6G/6H; only the spatial supervision target changed: a 65,536-class
spatial cross-entropy on the deterministic target **point-cell** plus a **bounded** own-vs-cross
target **probability-mass** preference, with Task 6G's BCE+Dice and Task 6H's raw-logit ranking
detached to zero gradient. Full detail: `docs/task6h1_bounded_point_counterfactual.md`, ADR-020,
`evaluation/task6h1_*.json`.

| | Value |
|---|---|
| Objective | `0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf`, `L_point` = CE over 65 536 spatial classes, `mass(P,M)=sum(P*M)` (bounded, no area division), `L_cf = -log(p_own/(p_own+p_cross))`, eps 1e-8, **no margin** |
| Retirement proof | `bce_dice_requires_grad: false`, `logit_ranking_requires_grad: false`, gradient contribution 0.0 |
| Gradient coverage | point-CE reaches query projection 0.881, query norm 0.0136, visual 1×1 0.109, LoRA 0.109, token rows 0.322; bounded `L_cf` reaches all five groups too |
| **H0-R** (10 pairs, 1500 pair steps) | inside-own **7/20**, top-1 cell 5/20, paired point **2/10**, bounded ranking **8/10**, own mass **0.1835** > cross **0.0456**, mean \|logit\| 13.2 → **failed** (gates 18/20, 9/10, 9/10, <10.0) |
| H0-R audit | point CE **11.0965 → 4.7427** (chance 11.090), entropy **11.09 → 6.21**, target-cell probability ×2 500, point error **0.4658 → 0.1673**, bounded margin **−0.000 → +0.1385**, but max **non-target** probability 0.0532 vs target 0.0515 |
| H1-R / H2-R | **not run** (section 13 stops at the failed H0-R) |

1. **The corrected objective is healthy and not degenerate**: the point cross-entropy falls from
   exactly chance to 4.74, the softmax sharpens (entropy 11.09 → 6.21 nats), the target cell gains
   ~2 500× probability, the argmax point error falls 0.466 → 0.167, and the bounded preference margin
   becomes clearly positive with own mass 4× cross mass. Task 6H's magnification shortcut is gone.
2. **H0-R still fails**: the largest non-target cell holds 0.0532 probability against a 0.0515 mean
   target probability, so top-1 is 25 % and inside-target 35 % — far from the 90 % gate. The single
   `[BOX]` query does not carry enough target-specific spatial signal even under a directly aligned,
   bounded objective.
3. **Task 6H's verdict and numbers are preserved**; only its causal wording was narrowed (section 3):
   6H learned the specified logit ranking but the loss was scale-degenerate, so it did not establish
   that the query signal is unlearnable — 6H.1 supplies that bounded test and shows real but
   insufficient spatial signal.
4. **Accepted verdict (section 22): `BOUNDED_POINT_OBJECTIVE_FAILED`** → per section 23 this is a
   clean failure and the task stops for review: the next candidates are representation-level
   (instruction-aware multi-query / iterative refinement, stronger MLLM, reference/relation
   grounding), **not another scalar loss**, and none may be implemented automatically. No SAM2
   training, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.

## Task 6H measured results

Task 6G's architecture is frozen; Task 6H changed only the training semantics: one optimizer step per
canonical same-image counterfactual pair (240 pairs / 240 images from the Task 6C `P` subset) plus an
own-vs-cross region-ranking loss. Full detail: `docs/task6h_counterfactual_pair_grounding.md`,
ADR-019, `evaluation/task6h_*.json`.

| | Value |
|---|---|
| Canonical pairs | **240 / 240 images**, all assertions enforced, identity hash `c56f0507da41fd55…`, 0/240 pairs overlap, no pair discarded |
| Pair step | one shared frozen SAM2 feature (bit-identical through the step), two query forwards with retained graphs, **one** backward; horizon counts **pair** steps (240/epoch) |
| Loss | `0.5*(reasoning_A+reasoning_B) + 1.0*(heatmap_A+heatmap_B) + 2.0*L_cf`, `L_cf` margin 1.0 on `s_XY = sum(H_X·M_Y)/sum(M_Y)` (logits) |
| Architecture check | identical to Task 6G (level `[1,32,256,256]`, head 270 593 params); `[BOX]` still causally clean |
| **H0** (10 pairs, 1500 pair steps) | pair ranking **10/10**, strict margin 10/10, mean margin **+5.58** — but point-inside-own **3/20** and paired point **0/10** → **failed** |
| H0 audit | `L_cf` 1.3133 → 0.0212, margin (logits) −0.0001 → +5.5788, **mean abs logit 0.148 → 16.32**, probability margin only **+0.112**, peakiness 1.6e-05 → 0.403, point-inside 0.00 → **0.15** |
| H1 / H2 | **not run** (section 12 stops on H0 failure) |

1. **The pair machinery is correct and verified**: canonical pair construction with hash and
   zero-overlap audit, one shared frozen feature, single backward per pair, correct loss sign,
   `L_cf` gradients reaching every trainable group (query projection, query norm, visual 1×1, Qwen
   LoRA, token rows), SAM2 bit-frozen through the step, Task 6G head unchanged.
2. **The specified ranking objective is scale-degenerate.** H0 satisfies ranking 10/10 and margin
   +5.58 by growing the heatmap logit scale ~110× while the probability-space margin stays +0.112
   and the argmax lands inside the target 15 % of the time. Error classification: 10/10 failing
   pairs are `pair_ranking_correct_point_outside`.
3. **Five readouts now fail with verified implementations** (6D.1 `[SEG]` probes, 6E coordinate
   tokens, 6F box regression, 6G dense map, 6H ranked dense map). Task 6H's contribution is the
   diagnosis: a future pair/contrastive objective for this project must be defined on a **bounded**
   scale (probability/softmax) or carry an explicit peak-sharpness term, otherwise the ranking gate
   keeps passing while localization stays at chance.
4. **WHU is not the binding constraint** of this result (the failure is uniform at 10/10 pairs on
   memorized records). No SAM2 training, no `[REF]`/SRE/SCL/4B/dataset migration/GUI.
5. **Machine verdict (section 20): `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`** — kept as the task's
   original verdict. **Corrected causal interpretation (Task 6H.1 section 3):** Task 6H successfully
   learned the specified own-vs-cross *logit* ranking, but the loss was **scale-degenerate** — it could
   be minimized by magnifying logits without moving the heatmap peak onto the target. Query
   learnability therefore remained **unresolved** until a bounded, localization-aligned objective was
   tested; Task 6H.1 provides that test.

## Task 6G measured results

Task 6F's global box regression is retired; Task 6G keeps the causally clean pre-reasoning `[BOX]`
query but matches its hidden against the frozen SAM2 dense spatial features to produce a target
heatmap whose argmax is the predicted point. Full detail: `docs/task6g_dense_spatial_grounding.md`,
ADR-018, `evaluation/task6g_*.json`.

| | Value |
|---|---|
| Grid oracle (snapped point → frozen SAM2) | 64: 0.4950/17-20; 128: 0.4894/17-20; **256: 0.4883/18-20** (selected: smallest with paired ≥ 18/20 and mIoU ≥ 0.4576) |
| Selected level | 256×256 = the **32-channel** SAM2 high-res feature (by spatial size, never list order) |
| Head | `LayerNorm→Linear(2048,128)` on `[BOX]`; `Conv1x1(32,128)` on frozen features; `dot/√128` + scalar bias (270 593 params) |
| Loss | `1.0·L_reasoning + 2.0·(1.0·BCEWithLogits + 1.0·SoftDice)` on the area-downsampled target mask |
| One-step smoke | `[BOX]`/`[SEG]` rows + all head params move with gradient; **SAM2 bit-frozen through a training step**; ordinary rows exactly unchanged; causal bit-identity re-verified |
| **G0** (20 records, 1500 steps) | inside-own **0.10**, heatmap Dice **0.125**, paired point selection **0/10** — gate failed at budget exhaustion |
| G0 implementation audit | **no defect**: train/eval query hiddens bit-identical; logits healthy (std 3.3); BCE/Dice gradient norms comparable (0.395/0.456); heatmap = near-flat image-dominated field (peakiness 0.086, errors 44–225 px) |
| G1 / G2 | **not run** (section 11 stops on G0 failure) |

1. **The point-oracle pathway is verified at every grid**: snapping the frozen interior point to
   256×256 cell centres costs nothing (0.4883 vs 0.4876 continuous, paired 18/20, 0.50 px mean
   displacement). The downstream frozen SAM2 point path is not the problem.
2. **The dense head is implemented faithfully and audited, not assumed**: train/eval bit-equivalence,
   healthy logits, comparable BCE/Dice gradients, SAM2 bit-frozen — yet **G0 cannot overfit 20
   records**. The heatmap converges to a near-flat image-dominated field (Dice term never moves from
   ~0.97), and both instructions of one image pin the identical argmax cell.
3. **Four readouts now fail identically**: 6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box head, 6G
   dense map — each implementation-verified, each collapsing to an image-dominated query. The
   measured bottleneck is the **query's instruction-dependent signal**, not the readout format.
4. **WHU is not the binding constraint** (diffuse heatmaps cannot be explained by target size/border
   flags; L1 simple cases fail too). No SAM2 training, no `[REF]`/SRE/SCL/4B/dataset change/GUI.
5. **Accepted verdict (section 18): `DENSE_GROUNDING_IMPLEMENTATION_FAILED`** (G0 cannot pass after
   the implementation audit; task stops before G1 per section 11).

## Task 6F measured results

Task 6E's coordinate vocabulary is retired; Task 6F inserts a fixed learned `[BOX]` query before
reasoning and regresses the box from its hidden state. Full detail:
`docs/task6f_target_aware_box_query.md`, ADR-017, `evaluation/task6f_*.json`.

| | Value |
|---|---|
| Vocabulary | `[SEG]` 151 669 + exactly one new `[BOX]` 151 670 (no `<loc_*>` tokens added) |
| Trainables | text LoRA (17 432 576) + `{[SEG], [BOX]}` rows (4 096) + `TargetAwareBoxHead` (1 055 236) |
| Loss | `1.0 · L_reasoning + 5.0 · L_box` (SmoothL1 vs Task 6D tight box), no mask loss |
| Causal placement | `[BOX]` at `prompt_length`, never predicted; hidden **bit-identical** under future-text mutation |
| F0 (20 records, ≤1500 steps) | train box IoU **0.9046**, paired **10/10**, non-identical same-image boxes (L1 0.2285) at step 1000 → `F0_PASS` |
| F1 (480 paired, ≤8 epochs) | train box IoU **0.050** (never converged), val box IoU **0.025**, center-inside **0.017**, paired **0/20** at epoch 8/8 |

1. **The query mechanism itself is verified, not assumed.** `[BOX]` sits at the generation prefix
   and is never a prediction target; replacing the entire reasoning tail leaves its hidden
   **bit-identical** (recorded on the real model). The one-step smoke shows both token rows and all
   six head parameters moving with non-zero gradient while ordinary rows and the base table stay
   exactly unchanged, and SAM2 stays frozen.
2. **F0 passes decisively** (train box IoU **0.9046**, paired **10/10**, same-image non-identical
   boxes with L1 0.2285 at step 1000), so the implementation is not the problem.
3. **F1 underfits rather than overfits**: the box head never fits even its own 480 training records
   (train SmoothL1 plateaus ≈0.003 ≈ 8 % RMS coordinate error; per-step train IoU 0.0 at most logged
   steps), the predictions are canonical and spread over the image but wrongly placed (val coord MAE
   0.154, box IoU 0.025, paired 0/20). Same pattern as Task 6E, now on the regression side: 20
   records memorize, 480 do not fit inside the spec's budget.
4. **Representation answer (section 13):** the query is no longer template-dominated, but it is
   **image-dominated** — same-image centered cosine 0.626 vs 0.018 across images — with moderate
   target-geometry variation (hidden-vs-GT-box distance correlation 0.476 vs ≈0 for the legacy
   `[SEG]`). Effective rank 7.07 vs legacy 8.33.
5. **Language compatibility is perfect** (exactly-one `[SEG]` 1.000, EOS 1.000, known template
   1.000), and **F2 was correctly not run** (section 15 gate). Error analysis: 120/120 failures,
   dominant class `wrong_target_valid_box` (98 %), `tiny_target` on 93 % — the localization-precision
   difficulty of WHU-scale targets is real, but the F1 failure is primarily budget-limited
   under-fitting, not a data defect.
6. **Accepted verdict (section 17): `TARGET_AWARE_QUERY_FAILED`**, with no escalation attempted
   (no `[REF]`, SRE, SCL, 4B, dataset change, full training, GUI) and SAM2 never trained. Task 6E's
   coordinate machinery stayed retired throughout.

## Task 6E measured results

The box became an explicit autoregressive target (`reasoning [BOX] <loc_x1> <loc_y1> <loc_x2>
<loc_y2> [SEG]`) over one shared 256-token location family, with `L = 1.0·L_assistant +
5.0·L_location` and no mask loss. Full detail: `docs/task6e_explicit_spatial_tokens.md`, ADR-016,
`evaluation/task6e_*.json`.

| | Value |
|---|---|
| Quantized oracle, `B = 32 / 64 / 128 / 256` | 0.5400 / 0.6473 / 0.7129 / **0.7343** mIoU, all paired **20/20** |
| Selected bin count | **`B = 256`** (smallest with paired 20/20 and mIoU ≥ `0.7506 − 0.03`) |
| Added vocabulary | `[BOX]` + `<loc_000>…<loc_255>` = **257** tokens; **258** trainable rows ([SEG] + [BOX] + all loc) |
| E0 (20 records, ≤1500 steps) | structural **20/20**, exact four-token **20/20**, box IoU **0.9166**, paired **10/10** at step 500 → `E0_PASS` |
| E1 (480 paired records, 3 epochs × 480 steps) | structural **0/120** at every epoch, paired **0/20** |
| E1 teacher-forced | reasoning 1.000, `[BOX]` 1.000, EOS 1.000, **loc 0.075, `[SEG]` 0.000** |
| E1 free generation | exactly one `[BOX]` in 120/120, then a **51–81-token location run**, `[SEG]` count 0 in 120/120 |
| E1 train location CE | 9.64 → **5.15** (256-way uniform floor `ln 256 = 5.545`) |
| E2 strict e2e segmentation | **not run** (section 14 gates it on the E1 gate) |
| Error classification | 120 structural, **0 localization**, 0 quantization-limited, `whu_data_quality_dominates: null` |

1. **The pathway is verified up to the box, and the box is where it stops.** Given a 256-bin
   quantized box, the frozen SAM2 prompt path still reaches **0.7343 mIoU / paired 20/20** (97.8 % of
   the continuous-oracle 0.7506). The tokenizer, the 258 trainable rows, the label/loss plumbing and
   the token-id parser are verified by a real one-step smoke and by E0's 20/20 exact-token overfit.
2. **E1's failure is a vocabulary-learning failure, not a formatting or leakage failure.** Teacher
   forcing reaches 1.000 on reasoning, `[BOX]` and EOS but only 0.075 on the location values and
   **0.000** on `[SEG]`; free generation therefore emits `[BOX]` and then a runaway location run with
   no `[SEG]`, so no sequence is structurally valid.
3. **The same code, loss and schedule overfit the first 20 of those same 480 records to 20/20** (E0),
   so this is an optimization/budget limit of the fixed ≤3-epoch schedule on a new 256-way vocabulary,
   not an implementation defect. The location CE does not even fit the *training* set (5.15 vs 5.545
   uniform floor).
4. **Neither the dataset nor SAM2 is the constraint.** 0/120 localization failures, 0 samples whose
   quantized GT box is below 0.5 IoU, and every failure is structural.
5. **Accepted verdict (section 16): `EXPLICIT_SPATIAL_TOKENS_FAILED`**, with no escalation attempted
   (no `[REF]`, SRE, SCL, 4B, dataset change, full training, GUI) and SAM2 never trained.


## Task 6C measured results

Valid, bit-reproducible 2×2 on the frozen ADR-013 stack. Same recipe, same seed, same validation material;
only the training subset (U/P) and the SAM bridge (C/L) differ. Full detail:
`docs/task6c_prompt_ablation.md`, ADR-015, `evaluation/task6c_comparison.json`.

| Arm | strict e2e mIoU | Dice | emission | paired | mean margin | projected effective rank | top-1 variance |
|---|---|---|---|---|---|---|---|
| `U_C` (Task 6B baseline) | 0.10516 | 0.17360 | 120/120 | **0/20** | −0.000001 | 1.516 | 0.9023 |
| `U_L` (language bridge) | 0.10872 | 0.17846 | 120/120 | **0/20** | −0.003530 | 1.491 | 0.9087 |
| `P_C` (paired data) | 0.10604 | 0.17851 | 120/120 | **0/20** | +0.000007 | **4.100** | **0.5892** |
| `P_L` (both) | 0.09284 | 0.15626 | 120/120 | **0/20** | −0.000003 | **3.371** | **0.6430** |

## Task 6C.5 measured results

Performance-only task; it changed no model, loss, optimizer, scheduler, data or sample order, and it does not
reinterpret Task 6C. Full detail: `docs/task6c5_training_optimization.md`,
`evaluation/task6c5_variants.json`, `evaluation/task6c5_final_benchmark.json`.

| | B0 (current) | Adopted (`skip_grad_norm_instrumentation`) |
|---|---|---|
| Throughput (interleaved, 2 rounds, 8 warmup + 64 steps, batch 1) | 2.152 samples/s | **2.373 samples/s → +10.24 %** |
| Repeat spread | 4.96 % | 0.47 % |
| GPU utilization | 37.3 % | 37.5 % (unchanged) |
| CPU utilization | 84.4 % | 92.5 % |
| Single clean sweep | 2.506 samples/s | 2.847 samples/s (+13.63 %) |
| Peak RSS / reserved VRAM | 3.30 GiB / 8.64 GiB | 3.29 GiB / 8.64 GiB |

1. **The CPU≈100 % / GPU≈40 % cause is per-operator launch overhead inside Qwen forward and backward, not
   data preparation.** Synchronized profiling puts `qwen_forward` 50.1 %, `backward` 38.5 %, `optimizer_step`
   7.7 % against **all host-side preparation ≈2.4 %** (processor 0.76 %, image I/O 0.62 %, SAM feature
   lookup 0.52 %, target-mask I/O 0.24 %, H2D 0.21 %). The processor costs 2.21 ms/sample against a ≈400 ms
   step.
2. **Every data-side candidate was measured and rejected**: source cache −5.80 %, Qwen preprocessing cache
   −2.92 % (it halves CPU utilization 90 → 56 % and still loses), pinned + non-blocking −12.98 %, prefetch
   with 2/4/8 threads −5.46/−5.69/−9.31 %. The preprocessing cache is the clean proof that the hypothesis was
   wrong: CPU utilization is not the binding constraint.
3. **The adopted change is one switch**: drop the per-step 528-tensor gradient-norm sweep that the Task 6C
   loop computes and discards. Default stays `collect_grad_norms=True`, so Task 6A/6B and their tests are
   unaffected. Bit-equivalent gate: prepared tensors (0 mismatches / 16 samples), 12-step losses, gradient
   fingerprints and post-step parameter fingerprints all identical, baseline control reproducible. A
   separate refactor control proves the audit's own restructuring of `train_step` (optional stage timers,
   split `torch.autocast` regions) is value-neutral against the pre-6C.5 step body.
4. **Rejected caches were still value-preserving.** `source_cache + preprocessed_cache + skip` is also
   bit-equivalent (second gate artifact) and 2.29 % faster than the adopted variant — inside the benchmark's
   own 4.96 % spread — so three extra switches were not taken for an unresolvable difference (section 20's
   ~5 % complexity rule, recorded as `adoption_rule`).
5. **Gradient checkpointing stays ON** (−8.58 % when off) and **strict determinism stays ON** (the
   determinism tax is only +1.78 %, well below the gate).
6. **Remaining bottleneck is batch-1 launch serialization**; data-side work is closed out. Raising arithmetic
   intensity per launch (batch > 1) is the only evidence-backed lever and is explicitly a different
   experiment, not a Task 6C.5 optimization.
7. **Measurement defect found and fixed**: the LoRA adapters' `dropout = 0.05` consumes the global RNG, so an
   equivalence gate must **re-seed per run** — with a single start-up seed the baseline did not reproduce
   itself. Any future equivalence claim needs the same per-run re-seeding rule.
8. **Network / Watt for 6C.5**: the accelerator was **pre-existing** (`watt_preexisting = true`) and was only
   used to push; nothing was closed, force-killed, or reconfigured, no hosts file was edited and no TLS
   verification was disabled. The MVP stack stayed offline (`HF_HUB_OFFLINE=1`) throughout.

## Task 6C.6 measured results

Performance-only task. The one runtime change it adopts is wiring Task 6C.5's winner into the formal
training loop; every other candidate was measured and rejected. Full detail:
`docs/task6c6_launch_optimization.md`, `evaluation/task6c6_*.json`.

| | Value |
|---|---|
| Formal-path integration | `training.collect_grad_norms: false` in `configs/mvp/task6c_2b_ablation.yaml`, passed explicitly by `scripts/task6c_train.py` |
| Integration equivalence | **`BIT_EQUIVALENT`** (prepared tensors, 12-step losses, gradients, post-step parameters) |
| Integrated baseline B0.6 | **2.505 samples/s** (final head-to-head mean; 2.545 / 2.465) |
| Pre-integration control | 2.184 samples/s (2.188 / 2.179; spread 0.41 %) |
| Integration gain (interleaved) | **+14.7 %** (+21.5 % in the 4-run baseline group) |
| New candidates adopted | **none** |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 3.295 GiB RSS, no OOM |

1. **The step is dispatch/launch dominated, now measured rather than inferred**: **57,341 CUDA kernels
   per step** at a **2.1 µs median**, **2,545 host↔device synchronizations per step**, the CUDA launch
   path consuming **31.9 % of CPU self time**, and the GPU only ~33 % utilized in unprofiled runs. Host
   data preparation is ≈2.4 % of the step (Task 6C.5) and is not a factor. The gradient-norm sweep that
   Task 6C.5 removed accounted for 3,589 kernels and 2,048 synchronizations per step.
2. **`torch.compile` cannot run on this install.** The inductor backend fails with `TritonMissing`
   (no Triton package, no MSVC). Everything that does run is slower: `cudagraphs` −24.7 %, combined
   `aot_eager` −21.2 %, decoder tail −11.3 %, Qwen `aot_eager` −10.4 %, Qwen `eager` −9.2 %. All compile
   candidates are also `NOT_EQUIVALENT` (max loss difference 0.042–0.234 against a 1e-3 tolerance), all
   inflate reserved VRAM by ~4 GiB, and `combined_aot_eager` exceeds the 14 GiB budget at 14.41 GiB.
   `backend="eager"` and `cudagraphs` additionally fail on a **sequence-length change** (291/321/295/318
   tokens) with a tensor-size mismatch — CUDA graphs are unsafe here without fixed-length padding, which
   is out of scope.
3. **Optimizer/clipping is already optimal.** `_default_to_fused_or_foreach` resolves to
   `foreach=True` for the 528 fp32 parameters, so `foreach=True` is a no-op; `fused=True` is inside the
   noise band and `NOT_EQUIVALENT`; `foreach=False` is slower. Two *code-identical* controls
   (`adamw_foreach`, `clip_foreach_true`) measured −2.85 % and −3.37 %, which fixes the **noise floor of
   a sequential group at 3.37 %** on this machine.
4. **SDPA is already mixed and correct.** Per step, 472 attention calls split into **248
   memory-efficient CUTLASS FMHA** and **224 math**, with 0 flash and 0 cuDNN. FlashAttention is not
   compiled into this PyTorch build; forcing flash or mem-efficient fails with `No available kernel`, and
   forcing math changes the numerics (9.4481 vs 9.4255) and is ~9.7 % slower. No accidental fallback to
   fix, no win available.
5. **Gradient checkpointing stays ON.** Interleaved ON/OFF gave OFF −2.03 % with the two rounds
   disagreeing in sign (spread ON 5.76 %, OFF 1.32 %) → Task 6C.5's −8.58 % single-sweep figure does not
   survive interleaving.
6. **This laptop's absolute throughput is not a stable quantity** (up to 19 % between identical runs;
   the first run in a process is systematically fastest). The reproducible quantity is the interleaved
   *ratio*, which is why every comparison brackets its candidates with reference runs and compares
   against the interpolated reference, and why a non-resolvable gain is never adopted.
7. **Methodology carried forward**: A/B/A/B interleaving for headline numbers, bracketed+interpolated
   references for screening, per-run re-seeding in every equivalence gate, and `PYTHONUTF8=1` when
   reading `torch.compile` errors on this Windows locale (otherwise the real `TritonMissing` cause is
   hidden behind a `UnicodeDecodeError`).

## Task 6D.1 measured results

Corrective audit of Task 6D. Full detail: `docs/task6d1_corrective_grounding_audit.md`,
`evaluation/task6d1_*.json`.

| | Value |
|---|---|
| Scheduler defect | horizon was one epoch (480) while two ran → **epoch 2 LR exactly 0**; end-of-epoch-1 factor 1.17e-05 |
| Scheduler fixed | horizon 960 → epoch-2 start factor 0.517, terminal only at the final step (5.4e-06 of peak) |
| G0-R (scheduler fix only) | emission **120/120** ✅, geometry paired **0/20** ❌, box IoU **0.0088** ❌ → G1 not run |
| Probe A (linear, raw) | overfit-20 0.903, train-480 0.027, val 0.004, paired 0/20 |
| Probe B (raw MLP, no LayerNorm) | overfit-20 0.645, train-480 0.036, val 0.008 |
| Probe C (Task 6D LayerNorm head) | overfit-20 0.778, train-480 0.051, val 0.007 |
| Label-shuffled control | real ≈ shuffled at 480 (0.027 vs 0.020; 0.036 vs 0.020); shuffled also fits 20 samples to loss 0 |
| LayerNorm | **not a confound** — with a fair LR the LayerNorm head matches/beats the raw MLP |
| Representation | norm ≈112, effective rank **8.33**, top-1 31.6 %; centered cosine same-image/diff-template **0.077** vs diff-image/same-template 0.531 |
| Verdict | **Case F — `PRACTICALLY_NOT_DECODABLE_GEOMETRY`** |

1. **The scheduler defect was real but not causal.** Task 6D built the cosine scheduler with one epoch as
   its horizon and ran two, so epoch 2 trained at LR 0 (which is why its epoch-1/epoch-2 metrics were
   bit-identical). The corrective rerun G0-R fixes only that and fails the same gate.
2. **LayerNorm is not a confound, and neither was the head.** The first probe run appeared to show
   LayerNorm destroying the signal; that was **my probe failing to converge** (plus unscaled norm-112
   inputs). After fixing both, all three readouts converge and the LayerNorm head performs as well as the
   raw MLP.
3. **The 20-sample overfit is not evidence**: a shuffled-label readout fits 20 permuted boxes to zero loss
   too. The informative number is the 480-sample fit, where the real fit is within a few IoU points of the
   shuffled fit.
4. **Accepted statement (weaker, per the spec's wording rule)**: after the implementation audit, the frozen
   `[SEG]` representation of the Task 6C `P_C` checkpoint contains **no practically decodable target
   geometry** under this training setup. This is not an information-theoretic absence claim, and it
   replaces Task 6D's "carries no target location" wording.
5. **The residual encodes the instruction wording, not the target**: with the shared component removed, the
   two instructions of one image are nearly orthogonal (centered cosine 0.077) while two different images
   with the same template are much more similar (0.531), and the within-image token variance (0.00999) is
   ~30× smaller than the between-target box L1 (0.306).
6. **Oracle conclusion unchanged**: correct geometry → 0.7506 mIoU, paired 20/20. The downstream machinery
   works; the failure is in what the read-out can express.

## Task 6D measured results

Architecture task: the first explicit spatial-grounding mechanism, and the first measurement that
localizes the Task 6C failure to a specific component. Full detail:
`docs/task6d_spatial_grounding_bridge.md`, `evaluation/task6d_*.json`.

| | Value |
|---|---|
| Oracle point (GT point prompt → SAM2) | mIoU **0.4876**, Dice 0.6055, paired **18/20** |
| Oracle box (GT box prompt → SAM2) | mIoU **0.7506**, Dice 0.8507, paired **20/20**, IoU(pred_A, pred_B) **0.0000** |
| Section 5 geometry choice | **BOX** (point missed the 0.50 mIoU gate; box passed it and led by +0.2630) |
| G0 (2 epochs × 480 paired records) | emission **120/120** ✅, geometry paired **0/20** ❌, box IoU **0.0082** ❌ |
| G1 | **not run** (section 9: stop on G0 failure, do not compensate) |
| Free-generation strict e2e mIoU | **0.0066** (Task 6C `P_C` reference 0.10604) |
| Paired mask probe | **0/20**, own ≈ cross ≈ 4e-10, IoU(pred_A, pred_B) **0.919** |

1. **The segmenter was never the problem.** Given the correct geometry, SAM2 recovers the target at
   **0.7506 mIoU** (box) against Task 6C's **0.10604**, with a paired probe of 20/20 and *zero* overlap
   between the two targets' predicted masks. The Task 6C failure was entirely in prompt generation.
2. **Superseded wording (Task 6E §2.1; accepted statement in Task 6D.1 §16): under the audited Task 6C
   training setup, the frozen `[SEG]` representation contains no *practically decodable* target geometry
   with the tested readouts. This is not an information-theoretic absence claim.** The Task 6D evidence: a
   1.05M-parameter head supervised on
   480 GT boxes did not fit even the training boxes; by epoch 2 it collapsed to the single constant box
   `(0.457, 0.455, 0.547, 0.539)` with per-coordinate spread `[0.000, 0.000, 0.001, 0.000]` — the optimal
   constant predictor under SmoothL1 for an uninformative read-out.
3. **The read-out is template-dominated, not merely instruction-independent.** Four-condition control on
   `[SEG]` hidden cosines: different image + **same template** 0.99988 (most similar) vs same image +
   different template 0.99898 vs different image + different template 0.99869 (least similar). Changing
   the whole image moves the hidden state *less* than changing the instruction wording.
4. **The language objective cannot fix it.** LM CE saturates at 0.0000 because the assistant target is one
   of only 21 distinct reasoning templates (Task 6C's finding), so it is memorized long before it could
   pressure the read-out to encode *which* building.
5. **This reproduces Task 6C with a different read-out.** Task 6C measured the projected prompt at
   effective rank 1.5 with same-image cosine > 0.9999; Task 6D measures a *supervised geometry head* on
   the raw `[SEG]` hidden and finds the same collapse. Two independent read-outs agreeing makes the
   diagnosis strong.
6. **Dataset is not the binding constraint (yet).** The 116/120 failures carry `tiny_target` (93 %) and
   `border_truncation` (30 %) flags, but the artifact's degeneracy guard shows all of them fail from one
   constant mask regardless of content, so a dataset-selection task is not indicated by this evidence.

## Task 6C.7 measured results

Performance-only task; two changes adopted, and the model/loss/data/optimizer semantics are untouched.
Full detail: `docs/task6c7_visual_cache_optimization.md`, `evaluation/task6c7_*.json`.

| | Value |
|---|---|
| Section 3 Phase-B duplicate H2D | **removed** (exact; also fixed a `del moved` `UnboundLocalError` the refactor introduced) |
| Frozen Qwen visual-feature cache | **adopted** — `BIT_EQUIVALENT`, **+9.94 %** warm paired throughput |
| Cache footprint | 4.0 MiB/image (`pooler_output` 1 MiB + 3 × `deepstack_features` 1 MiB, bf16), bound 512 images = **2.0 GiB** |
| Kernel / sync change | **−8.44 %** CUDA kernels, **−43.47 %** host↔device sync ops per step |
| Adopted runtime footprint | 8.582 GiB reserved VRAM, 4.751 GiB RSS + 128 MiB cache, no OOM |
| Config | `training.visual_feature_cache: true`, `visual_feature_cache_max_images: 512` |

1. **The visual tower is provably cacheable.** 315 visual parameters, all bf16 and all frozen; zero LoRA
   modules in the tower (`lora.text_only: true`); no dropout, so it consumes no RNG; recomputation is
   bit-identical (max abs difference 0.0); nothing in the tower moves during a real optimizer step
   (parameters *and* buffers checked); and 8 same-image instruction pairs produce identical
   `pixel_values` **and** identical visual features — the tower is instruction-independent.
2. **The cached boundary is the narrowest available**: `Qwen3VLModel.get_image_features(...)` output,
   i.e. `pooler_output` + `deepstack_features`, before they are mixed with trainable text hidden states.
   The wrapper is an instance-level replacement on the project's own Qwen module with the original bound
   method preserved, so the miss path is byte-identical. Nothing downstream of a trainable parameter is
   cached, and `last_hidden_state` is deliberately not cached (the language model never reads it).
3. **Equivalence is bit-exact** over 16 samples' features and a 12-step gate with per-run re-seeding, for
   both key strategies (source-image identity and pixel-content hash). First-step loss identical.
4. **The gain is real but only visible under a drift-robust design.** Whole-variant runs drift by −24.6 %
   and the four-run interleaved V1/V2 comparison produced rounds disagreeing in sign (+15.7 % V1, then
   +8.9 % V2). A paired ablation — alternating 8-step cache-off/cache-on blocks inside one runtime — gives
   **+10.41 % (all pairs) / +9.94 % (warm), 4/4 pairs in favour**. Warming the cache while it is disabled
   stores nothing, and that mistake alone had reported +2.31 % and would have rejected the change.
5. **The remaining ~1,051 scalar read-backs are PyTorch's, not ours.** Scoped windows plus a
   `sys.setprofile` `c_call` tracer attribute 99.5 % of them to `optimizer.step()` →
   `torch/optim/adam.py:770-776` converting each parameter's **CPU-hosted** `step` counter twice per step
   (`_get_value` → `.item()`); the upstream comment states the CPU hosting is deliberate. The optimizer
   window issues 1,024 reads with **one** sync op, so they are cheap; backward contributes 5 reads;
   disabling gradient checkpointing changes nothing. `fused=True` would remove them, but Task 6C.6
   measured it at −0.71 % and it is `NOT_EQUIVALENT`, and section 4 forbids patching third-party
   internals. Project-owned reads are ~7/step (<1 %), so V3 (deferred scalar logging) was not implemented.
6. **Batch-1 optimization is now exhausted on evidence**, which is what makes the batching recommendation
   earned: the redundant H2D is gone, the sync traffic is localized and dismissed, and the last large
   removable win is adopted. What remains is execution shape (~52,500 kernels/step, ~2 µs median, GPU
   ~40 %).

## What Task 6C changed in the project's understanding

1. **Neither of Task 6B's two confounds is the cause.** All four arms are 0/20 on the paired unseen probe
   with mean margins within ±0.0036 of zero. Paired counterfactual training, the removal of the fixed
   centre point, and their combination all fail to create instruction-conditional mask selection.
2. **But the factors are not inert.** Paired training materially changes the prompt representation
   (effective rank 1.50 → 3.74, top-1 variance 0.905 → 0.616) without changing the masks. The projection
   uses more dimensions; two instructions on one image still produce the same mask.
3. **The fixed positive centre point is not the culprit.** Its norm is 11.39 against a projected language
   norm of 266.8 — a **23.4×** ratio in the language vector's favour — and removing it (the `L` bridge)
   changes little. "The centre point overrides the language" is not supported by the measurement.
4. **The collapse is directional and low-rank, not constant.** Same-image projected cosine > 0.9999 with
   top-1 variance 0.59–0.91 and a non-zero effective rank. Task 6B's looser wording is corrected.
5. **The remaining problem is after the prompt**, i.e. in how the `[SEG]` hidden state is formed and how
   SAM's decoder turns a prompt direction into a region. Prompt normalisation, a multi-token prompt,
   auxiliary point supervision or `[REF]` are justified now, and were not before.
6. **Determinism is real now.** `training.deterministic` is consumed, strict deterministic algorithms are
   active with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and two independent processes produce identical losses,
   gradients and post-step parameters. The `U_C` arm was run before and after the cache fix and is
   bit-identical, which also proves the fix value-preserving.

## Correctness fixes carried forward (do not regress)

1. Determinism (above).
2. Per-arm checkpoint directories come from `cfg["paths"]["checkpoints"]` — no module-level constant.
3. The operation-chain lookup is built from the **train split only**; validation text never extends it.
4. The SAM2 CPU feature cache holds all 480 training images inside the budget by sharing the two
   image-independent tensors once (7.508 GiB instead of 11.25 GiB), and it never mutates stored entries.

## Measured limitations (carry into Task 6I)

1. **Instruction conditioning of the mask is still absent**, and Tasks 6D + 6D.1 localize it precisely:
   the frozen `[SEG]` hidden state is norm-112, effective-rank 8.33, and its instruction-dependent residual
   (centered cosine 0.077 between the two instructions of one image) is **not practically decodable**
   into target geometry — three converging readouts cannot fit even the 480 training samples better than a
   label-shuffled control. Task 6D's original wording ("carries no target location") is superseded by this
   audit (§16 of Task 6D.1).
2. **Five geometry readouts fail with verified implementations, and Task 6H identifies the objective as
   the culprit, not the plumbing.** The legacy `[SEG]` probes (6D.1), the coordinate vocabulary (6E:
   structural 0/120), the global box regression (6F: val box IoU 0.025), the dense map (6G: G0 inside
   0.10) and the counterfactually ranked dense map (6H: ranking 10/10 but inside-own 3/20) all collapse
   to an image-dominated query. Task 6H's audit shows the specified own-vs-cross margin on **unbounded
   logits** is minimized by magnifying the field (mean |logit| 0.148 → 16.32, probability margin only
   +0.112). A future ranking/contrastive term must be bounded or carry a peak-sharpness term, and the
   query's instruction dependence must be strengthened deliberately.
3. **Mask quality is not usable**: strict end-to-end mIoU 0.093–0.109, L3 nontrivial 0.076–0.088.
4. **Language metrics are template metrics**: `reasoning_zh` has 21 distinct values in the whole training
   mini-set, so exact match and operation-chain accuracy cannot support a reasoning claim.
5. **More prompt diversity did not become mask diversity**: effective rank 3.4–4.1 still yields
   IoU(pred_A, pred_B) 0.999.
6. **The best mIoU arm is not the most diverse arm** (`U_L` 0.1087 vs `P_L` 0.0928), so mIoU alone remains
   a misleading selection signal.
7. **Training throughput is launch-bound at batch 1** and the removable overhead is now exhausted:
   the adopted runtime is Task 6C.6's integration plus Task 6C.7's frozen visual-feature cache, worth
   +9.94 % on top (paired ablation). Task 6H adds one datapoint: a pair step (two query forwards, one
   shared frozen feature, one backward) costs ~1.6× a single-sample step, so 1500 pair steps ≈ 29 min.

## Current blockers

**The single `[BOX]` query does not carry enough target-specific spatial signal.** Six readouts have
failed with verified implementations (6D.1 `[SEG]` probes, 6E coordinate tokens, 6F box regression,
6G dense map, 6H logit-ranked dense map, 6H.1 point-aligned bounded dense map). The last two rule out
the readout format (6G: image-dominated feature map) and the objective's scale/sign (6H.1: point CE
falls to 4.74 and the bounded margin is clearly positive, yet the max non-target cell still rivals the
target cell). What remains is the query representation itself and the strength of the 2B MLLM's
instruction-conditioned spatial signal. Per Task 6H.1 section 23, further scalar-loss redesign is
**not** indicated.

## Recommended next task

**ChatGPT review of the pushed Task 6H.1 results**, then **exactly one** representation-level task,
with the evidence Tasks 6D.1–6H.1 produced:

1. **Instruction-aware query refinement (recommended).** Multiple query tokens (or a small number of
   refinement steps) over the frozen visual features, so the instruction can select among competing
   buildings in one image; the canonical pair material, the frozen point oracle ceiling and the
   point-CE + bounded-mass objective are ready-made measurement infrastructure.
2. **Stronger MLLM representation** (e.g. a larger Qwen) if the 2B hidden is the binding constraint —
   a scale question that the project has so far refused to answer by default and should now be posed
   explicitly against the measured bottleneck.
3. **Architecture-level reference/relation grounding** (`[REF]`-style reference tokens, SRE) only
   after a target-selection gate passes; the current evidence still does not justify SRE/SCL.
4. **Not indicated**: another scalar loss, dataset migration, full training, GUI.

Full detail: `handoff/FROM_DSH.md`, `docs/task6h1_bounded_point_counterfactual.md`,
`docs/task6h_counterfactual_pair_grounding.md`, `docs/task6g_dense_spatial_grounding.md`,
`evaluation/task6h1_verdict.json`, `evaluation/task6h1_objective_setup.json`,
`evaluation/task6h1_h0r_overfit.json`, `evaluation/task6h1_h0r_audit.json`,
`evaluation/task6h1_error_analysis.json`, `evaluation/task6h_*.json`, `evaluation/task6g_*.json`,
`docs/architecture_decisions.md` (ADR-016 … ADR-020).
