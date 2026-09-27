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

# FROM_DSH — Task 6J Report: Structured Proposal Grounding Feasibility

_This file holds the Task 6J report. The Task 6I report is preserved in git history and in
`docs/task6i_visual_query_refinement.md`; Task 6H.1 in `docs/task6h1_bounded_point_counterfactual.md`;
Task 6H in `docs/task6h_counterfactual_pair_grounding.md`; Task 6G in `docs/task6g_dense_spatial_grounding.md`._

Full design notes: `docs/task6j_structured_proposal_grounding.md`, ADR-022.

## 1. Verdict

**`PROPOSAL_QUALITY_LIMIT`.**

The structured route decomposes cleanly: the relation executor is **exact** with oracle candidates
(J0: 120/120, paired 20/20, zero abstentions, bit-level agreement with the frozen generator), the
instruction → canonical-program parser is **perfect** (J2: 1.000 accuracy / 1.000 macro F1 on the
fixed 120, paired 20/20, and 1.000 on the full 3,884-record val split), and the predicted-program +
oracle-candidate route is therefore also exact (J3: 1.000, paired 20/20). The binding failure is
the **frozen YOLO proposal chain**: recall passes (0.919/0.869/0.594 at IoU 0.25/0.5/0.75) and
oracle-program mIoU passes (0.3712 ≥ 0.30), but paired mask selection is **5/20** (gate 12/20)
with 35/120 executor abstentions driven by proposal-geometry mismatch — J4 was therefore correctly
**not run** (section 15). No SAM2 training, no YOLO retraining, no `[REF]`/SRE/SCL, no 4B, no
dataset migration, no GUI.

## 2. Strategic Pivot

Tasks 6D–6I froze as evidence (6C `P_C` 0.10604 / 0-20; point oracle ≈0.488/18-20; box oracle
0.7506/20-20; 6I I0 inside 13/20, paired point 5/10, ranking 10/10, norm-err 0.0838). Task 6I's
attention mass DID rise above initialization — the claim is "insufficient", never "zero
information". Task 6J is a deliberate architecture-path pivot, not a deletion of prior work:
`instruction → relation program → building candidates → explicit geometry execution → mask`.

## 3. Program Vocabulary

`evaluation/task6j_program_spec.json`: **20 canonical programs, 1:1 with the actual frozen
v0.1.1 query types** (no invented semantics). Each program's ordered operations were verified to
be the unique stored `reasoning_steps` pattern of its query type across train+val+test
(25,229 records). Operations: `argmin/argmax_centroid_{x,y}`, `argmax/argmin_area`,
`filter_relation(above|below|left_of|right_of)` (frozen `relation(subject, object)` convention),
`argmin_boundary_distance` (frozen `nearest_within`: boundary distance, non-border anchor, frozen
margins). Symbolic reference role `@1` = the step-1 anchor; L1 = single arg; L2-A = arg+nearest;
L2-B = arg+filter (unique kept); L3 = arg+filter+nearest.

## 4. J0 Oracle Executor

`buildreasonseg_mvp/structured_grounding.py` executor consumes ONLY program + candidate geometry
(mask/bbox/centroid/area/border flag); it never reads target id, GT reasoning or target mask.
On the fixed 120 val records + 20 paired images: exact accuracy **1.000 (120/120)**, paired
**20/20**, zero abstentions, L1/L2/L3 all 1.000, and **120/120 agreement** with the frozen
`recompute_target_from_steps` (bit-level executor-fidelity proof). Gate (≥0.98 / ≥19-20): **PASS**.

## 5. YOLO Baseline Provenance

Frozen YOLOv8m-seg-WHU (`WHU_Building_Segment/runs/segment/logs/whu_building_v1/weights/best.pt`,
100 epochs, imgsz 640): SHA-256 `d9a6a65b7e0819ce4ecbbd9d44a5c8f9dcd2e60ea78203ba8fdf90ba6aaa1f91`,
re-hashed before inference and re-verified after (match: true). Invoked read-only through the
existing `yolo_sam_env` (Python 3.10.20, ultralytics 8.4.67, torch 2.13.0+cu132, cuda:0). Nothing
installed; no legacy file modified; proposal cache gitignored under
`artifacts/task6j_yolo_proposals/` (131 unique images).

## 6. Proposal Recall

Target recall @ IoU 0.25/0.50/0.75 = **0.919 / 0.869 / 0.594**; mean/median best IoU 0.701/0.786;
missing-target rate 13.1 %; ≈9.5 proposals/image; duplicate pairs (IoU>0.7) 21/7,862; all-component
recall@0.5 target 0.869 vs non-target 0.851; tiny-component recall 0.391; border-component recall
0.797. Recall gate (≥0.75): **PASS** — but tiny components and 13 % missing targets foreshadow the
J1 result.

## 7. J1 Oracle Program + YOLO

Canonical templates executed over predicted proposal geometry (no GT in execution): strict mIoU
**0.3712** (gate 0.30 ✓), Dice 0.4174, **35/120 abstentions** (nearest_relation_invalid 13,
no_eligible 9, filter_multi 9, extreme_invalid 4), paired mask selection **5/20** (gate 12/20 ✗),
mean own IoU 0.131 vs cross 0.160. **J1 viability gate FAILS** → J4 not run. Failure attribution
over 120 rows: 55 good selections, 35 abstentions, 18 proposal-geometry-changes-outcome, 8
target-absent-or-poor, 4 correct-selection-poor-mask-quality.

## 8. ProgramHead Architecture

Text-only Qwen3-VL-2B: instruction (chat format, NO image tokens) → text-only LoRA → last prompt
position hidden (assistant-prefix representation) → `LayerNorm → Linear(2048, 20)` → program id.
Trainable: text-only LoRA + ProgramHead; frozen: Qwen base, visual tower, everything else.
query_type appears only as the CE target. No free-form generation.

## 9. J2 Program Parsing

Query-type-stratified train-only subset (100 × 20 = 2000 records), 5 epochs, batch 16, cosine over
625 steps, strict determinism. Fixed-120 accuracy **1.0000**, macro F1 **1.0000**, paired program
correctness **20/20**, full-val (3,884) accuracy **1.0000**. Gate (≥0.90/≥0.85/≥18-20): **PASS**.
Checkpoint `artifacts/checkpoints/task6j/j2_best.pt` (manifest recorded).

## 10. J3 Predicted Program + Oracle Candidates

Instruction → trained ProgramHead → program → oracle candidates → executor: selected-target
accuracy **1.0000 (120/120)**, paired **20/20**, zero failures, program accuracy 1.0000. No GT
program fallback anywhere. Gate (≥0.85/≥17-20): **PASS**.

## 11. J4 Structured End-to-End

**Not run** — section 15 gates J4 on the J1 viability gate, which failed on paired selection
(5/20 < 12/20). `evaluation/task6j_j4_structured_end_to_end.json` records `ran: false` and the
blockers; the J4 script exists and refuses without the gates.

## 12. Optional SAM Refinement

Not exercised (J4 gated off). The J4 source contains no SAM2 calls at all; if added later it must
consume only predicted proposal geometry.

## 13. L1/L2/L3 + Query Breakdown

J0: L1/L2/L3 all 1.000 (oracle executor). J2: per-query-type accuracy 1.000 for all 20 programs.
J1 (proposal chain): mIoU by query type ranges from 0.000 (smallest_to_left_of) to 0.790
(rightmost); the L2/L3 multi-step programs suffer the abstentions and geometry shifts documented
in §7.

## 14. Failure Attribution

`evaluation/task6j_error_attribution.json`: binding failure = **proposal_chain_j1_paired_selection**.
J3 failures: none. J1 failures: 35 executor abstentions, 18 proposal-geometry-changes-relation-outcome,
8 target-absent-or-poor-in-proposal-set, 4 correct-selection-poor-mask-quality. The parser and the
executor are not the problem; the proposal backbone is.

## 15. Dataset / Proposal Adequacy

Tiny-component recall@0.5 is 0.391, border-component recall 0.797, 13.1 % of targets have no
proposal at IoU ≥ 0.5, and 21 images contain duplicate (IoU>0.7) proposal pairs. Merged/touching
buildings are the known upstream limitation of the binary component maps. Recommendation: a
proposal-backbone/dataset task next (instance segmentation aligned with component semantics, or
proposal-level post-processing) — no automatic data migration.

## 16. Reusable Core API

`buildreasonseg_mvp/structured_grounding.py`: `parse_program(instruction, parser, template_map)`,
`extract_building_candidates(image)` (CandidateSet), `execute_program(program, candidates)`,
`execute_program_by_id(program_id, candidates)`, `predict_structured_mask(image, instruction, ...)`,
plus `build_program_spec`, `canonical_program_template`, `CandidateSet.from_component_map` /
`from_proposals`. No GUI.

## 17. Runtime / VRAM

YOLO inference: 131 images ≈ minutes on cuda:0 (read-only env). J2: 625 optimizer steps ≈ 8 min
on the RTX 5080 Laptop (text-only bf16, strict determinism). J0/J1/J3 executor runs: pure CPU
geometry over cached proposals/component maps. No new packages, no new environments.

## 18. Tests

`python -m pytest tests/ -q` → **444 passed** (32 warnings), including 26 new Task 6J tests
(`tests/test_task6j_structured_grounding.py`): vocabulary-from-frozen-query-types, frozen relation
convention, executor-cannot-read-target-id, oracle-candidates-diagnostic-only, synthetic unit
reproduction of every operation, real-data equivalence with the frozen recompute, YOLO provenance
hash checks, legacy read-only hygiene, no package installation, proposal geometry without GT,
recall matching evaluation-only, ProgramHead text-only/no-image-tokens, query_type as CE target
only, split hygiene, J3 no-GT-program fallback, J4 predicted-only inputs, no old direct-pixel
head, no `[REF]`/SRE/SCL/4B, strict determinism, no GUI, auditable failure attribution, and the
J0/J1/J2/J3 artifact gates.

## 19. Git / Watt

Task commit `5a5daf0` (`feat: audit structured proposal grounding`) followed by the `docs:`
handoff commit; YOLO weights, prediction caches and checkpoints stay gitignored (only
hashes/manifests tracked); no legacy files, no dataset JSONL edits. Watt already running from
earlier tasks; transport-only for the push, ownership rules respected — no hosts/cert/TLS edits.

## 20. Recommended Next Architecture Decision

**Proposal-backbone / dataset step.** The parser and the executor are solved and reusable; the
measured blocker is that YOLO instances do not match the component semantics the frozen relation
engine was calibrated on (split/merged/border instances → 35/120 abstentions and 5/20 paired
selection). Evaluate an instance-segmentation backbone whose instances align with the component
maps (or deterministic proposal post-processing such as split/merge against the relation
eligibility flags) and re-run the J1/J4 gates. Keep the 20-program vocabulary, the executor and
the ProgramHead frozen. Per section 26 the task stops here; no learned SRE, no `[REF]`, no SCL,
no YOLO retraining, no 4B, no dataset migration, no full training, no GUI — waiting for
ChatGPT review.
