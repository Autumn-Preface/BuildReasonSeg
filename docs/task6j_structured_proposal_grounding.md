# Task 6J — Structured Proposal Grounding Feasibility v0.1

> Task: `handoff/TO_DSH.md` · Verdict: **`PROPOSAL_QUALITY_LIMIT`**
> Evidence: `evaluation/task6j_*.json` · ADR-022 · Tests: `tests/test_task6j_structured_grounding.py`

## 1. What was tested

Task 6J pivots from direct pixel grounding (Tasks 6D–6I) to the project's original structured
route and audits it stage by stage:

```
instruction -> canonical relation program -> building candidates
            -> explicit geometry execution (frozen Task 3B relation engine) -> selected building mask
```

Four independent questions, four gates:

| Stage | Question | Gate | Result |
|---|---|---|---|
| J0 | Can the frozen relation semantics select the target from **oracle** candidates? | acc ≥ 0.98, paired ≥ 19/20 | **PASS** 1.000 / 20-20 |
| J1 | Can the frozen YOLO baseline provide a sufficient inference-time candidate set? | recall@0.5 ≥ 0.75, mIoU ≥ 0.30, paired ≥ 12/20 | **FAIL on paired** (0.869 / 0.371 / 5-20) |
| J2 | Can Qwen parse the instruction into the canonical program? | acc ≥ 0.90, macro F1 ≥ 0.85, paired ≥ 18/20 | **PASS** 1.000 / 1.000 / 20-20 |
| J3 | Predicted program + oracle candidates | acc ≥ 0.85, paired ≥ 17/20 | **PASS** 1.000 / 20-20 |
| J4 | Predicted program + predicted YOLO candidates | mIoU ≥ 0.20, paired ≥ 12/20 | **NOT RUN** (gated on J1 viability) |

## 2. Program vocabulary (section 2)

`evaluation/task6j_program_spec.json` derives **20 canonical programs 1:1 from the actual frozen
BuildSpatialReason v0.1.1 query types** — no semantics are invented. Each program's ordered
operations were verified to be the unique stored `reasoning_steps` pattern of its query type
across train + val + test (25,229 records). Operations: `argmin/argmax_centroid_{x,y}`,
`argmax/argmin_area`, `filter_relation(above|below|left_of|right_of)`, `argmin_boundary_distance`.
References are symbolic roles (`@1` = the step-1 anchor); L2/L3 templates:

- L1: `[arg_*]`
- L2-A (`X_to_nearest`): `[arg_*, nearest(ref=@1, candidates=all)]`
- L2-B (`X_to_{dir}`): `[arg_*, filter(dir, ref=@1, subjects=all)]` — answer = the unique kept candidate
- L3 (`largest_to_{dir}_to_nearest`): `[argmax_area, filter(dir, ref=@1), nearest(ref=@1, candidates=previous)]`

## 3. Independent executor (sections 3-4)

`buildreasonseg_mvp/structured_grounding.py` implements a compact executor that consumes ONLY a
program and candidate geometry (mask / bbox / centroid / area / border flag) and returns one
candidate id, reusing the frozen relation engine (`spatial_reasoning/relations.py`,
`thresholds.py`, `component_quality.py`) with the frozen v1 thresholds. It never reads the target
component id, GT reasoning text or the target mask. **J0 proof**: on the fixed 120 val records the
executor reproduces the frozen generator's own `recompute_target_from_steps` on **120/120**
samples (bit-level agreement), exact accuracy **1.000**, paired **20/20**, zero abstentions —
J0 gate passes and pins the executor semantics to the frozen Task 3B convention.

## 4. YOLO baseline (sections 6-7)

Frozen YOLOv8m-seg-WHU (`WHU_Building_Segment/runs/segment/logs/whu_building_v1/weights/best.pt`,
trained 100 epochs, imgsz 640) invoked **read-only** through the existing `yolo_sam_env`
(Python 3.10.20, ultralytics 8.4.67, torch 2.13.0+cu132, device cuda:0). Provenance is hashed
before inference (`d9a6a65b…`), re-verified after (match: true), and no package was installed,
no legacy file modified. 131 unique images (fixed 120 val + 20 paired) were segmented at
conf 0.25 / IoU 0.7 / max_det 300; proposals (mask/bbox/centroid/area/confidence) are cached under
gitignored `artifacts/task6j_yolo_proposals/`.

## 5. Proposal recall (section 8)

| Metric | Value |
|---|---|
| target recall @ IoU 0.25 / 0.50 / 0.75 | **0.919 / 0.869 / 0.594** |
| mean / median best proposal IoU | 0.701 / 0.786 |
| missing-target rate (best IoU < 0.5) | 0.131 |
| proposals per image (mean, 120-val sample rows) | 9.5 |
| duplicate proposal pairs (IoU > 0.7) | 21 of 7,862 pairs, 21 images |
| all-component recall@0.5 (1,257 components, 131 images) | target 0.869 vs non-target 0.851 |
| tiny-component recall@0.5 | 0.391 |
| border-component recall@0.5 | 0.797 |

Recall meets the J1 gate (0.869 ≥ 0.75). The weak spots are tiny components (0.391) and the
13.1 % missing-target rate.

## 6. J1 — oracle program + predicted proposals (section 9)

The canonical program template runs over the predicted proposal geometry with no GT id/mask in
execution. Result: strict mIoU **0.3712** (gate 0.30 ✓), Dice 0.4174, **35/120 abstentions**, and
paired mask selection **5/20** (gate 12/20 ✗) with mean own IoU 0.131 vs cross 0.160. Abstention
reasons: `nearest_relation_invalid` 13 (predicted anchors/candidates fail the frozen non-border
nearest eligibility), `no_eligible_candidates` 9, `filter_multi_candidate` 9,
`extreme_relation_invalid` 4. Failure attribution over the 120 rows: 55 successful selections,
35 executor abstentions, 18 proposal-geometry-changes-relation-outcome, 8 target-absent-or-poor,
4 correct-selection-poor-mask-quality. **The J1 viability gate fails on the paired selection, so
J4 is not run.**

The mechanism: the frozen semantics were calibrated on connected components; YOLO proposals
differ (split/merged instances, border-clipped anchors, different centroids/areas), so the same
program changes its outcome or abstains. Proposal recall is not the binding constraint — proposal
**geometry/instance semantics** is.

## 7. J2 — ProgramHead (sections 10-13)

Text-only branch: instruction (chat formatting, **no image tokens**) → Qwen3-VL-2B text-only
LoRA → last-prompt-position hidden → `LayerNorm → Linear(2048, 20)` → program id. Trainable:
text-only LoRA + ProgramHead; everything else frozen; query_type appears only as the CE target.
Training: query-type-stratified train-only subset, 100 records × 20 programs = 2000 records,
5 epochs, batch 16, cosine over 625 steps, strict determinism.

| Metric | Value |
|---|---|
| fixed-120 exact accuracy | **1.0000** |
| macro F1 | **1.0000** |
| paired program correctness | **20/20** |
| full val (3,884 records) exact accuracy | **1.0000** |

J2 gate passes with margin. (The instructions are template-generated and semantically complete,
so a perfect parser is expected — and it confirms program parsing is **not** the bottleneck.)

## 8. J3 — predicted program + oracle candidates (section 14)

Selected-target accuracy **1.0000** (120/120), paired **20/20**, zero failures — the parser +
executor + oracle candidate chain is exact. J3 gate passes.

## 9. Verdict and interpretation (sections 17-19)

**`PROPOSAL_QUALITY_LIMIT`** — J0, J2 and J3 are healthy; the frozen YOLO proposal chain is the
binding failure (J1 paired 5/20, gate 12/20; recall and single-mask mIoU pass). J4 was correctly
not run (section 15). Per the interpretation discipline: the result freezes nothing more than
"explicit program parsing + proposal-level geometry execution is a viable functional
target-selection architecture, conditional on a better proposal backbone", and does **not** claim
deterministic relation execution as final innovation, YOLO as final backbone, or open-vocabulary
reasoning.

The recommended next task is a **proposal-backbone/dataset step** (e.g. a frozen instance
segmentation model whose instances align with the component semantics — or proposal-level
post-processing — before touching the parser or the executor). No dataset migration, no YOLO
retraining, no `[REF]`/SRE/SCL, no 4B, no GUI.

## 10. Reusable API (section 20)

`buildreasonseg_mvp/structured_grounding.py`:

```python
parse_program(instruction, parser, template_map)          # ProgramHead -> program id
extract_building_candidates(image)                        # CandidateSet (proposal loader hook)
execute_program(program, candidates)                      # frozen-semantics executor
execute_program_by_id(program_id, candidates)
predict_structured_mask(image, instruction, ...)          # J4 pipeline
```

plus `build_program_spec` / `canonical_program_template` / `CandidateSet.from_component_map` /
`from_proposals`. No GUI.

## 11. Runtime / provenance notes

* YOLO inference: 131 images, ~9.5 proposals/image, GPU (cuda:0), read-only legacy env.
* J2 training: 625 optimizer steps ≈ 8 min on the RTX 5080 Laptop (text-only, bf16).
* J0/J1/J3 executor runs are pure CPU geometry over cached proposals/maps.
* `python -m pytest tests/ -q` → 444 passed (26 new Task 6J tests).

## 12. File map

| Piece | Path |
|---|---|
| Executor + program vocabulary + reusable API | `buildreasonseg_mvp/structured_grounding.py` |
| ProgramHead runtime + checkpointing | `buildreasonseg_mvp/program_parser.py` |
| Config | `configs/mvp/task6j_program_parser.yaml` |
| Stage scripts | `scripts/task6j_*.py` (spec, j0, j1, j2, j3, j4, attribution, verdict, yolo infer/manifest) |
| Evidence | `evaluation/task6j_*.json` |
| YOLO proposal cache | `artifacts/task6j_yolo_proposals/` (gitignored) |
| Tests | `tests/test_task6j_structured_grounding.py` |
