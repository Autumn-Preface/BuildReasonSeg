# Task 6S — Directional End-to-End Integration + Architecture Hardening Checkpoint

> Task: `handoff/TO_DSH.md` (Task 6S) · Base commit: `a3d59da` · Predecessor: Task 6R →
> `GRCL_NO_MEANINGFUL_RELATION_GAIN`
> **Verdict: `DIRECTIONAL_PARSER_HARDENING_REQUIRED`** · Dominant bottleneck: **`REFERENCE`**
> Entry point: `predict_buildreasonseg_directional.py` · Tests:
> `tests/test_task6s_directional_end_to_end.py` · Evidence: `evaluation/task6s_*.json`

The first natural-language directional end-to-end chain of the project, integrating **already frozen**
modules only. No module was retrained, no threshold was tuned, no algorithm was redesigned, and the test
split was never read.

## 1. The primary chain

```text
natural-language instruction
    -> frozen Qwen3-VL-2B ProgramHead          (text only, 20 canonical ids)
    -> canonical program id
    -> deterministic decomposition (8 supported programs)
    -> frozen Task 6Q YOLO26m-seg proposal reference resolver
    -> predicted reference mask
    -> GeometricRelationField v0.2  ->  P_rel
    -> frozen SAM2.1 Hiera Base+ visual feature + P_rel + relation embedding
    -> frozen Task 6O B3 target decoder
    -> target mask
```

**No GRCL v0.1 in this chain.** No oracle reference mask, no deterministic final-target proposal
selection, no `[REF]`, no nearest, no L3, no MLLM hidden-state fusion. The proposal system grounds the
*reference* only; the target is produced by the dense B3 visual decoder.

`buildreasonseg_mvp/task6s_directional_pipeline.py` implements the chain;
`predict_buildreasonseg_directional.py` is the CMD entry point (image + prompt + checkpoints only).

Exit codes: `0` answered · `3` reference abstention · `4` out-of-domain prompt (before ProgramHead and
before any downstream model) · `5` valid-but-unsupported canonical program (before proposal/SAM2/B3).

## 2. Why B3 and not R1/GRCL (recorded)

Task 6R measured: R0/B3 MiniVal mIoU `0.4299680351479113` vs R1/B3+GRCL `0.41185668634454997`; hard
relation accuracy `0.95` vs `0.9541666666666667`; PairedVal `14/20` vs `12/20`; proposal-reference
transfer frozen-6Q B3 chain `0.3045812554881724` vs R1 transfer `0.283782` with paired `10/20 → 2/20`.
GRCL v0.1 is therefore excluded from the primary chain and preserved only as a negative ablation.
Task 6R artifacts and its verdict were **not** modified.

**Task 6R gate erratum (recorded, not mutated):** the predeclared criterion
`R1 relation_accuracy >= R0 relation_accuracy + 0.08` was ill-posed, because the frozen baseline already
measured `R0 = 0.95`, leaving only 0.05 of headroom below the metric ceiling 1.0. This was a ChatGPT
experiment-design error, not a DSH implementation error; the independent measurements still support
excluding GRCL v0.1 (relation gain ≈ +0.0042, mIoU down, PairedVal down, transfer degraded).

## 3. Frozen-asset integrity audit (Part B)

`evaluation/task6s_frozen_asset_audit.json` = **`ASSETS_FROZEN`**.

| Asset | Verification |
|---|---|
| ProgramHead | SHA256 `eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3` **exact**; text only (no image input); frozen 20-program vocabulary; not retrained |
| Proposal resolver | SHA256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` **exact**; YOLO26m-seg, imgsz 640, conf 0.10, max_det 100, default NMS, no TTA, no tiling; eligibility/ranking untouched |
| B3 target decoder | SHA256 read from the frozen Task 6O artifact and matched exactly; 274,625 params; not retrained |
| Relation field | `geometric_relation_field_v02.py` unmodified; `alpha 1.2`, `tau 0.04`, `s_axis 0.02`, `s_margin 0.02` |

**Recorded path erratum (no improvisation):** section 5 names the default path
`artifacts/checkpoints/task6j/j2_best.pt`, which hashes to
`224f70c97f60d9eb21284cfefdd806eabf300b42375f10d45a25d19547097457` and therefore does **not** match the
section-5 expected hash. The checkpoint whose SHA256 equals the expected frozen hash is
`artifacts/checkpoints/task6m/program_parser_v02_best.pt`, which is exactly *"the existing text-only
Qwen3-VL-2B ProgramHead used by Task 6M structured CLI"* as section 5 describes. The expected **SHA256
was treated as authoritative**; both candidates and hashes are recorded in the audit. No checkpoint was
retrained, replaced or edited.

## 4. Parser audit on MiniVal240 (section 15)

`evaluation/task6s_parser_val.json` — frozen ProgramHead on every record's real natural-language query,
text only, in both languages (English is the primary gauge because it is the input language of the
parser's own frozen v0.2 validation evidence; Chinese is reported alongside).

| Language | exact program accuracy | reference-family accuracy | relation accuracy | unsupported classifications |
|---|---|---|---|---|
| English (primary) | **1.0000** (240/240) | 1.0000 | 1.0000 | **0** |
| Chinese | **1.0000** (240/240) | 1.0000 | 1.0000 | 0 |

Gate 1 (≥ 0.95) **PASS**. Every pair member also parsed correctly (40/40) and no pair was a parser-error
pair.

## 5. End-to-end MiniVal240 (section 16)

`evaluation/task6s_end_to_end_val.json` — 240 records, one frozen proposal run per tile (212 tiles,
YOLO executed exactly once per tile), inference path receiving only image + query + frozen checkpoints.

| Metric | Value |
|---|---|
| strict all-240 mIoU | **0.2969667241009681** (gate ≥ 0.28 ✓) |
| strict all-240 Dice | 0.3753212788292584 |
| strict Pr@0.5 | 0.2916666666666667 |
| answered-only mIoU (234 records) | **0.3045812554881724** (gate ≥ 0.2893521927137638 ✓) |
| answered-only Dice | 0.38494490136334186 |
| answered-only Pr@0.5 | 0.29914529914529914 |
| abstention rate | **0.025000000000000022** (6 records; all `reference_abstention`) |
| parser exact-program accuracy | 1.0 (0 unsupported) |
| per direction (left/right/above/below) mIoU | 0.335333 / 0.247051 / 0.329150 / 0.303450 |
| per reference family (largest/smallest) mIoU | 0.335005 / 0.273099 |
| border target (n=110) mIoU | 0.30496797891142324 (Pr@0.5 0.2909090909090909) |
| tiny target (n=4) mIoU | 7.42794373054799e-08 (≈ 0; Pr@0.5 0.0) |

**Comparison with the frozen Task 6Q chain** (which used oracle program ids): answered-only mIoU delta
**0.0** exactly, Dice 6Q `0.384945` vs 6S `0.38494490136334186`, abstentions 6 vs 6, PairedVal `10/20` vs
`10/20`, own-cross margin `0.27370032940000916` vs `0.27370032940000916`. Section 23's regression check
therefore reports `material_difference: false` — Task 6S measures **no incremental cost** of
natural-language parsing/integration on this pack, because the frozen parser is exact on these queries.

*Implementation difference found and fixed during the §23 investigation (recorded for transparency):* the
first Task 6S paired run reported 14/20 instead of 10/20. Cause: the Task 6S aggregate-IoU helper used the
project-wide `(intersection + 1e-6) / (union + 1e-6)` convention, which lets an **empty** predicted mask
score `1e-6/(union+1e-6)` and thereby win the own-vs-cross comparison whenever the counterpart target is
larger. The frozen Task 6Q/6R paired convention is `intersection / union` (empty → exactly 0.0). The
paired preference now uses a dedicated strict helper with that convention, and the Task 6S paired numbers
reproduce Task 6Q to the last digit. No model, threshold or algorithm was changed — only the metric
epsilon of a comparison, which is exactly the class of "implementation difference" section 23 asks to
investigate.

## 6. PairedVal20 (section 17)

`evaluation/task6s_end_to_end_paired_val.json` — each member uses its own natural-language query; for a
pair sharing image + reference source the same resolved proposal reference mask is reused and only the
relation differs.

| Metric | Task 6S | Frozen Task 6Q B3 chain |
|---|---|---|
| parser-correct members | **40/40** | 40/40 (oracle ids) |
| pass | **10/20** (gate ≥ 9 ✓) | 10/20 |
| mean own IoU | 0.2779456770946653 | 0.2779456770946653 |
| mean cross IoU | 0.00424534769465611 | 0.00424534769465611 |
| own − cross margin | **0.27370032940000916** (gate ≥ 0.22 ✓) | 0.27370032940000916 |
| reference-abstention pairs | 0 | 0 |
| parser-error pairs | 0 | — |

## 7. Fixed 24-prompt paraphrase/CLI audit (Part F)

`evaluation/task6s_cli_prompt_audit.json` — the **real CLI** executed as a subprocess, once per prompt,
on exactly the 24 fixed supported paraphrases (16 Chinese + 8 English from section 18) and the 8
unsupported controls (section 19).

| Item | Result |
|---|---|
| supported paraphrases parsed exactly | **21/24** (gate ≥ 22 **FAIL**) |
| failing paraphrases | #2 `找出最大建筑左边的建筑物。` → `smallest_to_left_of`; #5 `找出最大建筑右边的建筑物。` → `smallest_to_right_of`; #11 `找出最大建筑下面的建筑物。` → `smallest_to_below` |
| downstream branch reached correctly when parsed exactly | 21/21 |
| OOD controls (4) | **all exit 4** before parser/proposal/SAM2/B3 ✓ |
| out-of-scope controls (4) | #5 `分割面积最大的建筑物。` → `largest`, #6 `分割最左侧的建筑物。` → `leftmost` → both **exit 5** ✓; #7 `分割面积最大的建筑物右侧最近的建筑物。` and #8 `segment the building nearest to the right of the largest building` → parsed as **`largest_to_right_of`** → **exit 3** (reference abstention) instead of exit 5 ✗ |
| downstream never called for any control | ✓ |
| GT available to the CLI | **no** (`ground_truth_required: false` in every result JSON) |
| on-the-fly SAM2 | **proven**: tile `2_0` (not in MiniVal240, **no** feature-cache key) ran exit 0 with `sam2_feature = 0.206 s` versus 0.002 s from cache; four visual outputs written |

Two measured parser/scope facts are reported without repair (as required):

1. the frozen parser is exact on all 240 canonical dataset queries in both languages but confuses
   **largest vs smallest** on 3 of the 16 short Chinese paraphrases;
2. the frozen canonical vocabulary has **no "nearest" program**, so "nearest" prompts are silently mapped
   onto a *supported* directional program and exit the chain at the resolver (exit 3) rather than being
   rejected as out-of-scope (exit 5). The 6S scope check can only reject canonical programs outside the
   8; it cannot detect semantics the vocabulary cannot express.

## 8. CLI contract (Parts D)

`result.json` per successful prompt contains: status, prompt, parsed_program, reference_family, relation,
domain_gate result, parser/proposal/B3 checkpoint hashes, proposal count, eligible reference proposal
count, selected reference proposal index/confidence/area/bbox, explicit reference abstention
status/reason, relation-field min/max/mean, target positive-pixel count and the runtime breakdown
(parser / proposal-reference / SAM2 feature / field / target decoder / total), plus
`ground_truth_used: false`. Successful runs write `target_mask.png`, `overlay.png`,
`reference_mask.png`, `relation_field.png`.

## 9. Failure attribution and the dominant bottleneck (Part G)

`evaluation/task6s_failure_attribution.json` — one exclusive bucket per MiniVal240 record in the fixed
priority order (GT used only in this offline evaluator):

| Bucket | Count | Share |
|---|---|---|
| `PARSER_WRONG` | **0** | 0.0000 |
| `REFERENCE_NO_PROPOSALS` | 3 | 0.0125 |
| `REFERENCE_NO_ELIGIBLE` | 3 | 0.0125 |
| `REFERENCE_NOT_COVERED_IOU50` | **68** | 0.2833 |
| `REFERENCE_SELECTION_WRONG` | **42** | 0.1750 |
| `REFERENCE_GEOMETRY_POOR` | 1 | 0.0042 |
| `TARGET_FAIL_WITH_REFERENCE_OK` | 67 | 0.2792 |
| `TARGET_OK` | 56 | 0.2333 |

Aggregation: `parser_fail = 0` (0.0000 ≤ 0.05), `reference_fail = 117`, `target_fail = 67`
→ **dominant bottleneck `REFERENCE`** (reference_fail > target_fail). Breakdowns by
reference family, direction, border and expected program are in the artifact.

This label is for ChatGPT's next research decision only; DSH does not choose a repair and does not
prescribe an implementation.

## 10. Predeclared integration gates (section 22) and verdict (section 24)

| # | Gate | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | MiniVal240 parser exact-program accuracy | ≥ 0.95 | **1.0000** | ✓ |
| 2 | strict all-240 end-to-end mIoU | ≥ 0.28 | **0.2969667** | ✓ |
| 3 | answered-only end-to-end mIoU | ≥ 0.2893521927137638 | **0.3045813** | ✓ |
| 4 | PairedVal | ≥ 9/20 | **10/20** | ✓ |
| 5 | own-cross margin | ≥ 0.22 | **+0.2737003** | ✓ |
| 6 | supported 24-prompt paraphrase accuracy | ≥ 22/24 | **21/24** | **✗** |
| 7 | all OOD controls exit 4 | yes | yes | ✓ |
| 8 | all valid-but-out-of-scope controls exit 5 | yes | 2/4 | **✗** |
| 9 | no GT/annotation dependency in the CLI | yes | yes | ✓ |
| 10 | no test split | yes | yes | ✓ |

Section 24 priority order applied literally:

1. `INVALID_EXPERIMENT` — no (frozen artifacts unchanged, no test access, no GT dependency, no GRCL in the
   primary chain, no oracle reference in inference).
2. `PARSER_CHECKPOINT_UNAVAILABLE` — no (exact hash match).
3. `END_TO_END_INTEGRATION_REGRESSION` — no (answered-only mIoU delta exactly 0.0, identical answered set).
4. **`DIRECTIONAL_PARSER_HARDENING_REQUIRED`** — canonical parser accuracy 1.0000 ≥ 0.95 but the
   paraphrase pack is **21/24 < 22**. ← **verdict**
5. `DIRECTIONAL_CHAIN_BELOW_GATE` — not reached.
6. `DIRECTIONAL_END_TO_END_CHAIN_READY_FOR_HARDENING` — no.

Gate 8 also fails, but the fixed priority order evaluates the parser gate first.

## 11. Architecture-freeze checkpoint (section 25)

`evaluation/task6s_hardening_checkpoint.json` records the primary chain (parser: frozen Qwen3-VL-2B
ProgramHead · reference: frozen Task6Q proposal resolver · relation field: v0.2 · target decoder: frozen
Task6O B3 · `grcl_primary: false`), the proven positive modules (GeometricRelationField v0.2 + dense
visual target decoder), the negative/non-primary evidence (Task6P dense ReferenceMaskHead, Task6R GRCL
v0.1), the known technical debt (reference proposal coverage/extreme ranking especially smallest, tiny
buildings, directional-only scope, no nearest, no L3 multi-hop, no cross-dataset generalization), the
dominant bottleneck (`REFERENCE`) and `next_research_decision: WAIT_FOR_CHATGPT`.

## 12. Interpretation boundary

DSH reports measurements only. `READY_FOR_HARDENING`-style outcomes would not mean the final model or
paper is ready; this run did not reach that outcome. DSH does not repair the dominant bottleneck, does
not retrain the parser or reference, does not add GRCL/nearest/L3, does not start full-dataset training,
does not access the test split and does not build a GUI.

## 13. Reproduce

```text
python scripts/task6s_evaluate.py --stage assets       # Part B frozen-asset audit
python scripts/task6s_evaluate.py --stage parser       # section 15 parser audit
python scripts/task6s_evaluate.py --stage end-to-end   # sections 16-17 full chain + PairedVal20
python scripts/task6s_cli_audit.py --stage cli         # sections 18-19 + section 13 on-the-fly SAM2
python scripts/task6s_failure_attribution.py --stage attribution   # sections 20-21
python scripts/task6s_report.py                        # sections 24-25 verdict + checkpoint
```

Run in `.conda/buildreasonseg-proposal` (frozen SAM2 + torch stack) with `HF_HUB_OFFLINE=1` so the frozen
parser loads from the local cache and nothing is downloaded. Generated masks/overlays and the CLI working
directory live under the gitignored `artifacts/task6s/`.
