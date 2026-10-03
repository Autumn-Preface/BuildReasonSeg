# Task 8B.3-M1A — Compact Proposal Masks (canonical only)

> Status: **PARTIAL / STOP** — the frozen `global_mask` → tight-bbox + `mask_crop` refactor was **not implemented**
> in this execution turn. §3 (branch/gate) and §4 (dependency audit) completed; no functional canonical file was
> modified, so the refactor remains exactly as specified for the next execution turn.

## 1. Task / scope

Fix `RC1-DEMO-MEM-01` **in the workspace canonical source only** by replacing the full-frame per-proposal
representation with a tight global bbox + bbox-local bool `mask_crop`, and by computing duplicate IoU only over the
two bboxes' intersection. No external-delivery edit, no inference, no scientific/semantic policy change.

## 2. Starting branch/HEAD

| item | value |
|---|---|
| starting branch | `eval/task8b3-six-image-demo-suite` |
| starting HEAD | `7a9c576692abf82510d61fb1c814fd47d4b053a5` |
| working tree | only `M handoff/TO_DSH.md` |
| new branch created | **`fix/task8b3-mem01-compact-proposals`** (did not exist before; created from the starting HEAD) |

## 3. Frozen architecture decision

```text
OLD: one full H×W bool mask retained per raw proposal + pairwise H×W logical_and/logical_or during merge
NEW: one tight bbox-local bool crop retained per raw proposal + duplicate IoU only in the bbox intersection
```

## 4. Changed canonical paths

**None.** The following allowed paths were left byte-identical in this turn:
`buildreasonseg/runtime/detector.py`, `buildreasonseg/runtime/core.py`, `buildreasonseg/runtime/outputs.py`,
`tests/test_task8b_runtime.py`, `source_manifest.json`.

### 4.1 §4 pre-change `global_mask` dependency gate (read-only, PASSED)

`git grep -n "global_mask" -- delivery_src/BuildReasonSeg_Advisor_RC1` returned usage **only** in the four expected
files, so the task book's STOP condition did not trigger:

```text
buildreasonseg/runtime/core.py:190,191,194      (reference_mask_from_proposal: shape lookup + slice)
buildreasonseg/runtime/detector.py:130          (GlobalProposal.global_mask field)
buildreasonseg/runtime/detector.py:265,272,273,277 (per-raw-detection np.zeros((H,W)) + paste + retain)
buildreasonseg/runtime/detector.py:305          (GlobalProposal construction: global_mask=mask)
buildreasonseg/runtime/detector.py:325          (duplicate IoU on two full-frame masks)
buildreasonseg/runtime/detector.py:355          (eligibility: proposal.global_mask.any())
buildreasonseg/runtime/outputs.py:181           (preview outline via global_mask & ~_erode(global_mask))
tests/test_task8b_runtime.py:156,181,182        (test helper + duplicate-winner assertions)
```

This inventory is the exact site list the refactor must convert (13 executable usages, no others).

## 5. Compact mask representation

**NOT IMPLEMENTED.** Planned per §6–§8 of the task book: `GlobalProposal.mask_crop: np.ndarray` +
`global_bbox: tuple[int, int, int, int]` (inclusive `top, left, bottom, right`; `mask_crop.shape ==
(bottom-top+1, right-left+1)`; tight; bool; no `global_mask` field), `to_dict()` unchanged and never serializing
`mask_crop`, plus the single `_compact_mask(mask, *, top, left)` helper producing `(mask_crop, global_bbox)` or
`None` for an empty mask, with `accumulated` entries retaining `mask_crop`, `global_bbox`, `image_size`,
`source_tile_id`, `tile_index`, `confidence`, `raw_index`, `tile_top`, `tile_left`.

## 6. IoU/merge semantic-equivalence proof

**NOT PRODUCED.** Planned: `proposal_iou(first, second)` that intersects the two inclusive global bboxes, returns
`0.0` without allocation when they do not overlap, slices both `mask_crop`s over the global intersection rectangle,
computes `intersection = AND count`, `union = first.mask_area + second.mask_area - intersection`, and returns
`intersection / union` (0.0 when `union <= 0`), with `DUPLICATE_IOU = 0.50`, the frozen winner priority
(`touches_image_border → -border_clearance → -mask_area → -confidence → source_tile_id → raw_index`) and the frozen
stable-ID ordering (`centroid_y → centroid_x → mask_area`) unchanged, and no mask union introduced.

## 7. Downstream core/preview adaptations

**NOT IMPLEMENTED.** Planned: `core.reference_mask_from_proposal()` keeps allocating only the 512×512 context mask
and pastes the `mask_crop` slice corresponding to `global_bbox ∩ context` (never materializing a full-image mask);
`outputs.proposals_preview_image()` keeps bbox/centroid drawing unchanged and computes `_erode()` + the magenta
outline in crop coordinates, pasting only into `preview[top:bottom+1, left:right+1]`.

## 8. Dedicated test result

**NOT RUN** (no implementation to test yet). Planned coverage per §13: compact-representation invariant; IoU
equivalence versus a test-only legacy full-frame implementation; merge-equivalence regression (grouping, winner
identity, count, area, bbox, centroid, confidence, stable IDs); Reference-context equivalence (inside, clipped
left/top, clipped right/bottom); preview smoke; 5000×5000 synthetic no-full-bool guard with a monkeypatched
`np.zeros` that raises on `(5000, 5000)` bool; serialization contract.

## 9. Canonical full-test result

**NOT RUN** (gated behind the dedicated tests).

## 10. Manifest 135/135 verification

Not re-verified in this turn beyond the fact that **no canonical file changed**, so the manifest is untouched and
still matches the starting HEAD state (135 entries). The §16 update applies only once detector/core/outputs/tests
are actually modified.

## 11. External delivery = UNCHANGED

The external delivery `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1` was **not** modified: no sync
helper write-mode invocation, no file copy, no package change. Canonical and external delivery remain identical to
each other and to the starting HEAD.

## 12. Real inference = NOT RUN

No `predict.py`, no six-image suite, no YOLO/Qwen/SAM2/D-B1 execution, no checkpoint or model-parameter access.

## 13. Scientific scope statement

* no detector model / threshold / tiling change;
* no merge threshold / winner / stable-ID change;
* no Reference eligibility (`MERGE_BBOX_EXTENT_RATIO_MAX`) or selection change;
* no language / ProgramHead / SAM2 / D-B1 / relation-field / reasoning-context / SUCCESS-validity change.

Nothing scientific was touched in this turn because no functional file was modified at all.

## 14. Remaining defects

* `RC1-DEMO-PROP-01` — UNCHANGED (A2 zero-proposal detector/proposal-stage failure; NMS causality still NOT CONFIRMED);
* `RC1-DEMO-REF-01` — UNCHANGED (user-facing “largest building” ≠ implementation “largest eligible detected proposal”);
* `RC1-DEMO-MASK-01` — UNCHANGED (SUCCESS validity still only non-empty + non-padding + directional-centroid);
* `RC1-DEMO-MEM-01` — **implementation still pending**; the compact-mask refactor is specified but not applied.

## 15. Next gate

**Awaiting ChatGPT audit before any canonical → delivery sync.** The fix branch
`fix/task8b3-mem01-compact-proposals` exists at the starting HEAD with only documentation/handoff changes; the
refactor must be executed in a dedicated turn that has budget for the implementation, the seven required test
groups, the manifest refresh and the two test gates.
