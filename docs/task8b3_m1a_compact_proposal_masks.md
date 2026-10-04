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


---

## Task 8B.3-M1A.1 — implementation attempt (budget-limited, no functional change)

| item | value |
|---|---|
| branch | `fix/task8b3-mem01-compact-proposals` @ `8fc71ca49797112cd0a551faadf27ad7e1de29f1` |
| status | **PARTIAL / STOP** |
| functional canonical files changed | **none** |
| `source_manifest.json` | not updated (no canonical content change) |
| dedicated `tests/test_task8b_runtime.py` gate | NOT RUN |
| full canonical suite | NOT RUN (excluded by this task book) |
| external delivery | UNCHANGED |
| real inference / six-image Demo | NOT RUN |

**Why no code was changed:** the M1A.1 implementation requires coordinated edits to four canonical files
(`detector.py` representation + `detect_global` compaction + `proposal_iou` + merge loop; `core.py`
`reference_mask_from_proposal`; `outputs.py` `proposals_preview_image`; `tests/test_task8b_runtime.py` helper plus
the seven required test groups). Writing those edits safely needs an exact read of the current function bodies first,
and this execution turn ran out of budget before that read could be completed. Rather than risk a partially applied
refactor that would leave the canonical tree broken and the single dedicated-test gate failing for implementation
reasons, **no file was modified**; the branch remains exactly at its starting HEAD with only documentation/handoff
changes.

**Dependency inventory already established (M1A §4, unchanged):** all 13 executable `global_mask` usages live in
`buildreasonseg/runtime/detector.py` (field, `np.zeros((H,W))`, paste, retain, construction, `iou_of` call,
eligibility `.any()`), `buildreasonseg/runtime/core.py` (shape lookup + slice), `buildreasonseg/runtime/outputs.py`
(preview outline) and `tests/test_task8b_runtime.py` (test helper + duplicate-winner assertions). This site list is
the complete work list for the next turn.

**Suggested split for the next task book (factual, not a decision):** (1) representation + `detect_global`
compaction; (2) `proposal_iou` + merge loop; (3) `core.py` / `outputs.py` adaptation; (4) the seven test groups +
the 5000×5000 synthetic guard. Each is independently verifiable and keeps the dedicated gate meaningful.

**Unchanged defects:** `RC1-DEMO-MEM-01` still pending implementation; `RC1-DEMO-PROP-01`, `RC1-DEMO-REF-01`,
`RC1-DEMO-MASK-01` untouched.

**Scope compliance:** no detector parameter/tiling/IoU-threshold/winner/stable-ID/Reference/context/language/SAM2/
D-B1/validity change; no PROP-01/REF-01/MASK-01 fix; no manifest update; no canonical suite; no external delivery
sync or edit; no real predict or six-image Demo; no package change.

---

## Task 8B.3-M1A.2A — compact proposal representation implemented (canonical runtime only)

| item | value |
|---|---|
| branch | fix/task8b3-mem01-compact-proposals |
| starting HEAD | 2c51c48252fcc705da5d9ee914299775d985a725 |
| files changed | buildreasonseg/runtime/detector.py, buildreasonseg/runtime/core.py, buildreasonseg/runtime/outputs.py |
| tests / source_manifest / external delivery | NOT modified |
| py_compile gate | PASS (detector.py, core.py, outputs.py) |
| synthetic smoke gate | PASS — no model loading, SMOKE_FAILURES empty |
| real predict / six-image Demo | NOT RUN |

### What changed

1. GlobalProposal now carries `mask_crop` (tight bbox-local bool crop) + `global_bbox` (inclusive) instead of a
   full-frame `global_mask`; `image_size` was added as an optional field (default None) so existing constructor
   calls without it still work. `to_dict()` is unchanged and never serialises `mask_crop`.
2. `_compact_mask(mask, top, left)` computes the tight bbox via the existing `bbox_of`, slices the crop, and returns
   `(mask_crop, global_bbox)` or None for an empty mask.
3. `detect_global()` no longer allocates a full-frame bool array: each detection is compacted in source-pixel space
   and the accumulated entry stores `mask_crop`, `global_bbox`, `image_size`, tile/confidence/raw-index fields.
4. `proposal_iou(first, second)` computes duplicate IoU **only inside the two inclusive bboxes' intersection**:
   0.0 when they do not overlap, otherwise AND/OR counts over the intersection slices with
   `union = area1 + area2 - intersection`. `DUPLICATE_IOU = 0.50` is unchanged, and the legacy full-frame
   `iou_of()` is retained untouched for reference/comparison.
5. `merge_proposals()` builds proposals from `mask_crop`/`global_bbox`; `mask_area`, centroid, image-border touch
   and border clearance are computed in crop coordinates mapped to global pixels through the bbox and `image_size`
   (semantics identical to the previous full-frame computation). Winner priority, stable-ID ordering and the
   no-union rule are unchanged. Duplicate grouping now calls `proposal_iou`.
6. `eligible()` uses `proposal.mask_crop.any()`; `MERGE_BBOX_EXTENT_RATIO_MAX`, the eligibility rule and
   `select_reference()` are unchanged.
7. `core.reference_mask_from_proposal()` still allocates only the 512x512 context mask and pastes the crop slice
   corresponding to bbox intersect context; it reads `image_size` instead of the removed `global_mask.shape`.
8. `outputs.proposals_preview_image()` computes the magenta outline with `_erode()` in crop coordinates and writes
   only the affected global pixel indices; bbox/centroid drawing is unchanged.

### Evidence from the synthetic smoke

* compactness: 40x40 mask with a 10x10 block at (10,5) produced bbox (110, 205, 119, 214) and a 10x10 all-True crop;
* IoU equivalence: `proposal_iou` equals the legacy full-frame `iou_of` to within 1e-12 on overlapping masks;
* merge equivalence: grouping, retained count, mask areas, stable IDs and centroid ordering matched expectations;
* large-image guard: with `np.zeros` monkeypatched to raise on any bool array of shape (5000, 5000), the compact
  path (compaction + merge + core context crop + preview) completed with no such allocation and correct mask area;
* no model, checkpoint, dataset or delivery file was touched.

### Scope compliance

Detector parameters (TILE_SIZE/overlap/stride/IMGSZ/CONF/MAX_DET), `DUPLICATE_IOU`, winner priority, stable-ID
ordering, Reference eligibility/`MERGE_BBOX_EXTENT_RATIO_MAX`/selection, reasoning context, ProgramHead, SAM2, D-B1
and post-inference SUCCESS validity are unchanged; PROP-01, REF-01 and MASK-01 are untouched. No test file,
`source_manifest.json`, external delivery, pytest run, predict run or six-image Demo was involved.


---

## Task 8B.3-M1A.2A-R1 — core reference crop three-way intersection fix

| item | value |
|---|---|
| branch | `fix/task8b3-mem01-compact-proposals` @ `10d36117993f128c9a3d6bd2bea114dc63120121` |
| files changed | `buildreasonseg/runtime/core.py` only (+ report/handoff) |
| valid region | **original image ∩ 512 reasoning context ∩ `proposal.global_bbox`** (inclusive bbox) |
| py_compile gate | PASS (`core.py`) |
| legacy-equivalence smokes | 3 synthetic, model-free (context at proposal top-left; proposal clipped top-left; proposal clipped bottom-right) |
| `detector.py` / `outputs.py` / tests / `source_manifest.json` / external delivery | NOT modified |
| real predict / six-image Demo | NOT RUN |

Fix: `reference_mask_from_proposal()` previously intersected only the image and the context, so a context window
that extended past the proposal bbox produced negative crop offsets (and slices beyond `mask_crop`). The valid
window is now the three-way intersection, `valid_top = max(0, context.top, top)`,
`valid_left = max(0, context.left, left)`, `valid_bottom = min(image_height, context.top + CONTEXT_SIZE, bottom + 1)`,
`valid_right = min(image_width, context.left + CONTEXT_SIZE, right + 1)`, with the crop sliced by
`valid_* - bbox` offsets and pasted at `valid_* - context` offsets. No other semantic changed.

Legacy equivalence: for all three geometries the new function returns arrays identical to the pre-refactor
full-frame computation (`np.array_equal` true), which is the expected behaviour wherever the legacy result was
already inside the bbox; the fix additionally removes the out-of-bbox offsets that the legacy code only avoided
incidentally.

---

## Task 8B.3-M1A.2B — dedicated runtime-test adaptation (single run)

| item | value |
|---|---|
| branch | `fix/task8b3-mem01-compact-proposals` @ `39b36031294f1ec1ddab3f5e9681170af687af5a` |
| files changed | `tests/test_task8b_runtime.py` only |
| runtime files | NOT modified (`detector.py`, `core.py`, `outputs.py` untouched) |
| dedicated run | exactly once: `pytest tests/test_task8b_runtime.py -q` → **FAIL (STOP)** |
| tail | FAILED tests/test_task8b_runtime.py::test_select_reference_tie_break - KeyErr... | FAILED tests/test_task8b_runtime.py::test_merge_equivalence_with_compact_masks | 5 failed, 27 passed in 0.44s |
| manifest | UNCHANGED (dedicated gate failed) |
| external delivery / real inference | NOT touched / NOT RUN |

Adaptation: the test-local `_proposal()` helper now builds the compact representation
(`mask_crop` = tight crop of the supplied full-frame mask over its own bbox, `global_bbox`, `image_size`), a
`_to_full()` test-only reconstruction helper supports the legacy full-frame comparisons, and the two duplicate-winner
assertions use the reconstructed masks. The legacy full-frame `detector.iou_of()` is deliberately kept as the
comparison oracle for the new compact `detector.proposal_iou()`.

Added coverage (six groups): compact geometry tightness/global offsets; `proposal_iou` equivalence with the legacy
full-frame IoU plus the disjoint and identical cases; merge equivalence (grouping, count, areas, winner identity,
stable ids, no union); Reference-context equivalence against a locally re-implemented legacy full-frame computation
over four context placements; preview built from the crop (bbox window touched, outside untouched); serialization
contract (`to_dict()` carries no mask payload); and a 5000×5000 guard that monkeypatches `np.zeros` to raise on any
full-frame bool allocation while exercising compaction, merge, the reference crop and the preview.

---

## Task 8B.3-M1A.2B-R1 — complete compact-contract test adaptation (single run)

| item | value |
|---|---|
| branch | `fix/task8b3-mem01-compact-proposals` @ `e5229d60a869c1fc02536d41a039229f2936090b` |
| files changed | `tests/test_task8b_runtime.py` only |
| runtime files | NOT modified |
| legacy entry literals converted to `mask_crop`/`global_bbox`/`image_size` | 10 |
| equal-area duplicate construction | corrected (two equal 40×40 blocks, IoU ≈ 0.78) |
| 5000×5000 guard | real `DetectorRuntime.detect_global()` on a fake `(5000, 5000, 3)` shape with monkeypatched `plan_tiles` / `extract_tile` / `detect_tile` and a `np.zeros` guard; no real 5000×5000 RGB or bool oracle allocated |
| dedicated run | exactly once → **FAIL (STOP)** |
| tail | =========================== short test summary info =========================== | FAILED tests/test_task8b_runtime.py::test_merge_equivalence_with_compact_masks | FAILED tests/test_task8b_runtime.py::test_large_image_path_never_allocates_full_frame_bool | 2 failed, 30 passed in 0.40s |
| manifest | UNCHANGED (dedicated gate failed) |
| external delivery / real inference / full suite | NOT touched / NOT RUN / NOT RUN |

Per the task book, a failing single run is recorded here and the task stops without any runtime change, without a
rerun and without a manifest update.
