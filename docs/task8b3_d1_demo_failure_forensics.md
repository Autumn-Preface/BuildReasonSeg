# Task 8B.3-D1 — Demo Failure Forensics (read-only)

## 1. Task and scope

Read-only forensics of the already completed Task 8B.3-R4B six-image Automatic Demo baseline, plus static audit of
the canonical runtime source. **No inference ran; no delivery artifact, product/canonical source, harness or test
was modified; no threshold/config/checkpoint was touched; no Assisted Mode was used.** Only this report,
`handoff/FROM_DSH.md` and the task book changed in Git.

## 2. Frozen R4B baseline

```text
A1 language DIRECT_CORRECT    runtime SUCCESS
A2 language FALLBACK_CORRECT  runtime FAILED E401
A3 language DIRECT_CORRECT    runtime SUCCESS
A4 language FALLBACK_CORRECT  runtime SUCCESS
B1 language FALLBACK_CORRECT  runtime FAILED E502
B2 language DIRECT_CORRECT    runtime FAILED E502
language resolution 6/6 · runtime success 3/6 · manual end-to-end semantic success 0/6
```

The three R4B successes are `AUTOMATIC_RUNTIME_SUCCESS_PENDING_VISUAL_REVIEW`. The frozen ChatGPT visual audit
(A1/A3/A4 semantic FAIL) is reproduced as given and is not reinterpreted here.

## 3. A1/A3/A4 Reference-selection table

Facts read from each sample's `proposals.json` and `result.json`; eligibility applied exactly as frozen in
`detector.py` (non-empty ∧ not touching the original image border ∧ bbox extent ratio ≤ 0.20).

| sample | merged | selected_ref_id | ref_area | ref_bbox | ref_extent | ref_border | ref_conf | largest_id_by_area | largest_area | largest_eligible_under_rule | largest_eligible_id | largest_eligible_area | selected_is_largest_eligible | eligible_count | final mask area |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| A1 | 52 | 48 | 4853 | [811, 180, 865, 276] | 0.1895 | false | 0.1292 | 6 | 6470 | **false** | 48 | 4853 | **true** | 45 | 101 |
| A3 | 6 | 1 | 680 | [384, 867, 415, 895] | 0.0625 | false | 0.1897 | 4 | 2023 | **false** | 1 | 680 | **true** | 4 | 1439 |
| A4 | 77 | 30 | 3622 | [658, 41, 707, 119] | 0.1543 | false | 0.0573 | 29 | 10404 | **false** | 30 | 3622 | **true** | 61 | 359 |

Per sample:

1. **selection correctness:** in all three the implementation selected the largest **eligible** detected proposal
   correctly (A1 → 48, A3 → 1, A4 → 30);
2. **visual/structural meaning (frozen audit):** the selected proposal is **not** the user's intended visually
   largest complete building in any of the three cases;
3. **discrepancy class:** **both** eligibility filtering and proposal coverage/fragmentation —
   the largest-area proposals (A1 #6 6470 px, A3 #4 2023 px, A4 #29 10404 px) are all rejected by the frozen
   extent/border rule, and the surviving eligible proposals are small fragments of larger roof complexes.

No ground-truth building identity is claimed.

## 4. `largest` semantic-contract audit (canonical `detector.py`, read-only)

* **eligibility criteria** (`eligible()`): proposal non-empty **and** `touches_image_border == False` **and**
  `bbox_extent_ratio <= MERGE_BBOX_EXTENT_RATIO_MAX` (= 0.20); the `family="largest"` branch applies no area floor;
* **tie-break** (`select_reference()`): `sorted(candidates, key=(-mask_area, -confidence, proposal_id))`, i.e.
  predicted mask area desc → confidence desc → global proposal id asc, evaluated over the **eligible** subset only;
* **meaning of “largest”:** largest among **eligible** proposals, not among all proposals.

```text
user-facing:    largest building in the image
implementation: largest eligible detected proposal by predicted mask area
                (eligible = non-empty ∧ not touching the original image border ∧ bbox extent ratio ≤ 0.20)
```

**Semantic-contract mismatch: CONFIRMED** (documented, not fixed).

## 5. A1/A4 SUCCESS-validity audit (canonical `pipeline.py`, read-only)

Post-inference hard gates applied before `status = SUCCESS`:

1. mask non-empty (`if not mask_full.any(): E404 empty_target_mask`, line ~297);
2. mask has pixels outside the context padding (`E404 mask_only_in_padding`);
3. target centroid satisfies the requested direction relative to the reference centroid
   (`direction_satisfied(...)`, line ~304, else `E404 direction_constraint_violated`).

Factual answers:

* minimum target area gate — **none**;
* target connected-component / instance-integrity gate — **none**;
* requirement that the target correspond to an existing detector proposal — **none**;
* validation is **only** non-empty + non-padding + directional-centroid.

A1 recorded `mask_area = 101` and A4 `mask_area = 359` (A3 `1439`); both A1 and A4 therefore returned `SUCCESS`
with tiny fragments, which the frozen audit classifies as semantic failure. No threshold is proposed here.

## 6. A2 E401 forensics (existing transcript/result only)

```text
Qwen initial program      = largest_to_left_of_to_nearest (confidence 0.98)
suggested program         = largest_to_left_of_to_nearest (FALLBACK_CORRECT, Y sent)
final language status     = FALLBACK_CORRECT
tile count                = 9
raw proposal count        = 0
merged proposal count     = 0
detector warning text     = "WARNING NMS time limit 2.050s exceeded"
final E401 detail         = 整幅影像未检测到任何建筑实例。
detector timing           = not present in the stored artifacts (absent value not inferred)
```

Supported conclusion: this is a **detector/proposal-stage failure** — the tiled U-C1 pass produced zero raw
detections, so no merged proposal and no eligible Reference existed. The NMS time-limit warning is recorded as an
observed message only; **causality between the warning and the empty proposal set is NOT established** by the
available evidence.

## 7. B1/B2 E502 memory forensics

Facts (frozen artifacts + static reading of canonical `detector.py`; nothing was executed):

* B1/B2 image size = **5000 × 5000** RGB;
* exception (both samples): `Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool`;
* one full-frame bool array at 5000 × 5000 = **25,000,000 B = 23.84185791015625 MiB (~23.84 MiB)**.

### A. Retained full-frame proposal masks (`detect_global()`, line 265)

```python
global_mask = np.zeros((height_px, width_px), dtype=bool)
```

* allocated **once per raw detection that reaches this point**;
* stored in the `accumulated` entry as `"mask": global_mask`, so full-frame proposal masks can coexist — retained
  until merge runs;
* approximate payload at 5000 × 5000: `N × 23.84 MiB`, excluding Python/object and other-array overhead, where `N`
  is the number of retained raw proposals; the frozen artifacts do not expose how many masks had accumulated before
  the failure, so **`N` is not invented**;
* `GlobalProposal.global_mask` **reuses** the same mask object (`merge_proposals()` passes `global_mask=mask`
  straight from the accumulated entry); there is no explicit constructor copy.

### B. Pairwise merge temporaries (`merge_proposals()` → `iou_of()`)

```python
np.logical_and(first, second)
np.logical_or(first, second)
```

* the operands are full-frame proposal masks;
* each result is also a full-frame bool array, i.e. **23.84 MiB at 5000 × 5000**;
* the two expressions execute **sequentially** in the current Python code, so this report does not claim that both
  temporaries necessarily coexist simultaneously;
* for `N` proposals the pairwise merge can perform up to `N(N-1)/2` comparisons, i.e. allocation churn in addition
  to the retained masks.

### Conclusion

```text
Exception signature: EXACT_SIGNATURE_MATCH (shape/dtype/size)
Unique throwing allocation site: NOT CONFIRMED FROM EXISTING ARTIFACTS
RC1-DEMO-MEM-01: CONFIRMED
Exact failure threshold: NOT ESTABLISHED
```

`detector.py:265` is one directly matching allocation site; `np.logical_and` / `np.logical_or` can request the same
shape, and the frozen transcript/result contain no traceback pinning a source line, so no single line is described
as the uniquely proven throwing line.

## 8. Defect taxonomy

| ID | status | evidence | affected samples | layer | severity | needs product code change | needs scientific model/checkpoint change |
|---|---|---|---|---|---|---|---|
| **RC1-DEMO-REF-01** — Reference semantics/proposal quality: user-facing “largest building” may resolve to a fragmented or eligibility-filtered proposal rather than the visually largest complete building | **CONFIRMED** | §3 table (largest-area proposals A1 #6 6470, A3 #4 2023, A4 #29 10404 all ineligible under extent/border; selected = largest eligible fragment) + §4 contract mismatch | A1, A3, A4 (all three successes) | proposal/reference layer | **blocker** for Demo semantic value | yes (fragmentation handling / eligibility policy / possibly detector quality) | not yet known |
| **RC1-DEMO-MASK-01** — SUCCESS validity too weak to reject tiny/incomplete target fragments | **CONFIRMED** | §5: only non-empty + non-padding + directional-centroid gates exist; A1 SUCCESS with 101 px, A4 with 359 px | A1, A4 | post-inference validity layer | major | yes (validity guard) | no |
| **RC1-DEMO-PROP-01** — A2 detector/proposal path can return zero proposals on a building-containing Demo scene | **CONFIRMED** (zero proposals is fact; NMS causality NOT confirmed) | §6: raw 0 / merged 0 over 9 tiles, `WARNING NMS time limit 2.050s exceeded` | A2 | detector/proposal stage | major | not yet known (investigation first) | not yet known |
| **RC1-DEMO-MEM-01** — large-image proposal path uses full-frame boolean proposal masks plus full-frame pairwise merge temporaries and failed on 5000 × 5000 inputs | **CONFIRMED** | §7: one retained full-frame bool mask per retained raw proposal (`detector.py:265`, kept in `accumulated` until merge), plus full-frame `np.logical_and` / `np.logical_or` bool temporaries per pairwise comparison (up to `N(N-1)/2`); exception is an EXACT_SIGNATURE_MATCH in shape/dtype/size, while the unique throwing expression is NOT CONFIRMED from existing artifacts | B1, B2 | large-image runtime/engineering | **blocker for the demonstrated 5000 × 5000 B1/B2 Demo path** | yes | no |

## 9. Dependency/prioritization recommendation (no implementation)

Decision policy applied: (1) run-blocking bugs → (2) proposal/reference correctness → (3) target validity guard →
(4) packaging/output layout → (5) free manual Demo.

1. **RC1-DEMO-MEM-01** first: it blocks the **demonstrated 5000 × 5000 B1/B2 path**, so later B1/B2 evidence
   (proposal quality, reference behaviour at scale) is blocked until it is resolved. The exact image-size /
   proposal-count failure threshold is **not established**, and no claim is made that all images larger than 512 px
   fail (the 1024 × 1024 samples are counterexamples).
2. **RC1-DEMO-PROP-01** second: A2 shows the proposal stage can yield zero detections; without proposals nothing
   downstream can act, so proposal reliability must be understood first.
3. **RC1-DEMO-REF-01** third: once the pipeline runs at both scales with a functioning proposal stage, the
   “largest building” semantics (fragmentation + eligibility) is the dominant semantic blocker — it affects 3/3
   runtime successes in the frozen visual audit.
4. **RC1-DEMO-MASK-01** fourth: the validity guard depends on knowing what a legitimate target is, which follows
   from 1–3; it is the smallest self-contained change.
5. Packaging/output layout (Task 8B.4) and free manual Demo remain last.

Dependency note: MEM-01 is independent; PROP-01 and REF-01 both live in the detection/proposal layer and should be
investigated together before a fix is chosen; MASK-01 is independent of the first three but should not be finalized
before the intended target definition is settled.

Task 8B.4 stays deferred until ChatGPT reviews this report.

## 10. No-change statement

No inference was executed (`predict.py`, `scripts/task8b3_interactive_suite.py`, detector, SAM2, D-B1 and Qwen were
never invoked in this task). No file under the external delivery was created, modified or deleted; no existing
inference input/output/log/diagnostic artifact was touched; no product/canonical/harness/test source, threshold,
config, checkpoint or model asset was modified; no package was installed; nothing was downloaded; final-test data
was not accessed; Assisted Mode / `--reference-id` / `--inspect-proposals` were not used. All findings come from
reading the frozen R4B artifacts listed in the task book and the canonical source.

## 11. Next gate

**Awaiting ChatGPT audit.** No defect is fixed here; do not run the Demo, do not enter Task 8B.4 or Task 8C.

### Audit history

Task 8B.3-D1.1 corrected the original D1 memory-forensics report by adding pairwise full-frame IoU temporaries,
removing the unsupported >512-px generalization, and separating exact allocation-signature matching from unique
throwing-site attribution. The normalized §7–§9 above are authoritative.
