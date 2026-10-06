# A2 ground-truth and locked-case validity audit V1

Task: A2_GROUND_TRUTH_AND_LOCKED_CASE_VALIDITY_AUDIT_V1. Status: IN_PROGRESS.
Starting branch: fix/task8b3-detector-rgb-bgr-contract-v1, remote HEAD `104bcde03ff8bedbd563a1c0eba6fe230f221dad`.
Task branch: `audit/task8b3-a2-ground-truth-case-validity-v1`. All findings are evidence for Supervisor review; PROP-01 remains OPEN.

## Result dimensions

- provenance_status: `NOT_ESTABLISHED`
- gt_status: `NOT_EVALUATED_NO_EXACT_PROVENANCE`
- case_validity: `NOT_EVALUABLE_WITHOUT_PROVENANCE`
- prop01_interpretation_candidate: `PROP01_ROOT_CAUSE_REMAINS_UNRESOLVED`

## History and locked identity

Current external A2 file SHA256: `10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f`.
Decoded RGB SHA256: `d02b3597a389285d72874fea12a06dbc031c720bfc9506e986a184b6637940bf`; 1024 x 1024 RGB uint8, 1,677,040 bytes.
Actual external runtime imageio.load_image output equals direct Pillow RGB byte-for-byte.
File timestamp metadata is descriptive only and does not establish source provenance.

The first tracked suite definition is commit `1287d0a2bab50a278452d4cd0ec8d492347afdfe`
(2026-10-03, author recorded by Git). Its Supervisor task book specifies the A2 path, prompt and
largest_to_left_of_to_nearest program. This establishes introduction into the tracked suite,
not who created the original image. The pre-existing RC1 Task 8B.2 inventory already lists A2.
The R4B freeze/results fix the input hash and program, not a GT-backed construction.
P1D5, P1D8 and P1D10 records report unresolved origin; prior supported-domain conclusions are
not adopted. Search commands, inspected task-book versions, source hashes and exact line pointers
are saved in the JSON. No authoritative source raster/window or transformation chain was recovered.
The reason for selecting these exact pixels is not established by the inspected documentary records.

## Predeclared exact search

C1: four disjoint 512 x 512 quadrants. C2: five additional fixed 512 x 512 windows.
C3: three fixed 16 x 16 patches at (256,256), (496,496), (752,752), with both first-row
8-pixel halves searched exactly. The saved plan predates corpus execution and is checkpointed in Git.
The two halves cover tile-boundary crossings; every candidate must pass full patch equality.
C4 requires complete in-bounds 1024 x 1024 RGB equality. No transformations, fuzzy matching,
perceptual comparisons, threshold fitting or model calls are used.

Search coverage: `{}`.
Exact matches: 0. Full-window records: 0.
Limitations and unavailable-field reasons are saved explicitly in the JSON.

## Conditional GT and frozen semantics

GT mapping, image/GT alignment and relation classification require exact full-window provenance.
Native-vector tile index and Task 6K.1 mapping/validation evidence are inspected read-only.
Frozen lineage: spatial_reasoning/annotator.py generate_level3; semantic_policy.py
resolve_size_extreme, direction_candidates_over_visible and resolve_nearest; geometry.py
boundary distance; configs/spatial_relations_v1.yaml; BuildSpatialReason v0.2 manifest.
Global visible largest is checked for frozen ambiguity/eligibility; left_of is the frozen
subject-relative direction predicate; nearest is boundary distance over the full direction set,
with frozen margin/eligibility, without substituting an eligible runner-up.
No A2 GT-derived answer or 1024-window portability decision is made without exact provenance.
No overlay is created without exact GT-backed provenance.

## Validation and claim boundary

Validation: {}.
The audit makes no domain reclassification, detector improvement, semantic segmentation,
PROP-01 closure or replacement-case claim. The accepted detector zero result is historical evidence;
no inference is repeated. Only the authorized diagnostic, report, JSON and two handoff paths change.
External RC1 and WHU archive inventory identities are checked before and after read-only work.
No package installation, data/model download, external write or product edit occurs.

Next gate: CHATGPT_A2_GT_CASE_VALIDITY_REMOTE_AUDIT. STOP at READY_FOR_SUPERVISOR_AUDIT.
