# CURRENT_TASK — A2_GROUND_TRUTH_AND_LOCKED_CASE_VALIDITY_AUDIT_V1

## 0. Metadata

Task ID:
A2_GROUND_TRUTH_AND_LOCKED_CASE_VALIDITY_AUDIT_V1

Status:
AUTHORIZED

Decision owner:
ChatGPT Supervisor

Authorized executor:
CODEX

Required starting branch:
fix/task8b3-detector-rgb-bgr-contract-v1

Required starting remote HEAD:
104bcde03ff8bedbd563a1c0eba6fe230f221dad

Task branch:
audit/task8b3-a2-ground-truth-case-validity-v1

Previous milestone:
DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1

Previous milestone disposition:
ACCEPTED_BY_CHATGPT_SUPERVISOR

Next gate:
CHATGPT_A2_GT_CASE_VALIDITY_REMOTE_AUDIT


## 1. Supervisor Disposition

The ChatGPT Supervisor has independently audited and ACCEPTED:

DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1

Accepted engineering conclusion:

DETECTOR_RGB_BGR_INPUT_CONTRACT = CORRECTED

Accepted scientific boundary:

- the RGB/BGR input-contract defect was real;
- the corrected product now preserves the intended RGB network tensor;
- the correction did NOT recover A2 proposals;
- corrected A2 remains raw=0 / merged=0 / reference=None;
- PROP-01 therefore remains OPEN;
- A2 source provenance, GT availability and locked-case validity remain unresolved.

This milestone is forensic only.

It MUST NOT perform another detector repair.


## 2. Accepted Starting Facts

The following facts are accepted and MUST NOT be re-litigated unless contradictory new evidence is found.

### A2 locked identity

External complete RC1 path:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A2.png

File SHA256:

10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f

Decoded RGB:

1024 × 1024
RGB
uint8

Program:

largest_to_left_of_to_nearest


### Detector result

After the accepted RGB/BGR repair:

raw proposals = 0
merged proposals = 0
reference = none

Do NOT use proposal recovery as an acceptance criterion.


### Historical provenance state

Previous P1D8/P1D10-era audits established:

- no exact file identity between A2 and the 17,388 active 512×512 detector-domain tiles;
- A2 geographic/source provenance = NOT ESTABLISHED;
- A2 train/val/test membership = NOT ESTABLISHED;
- unknown provenance does NOT prove that A2 is outside the supported domain;
- the earlier supported-domain reclassification based on unknown provenance was withdrawn;
- PROP-01 remains OPEN_ENGINEERING_DEFECT in governance state.

Previous detector/domain probes are historical evidence only.
Do not repeat them.


### Available GT infrastructure

Existing repository evidence establishes:

- WHU whole-image ↔ 512-tile mapping has been validated;
- native EA.shp polygons are the primary building-instance truth;
- raster/vector alignment has already been validated at dataset level;
- datasets/whu_native_vector/v1.0/tiles/index.jsonl is tracked;
- native-vector stable source identity uses EA.shp record order;
- BuildSpatialReason v0.2 carries frozen relation metadata and native-vector references.

Therefore an A2 provenance/GT audit is technically meaningful if exact source provenance can be recovered.


## 3. Goal

Answer the following questions using the strongest available exact evidence:

1. Where did locked A2 originate?
2. Can A2 be mapped exactly to a canonical source raster/window?
3. If exact provenance is recovered, what ground-truth building instances exist in that exact window?
4. Under the already-frozen spatial-relation semantics, is A2 genuinely a valid:

   largest_to_left_of_to_nearest

   evaluation case?
5. Which interpretation is supported:

   - valid GT case where detector zero genuinely persists;
   - invalid/misconstructed locked evaluation case;
   - image/GT alignment defect;
   - provenance-unresolved case that cannot yet be scientifically classified?

This task does NOT repair PROP-01.

This task does NOT select a replacement Demo case.


## 4. Allowed Scope

Read-only inspection is allowed for:

- repository source;
- Git history;
- repository docs;
- evaluation evidence;
- historical handoff/report files;
- external complete RC1;
- existing local project artifacts referenced by repository provenance;
- existing WHU source archive, if present;
- existing native-vector dataset metadata and geometry resources;
- OS/task-local temporary scratch required for deterministic analysis.

Important permitted roots include:

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1

C:\D\resources\Satellite dataset Ⅱ (East Asia)

datasets/whu_native_vector/v1.0/**

datasets/build_spatial_reason/v0.2/**

existing Task 6K.1 / 6L artifacts and scripts.


### Allowed repository writes

Only:

scripts/diagnose_a2_ground_truth_case_validity.py

docs/task8b3_a2_ground_truth_case_validity_audit_v1.md

evaluation/task8b3_a2_ground_truth_case_validity_audit_v1.json

evaluation/task8b3_a2_ground_truth_case_validity_audit_v1_overlay.png
ONLY when exact GT-backed provenance is established

handoff/CURRENT_TASK.md

handoff/EXECUTOR_STATE.yaml


No other repository path is authorized.


## 5. Forbidden Scope

Do NOT:

- modify any delivery product/runtime source;
- modify detector.py;
- modify ProgramHead;
- modify Qwen;
- modify SAM/SAM2;
- modify GRF;
- modify reference ranking;
- modify spatial executor semantics;
- modify governance/PROJECT_STATE.yaml;
- modify governance/DECISIONS.md;
- change confidence threshold;
- change NMS;
- change imgsz;
- change max_det;
- change mask threshold;
- change tile size;
- change overlap/stride;
- run any threshold sweep;
- run detector inference;
- run Qwen inference;
- run SAM/SAM2 inference;
- run D-B1;
- run full predict CLI;
- train/fine-tune any model;
- install or update dependencies;
- download data/models;
- modify external RC1;
- modify WHU source files;
- replace A2;
- alter existing P1D10 locked Demo candidates;
- select a replacement case;
- declare PROP-01 closed;
- reclassify PROP-01 autonomously.


## 6. Decision Boundary

### L0-L1 — Executor may decide

Inside the frozen audit contract, Codex may choose:

- exact hashing implementation;
- exact-pixel indexing implementation;
- efficient chunked raster reads;
- deterministic scratch layout;
- deterministic evidence serialization;
- existing geometry helper reuse;
- performance-safe exact-search mechanics.

### L2-L4 — STOP + Supervisor

Codex MUST NOT independently:

- invent a new relation definition;
- redefine "largest";
- redefine "left_of";
- redefine "nearest";
- create a fuzzy similarity threshold;
- use perceptual matching as provenance proof;
- redefine case validity;
- change GT acceptance rules;
- invoke a model to resolve provenance;
- modify scientific conclusions.


## 7. Required Preflight

First read, in order:

1. AGENTS.md
2. governance/PROJECT_STATE.yaml
3. governance/DECISIONS.md
4. handoff/CURRENT_TASK.md
5. handoff/EXECUTOR_STATE.yaml

Because CURRENT_TASK in the starting commit belongs to the accepted predecessor milestone, install THIS Supervisor task book byte-for-byte after Git preflight and task-branch creation.

Then verify:

git branch

git HEAD

remote predecessor branch

remote predecessor HEAD

git status

git diff


Required starting remote state:

fix/task8b3-detector-rgb-bgr-contract-v1

HEAD:

104bcde03ff8bedbd563a1c0eba6fe230f221dad


If the actual remote has advanced:

DO NOT reset.

Inspect the new commits read-only.

If they are not obviously Supervisor task-install/governance-only commits for this exact milestone:

update EXECUTOR_STATE with the facts and STOP.


If the expected starting state is confirmed:

create:

audit/task8b3-a2-ground-truth-case-validity-v1

from the verified starting HEAD.

Do not modify the accepted predecessor history.


## 8. Provenance Evidence Levels

Use the following exact provenance levels.


### P0 — documentary exact provenance

A pre-existing authoritative record identifies:

- original source path/image;
- exact crop/window coordinates or equivalent deterministic construction;
- enough identity to reproduce A2.

P0 should still be pixel-verified when the source remains available.


### P1 — full exact-pixel provenance

A source raster/window is identified and:

decoded RGB of complete reconstructed 1024×1024 source window

is byte-identical to:

decoded RGB of locked A2.

P1 is sufficient for GT mapping.


### P2 — partial exact evidence

Examples:

- one or more A2 quadrants exactly match canonical tiles;
- an exact interior patch identifies a source scene;
- a candidate source location exists;
- but complete 1024×1024 equality cannot be established.

P2 is NOT sufficient for final GT case-validity classification.


### P3 — approximate / perceptual similarity

P3 is NOT provenance proof.

Do not use it as the basis for this milestone's scientific result.


## 9. Phase A — Historical Documentary Provenance

Search read-only through:

- Git history;
- historical Task 8B.3 task books/reports;
- six-case suite records;
- external suite logs;
- preserved diagnostics;
- external file metadata;
- previous A2 forensics;
- any pre-existing copy/source records.

Determine, if possible:

- who/what introduced A2;
- original source path;
- source dataset;
- crop/window coordinates;
- transformation chain;
- why A2 was selected into the six locked cases.

Every claim must be attached to an exact evidence source.

Classify each finding as:

AUTHORITATIVE

DESCRIPTIVE_ONLY

INFERRED

Only AUTHORITATIVE evidence may establish P0.


## 10. Phase B — Locked A2 Identity

Re-verify current external A2:

path

dimensions

mode

dtype

file bytes

SHA256

decoded-RGB SHA256


Required file SHA256:

10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f


Confirm:

runtime decode / Pillow RGB decode identity

using read-only operations.

No image rewrite or re-encoding.


## 11. Phase C — Exact Pixel Provenance Search

Use exact-pixel evidence only.


### C1. 512×512 quadrant search

Compute decoded RGB hashes for:

A2[0:512, 0:512]

A2[0:512, 512:1024]

A2[512:1024, 0:512]

A2[512:1024, 512:1024]

Compare against the complete available canonical WHU 512×512 tile set.

Do not assume that one match proves the whole image.


### C2. Deterministic interior windows

If quadrants do not resolve provenance, test a small predeclared deterministic set of additional exact 512×512 A2 windows.

The locations MUST be written to evidence before the corpus search.

No adaptive search based on "looks promising" results.


### C3. Bounded exact-patch search

If required, perform a bounded exact-pixel patch search.

Requirements:

- patch size/locations declared before search;
- exact byte equality only;
- no SSIM;
- no feature matching threshold;
- no histogram distance threshold;
- no perceptual nearest-neighbour acceptance.

An exact patch hit is only a candidate source position.


### C4. Full-window verification

Any candidate source location MUST be verified by reconstructing the complete 1024×1024 source window.

P1 requires:

reconstructed source-window decoded RGB
==
locked A2 decoded RGB

for every pixel/channel.


If full equality is not achieved:

do not claim exact provenance.


### C5. Computational bound

If an exact whole-source search becomes computationally unreasonable:

record the completed search coverage;

set the strongest justified provenance status;

do not invent a fuzzy rescue method.


## 12. Phase D — GT Recovery

Execute ONLY if P0/P1 yields an exact GT-backed source window.

Resolve:

- whole source raster;
- source pixel window;
- corresponding raster label;
- corresponding native EA.shp geometry;
- native-vector archive identity.

Use the existing validated WHU mapping.

Do not redefine it.


For every building intersecting the A2 window record:

- source_feature_id;
- stable source UID;
- clipped pixel area;
- full area where available;
- centroid in A2 coordinates;
- bbox in A2 coordinates;
- border-clipped state;
- visible fraction if available.


Record:

GT building instance count.


## 13. Phase E — Image / GT Alignment

If exact source provenance exists:

validate the exact A2 window against existing raster/vector truth.

Reuse existing Task 6K.1 alignment machinery/definitions.

Do NOT invent a new IoU or offset acceptance threshold.


Record:

- source raster identity;
- raster-label identity;
- EA.shp identity;
- crop/window coordinates;
- applicable alignment metric;
- any contradiction.

If the recovered image/window and GT genuinely conflict:

persist evidence

set:

gt_status = GT_ALIGNMENT_CONFLICT

STOP for Supervisor review.


## 14. Phase F — Frozen Relation Semantics

Locate the canonical already-frozen implementation/definition of:

largest_to_left_of_to_nearest

from the BuildSpatialReason / spatial_reasoning lineage.

Document the exact code/data source used.


Do NOT invent special semantics for A2.


First determine whether the frozen semantics are directly applicable to a 1024×1024 A2 GT window.

If applying them would require a new interface/contract:

set:

case_validity = SEMANTICS_NOT_PORTABLE_TO_A2

persist evidence

STOP semantic classification.


If directly applicable, derive from GT only:

1. largest reference building;
2. left_of eligible candidates;
3. nearest target according to frozen semantics;
4. tie/ambiguity state;
5. complete:

Reference
→ Relation
→ Target

chain validity.


No detector proposal may participate in this calculation.


## 15. Required Result Enums

Machine-readable evidence MUST keep these dimensions separate.


### provenance_status

Exactly one:

EXACT_DOCUMENTARY_AND_PIXEL_CONFIRMED

EXACT_PIXEL_CONFIRMED

PARTIAL_EXACT_EVIDENCE

NOT_ESTABLISHED

CONFLICTING_PROVENANCE_EVIDENCE


### gt_status

Exactly one:

GT_AVAILABLE_ALIGNED

GT_ALIGNMENT_CONFLICT

GT_UNAVAILABLE

NOT_EVALUATED_NO_EXACT_PROVENANCE


### case_validity

Exactly one:

VALID_REFERENCE_RELATION_TARGET_CHAIN

INVALID_NO_BUILDING_INSTANCES

INVALID_NO_VALID_REFERENCE

INVALID_NO_LEFT_OF_TARGET

INVALID_NO_NEAREST_TARGET

AMBIGUOUS_GT_RELATION

SEMANTICS_NOT_PORTABLE_TO_A2

NOT_EVALUABLE_WITHOUT_PROVENANCE


### prop01_interpretation_candidate

Exactly one:

VALID_GT_CASE_DETECTOR_ZERO_PERSISTS

LOCKED_CASE_CONSTRUCTION_DEFECT_CANDIDATE

IMAGE_GT_ALIGNMENT_DEFECT_CANDIDATE

PROP01_ROOT_CAUSE_REMAINS_UNRESOLVED


These are evidence classifications.

They do NOT authorize the Executor to change governance state.


## 16. Required Evidence

Create:

docs/task8b3_a2_ground_truth_case_validity_audit_v1.md

evaluation/task8b3_a2_ground_truth_case_validity_audit_v1.json


Preferred diagnostic implementation:

scripts/diagnose_a2_ground_truth_case_validity.py


If exact GT-backed provenance is established, also create:

evaluation/task8b3_a2_ground_truth_case_validity_audit_v1_overlay.png


Overlay requirements:

- original A2 background;
- GT building boundaries;
- reference clearly identified;
- target clearly identified;
- evidence only;
- must not influence the classification.


## 17. Required JSON Fields

At minimum:

task_id

status

starting_branch

starting_head

task_branch

locked_a2_path

locked_a2_file_sha256

locked_a2_decoded_rgb_sha256

historical_sources_inspected

source_roots_inspected

exact_search_plan

exact_search_anchors

exact_matches

full_window_verification

provenance_status

resolved_source_raster

resolved_source_window

gt_source_identities

gt_building_instance_count

gt_instances

relation_semantics_source

gt_reference

gt_left_of_candidates

gt_target

gt_status

case_validity

prop01_interpretation_candidate

no_model_inference = true

no_threshold_sweep = true

no_parameter_tuning = true

no_product_change = true

no_external_write = true

no_candidate_replacement = true


Unavailable fields must be null / empty with an explicit reason.

Never fabricate completeness.


## 18. Validation

Required:

- syntax check of any new diagnostic script;
- import check;
- JSON parse;
- required-enum assertions;
- deterministic saved-evidence finalize phase;
- changed-path audit;
- final Git diff audit;
- exact A2 SHA256 re-check.

Preferred script interface:

--phase provenance

--phase gt

--phase finalize


`--phase gt` must refuse to run unless saved evidence contains exact provenance sufficient for GT mapping.

`--phase finalize` must perform zero model inference and validate already-saved evidence.


No full pytest suite required.

No product regression suite required.

No model inference allowed.


## 19. Checkpoint Rules

### Checkpoint 1

Completed:

historical documentary search

+

locked A2 identity verification


Update EXECUTOR_STATE.

Commit + push.


### Checkpoint 2

Completed:

exact pixel provenance search


If result is:

NOT_ESTABLISHED

or:

PARTIAL_EXACT_EVIDENCE

this is a VALID completion path.

Do NOT rescue with fuzzy matching.

Finalize evidence and proceed to READY_FOR_SUPERVISOR_AUDIT.


If exact provenance is established:

continue automatically to GT/alignment audit.


### Checkpoint 3

Completed:

GT + relation validity audit


Update EXECUTOR_STATE.

Commit + push.


## 20. STOP / Escalation Conditions

STOP after preserving evidence if:

- documentary and pixel provenance conflict;
- source/GT alignment contradicts the validated mapping;
- relation semantics require a new L2 contract;
- exact search would require an unsafe/unbounded destructive operation;
- fuzzy/perceptual matching would be needed as proof;
- model inference would be needed;
- unknown workspace modifications exist;
- Git state is unsafe;
- an external source would have to be modified.


Do NOT perform another repair.


## 21. Scientific Claim Boundary

This milestone may establish only:

- exact or unresolved A2 source provenance;
- GT availability/alignment when exact provenance permits;
- locked A2 case validity under already-frozen semantics.


It does NOT establish:

- detector improvement;
- segmentation semantic correctness;
- new supported-domain boundary;
- PROP-01 closure by itself;
- replacement Demo case;
- detector adaptation;
- new relation semantics;
- architecture improvement.


## 22. Git Safety

Forbidden:

git reset --hard

git rebase

git commit --amend

git stash

git clean

force push

history rewrite

deleting unknown modifications


Unknown work must be preserved.


## 23. Final State

A complete forensic result includes a legitimate:

provenance_status = NOT_ESTABLISHED

if exact evidence cannot recover A2's origin.

Do not treat that as Executor failure.


On completion:

handoff/CURRENT_TASK.md

Status:

READY_FOR_SUPERVISOR_AUDIT


handoff/EXECUTOR_STATE.yaml

status:

READY_FOR_SUPERVISOR_AUDIT


Commit and push all authorized evidence.

Then STOP.


Next gate:

CHATGPT_A2_GT_CASE_VALIDITY_REMOTE_AUDIT


Do NOT automatically begin:

- detector tuning;
- detector adaptation;
- Demo-case replacement;
- Task 8B.4;
- final Demo.