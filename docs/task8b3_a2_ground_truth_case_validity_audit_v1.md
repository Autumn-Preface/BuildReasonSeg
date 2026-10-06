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

Search coverage: `{"C1": {"all_available_cropped_images_indexed": true, "counts_by_raster": {"test": 3726, "train1": 10044, "train2": 3618}, "errors": [], "matches": 0, "ordered_ref_decoded_rgb_hash_index_sha256": "d3c7e87b2802656eb61069a02a97ab621312fe464e38b9fa50c502df39bd2616", "tiles": 17388}, "C2": {"execution": "same complete decoded index, predeclared windows; no adaptive locations", "matches": 0, "tiles": 17388, "windows": 5}, "C3": [{"anchor_occurrences": 0, "bytes": 3001216122, "bytes_scanned": 3001216122, "decoder_exact_checks": [{"equal": true, "tile_id": "42", "window": [21504, 0, 512, 512]}, {"equal": true, "tile_id": "1306", "window": [32768, 9216, 512, 512]}, {"equal": true, "tile_id": "3723", "window": [33792, 27136, 512, 512]}], "dimensions": [35765, 27802], "exact_patch_hits": 0, "nonmonotonic_logical_tile_offsets": 164, "patch_candidates_checked": 0, "raster": "test.tif", "seconds_total_elapsed": 8.328, "source_file_sha256": "5325c8f38086fc5084dc64377da4cf63370da295a20284d197304135be263cde", "status": "COMPLETE_ALL_BYTES", "tile_count": 61040, "tile_dimensions": [128, 128]}, {"anchor_occurrences": 0, "bytes": 8006812933, "bytes_scanned": 8006812933, "decoder_exact_checks": [{"equal": true, "tile_id": "1_0", "window": [0, 0, 512, 512]}, {"equal": true, "tile_id": "1_4124", "window": [16384, 11264, 512, 512]}, {"equal": true, "tile_id": "1_10043", "window": [94720, 27136, 512, 512]}], "dimensions": [95521, 27801], "exact_patch_hits": 0, "nonmonotonic_logical_tile_offsets": 0, "patch_candidates_checked": 0, "raster": "train1.tif", "seconds_total_elapsed": 32.735, "source_file_sha256": "37b537d6f9cfc1e0814b04ed4a549b6479b1e125e303650e3cb2474160d36012", "status": "COMPLETE_ALL_BYTES", "tile_count": 162846, "tile_dimensions": [128, 128]}, {"anchor_occurrences": 0, "bytes": 2915467130, "bytes_scanned": 2915467130, "decoder_exact_checks": [{"equal": true, "tile_id": "2_1", "window": [512, 0, 512, 512]}, {"equal": true, "tile_id": "2_594", "window": [29696, 4096, 512, 512]}, {"equal": true, "tile_id": "2_3617", "window": [33792, 27136, 512, 512]}], "dimensions": [34772, 27802], "exact_patch_hits": 0, "nonmonotonic_logical_tile_offsets": 164, "patch_candidates_checked": 0, "raster": "train2.tif", "seconds_total_elapsed": 40.547, "source_file_sha256": "dc2f245d41832fbff9dd12beb4460bddbad9a5c3d3b81357c961550bad2a3aca", "status": "COMPLETE_ALL_BYTES", "tile_count": 59296, "tile_dimensions": [128, 128]}]}`.
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

Validation: {"script_syntax_import_checkpoint2": "PASS: compile(source), runpy.run_path with non-main name; -B; no model imports"}.
The audit makes no domain reclassification, detector improvement, semantic segmentation,
PROP-01 closure or replacement-case claim. The accepted detector zero result is historical evidence;
no inference is repeated. Only the authorized diagnostic, report, JSON and two handoff paths change.
External RC1 and WHU archive inventory identities are checked before and after read-only work.
No package installation, data/model download, external write or product edit occurs.

Next gate: CHATGPT_A2_GT_CASE_VALIDITY_REMOTE_AUDIT. STOP at READY_FOR_SUPERVISOR_AUDIT.
