# Task 8B.3-P1D12-R1 — Locked Demo Proposal Evidence Closure (read-only)

## 1. Task and scope

Read-only completion of the P1D12 evidence: no candidate, `predict.py`, detector or model was re-run. All facts below
come from the frozen rasters, the canonical source, the external delivery and the diagnostics produced by P1D12.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 75e01defde3ad6e9f8a247481028536e68ef4ee4
model / detector / candidate executions = NONE
```

## 2. External source/config integrity (135/135)

```text
external manifest-listed files equal Git canonical bytes = 135/135 PASS (mismatches: none)
external source_manifest.json == Git canonical control   = True
external setup checker: exit 0 · READY True (C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe)
```

## 3. Locked raster identity (authoritative bytes)

| relation | path | bytes | SHA256 | size | mode | format | SHA/dimension lock |
|---|---|---:|---|---|---|---|---|
| right | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif` | 791306 | `1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2` | 512×512 | RGB | TIFF | PASS |
| left | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif` | 791846 | `eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38` | 512×512 | RGB | TIFF | PASS |
| above | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif` | 791534 | `0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd` | 512×512 | RGB | TIFF | PASS |
| below | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif` | 792202 | `c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7` | 512×512 | RGB | TIFF | PASS |

All four rasters match the immutable SHA256 locks, are 512×512 and decode as RGB.

## 4. Frozen detector constants (canonical source, read-only)

| constant | canonical value | expected | result |
|---|---|---|---|
| `TILE_SIZE` | `512` | `512` | PASS |
| `TILE_OVERLAP` | `128` | `128` | PASS |
| `TILE_STRIDE` | `TILE_SIZE - TILE_OVERLAP` | `TILE_SIZE - TILE_OVERLAP` | PASS |
| `IMGSZ` | `640` | `640` | PASS |
| `CONF` | `0.05` | `0.05` | PASS |
| `MAX_DET` | `300` | `300` | PASS |
| `DUPLICATE_IOU` | `0.50` | `0.50` | PASS |
| `MERGE_BBOX_EXTENT_RATIO_MAX` | `0.20` | `0.20` | PASS |

The P1D12 diagnostics report `tile_count = 1` because the inspect-proposals path plans a single whole-image tile
(`tile_size = 512`, `overlap = 128` are still recorded by that path), so no tiled orchestration was involved.

## 5. Existing diagnostics inspect-only evidence

| relation | tile | status | raw | merged | items | eligible (recomputed) | tile_count | file set |
|---|---|---|---:|---:|---:|---:|---:|---|
| right | 1010 | SUCCESS | 6 | 6 | 6 | 4 | 1 | PASS |
| left | 1003 | SUCCESS | 66 | 53 | 53 | 42 | 1 | PASS |
| above | 1008 | SUCCESS | 9 | 9 | 9 | 4 | 1 | PASS |
| below | 1009 | SUCCESS | 7 | 6 | 6 | 3 | 1 | PASS |

For every candidate the recomputed `eligible_largest_count` (from `proposals.json` items only) equals the P1D12 report
value, `merged_proposal_count == proposals.count`, the file set is exactly
`['global_proposals.png', 'parsed_program.json', 'prompt.txt', 'proposals.json', 'result.json']`, and no mask/overlay file exists.

## 6. Non-execution proof

```text
result.json fields matching sam2 / relation / d-b1 / decoder / language / reference / qwen = NONE (all four)
mask or overlay artifacts = NONE (all four)
timing/field evidence therefore shows SAM2 = 0, relation fields = 0, D-B1 = 0
```

## 7. Explicit confirmations

```text
candidate re-runs / predict / detector / model executions = NONE
candidate replacement / GT access / visual judgement      = NONE / NONE / NONE
locked candidate identities                               = UNCHANGED (FINAL_METADATA_LOCK)
```

## 8. Outcome

```text
PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_CLOSED
NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

Every required evidence gate passes (raster identity 4/4, dimensions 4/4, RGB readability 4/4, external integrity
135/135, setup READY, detector constants MATCH, inspect-only proof PASS), so the evidence-closure outcome applies. The
next gate is not executed here.
