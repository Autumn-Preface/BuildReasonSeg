# Task 8B.3-P1D12 — Locked Demo Proposal-Only Gate

## 1. Task and scope

The four immutable locked candidates were run once each through the **proposal-only** `--inspect-proposals` path in the
frozen order right → left → above → below. No Qwen, SAM2, D-B1, reference selection, GT or manual visual judgement was
involved and no candidate was replaced.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 6651c4bf89663e5fff3e7854c9e2115d084ae003
candidates run = 4 of 4 · process failures = 0
pre-existing candidate diagnostics = NONE
```

## 2. Locked candidate identities (unchanged)

| relation | sample_id | source raster |
|---|---|---|
| right | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif` |
| left | `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif` |
| above | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif` |
| below | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif` |

## 3. Command form

```text
<RC1_PYTHON> predict.py --image "<LOCKED_ABSOLUTE_IMAGE_PATH>" --inspect-proposals
```

No `--prompt`, no `--reference-id`. Expected path: model-package verification → image load →
`DetectorRuntime.detect_global` → proposal merge → diagnostics save → return.

## 4. Summary table

| relation | sample_id | raw | merged | eligible_largest | exit | inspect-only |
|---|---|---:|---:|---:|---:|---|
| right | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` | 6 | 6 | 4 | 0 | yes |
| left | `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` | 66 | 53 | 42 | 0 | yes |
| above | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` | 9 | 9 | 4 | 0 | yes |
| below | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` | 7 | 6 | 3 | 0 | yes |

## 5. Per-candidate evidence

**right** — `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` (tile 1010)

```
source raster   : C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif
command         : <RC1_PYTHON> predict.py --image "<locked raster>" --inspect-proposals
exit code       : 0 · wall 5.42 s
diagnostics dir : C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010
status          : SUCCESS
raw / merged    : 6 / 6
proposal items  : 6
tile count/size/overlap : 1 / 512 / 128
eligible_largest_count  : 4
result.json sha256      : 54c964933fca667ff9c55593439d3059c2f8ec18d30951d16a417eac880cc518
proposals.json sha256   : 2b09041ee86d5e7d71c05c32f2cea8669bdfbb5b7f96cda33d5e12c99a0ac45b
diagnostics files       : ['global_proposals.png', 'parsed_program.json', 'prompt.txt', 'proposals.json', 'result.json']
```

**left** — `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` (tile 1003)

```
source raster   : C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif
command         : <RC1_PYTHON> predict.py --image "<locked raster>" --inspect-proposals
exit code       : 0 · wall 4.47 s
diagnostics dir : C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003
status          : SUCCESS
raw / merged    : 66 / 53
proposal items  : 53
tile count/size/overlap : 1 / 512 / 128
eligible_largest_count  : 42
result.json sha256      : d2cef24ea3b15a528a8f2dec29faeb9fa1992bb9385633cdb2e419ea9039ad8e
proposals.json sha256   : 62865e42b4d92077ce336522b97f33931b351d46ab4ec7443fb1afe2dfd798a6
diagnostics files       : ['global_proposals.png', 'parsed_program.json', 'prompt.txt', 'proposals.json', 'result.json']
```

**above** — `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` (tile 1008)

```
source raster   : C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif
command         : <RC1_PYTHON> predict.py --image "<locked raster>" --inspect-proposals
exit code       : 0 · wall 4.36 s
diagnostics dir : C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008
status          : SUCCESS
raw / merged    : 9 / 9
proposal items  : 9
tile count/size/overlap : 1 / 512 / 128
eligible_largest_count  : 4
result.json sha256      : bf70127106e5798b5be40a2aa317ac8921abe99b1e5015c9945a912e4657184c
proposals.json sha256   : 257297112fb2fe35f2225725e8c0a0e79338dcde10f9f4dbe20641c8fbb85e0b
diagnostics files       : ['global_proposals.png', 'parsed_program.json', 'prompt.txt', 'proposals.json', 'result.json']
```

**below** — `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` (tile 1009)

```
source raster   : C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif
command         : <RC1_PYTHON> predict.py --image "<locked raster>" --inspect-proposals
exit code       : 0 · wall 4.5 s
diagnostics dir : C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009
status          : SUCCESS
raw / merged    : 7 / 6
proposal items  : 6
tile count/size/overlap : 1 / 512 / 128
eligible_largest_count  : 3
result.json sha256      : ab75dd17e082c171e0b0f92eb9586329216bb32a1eafb317bcb2eab323e91507
proposals.json sha256   : 7c1a45690c860910b5626fc603fe942cc0497157bf6fc5d1645657029c167752
diagnostics files       : ['global_proposals.png', 'parsed_program.json', 'prompt.txt', 'proposals.json', 'result.json']
```

Internal consistency held for all four candidates: `status = SUCCESS`, `merged_proposal_count = proposals.count`,
`tile_count = 1`, `tile_size = 512`, `overlap = 128`.

## 6. Mechanical frozen eligibility count

Computed from `proposals.json` items only, with no reference selection:
`mask_area > 0 AND touches_image_border == false AND bbox_extent_ratio <= 0.20`.
Counts: right=4, left=42, above=4, below=3.

## 7. Core-chain non-execution

`result.json` carries no language/parse, reference-selection, relation or decoder field for any candidate, and no
candidate mask or overlay file was produced (each diagnostics directory contains only `global_proposals.png`,
`parsed_program.json`, `prompt.txt`, `proposals.json`, `result.json` — the standard inspect-mode set). No visual
inspection of any preview was performed.

## 8. Explicit confirmations

```text
visual inspection / manual quality judgement = NONE
GT access                                    = NONE
candidate replacement                        = NONE
Qwen / SAM2 / D-B1 / reference selection     = NOT EXECUTED
```

## 9. Outcome

```text
PROP01_LOCKED_DEMO_PROPOSAL_GATE_PASS
NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

No candidate has zero/empty proposals and every candidate has `eligible_largest_count > 0`, so outcome A applies. The
next gate is not executed here.
