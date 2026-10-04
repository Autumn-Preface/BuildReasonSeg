<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 8B.3-P1D10-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D10-R3` |
| Status | **COMPLETE** (canonical raster resolution + identity check) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `cbde8755f1cae0593274746f128760e13ebe86a1` |
| Model/test execution / functional modification | NONE / NO |
| Original root | `C:\D\resources\Satellite dataset Ⅱ (East Asia)` |
| Canonical tile index | `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\datasets\whu_native_vector\v1.0\tiles\index.jsonl` |
| Tile index identity | MATCH |
| Right raster | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif` \| `1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2` |
| Left raster | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif` \| `eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38` |
| Above raster | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif` \| `0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd` |
| Below raster | `C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif` \| `c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7` |
| All four raster dimensions | ['512x512'] |
| v0.2 identity cross-check | ALL MATCH |
| R2 §20.8 raster gap | RESOLVED |
| Locked candidates | UNCHANGED (4; no substitution) |
| Primary resolution / PROP-01 status | `PROP01_RESOLUTION_DEMO_POLICY` / `PROP01_OPEN_ENGINEERING_DEFECT` (unchanged) |
| Scientific freeze preserved | YES |
| Next gate | `PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION` (recommended, not executed) |
| Report | `docs/task8b3_p1d10_prop01_resolution_decision.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; locked candidates must not be run or replaced. |

Watt was not needed for Task 8B.3-P1D10-R3 (no downloads, no transfers).

No model, test, training or inference execution occurred; the rasters were read and hashed only, and no image was
copied into RC1.
