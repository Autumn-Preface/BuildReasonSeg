# RC1 Supported-Domain Policy (authoritative)

Authoritative supported-domain policy for the BuildReasonSeg Advisor RC1 delivery: positively verified scope, explicit
non-claims, the A2 policy and the two distinct Demo sets. Policy documentation only — no research metric, model,
threshold, seed or architecture changes.

## 5.1 Positively verified scope

```text
Research/evaluation domain:
  BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0
  source: WHU Building Dataset — Satellite dataset II (East Asia)
  split view: scene_disjoint_v1

image modality:
  RGB optical overhead/aerial imagery

instance concept:
  building instances represented by WHU native-vector annotations

reasoning scope:
  tile-relative spatial reasoning

formal RC1 semantics:
  largest -> left_of  -> nearest
  largest -> right_of -> nearest
  largest -> above    -> nearest
  largest -> below    -> nearest
```

Software input-format/size support is **not** demonstrated generalization: PNG/JPEG/TIFF reading and large-image tiling
establish no broad geographic or sensor robustness.

## 5.2 Explicit non-claims

```text
cross-city generalization          = NOT ESTABLISHED
broad geographic generalization    = NOT ESTABLISHED
cross-sensor generalization        = NOT ESTABLISHED
arbitrary aerial-image robustness  = NOT ESTABLISHED
SAR / infrared / raw multispectral = NOT SUPPORTED
unrestricted natural language      = NOT ESTABLISHED
```

Preserved v0.2 limitation: **scene-disjoint separation is scene separation, not proof of cross-city generalization.**

## 5.3 A2 policy

```text
A2 provenance/domain = NOT ESTABLISHED.
A2 is a fixed, documented persistent non-detection / stress case.
It remains part of the historical Task 8B.3 six-case diagnostic evidence.
It was not fixed by P1D1–P1D9.
It must not be described as proven out-of-domain.
It must not be silently removed or rewritten as a success.
```

Bounded evidence (no cause inferred beyond it):

```text
active YOLO26 tiled / full-frame                     -> zero
active YOLO26 at diagnostic conf = 0.001             -> zero
same-lineage epoch-18 YOLO26                         -> zero
independent WHU YOLOv8m baseline                     -> zero
one predeclared validation-moment photometric rescue -> zero
```

## 5.4 Two distinct Demo sets

### Historical diagnostic suite

```text
A1 / A2 / A3 / A4 / B1 / B2
```

Historical fixed defect/audit evidence; failures and manual findings are preserved; the suite is **not** retroactively
relabeled as a 6/6 success suite.

### Locked supported-domain qualitative candidates

Source: `BuildSpatialReason v0.2` **TEST** split. Selection: deterministic, metadata-only, performed after the Task 7J
final metrics had already been consumed; no detector, parser, runtime or manual visual result was consulted; reuse is
disclosed; no research metric changes.

| relation | immutable sample_id |
|---|---|
| largest_to_right_of_to_nearest | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` |
| largest_to_left_of_to_nearest | `buildsr_test_1003_3_largest_to_left_of_nearest_f3fcb14e14c3` |
| largest_to_above_to_nearest | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` |
| largest_to_below_to_nearest | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` |

These are **locked candidates, NOT yet demonstrated successes.** Future failure does not permit replacement without a
new ChatGPT decision explicitly recording that failed lock.

## 5.5 Test-reuse disclosure

The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

定性演示用例是在 Task 7J 最终冻结架构测试指标已被消耗之后，从冻结的 BuildSpatialReason v0.2 test 划分中以确定性方式
选出的；其定性复用不改变、不替换、也不重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。
