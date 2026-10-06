# Runtime source mapping（Task 8B）

交付 runtime 中的每一个数值步骤都来自研究中心冻结实现，移植方式为**逐字复制 + 仅改写 import 前缀**
（`buildreasonseg_mvp.` → `buildreasonseg.runtime._frozen.mvp.`），见
`buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md`。

| 交付模块 | 研究来源 | 作用 |
|---|---|---|
| `buildreasonseg/runtime/_frozen/mvp/task6n_relation_decoder.py` | `buildreasonseg_mvp/task6n_relation_decoder.py` | 冻结 SAM2.1 Hiera Base+ encoder 装载 |
| `buildreasonseg/runtime/_frozen/mvp/sam2_bridge.py` | `buildreasonseg_mvp/sam2_bridge.py` | SAM2 组网与 encode |
| `buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py` | 同名 | GeometricRelationField v0.2（P_dir） |
| `buildreasonseg/runtime/_frozen/mvp/nearest_boundary_field.py` | 同名 | NearestBoundaryField v0.1（P_near） |
| `buildreasonseg/runtime/_frozen/mvp/task6z_field_composition.py` | 同名 | `record_fields` 关系场组合 |
| `buildreasonseg/runtime/_frozen/mvp/task7d_global_competition_decoder.py` | 同名 | D-B1 `GlobalCompetitionDecoder`（W/A/q/C/logits） |
| `buildreasonseg/runtime/_frozen/mvp/task7d_data.py` | 同名 | 训练期 batch 组装（等价性对照用） |
| `buildreasonseg/runtime/_frozen/mvp/program_parser.py` + `qwen_seg.py` | 同名 | Qwen + LoRA + ProgramHead runtime |
| `buildreasonseg/runtime/_frozen/mvp/task6s_directional_pipeline.py` | 同名 | `parse_instruction`（top-1 softmax 置信度） |
| `buildreasonseg/runtime/_frozen/mvp/task6u_common.py` | `scripts/task6u_common.py` | 冻结 U-C1 配置（imgsz 640 / conf 0.05 / max_det 300） |
| `buildreasonseg/runtime/frozen_paths.py` | —（新增） | 把上述模块中的研究路径常量重定向到交付模型包 |

RC1 自己新增的工程层（不改变核心数学）：`runtime/imageio.py`（影像契约）、`runtime/detector.py`
（tiling + merge + eligibility）、`runtime/context.py`（512 reasoning context）、`runtime/core.py`
（核心链编排）、`runtime/program_head.py`（真实 Qwen runtime + suggestion 生成）、`runtime/outputs.py`
（mask/overlay/diagnostics）、`runtime/manifest.py`（Qwen 资产 manifest）、`runtime/pipeline.py`（端到端流程）、
`predict.py`（真实 CLI）。

## SUCCESS 语义职责映射

`runtime/pipeline.py` 负责 machine-readable SUCCESS semantics；`predict.py` 负责 user-facing CLI presentation。

冻结的机器字段为：

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

兼容性保持不变：`PipelineResult.ok` remains compatible with status == SUCCESS。

这里的 `SUCCESS` 仅表示当前结构性 runtime checks 已通过；semantic target correctness is not established。
