# Command Grammar — BuildReasonSeg Advisor RC1

## 1. 语言链（冻结）

```text
用户自然语言指令
      ↓
Qwen3-VL-2B-Instruct + Task 7C ProgramHead      ← 正常路径，必须经过
      ↓
parsed program（canonical program id）
      ↓
Hard Validator
      ↓
supported → 执行 | unsupported → 建议流程（Y/N 确认）
```

- 正常模式下**禁止**绕过 Qwen（不存在“标准命令/正则优先”的捷径）。
- 仅当 Qwen 资产缺失、加载失败或运行时失败时，才允许确定性 fallback，并打印
  `MODE: FALLBACK COMMAND PARSER` 且请求用户确认；fallback **永远不会**成为正常路径。

## 2. 正式支持 program（恰好四个）

| program | 中文示例 | 含义 |
|---|---|---|
| `largest_to_left_of_to_nearest` | 最大建筑左侧最近的建筑 | 以最大建筑为参考，左侧最近的建筑 |
| `largest_to_right_of_to_nearest` | 最大建筑右侧最近的建筑 | 以最大建筑为参考，右侧最近的建筑 |
| `largest_to_above_to_nearest` | 最大建筑上方最近的建筑 | 以最大建筑为参考，上方最近的建筑 |
| `largest_to_below_to_nearest` | 最大建筑下方最近的建筑 | 以最大建筑为参考，下方最近的建筑 |

以下**不属于**正式能力（即使研究代码中存在部分 primitive，也不得对外宣传）：

- `smallest_to_<direction>_to_nearest`；
- L1/L2 程序（如单纯的 `largest`、`largest_to_left_of`）；
- 未在支持列表中的任何组合。

## 3. 不受支持指令的处理（建议流程）

```text
1. 显示解析结果（program / 来源 / 置信度）
2. 基于固定支持算子列表提出最接近的可支持替代命令
3. 替代 program 必须再次通过 Validator
4. 询问：是否使用建议指令继续？ [Y/N]
5. Y → 执行；N → 终止当前样本，并给出命令示例
```

工程**不会**自动替用户改写命令。

## 4. Reference（参考建筑）选择

- 默认策略：**U-C1 确定性 largest selector**
  （eligibility：非空、不触边、bbox extent ratio ≤ 0.20；selector：最大预测面积 → 置信度更高 → 索引更小）；
- **Reference-Assisted Mode**：`--reference-id INT` 手动指定参考建筑（配合 `--inspect-proposals` 查看编号）；
- 学习式 selector **未被采纳**（Task 7G 未通过其内部门禁），因此不在链中。

## 5. Fallback（Qwen 不可用）

允许的三种原因：`qwen_asset_missing`、`qwen_load_failed`、`qwen_runtime_failed`。
出现在其它原因（例如“想更快”）时**不允许**启用 fallback。

## 6. Error code registry

| code | name | 说明 |
|---|---|---|
| E101 | INVALID_COMMAND | 指令格式无效 |
| E102 | UNSUPPORTED_PROGRAM | 解析为未支持 program |
| E201 / E202 / E203 | IMAGE_NOT_FOUND / IMAGE_UNREADABLE / UNSUPPORTED_IMAGE_TYPE | 输入影像问题 |
| E301 / E302 / E303 / E304 | MODEL_NOT_FOUND / MODEL_INCOMPLETE / MODEL_HASH_MISMATCH / MODEL_INCOMPATIBLE | 模型包问题 |
| E401 | NO_BUILDING_DETECTED | 未检测到可用建筑 |
| E402 | NO_ELIGIBLE_REFERENCE | 无合格参考建筑 |
| E403 | NO_DIRECTIONAL_TARGET | 该方向无目标 |
| E404 | REASONING_FAILED | 关系推理失败 |
| E501 / E502 | CUDA_OUT_OF_MEMORY / INFERENCE_RUNTIME_ERROR | 运行时错误 |
| E900 | NOT_IMPLEMENTED_IN_TASK_8A | 该功能属于后续构建阶段 |
| E901 | USER_ABORTED | 用户中止（Y/N 选择 N） |

用户只会看到简短中文错误 + 必要修复建议；完整 traceback 写入 `logs/`。

## 7. Task 8B runtime（真实执行）

| 阶段 | 实现要点 |
|---|---|
| 语言 | Qwen3-VL-2B-Instruct + Task 7C ProgramHead（verbatim 移植）；输出 parsed program + top-1 置信度 |
| Validator | 只放行四个 `largest_to_*_to_nearest`；未支持时走 Qwen suggestion + Y/N 确认 |
| Detection | ≤512 → reflection pad 到 512；>512 → 512 px tiles，overlap 128，stride 384，边缘 clamp |
| Merge | global mask IoU ≥ 0.50 视为 duplicate；每个 group 只保留一个（border → clearance → area → confidence → tile id → index）；**no union** |
| Reference | Automatic = 最大 eligible 面积（tie: 置信度 → 全局 id）；Assisted = `--reference-id` |
| Context | 唯一 512×512：right_of (0.30,0.50) / left_of (0.70,0.50) / above (0.50,0.70) / below (0.50,0.30)，`int(x+0.5)` |
| Core | SAM2 → P_dir/P_near → W/A/q/C → D-B1 → `logits > 0.0` |
| Validity | mask 非空、落在非 padding 区域、target centroid 满足方向硬约束；否则 E404（不产出假成功 mask） |
| 输出 | mask（0/255、原尺寸）、overlay（红、alpha 默认 0.45）、diagnostics（含 maps.npz 的 P_dir/P_near/W/A/C） |

新增 reason（仍用统一错误码）：`reference_too_large_for_rc1_context`、
`directional_candidates_outside_rc1_context`、`empty_target_mask`、`mask_only_in_padding`、
`direction_constraint_violated`。

Batch 退出码：全部成功 0；部分失败 2；全部失败 3（定义在 `errors.py`）。
