# BuildReasonSeg Advisor — RC1（Task 8B：真实 predict / 本地 CMD Demo）

BuildReasonSeg Advisor 是一个**本地可迁移**的建筑目标分割工程：用户用自然语言描述“最大建筑的某个方向上的
最近建筑”，工程在 **RGB 光学遥感影像**中分割出对应建筑，并输出 mask、可视化 overlay 与完整 diagnostics。

> **当前状态：Task 8B 已实现真实 `predict.py` 推理链**（detector → SAM2 → relation fields → D-B1），
> `train.py` / `prepare_dataset.py` / `evaluate.py` 仍为 Task 8A 冻结的**接口与占位实现**（未实现执行）。

---

## Quick Start

```bat
python check_setup.py
python predict.py --image inference/input/example.tif --prompt "先确定面积最大的建筑，再从它右侧的建筑中挑选并分割距离最近的那一个"
```

先看候选建筑编号（用于人工指定参考）：

```bat
python predict.py --image inference/input/example.tif --inspect-proposals
python predict.py --image inference/input/example.tif --prompt "..." --reference-id 12
```

批量（只扫描目录第一层）：

```bat
python predict.py --input-dir inference/input --prompt "分割最大建筑物右侧最近的建筑物"
```

模型包完整性校验：`python check_setup.py` → 全通过时输出 `BuildReasonSeg environment: READY`。

## 输入

- **RGB 光学遥感影像**（软件接受的输入模态；已验证研究/评测域见下文“已验证数据域与 Demo 边界”）；支持 `.png / .jpg / .jpeg / .tif / .tiff`；
- **uint8** 与 **uint16**（uint16 使用固定线性缩放 0→0、65535→255，**不做** histogram/percentile 拉伸）；
- 4 通道 RGBA 会**明确去掉 alpha**；灰度、2 通道、>4 通道、SAR、原始多光谱会被拒绝（E203）；
- **任意尺寸在软件层面支持**：≤512 反射 padding 到 512；>512 使用 512 px 滑窗（overlap 128、stride 384）
  做 tiled detection，再建立**一个**确定性的 512×512 reasoning context；
- **任意尺寸可读 ≠ 跨尺度泛化已验证**（跨城市/跨传感器/跨尺度泛化未建立）；
- GeoTIFF 输出为 PNG，**不保留 georeference**。

### 已验证数据域与 Demo 边界

- **软件接受的输入模态**：PNG / JPEG / TIFF，支持大图分块；这只是格式/尺寸支持。
- **已验证的研究/评测域**：WHU East Asia（Satellite dataset II (East Asia)）之上的 BuildSpatialReason v0.2 与
  WHU-EA-NativeVector v1.0，切分视图 `scene_disjoint_v1`，tile 相对空间关系推理。
- `scene_disjoint_v1` 的场景分离**不**等同于跨城市泛化证据。
- 任意航空影像的鲁棒性**未**得到保证；SAR / 红外 / 原始多光谱不在支持范围内。
- **A2 是已记录的持续未检出（persistent non-detection）压力样例，其来源/域为 NOT ESTABLISHED，且不得声称其“域外”。**
- 历史六案例诊断套件（A1/A2/A3/A4/B1/B2）保留原样，**不**改写为 6/6 成功套件。
- 四个锁定的 v0.2 TEST 用例仅为**定性候选**，**尚未**作为新的验收 Demo 运行。
- 定性候选的选择为确定性、仅基于元数据；其复用已披露，且不改变任何已报告的 Task 7J 指标/模型/阈值/种子/架构。

## 当前已验证语义

RC1 正式开放的空间语义**只有四类**（用“语义类型 / 命令示例”表述，不称“标准命令”）：

| 语义类型 | 命令示例 |
|---|---|
| 最大建筑 → 左侧 → 最近建筑 | 请分割最大建筑左侧最近的建筑 |
| 最大建筑 → 右侧 → 最近建筑 | 请分割最大建筑右侧最近的建筑 |
| 最大建筑 → 上方 → 最近建筑 | 请分割最大建筑上方最近的建筑 |
| 最大建筑 → 下方 → 最近建筑 | 请分割最大建筑下方最近的建筑 |

`smallest → direction → nearest` 等其它组合**不属于** RC1 正式能力（即使研究代码中存在部分 primitive）。

## Qwen（语言链）

- **所有正常自然语言都必须经过 Qwen**：`Qwen3-VL-2B-Instruct + Task 7C ProgramHead → parsed program →
  Hard Validator`；不存在“命中标准字符串就绕过 Qwen”的捷径；
- CMD 会**显示 parsed program**（含置信度）；`--confirm-command` 可人工确认后执行；
- 解析为**未支持** program 时：显示解析结果 → 由本地 Qwen 生成最接近的可支持替代指令 → 再次经 Validator →
  询问 `是否使用建议指令继续？ [Y/N]`，只有 Y 才继续；
- **当前语言鲁棒性仍有限**：受控语言/canonical program 接口在数据集自身指令上表现完好（Task 7C 全量 val
  accuracy 1.0 / 18,222 条），但**未见过改写会退化**（Task 7C: fixed24 5/24、compact 0/8、stress 0.6667）。
  工程**不**声称已解决 unrestricted natural language；
- 仅当 Qwen 资产缺失 / 加载失败 / 运行失败时，才允许进入有限兼容模式并显示
  `MODE: FALLBACK COMMAND PARSER`（`result.json` 记录 `"language_mode": "fallback"`）。

### 当第一次解析落到未开放语义时（fallback UX）

1. 所有正常自然语言输入**都经过** Qwen / ProgramHead，不经过关键词或正则捷径；
2. 当前 ProgramHead 对**训练分布外措辞仍然敏感**（同一意图换个说法可能被解析为 L1/L2 program）；
3. 若第一次解析落到未开放语义，系统会显示
   `当前解析结果不属于 RC1 已开放的四类空间推理语义。`，并尝试由**本地 Qwen** 给出一个**可支持替代建议**
   （结构化契约：`program` 只能是四类之一，或返回 `NO_SAFE_SUGGESTION`）；
4. 替代建议**不会自动执行**：必须由用户 `是否使用建议指令继续？ [Y/N]` 确认，且建议的 program 必须再通过
   Hard Validator；
5. 若建议失败（`NO_SAFE_SUGGESTION` 或未通过 Validator），用户可参照**命令示例**重新表述；系统不会声称已支持
   未开放的语义（例如 second largest / second nearest / 颜色属性 / between）；
6. 这**不是** unrestricted natural-language understanding。

## 运行模式

| 模式 | 命令 | 说明 |
|---|---|---|
| Automatic | `--image … --prompt …` | 自动选择 eligible 建筑中**预测面积最大**者为参考（U-C1 冻结优先级：面积 → 置信度 → 全局 id） |
| Reference-Assisted Diagnostic | `--image … --prompt … --reference-id N` | 人工指定参考（`result.json` 标记 `"reference_mode": "assisted"`）；**仅是诊断能力，不代表 automatic 端到端性能** |
| Inspect Proposals | `--image … --inspect-proposals` | 只跑 image loader + detector + merge，不加载/不运行 Qwen，输出编号候选图 |

## 输出

```text
inference/output/masks/<stem>_mask.png          # 原图尺寸，uint8，背景 0 / 目标 255
inference/output/overlays/<stem>_overlay.png    # 原图尺寸，红色 (255,0,0)，alpha 默认 0.45
inference/output/diagnostics/<stem>/            # prompt.txt, parsed_program.json, global_proposals.png,
                                                # proposals.json, selected_reference.png,
                                                # reasoning_context.png, reference_context_mask.png,
                                                # direction_field.png, nearest_field.png,
                                                # relation_weight.png, prototype_similarity.png,
                                                # maps.npz (P_dir/P_near/W/A/C/logits), result.json
```

已存在的输出**不会被静默覆盖**：自动使用 `_001`、`_002`… 后缀，mask / overlay / diagnostics 共用同一后缀。
`--no-save-diagnostics` 关闭诊断产物；`--alpha` 范围 `(0, 1]`。诊断图 `global_proposals.png` /
`selected_reference.png` 最长边限制为 2048（仅显示，不影响算法）。

## 模型包

```text
model/buildreasonseg_advisor/   decoder.pt (D-B1, seed 20261003) · detector.pt (YOLO26m-seg, U-C1)
                                model.yaml · metadata.json · metrics.json
model/components/sam2/          SAM2.1 Hiera Base+（冻结）
model/components/program_head/  Qwen3-VL-2B-Instruct + Task 7C ProgramHead（冻结）
                                qwen_asset_manifest.json（11 文件逐个 SHA256）
```

所有路径相对交付根解析；整个 `BuildReasonSeg_Advisor_RC1` 可移动到任意位置后继续运行
（已通过“复制到新路径后 `check_setup.py` + 真实 `predict.py`”的 portability smoke）。

## 限制（必须在任何对外说明中保留）

- **Reference selection 仍是 practical bottleneck**（Task 7F/7H）：冻结 U-C1 确定性 largest selector 留在链中，
  因为没有学习式替代通过采纳门禁（Task 7G）；Automatic 失败而 Assisted 成功通常意味着参考选择问题；
- **unrestricted natural language 未解决**；仅正式开放 `largest → direction → nearest` 四类；
- **固定 512 reasoning context**：参考建筑 bbox 超过 `0.80 × 512` 时报
  `E404 reference_too_large_for_rc1_context`；该方向上所有候选都落在 context 之外时报
  `E404 directional_candidates_outside_rc1_context`；**direction 过远即失败，不做自适应重裁**；
- **decoder 层面优势 ≠ 端到端优势**：D-B1 在 oracle reference 下明显优于 Z-B3（Task 7I val +0.0510、
  Task 7J final test +0.0548，3/3 seeds），但在冻结的 predicted reference 下增益很小（+0.0065 / +0.0029）；
- Task 7J 是 measurement stage，正确表述为 **final frozen-architecture test evaluation**（**不是** untouched test），
  该 split 状态为 `FINAL_TEST_CONSUMED`；
- 不声称 unrestricted NL、不声称 unseen-city/cross-city/cross-sensor 泛化、不声称“first”；
- GeoTIFF 输出不保留 georeference；不支持 SAR / 红外 / 原始多光谱。

## 其它入口（Task 8A 契约，尚未实现执行）

```bat
python prepare_dataset.py --dataset datasets/MyDataset --format coco
python train.py --dataset datasets/MyDataset --name my_city_model
python evaluate.py --model buildreasonseg_advisor --dataset datasets/MyDataset --split val
```

这三者已完成参数解析与路径校验；真正执行会在后续任务实现（当前返回
`NOT_IMPLEMENTED_IN_TASK_8A` 且退出码非 0）。

## 文档

`docs/model_card.md`（模型/指标/限制）、`docs/command_grammar.md`（命令语法与错误码）、
`docs/dataset_format.md`（数据集约定）、`RC1_TASK8B_REPORT.md`（本任务构建与验证报告）。

## SUCCESS 状态语义

为保持既有接口兼容，正常推理仍保留 `status="SUCCESS"`。这里的 `SUCCESS` 只表示 RC1 runtime 已完成并通过当前的结构性检查，不表示目标建筑已经在语义上被正确识别或分割。

机器可读字段固定为：

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

单图 CLI 对应显示：

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
```

因此，semantic target correctness is not established。

这属于 `status-contract` 的语义澄清，不是模型性能修复，也不是 `mask-quality` 修复。
