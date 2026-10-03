# RC1_TASK8B_REPORT — BuildReasonSeg Advisor RC1 真实 `predict.py` 与本地 Demo 推理链

> 任务：**Task 8B**（交付工程 / Demo 推理实现，不是新算法研发、不是新实验）
> 目标目录：`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> 研究基线：Task 7J 后 `6c2b915dbc64acdeb099d194005d74c7180c95fa`（只读）
> **状态：实现与全部工程 gate / smoke 已完成；因 §20 canonical 措辞与冻结 parser 事实冲突，按 §20/§40 STOP，等待 ChatGPT 决策。**

---

## 0. 摘要（先看这里）

| 项目 | 结果 |
|---|---|
| `check_setup.py` | **READY**，exit 0（含 Qwen 11 文件 manifest 全量 hash 校验） |
| core-equivalence gate（§18） | **4/4 PASS，全部 bit-exact**（`P_dir/P_near/W/A/C/logits` max abs diff **0.00e+00**，binary mask identical） |
| detector regression gate（§19） | **PASS**（research 9 vs delivery 9 detections，confidence diff 0.0，resized mask diff 0.0） |
| ProgramHead runtime（§5） | **真实可运行**；冻结 v0.2 数据集指令 **8/8 正确，置信度 1.000**（zh+en） |
| **§20 spec canonical 措辞** | **0/4** → 解析为 L2 类（`largest_to_left_of` 等）→ **STOP 项**（见 §7） |
| Automatic smoke（§32.1） | **4/4 SUCCESS**（四个方向，真实 v0.2 自然语言 prompt） |
| Assisted smoke（§32.2） | **4/4 SUCCESS**（`--inspect-proposals` + `--reference-id`） |
| >512 mosaic smoke（§32.3） | **SUCCESS**（9 tiles、51 merged proposals、mask 映射回 1024×1024） |
| Portability actual-predict（§33） | **PASS**（复制到新路径后 `check_setup` READY + 真实 `predict` SUCCESS，输出写在复制目录内） |
| pytest | **86 passed**（Task 8A 原有 61 条全部保留 + 新增 25 条） |
| 峰值显存 | **5.97 GB**（16 GB GPU，单图/大图均未 OOM） |
| no training / no test access / no GitHub push / no download / no research repo write | **全部确认**（见 §18） |

---

## 1. 修改 / 新增文件列表（仅 delivery 内）

**新增 runtime（工程层）**

```text
buildreasonseg/runtime/__init__.py
buildreasonseg/runtime/frozen_paths.py      # 研究路径常量 → 交付资产重定向
buildreasonseg/runtime/imageio.py           # 影像契约（RGB/RGBA/uint8/uint16、padding、overlay、PNG）
buildreasonseg/runtime/detector.py          # tiled U-C1 detector + global merge + eligibility
buildreasonseg/runtime/context.py           # 唯一确定性 512×512 reasoning context
buildreasonseg/runtime/core.py              # 冻结核心链：SAM2 → fields → D-B1
buildreasonseg/runtime/program_head.py      # 真实 Qwen + ProgramHead runtime（+ suggestion generation）
buildreasonseg/runtime/manifest.py          # Qwen / ProgramHead 资产 manifest 与完整性校验
buildreasonseg/runtime/outputs.py           # mask / overlay / diagnostics / result.json / 命名冲突后缀
buildreasonseg/runtime/pipeline.py          # 端到端编排（automatic / assisted / inspect / batch）
```

**冻结实现移植（逐字复制 + 仅改写 import 前缀）**

```text
buildreasonseg/runtime/_frozen/__init__.py
buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md
buildreasonseg/runtime/_frozen/mvp/**       # 67 个模块（含 task6u_common.py）
```

**CLI / 文档 / 测试 / 证据**

```text
predict.py                                  # 由 Task 8A 占位改为真实实现
check_setup.py                              # §30：新增 Qwen manifest 全量校验
README.md                                   # 更新为实际 Demo 说明（§34）
docs/model_card.md                          # 追加 RC1 runtime 描述（§35）
docs/command_grammar.md                     # 追加 Task 8B runtime 表
docs/runtime_mapping.md                     # 新增：runtime ↔ 研究来源映射
model/components/program_head/qwen_asset_manifest.json   # §30：11 文件 path/bytes/sha256
tests/test_task8b_runtime.py                # 新增 25 条（image loader / tiling / merge / context / output）
tests/test_cli_contract.py, tests/test_setup_checker.py  # 少量更新以匹配真实 runtime
logs/task8b_core_equivalence.json           # §18 gate 证据
logs/task8b_gates.json                      # §19/§20/§30 gate 证据
logs/task8b_program_head_fixture_conflict.json  # §20 冲突证据
logs/task8b_smoke.json                      # §32 smoke 证据
logs/task8b_portability_predict.json        # §33 portability 证据
logs/build_scripts/build8b_*.py             # 本次构建/回归脚本（可复跑）
```

**未改动**：研究仓库（HEAD `6c2b915`，无 tracked/untracked 变化）、Task 8A 的全部资产与契约、
任何冻结 checkpoint。

## 2. `check_setup.py` final result

```text
[OK] Python — 3.11.16   [OK] OS — Windows 10 (AMD64)   [OK] PyTorch — 2.13.0+cu132
[OK] CUDA — 可用   [OK] GPU — NVIDIA GeForce RTX 5080 Laptop GPU
[OK] delivery root 可写   [OK] required directories — 16 项齐备   [OK] writable paths
[OK] model package   [OK] model.yaml — decoder=D-B1 seed=20261003   [OK] metadata.json — 6 个资产记录
[OK] D-B1 decoder — sha256=9187b133… bytes=1117495
[OK] U-C1 detector — sha256=ef852b58… bytes=54480241
[OK] SAM2 assets — sha256=a2345aed…
[OK] Qwen / ProgramHead assets — ProgramHead sha256=c1505736…
[OK] Qwen asset manifest — 11 个文件全部校验通过 (4.34 GB)
[OK] portability — 无 workspace 依赖；项目可整体移动

BuildReasonSeg environment: READY
```

## 3. Qwen full asset manifest integrity（§30）

`model/components/program_head/qwen_asset_manifest.json`：**11 个文件、4,336,731,019 bytes**，每个文件记录
relative path / bytes / SHA256；`check_setup.py` 与 portability smoke 均执行**全量 hash 校验并通过**。
`predict.py` 启动路径对全部条目做存在性 + 大小校验，小文件即时 hash，`model.safetensors`（4.25 GB）使用
metadata + size + 完整性缓存（首次/显式 setup 必须全量 hash）。**不存在“Qwen 缺文件仍 READY”的路径**。

## 4. ProgramHead runtime source mapping（§5）

- Qwen base：`model/components/program_head/Qwen3-VL-2B-Instruct/`（Task 8A 已复制的 10 文件快照）；
- ProgramHead：`program_parser_l3_rehearsal_v1.pt`（step 2，SHA256 `c1505736…d58d9a`）；
- 结构：`Qwen（text-only）+ LoRA(rank16/alpha32/dropout0.05) + LayerNorm→Linear(2048,20)`，与
  `configs/mvp/task6j_program_parser.yaml` 的冻结配置一致；
- runtime 来源：`_frozen/mvp/program_parser.py`（`ProgramParserRuntime`）、`_frozen/mvp/qwen_seg.py`
  （`load_qwen` / `setup_seg_token` / `attach_lora`）、`_frozen/mvp/task6s_directional_pipeline.py`
  （`parse_instruction`：top-1 softmax 置信度）；
- 装载核对：`lora_and_token_state` 中 **394 个 key 全部成功加载（unexpected 0；missing 626 = 冻结 base 权重，
  checkpoint 本就不含，属预期）**；
- 独立验证：研究实现（`build_programseg_mvp.program_parser`）在同一 checkpoint 上对同一 prompt 给出**完全相同**
  的输出 → delivery 与研究 runtime 等价。

## 5. D-B1 / field / SAM2 research source mapping（§17）

| 交付 | 研究来源 | 关键参数 |
|---|---|---|
| SAM2 dense feature | `task6n_relation_decoder.load_frozen_sam2_encoder` + `sam2_bridge` | SAM2.1 Hiera Base+，256×64×64，frozen |
| `P_dir` | `geometric_relation_field_v02.py`（v0.2） | ALPHA 1.2 / TAU 0.04 / S_AXIS 0.02 / S_MARGIN 0.02 |
| `P_near` | `nearest_boundary_field.py`（v0.1） | SIGMA_DIAG 0.05 |
| 组合 | `task6z_field_composition.record_fields` | `W = clamp(P_dir * P_near)` |
| D-B1 | `task7d_global_competition_decoder.GlobalCompetitionDecoder("D-B1")` | `A` 全局竞争注意力、`q` prototype、`C` cosine map、trunk logits |
| 阈值 | 研究 frozen 规则 | `bilinear upsample → logits > 0.0` |
| detector | `scripts/task6u_common.CONFIGS["U-C1"]`（已移植） | imgsz 640 / conf 0.05 / max_det 300 / 默认 NMS / no TTA |

**无任何后处理改变 mask**：无 morphology、CRF、SAM refinement、连通域挑选、proposal snap、按结果重跑。

## 6. core-equivalence gate（§18，4 fixtures，validation only）

fixture 在**模型运行前**按 `sample_id` 排序取每个 direction 第 1 条（禁止 test）：

| sample_id | direction | P_dir | P_near | W | A | C | logits | mask identical |
|---|---|---|---|---|---|---|---|---|
| `buildsr_val_2_1005_3_largest_to_left_of_to_nearest_2a9d76b06182` | left_of | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✅ |
| `buildsr_val_2_1001_3_largest_to_right_of_to_nearest_feffa8e34363` | right_of | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✅ |
| `buildsr_val_2_1002_3_largest_to_above_to_nearest_486ab4f09c4e` | above | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✅ |
| `buildsr_val_2_1005_3_largest_to_below_to_nearest_bdf20085ef01` | below | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✅ |

**all_passed = True**，容差 1e-5，实际达到 **bit-exact**。首次运行曾出现 ~7e-2 logit 差异，定位为**对照脚本**未
使用研究评估路径的 `autocast(bfloat16)`（交付侧使用）；统一精度路径后完全一致 —— 属测试脚本 bug，非实现差异。

## 7. ⚠️ §20 canonical 措辞冲突（**STOP 项**）

`logs/task8b_program_head_fixture_conflict.json` 记录（同一 real runtime、同一 checkpoint）：

| 输入 | 期望 | 实际 parsed | 置信度 | 结论 |
|---|---|---|---|---|
| 分割最大建筑物左侧最近的建筑物 | `largest_to_left_of_to_nearest` | **`largest_to_left_of`** | 1.000 | 冲突 |
| 分割最大建筑物右侧最近的建筑物 | `largest_to_right_of_to_nearest` | **`largest_to_right_of`** | 1.000 | 冲突 |
| 分割最大建筑物上方最近的建筑物 | `largest_to_above_to_nearest` | **`largest_to_above`** | 1.000 | 冲突 |
| 分割最大建筑物下方最近的建筑物 | `largest_to_below_to_nearest` | **`largest_to_below`** | 1.000 | 冲突 |
| **冻结 v0.2 数据集指令（zh，4 条）** | L3 四类 | **全部正确** | 1.000 | ✅ |
| **冻结 v0.2 数据集指令（en，4 条）** | L3 四类 | **全部正确** | 1.000 | ✅ |

事实链（均已交叉验证，非推测）：

1. delivery runtime 与 research runtime 对同一 prompt 输出**完全相同**（§4）→ **不是移植/实现错误**；
2. 同一 checkpoint 在 Task 7C 自己的全量 val 上记录 **accuracy 1.0 / 18,222 条**（`evaluation/task7c_full_val.json`）
   → runtime 与权威 checkpoint **正确对应**；
3. §20 的示例措辞（`最大建筑物左侧最近的建筑`）被该冻结受控语言 parser 稳定映射到**同一 20 类 registry 中的
   L2 类** `largest_to_left_of`（置信度 1.000），即 §20 示例与 parser 的训练/验证措辞分布不一致；
4. 这与 Task 7C 已知的措辞敏感性一致（fixed24 5/24、compact 0/8、stress 0.6667）。

**按 §20（“canonical 四条若不能正确 parse：STOP”）与 §40，DSH 在此 STOP，不自行替换 prompt、不修改
parser、不重新训练、不使用 test 数据，等待 ChatGPT 决策。** 可选处置（供审核选择）：

- **(A)** 接受“数据集冻结指令措辞”作为 canonical fixture（§20 四例改为 warning），Task 8B 其余项已全部通过；
- **(B)** ChatGPT 指定新的 canonical fixture 措辞，DSH 仅重跑 §20 gate 并如实记录；
- **(C)** 认定需要改进受控语言覆盖 → 属于**新研发任务**（需训练/数据变更），不在 Task 8B 范围。

## 8. detector regression（§19）

同一非-test validation 图、同一 imgsz/conf/max_det/NMS：

```text
research detections = 9 ; delivery detections = 9 ; count_match = True
confidence max abs diff = 0.0 ; sorted confidence order match = True
resized mask mean abs diff = 0.0 ; identical masks = True
```

**未调整 conf / NMS 以对齐。**

## 9. ProgramHead 4 canonical + 8 paraphrase（§20，实际解析逐条记录）

8 条 paraphrase 在运行前**预先写死**（每方向 2 条，见 `logs/task8b_gates.json`）：**3/8 正确**
（`largest_to_left_of_to_nearest` / `..._to_above_to_nearest` ×2 正确；其余映射到 L2 或 `largest`）。
canonical 4 条见 §7。**结果如实记录，未为通过而修改 prompt / checkpoint。**

## 10. Automatic 4 validation smoke（§32.1）

使用冻结 v0.2 数据集自身的自然语言指令（zh），原图 512、Automatic Mode、**不使用 GT 纠正**：

| direction | parsed program | 置信度 | 状态 | mask area | raw/merged proposals |
|---|---|---|---|---|---|
| left_of | `largest_to_left_of_to_nearest` | 1.000 | **SUCCESS** | 2410 | 记录于 `logs/task8b_smoke.json` |
| right_of | `largest_to_right_of_to_nearest` | 1.000 | **SUCCESS** | 1988 | 同上 |
| above | `largest_to_above_to_nearest` | 1.000 | **SUCCESS** | 1719 | 同上 |
| below | `largest_to_below_to_nearest` | 1.000 | **SUCCESS** | 1822 | 同上 |

**Automatic 4/4 SUCCESS。** 全部 mask/overlay/diagnostics 与 `result.json` 已落盘。

## 11. Assisted 4 validation smoke（§32.2）

`--inspect-proposals` 在 fixture 0 上得到 **15 个 merged proposals**；随后 `--reference-id` 逐条运行：

| direction | reference_id | 状态 | mask area | `reference_mode` |
|---|---|---|---|---|
| left_of | 6 | **SUCCESS** | 2410 | assisted |
| right_of | 5 | **SUCCESS** | 1988 | assisted |
| above | 6 | **SUCCESS** | 1719 | assisted |
| below | 6 | **SUCCESS** | 1822 | assisted |

`result.json` 正确记录 `"reference_mode": "assisted"`、`"reference_override": true`、`parsed_reference_selector: largest`
（**未伪装成 automatic**）。**Assisted 4/4 SUCCESS。**

## 12. >512 large-image mosaic smoke（§32.3）

**预先确定**的 2×2 mosaic（fixture 顺序 0,1,2,3，1024×1024，仅用于测试大图工程路径）：

```text
tiled detector: 9 tiles (512/128/384) → 51 merged global proposals → reference_id 29
reasoning context: 确定性 512×512 → frozen chain → mask 映射回 1024×1024
status = SUCCESS ; mask_area = 2165 ; mask_full_size_correct = True
```

覆盖：tiled detector、坐标映射、merge、stable global ids、global reference、reasoning context、full-size mask mapping。

## 13. Portability actual-predict smoke（§33）

复制整个 RC1 到 `delivery\_portability_smoke\BuildReasonSeg_Advisor_RC1`，复制一张固定 non-test fixture 到
`inference/input/`，在该目录运行：

```text
check_setup.py → exit 0, READY
predict.py --image inference/input/fixture1_right.png --prompt "…右侧…" → exit 0, status SUCCESS
输出全部写在复制目录内（outputs_inside_copy = True）
```

**runtime_portable = True，sample_result = SUCCESS。** 临时副本已删除；未使用 junction / symlink。

## 14. 输出示例路径

```text
inference/output/masks/fixture0_left_mask.png
inference/output/overlays/fixture0_left_overlay.png
inference/output/diagnostics/fixture0_left/{prompt.txt, parsed_program.json, global_proposals.png,
    proposals.json, selected_reference.png, reasoning_context.png, reference_context_mask.png,
    direction_field.png, nearest_field.png, relation_weight.png, prototype_similarity.png,
    maps.npz (P_dir/P_near/W/A/C/logits/probability), result.json}
```

文件名冲突处理已验证：重复运行产生 `_001`、`_002` 后缀，mask/overlay/diagnostics 后缀**保持配对**，不静默覆盖。

## 15. pytest result

```text
python -m pytest tests -q
86 passed
```

Task 8A 原有 **61 条全部保留**（两处按 Task 8B 真实行为更新：`--inspect-proposals` 不再返回 E900、
setup fixture 需带 Qwen manifest），新增 **25 条**：image loader（uint8/uint16/RGBA/灰度拒绝/2&6 通道拒绝/
扩展名/缺失文件）、tiling（≤512 pad、>512 覆盖、overlap=128、边缘 clamp、pad 映射、坐标映射）、
merge（IoU≥0.5 判定、no union、确定性 winner、stable ids、eligibility、tie-break）、
reasoning context（四 anchor、reflection pad、reference too large、out-of-context guard、方向硬约束）、
output（原尺寸 0/255、overlay、alpha 边界、非覆盖后缀、diagnostics 必填字段）。

## 16. 性能 / 显存

| 阶段 | 观测 |
|---|---|
| 模型装载（Qwen + ProgramHead） | 约 15 s（单进程仅一次） |
| 单图 automatic 全链 | 记录在 `logs/task8b_smoke.json` 的 `timings`（language/detector/merge/sam2/relation_fields/db1/total） |
| 1024×1024 mosaic（9 tiles） | 全链 SUCCESS；tiled detector + 单次 512 core chain |
| **峰值显存** | **5.965 GB**（`torch.cuda.max_memory_allocated`），16 GB GPU 下无 OOM |
| cache | 同一进程内 Qwen / detector / SAM2 / D-B1 各加载一次；未量化、未 TensorRT/ONNX、未改精度路径 |

## 17. 所有 failure 与未完成项

**Failure（如实记录）**

1. §20 spec canonical 措辞 4 条全部解析为 L2 类（§7）——**STOP 项**；
2. §20 的 8 条 paraphrase 仅 3/8 正确——与 Task 7C 已知措辞敏感性一致，**未做任何调参**；
3. 首次 core-equivalence 运行的 logit 差异为对照脚本未对齐 `autocast` 精度路径（已定位并修正，非实现差异）。

**未完成项**

1. `train.py` / `prepare_dataset.py` / `evaluate.py` 真实执行（Task 8A 契约 + 占位，属后续任务）；
2. 三种 dataset adapter 完整实现；
3. 大图 tile/overlap/merge 之外的更细参数（如 context 之外的多 context 策略）——按 §16.5 故意不做；
4. GUI / Web / 阶段报告 / GitHub 发布 / 网盘资产版（按任务边界不做）。

## 18. 确认

| 项目 | 状态 |
|---|---|
| no training | ✅ 未训练/续训任何模型（含 parser、detector、decoder、selector） |
| no final test access | ✅ 未访问、未重跑 final test（仅使用 Task 7I validation 与数据集 val 指令；test 仍 `FINAL_TEST_CONSUMED`） |
| no GitHub push | ✅ 未 commit / push 任何内容 |
| no download | ✅ 未联网下载任何模型或依赖（全程 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`） |
| no research repo write | ✅ 研究仓库 HEAD 仍 `6c2b915`，无 tracked/untracked 变化；仅只读读取 val pack / v0.2 指令 / 掩膜 / 缓存特征 |
| no algorithm change | ✅ 未改 D-B1、relation field、ProgramHead、detector 权重或阈值；未加任何 mask 后处理 |
| no checkpoint change | ✅ decoder / detector / SAM2 / ProgramHead SHA256 与 Task 8A/7J 记录一致 |

---

## 19. 结论与请求

**Task 8B 的工程实现、等价性 gate、detector gate、ProgramHead runtime、automatic/assisted smoke、
大图 smoke、portability smoke 与测试均已完成并通过**；唯一未通过项是 **§20 示例措辞的 canonical fixture**
（其与冻结 parser 的事实冲突见 §7）。

请将本文件 `RC1_TASK8B_REPORT.md` 上传至 ChatGPT，由 ChatGPT 审核实际 predict / Demo runtime，并就 §7 的
canonical fixture 冲突给出处置决定（A/B/C）后，再决定是否进入用户 Demo 测试或后续 Task 8C。

> **Task 8B 已完成后停止。请将 `RC1_TASK8B_REPORT.md` 上传至 ChatGPT，由 ChatGPT 审核实际 predict / Demo runtime 后，再决定是否进入用户 Demo 测试或后续 Task 8C。**
