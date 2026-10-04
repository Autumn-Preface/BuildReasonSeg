# Model Card — BuildReasonSeg Advisor RC1 default package

## 1. 模型定位

- **Model**：BuildReasonSeg **D-B1 RC1 default package**（`buildreasonseg_advisor`）；
- **decoder seed**：**20261003**；
- **选择依据**：Task 7I **validation**（936 条 oracle-reference v0.2 L3 val，按 val mean mIoU 选择；三个 D-B1
  seed 中 0.387688 最高）。**不是**由 Task 7J final test 选择，也不做 seed ensemble；
- **detector**：YOLO26m-seg（冻结 U-C1 proposal model，imgsz 640 / conf 0.05 / max_det 300 / 默认 NMS /
  无 TTA / 无 tiling）；
- **visual encoder**：SAM2.1 Hiera Base+（冻结，`V ∈ R^(256×64×64)`）；
- **language**：Qwen3-VL-2B-Instruct + Task 7C controlled-language ProgramHead（20 个 canonical program）。

## 2. 数据

- **BuildSpatialReason v0.2** over **WHU-EA-NativeVector v1.0**（`scene_disjoint_v1`）；
- 输入域：**RGB 光学遥感影像**（不支持 SAR / 红外 / 原始多光谱）；
- 正式语义：**largest → direction → nearest**（四个 L3 program）。

- 已验证域：WHU East Asia（Satellite dataset II (East Asia)）上的 native-vector 标注
  （WHU-EA-NativeVector v1.0），切分视图 `scene_disjoint_v1`。
- 推理设定：tile 相对（tile-relative）空间关系推理。
- 正式 L3 程序：`largest_to_left_of_to_nearest`、`largest_to_right_of_to_nearest`、
  `largest_to_above_to_nearest`、`largest_to_below_to_nearest`。

## 3. 指标（均为已完成的正式结果，未在 RC1 重新计算）

### 3.1 Task 7I validation（936 条，oracle reference）

| decoder | seeds | mean ± std (ddof=1) |
|---|---|---|
| Z-B3（baseline/ablation） | 0.319367 / 0.339382 / 0.338579 | 0.332443 ± 0.011331 |
| **D-B1（本包默认）** | 0.379934 / 0.382651 / **0.387688** | **0.383424 ± 0.003934** |

matched-seed Δ(D-B1 − Z-B3)：+0.060567 / +0.043269 / +0.049109（**3/3**）。反事实（326 对）：
oracle pair pass Z-B3 0.869121 / D-B1 0.844581，margin +0.316684 / +0.377485。Task 7I 判定：
**`DB1_FORMAL_VAL_NOT_CONFIRMED`**，唯一失败门禁为实践链路增益 +0.006535 < +0.02。

### 3.2 Task 7J final frozen-architecture test（736 条 / 462 tiles，一次性授权）

| 模式 | Z-B3 | D-B1 |
|---|---|---|
| oracle reference mIoU | 0.337488 ± 0.005273 | **0.392304 ± 0.005889**（Δ +0.054816，3/3） |
| oracle pair pass / margin | 0.849148 / +0.318551 | 0.823601 / **+0.373509** |
| predicted reference strict mIoU | 0.207942 ± 0.001604 | 0.210805 ± 0.005562（Δ **+0.002863**） |
| predicted pair pass / margin | 0.413625 / +0.180074 | 0.316302 / +0.180058 |

predicted reference（U-C1 + 确定性 largest）在该 test 上的参考质量：mIoU 0.375155、`REFERENCE_OK 319` /
`SELECTION_WRONG 369` / `NOT_COVERED 46` / 弃权 2、coverage@0.50 0.934783。
oracle→predicted retention（D-B1）≈ **0.5375**。

> **Task 7J 是 measurement stage，没有性能 PASS/FAIL 判定。** 该 test split 已在 Task 6M 被访问过一次，
> 之后架构选择未使用 test 指标，因此正确表述是 **final frozen-architecture test evaluation**，
> **不得**称为 “untouched test”。该 test 状态为 `FINAL_TEST_CONSUMED`。

### 3.3 语言（Task 7C）

canonical full-val accuracy 1.0000 / macro F1 1.0000 / 每类召回 1.0000 / Z-MiniVal240 240/240 /
Z-Paired 40/40；但 fixed24 **5/24**、compact **0/8**、stress accuracy **0.6667**。
→ RC1 只提供 **controlled-language / canonical interface**。

## 4. 已知限制（与 `metadata.json` 一致）

1. **Reference selection 是实际链路的主导瓶颈**；冻结 U-C1 确定性 largest selector 仍在链中，因为没有学习式
   替代通过采纳门禁（Task 7G）。
2. **实践链路增益很小**（+0.006535 validation / +0.002863 test），而 oracle 参考下增益显著
   （+0.050982 / +0.054816）——两者必须分开报告，**不得**合并成一个数字或表述为 end-to-end 显著提升。
3. **D-B1 并非普遍提升 counterfactual pair pass rate**：其 margin 更大，但 pass rate 在两种 reference 下
   都低于 Z-B3。
4. **unrestricted natural language 未验证**。
5. **unseen-city generalization 未建立**。
6. `nearest-only` 为实验性/受限，不是正式能力。
7. 不声称 “first”，不声称 novelty。
8. 大图 tile size / overlap / merge threshold / context 参数**尚未**冻结。

## 5. 使用与完整性

- 完整性校验：`python check_setup.py`（逐一核对 decoder / detector / SAM2 / Qwen-ProgramHead 的 SHA256）；
- 模型包契约：`model/buildreasonseg_advisor/model.yaml` + `metadata.json`；
- 任何新训练产出的模型包都必须与 `buildreasonseg_advisor` **同构**。

## 6. RC1 runtime（Task 8B）

RC1 的 `predict.py` 执行**冻结研究实现**（`buildreasonseg/runtime/_frozen/mvp/`，逐字移植、仅改写 import
前缀），因此不存在“近似版”D-B1：

```text
RGB 输入 → 512 px tiled detector（U-C1, imgsz 640 / conf 0.05 / max_det 300 / 默认 NMS / no TTA）
        → global merge（global mask IoU ≥ 0.50，保留一个 proposal，不做 union）
        → Automatic/Assisted Reference（U-C1 eligibility：非空、不触原始图边界、bbox extent ≤ 0.20）
        → 唯一确定性 512×512 reasoning context（方向 anchor；reflection pad；不做自适应重裁）
        → 冻结 SAM2.1 Hiera Base+ dense feature (256×64×64)
        → GeometricRelationField v0.2 (P_dir) + NearestBoundaryField v0.1 (P_near) → W = clamp(P_dir*P_near)
        → D-B1 GlobalCompetitionDecoder("D-B1")（A = 全局竞争注意力、q = target prototype、C = cosine map）
        → bilinear upsample 到 512 → 冻结阈值 `logits > 0.0`
```

**没有任何后处理改变 mask**：无 morphology、无 CRF、无 SAM refinement、无连通域挑选、无 proposal snap、
无“根据结果重跑”。只允许为写文件做 dtype 转换与可视化 normalization。

等价性证据（Task 8B section 18/19）：4 条 validation fixture 上 delivery 与 research 实现的
`P_dir / P_near / W / A / C / logits` **全部 bit-exact（max abs diff = 0.00e+00）**、binary mask 完全一致；
detector 与 research 冻结路径**检测数、置信度、mask 完全一致**（9 vs 9，diff 0.0）。

### Demo policy

- 历史 A1–B2 诊断套件保留（含失败与人工结论），不改写为 6/6 成功。
- 锁定候选来自 BuildSpatialReason v0.2 **TEST**，按确定性、仅元数据规则选出。
- 定性复用已披露：不改变任何 Task 7J 指标、模型、阈值、种子或架构。
- 锁定候选**尚未**被声称成功；未来失败不得在无新 ChatGPT 决定的情况下替换。
- `scene_disjoint_v1` 的场景分离 ≠ 跨城市泛化证据；任意航空/跨传感器鲁棒性 **NOT ESTABLISHED**。
- A2 为持续未检出样例，来源/域 **NOT ESTABLISHED**；**不得**判定为已证实的域外样本。
