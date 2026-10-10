# TASK AUTHORIZATION

**Task ID:** `TASK8F_USER_DEMO_LIVE_PIPELINE_TRACE_V1`

**Mode:** USER-FACING DEMO ENHANCEMENT / OBSERVABILITY ONLY

**Accepted predecessor:** `TASK8E_USER_FACING_DEMO_PACKAGE_V1 = ACCEPT`

**Repository:** `Autumn-Preface/BuildReasonSeg`

**Accepted predecessor branch:** `delivery/task8e-user-demo-v1`

**Accepted predecessor HEAD:** `93f66bc9a4fcb7b7ce5699f1c1afe412d18c43d4`

**Required new branch:** `delivery/task8f-user-demo-live-trace-v1`

**New external destination:**

`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2`

**Next gate:** `CHATGPT_TASK8F_USER_DEMO_LIVE_TRACE_AUDIT`

## 1. 核心目标

在已接受的 BuildReasonSeg Demo V1 基础上，建立一个具有实时推理过程展示能力的用户 Demo V2。

必须采用**两个可切换的平级页面**：

- 结果
- 推理过程（Pipeline Trace）

推荐使用 Tkinter `ttk.Notebook` 或功能等价的页面管理结构。

两个页面必须处于同一主窗口的平级层次。

禁止：

- 把推理过程嵌套在原结果页里面；
- 用独立弹窗代替新页面；
- 把推理过程设计成另一个独立程序；
- 切换页面时重新推理；
- 为了展示中间量而修改算法。

结果页保留 V1 已有交互与功能。

两个页面共享同一个 Worker、同一次 inference、同一份结果状态。

## 2. 冻结科学链路

不得改变：

Qwen / ProgramHead → Validator / suggestion → YOLO26m-seg → automatic Reference → 512 reasoning context → SAM2 → P_dir / P_near → W / A → q / C → D-B1 → Mask → structural guard。

不得修改：

- 模型权重和 checkpoint；
- YOLO 推理参数与 RGB/BGR 契约；
- proposal merge；
- reference eligibility / ranking；
- context 规划；
- 方向和邻近场公式；
- W / A / q / C 的数值计算；
- D-B1 的网络与输出阈值；
- padding、方向和其他结构 guard；
- Qwen 解析、建议与用户确认规则。

`SUCCESS` 必须继续表示 `RUNTIME_STRUCTURAL_ONLY`，不能自动声称语义正确。

## 3. 新增九个可视化阶段

### Stage 1：语言理解

显示用户原始输入、ProgramHead 初始 program、是否受支持、Qwen suggestion、用户确认后的最终 program。

若首次解析缺少 nearest，真实展示该事实。

置信度不得写成语义正确率。

拒绝建议或取消时，后续阶段显示“未执行”。

### Stage 2：YOLO proposals

模型检测并完成 merge 后立即显示：

- 原图上的半透明 proposal masks；
- 各实例的 proposal ID；
- 必要时标示 bbox；
- raw / merged proposal counts；
- 检测耗时。

保留所有合格输出候选，不得为美观隐藏错误候选。

可视化不得修改 proposal mask。

E401 时保留 0 候选的检测结果，并停止后续阶段。

### Stage 3：Automatic Reference

在候选可视化基础上高亮自动选择的 Reference。

展示：

- selected proposal ID；
- mask area；
- confidence；
- bbox；
- automatic mode。

不得暴露可改变算法的 `reference-id` 输入功能。

E402 时明确说明“没有可用 Reference”。

### Stage 4：Reasoning Context

显示：

- 原图中 512×512 context 的实际位置；
- 真正传入核心链的 context RGB 图；
- 参考 Mask；
- 方向；
- padding 状态。

不得通过重新裁剪推测 context，必须使用真实运行计算得到的坐标与数据。

### Stage 5：SAM2 Visual Features

显示视觉编码已完成、原始特征形状、耗时。

可以提供简洁的特征网格示意，但不能把任何虚构热力图冒充 SAM2 输出。

不必显示 256 个 feature channels，不保存原始大特征张量。

### Stage 6：Spatial Reasoning Fields

实际空间场计算完成后展示：

- `P_dir`
- `P_near`
- `W`
- `A`

采用四宫格热力图，尽量叠加 Reference 边界。

所有图标注其真实名称。

显示颜色映射范围与使用的显示归一化方式。

不得修改原始数值；不得把显示归一化后的颜色当作原始概率。

### Stage 7：Prototype & Similarity

显示：

- q 已形成；
- q 的维度；
- C similarity heatmap；
- 对应的 context 区域。

q 不需要显示 128 个维度的数值。

C 只表示视觉相似度，不是目标正确概率。

### Stage 8：D-B1 Segmentation

D-B1 计算完成后立即显示：

- logits / probability heatmap；
- 二值 Mask；
- context 中的预测结果。

注意：若后续 structural guard 返回 E404，这里显示的是**未经最终结构验证的 D-B1 中间输出**。

必须明确标注：

“D-B1 已生成候选 Mask，但尚未通过最终结构检查。”

不得把它保存或称作已接受的最终目标分割结果。

### Stage 9：Final Guard & Result

显示：

- guard 结果；
- SUCCESS / E401 / E402 / E403 / E404；
- 最终输出是否有效；
- Overlay / Mask；
- elapsed time。

不得声称已经通过 GT 或目标实例身份验证。

失败时，未执行的阶段保持“未执行”。

## 4. 真正的实时更新

不是“推理结束后一次性展示”。

要求每完成一个计算阶段，通过 observation-only StageEvent 推送结果：

Scientific runtime → read-only observation → worker queue → Tkinter main thread → Trace UI。

事件可包含：

- run ID；
- stage ID；
- stage state；
- 时间与耗时；
- 快照路径或只读数据副本；
- 简要元数据；
- 错误码。

只能在 Tkinter 主线程更新界面。

严禁模型线程直接操作 Tk 控件。

事件顺序必须符合真实执行顺序，不得为了视觉效果伪造阶段完成事件。

切换结果页与推理过程页，不得新增模型调用。

## 5. 观测接口与科学隔离

当前已验收 RC1 在成功后集中保存多数 diagnostics，失败时可能没有完整中间快照。

因此授权增加有限的**纯观测 Hook**，但只能在 Demo V2 自有的引擎副本中生效。

不得修改：

- canonical `delivery_src/BuildReasonSeg_Advisor_RC1/**`；
- 已存在的 `BuildReasonSeg_Advisor_RC1`；
- 已存在的 `BuildReasonSeg_Demo_V1`；
- Task 8C / 8D / 8E 证据。

优先采用：

- 可选 observer 回调；
- 阶段完成时对已有结果制作只读快照；
- 快照异步转交 UI；
- 观测代码不向科学计算传递任何修改后的输入。

禁止通过第二次检测、第二次 SAM2 或第二次 D-B1 前向调用来补充展示结果。

禁止重复实现科学算法。

确有必要在 V2 的 pipeline/core/Db1Runtime 副本内增加观测边界时：

1. 必须局部且最小化；
2. observer=None 时保持原执行路径；
3. 不改变科学计算函数的数学表达和参数；
4. 不改变模型前向次数；
5. 不改变随机状态与权重；
6. 不改变最终 Mask、Overlay 和 error code；
7. 在报告中逐一列出与冻结源码不同的纯观测代码；
8. 配套 fake/equivalence tests。

任何改变数学结果的需求，立即 STOP。

不得暗中修改冻结算法。

## 6. 失败时保存已经完成的阶段

必须保证：

- E401：保留语言阶段与 0 proposals 的检测记录；
- E402：保留已有 proposals；
- context guard 失败：保留 language、YOLO、Reference 与已有 context 信息；
- E404：若已经完成 D-B1，保留真实 D-B1 中间 Mask、logits/probability 等已生成数据；
- E502：保留崩溃之前实际成功完成的阶段，但不伪造后续阶段。

区分：

- 已完成；
- 进行中；
- 失败；
- 未执行；
- 用户取消。

不得因最终失败清空前面阶段卡片。

Observation failure 也不能被静默记录成科学 inference SUCCESS。

## 7. Trace 存储

为每次用户运行建立独立 Trace 记录。

建议：

`results/<run_slug>/trace/`

包含：

- `stage_events.jsonl`；
- `trace_manifest.json`；
- 需要的预览 PNG；
- 简洁的阶段数据摘要。

与现有 `result_summary.json` 共存。

只保存可视化需要的轻量文件，不复制多 GB 权重或大特征张量。

Trace 文件应关联同一次真实运行的 engine run root。

最终 FAIL 不得伪造 `mask.png` 或 `overlay.png`。

必须防止多次运行互相覆盖、跨 run 混入旧图、同名图碰撞。

保留结果页原有的保存与打开结果文件夹功能。

## 8. UI 与交互要求

采用独立平级页面：

`结果 | 推理过程`

过程页使用可以上下滚动的两列阶段卡片，依真实运行逐项更新。

原有结果页：

- 选择图片；
- 输入指令；
- 四个快捷示例；
- 开始分析；
- 原图与 Overlay；
- 查看 Mask；
- 打开结果文件夹；

继续保留。

运行中允许切换页面，不能重复启动 inference。

一个用户点击对应最多一次实际推理。

新的运行开始时创建新的 trace session；旧结果不应混入新结果。

各阶段状态、图片和数据应来自同一 run ID。

不得在主页面显示大量开发日志、traceback、隐藏权重路径。

必要时可以增加只读图像放大/查看功能，但不得影响模型选择或推理结果。

## 9. 展示图片要求

YOLO 使用：

半透明实例 Mask + proposal 编号。

Reference 使用醒目但不遮盖屋顶的高亮边界。

Context 显示原图位置及真实裁剪区域。

P_dir、P_near、W、A 与 C 使用图例清楚的热力图。

D-B1 显示 probability 与 binary mask。

所有图保持比例与坐标对应关系。

图像 resize、上色、alpha blend 只能用于展示，不能回写模型数组。

对多个图像、图块、大尺寸输入需要限制预览占用，避免 GUI 内存无限增长。

## 10. V2 外部交付

创建：

`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2`

保留 V1 完整不变。

V2 应包含：

- 双击启动器；
- Tkinter 用户界面；
- 结果与推理过程两个平级页面；
- 独立引擎和模型资产；
- `results/`；
- 使用说明；
- 版本说明；
- 内部 runtime cache；
- 完整 package manifest。

不得包含：

- handoff；
- governance；
- evaluation；
- tests；
- Git 元数据；
- 研究任务书；
- 旧模型实验结果。

V2 继续支持已配置电脑上的文件夹整体迁移。

无需实现跨电脑免配置运行。

禁止安装新依赖，除非出现无法继续的阻塞并经过 Supervisor 重新授权。

## 11. 组装脚本与源码管理

基于当前 Task 8E 的用户 Demo 代码继续实现。

新增或扩展：

- 用户 UI 页面；
- StageEvent 与只读 TraceAdapter；
- 预览渲染工具；
- V2 组装脚本；
- 测试；
- 用户使用说明；
- 交付报告与证据。

模型资产不进入 Git。

组装脚本必须能重复构建 V2，使用 staging + manifest/hash 校验。

禁止仅通过手工复制文件完成交付。

V2 中如存在经批准的纯观测引擎源码差异，必须明确标注其 provenance 和 SHA；不得仍声称所有源码均与 RC1 byte-identical。

## 12. 必须通过的测试

至少覆盖：

1. 两个平级页面存在且能切换。
2. 切换页面不触发 inference。
3. Worker 单次运行不被重复启动。
4. StageEvent 顺序正确。
5. 同一 run 的事件不会串入其他 run。
6. YOLO 原始候选与展示候选一致。
7. 候选可视化不改变原 mask。
8. Reference 高亮使用真实 selected proposal。
9. Reference 不能被用户手动改写。
10. Context 预览使用真实 origin/size/padding。
11. P_dir/P_near/W/A/C 的显示不修改原数组。
12. 真实运行阶段才可标记完成。
13. 中间图像不能由额外模型推理产生。
14. D-B1 Mask 与最终通过 guard 的 Mask 在状态上严格区分。
15. E401、E402、E404 失败场景保留先前已完成阶段。
16. 无产物时不伪造截图。
17. 失败时没有虚假最终 Mask/Overlay。
18. 结果页原有功能不退化。
19. 语言 suggestion 仍需要用户确认。
20. 快捷按钮不绕过 Qwen。
21. 没有 reference override、threshold control、inspect mode。
22. 观测 Hook 启用/关闭不改变科学输出与错误语义。
23. 观测不增加 detector/SAM2/D-B1 forward count。
24. V1 与 Advisor RC1 byte-identical。
25. V2 的模型资产来自接受版本并通过 hash 校验。
26. 旧 Task 8C/8D/8E 证据完全不变。
27. V2 根目录无开发 handoff 与历史诊断。
28. 实际 GUI 运行不会因为创建预览图而阻塞主线程。
29. 完成与失败阶段的事件能持久化并重新打开。
30. 显示 SUCCESS 时不会声称目标身份正确。

测试优先使用 fake runtime 和固定模拟阶段输出。

不得使用已经冻结的 Task 8C 四个样例重新推理。

## 13. 真实推理 Smoke

先完成全部 fake tests、静态科学源差异审计和 V2 文件完整性门禁。

允许在确有必要时，选取一个非 Task 8C 的本地输入执行**最多一次**端到端真实 V2 smoke。

不得尝试多图择优。

不得通过重跑或修改阈值救场。

真实 smoke 只验证：

- 阶段事件真实更新；
- 页面切换；
- 真实快照；
- 正确保留失败；
- 无新增模型调用；
- 最终结果与 Trace 属于同一次执行。

不将其计入正式研究评价，也不据此声称模型效果提升。

如果没有合适输入或存在安全阻塞，可推迟给用户自行测试，并在报告中明确披露。

## 14. 交付说明

更新 V2 用户使用说明，说明：

1. 如何双击启动。
2. 如何进入“结果”页。
3. 如何进入“推理过程”页。
4. 如何在运行中切换。
5. 每个阶段代表什么。
6. 如何查看 YOLO proposals。
7. 如何查看自动 Reference。
8. 如何查看空间热力图。
9. 如何查看 D-B1 分割图。
10. E401/E404 时前面阶段为何仍可查看。
11. Trace 展示只是模型中间状态，不是 GT 验证。
12. 如何找到结果文件夹和 Trace 文件。

## 15. Git 与执行安全

首先读取：

1. `AGENTS.md`
2. `governance/PROJECT_STATE.yaml`
3. `governance/DECISIONS.md`
4. `handoff/CURRENT_TASK.md`
5. `handoff/EXECUTOR_STATE.yaml`

然后 fetch 并验证 accepted predecessor HEAD。

任何 branch、HEAD 或未知改动冲突：inspect 后 STOP。

新建：

`delivery/task8f-user-demo-live-trace-v1`

禁止 reset --hard、rebase、amend、stash、clean、force push。

阶段性 commit/push：

1. Trace/UI 设计与 fake tests。
2. 只读 observer 实现与科学等价性验证。
3. V2 打包、完整性、页面验收。
4. 最终证据与交付报告。

不得为了继续执行而隐藏失败测试或破坏已有用户数据。

## 16. 最终报告与验收

建立：

`docs/task8f_user_demo_live_trace_v1.md`

以及：

`evaluation/task8f_user_demo_live_trace_v1.json`

记录：

- V2 路径、包大小和 manifest；
- 页面结构；
- 九阶段事件与图像；
- 只读观测 Hook 的精确源码差异；
- 观测与算法输出等价性测试；
- 失败留存测试；
- 有无真实 smoke；
- 运行时版本和路径迁移；
- 原 V1、Advisor RC1、Task 8C/8D/8E 完整性；
- 已知限制。

最终状态：

`CURRENT_TASK = READY_FOR_SUPERVISOR_AUDIT`

`EXECUTOR_STATE = READY_FOR_SUPERVISOR_AUDIT`

`next_gate = CHATGPT_TASK8F_USER_DEMO_LIVE_TRACE_AUDIT`

Commit + push，随后 STOP。

不得自行开始算法修复或新架构实验。

## 17. 最终交付摘要

Codex 必须说明：

- V2 最终所在路径；
- 启动方式；
- 两页切换方式；
- 是否实现真实逐阶段更新；
- E401/E404 如何呈现；
- Trace 保存位置；
- 新增的只读观测 Hook 是否改变科学计算；
- 所有测试结果；
- 原 V1 是否保留；
- 是否完成真实模型 smoke；
- 有哪些仍需用户手动验收的功能。


## Executor checkpoint

Status: IN_PROGRESS
Current step: OBSERVATION_SCIENTIFIC_EQUIVALENCE_PASS
Next gate: CHATGPT_TASK8F_USER_DEMO_LIVE_TRACE_AUDIT
No model inference. V2 external assembly pending.
