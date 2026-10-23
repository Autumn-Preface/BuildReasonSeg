# DEMO-R1 科学等价性证据与界限

## 源码和资产身份

原 RC1 科学实现的锚点为历史 V1 接受提交 `93f66bc9a4fcb7b7ce5699f1c1afe412d18c43d4`。V2 观测实现取自 `e06777b0a0844c5c29325701eb2e819bd193ff04`；已核对仓库 V2 文件与历史 Git 内容（仅允许 checkout CRLF 差别），原构建器未改。

V3 选择 71 个引擎源码/资产路径，身份对照原 V2 验收记录中的 RC1 清单。未修改模型字节。三处引擎源码仅加入原 V2 的只读观测 Hook，另增加 observation 传输模块；最终派生文件与原 V2 patch 输出逐字节一致。具体路径、前后 SHA 和审计记录见 [preflight.json](preflight.json) 中 observation_source_differences。V3 UI 变化不写回模型数组。

V3 backend 仅将用户摘要 demo_version 改为 V3；语言规则、确认路径、参数、原有候选合并、reference 选择、512×512 context、模型解码、guard 和输出契约未改。UI 的中文标题、布局、预填参数和 readonly 预览不改变这些决策。

## 已执行的工程等价性检查

`tests/test_demo_r1.py` 继承历史 V2 固定替身/事件测试并切换到 V3，另加预填控件断言；通过 57 项。autouse fixture 阻止真实 torch.load、nn.Module 构造及模型 runtime load/parse/detect/forward；测试不加载模型权重，不做真实训练或 Task7J 评价。

| 性质 | 工程证据 |
| --- | --- |
| 科学源码语义 | 在启用和禁用 observer 两条路径移除纯观测节点后 AST 一致；故意改变公式/视觉输入会被负例拒绝。 |
| 最终输出 | 原源码、派生关闭、派生开启三路固定输入比较 logits、概率、Mask、Overlay 等输出及错误状态；不是依赖截图肉眼相似。 |
| 中间计算 | P_dir/P_near/W/A/q/C 对应同一次真实 forward 的已存在状态；替身故意区分诊断与正式 forward，测试确认采集后者。 |
| 调用与随机状态 | 固定替身的 SAM、projection、competition、forward 调用计数不变，Python/NumPy/Torch RNG 状态不变。原算法已有的诊断计算次数保持原样，不额外推理生成美化图。 |
| 选择与失败 | 固定 reference ID/bbox/Mask 比较；覆盖检测失败、参考失败、context 失败、编码/解码异常及候选后 guard 失败。 |
| 渲染/回放隔离 | immutable snapshot、颜色归一化不回写、跨 run/篡改拒绝、主线程 UI 更新、记录回放无模型调用；失败不能伪装成功。 |

测试通过是工程证据，不等于科学正确性，也不证明所有真实输入上的行为。

## 真实运行证据

用户确认的单次 V3 真实运行 `20261023_230642_1034` 完成九阶段，并保存 Mask/Overlay。实际事件的 stage 6/7 来源标记为实际 D-B1 forward，stage 8 标记候选，stage 9 单独标记结构成功。运行未改变算法以获取成功。

**真实 RC1（观测前）对 V3（观测后）的成对模型推理比较：NOT_PERFORMED。**
因此真实 Mask 数组/Overlay 的跨版本相等、真实模型 forward 次数及真实 GPU 随机行为没有在本任务中获得独立成对实测证明。实际单次运行的 SUCCESS 只能说明本机工程流程可用，不能升级为全面等价、目标语义正确或精度提升结论。

## 资产与治理

[最终检查](final_verification.json) 对当前 RC1 构建前后 2184 文件清单（大小、SHA、mtime）一致，归档 ZIP 未变，V3 所有清单文件 SHA 正确且 UI 与仓库一致；旧源码、正式评价和 Damage-P0 Git 路径无修改。无训练、无阈值/排名/公式调节、无新模型分支。独立 reviewer 已复核差异与证据；最终接收或进一步真实配对验证由 Supervisor 决定。
