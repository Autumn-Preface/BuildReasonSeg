# DEMO-R1 独立工程复查

复查者：独立 reviewer agent；日期：2026-10-23。基于提交 `4f966cba71e019e6e73ab037eee470239d18a9d5` 及本轮未提交增量。只读检查源码、diff、日志与已保存真实记录；仅本报告获准写入。没有启动模型、推理或训练，没有修改其他文件。本结论不代替 Supervisor 验收。

## 问题与修复复核

- P2 已关闭：`_ui/backend.py` 的结果摘要版本原为 V2，现为 V3，并有版本断言。实际运行摘要也记录 V3。
- P2 已关闭：启动环境检查会禁用输入框，使原来的 `--prompt` 直接插入失效。现由 `prefill_prompt()` 临时切换状态、预填后恢复；新增测试覆盖 disabled 状态、文字内容及未启动 worker。`--image` 只读取预览、记录路径；两参数均不触发推理或替代确认。
- 原布局风险已有工程缓解：结果页侧栏改为独立滚动区域。中文阶段标题、用途说明仅影响展示，不改阶段数据、状态或模型配置。另已抽看 `simulated_failure_1180x760.png`，最小窗口下可见失败状态与明确模拟标签；不同 DPI 的全面兼容性未据此建立。

## 本轮核对结论

独立构建器沿用原 V2 的 accepted blobs、依赖闭包和 observation patch，核对历史 Git 源码及 RC1 选中资产身份，并保留禁止覆盖、链接拒绝和源资产前后核验；没有修改或绕过原构建器的 preservation 函数。新科学公式、阈值、权重、排名或模型选择变更未见。

新运行/成功选图会清除最终 Mask；失败不启用 Mask，结果放大要求当前结果成功。历史回放先验证 run、事件序列和预览哈希，再发 UI 事件，不重新推理。结果页与过程页的真实记录均保留结构成功和语义未验证的区别。

`scripts/demo_r1_ui_evidence.py` 在 runtime 目录之外，不由构建器打包。replay 分支只读取已保存 trace；failure 分支只调用固定 fixture 和 TraceSession，写入 Git 忽略的 `.tmp/demo_r1_fake_evidence`。模拟窗口标题、失败 reason 明示模拟，未发现加载权重、启动推理、自动确认或修改真实记录的代码。失败展示只能作为模拟 UI 证据，不能写成真实模型失败。

已读 `tests_57_final.txt`：57 passed；本 reviewer 未重跑测试。已核对真实 `20261023_230642_1034` 摘要与事件：qwen 初始 program 缺 nearest，建议通过 validator，记录 `user_confirmation=Y`，最终 below-to-nearest；24 条事件包含九阶段完成，最终 SUCCESS，仍为 `RUNTIME_STRUCTURAL_ONLY / NOT_EVALUATED`。确认身份由主执行者的人机交互证据支持，本 reviewer 只核对保存记录。另抽看结果页和最终结构检查截图。

原测试曾发生 Tcl 初始化失败，显式现存 Tcl 路径重跑通过；不能以 import tkinter 或单次测试通过证明所有启动环境兼容。主执行者已补标准 BAT 实开证据，reviewer 抽看的 `launcher_ready.png` 可见环境检查通过，应与原失败日志共同报告。真实 RC1 与 V3 成对模型等价实验为 **NOT_PERFORMED**；现有 AST、固定输入测试及源码身份检查不应被描述为全面真实模型等价证明。

截至本次增量复核，没有发现剩余确定的交付阻断缺陷。截图归档、最终保护核对、提交推送及 Supervisor 审核由主执行者继续完成；本报告不是最终验收。
