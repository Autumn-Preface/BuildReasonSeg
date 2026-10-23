# DEMO-R1 完整工作报告

日期：2026-10-23（Asia/Shanghai）。状态：工程交付完成，等待 ChatGPT Supervisor 审核。研究结论、模型精度与语义正确性未自行验收。

## 交付与入口

独立 V3 已恢复到 `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V3`，可双击 `启动BuildReasonSeg Demo.bat`。源码在 `demo/user_demo_v3/`，恢复构建器为 `scripts/build_demo_r1.py`，使用说明随包提供。

结果页包含输入侧栏、原图/最终 Overlay/最终 Mask 切换、点击结果图查看较大预览、状态摘要、结果目录入口。推理过程页包含九阶段导航、中文阶段标题及用途、状态卡片、真实预览和只读历史回放。侧栏可独立滚动；阶段卡片按可用宽度调整列数。第八阶段保留“候选、未通过最终结构检查”的说明，第九阶段单独报告最终结构状态。

分支：`delivery/demo-r1-restore-ui-20261023`。初始 HEAD 为 `0ef69a2ed6f80251dc14760bab81053a8468954e`，其包含历史 V2 和最新 Damage-P0 治理；恢复 checkpoint `4f966cba71e019e6e73ab037eee470239d18a9d5` 已推送。最终提交应从 Git 实际 HEAD 获取，不在提交内伪造自身 SHA；没有合并 main。

## 只读调查与恢复来源

| 检查项 | 实际发现与处理 |
| --- | --- |
| RC1 | 完整目录存在，所需源码及模型资产可核实；已有兼容环境可用。当前全目录 2184 个文件的 SHA256、大小及修改时间与构建前清单一致。 |
| 原 V1/V2 | 已知部署目录均不存在；清理记录和两份 non-model ZIP 仍在。不是模型权重丢失。 |
| 历史 Git | 已从 GitHub 获取并核对 V2 `e06777b0a0844c5c29325701eb2e819bd193ff04`、V1 接受锚点 `93f66bc9a4fcb7b7ce5699f1c1afe412d18c43d4`；展示程序、observer patch、构建器均可恢复。 |
| 旧构建器依赖 | V2 构建器依赖已删除 V1 的完整清单/身份门禁，不能按原流程直接重建。原 V1/V2 构建器保持不变。 |
| 新恢复方式 | 独立构建器核对历史 Git 源码、原验收记录中的选中 RC1 文件身份，复用原 V2 依赖闭包及 observer AST 审计，再复制到全新 staging；完成全量 SHA 检查后同卷改名到不存在的 V3 目标。未伪造历史 V1 验收。 |
| 空间预算 | 构建前剩余 142191038464 字节，足够约 4.4 GiB 的独立包及余量；没有清理用户资产、下载权重或使用硬链接。 |
| 环境 | 使用已有 Conda 目录中的 Python 3.11.16、PyTorch 2.13.0+cu132、Ultralytics 8.4.164、Transformers 5.17.0、Tk 8.6；GPU 模型路径真实执行成功。未更换或安装依赖。环境不随包附带。 |

详细来源、选中文件、观测差异与构建前完整清单见 [preflight.json](preflight.json)；初次独立构建见 [build.json](build.json) 和 [build.txt](build.txt)。初次构建后只同步本任务 V3 界面文件并更新包清单，其最终身份由 [final_verification.json](final_verification.json) 复核；旧 RC1 未被同步或修改。

两份归档 SHA256：

- V1 non-model ZIP：`6d117df063913488328ae9ffa22b1d2f218cd4e8f0ac76acf1d1269c01cc111e`。
- V2 non-model ZIP：`38b7188237d15b95011cccdc80f99a5d8bce051e8d9b0d11a457b968dd3171b4`。

## 九阶段数据来源与真实性

九阶段沿用 V2 在真实计算位置发出的 observer 事件，并由同一个 TraceSession 保存。没有按最终文件回填虚构的阶段；没有用于美化的第二次模型推理。

| 阶段 | 数据来源与限制 |
| --- | --- |
| 1 语言解析 | ProgramHead 原始解析、支持性、真实 Qwen 建议、validator 与用户确认记录。 |
| 2 建筑候选 | 实际 YOLO 原始及合并候选、编号、Mask 和检测耗时。 |
| 3 参考选择 | 原选择逻辑返回的实例编号、边界、面积和模式。 |
| 4 推理区域 | 真正传入核心链的 RGB、参考 Mask、原图裁剪起点及 padding。 |
| 5 视觉编码 | SAM2 实际特征形状和编码耗时；不虚构特征热力图，不保存大型原始特征张量。 |
| 6 空间关系 | 实际 D-B1 forward 的 P_dir、P_near、W、A；保存形状、范围和数组摘要，预览含颜色范围/归一化说明。 |
| 7 原型与相似度 | 同一次 forward 的 q 形状及 C 图；C 是视觉余弦相似度，不是目标正确概率。 |
| 8 候选分割 | 同次解码 logits、sigmoid 图和二值候选，明确未执行最终 guard。 |
| 9 结构检查 | 原始 guard 状态、错误码及最终合法输出；SUCCESS 的范围仍为 RUNTIME_STRUCTURAL_ONLY。 |

运行、完成、失败、未执行及用户取消独立呈现；观测失败不能显示为无条件成功。渲染在后台，Tk 更新在主线程；切页不启动 worker 或模型。历史回放先核对 run_id、事件顺序和预览 SHA，再发布界面事件，不调用推理。

## 测试结果

| 项目 | 结果与证据 |
| --- | --- |
| 最终固定输入/Tk 测试 | **57 passed in 19.65s**，[最终日志](tests_57_final.txt)。覆盖成功、检测/参考/context/编码/候选后检查失败、取消、观测错误、跨 run 拒绝、记录篡改、回放、页面切换、响应性与语言确认。 |
| 观测等价性 | 原始源码、派生 observer 关闭、派生 observer 开启三路，固定替身输入的数组、输出、错误码、参考选择、调用计数与 RNG 状态相同。科学 AST 在 enabled/disabled 路径一致，故意修改公式的负例会被拒绝。详见 [科学等价性报告](scientific_equivalence.md)。 |
| 包完整性 | 最终 85 个清单文件全量 SHA 校验通过；界面与仓库一致；三处派生引擎差异及传输模块与原 V2 patch 结果逐字节一致；无硬链接/禁止目录。 |
| Windows 正式启动 | 实际调用随包 BAT 入口，打开主窗口并显示环境检查通过；[启动截图](launcher_ready.png)。最终全哈希自检亦通过：[日志](self_check_final.txt)。 |
| 历史回放 | 实际记录完整验证后在新版 UI 回放；文件清单前后不变，未运行模型。[1280×800 回放截图](real_replay_1280x800.png)。取证脚本直接调用与 UI 相同的 reopen_trace，未把文件选择器自动操作当作测试覆盖。 |
| 失败界面 | 固定模拟事件生成候选后 E404，候选保留，第九阶段失败且最终输出无效；窗口显式标注模拟。测试窗口 1180×760：[截图](simulated_failure_1180x760.png)。 |
| 独立复查 | [复查报告](independent_review.md)：版本标识、预填 disabled 控件问题已修复；未发现剩余确定阻断缺陷，最终验收归 Supervisor。 |

测试过程中的失败没有隐藏：初轮 55 通过、1 次 Tcl 初始化失败见 [tests.txt](tests.txt)；指定现有 Tcl/Tk 目录后原 56 项通过见 [tests_explicit_tcl.txt](tests_explicit_tcl.txt)。本轮曾因受限默认临时目录不可写、随后工作区临时父目录不存在而发生 setup 错误，创建任务专属父目录并使用新 basetemp 后通过。没有削弱断言。环境目录/依赖未改；这些测试进程的 Tcl 路径不作为修改系统配置的理由。补充失败摘要见 [测试尝试记录](test_attempts.md)。

最终测试命令（工作区根目录，PowerShell；`.tmp` 父目录须已存在，basetemp 取全新路径）：

```powershell
$env:TCL_LIBRARY=(Resolve-Path '.conda/buildreasonseg-proposal/Library/lib/tcl8.6').Path
$env:TK_LIBRARY=(Resolve-Path '.conda/buildreasonseg-proposal/Library/lib/tk8.6').Path
& ./.conda/buildreasonseg-proposal/python.exe -B -m pytest tests/test_demo_r1.py -q --basetemp .tmp/demo_r1_57_final
```

不要对已有未知目录使用 pytest basetemp，因为 pytest 会管理其内容。上述是本次已执行的路径；复跑应使用新的任务目录名。

## 单次真实模型烟雾运行

影像由用户选择：`C:\Users\ROG\Pictures\1034.tif`；指令为“最大建筑下方最近的建筑”。用户说明此影像曾测试成功；本次没有运行 Task7J 正式评价或使用其冻结验收结果。

run_id：`20261023_230642_1034`。Qwen 初始解析 `largest_to_below`，因缺 nearest 而不支持；真实建议为 `largest_to_below_to_nearest`，validator 通过。执行者发出明确请求，用户亲自点击确认并在对话中说明已完成测试；保存记录为 `user_confirmation=Y`。没有自动代确认。

真实记录共 24 个事件，九阶段均完成；检测 raw 16、merged 14；视觉特征 `[1,256,64,64]`，候选 Mask 2791 像素。最终运行状态 SUCCESS、观测 OK、总耗时 57.825 秒（包含等待确认），语义状态 NOT_EVALUATED。核心阶段耗时与总耗时统计口径不同，不混用。

保存的 [摘要](real_run/result_summary.json)、[事件](real_run/stage_events.jsonl)、[trace 清单](real_run/trace_manifest.json)、[最终 Mask](real_run/mask.png)、[Overlay](real_run/overlay.png) 是本次运行的真实副本。完整中间预览保留在本地 V3 的对应 results/trace 目录，未复制全部模型/研发目录到 GitHub。

该运行没有对每个真实模型 forward 加独立 profiler 计数；单 worker/无额外推理的证据来自源码审计、固定替身调用计数和一份真实 run 记录。不能将其描述为独立测得的真实模型调用计数证明。

## 截图索引

| 文件 | 类型与说明 |
| --- | --- |
| [real_result.png](real_result.png) | 真实运行结束后的结果页，Overlay 和结果摘要。 |
| [real_stages_1_2.png](real_stages_1_2.png) | 真实运行，语言初始失败、真实建议、确认及实际候选。 |
| [real_fields.png](real_fields.png) | 真实运行，空间图及 SAM2 形状。 |
| [real_candidate.png](real_candidate.png) | 真实运行，C 与第八阶段候选，明确候选语义。 |
| [real_final_guard.png](real_final_guard.png) | 真实运行，第九阶段结构成功及语义未验证。 |
| [real_replay_1280x800.png](real_replay_1280x800.png) | 同一真实记录在最终中文卡片上的只读回放；1280×800 逻辑窗口。 |
| [real_replay_final_guard.png](real_replay_final_guard.png) | 同一真实记录在最终中文第九阶段卡片上的只读回放。 |
| [simulated_failure_1180x760.png](simulated_failure_1180x760.png) | 固定模拟 E404，仅证明失败 UI；1180×760 逻辑窗口。 |
| [launcher_ready.png](launcher_ready.png) | 正式 BAT 打开后的空闲/环境检查通过，尚未推理。 |

截图未重绘、拼接或用生成图替代。早期真实运行截图的阶段标题仍为继承 V2 的英文；最终 UI 已增加中文标题与用途，以回放截图为准。部分截图边缘包含当时桌面已有悬浮组件；它们不是 Demo 功能。当前桌面缩放下截图像素约为逻辑窗口的 1.5 倍，未据此声称所有 Windows DPI 组合均验证。

## 科学保护、限制及审核入口

最终保护检查为 [final_verification.json](final_verification.json)：RC1 完整文件清单、归档 ZIP、V3 模型及源码身份通过；Git 对 legacy 源码、V1/V2、正式评价及 Damage-P0 保护路径没有差异。模型资源独立复制，未训练、未调整阈值/公式/排名、未新增模型分支。Damage Research 未推进。

仍有以下边界：

1. **真实 RC1/V3 成对模型等价性未执行（NOT_PERFORMED）**；不能以 57 项替身测试和单次真实 SUCCESS 宣称全面真实模型等价。
2. 本机运行环境未打包；不同电脑、不同 DPI 的全面兼容性仍需人工验证。阶段弹窗是有界只读预览，不是完整特征张量检查器。
3. 真实烟雾运行只覆盖用户选定的一张影像和一个方向；真实失败页面使用固定模拟事件测试，没有为获取失败截图反复调用模型。
4. 旧 RC1 既有参考选择/语义残余限制继续保留；本任务没有通过调模型改善展示结果。

Supervisor 可先运行 BAT，查看真实记录并检查结果，再对照科学等价性报告、独立复查和实际 Git diff 审核。此交付不自授权 P1、算法修改或科学结论变更。
