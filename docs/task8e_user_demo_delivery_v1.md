# Task8E 用户 Demo 交付审计

状态：**READY_FOR_SUPERVISOR_AUDIT**。最终交付为 Tkinter GUI 的独立本地 Demo，等待 Supervisor 远程与本机交付审核。

外部路径：`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1`。

分支：`delivery/task8e-user-demo-v1`。接受的科学前置版本为 `b569dcefa26af9788a9ec75eb52cbb8317305bea`；科学产品源码、模型、阈值、排序、grammar、automatic largest reference 和512上下文保持不变。任务无真实推理、无 GT 调用、无科学修复。

## 交付范围和用户操作

既有环境 Tkinter 检查为 TK_OK，未安装依赖。主界面包括图片选择、指令输入、四个文本快捷填充、单项后台分析、原图和 Overlay、结果摘要、Mask 查看、结果目录与使用说明入口。[界面截图](task8e_user_demo_ui.png)为无模型空白界面，已检查控件完整可见。

启动动作：双击根目录 **启动BuildReasonSeg Demo.bat**。

六步流程：选择 RGB 遥感图片 → 输入支持的指令 → 既有语言解析与 Validator → 明确确认理解/建议 → 一次 automatic 推理 → 查看并人工核验 Mask/Overlay 与保存结果。

仅支持最大建筑左、右、上、下方最近建筑。快捷按钮只填文字；正常文本必须经过既有 Qwen/ProgramHead。直接支持需确认；unsupported 使用现有建议辅助函数和 Validator 再询问；fallback 先明确许可，再确认解释。没有自动 Y，没有诊断、手选 proposal、阈值/排名/模型、训练或评估控制。

SUCCESS 显示：“推理流程已完成。结果已生成，请结合原图人工核验目标是否正确。”同时注明语义未自动验证。此状态仅说明既有运行结构 guard 通过，不证明 reference、target 身份或 nearest 关系正确。本 Demo 不读取 GT。

一个后台 worker 串行处理操作，主线程通过队列显示确认和结果。忙时禁用重复运行，关闭操作等待完成；不强杀进程。错误保留原代码并显示短中文说明，不自动重试，完整日志仅内部保存。

## 最终包布局

```text
BuildReasonSeg_Demo_V1/
  启动BuildReasonSeg Demo.bat
  BuildReasonSeg_Demo.py
  使用说明.md                     14个非开发者章节
  版本说明.txt
  results/                        初始为空
  _ui/                            UI、环境与启动辅助
  _runtime_cache/
    yolo/Ultralytics/settings.json 私有运行缓存
    matplotlib/
  _engine/
    predict.py
    buildreasonseg/                accepted source静态导入闭包
    configs/
    model/                         自有模型与组件副本
    VERSION
    package_manifest.json
    inference/{input,output}/      初始为空
    logs/                         初始为空
```

结果保存在 `results/<时间>_<图片名>[_序号]/`，冲突时增加序号，不覆盖。成功只复制 `mask.png`、`overlay.png` 和独立的 `result_summary.json`，不修改内部原始产物，也不把用户摘要冒充科学 result.json。失败仅保存摘要，不伪造图像。原图从不修改，显示缩放保持纵横比，读取复用 accepted image contract。

实际包82文件、**4,716,487,685 bytes**，约 **4.716 GB / 4.393 GiB**。清单80项内容为4,716,468,600 bytes；清单自身18,600 bytes，另有运行生成私有缓存485 bytes。完整清单和逐项SHA256见仓库[证据JSON](../evaluation/task8e_user_demo_delivery_v1.json)及外部 `_engine/package_manifest.json`；manifest自身SHA256为 `2533209be63385c6a571e998669794a7015e8a7abdda1e1fd786f38f9b7e3432`。

## 组装和资产来源

`script`为 `scripts/build_task8e_user_demo.py`。脚本使用 accepted Git HEAD 的 canonical blobs 验证清单；AST分析覆盖函数内部导入与包初始化。54个必需 source/config 文件、VERSION 和16个模型/组件资产来自 accepted RC1，逐项复制校验。模型资产 SHA256 全部与旧 RC1 一致；多GB资产未加入Git。

少量冻结研究辅助模块是运行代码传递导入所需，保留其源文件；没有研究数据、历史诊断、报告、测试或训练/评估入口。optional metrics.json 和 qwen_integrity_cache.json 无运行依赖，未打包。

只创建此前不存在的 sibling staging；不合并、覆盖或删除任何既有交付。组装前后核验旧 RC1 全量清单，整包移动前再次核验80项哈希和禁入内容。没有链接、junction 或 symlink；移动未再次复制模型。

## 环境、缓存和移动性

Python发现顺序：BUILDREASONSEG_PYTHON → 可选相对 runtime/python.exe → 本机已验证环境 → PATH。轻量探测至少检查 Python、torch、Ultralytics、transformers、numpy、Pillow、opencv、scipy；也检查 accepted运行所需其他依赖。Ultralytics为冻结8.4.164。无兼容环境时中文提示停止，不安装或下载依赖。

本机已验证 Python 为 `.conda/buildreasonseg-mvp/python.exe`。`_ui/runtime.json` 保存其环境定位路径，这是任务第18节要求的本机候选，并非科学源码/模型来源。GUI/engine/model/results路径全部从 Demo 自身目录解析，未依赖旧RC1或研究工作区的科学源码和资产。

实测先组装 staging，再整包重命名至最终目录，在无关 TEMP 工作目录启动新的检查进程：环境、80项完整SHA256、真实 accepted API绑定和正常 GUI启动均通过；模型路径全部存在且在最终 `_engine` 下，窗口ready后安全关闭。结论为 **PATH_PORTABLE_ON_CONFIGURED_MACHINE**，folder/source/model portability = SUPPORTED；Python/CUDA/第三方运行环境 portability = **NOT_BUNDLED_V1**。不声称跨PC免配置。

私有可写缓存先创建后使用；显式设置 YOLO_CONFIG_DIR、MPLCONFIGDIR、HF_HUB_OFFLINE、TRANSFORMERS_OFFLINE、无字节码和UTF-8变量。仅新包私有缓存生成一项 Ultralytics/settings.json。最初验证在repo工作目录导入库导致缓存默认目录提示含该位置，已仅对新缓存的三个目录提示设为Demo相对路径，其他设置不变；无源码/模型/科学配置改变。根目录没有意外Ultralytics配置。

PowerShell UTF-8 BOM及原生解析通过，真实候选环境探测在staging与final均通过。最终 Start-Process 在验证中被拦截以记录路径、参数、WorkingDirectory和Hidden helper，避免留下孤立窗口；GUI本体另经正常启动/ready/安全关闭核验。

## 验收证据

| Gate | 结果 |
| --- | --- |
| 当前 dedicated revision | 57 passed in 2.76s，exit 0 |
| staging/final self-check | 环境通过，80/80全SHA256通过 |
| accepted API binding | 无模型构造/加载/推理护栏下通过，automatic only |
| GUI startup | ready，通过；未点击Run，安全关闭 |
| launcher discovery | 原生探测与相对启动参数通过 |
| copied model assets | 16/16 SHA256 MATCH |
| accepted source/config + VERSION | 55/55 MATCH |
| forbidden content | PASS，0 offenders，0 links |
| 初始results/input/output/logs | 全空 |
| practical folder move | PASS，不复制第二份模型 |
| 旧RC1 Git-canonical source/config | checked135 / match135 / missing0 / mismatch0，exit0 |
| 旧RC1全量 | 全2184 bytes/SHA256/mtime/file set不变 |
| Task8C / Task8D | 10/10、13/13锁定文件不变 |
| canonical/治理基线 | 139/139不变 |
| real model smoke | REAL_MODEL_SMOKE_DEFERRED_TO_USER_ACCEPTANCE_TEST，calls0 |

57项测试覆盖确认/建议/fallback、快捷填充不绕过Qwen、单请求和重复点击、失败无伪造图、摘要拷贝不改原件、路径移动、环境发现、友好失败、干净组装、禁入内容、原图显示与14节中文文档。Fake测试禁止torch.load、模型构造/加载、detector、SAM2、D-B1、ProgramHead及pipeline实调用，只有显式fake例外。

前两次正常TEMP调用为57 passed in3.29s、57 passed in3.02s；当前最终修订完整重跑通过。首次沙箱TEMP调用为 **57 fixture setup errors / 0 assertion failures / exit1**，14.36s，未进入断言；记录保留，不伪装PASS。

额外验证调用的L0失败记录全部保留：engine绑定顺序、import护栏范围、GUI队列等待、PowerShell验证变量scope，以及误用CRLF working-tree manifest比较LF Git-canonical bytes。仅纠正验证机制；交付代码/tests未改。external manifest 与权威Git blob完全一致，SHA256 `df9a870d72a25421311698f7a8865e07b4a4e11e9bfe52df18975da17b3bcf7e`，工作树换行差异并非漂移。

旧RC1的历史output、input、protected模型/Qwen/SAM2及grandfathered606-byte Ultralytics/settings.json均由全量基线覆盖并保持不变。Task8C/8D和governance未修改。最终外部包不含handoff、governance、evaluation、Git材料、历史研究产物、测试套件或开发入口。

## 已知边界和后续gate

本任务以accepted运行证据、fake测试和无模型集成检查验证界面/打包，**没有真实模型smoke**，未使用1003/1008/1009/1010 locked TIFF。真实模型操作延后至用户验收；不能把测试通过当作科学正确性，也不修复既有失败案例。

仍需已配置PC上的Python/GPU依赖；模型推理可能失败或产生需人工核验的错误目标。V1不提供并行分析、取消/强杀、自动重试、诊断或科学调整。

CURRENT_TASK与EXECUTOR_STATE均为READY_FOR_SUPERVISOR_AUDIT。Next gate：`CHATGPT_TASK8E_USER_DEMO_REMOTE_AND_LOCAL_DELIVERY_AUDIT`。完成commit/push后STOP；不进入Task8F或科学修复。Git最终SHA由推送后实际状态报告，不写入自引用占位符。
