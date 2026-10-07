# Task8E 用户 Demo 交付审计

状态：IN_PROGRESS。任务为用户界面和独立打包，科学运行链未修改。

最终路径：`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1`。

接受的科学前置版本为 `b569dcefa26af9788a9ec75eb52cbb8317305bea`。旧 Advisor RC1 仅用于读取和逐字节复制，禁止写入；Task8C/8D 和治理材料作为完整性基线保存。

## 范围与界面

既有环境的 Tkinter 检查为 TK_OK，因此实现 Windows Tkinter GUI，未安装依赖。主界面包括图片选择、四个文本快捷填充、指令框、确认、单一后台分析、原图与 Overlay、Mask 查看和结果目录入口。默认窗口完成本机显示核验，见 [界面截图](task8e_user_demo_ui.png)。截图使用空白界面，不代表模型结果。

语言输入复用既有 ProgramHead、Validator 和 CLI 的建议辅助函数。快捷按钮仅填文本。直接支持、Qwen 建议、fallback 均要求明确确认；有限兼容模式还展示最终解释。所有模型执行保持 automatic reference。没有手选 proposal、阈值、排名或模型控制。

一个 worker 同时仅处理一项操作，主线程读取消息队列并显示确认。分析期间禁用重复操作，关闭请求等待当前操作完成；不强杀推理。SUCCESS 展示“推理流程已完成。结果已生成，请结合原图人工核验目标是否正确。”，语义状态为 NOT_EVALUATED。

## 包布局与用户流程

```text
BuildReasonSeg_Demo_V1/
  启动BuildReasonSeg Demo.bat
  BuildReasonSeg_Demo.py
  使用说明.md
  版本说明.txt
  results/                         初始为空
  _ui/                            界面、环境与启动辅助代码
  _runtime_cache/{yolo,matplotlib}/ 私有运行缓存
  _engine/
    predict.py
    buildreasonseg/                已接受源码的静态导入闭包
    configs/
    model/                         自有、校验后的资产副本
    VERSION
    package_manifest.json
    inference/{input,output}/      初始为空
    logs/                          初始为空
```

用户双击启动文件，然后：选择图片 → 输入指令 → 语言解析 → 确认理解 → 自动推理 → 查看并人工核验结果。支持最大建筑左、右、上、下方最近建筑。每次结果位于 `results/时间_图片名[_序号]/`，成功只复制 mask、overlay 和独立的用户摘要；失败仅保存摘要，不伪造图片，不修改内部原始产物。

## 环境与移动性

启动顺序为 BUILDREASONSEG_PYTHON、未来的相对 runtime/python.exe、本机已验证 Python、PATH。候选通过轻量导入与版本检查后才启动；Ultralytics 必须为既有 8.4.164，无兼容环境则中文提示并停止，不安装依赖。PowerShell 启动辅助使用 UTF-8 BOM。

所有 Demo 源码、模型和结果均相对于自身目录。`_ui/runtime.json` 中唯一机器定位信息是已验证的 Python 环境位置，这是任务第18节要求的本机环境候选，并非科学源码或模型定位依赖。Python/CUDA 环境未打包，跨电脑零配置不在 V1 范围。

目标为 PATH_PORTABLE_ON_CONFIGURED_MACHINE；将 staging 整包移动至最终目录并在两处检查，不重复复制模型，不使用链接。移动实测和最终检查待完成。

启动前先建立可写 YOLO_CONFIG_DIR、MPLCONFIGDIR，配置离线和无字节码环境变量。私有缓存允许运行时生成文件，不属于科学产物。

## 可复现组装和资产来源

`scripts/build_task8e_user_demo.py` 从接受 Git HEAD 校验 canonical manifest，执行包含函数内部导入和包初始化的 AST 闭包分析。54 个必需 source/config 文件与 Git 字节一致，16 个模型/组件文件及 VERSION 从旧 RC1 复制并逐个 SHA256 校验。少量冻结研究辅助模块被运行代码传递导入，因此保留其源文件；不交付历史数据、诊断或研究入口。

使用专属且此前不存在的 sibling staging，不合并或覆盖已有交付。组装前后核验旧 RC1 的全2184文件 bytes/SHA256/mtime/file set；重命名前再检查 manifest 和禁入内容。没有多 GB 模型进入 Git。包清单、实际大小和最终禁入审计将在完成后追加。

## 测试证据

当前 revision：`tests/test_task8e_user_demo.py` **57 passed in 2.76s / exit 0**。前两次正常 TEMP 调用分别为 57 passed in 3.29s 和 57 passed in 3.02s；当前修订完整重跑已通过。测试覆盖语言确认、建议与 fallback、快捷填充、单请求与重复点击、失败无伪造图、原产物和原图不变、相对路径、环境发现、友好失败、整包移动、干净组装、禁入内容及中文文档。

首次沙箱 TEMP 调用为 **57 fixture setup errors / 0 assertion failures / exit 1**（14.36s）。未进入断言，未组装或推理。该 L0 环境失败记录保留在 JSON；改用父目录存在的实际 TEMP 后通过。

Fake tests 对 torch.load、模型构造/加载、detector、SAM2、D-B1、ProgramHead 和 pipeline 调用设禁止护栏，显式 fake 例外。没有真实模型调用。真实模型 smoke 状态为 REAL_MODEL_SMOKE_DEFERRED_TO_USER_ACCEPTANCE_TEST，不使用四个 Task8C locked 样例，也不重试或修复科学结果。

## 已知边界和待完成项

该界面改善本地操作，不证明模型语义正确；既有能力边界和结构 guard 保持冻结。推理可能失败或产生需人工核验的目标。预览保持纵横比，读图复用 accepted image contract。

下一步为 staging 实包、无模型环境/绑定检查、整包移动和最终完整性核验。最终审批属于 Supervisor。

## Staging checkpoint

独立实包已完成。清单80项全部 SHA256 通过；正常环境探测通过；真实 accepted API 绑定在禁止模型构造、加载和推理护栏下通过；正常 GUI 启动完成并安全关闭。原生 PowerShell launcher 探测选中已验证 Python，其最终 Start-Process 在验证中被拦截记录，以免留下孤立窗口。

验证脚本曾出现绑定顺序、导入护栏范围、GUI poll 等待和 PowerShell 验证变量 scope 四项 L0 调用失败。所有真实记录保留在 JSON，随后仅调整验证调用机制后通过；交付代码和57-test revision 未改变。staging 私有 YOLO_CONFIG_DIR 下生成 Ultralytics/settings.json，未落入根目录或旧 RC1。
