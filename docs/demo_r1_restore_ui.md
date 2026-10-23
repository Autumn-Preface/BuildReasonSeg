# DEMO-R1 恢复与界面升级

执行日期：2026-10-23。工程交付已完成，等待 ChatGPT Supervisor 审核；不构成科学验收。

完整报告及真实/模拟截图索引：[工作报告](../reports/demo_r1/20261023/README.md)。

## 运行

本机独立目录：`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V3`。
双击其中的 `启动BuildReasonSeg Demo.bat`，选择 RGB 影像并输入指令；任何语言解释或建议仍须用户确认。
这是空间推理原型演示，SUCCESS 仅表示运行结构检查通过，目标语义正确性须人工核验。

包内约 4.4 GiB 模型及必需资源为独立复制并经过 SHA256 校验，无硬链接；Python 使用本机已有环境，未附带安装环境。
新电脑须提供兼容 Python，可通过 `BUILDREASONSEG_PYTHON` 指定。禁止把整套研发目录或权重提交 Git。

## 实现与验证入口

- `scripts/build_demo_r1.py --assemble`：从历史 Git 源码、原 V2 observer 和校验后的当前 RC1 资产构建。目标或 staging 已存在时拒绝覆盖；当前 V3 已存在，不应直接重复构建。
- `tests/test_demo_r1.py`：固定输入与替身引擎测试，阻止加载真实权重，包含真实 Tk 控件测试。
- `scripts/verify_demo_r1_delivery.py`：复核 RC1 全清单、V3 包清单、历史 ZIP、派生引擎字节及保存记录只读回放。
- `scripts/demo_r1_ui_evidence.py`：仅开发取证使用，`replay` 读取本次真实记录；`failure` 生成显式标注的固定模拟 E404。脚本和模拟文件均不进入交付包。

测试使用已有环境 `.conda/buildreasonseg-proposal/python.exe`；进程内指定现存 `TCL_LIBRARY`、`TK_LIBRARY`，并将 pytest 临时目录置于工作区已有父目录下。最终 57 项通过，完整命令与限制见报告。

## 行为边界

V3 沿用 V2 的一次请求、单 worker、真实阶段事件、只读预览和历史回放机制；只改展示层与恢复构建方式。
观测 Hook 只存在于 V3 引擎副本。旧 RC1、V1/V2 历史源码及 ZIP、Damage-P0 材料保持原样。
源码 AST 与固定替身测试通过，单次真实模型烟雾运行成功；没有进行真实 RC1/V3 成对推理等价性实验，不宣称全面真实模型等价或精度改善。
