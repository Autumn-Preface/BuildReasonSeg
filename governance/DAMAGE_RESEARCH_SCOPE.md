# Damage Research 授权与隔离边界

授权：2026-10-19 DAMAGE-P0 正式任务书，原文见 [authorization](../reports/damage_p0/20261019/authorization.md)。
保留项目中文名“空间推理引导的建筑物结构 MLLM 相关推理分割方法”。

本线研究无人机 RGB 外立面可见裂缝、剥落的分割。P0 仅调查公开数据、文献、模型、
本机资源，检查少量合法样本，提出协议草案、研究问题和存储制度。不实现新模块、不训练、
不升级驱动或既有环境、不确定最终主干。不以 MLLM 或空间推理为基础视觉研究前置条件。

旧 RC1 的架构、代码、权重、Task7I/7J、GT/split、评价逻辑、Demo ZIP 和历史结论继续冻结。
GOV-D001 至 GOV-D008 的历史含义保留；GOV-D009 是独立研究线授权，不是重开旧结论。
本阶段候选/协议均为建议，P1 数据、模型、预算和创新方向待 Supervisor 独立审查后确定。

写范围：AGENTS 增量、governance 增量、handoff、docs/research/damage_p0、datasets/damage、
evaluation/damage_p0、reports/damage_p0/20261019。GitHub 保存源码/报告/小证据，原始资源单份本地管理。
执行者只能声明 EXECUTION_COMPLETE/STOP/FAILED，不能声明 ACCEPTED。
