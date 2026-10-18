# Damage-P0 独立材料复核记录

2026-10-19；reviewer：独立子代理 `/root/independent_p0_review`，未参与材料编写。
依据AGENTS第6节独立reviewer偏好执行只读复核；不能替代ChatGPT Supervisor验收。

范围：八份研究文档、数据/模型registry、主要检查脚本、环境事实与两张实际样本图；
抽查本地原始论文DeepCrack、S2DS、dacl10k、OmniCrack、EfficientCrackNet、MixerCSeg数字。
reviewer未重复执行整库hash，未运行模型/训练、未修改文件。执行者另查全部9张证据图和保护哈希。

|发现|严重度|修复及读回复核|
|---|---|---|
|附加BF1/clDice在两侧均非空但零交集时分母0，可能被当NA排除；small重叠实例像素需去重|P2|协议明确BF_P+BF_R=0→BF1=0、Tprec+Tsens=0→clDice=0；非空mask空骨架列异常；smallRecall按有效GT并集。reviewer读回关闭|
|误把EfficientCrackNet正文0.61M当自身参数，与0.26M表格制造虚假矛盾|P2|原文§4.6为比LMM/MobileNetv3少0.61M/0.80M；表2自身0.26M，三者一致。报告、来源、生成脚本及registry全量同步修正，reviewer读回关闭|

最终reviewer结论摘记：两项P2已关闭，未发现剩余P0交付阻断问题；主指标空mask、提示分组、
Oracle限定、定位漏检计FN、原生网格重建、预算非实测、未知许可与泄漏风险声明明确；
Git范围与双研究线隔离相符。

剩余审定项：CUBIT类别ID、建筑/序列分组、正式转换器、资源许可准入及协议TBD。
这些是已公开的P1准入/补充P0决策事项，执行者不自行冻结。
本记录是独立材料复核，**不是ACCEPTED、科学正确性证明或Damage-P1授权**。
