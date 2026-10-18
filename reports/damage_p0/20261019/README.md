# BuildReasonSeg — Damage-P0 执行报告

任务：DAMAGE-P0，授权日期2026-10-19。状态：**EXECUTION_COMPLETE，等待Supervisor独立验收**。
本报告不是 ACCEPTED，不授权Damage-P1。验收权归ChatGPT Supervisor。

## 1. 执行摘要

已建立独立Damage Research研究线，保留旧RC1冻结边界；形成8个数据候选、9个模型/实现候选、
15篇主要文献的结构化调查，以及环境、实验协议、研究问题、存储制度。
抽取并检查9组训练原图/人工标注，产生带原生裁剪的GT叠加证据；未训练、未运行分割模型、
未下载新模型权重、未改旧模型/GT/评价逻辑。

首要发现：CUBIT-InSeg任务最贴近UAV立面两类损伤，但样本并非全部4800×3200；同表面相近视角明显，
还不能证明建筑/序列隔离。首轮建议U-Net、SegFormer-B0、YOLO11s-seg三条全自动路线，
另设SAM2.1-tiny提示诊断；全部是待审建议，没有选定最终主干。

## 2. Git 起点与治理

启动按AGENTS规定读取五份治理/handoff文件，实际工作树干净。
本地与新fetch核实的远端清理分支HEAD均为
`0e7ee93c8ca62426e5f79b39fc874f1a2b7e7ee4`；远端main为
`57b368d5647e842d8f31d6d1a9997bf1df3cc0fb`，未包含起点之后150个提交。
从核实的`origin/codex/legacy-cleanup-20261019`创建`codex/damage-p0-20261019`，不合并main。
PATH Git缺HTTPS helper，改用现有bundled Git正常fetch/push，没有重装Git或改全局配置。

治理检查点：`40aa16398e850578f6dc0bc79758736147b9d835`，已推送。
调查/协议检查点：`bbfd017b71431a88e33503e5b02e75fef0bad9e7`，已推送。
后续调查和最终交付记录在分支Git log中；最终实际本地/远端HEAD在push后从Git读取并向Supervisor报告，
不预写自身SHA再制造额外提交。[git_startup.json](../../../evaluation/damage_p0/git_startup.json)记录起点观察。

新增GOV-D009与[DAMAGE_RESEARCH_SCOPE](../../../governance/DAMAGE_RESEARCH_SCOPE.md)，
AGENTS仅追加新线边界，PROJECT_STATE保留历史字段并追加research_lines。
旧科学结论、RC1代码、Task7I/7J、权重、GT/split与Demo本地ZIP仍冻结。原始授权见[authorization](authorization.md)。

## 3. 数据主要发现

[调查明细](../../../docs/research/damage_p0/dataset_survey.md) / [CSV清单](../../../evaluation/damage_p0/dataset_registry.csv)。

|候选结论|依据|尚未解决|
|---|---|---|
|CUBIT-InSeg为首选训练候选|官方UAV立面crack/spalling、CC BY4；train ZIP实查5596图/5596标签；抽4组|两个CR样本3000×4500，两个SP样本4800×3200；类别ID权威说明、全部尺寸分布、建筑/航线分组|
|CiF、S2DS、dacl作为双损伤外域候选|真实基础设施、像素/实例标签，可评价两类|桥梁/外立面与剥落定义偏移；CiF Full/Tiled同源；S2DS禁止再分发；dacl AWS403且隐藏测试GT|
|Facade390、DeepCrack为crack-only外域候选|作者明确裂缝任务，Facade390 CC BY4；DeepCrack非商业研究/教育|未知spalling不能当负类；聚合库预训练可能污染外测；Facade390本次未做配对抽样|
|OmniCrack为辅助/预训练调查材料|论文明确重复/标签定义问题，多子集迁移|数据需申请、组件许可；与组成子库不独立；本次不申请邮件或下载|
|原始CODEBRIM不作像素主基线|原始为多标签分类|不与SegCODEBRIM混同；无像素GT不能评价像素IoU|

真实抽样：[9组证据图与署名](sample_evidence/README.md)，[逐样本尺寸/哈希/解析结果](../../../evaluation/damage_p0/sample_checks.json)。
四组CUBIT、三组DeepCrack、两组CiF可读且无坐标越界/整图配准错误；执行者可视观察与限制已逐项记录。
有局部边界外扩/宽度差异，不宣称全库无漏标；小样本不是代表性质量抽查或独立科学证明。

## 4. 论文与已有方法

[文献报告](../../../docs/research/damage_p0/literature_review.md) / [15篇结构化记录](../../../evaluation/damage_p0/literature_registry.json)。
层次边缘、多尺度、边界loss、注意力、连接性loss与领域SAM微调均有充分先行研究。
2026 MixerCSeg已经联合方向边缘/局部全局/多尺度；CiF直接研究基础模型在真实基础设施上的局限。
不能将这些模块重新组合就认定创新。主要论文结果附原始数据/指标/阅读范围；没有跨库分数总排名。

部分记录只达到摘要/作者资源级（如Crack-EdgeSAM），部分数值表未完整取得；明确unknown并说明影响。
EfficientCrackNet表2参数0.26M；正文0.61M是相对LMM减少的参数量，已按独立复核纠正初稿误读。
DeepCrack原文最佳F阈值口径、HMA/OmniCrack容忍指标、CiF实例AP均不能当本协议固定阈值像素IoU。

## 5. 模型资源与候选

[模型报告](../../../docs/research/damage_p0/model_survey.md) / [模型CSV](../../../evaluation/damage_p0/model_registry.csv)。
U-Net PyTorch候选明确为第三方实现；SegFormer官方旧栈需兼容迁移，代码与HF权重条款分记；
YOLO11s-seg官方checkpoint HEAD200、20,669,228 bytes；SAM2.1-tiny HEAD200、156,008,466 bytes。
这些只是访问/体积核查，不是下载/模型运行成功。SAM3匿名权重HEAD401，需要授权访问条件；
官方真实文本API与SAM2点/框API不同，不能将SAM2当原生文本分割器。

CT-CrackSeg、CrackSAM、MixerCSeg保留为领域方案；权重特定条款/现代CUDA兼容仍需核查。
EfficientCrackNet未确认作者实现，暂不列可执行项。源码查询提交记录见[resource_pins](../../../evaluation/damage_p0/resource_pins.json)。

## 6. 本机硬件/软件

[环境报告](../../../docs/research/damage_p0/environment_audit.md)有全部命令与JSON证据。
实测Ultra9 275HX、24核/24线程、约31.38GiB内存；RTX5080 Laptop约16GiB显存、驱动617.42；
Windows11 build26300。proposal环境Python3.11.16、torch2.13.0+cu132能调用CUDA，
64×64矩阵乘总和262144与期望一致。只是算子烟测，不证明模型精度、吞吐或训练稳定性。

WSL3.0.3.0/WSL2存在，但仅列Stopped docker-desktop；未测WSL内GPU，无Ubuntu研究环境ready证据。
PATH没找到nvcc；没有改驱动/Toolkit/Conda依赖。卷C:审计时空闲约144GiB，项目不是另一个D:分区。
显存、batch和存储建议均标估算，实际模型兼容性仍待获准P1检查。

## 7. 建议的基线与监督条件

自动A：U-Net、SegFormer-B0、YOLO11s-seg；提示B：同SAM2.1-tiny分别用真实人工/自动/GT Oracle框；
文本C只预留定义，SAM3不要求P0实现。自动系统将定位漏检完整计入FN，不能只给检出目标算mask质量。
Oracle暴露GT位置和目标数，独立标注，不称自动，也不保证实际分数必然为理论上界。
预训练条件不同、tile与整图条件不同的结果以系统轨报告；不能伪装为纯架构受控比较。

## 8. 正式协议草案

[experiment_protocol.md](../../../docs/research/damage_p0/experiment_protocol.md)明确：

- crack/spalling独立通道、未知标注ignore、重叠标签不强制互斥、不可统一类不随意合并。
- 官方split优先，建筑/序列/原图/重叠tile隔离；test不参与阈值、提示、模型与超参选择。
- 原生网格重建评价；IoU/Dice/P/R的公式、按类全局像素累计、前景宏类均值、原图宏平均与NA分母。
- 空GT空预测不补1刷分；负图假阳性率另报；small边界只由train分位数预注册，数值TBD。
- 固定阈值草案与提示预算、BoundaryF1/clDice候选、种子、训练/搜索上限、checkpoint与计时/显存规则。

所有配置仍DRAFT；P1需审查数据版本/分组、模型/权重许可、最终优化器/预算、small阈值数值等。
没有将这些草案值包装成实测最优参数，没有正式评价算法实现或训练。

## 9. 研究问题

[research_gaps](../../../docs/research/damage_p0/research_gaps.md)列6个候选问题：分辨率丢失、背景纹理混淆、
裂缝断裂/误连、剥落边界与标注不确定性、prompt定位瓶颈、双类有限预算兼顾。
每项有先行研究、验证数据/指标、受控对照及否证条件。当前没有证据宣布最终创新模块。

## 10. 风险、未解决项与建议

P1准入重点是CUBIT版本/类ID与分组泄漏，不能只看到官方下载便跳过；
本阶段未扫描全库分辨率、未拿到所有数据的原始标签/建筑ID、未复现论文精度、未加载候选模型。
Facade390/S2DS/dacl/OmniCrack未做配对抽样，状态和原因已记；是否补充P0由Supervisor判断。
部分checkpoint许可和现代CUDA兼容未证实；SAM3访问受限；新环境和完整数据下载需要P1明确计划。
没有向作者发邮件、申请账号权限、购买资源或擅自改研究目标。

## 11. 新增存储与保护

[storage_policy](../../../docs/research/damage_p0/storage_policy.md)确立单份资源、下载前bytes预算、
缓存重建清单和旧Demo/冻结资产保护规则。本阶段新目录逻辑字节约48MiB，确数见
[storage_final.json](../../../evaluation/damage_p0/storage_final.json)；不含Git对象、自身JSON及既有治理小增量，
不是物理分配空间或卷空闲变化。远端14GB CUBIT包没有整包落盘；没有新权重下载或旧缓存清理。
本地源快照/样本留待审查，列为可重建候选，未自动删除。

## 12. 自检、证据与交接

[integrity_final.json](../../../evaluation/damage_p0/integrity_final.json)检查改动白名单、旧保护资产SHA256、
本地引用链接、JSON/CSV、样本hash和GPU算子记录。6443保护记录/6434唯一文件fresh hash无错误；
最终核验PASS（范围、引用、登记、样本完整性）。[validation.json](../../../evaluation/damage_p0/validation.json)
记录9个Python脚本AST及治理/handoff YAML解析PASS。未运行旧产品完整测试，
因为本任务不修改产品逻辑，不能以旧测试PASS冒充损伤科学正确性。

已完成[只读独立材料复核](independent_review.md)，发现的2项P2均修复并经reviewer读回关闭，
未发现剩余P0交付阻断问题；仍不替代Supervisor验收。最终handoff记录执行结束、等待Supervisor审查；
所有重要成果提交推送任务分支，工作树/本地远端HEAD实际核对后报告并停止。

证据入口：

- [原始来源索引](../../../docs/research/damage_p0/sources.md)、[source_access](../../../evaluation/damage_p0/source_access.json)、[weight_access](../../../evaluation/damage_p0/weight_access.json)。
- [sample_access](../../../evaluation/damage_p0/sample_access.json)、[sample_checks](../../../evaluation/damage_p0/sample_checks.json)、[CUBIT目录元数据](../../../evaluation/damage_p0/cubit_folder_inventory.json)。
- [environment_facts](../../../evaluation/damage_p0/environment_facts.json)、[Python GPU探针](../../../evaluation/damage_p0/python_proposal.json)、[WSL解码记录](../../../evaluation/damage_p0/wsl_facts.json)。
- [CURRENT_TASK](../../../handoff/CURRENT_TASK.md)、[EXECUTOR_STATE](../../../handoff/EXECUTOR_STATE.yaml)。

下一步只等待ChatGPT Supervisor审查；**禁止自动启动Damage-P1**。
