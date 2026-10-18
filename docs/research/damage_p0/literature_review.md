# Damage-P0 文献与已有方法分析

截至2026-10-19。目标是支撑可验证问题与基线协议，不做跨论文分数排行榜。
引文/阅读范围见 [sources](sources.md)，结构化主要论文记录见
[literature_registry.json](../../../evaluation/damage_p0/literature_registry.json)。
下面数字全是**论文报告值，项目实测分割性能为零项**。全文阅读与摘要级核查分别标注，
没有取到的表格结果不补造。模型能运行、论文分数高，均不证明本研究目标正确。

## 主要工作及可复现边界

|工作、作者、年份/出处|任务、方法与贡献|报告结果/指标（仅在原实验条件下）|局限与实现条件|
|---|---|---|---|
|[U-Net](https://arxiv.org/abs/1505.04597)，Ronneberger/Fischer/Brox，2015 MICCAI|医学像素分割；编码-解码与skip细节恢复、增强|原文报告512²图像GPU推理<1秒；不是外墙成绩|生物图像数据，原实现Caffe；后续PyTorch第三方版本需注明；没有直接证明微裂缝适配|
|[DeepCrack](https://doi.org/10.1016/j.neucom.2019.01.036)，Liu/Yao/Lu/Xie/Li，2019 Neurocomputing|裂缝；层次多尺度特征与边缘式监督；对应作者537图数据|原文表2：DeepCrack-BN F=85.1%，阈值0.87；DeepCrack-Aug F=86.5%，阈值0.86；不是本协议固定阈值结果|原文按最佳F阈值报告其他指标，不能等同本项目锁阈值测试；与另一个同名DeepCrack模型不能混用；旧预处理和标签宽度影响复现|
|[SegFormer](https://arxiv.org/abs/2105.15203)，Xie等，2021 NeurIPS|层次Transformer+MLP、多尺度局部/全局表达，无位置编码|摘要：B4/ADE20K mIoU50.3%，B5/Cityscapes val84.0%|通用场景指标不等于损伤指标；官方实现旧mmcv/PyTorch，不可直接在Blackwell安装旧CUDA包|
|[HMA/S2DS](https://ben-z-original.github.io/2022_Benz_DetectionStructuralDefects.pdf)，Benz/Rodehorst，2022 GCPR|多损伤；迁移HRNet-OCR和层次多尺度注意力，提出线容忍IoU|表2：S2DS test多尺度[0.25,0.5,1] mIoU92%，crack/spalling各91%；含背景/其他类，非本协议前景mIoU|测试偏向能看清损伤的图块；line-tolerant指标对裂缝宽度不敏感，不能取代面积/宽度评价。作者代码detectionhma与权重需另核验|
|[CT-CrackSeg](https://arxiv.org/abs/2302.11728)，Tao/Liu/Cui/Zhang，2023 ICIP，doi:10.1109/ICIP49359.2023.10222276|DRB局部特征+轻量Transformer+BAM边界监督；裂缝|原文表1：Crack500 F1=73.3%，DeepCrack F1=88.3%，22.88M；§3.2训练256²、batch2、TITAN Xp12GB|Dilated label监督已有先行工作；Crack500裁剪/过滤改变样本分布；旧torch1.10/cu111及checkpoint适配需P1验证；非本机实测|
|[dacl10k](https://openaccess.thecvf.com/content/WACV2024/papers/Flotzinger_dacl10k_Benchmark_for_Semantic_Bridge_Damage_Segmentation_WACV_2024_paper.pdf)，Flotzinger/Rösch/Braml，2024 WACV|真实桥梁多标签损伤；DeepLabV3+/FPN/SegFormer基线，工程师与模型比较|v1论文表3：FPN+EfficientNetB4+辅助分类val mIoU0.414；SegFormer0.400；512²，30epoch|此处v1为18通道；公开挑战v2是19类，不能将v1数值称作v2复现。相似损伤纹理与重叠标签易混淆|
|[OmniCrack30k](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.pdf)，Benz/Rodehorst，2024 CVPRW|20库统一裂缝研究、重复分析、nnU-Net迁移比较|表4：nnU-Net跨子集平均clIoU4px63.7%；表3共30017样本|容忍指标与IoU不可混比；不同子集像素/线标注不一致；共享组成库不可充独立泛化测试；数据需申请|
|[CrackSAM](https://doi.org/10.1016/j.conbuildmat.2024.136573)，Ge/Wang/Guo/Tang/Hu/Chen，2024 CBM431:136573（2023预印本）|SAM视觉编码器adapter/LoRA微调，裂缝语义分割及外域鲁棒性；非SAM2/3|摘要声称外域/噪声优势；未完整表格复核，数值记unknown，不将其声称改写为已验证优越|作者train/test和delta checkpoint公开；基础SAM权重与增量权重分别审许可；是否“无人工提示”需按具体测试代码输入审计|
|[EfficientCrackNet](https://openaccess.thecvf.com/content/WACV2025/papers/Zim_EfficientCrackNet_A_Lightweight_Model_for_Crack_Segmentation_WACV_2025_paper.pdf)，Zim/Iqbal/Al-Huda/Malik/Kuribayashi，2025 WACV|轻量DSC+MobileViT、EEM边缘、ULSAM注意力|表1：Crack500 F1=79.10%、mIoU81.33%；DeepCrack F1=85.88%；表2/摘要0.26M、0.483GFLOPs|§4.6的0.61M/0.80M是相对LMM/MobileNetv3减少的参数量，不是自身参数。未找到经作者确认代码/权重。注意含背景mIoU定义，不能与前景IoU直接比|
|[MixerCSeg](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.html)，Zhao/Ding/Niu/Sun/Guo，2026 CVPR17969–17978|TransMixer局部/全局/顺序特征、方向边缘门控、SRF多尺度融合|原文主比较表：DeepCrack mIoU0.9151、F1=0.9205；Crack500 mIoU0.7824、F1=0.7755；作者2.54M、2.05GFLOPs|已同时覆盖边缘、纹理、方向、融合，不能把这些词的组合宣称新颖。ODS/OIS有阈值扫描含义，与固定阈值F1分开；MIT代码，有checkpoint链接；旧CUDA selective_scan与RTX5080兼容未测|
|[CUBIT-InSeg](https://doi.org/10.1016/j.autcon.2026.106980)，Zhao/Zhang/Huang/Chen/Chen，2026 AIC188:106980|UAV立面实例分割/物理量化；多类基准与跨域验证|公开正文报告80余模型、mask/box AP50与AP50:95；未完整核查各模型精度表，本项目不采纳论文最终主干选择|采集距离/物理量化假设不是本项目已满足条件；与当前下载样本尺寸有差异；需建筑分组检查|
|[CiF](https://arxiv.org/html/2605.18413v1)，Farronato等，2026 arXiv（本次未核定同行评议去向）|基础设施六类实例数据；监督、提示基础模型的域差异，中心偏置与细形状|表3：YOLO11-small tiled mask mAP50:95=20.7%，crack AP=7.5%；实例指标，非语义IoU|不能从其总体基础模型失败断言本项目必需某个新模块；full/tiled计数、公开HF聚合数不同；源图隔离不等于建筑隔离|
|[clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)，Shit等，2021 CVPR|骨架与区域相交构成连接性度量/可微loss，适用管状结构|原文5个血管/道路/神经元数据比较连接性；本次未转抄数值|在特定假设下拓扑论证，不保证所有真实裂缝物理连通；误连与断裂必须分别看|
|[Boundary loss](https://proceedings.mlr.press/v102/kervadec19a.html)，Kervadec等，2019 MIDL|距离场边界loss补区域loss，处理类别极不平衡|原文医学数据ISLES/WMH，报告最高8% Dice及10% Hausdorff改善（原文措辞），不是外墙结果|医学域可迁移性只是待验证假设；不能以加边界loss本身作为创新|
|[Crack-EdgeSAM](https://arxiv.org/abs/2412.07205)，2024 arXiv|YOLOv8生成框+微调EdgeSAM，自动提示系统先行路线|本次摘要级阅读，细表/资源未核验|已存在自提示损伤分割；检测漏检会传递到分割，必须端到端计算而非只算检出目标|

## 已有技术—不足—验证方式

|技术主题|明确先行研究|仍需验证的问题（执行者推断）|受控验证方式|
|---|---|---|---|
|边缘与纹理联合表达|DeepCrack、CT-CrackSeg、EfficientCrackNet、MixerCSeg|在墙砖缝/阴影/粗糙涂层上是否引入更多假阳性|同数据/分辨率/预算，纹理混淆负样本分层，报告Precision及FP面积；先基线诊断|
|多尺度融合|HMA、SegFormer、MixerCSeg|提升来自保留原生像素还是复杂模块，是否损伤小裂缝宽度|同训练像素与算力预算比较resize/tiling；单尺度/多尺度分开计时，不先归因于新特征|
|连续性/几何先验|clDice、OmniCrack线指标|断裂减少是否以跨过接缝的误连接为代价|clDice+像素IoU+人工审查错误连桥；不只追单个拓扑分数|
|边界监督|Boundary loss、CT-CrackSeg BAM|标注边界主观性与分辨率变化能否超过模型差异|原生网格BoundaryF1和IoU；固定容忍度，单独标记标签歧义，不在测试后扩大容忍|
|注意力增强|HMA/ULSAM/TransMixer|增加容量是否才是提升来源，外域能否保持|参数/训练预算受控的现有方法复现对照，跨材质外测；当前不实施新模块|
|类别专用先验|HMA多损伤、CUBIT/CiF细长裂缝与面积剥落|一套配置是否偏向剥落而漏裂缝|两类独立IoU/Recall，按面积分层；不合并damage得到高分|
|基础模型提示/微调|CrackSAM、Crack-EdgeSAM、SAM2/3、CiF|提示质量、漏定位与像素细节谁是瓶颈|同一个分割器+真实人工/自动/Oracle条件分报告；Oracle使用全部GT位置也不称自动|

当前没有证据支持“边缘+纹理+多尺度+注意力”组合本身为独立创新，也不能证明某个模块
能解决无人机外立面两类损伤。可研究的问题见 [research_gaps](research_gaps.md)，
须由P1基线与更系统先行研究检索支持后，Supervisor才可选择方向。
