# 原始来源与证据导航

核查日期2026-10-19；只将原始论文、作者仓库、官方发布平台作为事实依据。
[source_access.json](../../../evaluation/damage_p0/source_access.json)记录HTTP状态、取回字节与SHA256；
[resource_pins.json](../../../evaluation/damage_p0/resource_pins.json)记录源码查询时提交；
动态README链接不代表未来仍保持相同内容。全文快照/PDF为本地参考缓存，不公开整篇复制。
论文实测性能一律没有在本项目复现。下列来源ID用于报告交叉定位。

|ID|来源|用途/阅读范围|
|---|---|---|
|D1|[CUBIT-InSeg 作者发布](https://github.com/BenyunZhao/CUBIT-InSeg)，[论文DOI](https://doi.org/10.1016/j.autcon.2026.106980)|README许可/入口；论文公开摘要、数据卡和方法/评估章节网页内容；ZIP目录和4个配对样本实测。论文网页后续访问不稳定，不声称完整本地PDF审读|
|D2|[DeepCrack 作者库](https://github.com/yhlleo/DeepCrack)，[DOI](https://doi.org/10.1016/j.neucom.2019.01.036)|数据约束、ZIP目录、3组训练样本；同名论文需区分|
|D3|[S2DS 作者库](https://github.com/ben-z-original/s2ds)，[作者论文](https://ben-z-original.github.io/2022_Benz_DetectionStructuralDefects.pdf)|许可、表1/2、采集/拆分偏差和line-tolerant指标；不再分发样本|
|D4|[dacl10k v2 工具](https://github.com/phiyodr/dacl10k-toolkit)，[WACV论文](https://openaccess.thecvf.com/content/WACV2024/papers/Flotzinger_dacl10k_Benchmark_for_Semantic_Bridge_Damage_Segmentation_WACV_2024_paper.pdf)|v2定义/许可与v1论文表3/4区别；AWS403|
|D5|[CiF论文v1](https://arxiv.org/html/2605.18413v1)，[IBM仓库](https://github.com/IBM/cracks-in-the-foundation-dataset)，[官方HF](https://huggingface.co/datasets/ibm-research/cif-dataset)|论文§3/4、表1/3；API split/size；2组train_full，非聚合总行数|
|D6|[OmniCrack作者库](https://github.com/ben-z-original/omnicrack30k)，[CVPRW论文](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.pdf)|表2去重、表3规模、表4 clIoU4px；源码许可不等于数据许可|
|M1|[U-Net论文](https://arxiv.org/abs/1505.04597)，[原作者项目](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)，[milesial实现](https://github.com/milesial/Pytorch-UNet)|原实现是Caffe；PyTorch库是第三方；训练入口及GPL|
|M2|[SegFormer论文](https://arxiv.org/abs/2105.15203)，[官方代码](https://github.com/NVlabs/SegFormer)，[NVIDIA权重卡](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)|方法、旧依赖、非商业代码许可、另行核对权重卡|
|M3|[SAM2官方](https://github.com/facebookresearch/sam2)，[论文](https://arxiv.org/abs/2408.00714)|2.1 tiny配置、点/框API与AMG、训练入口、Apache2，不能推成文本接口|
|M4|[SAM3官方](https://github.com/facebookresearch/sam3)，[HF模型](https://huggingface.co/facebook/sam3)，[许可](https://github.com/facebookresearch/sam3/blob/main/LICENSE)|文本接口、环境/授权要求；3.1发布记录与本次调查的SAM3要分别记版本|
|M5|[YOLO11官方](https://docs.ultralytics.com/models/yolo11/)，[分割任务](https://docs.ultralytics.com/tasks/segment/)，[许可说明](https://www.ultralytics.com/license)|yolo11s-seg具体候选、实例输出、AGPL/企业条款；不随新名称自动换候选|
|L1|[CT-CrackSeg论文](https://arxiv.org/abs/2302.11728)，[作者代码](https://github.com/HqiTao/CT-crackseg)|DRB/BAM、数据转换、训练/测试入口、checkpoint链接。一次错误ID2302.11707已排除，访问日志保留纠错|
|L2|[CrackSAM论文](https://arxiv.org/abs/2312.04233)，[DOI](https://doi.org/10.1016/j.conbuildmat.2024.136573)，[作者代码](https://github.com/KG-TSI-Civil/CrackSAM)|adapter/LoRA，Facade390许可；checkpoint目录存在，权重条款待清楚，不将SAM原权重许可自动外推|
|L3|[EfficientCrackNet WACV论文](https://openaccess.thecvf.com/content/WACV2025/papers/Zim_EfficientCrackNet_A_Lightweight_Model_for_Crack_Segmentation_WACV_2025_paper.pdf)|表1/2及消融；0.26M是自身参数，正文0.61M是相对差值。未确认作者实现，不用同名第三方代替|
|L4|[MixerCSeg CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.html)，[作者代码](https://github.com/spiderforest/MixerCSeg)，[全文](https://arxiv.org/html/2603.01361v1)|TransMixer/方向边缘/融合；安装、权重、第三方CUDA内核条件|
|L5|[clDice CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)|骨架-区域度量与先行拓扑损失；不能直接声称裂缝任务新颖|
|L6|[Boundary loss MIDL2019](https://proceedings.mlr.press/v102/kervadec19a.html)，[作者代码](https://github.com/LIVIAETS/boundary-loss)|边界损失先行研究；医学域结果不当外墙结果|
|L7|[Crack-EdgeSAM原文](https://arxiv.org/abs/2412.07205)|YOLOv8自动框+微调EdgeSAM先行路线；只读摘要，细表/代码许可未验证|

网页检索失败与本地HTTP成功分别记载；任何缺失下载/许可/性能一律保留unknown。
