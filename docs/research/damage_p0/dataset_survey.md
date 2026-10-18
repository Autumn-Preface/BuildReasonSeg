# Damage-P0 数据集调查

核查日：2026-10-19。结论为候选建议，不是 P1 数据冻结。来源与访问记录见
[引用索引](sources.md)、[访问日志](../../../evaluation/damage_p0/source_access.json)及
[抽样日志](../../../evaluation/damage_p0/sample_access.json)。数字标为“论文”时未经全库本地复核。

## 候选比较矩阵

|数据/版本与原始来源|场景、模态、规模与分辨率|标注、类别、官方划分|许可证与获取状态|建议用途、限制|
|---|---|---|---|---|
|[CUBIT-InSeg / 2026 release](https://github.com/BenyunZhao/CUBIT-InSeg)，Zhao 等，AIC，doi:10.1016/j.autcon.2026.106980|UAV RGB 建筑外立面；论文 6996 图、62187 实例；数据卡称 4800×3200。**实测样本含 3000×4500 和 4800×3200 两种尺寸**|裂缝/剥落，实例多边形及框；论文 train/val/test=5596/699/701；训练 ZIP 目录实查 5596 图、5596 标签；本地读到 YOLO 归一化多边形 TXT|官方 README 明示数据 CC BY 4.0；Google Drive/Baidu。范围读取成功，无全库下载|**首选训练候选，附准入条件**：核实类别 ID、版本、建筑/序列 group 和跨 split 近重复；不能宣称已具备无泄漏测试集|
|[Facade390](https://github.com/KG-TSI-Civil/CrackSAM)，Ge 等，CBM 2024，doi:10.1016/j.conbuildmat.2024.136573|建筑外立面裂缝 RGB；作者以 Facade390 发布；本次未全库确认 390 计数及尺寸分布；拍摄平台未证实全为 UAV|裂缝像素分割；仓库有 lists_facade390/test_vol.txt，作为作者外部测试使用；完整 train/val/test 组织待获取核查|作者明确 Road420/Facade390 为 CC BY 4.0；Drive 链接存在；未抽取图/GT，不声称下载成功|外立面裂缝跨域候选；只可评价 crack，spalling 未标注不能当背景真值；与 CrackSAM 训练源重叠须查|
|[DeepCrack / Liu 2019 archive](https://github.com/yhlleo/DeepCrack)，Neurocomputing，doi:10.1016/j.neucom.2019.01.036|多场景 RGB 裂缝；作者手工标注；ZIP 目录 **300 train+237 test**（排除目录条目）；3 组样本均 544×384；全库尺寸分布未扫描|二值 PNG，样本值 {0,255}；无官方 val，P1 须从 train 按源组留 val。这里不是另一篇同名 DeepCrack|作者限制非商业研究/教育用途；原始图版权部分归作者；公开 67,708,808-byte ZIP；只用范围读抽 3 组|裂缝跨域或辅助训练；非 UAV 外立面专库；若采用 OmniCrack/CrackSeg9k 预训练，不能直接视作独立外测|
|[S2DS](https://github.com/ben-z-original/s2ds)，Benz & Rodehorst，GCPR 2022，doi:10.1007/978-3-031-16788-1_21|真实巡检，DSLR/手机/UAS 混合；RGB，743 个 1024² 图块，来自更大原图|背景、crack、spalling、corrosion、efflorescence、vegetation、control point；像素掩码；563/87/93（论文表1）|仅学术使用，**不得分享/再分发数据**；Drive；本次不复制样本到公共证据|双损伤外域候选。测试偏向可辨认图、训练含较差质量图，分布有选择偏差；同一原图/巡检场景分组仍需核实|
|[dacl10k v2](https://github.com/phiyodr/dacl10k-toolkit)，Flotzinger/Rösch/Braml，WACV 2024|桥梁 RGB 实际巡检，9920 图，多尺寸；不是纯建筑外立面，也不能认定全 UAV|LabelMe-like JSON 多边形，**多标签像素可重叠**；v2 13 damage+6 object；train6935/val975/testdev1012/testchallenge998；测试 GT 不在公开 dev 目录|数据 CC BY-NC 4.0；AWS dev ZIP HEAD 返回 HTTP403，本次未取到原图+JSON；官网工具/示例图不冒充原始数据检查|双损伤跨域候选；原始论文 v1 的18类/划分和 v2 19类不能混记；隐藏测试需官方服务，val不能一边调参一边充独立测试|
|[CiF / official HF default](https://huggingface.co/datasets/ibm-research/cif-dataset)，Farronato 等，arXiv:2605.18413v1，2026|欧洲基础设施，含桥梁、现场与 UAV 受控采集；RGB。论文：12896 full、148642 tiled；原生 0.23–42.18 MP，平均16.66 MP；tile1024²。HF聚合展示行数不可当独立原图数|6类：Algae/Crack/Net-Crack/Crack with Precipitation/Rust/Spalling；COCO-style 实例多边形。full9096/1942/1858；tiled105139/22368/21135，API与论文一致|官方数据卡 CDLA Permissive 2.0；API download_size=122,957,087,885 bytes；只取2个 train_full 配对样本，均3024×4032|很有价值的跨域候选，**不全量下载**。full/tiled是同源两种视图，不是独立数据集；论文承诺源图分割，不等于建筑/航线严格隔离|
|[OmniCrack30k](https://github.com/ben-z-original/omnicrack30k)，Benz & Rodehorst，CVPRW 2024|20子集，多材质 RGB 裂缝；论文表3总30017（22158/3277/4582）；多尺寸；包含UAV75、S2DS、DeepCrack等|重新组织/统一裂缝标签；包含线与面积标注差异；作者分析集合重复并去重|代码 GPL-3.0 与**数据许可证不同**；数据需高校邮箱申请并遵守各子集条款；本次未申请、未下载|辅助预训练/重复风险研究；不作为与所有组成子集独立的外测库。不能把代码 GPL 当全体数据许可|
|[CODEBRIM original](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html)|桥梁混凝土，部分UAV；RGB缺陷图块|原始任务为多标签**分类**，含crack/spalling等；原始类别标签不等于像素GT|本次未核实完整许可/下载，不使用数据|不作为像素主基线；不要与另行像素标注的 SegCODEBRIM 混淆。其余计数/划分待需用时查|

CUBIT 和 CiF 可同时支持两类损伤；小型裂缝库不能弥补剥落监督缺失。桥梁上的剥落和
外立面涂层剥落是否同义，尚不能由数据集名称推出；必须经标签定义和抽样确认后再映射。

## 抽样方法与可复核发现

脚本：[acquire_samples.py](../../../reports/damage_p0/20261019/acquire_samples.py)、
[add_sample_evidence.py](../../../reports/damage_p0/20261019/add_sample_evidence.py)、
[inspect_samples.py](../../../reports/damage_p0/20261019/inspect_samples.py)。
从训练资源确定性选取排序靠前样本；CUBIT 为覆盖两标签补取 SP0 前两项。不是随机代表样本，
也不用于任何性能估计。原图/标注保存在 `datasets/damage/local/`（Git忽略），逐文件SHA256见
[sample_checks.json](../../../evaluation/damage_p0/sample_checks.json)。全部9组图像可读取、尺寸可解析、
坐标未越界、掩码配准无明显整图错位；PIL 多边形栅格化的边界包含规则只是检查器口径，
P1 必须锁定官方/COCO兼容转换器并用人工小例核对，不能把该检查器当正式评价实现。

|样本|人工可视检查（执行者观察，不是独立标注审查）|局限/风险|
|---|---|---|
|CUBIT CR1_1、CR1_10|可见竖向裂缝与class0多边形基本对应；原生256px裁剪能看到裂缝与边缘。前景约1.01%/1.12%|两个相近视角的同一外墙；局部标签有外扩，细边界非逐像素绝对标准；尚未跨split寻找此场景|
|CUBIT SP0_1、SP0_10|class1覆盖墙角表面剥落，约4.88%/4.91%；两图均4800×3200|相同破损墙角近视角；裂缝状纹理在剥落区域内部，是否重叠标裂缝需要指南，不自行补标|
|DeepCrack 11111–11113|3组裂缝与二值GT主干对应；像素值0/255；544×384|多为路面近景，材质/尺度不代表立面无人机；细分支与边界宽度存在主观性；水印/低清区域可能带偏差|
|CiF IMG_8896、IMG_0345|多边形类别2/6对应裂缝/小剥落；能解析成双通道掩码|可见复杂背景；IMG_8896宽黑裂隙部分边缘与标注宽度不一致，不能把所有黑色当裂缝；部分细区域标签窄于可见裂隙|

图像证据：[sample_evidence](../../../reports/damage_p0/20261019/sample_evidence/README.md)。
总览是缩略显示，另带同位置原生256px裁剪；没有模型预测、伪标签或手绘替代GT。
512²强制缩放后的面积比例仅作解析诊断，**不能证明细分支保持或模型可分割**。

## P1 数据准入条件（待 Supervisor 冻结）

1. CUBIT确认发布版本、全部图尺寸分布、类别ID/标签指南、官方split及原图/建筑/航线映射。
   0→crack、1→spalling目前仅由文件前缀与抽样视觉支持，未取得权威class YAML。
2. 对拟用 split 建立原始文件hash、source_image_id、building_id、sequence_id、tile_parent；
   exact hash + 感知相似检索 + 人工复核相近视角。无group信息不得声称建筑级无泄漏。
3. 若官方划分有重叠，保留官方复现轨与独立分组研究轨，提交重分方案审查，不静默改官方GT/split。
4. 所有跨库测试先排查训练/预训练源重叠。外测只报告已标注类别；未知标签用ignore，不默认阴性。
5. S2DS不进入公开GitHub证据图；dacl访问失败单独记录；CiF按所需full或tiled单份管理，
   不同时下载两者制造重复；任何后续完整下载前核算空间。
