# Damage RGB Visible Segmentation Protocol v0.1

状态：**DRAFT_PENDING_SUPERVISOR_REVIEW**，不是冻结P1配置。2026-10-19。
以下是任务书授权的协议设计，不是旧RC1阈值修改，也不授权执行训练。
P1开始前需Supervisor批准数据版本/split、候选与预算及本草案；所有TBD填妥才可锁测试。

## 1. 任务、标注与转换

输入单张RGB外立面图像；输出原生网格上两个独立二值mask：crack、spalling。
**主要指标是每类像素分割**，并不测结构安全或不可见内部损伤。背景为相应已标注类的非前景，
其他损伤按原库定义保留；裂缝与剥落不合成单个damage类充当主指标。

|数据|拟议映射|不可直接统一处|
|---|---|---|
|CUBIT|class0→crack，class1→spalling，**待官方ID说明确认**；同类实例取并集|实例原ID保留供AP；跨类重叠不靠强行优先级改GT|
|S2DS|crack/spalling分别为前景，其他已完整标注类属于各二值任务的负像素|control point非损伤；涂层/混凝土剥落定义需核查；不传播其他类的模糊语义|
|dacl v2|Crack→crack，Spalling→spalling|ACrack不自动合并为Crack；Rockpocket/Hollowareas/Cavity/WConccor不合并为Spalling；原始多标签重叠保留|
|CiF|2→crack、6→spalling|3 Net-Crack、4 Crack+Precipitation不自动合入2；如后续扩为“all crack”，另设审定任务版本和报告|
|DeepCrack/Facade390|只有crack已标注；spalling整通道ignore|未标注剥落≠没有剥落；不得把它们用于两类mIoU的缺失类补零/补一|

像素标签、线标签、polygon、instance不视作等价。Polygon用锁定版本的COCO/官方转换器栅格化，
记录坐标原点、pixel-edge/inclusion约定、孔洞/RLE/crowd/ignore。二值PNG先查正负值，
禁止凭颜色猜ID。训练mask几何变换最近邻；图像双线性。评价GT不resize、不膨胀、不补裂缝。
线标签只作单独线指标轨，不能膨胀成面积GT并与原面积IoU混比。
本次P0 PIL解析仅验证可读与大致配准，不作为P1正式转换器。

若确证标签互斥，可另报告背景+两损伤三类softmax任务；其mIoU明确包含背景，
不与本协议双通道前景mIoU混在同列。二分类仅适用于单一已标损伤或单独每类任务。

## 2. 数据清单与泄漏控制

先可靠官方split；记录版本、许可、SHA256和原始来源，统计单位为**原图/建筑group**，不是图块。
至少保存：dataset_version, file_sha256, native_shape, official_split, source_image_id,
building_id, sequence_id, tile_parent_id, annotation_version, known_class_channels, group_confidence。
采样图块全部在split之后生成；同原图/建筑/连续航线及重叠图块不得跨集合。

跨split exact hash、相似图检索与人工相近视角核查。缺group不能标“无泄漏”；若官方split不可靠，
保留官方复现轨，另提group70/15/15、seed20261019的研究轨，按组分配并检查两类覆盖。
这只是划分建议，**未实际重分任何数据**；建筑组少时采用leave-one-building-out建议，需新审查。
只有train/test的库从train内留验证组；本来作为外测的库不新增用其test调参。

train负责梯度、增强统计和面积分层边界；val负责有限超参数、checkpoint、固定词表/自动提示规则；
test只在全部冻结后一次最终报告，不用于调阈值、调提示、挑种子或筛模型。P0已查看的训练样本不移作独立测试。
外测不进行目标域微调；所有预训练库与外测库先排查成员重叠，不能确认时标“独立性未建立”。

## 3. 三组输入条件

|组别|输入/固定规则草案|报告与成本|
|---|---|---|
|A 自动|RGB；U-Net、SegFormer-B0、YOLO11s-seg候选；没有人工测试位置、没有GT提示|主比较。全图上每类IoU等；实例模型并集后像素评价，另附实例AP。检测漏检全部计FN|
|B-human 真实人工|看RGB的标注者每图最多5个框（两类合计），0次模型反馈修正；先记录类、框、用时和漏定位，再推理；不看GT|人工预算与分割质量同报，不能称全自动。多标注者/盲性/顺序在P1计划中固定；未实例完备提示的漏目标仍计FN|
|B-auto 自动提示|只从train/val训练/选择定位器；建议复用YOLO11s-seg boxes，固定conf0.25/NMS0.70、每图总最多5框；按score降序，稳定tie按类ID及坐标；不读test GT|SAM2.1-tiny接受同一规则boxes；计入检测、预处理、编码、后处理全部时间和显存；定位漏检计FN。参数是待审草案值，不是调参结果|
|B-oracle 诊断|由GT每个已标实例生成tight box，一实例一框，**提示数量随GT变化且完整暴露目标位置/数量**；无扰动、无后验挑选mask；记录GT版本/框数|单独标Oracle，不称自动/公平A基线；是信息上限条件的诊断，不能保证数值必然上界。单独tight-box与点实验不可混列；点版暂不强制|
|C 文本预留|RGB+固定文本/语言/模板；SAM3实际支持set_text_prompt，SAM2不支持；未来可加MLLM但本阶段不实现|固定如英语“crack”“spalling”的候选词，最终词表由train/val与Supervisor冻结；不看测试图结果逐图改词。与A/B分表；记录文本和外部模型预算|

SAM2自动mask generator是无语义候选生成，未解决损伤类别识别；增加任何筛选器必须登记监督与预算。
B各设置输入信息不同，不共享一个总排行榜。每个prompt的multimask只用模型自身最高预测质量输出，
不得拿GT IoU选择最佳mask；自动框NMS/上限是系统协议条件，任何改变都先改版本后复核。
若改用点提示，另冻结数量、正负点、生成规则和人工预算；目前不把GT点模拟称真人操作。

## 4. 指标、阈值与空掩码

令每个类在有效原图像素域的TP=|P∩G|、FP=|P\G|、FN=|G\P|，ignore像素不计。

- IoU=TP/(TP+FP+FN)；Dice/F1=2TP/(2TP+FP+FN)。在相同二值统计下Dice=F1。
- Precision=TP/(TP+FP)；Recall=TP/(TP+FN)。先累计全测试有效像素再算每类（micro-over-images）。
- 主mIoU_fg是固定评价任务内已标注类的各类全局IoU等权平均；不含背景。两类任务必须两类均有支持，
  若某类整个split没有正样本，主两类mIoU标NA并补报单类；不同类支持的mIoU不直接排名。
- 类间像素微平均另标micro-all-classes（大类主导），不代替逐类表。按原图宏平均另报，给出有效图数。

固定二值决策草案：语义独立logit经sigmoid≥0.5；YOLO实例conf≥0.25，mask≥0.5，
若该pin模型使用NMS则IoU0.70，max_det不低于100并记录实际实现（B-auto额外限5prompt）；
所有设置须P1前锁定，不进行测试阈值寻优。原生网格上先重建概率再二值。
实例AP可按官方COCO连续confidence排序另算，不等于像素阈值指标。

|单图/类情况|IoU、Dice|Precision|Recall|纳入方式|
|---|---|---|---|---|
|GT空、预测空|NA|NA|NA|不人为补1刷宏平均；记empty-correct计数；全局TP/FP/FN贡献0|
|GT空、预测非空|0|0|NA|计FP、宏IoU=0，并记negative false-positive image rate|
|GT非空、预测空|0|NA（无预测分母）|0|所有漏检计入IoU/Dice与全局Recall；Precision不偷偷补1|
|类通道未标注|NA|NA|NA|该通道所有像素ignore，报告known/ignored图数|

主要每类指标由全局TP/FP/FN计算；分母0则NA。原图宏IoU/Dice排除仅GT空且预测空的图，
同时报告分母和负样本FP率（负GT图里预测非空比例）及平均FP面积比例，防止仅排空图的偏差。
每图CSV公开混淆统计/支持数，可追溯宏微口径；不把批平均当整库指标。

### 小损伤与附加结构指标

面积分层边界**TBD**：在train原生GT实例面积占有效图面积比例的非空分布，分别按类计算Q25/Q75，
small≤Q25、medium(Q25,Q75]、large>Q75；分位数插值规则linear、相等边界可合并并登记。
数值、支持数与hash在P1测试前冻结，绝不从test分布/结果回调。只有语义GT的库改用8连通域作为
明确标注的proxy实例，不与真实例混报。train样本过少/分位数退化时停止该分层并提案，不随意换阈值。

小损伤像素Recall_small=同类所有small GT实例有效像素**并集**内被预测覆盖像素数/该并集像素数，
同类重叠实例不重复累计像素；
实例命中率为small实例中GT覆盖率≥0.5的比例，阈值待审固定0.5；不让一小段命中算完整检测。
同时报告对应原图的类Precision/FP面积，避免单用Recall奖励整图预测。实例跨tile在原图重组后统计。

附加建议（P1可选，未实现）：BoundaryF1用原生二值mask的8邻域内边界
`B(M)=M minus erode_3x3(M)`（外域视背景）；预测/GT边界像素在欧氏距离≤2px内各自匹配，
BF_P=匹配预测边界/预测边界、BF_R=匹配GT边界/GT边界，BF1为其调和平均。
无两边界时NA，仅一侧有边界时BF1=0；两侧边界均非空而BF_P+BF_R=0时BF1=0，不能作为NA剔除。
记录原生2px不等于物理2mm，不因结果差扩大容忍。
可附1/2/4px敏感性表且全部展示，不能选择最有利列为主结果。

连接性候选clDice=2*Tprec*Tsens/(Tprec+Tsens)，
Tprec=|S(P)∩G|/|S(P)|、Tsens=|S(G)∩P|/|S(G)|，骨架算法/版本、8邻接先锁定；
空域规则沿用主IoU/Dice；P/G均非空而Tprec+Tsens=0时clDice=0。
非空mask经骨架算法却得到空骨架时单列实现异常并停止该指标交付，不静默排除。
需要另看误连桥与断裂，clDice高不证明裂缝物理连续。
OmniCrack的clIoU4px是另一指标，不混名。背景复杂度分层需train/val定义可盲标的材质/阴影/
接缝/低对比因素，test标签在冻结指南下盲标，不根据模型失败挑困难例；当前无强制复杂度阈值。

## 5. 分辨率、增强、监督与预算

以下为可审查的默认方案，不以公平口号掩盖资源差异。

- 主能力轨：train原生512²图块，先按原图均匀取样再随机crop，保留无损伤区域；不足512先pad，
  pad标签ignore。每原图每epoch最多4crop、seed清单固定，不能只保留正块。每模型同一crop清单。
  eval用512²窗口、128px overlap、原图边界锚定末窗；pad只算有效区；重叠概率等权平均后裁回原图。
  实例模型若不能等价输出密集概率，则该能力轨不宣称同条件，用下面系统轨比较，差异明确登记。
- 系统轨：U-Net/SegFormer按上述tile；YOLO11s-seg默认640²letterbox，SAM2按官方1024²编码。
  另提议YOLO原生tile系统做分辨率控制对照，**不在P0实施**。resize整图与tile是两条不同预算轨，
  包括tile数、总处理像素、延迟；不能把高分直接归因架构。
- 增强草案：train水平/竖直翻转各p0.5，90°倍数旋转等概率；亮度/对比度±0.2、p0.3；
  不做抹掉裂缝的强模糊、形态学改GT或mosaic/copy-paste默认开启。val/test无随机增强、无TTA。
  若框架不能关掉默认增强，记录并归系统轨；不能假装条件统一。
- 预训练逐权重记录source/hash/license/预训练数据。首轮U-Net scratch、SegFormer ImageNet、YOLO COCO、
  SAM2通用基础权重是不同监督条件；诊断性结果不是纯架构归因。匹配监督scratch轨是否追加另审。
- 预算建议：每A路线最多60epoch、effective batch8；microbatch据显存等效累积；
  每路线最多2个LR候选、最多12GPU小时用于配置试验，随后3个独立种子各≤12GPU小时，
  总上限约48GPU小时/路线、3路线144GPU小时（规划上限，不是耗时估计或P1授权）。
  若上限内不足60epoch，记录完成epoch/step，另报等预算轨，不能称相同收敛预算。
- 优化器草案：U-Net AdamW；SegFormer AdamW；YOLO先官方SGD/AdamW中**固定一种待Supervisor确定**。
  各路线2个预注册LR（如U-Net/SegFormer1e-4、3e-4；YOLO1e-3、1e-2）只看val；
  cosine schedule、5%steps warmup、weight_decay0.01为语义候选值；YOLO如沿官方默认须独立记录。
  各自合理训练recipe是系统公平，严格同优化器轨是另一个问题。
- 固定seed20261019/20261020/20261021；数据划分seed与训练seed分开存；保存Python/numpy/torch/CUDA种子、
  cudnn/deterministic/TF32/AMP/编译设置和版本。不能宣称跨硬件bitwise复现。
- checkpoint按val两类mIoU_fg最高选择；并列选更早epoch，再更早step；单类任务用crackIoU；
  不按test或人工看test图选择。验证每epoch一次，训练损失不充当最终模型选择标准。

## 6. 报告、统计与性能测量

每库/每组/每分辨率轨单表，列逐类TP/FP/FN、IoU/Dice/P/R、支持数、empty/ignore数、小损伤召回、
每种子结果及3种子均值±样本标准差（ddof1）。置信区间建议以建筑/源图group bootstrap1000次、
固定seed20261022取2.5/97.5分位；无法恢复建筑group时仅称源图级区间，不装作独立建筑样本。
不把不同数据集的IoU/F1或不同输入组排序为同一个优劣结论。

推理统一本机GPU、batch1、eval/no_grad、记录精度；固定20图warmup、100图正式测量（不足100用全部，
重复次数与唯一图数均报），CUDA synchronize或events围住计时区间。
分别报纯GPU网络时间、含预后处理的全图系统时间、含磁盘读取端到端时间；prompt编码/自动检测/
tile聚合计入相应系统时间，人工用时另报。CUDA events不冒称包含CPU时间。
报均值/中位/P95、原图/模型输入尺寸、tile数；模型load耗时与冷启动另报，不用官方A100 FPS当本机值。
`reset_peak_memory_stats`后记录allocated/reserved峰值与nvidia-smi环境背景；训练峰值另表。
先独占或记录并发桌面负载，不能把随机一次显存读数当模型峰值。

P1每run保存不可变run_id、config、代码commit、环境lock、数据manifest/split hash、权重hash、
预测概率/二值输出的重建规则、每图统计CSV、计时JSON。测试结果只在冻结commit上运行。
本草案没有正式评价程序；转换器/指标实现及合成边界例验算属于获准P1的工程准备。

## 7. 冻结前待审事项

CUBIT版本/类ID/建筑序列组；外测选择与许可；3–4路线执行集合；实际训练时限；
优化器及LR清单；像素能力轨vs系统轨；prompt5框预算；small分位数数值；附加指标是否必跑；
图像/掩码转换器和重叠处理。未完成这些事项不启动训练或偷偷填入更有利默认值。
