# BuildReasonSeg 中期检查实验成果素材审查

项目：空间推理引导的建筑物结构 MLLM 相关推理分割方法。

审查日期：**2026-10-17（Asia/Shanghai）**。状态：**证据整理完成，待ChatGPT与用户共同审查**。

本次整理出 **10张历史原始图片的逐字节副本 + 2张根据已有实验记录重新绘制的图表**，共12项可供选择；原始图片没有裁剪、修改或覆盖。总览、CSV清单、图注、使用建议和来源核查均在本目录。

**核心结论：已有证据可以支撑一份诚实、可追溯的阶段性研究附件，内容为遥感建筑实例分割baseline、数据构建/质量控制、空间关系中间量、冻结实验对照与原型系统开发。不能据此声称已经完成建筑外立面损伤识别/损伤分割，也不能声称当前自然语言端到端目标身份选择已可靠解决。**

## 一、交付入口与目录

- [素材总览](contact_sheet.png)：编号定位用，不作为精读原图。
- [完整清单、逐图中文图注及限制](evidence_manifest.csv)：UTF-8 BOM CSV；包含完整来源路径、尺寸、日志、用途、SHA256及同样本核查依据。
- [PPT使用建议](ppt_recommendations.md)：建议的六页成果段落和可用措辞。
- [缺失与不宜展示的证据](missing_evidence.md)：明确哪些需要用户补充，哪些只能由既有数据重绘。

```text
reports/midterm_evidence_review/
├── README.md                         本审查报告
├── evidence_manifest.csv             12项清单：10原图+2重绘
├── selected/                         E01—E10；原始素材未修改副本
├── derived/                          E11/E12；显式标记已有记录重绘
│   └── formal_metrics_source.csv     两种Reference、全部三seed、原均值/SD
├── contact_sheet.png                 带编号总览
├── ppt_recommendations.md
├── missing_evidence.md
├── source_records/                   外部Demo小型原始记录/原CSV与配置副本
├── audit/                            清点、来源、排除项、保护基线与校验结果
├── review_audit.py                    只读清点/完整性校验
└── build_review_pack.py               复制及绘图生成流程，不加载模型
```

## 二、检查范围与方法

### GitHub与分支

已访问用户指定的[GitHub仓库](https://github.com/Autumn-Preface/BuildReasonSeg)，并成功fetch origin。当前审查从实际干净HEAD `e06777b0a0844c5c29325701eb2e819bd193ff04` 创建独立分支 `codex/midterm-evidence-review-20261017`；没有更新主分支、合并历史分支或改写历史。

fetch时默认main为 `57b368d5647e842d8f31d6d1a9997bf1df3cc0fb`，Task8F远端为上述e06777b。扫描了 **48个remote refs（含origin/HEAD）** 的树，得到 **200条唯一图像路径**。后期Task8C/D/E/F资料在相应分支，不把默认main的早期README当作最新验收结论。清单中repo素材/记录使用不可变commit链接；本地独有Demo快照说明本地来源，并将未改图片与对应小型记录保存到本分支。

### 实际定位的本地根目录

| 实际路径 | 请求扩展名图像路径数 | 范围说明 |
| --- | ---: | --- |
| `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg` | 25,864 | 项目/历史工程/交付目录实际存在；路径与header清点 |
| `C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment` | 927 | 项目/历史工程/交付目录实际存在；路径与header清点 |
| `C:\D\resources\project\WHU_Building_Segment` | 927 | 项目/历史工程/交付目录实际存在；路径与header清点 |
| `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1` | 1,066 | 项目/历史工程/交付目录实际存在；路径与header清点 |
| `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V1` | 108 | 项目/历史工程/交付目录实际存在；路径与header清点 |
| `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Demo_V2` | 48 | 项目/历史工程/交付目录实际存在；路径与header清点 |

以上合计 **28,940个图像文件路径**，不是28,940项独立成果；包含大量GT/component缓存、数据标签与重复副本。没有把数据文件数当作实验次数或成果数量。检索PNG/JPG/JPEG/SVG/WebP，读取图像header，清点范围内未发现损坏header。大权重、环境、local_cache与model目录不参与图像遍历；不读取或反序列化checkpoint来做预测。

补充定位了 `C:\D\resources\建筑物语义分割`、`C:\D\resources\挑战杯中期审查`，按请求图像扩展名未找到独立图片，只有既有PDF/Office材料。只查看目录与图像路径，没有编辑或拆取Office嵌入媒体。未进行整盘或无关私人目录扫描；因此不能断言其他位置绝无截图。

用户列明的六组起点均实际存在并检查：`baseline/yolo_whu/plots/`、`evaluation/relation_samples/`、`evaluation/task6d_panels/`、`evaluation/task6k1_samples/`、`evaluation/task6l_samples/`、`evaluation/build_spatial_reason_v0.1.1_samples/`。其余重点为Task7I/J记录、Task8C/D审核图、真实原型截图，以及external RC1、Demo V1/V2保存的输出。

### 核查强度

1. 范围内文件路径/header清点与手工内容审查分开：并未声称逐张人工检查全部28,940路径。10张入选原图均已实际打开检查；两张新图和总览也已渲染目视复核。另抽查了小尺寸/拥挤/失败等候选，见[排除记录](audit/rejected_candidates.csv)。
2. 从图片回查JSON/CSV、训练args、生成/拼图脚本、样本stem、冻结case/run ID及文件SHA，不按视觉相似度猜来源。入选repo原图字节与e06777b对应Git blob相等。
3. 本地Demo选择**两个既有V2 run中的首个时间顺序记录**，不按表现选优。E07—E10来自同一 `20261012_155017_1034`。24条事件/12个预览PNG的序号、run ID、尺寸和hash均核验；现有输入1034.tif的RGB像素hash与Stage2保存摘要完全一致；summary/trace/engine run root一致，最终Mask/Overlay与engine原文件hash相同。详见[Demo来源核查](audit/demo_provenance.json)。这证明保存记录内部可追溯，不是新的模型重现或独立GT正确性证明。
4. E04由已有Task8C原图/GT轮廓/保存输出生成，样本1008及审查记录明确；E02/E03由同一stem/tile_id拼接，脚本/索引说明面板含义。E03两幅彩色前景均为GT核对，绝非预测/GT对照。
5. Task7I的六个冻结checkpoint均仅以文件字节/hash核对存在性，与Task7J manifest一致；**没有torch.load、推理或重新选权重**。据此可说明已保存训练产物，不能由checkpoint存在单独推出性能。
6. 所有科学数字仅引用已有记录。E11逐列读取100行CSV，E12核对全部三seed与原mean/sample SD。没有模型调用、训练、阈值修改、新测试评价或已消费测试上的模型选择。

## 三、精选清单：每张图能证明什么

| 编号 | 图片 | 可证明的内容 | 使用边界 |
| --- | --- | --- | --- |
| E01 | [YOLO历史训练原图](selected/E01_results.png) | 100轮YOLO训练/验证原曲线与配置、CSV确实保存 | 历史YOLOv8m-seg，val而非test；不等于MLLM效果 |
| E02 | [原生矢量与伪实例GT审查](selected/E02_1_0_panel.png) | 开展过native-vector与pseudo实例GT的差异审查 | 五列是数据/GT，不是模型预测 |
| E03 | [矢量GT缓存一致性原图](selected/E03_1_0_overlay.png) | v0.2 native GT迁移及缓存核对有已有图/审查记录 | 绿/蓝都是GT；不能说预测IoU=1 |
| E04 | [正式Demo目标错误对照](selected/E04_above_review.png) | 真实正式运行可被GT后验检查，并暴露目标错误 | 失败案例，不能当正例或损伤结果 |
| E05 | [空间权重与attention诊断](selected/E05_04_W_and_A.png) | 空间场和attention等中间量有真实保存诊断 | 不证明最终正确或某模块是错误根因 |
| E06 | [真实原型界面空载截图](selected/E06_task8f_final_results.png) | 本地两页GUI与交互控件已经实现 | 真实空载界面，非模型推理完成截图 |
| E07 | [真实运行自动Reference快照](selected/E07_stage_03_event_010_1.png) | 真实run保存了候选和automatic Reference ID9 | Reference未做GT身份验证 |
| E08 | [真实运行四种空间场快照](selected/E08_stage_06_event_018_1.png) | 同run产生了P_dir/P_near/W/A及范围说明 | 颜色不是目标正确率 |
| E09 | [真实运行D-B1候选快照](selected/E09_stage_08_event_022_1.png) | 同run的D-B1候选概率/二值图/context可追溯 | 中间候选；不能等同已接受目标 |
| E10 | [真实运行最终结构输出](selected/E10_overlay.png) | 同run有最终保存的结构输出 | SUCCESS结构含义，语义NOT_EVALUATED，无损伤类别 |
| E11 | [历史训练CSV重绘摘要](derived/E11_yolo_record_redrawn.png) | 用真实100轮CSV更清楚地呈现损失与val指标 | 显式重绘，不重新训练或选择峰值权重 |
| E12 | [冻结测试对照重绘](derived/E12_formal_metrics_redrawn.png) | 完整呈现三seed、两种reference下历史对照与局限 | 显式重绘，不是新测试、全面提升或显著性证明 |

**正文优先：E02、E11、E07/E08、E12、E06、E04。** 这些图分别覆盖数据、训练、方法实现、客观对照、原型和问题认识。E01原曲线及E03/E05/E09/E10更适合成果附件。逐图可直接使用的中文图注已写入CSV，不需要另猜图片语义。

## 四、指标与解释核查结果

### A. YOLO历史baseline

100行epoch1—100与args.yaml存在；repo的results.png、results.csv、args.yaml与两个实际legacy目录同名记录逐项hash一致。最终val Mask mAP50 **0.80693**；历史最高 **0.84373/epoch44**，两者没有混淆。最终val Mask mAP50-95为已有manifest记载 **0.43388**。原CSV train/val sem_loss全为0，不应把平线当作新增语义监督训练成功。后期val分割损失回升、最终指标低于峰值均保留，不平滑掩盖。

现有903张预测JPEG不等于该baseline的正式test评价；其README/manifest明确没有正式test-set指标。它是早期 **YOLOv8m-seg**，与后续proposal path的 **YOLO26m** 不应混称；不是MLLM本身，也不能宣称建筑损伤类别识别。

### B. 数据准备与版本

v0.1.1质量记录为 **25,229条**，其scene-level disjointness为 **unverified**；“没有字节重复”不等于地理隔离。现成小图29张实际存在，但抽查的非平凡L3样例尺寸小且长指令截断，未列主精选。

后续v0.2 native-vector质量记录按split合计 **28,108条问答**（train12,778/val9,111/test6,219）；level3为3,016。其native-vector索引 **17,388个tile** 中含 **11,909空tile**；tile数不是问答数，也不是有建筑的样本数。40样本GT一致性核查是数据质量结果，不是分割模型在40图上IoU=1。版本、GT来源和划分口径均不能混用。

### C. Z-B3/D-B1历史冻结实验

Task7J的既有终局为 `FINAL_FROZEN_TEST_COMPLETE`，这是程序性完成，不是性能显著改善验收。测试数据已经消费，736条四类canonical L3记录，462个tile；成对指令集合274对。E12只重画全部三个seed的既有结果：

| 指标 | Z-B3 mean | D-B1 mean | 可用解释 |
| --- | ---: | ---: | --- |
| GT Reference mIoU | 0.337488 | 0.392304 | 已知正确参考的条件下，D-B1该项均值较高 |
| 预测Reference strict mIoU | 0.207942 | 0.210805 | 实际reference设置增量仅0.002863，不是大幅提升 |
| GT Reference paired pass rate | 0.849148 | 0.823601 | D-B1并非所有指标更好 |
| 预测Reference paired pass rate | 0.413625 | 0.316302 | 实际reference下该项下降，必须披露 |

误差线是原记录**三个seed样本标准差（ddof1）**，不是95%置信区间、不是重新进行显著性检验。Task7I原有practical增量gate未通过、结论 `DB1_FORMAL_VAL_NOT_CONFIRMED` 不变。冻结指标不是“全新未触碰test”或自然语言端到端准确率；不能删除不利面板，只展示oracle提升。

### D. 正式Demo、实际用户记录与系统截图

Task8C四例真实正式进程均runtime SUCCESS，但既有GT后验审核：**0完整chain identity match、1仅target identity mismatch、3 reference+target同时mismatch**；runtime-success样本的mean target IoU记录为 **0.073331**。E04清晰保留above target IoU=0.077622和TARGET_IDENTITY_MISMATCH。不是新推理、不是新的generalization测量，也不能由单图声称高质量成功。

另发现V2已有 **2026-10-12 15:50左侧查询** 与 **15:52下方查询** 两条保存run（summary时间分别是完成记录）；均有完整九阶段和结构SUCCESS，semantic NOT_EVALUATED。本次只读取，未在V2再点击分析，未重新打开GUI或创建新的科学输出。E07—E10来自首条同run；不能因已有output就宣称最近建筑身份正确。

E06是已存在的真实GUI空载截图；`docs/task8f_fake_*.png`明确是fake测试，已排除。当前有可用界面图及真实中间量，不必为了附件虚构截图。若希望展示正在运行的完整界面，仍建议用户额外提供真实结果页与过程页截屏，并保留run ID/时间。找不到的不是所有Demo素材，而是特定的完整运行现场GUI帧。

## 五、来源、可读性及不足

全部入选原图的绝对路径、源图/副本SHA256、尺寸和日志路径见CSV与[selection_records](audit/selection_records.json)。查阅事实文件清单及hash见[read_records](audit/read_records.json)，包括基线CSV/args、v0.1.1/v0.2质量记录、Task6K.1/6L面板源脚本、Task6D错误记录、Task7I/J训练/测试manifest、Task8C/D审核、Task8F界面验收和外部真实run记录。

`source_records/`保留来源文件的原始Windows CRLF等字节，不为格式检查改写历史记录。仅本审查目录的`.gitattributes`禁止对归档副本自动换行转换，并将CRLF视为合法行尾；10张原图及7份记录副本已核对Git暂存字节与来源副本一致。

总览是缩略定位图，不要求其小字可替代原图。E01原图4000×1200保留全部十八面板，正文用E11更易读；E02/E03没有图内列标题，必须保留中文图注；E08/E09是竖版，适合放大而非小角落。没有通过裁改预测、删除不利实例或去除失败标记提升外观。

本次没有找到可直接支撑**外立面损伤GT/损伤类别/损伤分割精度**的项目证据，也没有进行该方向的新数据检索或模型实验。当前主精选含清楚的失败/未验证示例，缺少“同一样本四联且已确认正确目标”的高分辨率正例。这不阻止阶段性进展附件，但限制模型成熟度和效果表述。完整缺失清单及“能否只用旧数据补绘”见missing_evidence.md；没有为填页面伪造图或补跑测试。

## 六、六项验收问题的直接回答

1. **已保留哪些直接可用真实图片？** E01—E10十张，分别是训练原曲线、GT差异/一致性、正式失败审核、中间场诊断、真实原型空界面和同run候选/场/分割/最终输出。E11/E12另为已有记录的透明重绘。
2. **哪些最值得放中期检查？** 数据E02、训练E11、实现E07/E08、客观评价E12、系统E06、问题E04；其余放附件。
3. **各图能证明什么？** 第三节表与逐项CSV均给出“可以证明”及限制；不把中间量、GT本身、runtime状态或测试PASS当科学正确性。
4. **哪些易误解、不适合展示？** fake UI；标注文字覆盖严重的baseline batch预测；小尺寸/截断样例；退化失败冒充正例；GT与GT比较冒充预测IoU=1；只截oracle提升图；抹掉NOT_EVALUATED的Demo输出。排除表保留具体路径与检查方式。
5. **是否要用户另供Demo截图？** 基本GUI与真实运行输出不缺。若正文需要带输入/结果/实时过程的完整现场GUI截图，建议用户补两张；本次不伪造或新推理。
6. **现有证据够不够有说服力的阶段附件？** **够**，可客观说明已经开展数据、训练、对照、诊断与系统实现；**不够**证明新方向损伤效果、高可靠最近目标选择、未消费测试泛化或全面显著提升。建议以“完成工作—记录结果—识别局限”组织，而不是包装成最终成熟系统。

## 七、执行边界与完整性

[完整性结果](audit/integrity_result.json)：**1450个原有tracked文件hash不变、111,308条既有文件元数据不变、10张原始副本源/目标hash一致**。元数据覆盖本次扫描范围，排除环境/大模型目录，不把它夸大为所有电脑文件的完整hash校验。科学checkpoint额外核查六个冻结best.pt均与原记录一致（仅读字节，无权重选择）。允许改变的既有文件只有CURRENT_TASK/EXECUTOR_STATE；它们的上一任务快照已归档，其余产物全部新增于本目录。

模型推理0、训练0、新阈值/模型选择0、历史实验/图片/指标/日志修改0、治理与项目产品代码修改0、Office文档修改0。所有新图只使用已保存记录，已显式标重绘。本报告不接受/改写此前Task8F的待审状态或任何冻结科学结论；最终素材选择和项目措辞等待ChatGPT与用户共同审查。

复核可用既有Python运行 `review_audit.py --verify`；生成脚本对已有目标拒绝覆盖，重建应使用新的独立工作副本/目录，不能删除现有证据后重做。本任务不启动算法修复、调参或新的模型训练。
