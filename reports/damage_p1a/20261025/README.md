# DAMAGE-P1A 工作报告

日期：2026-10-25（Asia/Shanghai）。执行结果：**STOP_DATA_ADMISSION，已完成允许的数据与工程部分，真实训练未启动，等待 Supervisor 审核。**

## 结论

没有强行生成第一个训练分数。本次发现两项独立的准入阻断：原始 YOLO TXT 的 0/1 权威语义映射尚未取得；官方 train/val 至少两组图片已经证实字节及解码像素完全相同。另有剥落优先标注下的独立 crack 通道 validity/ignore 规则待审。按任务书的停止条件，真实训练、真实验证精度、checkpoint 和预测错误分析均停止，而非补写结果。

已经交付：官方数据清单与全量数字标签审计、重复泄漏证据、独立 Conda/GPU 环境、可复用的显式有效域/COCO 转换/图块/原图重建/指标模块、标准 U-Net 及有门禁的训练框架、固定工程配置、38 项合成测试和独立复查。工程通过不代表数据准入或科学验收通过。

## 交付索引与状态

| 交付项 | 状态 | 入口 |
| --- | --- | --- |
| 数据准入 | 已调查；结论 BLOCKED | [详细报告](data_admission.md)、[门禁](admission.json) |
| 官方 split/数字类别/来源与哈希 | 已完成目录和全部标签审计；全图片 SHA/尺寸未完成 | [归档清单](official_archive_inventory.json)、[标签统计](label_audit.json)、[逐标签 hash](label_file_manifest.json) |
| 重复与建筑组 | 222 跨 split CRC/大小候选，2 组 train/val 已确证；建筑/航线及全近重复未完成 | [证据 JSON](duplicate_sample_audit.json)、[实际图像](sample_evidence/confirmed_train_val_duplicate.png) |
| 原图—GT 配准 | 4 个有来源区分的确定性小样本 | [样本记录](sample_alignment.json)、[图片说明](sample_evidence/README.md) |
| 独立训练环境 | 已建立、实际 CUDA/梯度通过 | [工程 GPU 记录](engineering_gpu.json)、[模块路径](runtime_paths.json)、[版本清单](requirements_freeze.txt) |
| 转换/加载/评价/U-Net | 已实现与合成验证；真实完整训练未验证 | [工程协议](../../../docs/research/damage_p1a/engineering_protocol.md)、`research/damage_p1a/` |
| 自动化测试 | **38 passed / 8 warnings** | [最终日志](tests_final.txt) |
| 真实小规模训练、开发种子 | **NOT_RUN**，0 个真实损伤训练 step | 不存在真实 checkpoint、val 指标或收敛轮次 |
| 模型错误分析 | **NOT_RUN**，只有数据观察与后续假设 | [状态说明](preliminary_error_analysis.md) |
| 独立审查 | 已完成准入与工程复查，未代替 Supervisor | [准入复查](independent_admission_review.md)、[工程复查](independent_engineering_review.md) |
| 旧资产保护 | 检查点前后 2324 文件 fresh SHA/大小/mtime 一致，受保护 Git 路径无改动 | [保护结果](protection_final.json) |
| 最终交付审计 | JSON/YAML、源码 AST、相对文档链接、归档和重复样本 hash、训练门禁及空间余量通过 | [最终审计](final_audit.json) |

## 数据关键发现

原版归档 train/val/test=5596/699/701，标签配对完整，共 62187 数字多边形实例。原始 ID 0 的实例数依次 41811/5029/5025，ID 1 为 8224/1106/992。这里**没有**将它们擅自命名为 crack/spalling。全部标签均只含单一数字类别，空标签为 0；该事实不能证明另一通道完整标注或数据拥有可靠阴性图。

按归档 CRC+大小找到 589 重复候选组，其中跨 split 222 组（train/val 99、train/test 110、val/test 13）。仅两组 train/val 完成密码学及解码核验：`SP1_813→SP0_818`、`SP1_1091→SP0_1102`。独立 reviewer 重算确认。其他候选不冒充全部确证；没有重新分配或删除任一官方 split 文件。

作者公开的 MMDetection 类别顺序和 YOLO 测试日志不足以直接确认原始 TXT ID；其 args 指向的 `infra-seg.yaml` 未取得，论文示例只有 `<class_id>`。官方扩展挑战页的规模/类别/比例与原版不同，不混版本。完整证据链与许可见 data_admission.md。

## 环境、工程与真实消耗

新环境 `.conda/buildreasonseg-damage-p1a` 为离线独立复制，已验证 torch/numpy/PIL/pycocotools 实际从新目录加载。关键 Python/torch/numpy 文件与原环境 SHA 相同但文件 ID 不同；原环境仍无新装的 pycocotools。没有升级驱动或修改旧环境依赖。克隆保留已有额外包，但损伤框架不导入 MLLM、RC1、Demo 或空间推理实现。

本机实际 GPU：RTX 5080 Laptop，16303 MiB，当前驱动 621.01，torch 2.13.0+cu132（运行时 CUDA 13.2）；不是将旧 P0 驱动版本直接当当前状态。64²矩阵乘检查总和 262144。标准 U-Net 两次 512²**合成** AdamW 步骤：31,031,810 参数，耗时 7.479 秒（含初始化），peak allocated 8594177024 B，reserved 13314818048 B。它只证明本机工程执行与有限梯度，不是损伤训练/吞吐/收敛/准确率。未保存其随机权重作为研究 checkpoint。

配置：seed20261025、512²、microbatch1、累计8、原图均匀4crop、H/V/rot90、RGB/255、FP32；0.5 masked BCE+0.5 masked Soft Dice；AdamW LR1e-4/WD.01；阈值0.5；epoch<=60/保守wall<=12h；best 仅完整 val 原图前景 mIoU。数据 manifest、映射和 validity 尚未绑定，因此不能宣布科学协议冻结或直接翻开训练开关。

所有图片预算先于下载核算，完整潜在方案加 50 GiB 系统余量可容纳；但实际只取得三份 labels ZIP 共310486337 B及少量 metadata/JPG。新 artifacts 目录逻辑字节约326879313，新环境约3992902931（读取时刻快照，日志可能再增长）；不是 NTFS 实际分配或卷空闲变化。没有整库图片、全分辨率 Mask 缓存或新模型权重。存储来源单份登记，不清旧缓存。

## 工程检查、失败与修复

合成测试覆盖原生尺寸概率回拼、pad 有效域、固定 COCO 像素边界、重叠/空/未知通道、按类像素累计、阴性图 FP、缺类 NA、split-parent 泄漏和训练门禁。手构造 RGB+多边形的 evaluate API 通过，但其预期完美分数只在合成单元测试内，未伪装成模型性能。

独立复查发现并关闭五项 P2：真布尔 validity 校验、tile_parent_id 跨 split 检查、概率 shape 禁止静默广播、退化 polygon 拒绝、NaN/无穷预算拒绝。每项补反例断言，最终38通过。pycocotools 的 NumPy copy 接口产生8条 DeprecationWarning，原日志保留，未屏蔽。

失败同样保留：

- [首次 Conda 克隆](environment_clone.txt)因缓存目录分隔符无效失败；[修复日志](environment_clone_repair.txt)记录在任务自建未完成环境上继续复制成功。
- [图像取样尝试1](sample_alignment_attempt1.txt)因单range4MiB上限拒绝；改为仍有界的8MiB后，[尝试2](sample_alignment_attempt2.txt)遇端点HTML/长度0而失败，不据此声称官方ZIP损坏。
- 后续显式 `--existing-p0` 仅只读使用历史 CR 样本并标明来源，SP 样本仍为本轮已取得原图；没有静默替换资产。
- 没有真实训练失败/收敛日志，因为真实训练根本未准入。

## Git、隔离与可恢复状态

起点：fetch 后最新可信 `a2f43aa2f5379920b9db6452dcc18b80481cdb76`，包含 P0 与 DEMO-R1；新分支 `research/damage-p1a-visual-baseline-20261025`。未从滞后 main 启动或改写原分支。

已推送检查点：`2955c4ad029306be8721d9fed5928dcf0d5b611f`。后续修复与完整报告独立提交；最终本地/远端 SHA 由 Git push 后实际核对，不写自指向伪 SHA。未合并 main。

Git 仅包含新研究代码/配置、数值清单、报告和小图；原始数据 ZIP、完整图像、新环境、缓存与任何完整权重不提交。旧 RC1、Demo V3、P0 文档/原图、旧正式评价均未修改。P1A 授权单独追加治理，历史 P0 的 pending 结论未重写成 accepted。

## 停止与恢复要求

当前 `STOP_DATA_ADMISSION` 是任务书允许的停止结果，**不是成功完成首个真实训练基线**。恢复须补原始类别 ID 权威定义、validity/ignore 规则、可靠的去重/建筑组验证方案及相应 manifest，然后由 Supervisor 审查是否按探索性或其他明确协议继续。官方测试集不能擅自重分或用于选参。

不自动下载其他数据集替代，不自动进入 P1B/P2、多模型/多种子实验或创新模块；不声称视觉创新、跨建筑泛化或损伤语义正确性。完整原图哈希/尺寸/近重复扫描、真实训练与验证、真实错误分析均留在“未完成”栏，绝不补写缺失结果。
