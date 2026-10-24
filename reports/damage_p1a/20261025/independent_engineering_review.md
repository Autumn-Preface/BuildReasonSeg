# DAMAGE-P1A 独立工程复查

日期：2026-10-25。reviewer：独立子代理 `/root/independent_p0_review`。复查时观察 HEAD 为 `2955c4ad029306be8721d9fed5928dcf0d5b611f`，并包含本轮工作树修复增量；实际 Git 状态优先。

范围：`research/damage_p1a/{data,metrics,model,train}.py`、`configs/damage_p1a/unet_baseline.json`、`tests/test_damage_p1a.py`、当前 admission 与重复样本证据。reviewer 只读取源码和证据、执行无模型内存反例及哈希复核；没有运行真实训练、模型构造/推理、GPU 运算或访问测试预测。按授权仅写本报告，不改代码。此复查不是 Supervisor 验收。

## 发现、修复与独立读回

|问题|严重度|修复及独立核对|
|---|---|---|
|`known_channels` 原只查长度；字符串 `"false"` 会在 NumPy bool 转换后成为 True，未知通道变成已知负类|P2，已关闭|`data.py:_known_channels()` 要求恰好两个真正的 bool，manifest 和 rasterize 均调用。reviewer 重跑 `['false','false']` 反例，明确拒绝；不再产生 32/32 有效像素。|
|manifest 有 `tile_parent_id` 时原未检查跨 split 原图归属|P2，已关闭|`validate_manifest()` 将该字段纳入 source/hash/building/sequence 的跨 split 排他检查。reviewer 用不同 source ID/hash、相同 parent 的 train/val 两行重跑，明确拒绝。|
|原图回拼未检查概率空间维度，`(2,1,1)` 可被广播成整个 tile 的概率|P2，已关闭|`Reconstructor.add()` 检查 ndim、两通道、空间覆盖及 tile 原图范围。reviewer 重跑 singleton 概率反例，明确拒绝；不再静默扩展成 `(2,4,4)`。|
|三个相同顶点的正标注可通过 parser，COCO 栅格后为零像素并默默成为背景|P2，已关闭|`parse_yolo()` 拒绝不足三个独立顶点或零有向面积的退化 polygon。reviewer 重跑三同点反例，明确拒绝；未引入按结果调节的面积阈值。正常细目标的 COCO 栅格语义仍应在真实准入后核查。|
|预算原只有 `hours>12`，Python JSON 可接受的 NaN 会绕过该比较及后续 deadline 比较|P2，已关闭|`train.py:validate_config()` 要求有限正预算且不超过 12 小时、1..60 整数 epoch、明确禁止 test inference。reviewer 独立核对正常配置通过，NaN/Infinity/零/超时限/非整数或超限 epoch 均拒绝。|

已读 [tests_final.txt](tests_final.txt)：**37 passed，7 warnings**。reviewer 没有重复运行整个测试集，不能把执行者的全套日志冒充 reviewer 自行复现；上述具体反例由 reviewer 在独立 `python -B` 进程重跑确认。

## 当前门禁是否会意外训练

[admission.json](admission.json) 仍为 `BLOCKED`、`training_allowed=false`，mapping/validity/manifest identity 尚未准入。`run()` 首先调用 `require_admission()`；在该状态下，训练入口先抛 `DATA_ADMISSION_BLOCKED`，不会进入 manifest 解析、torch 导入、模型构造、输出目录预留或 optimizer。

reviewer 独立调用当前入口时，将 `Path.read_bytes` 替换为一调用就抛错的探针；结果仍先收到 admission 异常，没有触发字节读取探针，且 `sys.modules` 中没有 torch。结合代码顺序，**当前配置没有发现意外开始真实训练的路径**。该结论不授权把 gate 改为 ADMITTED。

## 数据、指标与训练逻辑

- 原始 polygon 经显式 raw ID→通道映射，使用 `pycocotools.frPyObjects/decode` 栅格；同类取并集、跨类重叠保留，unknown/ignore 与 target 分离。不根据文件名推断通道；未知通道出现正标注会拒绝。spalling-priority 的实际 validity/ignore 政策仍未冻结，代码能力不等于数据政策已批准。
- pad 的 valid 为 false；随机 crop/翻转/90°旋转不 resize GT。manifest 排他检查在 crop 创建前进行。缺建筑/航线信息不能据此宣称跨建筑泛化；编码不同但视觉近重复也不能单凭 encoded SHA 检查排除。
- 验证在原图坐标汇总重叠 tile 概率，再以固定阈值二值化。未覆盖像素拒绝；没有按 tile IoU 取平均冒充原图指标。
- TP/FP/FN 只累计有效像素，主指标按整库逐类 micro 计算，再等权前景 mIoU。双空为 None，空 GT 假阳性计 FP，unknown 整通道不计；没有正 GT 支持的类不会被补 1 进入主两类 mIoU。另报原图宏 IoU 及负图 FP 率/面积。
- 模型为两输出的普通 U-Net，无预训练/额外模块。loss 用 valid 屏蔽 BCE 与 Soft Dice，已知全阴性像素仍受 BCE 监督；完全无监督 batch 明确报错，不伪造梯度。未来真实 validity 政策若产生这种 crop，须先明确处理，不能临时补负类。
- 仅 train 参与梯度；仅完整 val 原图指标选择 best，严格 `score>best` 保留更早的并列 checkpoint。test 行不进入训练或验证推理。超时中断验证不返回部分分数，预算包含主循环的数据与验证壁钟时间；不是测得的纯 GPU 占用时间。
- `accumulate=8` 是累积微批 loss 的梯度；Soft Dice 非线性，不能将它描述为与直接 batch=8 的 Soft Dice 完全等价。当前 recipe 已明确 microbatch/accumulate，尚无跨模型公平性结论。

## 明确的官方 split 泄漏证据

[duplicate_sample_audit.json](duplicate_sample_audit.json) 中 589 个 CRC/大小候选组、222 个跨 split 候选组，不等于全部被 SHA 验证。reviewer 仅独立读取以下四张 **train/val** 原图，重算 JPG 字节 SHA256 与解码 RGB SHA256：全部与报告吻合，两组分别完全相同。

- train `SP1_813.JPG` / val `SP0_818.JPG`：encoded SHA256 `ad5ef01c9a6a81eb183ba19e5ee8a2f419eaaad6e567c8c7c8e2a65c7ff7e781`；decoded SHA256 `a9b5592618237bf180c743c833d92cc8fa21e6d11a0c2639d0b9cbbe9eedda6b`。
- train `SP1_1091.JPG` / val `SP0_1102.JPG`：encoded SHA256 `f64361a7117be8879813cfb373131719a4911b440265441bc532710f3eddd0ff`；decoded SHA256 `ffff457343ff805be1460f84016f7b0e748cc26bb1ea75f7d94f4130cd9cd504`。

这两组是已验证 train/val 内容泄漏，不能再把原版官方划分笼统描述为仅“可能重叠”。其他候选仍保留候选身份，reviewer 没有读取测试图像外观或测试预测。不得静默移动官方 test、删标签、按测试表现重分，或自行将候选组全部认定为确证重复。

## 结论与限度

本轮五项 P2 均已修复并对具体反例独立关闭；当前没有发现合成工程交付的剩余确定阻断缺陷。**真实训练准入继续阻断**：原始数字语义 mapping 未直接确认、独立通道 validity/overlap 政策未冻结、官方 train/val 已有确证重复。环境/转换/指标/U-Net 工程通过不能关闭这些准入门禁，也不能证明真实数据上的资源适配、训练稳定性或科学效果。

后续仅按已有授权继续 synthetic-only 工程及数据审计；恢复真实训练和最终验收由 Supervisor 决定。本报告不自授权重分、不启动开发种子、不接受模型或指标结果。

## 最终增量复核（2026-10-25）

保留上文首次观察的 **37 passed / 7 warnings** 历史事实。此后新增非零几何但 subpixel 空栅格反例；reviewer 已读更新后的 [tests_final.txt](tests_final.txt)：**38 passed / 8 warnings，2.24 秒**。没有将后续测试结果倒写成此前已经观察到的结果，也没有由 reviewer 重跑整个套件。

`data.py:rasterize()` 现对 COCO decode 后没有任何前景像素的 polygon 显式报错，不再将正标注静默变成背景。reviewer 独立运行三个不同顶点、归一化几何面积约 `5.0000000008e-09` 的正 polygon：几何 parser 通过，8×8 COCO 栅格为零后明确拒绝。该检查没有增加可调最小面积或宽度阈值，也不膨胀/重画 GT。对真实标注出现此异常时应保留并报告，不为继续训练擅自修补。

`train.py:evaluate()` 的计时边界已分开：

- `input_and_gt_load_check_seconds`：RGB/GT 加载、解析与身份检查。
- `network_seconds_synchronized`：每个窗口前后同步包围的网络与 sigmoid 执行壁钟累计；不是 GPU events 得出的纯核时间。
- `inference_system_seconds_excluding_input_and_gt_loading`：加载后开始，到所有 tile 概率回拼完成，包含该区间的预处理、执行、传回与聚合。
- `evaluation_end_to_end_seconds_including_gt_and_metrics`：含输入/GT 加载与指标计算的完整评价时间；不称作纯推理总耗时。

源码字段与实际区间对应，工程协议对此也明确区分；没有据此新增真实运行耗时或损伤性能声明。reviewer 未运行真实模型、训练或 GPU 测量。

另已读最终 README、data_admission 和 engineering_protocol，未发现新的重要结论误述；已告知主执行者将工程协议最新 warning 数从 7 同步为 8（轻微文档更新，上文历史测试数不改）。两组 train/val 确证重复与其他候选的区别、映射未确认、unknown/overlap 未冻结、真实步骤为 0 等限制均保留。当前结论仍为工程增量无新确定阻断，**真实训练准入 BLOCKED，Supervisor 最终审核**。
