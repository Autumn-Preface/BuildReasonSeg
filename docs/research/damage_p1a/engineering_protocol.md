# DAMAGE-P1A 工程配置与复现入口

日期：2026-10-25。**工程参数固定供审查，数据/科学训练协议未准入**。数据证据见 [准入报告](../../../reports/damage_p1a/20261025/data_admission.md)。

## 隔离环境

新环境：`.conda/buildreasonseg-damage-p1a`。采用已有 proposal 环境的离线 `--copy --clone`，避免重新下载 PyTorch，并保持新环境文件独立。随后仅在新环境安装 `pycocotools==2.0.11`，缓存放在 `artifacts/damage_p1a/`。Conda 首次因缓存目录列表分隔符错误失败，保留日志后修复任务自建的未完成前缀；没有删除旧环境、升级驱动或安装到旧环境。

实际核心版本：Python 3.11.16，torch 2.13.0+cu132，NumPy 2.4.6，Pillow 12.3.0，SciPy 1.17.1，pytest 9.1.1，pycocotools 2.0.11。完整观察清单 [requirements_freeze.txt](../../../reports/damage_p1a/20261025/requirements_freeze.txt) 包含克隆环境已有额外包，**不是跨机器可直接安装的最小依赖文件**。实际模块加载路径均在新环境，见 [runtime_paths.json](../../../reports/damage_p1a/20261025/runtime_paths.json)。

## 标准模型与固定 recipe

[配置](../../../configs/damage_p1a/unet_baseline.json)：RGB/255；两独立 logit 通道；普通 same-padded U-Net，宽度 64/128/256/512/1024，双 3×3 Conv+ReLU、MaxPool、转置卷积上采样及 skip concat；不含预训练、BN 或新模块。结构来自标准 U-Net 思路的独立实现，不引入第三方损伤模型代码或预训练权重。[原论文](https://arxiv.org/abs/1505.04597)

- 31,031,810 参数；从头初始化。
- 512²原生图块，microbatch 1，梯度累计 8。每原图每 epoch 4 个均匀随机 crop，不只保留正块。
- 图像不足尺寸则 pad；pad 标签 validity=false；GT 不缩放、不插值或膨胀。增强仅 H/V flip 与 90°旋转，RGB 和 GT 同步。
- 损失固定 `0.5*masked BCEWithLogits + 0.5*masked Soft Dice`，epsilon=1e-6；未知通道/像素不贡献 loss；全无有效监督报错。
- AdamW，固定 LR=1e-4、weight_decay=.01；没有 LR 搜索或测试阈值调节。
- seed=20261025；FP32、deterministic algorithms、TF32=false、cudnn benchmark=false；不宣称跨硬件逐位相等。
- 60 epoch 上限和 12 小时保守 wall-time 上限，含数据加载与验证；不是已经运行的时长。无穷/NaN/非正预算、无效 epoch 与 test inference 被拒绝。
- 按完整原图 val 的两个前景类 micro IoU 均值选 best，严格大于才更新，平局留最早 epoch；不按 train loss 或 test 选模型。若 val 不完整或类别无支持，不发布部分分数。

这份 recipe 没有绑定可训练数据 manifest，因为权威 class_map 与 validity policy 仍为空；其文件 SHA 记录在 runtime_paths.json。不能仅把 admission 的布尔开关翻转后启动，需要补全并审核数据/划分/映射证据。

## 数据、坐标与指标

`research/damage_p1a/data.py`：显式 raw ID 映射；COCO polygon 栅格化；同类并集、跨类重叠保留；unknown 和显式 ignore 独立。对非法点数/范围/NaN/退化几何拒绝，不静默裁剪或修 GT。known_channels 必须真布尔值，字符串 `false` 不会变成已知负类。

manifest 在切图前验证 source_image_id、图像 SHA、已知 building/sequence/tile_parent 在 split 间不能重用。group 缺失必须明示，不能据名称猜组。原图和标签每次按清单 SHA 检查；当前正式 manifest 不存在，门禁会先拒绝读取它。

验证窗口 512²、overlap=128，末窗锚定图像边界。只在实际原图范围累计概率，重叠等权平均，覆盖完整后才 `>=0.5` 二值化。没有 resize 原图来替代评价，没有平均 tile IoU。原图尺寸、tile 数、同步网络时间、排除输入/GT加载的推理系统时间，以及包含 GT/指标的评价总时间分别记录。后者不能冒充纯推理耗时。

`metrics.py`：原图有效像素上累计逐类 TP/FP/FN；IoU、Dice、Precision、Recall 分母为 0 时 None；两类均有 GT 支持才发布前景 mIoU。空 GT+空预测不补 1；阴性图 FP 图像率、平均 FP 面积比例单列；unknown 通道不进入分母。runner 保留每图统计 CSV 接口。本任务没有真实 val CSV 或模型精度表。

## 执行入口和当前门禁

在仓库根目录使用新环境：

```powershell
& ./.conda/buildreasonseg-damage-p1a/python.exe -B -m pytest tests/test_damage_p1a.py -q --basetemp .tmp/damage_p1a_fresh_unique_directory
& ./.conda/buildreasonseg-damage-p1a/python.exe -B -m research.damage_p1a.train --config configs/damage_p1a/unet_baseline.json
```

pytest 临时父目录须已存在，传入全新且仅属于本任务的 basetemp，避免 pytest 管理未知目录。第二条命令当前预期退出 2 并报告 `DATA_ADMISSION_BLOCKED`，不是启动训练的建议。门禁位于 manifest 访问、模型构造和 run 目录创建之前。训练框架在实际数据上未完成端到端训练验证；测试 PASS 不升级为训练实验完成。

## 已执行验证

- 38 项合成工程测试通过：手算混淆统计、空域/unknown、COCO 方形像素边界、重叠、padding、非整倍原图重建、split 泄漏、退化几何、shape 广播拒绝、预算 NaN 拒绝及训练先拒绝。
- synthetic RGB+多边形的完整 evaluate API 已在 CPU 上验证原图 mIoU=1 的构造例；这是**合成算法单元预期**，不进入真实模型精度表。
- 实际 RTX 5080 Laptop 上 64²矩阵运算以及同一标准 U-Net 的两次 512²合成优化步骤完成。7.479 秒包含初始化；peak allocated 8594177024 B、reserved 13314818048 B。数值仅适用于该合成过程，不能外推真实训练吞吐或总耗时。没有保存其随机权重为损伤 checkpoint。
- pycocotools/NumPy 接口有 8 条 DeprecationWarning，原日志保留，未屏蔽。几何与指标断言通过；不因此承诺所有平台兼容。
- 独立 reviewer 修复复核见 [工程复查](../../../reports/damage_p1a/20261025/independent_engineering_review.md)。当前无确定的合成工程阻断缺陷，数据准入仍关闭。

完整 checkpoint、真实学习曲线、真实验证指标与预测错误分析均未执行。预算终止/真实模型训练的完整运行路径也没有用实际损伤数据验证，恢复时须先小规模真实 smoke 再决定是否继续开发种子。
