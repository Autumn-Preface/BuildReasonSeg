# CUBIT-InSeg 数据准入报告

核查日期：2026-10-25。结论：**不准入真实标签训练（BLOCKED）**。这不是数据集不可用的泛化结论，而是当前任务所需的类别与独立验证证据未满足。

## 来源、版本与许可

官方入口：[BenyunZhao/CUBIT-InSeg](https://github.com/BenyunZhao/CUBIT-InSeg)，[作者 README](https://github.com/BenyunZhao/CUBIT-InSeg/blob/master/README.md)，[原论文 DOI](https://doi.org/10.1016/j.autcon.2026.106980)。作者 README 明确图像、标注及元数据采用 CC BY 4.0；本任务保存本地许可快照，并在公开证据图中注明来源。Git 版本由官方 ref 响应确认：`8fee6385a2a1a714cdeab53321a8e275f652799f`。见 [official_git_ref.json](official_git_ref.json)、[official_git_tree.json](official_git_tree.json)。

本次审计绑定 README 原始 Google Drive 根目录 `1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b`。下载资源没有稳定的语义版本号或签名发布清单；用 Git ref、Drive ID、体积、HTTP 版本头、归档目录 CRC/大小及实际取得文件 SHA256 共同登记，不能把目录 CRC 当作全体图像的 SHA 认证。全体图片尚未下载，因此“全图片版本已哈希锁定”**未完成**。

新 [CUBIT Challenge 页面](https://benyunzhao.github.io/CUBIT-Dataset/) 描述的扩展规模、Moisture 和 70/10/20 协议与这里的 6996 张、两类、5596/699/701 资源不同。没有把扩展页规则覆盖到原始归档。

## 官方归档与完整空间预算

完整清单：[official_archive_inventory.json](official_archive_inventory.json)。

| 官方 split | images.zip 字节 | labels.zip 字节 | 图像条目 | 标签条目 |
| --- | ---: | ---: | ---: | ---: |
| train | 15034071180 | 248950377 | 5596 | 5596 |
| val | 1853116822 | 30763780 | 699 | 699 |
| test | 1861617729 | 30772180 | 701 | 701 |

全部归档共 **19059292068 B**，全部解压内容共 **20530647832 B**。实际下载前按归档+解压双份、环境/缓存 8 GiB、输出 4 GiB、临时空间 2 GiB、系统余量 50 GiB 预算，合计 **108309416636 B**；当时可用 **126897930240 B**，满足预算。见 [storage_budget_exact.json](storage_budget_exact.json)。早期粗预算仅用于元数据范围读取；实际数据取得采用此处更严格的 50 GiB 余量。

发现准入问题后没有下载整库图片，也没有生成全分辨率稠密 Mask/概率缓存。仅保存三份标签归档共 **310486337 B**、有界元数据、四张用于重复核验的图片和本地审计参考。两张裂缝配准图明确读取既有 P0 样本，不冒充本次重新下载成功。没有清理旧资产。

## 类别 ID 与标注规则

原始格式是每行 `integer_id x1 y1 x2 y2 ...` 的归一化多边形，不是可由颜色或文件名前缀确定的语义 Mask。

权威证据的层次不同：

1. 官方仓库和论文确认任务类别名称为 crack 与 spalling。
2. 作者直接链接的 MMDetection 配置给出模型类别顺序 `('crack', 'spalling')`。它不证明原始 YOLO TXT 的数字映射，因为不同格式可重映射类别。
3. 作者 YOLO `args.yaml` 引用其机器上的 `infra-seg.yaml`，公开文件中未取得该文件的 `names`。
4. 独立 reviewer 实际核对原论文 Fig.5：LabelMe 示例有 spalling 名称，TXT 示例却仅写 `<class_id>` 占位符，没有 0/1。

因此，**原始 TXT 的 0/1→crack/spalling 权威映射仍未关闭**。P0 的视觉观察和本轮数值统计不能替代权威定义。用户也确认没有自行下载过可补充的相关文件。没有猜测 mapping、发送作者邮件或将 MMDet tuple 偷换为已确认数据 ID。

[独立准入复查](independent_admission_review.md)记录官方小文件的 URL/Drive ID、字节数与 SHA256，包括 `mask-rcnn_convnext-t_fpn_amp_building2.py`、YOLO args 和官方测试日志。原论文 §3.2 还说明：只标可靠可辨识的缺陷，重叠时优先 spalling，细裂缝要能可靠追踪。这使“剥落区没有 crack 标签”不能直接等价为“物理上无裂缝”。双独立通道的有效/ignore 策略尚未冻结，不静默把未知区当负样本，也未修改原 GT。

## 全量数字标签审计

以下统计只使用 **原始数字 ID**，不将其命名为裂缝/剥落。[label_audit.json](label_audit.json)与 [逐标签 SHA256](label_file_manifest.json)可复核。

| split | ID 0 实例 / 图像 | ID 1 实例 / 图像 | 多类别图像 | 空标签 | 缺失/孤立标签 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 41811 / 2535 | 8224 / 3061 | 0 | 0 | 0 / 0 |
| val | 5029 / 299 | 1106 / 400 | 0 | 0 | 0 / 0 |
| test | 5025 / 320 | 992 / 381 | 0 | 0 | 0 / 0 |

总计 **62187 个多边形实例**。全部 6996 个标签文件可解码，行结构/点数/数值有限性/归一化坐标范围检查未发现错误；坐标范围实测 0.001042–0.998958。该全量审计检查的是上述语法/范围，**不是全量多边形拓扑、栅格后面积或漏标质量认证**。新正式转换器会另行拒绝退化几何；没有全量完成此门禁，更不能因此写“全部标签科学正确”。

没有一张归档标签同时含两种数字 ID，这不能证明另一类物理损伤不存在或已完整标注，负类/未知规则仍需来源依据。正式测试集这里只用于许可范围内的文件、数字标签和重复准入审计；没有模型预测、性能评分、阈值选择或测试集重分。

## 重复、近重复与验证独立性

文件 stem 在 train/val/test 间无交集，但这**不是无泄漏证据**。按图像归档条目的 `(uncompressed_bytes, CRC32)` 检索得到 **589 组重复候选**，其中 **222 组跨 split**：train/val 99，train/test 110，val/test 13。这些均为筛查候选，不把 CRC32 当密码学相等证明。

按归档顺序取前两组 train/val 候选，范围读取完整 JPG，并计算文件 SHA256 和解码 RGB SHA256。结果均相同：

| train / val | 文件 SHA256 | 结果 |
| --- | --- | --- |
| SP1_813.JPG / SP0_818.JPG | `ad5ef01c9a6a81eb183ba19e5ee8a2f419eaaad6e567c8c7c8e2a65c7ff7e781` | JPG 字节和解码像素相同 |
| SP1_1091.JPG / SP0_1102.JPG | `f64361a7117be8879813cfb373131719a4911b440265441bc532710f3eddd0ff` | JPG 字节和解码像素相同 |

[详细证据](duplicate_sample_audit.json) / [实际图像对照](sample_evidence/confirmed_train_val_duplicate.png)。独立 reviewer 另行重算四个文件的两种 SHA 并确认。只对这两组称“已证实”，没有外推 222 组均相同。

因此当前官方验证集**不能直接作为可靠的独立验证证据**。没有擅自删除重复样本、重划 test 或把官方 split 换成自造 split。尚未取得建筑/航线级 group；不能把 CR/SP 前缀当建筑 ID。全量解码哈希、感知相似/相近视角审计、完整尺寸分布及建筑组独立性未完成，仍需补充。P0 已观察到 CR1_1/CR1_10 的同墙相近视角；这只是具体观察，不是全量分组。

## 原图—多边形配准样本

[sample_alignment.json](sample_alignment.json)记录四个小样本的原图/标签 hash、尺寸、数字实例数、原生裁剪和来源类型。两张 CR 历史原图为 3000×4500，两张本次 train SP 原图为 4800×3200；两张对应 val 重复图也是 4800×3200。不能把这些样本尺寸当作全库尺寸分布。

- CR 原图来自受保护的既有 P0 样本，只读使用；大小与 CRC 匹配当前官方目录，但没有伪称重新取回其当前归档完整 SHA。标签从本轮官方 labels ZIP 读取。
- SP 原图来自本轮官方范围下载，完整字节 SHA 已计算。
- 使用固定 pycocotools 2.0.11 将归一化多边形按 `(x*width, y*height)` 转为 COCO 像素域；叠加图仅标 numeric ID 0/1。执行者实际查看样本，未见整图错配；裂缝局部标签较可见纹理外扩，不能承诺细边界逐像素真值。
- 图中不存在模型预测。原图、标签均未重写。第一次取样受 4 MiB 单 range 上限拒绝；修为仍有界的 8 MiB 后，端点返回 HTML/长度 0，取回失败。两次失败日志保留，随后显式选择只读 P0 样本路径，未静默替换。

## 准入结论与恢复条件

[admission.json](admission.json)：`status=BLOCKED`、`training_allowed=false`。停止真实训练及真实验证结果生成。

恢复前至少需要：原始 TXT ID 的权威定义；与原标注语义一致的双通道 validity/ignore 规则；可审查的去重/建筑组验证方案。若只保留探索性官方复现轨，也须明确污染与未知通道限制并交 Supervisor 审核，不能把它当跨建筑泛化结果。完整测试集不得擅自重划或用于选择方案。本任务没有自动改用其他数据集。
