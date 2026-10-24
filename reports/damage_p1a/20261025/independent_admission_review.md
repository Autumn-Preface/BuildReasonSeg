# DAMAGE-P1A：CUBIT-InSeg 独立数据准入复查

日期：2026-10-25。reviewer：独立子代理 `/root/independent_p0_review`。观察分支：`research/damage-p1a-visual-baseline-20261025`，本轮开始观察 HEAD：`a2f43aa2f5379920b9db6452dcc18b80481cdb76`。实际 Git 状态优先于此观察记录。

本轮先读仓库启动文件、[P0 数据调查](../../../docs/research/damage_p0/dataset_survey.md)及 [P0 独立复核](../../damage_p0/20261019/independent_review.md)，再只读访问作者公开来源。没有运行模型、训练、下载完整数据集、执行作者配置或修改其他文件；按主执行者授权仅新增本报告。本结论不是 Supervisor 验收。

## 准入结论

**`MAPPING_UNCONFIRMED`：原始 YOLO polygon TXT 的数字类别定义尚未获得直接权威确认，真实标签训练门禁应继续关闭。**

官方训练附件已确认作者使用 crack/spalling 两类及该顺序，但 MMDetection 的模型类别顺序、YOLO 评估日志的类别显示顺序、文件名前缀和抽样视觉对应，都不能独立替代原始 TXT 的 ID 定义。不得将它们改写为“已找到原始 `names: {0: crack, 1: spalling}`”。

可继续独立环境、原始文件及数字标签审计、synthetic-only 转换/指标/U-Net 管线。不得将数字 ID 审计通过解释为语义映射已确认；真实数据训练须先关闭本门禁及下述标签语义门禁。主执行者已采纳停止真实训练建议。

## 官方来源链与版本边界

1. [作者 CUBIT-InSeg 仓库](https://github.com/BenyunZhao/CUBIT-InSeg)，本次只读核对的树只有 LICENSE、README 与 figures；没有直接数据 YAML 或类别映射表。README 明确 CC BY 4.0，并直接链接原始数据 Drive 与 benchmark 扩展页。
2. [原始数据 Drive](https://drive.google.com/drive/folders/1HqS0rMWgwsy70TbaZYFqwzgFxLL_hp_b)。原版对应论文 6996 图、crack/spalling 两类、train/val/test 为 5596/699/701；具体归档内容仍须与所用 release 及文件身份绑定。
3. [作者 benchmark 扩展页](https://benyunzhao.github.io/CUBIT-Dataset/)另描述 12,500+ 图、包含 Moisture、70/10/20 划分。**这不是把原版归档改成新划分或增加类别的授权，也不能证明当前原版已完成建筑级隔离。**
4. 扩展页直接链接 [CUBIT-InSeg-log-yaml 公开资料目录](https://drive.google.com/drive/folders/1smAGS7WpTrNQPt0enS1FqQt9n4DC9bKu)。reviewer 只列目录并取下表小文件；没有下载其中模型权重。

## 实际读取的小文件身份

下列 SHA256 对应 HTTP 200 返回的原始字节，不是重排、解析或渲染后的内容。只读网络核查日期为 2026-10-25。

|资源|公开定位与父目录|bytes|SHA256|能证明什么、不能证明什么|
|---|---|---:|---|---|
|`mask-rcnn_convnext-t_fpn_amp_building2.py`|Drive ID `1yzXGmIBf0faWiS90UExCiQ5CzseTzVS2`；[直接取回地址](https://drive.usercontent.google.com/download?id=1yzXGmIBf0faWiS90UExCiQ5CzseTzVS2&export=download&confirm=t)；父目录 `2.MMDetection-frame/ConvNeXtV2-t`（`1iAeE70z40ayrXdf7n9ittDphSf4djSGZ`）|11613|`60de2ef68c0c686cfcfe4c2d19219d6f713ef9679ca1de62632de2d26ebcf044`|`metainfo.classes=('crack','spalling')`、两类模型，数据根指向 `CUBIT-Seg/building2`。这是作者模型类别顺序；COCO 类别可重映射，不能单独证明原 YOLO TXT 的数值 ID。|
|YOLO11n `args.yaml`|Drive ID `1oYIDshJDP3cJAtUlsrCxfa4gxIAItpLr`；[直接取回地址](https://drive.usercontent.google.com/download?id=1oYIDshJDP3cJAtUlsrCxfa4gxIAItpLr&export=download&confirm=t)；父目录 `1.Ultralytics/YOLO11/YOLOv11_v11n`（`1Asf834rlfCl9wQFR68xK22hs9ZhzuUzY`）|1623|`0f8554fa7111a66eb8372a5518a2af8176628c68c53fcfed07e5b54073c134af`|segment 训练配置引用作者机上的 `ultralytics/cfg/datasets/infra-seg.yaml`；附件本身没有 `names`，没有取得被引用的 dataset YAML。|
|`v11n-640-seg-mawan811-test.log`|Drive ID `1FT8XZcNbgCePxsLdadS7JGSvO80Dc-ia`；[直接取回地址](https://drive.usercontent.google.com/download?id=1FT8XZcNbgCePxsLdadS7JGSvO80Dc-ia&export=download&confirm=t)；与上项同父目录|11051|`440622388b3785887230f1c75341dccb8fff14b82366c1cd29986ff9aeb6f5c7`|记录 `building811/test` 701 图，crack GT 支持 320 图/5025 实例，spalling 381 图/992 实例；320+381=701，在此日志统计口径下没有共同支持两类的图。这不证明另一类实际不存在或全图标注完备；没有把论文性能作为本项目实测。|

这些新取回文件只在 reviewer 进程内读取，未由 reviewer 保存到本地缓存；主执行者若需要持久化，可按地址重新获取并核对上述 bytes/SHA256。作者机绝对路径仅用于对应来源，不构成版本或建筑 ID 的保证。

## 原论文 Fig.5 直接核查

[原论文 DOI](https://doi.org/10.1016/j.autcon.2026.106980)，Fig.5 的[原始大图](https://ars.els-cdn.com/content/image/1-s2.0-S0926580526002219-gr5_lrg.jpg)：549964 bytes，SHA256 `4544759ea59fd6906ee3e834d90af13d96c799d1f3487e8b0ce9c87157adc6ad`。

reviewer 已实际视觉查看：左下 LabelMe JSON 示例为 `label: spalling`，右上 YOLO 格式说明只写 `<class_id>` 占位符，没有 0 或 1。**Fig.5 不能关闭原始数字 ID 映射门禁。**原图仅内存查看，没有写图像文件或复制整篇论文。

## 标签完整性与 overlap 语义

原论文标注章节说明：可可靠辨识并描绘的缺陷应完整标注；微细裂缝只在其范围可可靠视觉追踪时标注；crack 与 spalling 重叠时优先 spalling，避免同一区域双标。因此“作者的完整标注原则”不等于所有物理裂缝均被标出，剥落区 crack=0 也不等于没有裂缝。

后续转换和损失/评价必须明确是复现作者优先级标注任务，还是项目的独立双通道可见损伤任务。若采用后者，crack 在 spalling 覆盖区如何 ignore、哪些通道/区域为已知负类，必须事先审定；reviewer 不自行新增或冻结规则。不能凭 CR/SP 文件名前缀把未出现的另一类自动视为完整阴性，也不能自行补画重叠裂缝。

## 划分与其他未关闭项

- 官方原版数量可对应 80/10/10，但本次未找到建筑/航线/序列 group 的权威映射或隔离证明。P0 已观察相近视角，不能据图像级划分声称建筑级泛化无泄漏。
- 在原图/建筑/连续序列归属、exact/near duplicate 与 split 交集核查完成前，不应关闭独立验证/测试的泄漏门禁。裁剪必须在划分之后，保留 parent ID；测试集不得参与分组调优或训练选择。
- 还需锁定版本/文件 manifest、正式栅格转换器和 pixel inclusion、异常/缺标签/未知通道处理及 native-resolution 样本核对。数字类别计数没有语义确认效力。
- 没有发送作者联系邮件或向外部人员发消息；这需要用户明确指示。

## 已有本地证据定位

以下是仓库现有可读路径，不能替代尚未取得的原始 ID 定义：

- [P0 原作者 README 缓存](../../../evaluation/damage_p0/sources/cubit_readme.txt)：本地忽略的历史参考缓存，可能陈旧；本轮同时核查了线上原作者页。
- [P0 归档目录盘点](../../../evaluation/damage_p0/cubit_folder_inventory.json)、[P0 样本检查及 hash](../../../evaluation/damage_p0/sample_checks.json)：历史目录/抽样证据，不是新版本全库认证。
- 本任务目录已有 `official_git_tree.json`、`official_archive_inventory.json`、`source_access.json`、`label_audit.json` 和 `label_file_manifest.json`，由主执行者生成；本 reviewer 未重复独立执行它们的全量审计，不能把它们称作本 reviewer 的独立复现结果。

最终状态：**真实训练准入继续阻断；允许范围内的环境、数字审计与 synthetic-only 工程可继续。**恢复真实训练前须有可审查的映射证据、标签语义/unknown 规则和适用划分结论，由 Supervisor 决定；本报告不自授权训练、不冻结协议、不接受科学结果。
