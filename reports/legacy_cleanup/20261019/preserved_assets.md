# 关键保留资产

已校验 6443 条 SHA256 记录，逐路径/字节数/hash 见 preserved_hash_index.csv。初始 1500 个 Git 跟踪文件除 handoff 外的 1498 个字节保持一致，完整历史没有重写。

|资产|保留位置|保留依据|
|---|---|---|
|核心源码、治理、历史科研报告/指标/图像|当前 Git 工作树及 origin 各历史分支|全部原内容保留；不更新冻结结论|
|中期成果图片及来源证明|reports/midterm_evidence_review/|12 项成果清单、10 原图副本、2 基于记录绘图、原始 JSON/CSV/配置证据和审查结果均保留|
|一套完整旧系统/模型资产|C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1|全目录哈希保持不变，含 Qwen/SAM2/ProgramHead/YOLO/D-B1 及历史 inputs/outputs/logs|
|Task7I 六个冻结最终权重|artifacts/checkpoints/task7i/{Z-B3,D-B1}/{20261001,20261002,20261003}/best.pt|六项与既有 frozen manifest 一致，不做权重选择|
|各阶段/消融唯一历史权重|artifacts/checkpoints/；旧 WHU baseline weights|SHA256 不同的状态保留待审；包括 Task6A best_run5；仅三个相同 hash 副本移除|
|关键数据/标注/split/GT/索引|datasets/ 与 artifacts/whu_native_vector/ 及原始 WHU 根|关键记录哈希和未授权变更对比通过；未改 GT/alignment|
|原始 WHU 数据（范围外，只读）|C:\D\resources\Satellite dataset Ⅱ (East Asia)|17,388 个原始图像 file ID、大小和 mtime 与被移除导出链接前一致；不写任何原始文件属性或字节|
|旧 WHU 基线源码、训练记录和权重|C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment|代码、配置、曲线、评价、best/last 均保留；数据副本有可靠外部整树保留副本|
|旧基线数据集保留副本（范围外，只读）|C:\D\resources\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset|8078 项 SHA256 精确对应，删除后再次核验|
|保留 Python 环境|.conda/buildreasonseg-proposal|整个环境未变，版本清单 environment_versions.json；无真实模型导入|
|Codex/ChatGPT/DSH 安装、运行依赖、个人配置|实际安装根见 registered_appx.json/registered_desktop_apps.json|未删除任何安装文件或身份资料；C:\D\DeepSeek Harness 与 C:\D\DeepSeekHarness 是不同目录|

## Demo 实现与全部非模型运行证据

- `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\artifacts\legacy_archive\20261019\BuildReasonSeg_Demo_V1_non_model.zip`：277 项，44,740,017 bytes，SHA256 `6d117df063913488328ae9ffa22b1d2f218cd4e8f0ac76acf1d1269c01cc111e`。
- `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\artifacts\legacy_archive\20261019\BuildReasonSeg_Demo_V2_non_model.zip`：133 项，11,146,516 bytes，SHA256 `38b7188237d15b95011cccdc80f99a5d8bce051e8d9b0d11a457b968dd3171b4`。

ZIP 为本地研究保留档案（Git 忽略的 artifacts 内），不将大模型/垃圾副本上传 GitHub。源码/构建器/技术文档和中期精选来源记录已在 GitHub；全部外部结果另在上述小型档案中。归档逐 entry SHA256 已在删除前校验，删除后整包 SHA256/CRC/数量再次通过。

旧报告中的原始 Demo 路径保留历史含义，不伪装为当前仍可直接访问；恢复映射在 duplicate_hash_proof.csv.gz。唯一未退役的 MVP 目录内容为 Typer 随包 `.agents/skills/typer/SKILL.md`，6709 bytes，保持 hash；没有可执行 MVP Python。
