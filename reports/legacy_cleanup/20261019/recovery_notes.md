# 恢复说明

默认不恢复、不重新部署旧 Demo、不启动推理。旧研究已结束，以下只为必要追溯保留。

1. **代码/治理/科研结论与成果附件**：保留在 Git 当前树、原始分支和完整历史中。起始成果分支 `codex/midterm-evidence-review-20261017` 未改写。所有新增报告在独立清理分支。
2. **Demo V1/V2**：从 artifacts/legacy_archive/20261019 相应 ZIP 解压至新的审阅目录；先对照 archives.json 校验 hash，再按 duplicate_hash_proof.csv.gz 中原路径映射，从 RC1 的对应 model 文件复制被省略的模型。archive entry 与被删除文件一一对应，不能用新预测替代历史结果。若未来需要运行，机器 Python locator 应与保留 proposal 环境核对；本次没有改档案或启动 Demo。
3. **Qwen 本地缓存**：完整 RC1 的 `model/components/program_head/Qwen3-VL-2B-Instruct/` 十个文件对应原 HF snapshot revision `89644892e4d85e24eaac8bacfd4f463576704203`。缓存 refs/树记录仍留存，按 proof 逐文件复制回原 snapshot 并验 hash 即可，不必联网更换版本。
4. **SAM2 与三个历史重复 checkpoint**：duplicate_hash_proof.csv.gz 给出逐字节一致的保留副本路径。只恢复同一 hash，不做新模型选择。
5. **旧基线数据副本**：从 C:\D\resources\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset 按对应相对路径恢复；8078 项 hash 已验证。此保留副本位于清理范围外，未移动/修改。
6. **Task6M YOLO 影像导出**：源 tile index、data.yaml、labels 和原始 TIFF 都保留。按 source_image_ref 恢复相同链接或副本即可。历史 scripts/task6m_build_export.py 自带删除整个导出目录的行为，**本次未执行它，日后不得为了恢复影像而盲目执行破坏性整树重建**。不要编辑仍有多个链接的原始影像。
7. **Task6N feature cache**：是冻结 SAM2.1 Hiera Base+ 对原始 RGB 的 float32 memoization，配置/encoder 源码/精确 pretrained hash/原始数据保留。需要时可重新计算，耗时/显存和跨运行逐位一致性不承诺；本次没有重新编码或推理。
8. **旧 MVP 环境**：environment_versions.json 和 environment/task6a_requirements_lock.txt 保留依赖信息；proposal 环境整套保留。被删除 clone 不承诺逐字节恢复，不能把版本清单当环境镜像；不得重新安装/破坏正在运行的 Codex。
9. **pip/bytecode/pytest cache**：按既有版本下载或正常使用时自然重建，不包含唯一科研结论；本次未运行模型或重试实验。

两个 Demo ZIP 只保留在本机，没有把重复大模型上传 GitHub。若本机存储未来丢失，ZIP 中未另行提交的全部原始运行细节不能仅从 GitHub 完整恢复；精选中期原图和来源记录已经在 Git。请将这约 53.3 MiB 保留档案纳入日常研究备份。本次不擅自访问外部备份服务。

所有唯一历史模型状态均保留，没有删除不可恢复的唯一训练权重。无法确认价值或来源的项目文件、认证与个人设置、不相关课程项目和不可读目录不处理。
