# BuildReasonSeg 本地存储清理报告

任务日期：2026-10-27；用户暂停后，于 **2026-10-28** 明确恢复并完成最终检查。
执行结果：维护工作完成，等待 Supervisor 审核；不推进任何研究阶段。
分支：`codex/local-storage-cleanup-20261027`。起点：`4dcabff660de82a0b8f585da6c2c22dea9bafc9f`。

## 实际空间与删除结果

本机 `C:\D` 位于 C 卷，总容量 **991902896128 字节**。删除前已保存逐文件路径、大小、SHA256、依据及保留副本映射，并将固定计划推送至 GitHub。计划 SHA256：`79e8015a4eb6f53ddfe368d7890e922c54d3ba43eddeb2721965901d8801670f`。

|删除类别|文件数|逻辑字节|位置/依据|
|---|---:|---:|---|
|已完成的模拟/固定测试临时文件|670|6490476|`.tmp` 中六个固定测试目录，正式记录保留|
|P1A 私有安装缓存|9|145265|`artifacts/damage_p1a/pip_cache`，不含已安装依赖|
|已解析的数据目录 HTML 缓存|4|1292384|`artifacts/damage_p1a/source`，完整目录科学字段已有 Git 记录|
|原图精确重复副本|2|7129386|`duplicate_samples` 中两张 val JPG，完全相同字节的 train 副本保留|
|合计|**685**|**15057511**|约 **14.36 MiB** 逻辑文件大小|

同时以非递归方式移除 **1049 个空目录**。没有删除任何 Git 跟踪的历史文件，没有清理环境或全局缓存，没有发现可安全删除的大体积失败下载片段。清理量有限，不能宣称已为完整数据集腾出大量空间。

|实际磁盘读数（北京时间）|空闲字节|
|---|---:|
|2026-10-27 17:59:42，删除前|125446184960|
|2026-10-27 17:59:53，删除后|125461061632|
|删除窗口内净增加|**14876672（约 14.19 MiB）**|
|2026-10-28 00:33:49，恢复后最终实时读数|100488908800|

证据：[删除前读数](disk_immediately_before.json)、[删除后读数](disk_immediately_after.json)、[逐文件删除日志](deletion_log.jsonl)、[完整清单](cleanup_list.csv)。逻辑字节不等于卷净变化；分配单位和其他进程也会影响空闲量。恢复当天最终实时读数另存 [disk_final_resumed.json](disk_final_resumed.json)，跨夜变化不能归因于本次清理。最初扫描读数为 125614800896 空闲字节，也不能与删除窗口结果混用。

## 精确范围与可恢复映射

六个已清测试目录：`.tmp/demo_r1_57_final`、`.tmp/demo_r1_final_2334`、`.tmp/demo_r1_fake_evidence`、`.tmp/damage_p1a_tests_04`、`.tmp/damage_p1a_tests_05`、`.tmp/damage_p1a_tests_06`。固定模拟产物不是真实模型 checkpoint；完整原始失败日志另行保留。

两张已删重复图片原位于 `artifacts/damage_p1a/duplicate_samples`。未来如重跑历史审计需要原路径，可从同目录留存副本按原名复制恢复，不能据此重新解释官方划分或编辑历史报告：

|已删除名称|留存的相同字节副本|SHA256|
|---|---|---|
|`val_SP0_818.JPG`|`train_SP1_813.JPG`|`ad5ef01c9a6a81eb183ba19e5ee8a2f419eaaad6e567c8c7c8e2a65c7ff7e781`|
|`val_SP0_1102.JPG`|`train_SP1_1091.JPG`|`f64361a7117be8879813cfb373131719a4911b440265441bc532710f3eddd0ff`|

## 保留资产及待确认项目

- 三份完整 `artifacts/damage_p1a/raw_labels/{train,val,test}_labels.zip`，合计 **310486337 字节**。它们不是不完整片段。官方链接的 HEAD 检查返回 HTML/长度 0，未确认同版本可恢复，故按用户边界保留。此检查没有下载数据正文，也不能证明远端永久失效。待用户取得完整新版后，仍需核对版本、哈希和来源再决定是否删除旧份。
- `artifacts/legacy_archive/20261019` 中 V1/V2 历史 ZIP，合计 **55886533 字节**，含 GitHub 未完整保留的唯一历史运行证据，故保留为研究档案。交付目录实际只有 `BuildReasonSeg_Advisor_RC1` 和 `BuildReasonSeg_Demo_V3`；不存在额外 V1/V2 日常运行或独立交付目录。RC1 保留科研用途。
- `.tmp/demo_r1_raw_test_attempts` 两份唯一完整失败日志（432859 字节）、P1A 历史保护基线（673974 字节）、源 README/LICENSE 均保留。
- RC1 完整交付、全部科研 checkpoint/实验日志、`datasets/damage/local` 的 P0 原始样本/标注、Git 历史源码/报告、V3 模型和结果、全部 Conda 环境及 SAM2 editable 源依赖均保留。
- 启动时工作树干净；未发现尚未提交的有效实验成果或进行中的训练/下载任务。P1A 仍是数据准入停止状态，真实训练步数为 0。

## 清理后检查

[verification.json](verification.json) 记录真实检查结果：

- 685 项删除日志与固定计划完全一致，目标文件均已不存在。
- **9482 项保留文件 SHA256/大小核验通过**，包括旧科研源码/报告、权重、数据和外部交付。原快照 9484 项中，明确排除本任务预期更新的两份 handoff 文件；其旧内容另有快照，不宣称它们未变。
- **85250 项环境文件集合、大小、mtime 未变**，选定 Python/torch CUDA DLL/numpy/pycocotools 关键文件 SHA256 通过。没有安装、升级或清理环境依赖。此项不等于重新执行训练或 GPU 基准。
- V3 引擎 package manifest 的 **85 项文件**大小/哈希通过；V3 其他入口、配置、模型、结果也纳入保留快照检查。正常启动只允许自身运行缓存/日志变化。
- 恢复后完整自检输出“环境检查：通过”，正常 BAT 启动成功，实际观察到 V3 主窗口及同一通过状态。见 [自检输出](v3_self_check_resumed_unrestricted.txt)、[窗口观察](v3_window.json)、[窗口截图](v3_window.png)。截图中右侧桌面悬浮组件不属于 Demo。没有选择影像、执行推理或加载新模型；本次检查窗口随后关闭。
- 恢复时第一次受限自检因交付目录 `_runtime_cache` 写入探针被拒绝；随后以正常权限重试通过，未修改 ACL、入口或环境。失败现场保留于 [v3_self_check_resumed.txt](v3_self_check_resumed.txt)。
- Git 变更范围仅为本维护报告目录和两份 handoff；旧治理结论及研究源码/报告没有被改写。

## 审计与执行限制

独立代理已在删除前核对全部 685 目标身份、范围、重复副本及执行器，见 [independent_review.md](independent_review.md)。这是删除前独立范围复查；清理后核验由执行者完成，最终审核仍属 Supervisor。

执行器使用明确白名单、逐文件身份核对与字面路径删除，只移除已空目录；未使用递归无界删除、git clean/reset/stash、全局缓存清理。删除前检查没有项目任务依赖这些文件。此前受限读取问题通过只读正常权限复查解决，没有改 ACL。

本次没有下载任何数据集正文、生成标注文本、训练、调参或改算法。删除已完成，**不要重跑 execute_plan.ps1**。下一步仅等待维护审核；CUBIT 完整下载由用户自行进行，P1A/P1B/P2 不自动启动。
