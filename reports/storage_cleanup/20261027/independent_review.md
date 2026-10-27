# 本地存储清理独立范围复查

日期：2026-10-27。reviewer：独立子代理 `/root/independent_p0_review`。观察分支 `codex/local-storage-cleanup-20261027`，起点 `4dcabff660de82a0b8f585da6c2c22dea9bafc9f`。本轮先读仓库启动文件，再审授权、清单、保护快照及执行器。reviewer 没有删除文件、下载资源、训练或运行模型；仅按授权写本报告。

## 被审查的明确计划

[plan.json](plan.json) 的 SHA256：

`79e8015a4eb6f53ddfe368d7890e922c54d3ba43eddeb2721965901d8801670f`

|类别|文件数|拟删除逻辑字节|独立核对依据|
|---|---:|---:|---|
|6 个明确历史 pytest/fake 临时目录|670|6490476|固定 fixture 测试来源；没有模型权重；生成源码、正式测试日志及截图在 Git 中；唯一完整原始错误日志另行保留|
|任务私有 `artifacts/damage_p1a/pip_cache`|9|145265|仅 pip HTTP/selfcheck 安装缓存；已安装独立环境及版本/路径证据保留，不是模型运行时输入|
|4 份 Drive 目录 HTML|4|1292384|只是传输/目录缓存；对应原始目录科学字段已存已提交 inventory/source_access；README/LICENSE 保留|
|2 张 val 的精确重复 JPG 副本|2|7129386|各自有完整字节 SHA256 相同的 train 名称副本留在本地，可不依赖远端恢复|
|合计|685|15057511|约 14.36 MiB 逻辑数据；不是已释放空间，也不应宣传为大规模腾空|

临时目录限 `.tmp/demo_r1_57_final`、`.tmp/demo_r1_final_2334`、`.tmp/demo_r1_fake_evidence`、`.tmp/damage_p1a_tests_04`、`.tmp/damage_p1a_tests_05`、`.tmp/damage_p1a_tests_06`。其中 `.npz` 位于固定 fake pipeline 的 diagnostics，`.log` 为零字节 fixture 日志；没有把它们当作真实科研 checkpoint/运行结果。`.tmp/demo_r1_raw_test_attempts` 的两份唯一原始失败日志不在删除清单内。

## reviewer 实际执行的只读检查

- 对 **685 个目标逐一重算 SHA256、大小、mtime_ns**，与计划全部相符；硬链接数均为 1。逐级检查没有 reparse point，解析后的目标仍在本仓库范围内；没有已跟踪 Git 文件或 `.pt/.pth/.ckpt/.safetensors` 目标。
- 拟删空目录均位于上述七个明确树根内（六个临时树加私有 pip 缓存），没有包含 datasets、环境、RC1、V3 或其他外部目录。
- 读取 **9484 条保留哈希记录**及 **85250 条环境元数据记录**，与删除目标集合交集均为空。此处是集合/范围审查，**不是 reviewer 重算了全部保留文件哈希**；主执行者仍需完成删除后保护核验。
- 重新读取并哈希两张拟删 val JPG 及相应 train 副本，大小及文件 SHA256 相等：`val_SP0_818.JPG → train_SP1_813.JPG` 为 `ad5ef01c9a6a81eb183ba19e5ee8a2f419eaaad6e567c8c7c8e2a65c7ff7e781`；`val_SP0_1102.JPG → train_SP1_1091.JPG` 为 `f64361a7117be8879813cfb373131719a4911b440265441bc532710f3eddd0ff`。plan 中已有 retained_copy 路径及身份映射。旧研究报告的 val local 路径会变为历史位置，未来用留存字节副本恢复即可；不修改历史 manifest 或重解释 split。
- 从 Git HEAD 直接读取已提交 `official_archive_inventory.json`，重解析四份 HTML，其名称/ID 对应全部吻合（根目录 3 条，三个 split 各 2 条）。另读 `verify_html_cache.py`，其完整 description 比对结果由执行者保存。删除的是原始传输 HTML，不宣称它能按旧日期逐字节重新下载；科学目录信息和来源身份并未只留在 HTML 中。

初次读历史 pytest 目录在沙箱下遇权限拒绝；经授权的只读扩大权限重查后，685 项检查完整通过。没有通过更改 ACL 或删除文件来绕过检查。

## 保留决策与依赖

保留完整 RC1、V3、全部 checkpoint/环境、P0 原样，以及原研究源码/报告/正式输出，符合当前任务边界。未发现运行时或正式实验必须读取本计划中的 pip/HTML/固定 fake 缓存；不只根据目录名字推断安全，而是结合代码入口、文件清单和既有测试/证据记录核对。

两份旧 Demo 非模型 ZIP 含唯一历史运行记录，不能因存在 V3 就当成普通冗余运行包删除。三份 CUBIT 标签 ZIP 也保留：[source_header_checks.json](source_header_checks.json) 显示当前 HEAD 为 HTML/长度 0，**不能证明同版本可恢复**；HTTP 200 不等于 ZIP 可下载。用户之后可能从百度取得新资源，不构成删除唯一旧版本标注的现时依据。保留源 README/LICENSE、`protected_baseline.json` 和错误原日志亦合理。

## 删除执行器抽查与后验要求

已读 `execute_plan.ps1`：硬编码仓库根、七个树根及六个单文件白名单；强制提供被审 plan 哈希；拒绝越界/reparse；全清单和保留副本在首次删除前先哈希检查；每个文件删除前再次核 SHA；使用 `Remove-Item -LiteralPath` 逐文件删除，没有递归删除。仅对已空目录调用非递归 `Directory.Delete(path,false)`，新未知子项不会被顺带移除。已存在执行日志时拒绝盲目重跑，有利于保留部分执行现场。

活动进程门禁的初版按 CommandLine 中完整项目路径筛查；经 reviewer 建议，执行者已加入 ExecutablePath 位于项目内的判断及外部 BuildReasonSeg 交付路径匹配，reviewer 已读回确认。实际执行时仍须保证没有当前测试、下载或训练使用这些目录；本 reviewer 没有将静态文件审查冒充完整活动进程验证。

已读 `verify_cleanup.py`：核对确切删除日志/目标缺失、保留文件 SHA/大小、全环境大小/mtime/文件集合及关键 SHA、V3 manifest 和维护分支 Git 范围。两份预期 handoff 更新应单独留旧副本并明示排除，不能据此放宽其他科研文件保护。清理后的实际结果、V3 启动与磁盘读数尚由主执行者检查，不由本报告预先宣称通过。

## 范围结论

**对上述固定哈希计划，没有发现用户既有清理授权范围内实施它的确定阻断项。**安全依据依赖当前清单与所核文件身份，不适用于追加未知文件或扩大目录。若路径/内容/留存副本/活动依赖发生变化，应停止相应项并保留现场。

本结论仅为删除前的独立工程范围复查，不是已经删除、已经释放磁盘空间、最终保护核验或 Supervisor 科学验收。P1A 准入仍关闭；不触发下载、训练或研究阶段推进。
