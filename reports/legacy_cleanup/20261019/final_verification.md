# 最终完整性与软件检查

结果：PASS（执行者自检，保留用户审查）。依据 integrity_result.json、hash_gate_after.json、execution_result.json、software_verification.json。

|检查|结果|
|---|---|
|删除前 Git branch/local HEAD/remote HEAD|独立分支、清单 checkpoint 已推送；执行时与 upstream 同 SHA|
|执行安全检查|所有 71,578 路径先 resolve、file ID/size/mtime、跟踪/保留排除；仅白名单 unlink|
|实际删除与失败|71,578 删除，0 失败；没有额外删除|
|起始跟踪文件|1498/1498 SHA256 一致（两份 handoff 授权更新）|
|关键资产|6443 条保留记录 PASS；15947 个唯一文件 post hash 检查|
|冻结最终权重|6/6 SHA256 一致；唯一历史权重保留，无模型加载/挑选|
|外部完整 RC1|整目录保留 hash；历史 input/output/logs 与既有 grandfathered Ultralytics 配置全部不变|
|canonical source/config|135/135 SHA256 一致；external source_manifest 与 canonical byte/hash 一致|
|中期检查成果及原始来源|全部原字节保留|
|原始数据与 split/GT|关键哈希一致；17,388 原始 tile 的身份/大小/mtime 一致；8078 项外部保留副本 hash 一致|
|两套退役 Demo|原部署目录已移除；410 个非模型文件保留在 2 个完整 CRC/hash 校验 ZIP|
|其他路径变更|unexpected missing/changed/additions = 0（仅允许报告/归档/Git/handoff 的授权变动）|
|安装/认证/其他课程项目|未删除，已扫描安装根文件路径/大小/mtime 对比无异常；拒绝访问目录保留|
|Codex|活动进程仍存在；当前工具会话持续可用，未停机/重装/清理其依赖|
|Git / 保留 Python|Git version、proposal Python stdlib 与已安装包元数据读取通过；不加载真实模型|
|DSH|安装程序存在且元数据未变；未运行，未进行完整 GUI 启动测试|
|治理 / 源码 / 科学结论|无修改|
|真实模型推理 / 训练 / 算法优化|0 / 0 / 0|

清理前 git fsck --connectivity-only exit 0，两项 dangling blob 属既有状态，未清理 Git 数据。清理后同项结果见 git_fsck_after.txt，exit code 0。

补充只读核对曾把 Windows CRLF 工作树 manifest 与 external LF 比较而记录 false；该结果保留在 integrity_first_worktree_comparison.json。manifest_line_ending_forensics.json 证实起始工作树就有 686 个 CRLF、内容语义相同，external 与 Git 原始 blob 完全 byte/hash 一致；正式检查改用真实 Git bytes。没有同步、改文件、豁免指标或改变科学/验收规则。

已知保留：唯一历史 checkpoint、完整 RC1/proposal 环境、新方向可能复用的原始数据/通用 pretrained 权重、全部科研记录；未把全部 best/last 当同一文件。旧 MVP 残留 6709-byte Typer 随包 skill，是删除前 guard 保留项。其他项目不可读 pytest cache 未测量、未清理。

本次没有测试模型科学性能，也未重跑历史 suite/Final Demo。旧目录清理不改变历史评价结果；旧报告中的历史原路径与本次恢复索引应结合阅读。

最终 handoff 使用 AGENTS 规定的 NONE/AWAITING_SUPERVISOR neutral idle，branch/head/last_checkpoint 为 null。实际最终 HEAD 在 push 后从 Git 核对并向用户报告，不写自引用 SHA。停止旧架构主动研究。
