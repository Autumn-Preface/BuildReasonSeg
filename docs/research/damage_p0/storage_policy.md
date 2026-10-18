# Damage Research 存储管理 v0.1

适用新研究目录；旧RC1冻结资产、Task7I/7J、GT/split、完整交付和Demo归档继续受原保护，
不按以下缓存规则清理。P0未删除旧文件、未修改旧环境依赖。

## 单份资产与登记

- Git：治理、源码、独立配置、CSV/JSON登记、Markdown、带署名的少量证据PNG。
- 原始数据：`datasets/damage/local/raw/<dataset>/<version>/`（未来P1使用）；P0定向样本在
  `datasets/damage/local/{cubit,deepcrack,cif}/`。整个local由目录内.gitignore排除。
- 官方文档/论文参考缓存：P0为`evaluation/damage_p0/sources/`，目录内.gitignore仅跟踪自身；
  不把整篇PDF/网页复制到GitHub。访问元数据、原文链接、hash在上层跟踪，报告给出判断依据。
- 新权重：未来在`datasets/damage/local/models/<provider>/<revision>/`或经审定单个外部管理根，
  不混入旧`artifacts/checkpoints`；**P0未下载新权重**。外部路径需有具体写授权。
- 每资源登记source URL、license核查日期、版本/revision、下载前bytes、实际bytes、SHA256、
  解压比例、责任任务和保留原因；只有真实下载后填本地SHA256，HEAD元数据不是文件hash校验。
- 原始归档和解压版原则选一个作为长期源；如须保留两者，预算双份并写恢复理由；不同时下载
  CiF Full/Tiled作为两套“独立”数据，也不把同模型不同框架缓存各存一套。

## 下载预算与生命周期

本机卷起始空闲约144GiB。建议（待P1审定）保持系统余量≥50GiB、Damage活动资产软上限60GiB。
下载前测可用空间和Content-Length/官方体积，估计下载+解压+转换+checkpoint+临时峰值；
未知大小用限制范围/流量，超预算停止受影响下载并提具体扩容/删哪些新缓存的方案。
CUBIT训练images.zip实测15034071180 bytes、labels.zip248950377 bytes；仅train原始约14.23GiB，
尚不含val/test/解压/其他派生。CiF官方API全包约122.96GB，当前不全量获取。

范围读取先确认HTTP206和Content-Range，单请求≤8MiB、每远端ZIP≤24MiB；不支持Range就拒绝回落
整包下载。P0全部取回量和本地占用分别记录，不能把14GB远端包大小当本次下载量。

缓存可重建性分类：

|资产|删除/保留规则（不自动执行）|恢复依据|
|---|---|---|
|原始数据、人工标注、license快照与manifest|长期单份；只有可验证备份且明确授权才删|原来源、版本/hash；网络入口存在不保证可恢复|
|派生tiles/栅格mask|正式split/转换器/config/hash固定后可重建；run结束审查删除|源图、GT、转换器commit、参数、seed，禁止删唯一GT|
|特征缓存、下载残片|登记任务与依赖；任务关闭后列清单再安全删除|可重复pipeline；不可重建内容转长期资产|
|checkpoint|保留最终审定模型、恢复训练last、对应配置/种子/指标；其他只按本研究明示保留策略清理|不能仅看best/last名称或体积；unique scientific状态默认保留待审|
|P0源快照与样本|留作待审证据，约几十MiB；审查后可重取缓存另列清理清单|source_access/sample_access及hash；样本不是整库替代|
|旧Demo ZIP与冻结实验|**不属于缓存**，P0禁止清理|legacy_cleanup/preserved_assets及hash索引|

任何删除先解析absolute path，确认处于本研究授权根、未知文件保留、列清单/恢复来源/hash、
必要时提交推送检查点再操作。不用git clean/stash/reset等替代管理，也不清用户账户全局缓存。

## 本阶段增量检查

[storage_final.json](../../../evaluation/damage_p0/storage_final.json)按新目录逐文件逻辑字节统计，
包括Git忽略样本/源缓存，列出可重建缓存。它不是NTFS实际分配空间，也不等于卷空闲变化。
Git对象新增和其他应用并发写入会使卷空闲差额不同；不宣称释放了空间。P0不清理全部缓存。
最终资产保护检查见 [integrity_final.json](../../../evaluation/damage_p0/integrity_final.json)。
