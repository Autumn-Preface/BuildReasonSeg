# 试标注记录规范修订草案 v0.2

本文件为 CUBIT-TEXT-PILOT-01 工程记录草案，等待 Supervisor 审定。继承 v0.1 的只读原始数据、未知域、未确认类别、重复隔离和人工盲读原则；不修改原草案、不冻结科学协议。

## 记录与版本

`schema_version=cubit-text-pilot-draft-1`。每条指令单独 annotation_id（UUID hex），每张图可多条草稿。source 固定保存原 image_id、图像/标签 SHA256、官方 train、原尺寸；source_instance_ids 保存本图所有稳定 ID，edit.target_instance_ids 保存人工选择集合。稳定实例 ID 沿用原图 ID / 原始行号 / 原始行文本哈希前16位；完整行哈希及原数字类别在 samples.json 可回查。

edit 保存 operator、reviewer、review_status、instruction_text、task_level、reasoning_kind、references、relations、regions、valid_domain、risks、notes、visual_class_provisional、change_reason。创建/更新时间为服务器 UTC ISO8601；每版保存完整快照和 history，含操作人与审核人、修改说明、状态及上一版文件 SHA256。身份字段为本地自报，未经身份认证；不能据此证明独立双人复核。空白审核人不算已审核。

原子保存：同目录临时文件写完、flush/fsync 后，以不覆盖的本地硬链接发布 revision；只读加载最新完整版本。过期 expected_revision 被拒绝。没有自动合并、自动通过、数据库或正式导出。发布失败不会覆盖上一版；异常断电/人为编辑 JSON 不属于已验证恢复保证，错误必须保留现场。

## 三层内部元数据

- direct：直接指定损伤；reasoning_kind=none。
- position：位置或对应关系；reasoning_kind=none。
- reasoning：需推理；reasoning_kind=visual_semantic 或 spatial。

这些字段及其中文类别名、原数字类别、文件名前缀、目标 ID、筛选规则、框坐标和答案元数据不得拼入 instruction_text。界面只让人工编辑文字，服务器拒绝常见内部标识；正则校验不等于彻底防止语义泄漏，仍须人独立盲读。没有自动模板生成和模型调用。

## 几何、有效域和 Mask

原多边形永不编辑。坐标沿原始未旋转图像，x向右、y向下，像素范围 [0,W]×[0,H]。参照物/有效域/未知域用 bbox_normalized=[xmin/W,ymin/H,xmax/W,ymax/H]，0至1、严格非空、有限数；参照物可多个，关系指向有效 ref ID 与当前目标 ID 子集。构件类别名称为人工可见描述，不代表结构功能鉴定或成因判断。最近/最大等关系须在 evidence 说明比较集合和量度；不能把图像距离当真实三维距离。

regions.kind=domain_proposed 或 unknown；valid_domain 默认 unknown，只可记录 target_region_proposed/class_region_proposed/full_image_proposed 建议，不把背景或未标类变成负样本。显示 Mask 采用 Canvas even-odd 多边形填充；各原实例独立显示，重叠会视觉累积透明度，没有生成合并 GT 或正式训练 Mask。自交、边缘精度和栅格器差异均待人工审核。

## 状态和风险

review_status 只允许 draft、needs_clarification、rejected、ready_for_human_review。退回须 notes 有原因；送审须有指令和目标集合。送审不是 accepted，不自动改变训练准入。正式接收状态与训练导出需另立审查任务。

risks 保存 target_uniqueness、fragmentation、missing_annotation、occlusion、truncation、alignment、valid_domain、spatial_relation、visibility；勾选代表待核问题，无勾选不等于已通过检查。具体情况/物理损伤范围/组合依据写入 notes。历史13零面积记录、222 CRC+size候选而仅两组可定位SHA证据、官方类映射/分组/完整性未知均不修正。

本轮 human_confirmations=0，正式训练样本=0。仅工具测试记录使用 TOOL_TEST_NOT_HUMAN，独立存于 qa_drafts，不参与人工试标注。
