# Task8F 实时推理过程 Demo V2

Status: IN_PROGRESS

范围：两个平级页面、同一个worker/同次推理、九阶段只读观测和轻量可视化。科学契约冻结，仅V2引擎副本可增加局部观测边界。

前置分支/HEAD精确核验，工作树干净。旧RC1全2184文件不变；V1的80项immutable清单不变，现有295文件含213项用户运行新增文件，全部锁定为保留基线，不删除、修改或复制进V2。Task8C/D/E与canonical/治理身份已锁定。

设计：结果 | 推理过程，过程页可滚动两列九阶段卡片。模型阶段快照为只读副本，通过有界后台渲染队列落盘与推送主线程。失败留存已完成阶段，D-B1候选与最终guard结果严格分离。Stage6/7的W/A/q/C观测只激活于既有实际模型forward，不使用其前置诊断计算冒充真实forward状态。无额外模型调用、无GT、无阈值/算法变化。

计划checkpoint：Trace/UI fake tests；V2只读observer与等价性；组装和页面/迁移门禁；最终报告与证据。

## Trace/UI checkpoint

当前36项fake tests PASS (3.34s/exit0)：两页切换、主线程/worker隔离、同run事件、只读数组、实时显示先于worker结束、各失败留存、缺失观测不能静默当成功、持久化/重开。首次32passed/2failed为TracePage重置卡片索引错误及其后继UI事件失败，真实记录留存在JSON，已L1局部修正并停止关闭时的progressbar timer。无外部组装或真实推理。
