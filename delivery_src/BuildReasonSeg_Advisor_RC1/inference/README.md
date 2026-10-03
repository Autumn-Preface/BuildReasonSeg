# inference/ — 推理输入输出目录（Task 8A 为空）

```text
inference/
├─ input/                        # 放入待推理影像（RGB 光学遥感影像）
└─ output/
   ├─ masks/                     # 目标建筑掩膜
   ├─ overlays/                  # 可视化叠加（透明度由 --alpha 控制，默认 0.45）
   └─ diagnostics/               # 诊断产物（可用 --no-save-diagnostics 关闭）
```

诊断产物契约（Task 8A 冻结名称，实现待后续任务）：

- `proposals.json` — 检测/合并后的候选建筑及编号；
- `reference_selection.json` — 参考建筑选择结果（U-C1 确定性 largest 或 `--reference-id`）；
- `relation_fields.npz` — P_dir / P_near 关系场；
- `command_trace.json` — 指令解析与 validator 决策轨迹；
- `timing.json` — 各阶段耗时。

> Task 8A 尚未实现真实推理，因此本目录不会产生任何结果文件。
> 大图 tile size / overlap / merge threshold / context 参数**尚未冻结**，将在后续任务定义。
