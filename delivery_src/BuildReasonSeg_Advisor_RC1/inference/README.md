# inference/ — RC1 推理输入与独立 run 输出目录

```text
inference/
├─ input/                        # 放入待推理影像（RGB 光学遥感影像）
└─ output/
   └─ <run_slug>/
      ├─ diagnostics/            # 诊断文件直接放在此处，不再嵌套 <stem>/
      ├─ masks/                  # <stem>_mask.png 或 <stem>_mask_001.png 等
      └─ overlays/               # <stem>_overlay.png 或 <stem>_overlay_001.png 等
```

首个可用 run 为 `<stem>`，后续为 `<stem>_001`、`<stem>_002`…。目录以独占创建方式保留，
失败 run 即使没有 mask，也占用自己的 run 名；已有 run 输出不会被静默覆盖。
Task 8B.4 之前的 shared `masks/`、`overlays/`、`diagnostics/` 及 review 历史产物保持原样，
不删除、不迁移、不重命名。如果输入 stem 与已有历史根目录名称相同，分配下一个 suffix。

RC1 runtime 的 diagnostics 文件名保持不变：

```text
prompt.txt, parsed_program.json, global_proposals.png, proposals.json,
selected_reference.png, reasoning_context.png, reference_context_mask.png,
direction_field.png, nearest_field.png, relation_weight.png,
prototype_similarity.png, maps.npz, result.json
```

`--inspect-proposals` 使用相同 run 目录，仅保存 diagnostics，不创建 mask 或 overlay。
`--no-save-diagnostics` 抑制诊断文件，diagnostics 目录可以存在但为空。
mask 保持原图尺寸、uint8、背景 0 / 目标 255；overlay 保持原图尺寸，透明度由 `--alpha`
控制（默认 0.45）。本任务不改变已有 tiling、模型、阈值或运行状态契约。

## Runtime result semantics

正常推理生成的 `result.json` 若记录：

```text
status = SUCCESS
```

则同时携带：

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

这只表示当前 RC1 runtime 已完成并通过现有结构性检查；semantic target correctness is not established。

因此 `result.json` 中的 `SUCCESS` 不能被解释为目标建筑在语义上已经正确匹配，也不能被解释为 mask 质量验收通过。
