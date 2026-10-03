# datasets/ — 用户数据目录（Task 8A 为空）

本目录用于放置用户自有建筑数据集。约定：

```text
datasets/
└─ <name>/
   ├─ raw/          # 原始影像与原始标注（**永不被修改**）
   └─ prepared/     # 由 prepare_dataset.py 生成（可删除重建）
```

- 完整 BuildReasonSeg 训练需要**建筑实例级标注**（每栋建筑一个实例 id）；无法可靠恢复实例的二值 semantic
  mask 会被明确拒绝；
- **scene split 必须在 tiling 之前完成**；用户已有 split 会被尊重，否则之后使用确定性的 source-scene split；
- 输入模态：**RGB 光学遥感影像**（不支持 SAR / 红外 / 原始多光谱）。

用法（契约已冻结，实现待后续任务）：

```bat
python prepare_dataset.py --dataset datasets/MyDataset --format coco
```

Task 8A 不包含任何数据集 adapter 实现，也不会在此目录生成任何数据。
