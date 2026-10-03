# Dataset Format — BuildReasonSeg Advisor RC1

## 1. 目录约定

```text
datasets/
└─ <name>/
   ├─ raw/          # 原始数据：影像 + 原始标注（**永不被修改**）
   └─ prepared/     # 由 prepare_dataset.py 生成（可删除重建）
```

命令行：

```bat
python prepare_dataset.py --dataset datasets/MyDataset --format coco
python prepare_dataset.py --dataset datasets/MyDataset --format instance-mask --overwrite
```

- `--format {coco, instance-mask, vector}`；
- `--config PATH`（默认 `configs/train.yaml`）；
- `--overwrite`：重建 `prepared/`；**`raw/` 永远不会被修改**；
- 若 `prepared/` 已存在且未指定 `--overwrite`，命令会拒绝覆盖并给出提示。

## 2. 标注要求（重要）

完整的 BuildReasonSeg 训练需要**建筑实例级标注**（每个建筑一个独立实例 id），因为推理链依赖
“最大建筑参考 → 方向筛选 → 最近建筑”的实例级关系。

- **拒绝**：普通二值 semantic mask 且无法可靠恢复实例的情形（会明确报错说明原因，而不是静默降级）；
- **接受**：instance mask（每实例独立 id / 连通域可唯一恢复）、COCO instance annotations、
  矢量（vector）多边形并带实例标识。

## 3. 划分（split）规则

1. **scene split 必须在 tiling 之前完成**；
2. 用户已提供 split（`prepared/splits.json` 或数据集自带的 split 文件）→ **尊重用户划分**；
3. 用户未提供 → 之后使用**确定性的 source-scene split**（同一 source scene 不得跨 split，避免同源泄漏）；
4. `train` 用于梯度更新，`val` 用于模型选择，`test` 只在用户显式 `--split test` 时才可访问。

## 4. 影像输入约定

- 模态：**RGB 光学遥感影像**；
- 默认瓦片 512×512；更大的影像由外层大图编排处理（tile size / overlap / merge threshold 参数
  **尚未冻结**，将在后续任务定义）；
- 支持的扩展名（推理入口）：`.tif/.tiff/.png/.jpg/.jpeg/.bmp/.webp`；
- 不支持：SAR、红外、原始多光谱。

## 5. Task 8A 范围

Task 8A **只冻结** CLI、目录 schema 与 validation contract；三种 adapter（COCO / instance-mask / vector）
的完整实现属于后续任务（Task 8C）。当前执行 `prepare_dataset.py` 会返回
`NOT_IMPLEMENTED_IN_TASK_8A`（非零退出码），不会生成任何伪造产物。
