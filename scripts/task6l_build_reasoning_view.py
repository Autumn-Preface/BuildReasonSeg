"""Task 6L Part C/D: materialise the reasoning view of the canonical native-vector dataset.

The frozen reasoning stack (annotation + relation execution) consumes

    metadata JSONL (image_id, split, width, height, component_map, components[...])
    + a uint8 instance-index map per tile

so this step writes exactly that view for one split view of `WHU-EA-NativeVector` v1.0. It is a
large regenerable cache and therefore lives under the gitignored `artifacts/whu_native_vector/`.

    python scripts/task6l_build_reasoning_view.py --split-view scene_disjoint_v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from PIL import Image  # noqa: E402

from buildreasonseg_mvp.native_vector_adapter import (  # noqa: E402
    SPLITS,
    NativeVectorDataset,
    image_record_for_reasoning,
)
from buildreasonseg_mvp.whu_native_vector import (  # noqa: E402
    CACHE_REL,
    DATASET_ROOT,
    REASONING_VIEW_ROOT,
    write_json,
)


def view_root(split_view: str) -> Path:
    return REASONING_VIEW_ROOT / split_view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-view", default="scene_disjoint_v1")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    dataset = NativeVectorDataset(args.dataset_root)
    root = view_root(args.split_view)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    summary = {"split_view": args.split_view, "splits": {}}
    for split in SPLITS:
        component_dir = root / "components" / split
        component_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "metadata" / f"{split}.jsonl"
        tiles = 0
        instances_total = 0
        writer = metadata_path.open("w", encoding="utf-8", newline="\n")
        try:
            for view in dataset.iter_tiles(split, split_view=args.split_view):
                label_map = dataset.label_map(view.tile_id)
                if label_map is None:
                    continue
                relative = f"{CACHE_REL}/reasoning_view/{args.split_view}/components/{split}/{view.tile_id}.png"
                Image.fromarray(np.asarray(label_map, dtype=np.uint8), mode="L").save(
                    component_dir / f"{view.tile_id}.png", optimize=True
                )
                record = image_record_for_reasoning(
                    dataset, view.tile_id, split, split_view=args.split_view, label_map_path=Path(relative)
                )
                writer.write(_json_line(record))
                tiles += 1
                instances_total += len(record["components"])
                if not args.quiet and tiles % 2000 == 0:
                    print(f"[6l.view] {split}: {tiles} tiles ({time.time() - started:.0f}s)", flush=True)
        finally:
            writer.close()
        summary["splits"][split] = {
            "tiles": tiles,
            "instances": instances_total,
            "metadata": f"{CACHE_REL}/reasoning_view/{args.split_view}/metadata/{split}.jsonl",
        }
        print(f"[6l.view] {split}: {tiles} tiles, {instances_total} instances", flush=True)

    summary["runtime_seconds"] = round(time.time() - started, 2)
    summary["dataset_version"] = dataset.dataset_version
    write_json(root / "view_manifest.json", summary)
    print(f"[6l.view] wrote reasoning view for {args.split_view} in {time.time() - started:.0f}s", flush=True)
    return 0


def _json_line(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
