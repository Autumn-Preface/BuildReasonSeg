#!/usr/bin/env python
"""Task 6B: build the deterministic train/val/paired subsets.

    python scripts/task6b_select_subsets.py

Writes `evaluation/task6b_subset_ids.json`. Nothing is trained here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import subsets  # noqa: E402

OUTPUT = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    payload = subsets.build_all()
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"train   : {payload['sizes']['train']} records, {payload['train']['n_images']} images")
        print(f"          by level {payload['train']['by_level']}")
        print(f"          by query type {payload['train']['by_query_type']}")
        print(f"          image reuse {sum(v['image_reuse_count'] for v in payload['train']['per_level_selection'].values())}")
        print(f"val     : {payload['sizes']['val']} records, {payload['val']['n_images']} images")
        print(f"          by level {payload['val']['by_level']}")
        print(f"paired  : {payload['sizes']['paired']} records / {payload['paired_probe']['n_pairs']} pairs")
        print(f"test used: {payload['test_split_used']}")
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
