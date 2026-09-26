#!/usr/bin/env python
"""Task 6C: build the frozen U and P training subsets.

    python scripts/task6c_select_subsets.py

Writes `evaluation/task6c_subset_ids.json` (U = Task 6B's exact 480 records with
unchanged ids; P = 240 unique images x 2 counterfactual instructions).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import subsets as subsets_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

OUT = REPO_ROOT / "evaluation" / "task6c_subset_ids.json"


def main() -> int:
    payload = subsets_mod.build_task6c()
    write_json(OUT, payload)
    u, p = payload["U"], payload["P"]
    print(f"[task6c] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    print(f"[task6c] U: {u['n_records']} records / {u['n_images']} images / levels {u['by_level']}")
    print(f"[task6c] P: {p['n_records']} records / {p['n_images']} images / levels {p['by_level']}")
    print(f"[task6c] P: {p['n_pairs']} pairs, different targets={p['all_pairs_different_targets']}, "
          f"different query types={p['all_pairs_different_query_types']}")
    print(f"[task6c] P: exact 80/80/80 feasible={p['composition_feasible_exactly']} "
          f"(eligible images {p['eligible_images_per_bucket']})")
    if not p["composition_feasible_exactly"]:
        print("[task6c] WARNING: the preferred composition was not exactly feasible; see the audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
