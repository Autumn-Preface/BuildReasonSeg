"""Task 6H section 4: build the canonical counterfactual pair manifest.

Verifies the 240 Task 6C `P` pairs (one source image, two different instructions, two different
target components, different masks, deterministic order), records the downsampled soft-target
overlap of every pair (never discarding overlapping pairs) and hashes the canonical pair identity
list. Writes `evaluation/task6h_pair_manifest.json`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.counterfactual import mask_overlap_diagnostics, pair_manifest  # noqa: E402

from task6h_common import (  # noqa: E402
    EVAL,
    PAIR_MANIFEST,
    canonical_train_pairs,
    load_pair_config,
    write_json,
)


def main() -> int:
    started = time.time()
    cfg, payload = load_pair_config()
    grid = int(cfg["dense_grounding"]["grid"])
    triples = canonical_train_pairs(payload)

    overlap_rows = []
    for pair, record_a, record_b in triples:
        sample_a = data_mod.to_sample(record_a)
        sample_b = data_mod.to_sample(record_b)
        overlap_rows.append(
            mask_overlap_diagnostics(sample_a.target_mask(), sample_b.target_mask(), grid)
        )

    manifest = pair_manifest(triples, grid, overlap_rows=overlap_rows)
    manifest["seconds"] = round(time.time() - started, 2)
    manifest["train_split_only"] = True
    write_json(PAIR_MANIFEST, manifest)
    print(
        f"[task6h:pairs] {manifest['pair_count']} pairs over {manifest['image_count']} images; "
        f"hash {manifest['pair_identity_sha256'][:16]}; overlapping pairs "
        f"{manifest['overlap_summary']['pairs_with_overlapping_targets']} "
        f"(max soft IoU {manifest['overlap_summary']['max_raw_iou']:.3f})",
        flush=True,
    )
    print(f"[task6h:pairs] wrote {PAIR_MANIFEST.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
