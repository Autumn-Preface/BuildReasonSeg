"""Task 6M section 2: freeze the deterministic native-vector evaluation packs.

Fixes the Task 6L `paired_counterfactual_availability: null` gap before any training happens:

* `val fixed120` / `test fixed120` — 120 BuildSpatialReason v0.2 records, stratified over L1/L2/L3
  and over the program families;
* `val paired20` / `test paired20` — 20 same-image counterfactual pairs whose two members share the
  tile but have DIFFERENT native target instances (so a model cannot win by answering the same mask);
* a manifest that freezes the construction policy, the seeds and the file hashes.

Nothing here reads test metrics; the packs are frozen before training and before any threshold tuning.

    python scripts/task6m_freeze_eval_packs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"

#: Frozen construction constants (changing any of them invalidates the packs).
SEED = 20260810
FIXED_N = 120
PAIR_N = 20
PROGRAM_FAMILIES = {
    "extreme": ("leftmost", "rightmost", "topmost", "bottommost"),
    "size_rank": ("largest", "smallest"),
    "reference_to_nearest": (
        "largest_to_nearest", "smallest_to_nearest",
    ),
    "reference_to_direction": (
        "largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below",
        "smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below",
    ),
    "triple_composition": (
        "largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
        "largest_to_above_to_nearest", "largest_to_below_to_nearest",
    ),
}


def family_of(query_type: str) -> str:
    for family, programs in PROGRAM_FAMILIES.items():
        if query_type in programs:
            return family
    return "other"


def load_v02(split: str) -> list[dict]:
    path = V02 / f"{split}.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic stratified sample over (level, program family) with unique tiles.

    Cells are filled from ALL records first (not from a tile-deduplicated list, which would bias
    towards whichever program the generator emits first for a tile). Picks are round-robin over the
    cells, and a pick is accepted only when its tile has not been used yet.
    """

    by_cell: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        by_cell[(int(record["level"]), family_of(str(record["query_type"])))].append(record)

    rng = random.Random(seed)
    for cell in by_cell:
        by_cell[cell].sort(
            key=lambda r: (str(r["query_type"]), str(r["image_id"]), str(r["sample_id"]))
        )
        rng.shuffle(by_cell[cell])

    cells = sorted(by_cell)
    picked: list[dict] = []
    used_tiles: set[str] = set()
    exhausted = {cell: False for cell in cells}
    while len(picked) < n and not all(exhausted.values()):
        for cell in cells:
            if len(picked) >= n:
                break
            if exhausted[cell]:
                continue
            bucket = by_cell[cell]
            # find the next record in this cell whose tile is still unused
            chosen_index = None
            for index in range(len(bucket) - 1, -1, -1):
                if str(bucket[index]["image_id"]) not in used_tiles:
                    chosen_index = index
                    break
            if chosen_index is None:
                exhausted[cell] = True
                continue
            record = bucket.pop(chosen_index)
            used_tiles.add(str(record["image_id"]))
            picked.append(record)

    # Top up so EVERY program id appears at least once: swap one record of the most-represented
    # program for a record of each still-missing program (tile uniqueness is preserved).
    present = {str(r["query_type"]) for r in picked}
    all_programs = sorted({str(r["query_type"]) for r in records})
    missing = [program for program in all_programs if program not in present]
    for program in missing:
        candidate = None
        for record in records:
            if str(record["query_type"]) == program and str(record["image_id"]) not in used_tiles:
                candidate = record
                break
        if candidate is None:
            continue
        counts = Counter(str(r["query_type"]) for r in picked)
        for donor_program, _count in counts.most_common():
            if donor_program == program:
                continue
            for index, record in enumerate(picked):
                if str(record["query_type"]) == donor_program and str(record["query_type"]) in present:
                    used_tiles.discard(str(record["image_id"]))
                    picked.pop(index)
                    break
            else:
                continue
            break
        picked.append(candidate)
        used_tiles.add(str(candidate["image_id"]))
        present.add(program)
    picked.sort(key=lambda r: str(r["sample_id"]))
    return picked


def build_pairs(records: list[dict], n: int, seed: int, dataset: NativeVectorDataset) -> list[dict]:
    """Same-image counterfactual pairs with different native targets.

    A pair is (reference query, swapped query) on one tile: both members name the SAME reference
    instance but ask for different relations/targets, and their targets must be different native
    instances, so a position-only heuristic cannot pass both.
    """

    by_tile: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_tile[str(record["image_id"])].append(record)
    candidates = []
    for tile, group in by_tile.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda r: str(r["query_type"]))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]
                if left["query_type"] == right["query_type"]:
                    continue
                left_target = int(left["target_component_id"])
                right_target = int(right["target_component_id"])
                if left_target == right_target:
                    continue
                # both members must reference the same native instance set (same-image counterfactual)
                if sorted(left.get("reference_component_ids", [])) != sorted(right.get("reference_component_ids", [])):
                    continue
                if int(left["level"]) != int(right["level"]):
                    continue
                candidates.append((tile, left, right))
    candidates.sort(key=lambda item: (item[0], item[1]["query_type"], item[2]["query_type"], item[1]["sample_id"]))

    rng = random.Random(seed)
    used_tiles: set[str] = set()
    pairs: list[dict] = []
    # prefer pairs whose two members come from DIFFERENT program families (a stronger counterfactual),
    # then fall back to any remaining pair in a deterministic order.
    pool = [(tile, left, right) for tile, left, right in candidates
            if family_of(str(left["query_type"])) != family_of(str(right["query_type"]))]
    fallback = [(tile, left, right) for tile, left, right in candidates if (tile, left, right) not in pool]
    for source in (pool, fallback):
        while len(pairs) < n and source:
            index = rng.randrange(len(source))
            tile, left, right = source.pop(index)
            if tile in used_tiles:
                continue
            used_tiles.add(tile)
            pairs.append(
                {
                    "pair_id": f"{tile}::{left['query_type']}::{right['query_type']}",
                    "tile_id": tile,
                    "split": left["split"],
                    "level": int(left["level"]),
                    "query_type_a": str(left["query_type"]),
                    "query_type_b": str(right["query_type"]),
                    "target_instance_a": int(left["target_component_id"]),
                    "target_instance_b": int(right["target_component_id"]),
                    "source_feature_a": ((left.get("native_vector") or {}).get("target") or {}).get(
                        "source_feature_id"
                    ),
                    "source_feature_b": ((right.get("native_vector") or {}).get("target") or {}).get(
                        "source_feature_id"
                    ),
                    "instruction_zh_a": left["instruction_zh"],
                    "instruction_zh_b": right["instruction_zh"],
                    "instruction_en_a": left["instruction_en"],
                    "instruction_en_b": right["instruction_en"],
                    "sample_id_a": left["sample_id"],
                    "sample_id_b": right["sample_id"],
                }
            )
    pairs.sort(key=lambda p: p["pair_id"])
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    dataset = NativeVectorDataset()
    outputs = {}
    for split in ("val", "test"):
        records = load_v02(split)
        fixed = stratified_sample(records, FIXED_N, SEED + (0 if split == "val" else 1))
        pairs = build_pairs(records, PAIR_N, SEED + (100 if split == "val" else 101), dataset)
        if len(pairs) < PAIR_N:
            print(f"warning: only {len(pairs)} pairs available for {split}", file=sys.stderr)

        fixed_payload = {
            "_doc": (
                f"Task 6M section 2: frozen {split} fixed120 evaluation pack. Deterministic stratified "
                "sample over (level, program family), one record per tile."
            ),
            "task": "6M",
            "split": split,
            "construction": {
                "seed": SEED + (0 if split == "val" else 1),
                "target_size": FIXED_N,
                "stratification": ["level", "program_family"],
                "program_families": PROGRAM_FAMILIES,
                "one_record_per_tile": True,
                "frozen_before_training": True,
                "dataset": "BuildSpatialReason v0.2",
                "instances": "WHU-EA-NativeVector v1.0",
            },
            "counts": {
                "total": len(fixed),
                "by_level": dict(sorted(Counter(int(r["level"]) for r in fixed).items())),
                "by_family": dict(sorted(Counter(family_of(str(r["query_type"])) for r in fixed).items())),
                "by_query_type": dict(sorted(Counter(str(r["query_type"]) for r in fixed).items())),
            },
            "records": [
                {
                    "sample_id": r["sample_id"],
                    "tile_id": r["image_id"],
                    "level": int(r["level"]),
                    "query_type": r["query_type"],
                    "family": family_of(str(r["query_type"])),
                    "target_instance": int(r["target_component_id"]),
                    "target_source_feature": ((r.get("native_vector") or {}).get("target") or {}).get(
                        "source_feature_id"
                    ),
                    "reference_instances": [int(v) for v in r.get("reference_component_ids", [])],
                    "instruction_zh": r["instruction_zh"],
                    "instruction_en": r["instruction_en"],
                }
                for r in fixed
            ],
        }
        pair_payload = {
            "_doc": (
                f"Task 6M section 2: frozen {split} paired20 counterfactual pack. Each pair is two "
                "programs on the SAME tile with the SAME reference instance but DIFFERENT native "
                "targets."
            ),
            "task": "6M",
            "split": split,
            "construction": {
                "seed": SEED + (100 if split == "val" else 101),
                "target_size": PAIR_N,
                "same_tile": True,
                "same_reference_instances": True,
                "different_target_instances": True,
                "same_level": True,
                "frozen_before_training": True,
            },
            "counts": {
                "total": len(pairs),
                "by_level": dict(sorted(Counter(int(p["level"]) for p in pairs).items())),
            },
            "pairs": pairs,
        }
        for name, payload in (
            (f"task6m_{split}_fixed120.json", fixed_payload),
            (f"task6m_{split}_paired20.json", pair_payload),
        ):
            path = EVAL / name
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                            encoding="utf-8")
            outputs[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if not args.quiet:
                print(f"[6m.packs] {name}: {payload['counts']}", flush=True)

    manifest = {
        "_doc": (
            "Task 6M section 2 manifest. Freezes the evaluation-pack construction policy, the seeds, "
            "the source dataset hashes and the resulting pack hashes BEFORE any training or tuning."
        ),
        "task": "6M",
        "purpose": "close the Task 6L paired_counterfactual_availability gap with frozen eval packs",
        "construction_policy": {
            "seed_base": SEED,
            "fixed120": "deterministic stratified sample over (level, program family), one record per tile",
            "paired20": "same tile, same reference instances, same level, different native target instances",
            "frozen_before_training": True,
            "test_packs_not_inspected_during_development": True,
        },
        "sources": {
            "build_spatial_reason_v0.2": {
                "train": "datasets/build_spatial_reason/v0.2/train.jsonl",
                "val": "datasets/build_spatial_reason/v0.2/val.jsonl",
                "test": "datasets/build_spatial_reason/v0.2/test.jsonl",
            },
            "canonical_instances": "datasets/whu_native_vector/v1.0/manifest.json",
            "split_view": "scene_disjoint_v1",
        },
        "artifacts": outputs,
        "runtime_seconds": round(time.time() - started, 2),
    }
    path = EVAL / "task6m_eval_pack_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[6m.packs] wrote manifest with {len(outputs)} packs in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
