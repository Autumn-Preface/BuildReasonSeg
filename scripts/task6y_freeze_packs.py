"""Task 6Y Parts E — freeze the nearest-only packs from BuildSpatialReason v0.2.

Deterministic (seed 20260930) stable-hash selection from the frozen v0.2 train/val splits only, for exactly
the two canonical nearest program ids `largest_to_nearest` and `smallest_to_nearest`:

* `Y-Overfit20`   — train, 10 + 10 unique ids, at least 15 unique tiles when possible
* `Y-MiniTrain1000`— train, exactly 650 largest_to_nearest + 350 smallest_to_nearest
* `Y-MiniVal240`  — val, exactly 120 + 120
* `Y-PairedVal`    — val pairs on the same tile with different reference ids and different target ids;
                     first 20 by stable hash if >= 20 exist, otherwise all if >= 12, else STOP
                     `NEAREST_PAIRED_SET_INSUFFICIENT`

No directional or L3 record may enter a pack, and the test split is never touched. Packs are written to the
gitignored `artifacts/task6y/packs/`; `evaluation/task6y_pack_manifest.json` records ids and SHA256.

    python scripts/task6y_freeze_packs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6y" / "packs"
OUT = EVAL / "task6y_pack_manifest.json"
SEED = 20260930
NEAREST_PROGRAMS = ("largest_to_nearest", "smallest_to_nearest")
PAIRED_MINIMUM = 12
PAIRED_TARGET = 20


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(name: str) -> list[dict]:
    records = []
    with (DATA / f"{name}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("query_type") in NEAREST_PROGRAMS:
                records.append(record)
    return records


def tile_of(record: dict) -> str:
    """Tile id of a raw v0.2 record (`native_vector.tile_id` / `image_id`) or of a frozen sample."""

    if record.get("tile_id"):
        return str(record["tile_id"])
    native = record.get("native_vector") or {}
    return str(native.get("tile_id") or record.get("image_id"))


def image_path_of(record: dict) -> str:
    """Resolve the v0.2 `image_path` against the WHU source root (exactly as Task 6N packs do)."""

    from buildreasonseg_mvp.task6n_relation_decoder import WHU_SOURCE_ROOT

    relative = str(record["image_path"]).replace("/", "\\")
    resolved = Path(WHU_SOURCE_ROOT) / relative
    return str(resolved)


def to_sample(record: dict) -> dict:
    native = record["native_vector"]
    reference = native["references"][0]
    target = native["target"]
    return {
        "sample_id": str(record["sample_id"]),
        "tile_id": tile_of(record),
        "program_id": str(record["query_type"]),
        "relation": "nearest",
        "image_path": image_path_of(record),
        "reference_source_feature_id": int(reference["source_feature_id"]),
        "target_source_feature_id": int(target["source_feature_id"]),
        "target_instance_id": int(target["tile_instance_id"]),
        "reference_instance_id": int(reference["tile_instance_id"]),
        "level": int(record.get("level", 2)),
        "split": str(record.get("split", "")),
        "pair_key": tile_of(record),
        "metadata": {
            "reference_source": "oracle_native_gt",
            "component_map_path": str(record["component_map_path"]),
            "candidate_component_ids": list(record.get("candidate_component_ids", [])),
            "target_component_id": int(record.get("target_component_id", 0)),
            "trivial_selection": bool(record.get("trivial_selection", False)),
            "distance_metric": "boundary_distance",
            "margin_px_floor": 2.0,
            "margin_diag_fraction": 0.005,
            "margin_mode": "normalized_with_absolute_floor",
            "relation_config_version": str(record.get("relation_config_version", "")),
        },
    }


def select_deterministic(records: list[dict], count: int) -> list[dict]:
    ordered = sorted(records, key=lambda record: stable_hash(str(record["sample_id"])))
    return ordered[:count]


def select_diverse(records: list[dict], count: int) -> list[dict]:
    """Deterministic hash order, but prefer tiles not yet used (maximises unique tiles)."""

    ordered = sorted(records, key=lambda record: stable_hash(str(record["sample_id"])))
    chosen: list[dict] = []
    used_tiles: set[str] = set()
    for record in ordered:
        if len(chosen) >= count:
            break
        if tile_of(record) in used_tiles:
            continue
        chosen.append(record)
        used_tiles.add(tile_of(record))
    for record in ordered:
        if len(chosen) >= count:
            break
        if record not in chosen:
            chosen.append(record)
    return chosen


def write_pack(name: str, records: list[dict]) -> dict:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    path = PACK_ROOT / f"{name}.json"
    payload = {
        "_doc": (
            "Task 6Y frozen nearest-only pack (oracle reference). Program ids are restricted to "
            "largest_to_nearest and smallest_to_nearest; boundary_distance semantics are frozen in "
            "configs/spatial_relations_v1.yaml and were not regenerated."
        ),
        "task": "6Y", "pack": name, "seed": SEED,
        "relation": "nearest", "distance_metric": "boundary_distance",
        "reference_source": "oracle_native_gt",
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = Counter(record["program_id"] for record in records)
    return {
        "name": name, "path": str(path), "records": len(records),
        "by_program": {program: counts.get(program, 0) for program in NEAREST_PROGRAMS},
        "unique_tiles": len({tile_of(record) for record in records}),
        "unique_sample_ids": len({record["sample_id"] for record in records}),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    train = read_split("train")
    val = read_split("val")
    train_by_program = defaultdict(list)
    val_by_program = defaultdict(list)
    for record in train:
        train_by_program[record["query_type"]].append(record)
    for record in val:
        val_by_program[record["query_type"]].append(record)
    availability = {
        "train": {program: len(train_by_program[program]) for program in NEAREST_PROGRAMS},
        "val": {program: len(val_by_program[program]) for program in NEAREST_PROGRAMS},
    }
    print(f"[6y.pack] availability {availability}", flush=True)
    if any(availability["train"][program] < count for program, count in
           (("largest_to_nearest", 650), ("smallest_to_nearest", 350))):
        write_json(OUT, {"_doc": "Task 6Y section 8.", "task": "6Y",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "insufficient nearest train records", "availability": availability})
        return 2

    packs = {}
    overfit_records = []
    selected_ids: set[str] = set()
    for program, count in (("largest_to_nearest", 10), ("smallest_to_nearest", 10)):
        picked = select_diverse([record for record in train_by_program[program]
                                 if record["sample_id"] not in selected_ids], count)
        for record in picked:
            selected_ids.add(record["sample_id"])
        overfit_records.extend(picked)
    packs["y_overfit20"] = write_pack("y_overfit20", [to_sample(record) for record in overfit_records])

    mini_train = []
    for program, count in (("largest_to_nearest", 650), ("smallest_to_nearest", 350)):
        mini_train.extend(select_deterministic(train_by_program[program], count))
    packs["y_mini_train_1000"] = write_pack("y_mini_train_1000",
                                           [to_sample(record) for record in mini_train])

    mini_val = []
    for program, count in (("largest_to_nearest", 120), ("smallest_to_nearest", 120)):
        mini_val.extend(select_deterministic(val_by_program[program], count))
    packs["y_mini_val_240"] = write_pack("y_mini_val_240", [to_sample(record) for record in mini_val])

    by_tile = defaultdict(dict)
    for record in val:
        by_tile[tile_of(record)][record["query_type"]] = record
    pair_candidates = []
    for tile_id, table in by_tile.items():
        if len(table) < 2:
            continue
        largest, smallest = table.get("largest_to_nearest"), table.get("smallest_to_nearest")
        if largest is None or smallest is None:
            continue
        largest_ref = largest["native_vector"]["references"][0]["source_feature_id"]
        smallest_ref = smallest["native_vector"]["references"][0]["source_feature_id"]
        largest_target = largest["native_vector"]["target"]["source_feature_id"]
        smallest_target = smallest["native_vector"]["target"]["source_feature_id"]
        if largest_ref == smallest_ref or largest_target == smallest_target:
            continue
        pair_candidates.append({"tile_id": tile_id, "largest": to_sample(largest),
                                "smallest": to_sample(smallest),
                                "pair_id": f"{tile_id}:{largest['sample_id']}:{smallest['sample_id']}"})
    pair_candidates.sort(key=lambda entry: stable_hash(entry["pair_id"]))
    if len(pair_candidates) >= PAIRED_TARGET:
        pairs = pair_candidates[:PAIRED_TARGET]
    elif len(pair_candidates) >= PAIRED_MINIMUM:
        pairs = pair_candidates
    else:
        write_json(OUT, {
            "_doc": "Task 6Y section 12.", "task": "6Y",
            "verdict": "NEAREST_PAIRED_SET_INSUFFICIENT",
            "n_pair": len(pair_candidates), "required_minimum": PAIRED_MINIMUM,
            "availability": availability, "packs": packs,
        })
        print(f"[6y.pack] STOP NEAREST_PAIRED_SET_INSUFFICIENT ({len(pair_candidates)} pairs)", flush=True)
        return 3

    pair_records = []
    for entry in pairs:
        pair_records.append({"pair_id": entry["pair_id"], "tile_id": entry["tile_id"],
                             "largest": entry["largest"], "smallest": entry["smallest"]})
    packs["y_paired_val"] = write_pack("y_paired_val",
                                       [record for entry in pair_records
                                        for record in (entry["largest"], entry["smallest"])])

    programs_in_packs = set()
    for name in packs:
        payload = json.loads(Path(packs[name]["path"]).read_text(encoding="utf-8"))
        programs_in_packs.update(record["program_id"] for record in payload["records"])

    payload = {
        "_doc": (
            "Task 6Y sections 8-12. Frozen nearest-only packs built deterministically (seed 20260930) "
            "from the BuildSpatialReason v0.2 train/val splits, restricted to largest_to_nearest and "
            "smallest_to_nearest with the frozen boundary_distance semantics. No directional or L3 record "
            "and no test record is included."
        ),
        "task": "6Y", "stage": "E-freeze-packs", "seed": SEED,
        "source": {"train_jsonl": str(DATA / "train.jsonl"), "val_jsonl": str(DATA / "val.jsonl"),
                   "train_sha256": sha256_file(DATA / "train.jsonl"),
                   "val_sha256": sha256_file(DATA / "val.jsonl"),
                   "dataset_version": "v0.2"},
        "availability": availability,
        "programs": list(NEAREST_PROGRAMS),
        "pack_root": str(PACK_ROOT), "gitignored": True,
        "packs": packs,
        "paired": {"n_pair": len(pair_records), "target": PAIRED_TARGET,
                   "minimum": PAIRED_MINIMUM, "rule": "first 20 by stable hash if >= 20, else all if "
                                                      ">= 12, else STOP",
                   "pair_requirements": ["same tile has both nearest programs",
                                         "different reference source feature ids",
                                         "different target source feature ids",
                                         "both canonical valid records"]},
        "integrity": {
            "programs_present": sorted(programs_in_packs),
            "only_nearest_programs": set(programs_in_packs) <= set(NEAREST_PROGRAMS),
            "directional_records": 0, "l3_records": 0, "test_records": 0,
            "unique_ids_per_pack": {name: packs[name]["unique_sample_ids"] for name in packs},
        },
        "overfit20_tiles": packs["y_overfit20"]["unique_tiles"],
        "nearest_semantics": {
            "distance_metric": "boundary_distance", "margin_px_floor": 2.0,
            "margin_diag_fraction": 0.005, "margin_mode": "normalized_with_absolute_floor",
            "config": str(REPO_ROOT / "configs" / "spatial_relations_v1.yaml"),
            "labels_regenerated": False,
        },
        "verdict": "PACKS_FROZEN",
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    for name in ("y_overfit20", "y_mini_train_1000", "y_mini_val_240", "y_paired_val"):
        entry = packs[name]
        print(f"[6y.pack] {name}: {entry['records']} records {entry['by_program']} "
              f"tiles {entry['unique_tiles']}", flush=True)
    print(f"[6y.pack] paired N_pair {len(pair_records)} -> PACKS_FROZEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
