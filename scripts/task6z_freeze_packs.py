"""Task 6Z Part E — freeze the L3 direction × nearest packs from BuildSpatialReason v0.2.

Deterministic (seed 20260930) stable-hash selection from the v0.2 train/val splits for exactly the four
canonical L3 programs `largest_to_{left_of,right_of,above,below}_to_nearest`:

* `Z-Overfit20`      — train, 5 above + 5 below + 5 left + 5 right, unique ids, >= 15 unique tiles when
                       possible
* `Z-MiniTrain1200`  — train, exactly 300 per direction
* `Z-MiniVal240`     — val, exactly 60 per direction
* `Z-PairedVal20`    — val same-reference directional counterfactual pairs: same tile, same largest
                       reference source feature id, two different L3 directions, two different target ids,
                       both canonical valid; first 20 by stable hash if >= 20, else all if >= 12, else STOP
                       `L3_PAIRED_SET_INSUFFICIENT`

No L1/L2 record and no test record may enter a pack; no `smallest_to_*_to_nearest` program exists in the
frozen v0.2 vocabulary and none is introduced. Packs go to the gitignored `artifacts/task6z/packs/` and
`evaluation/task6z_pack_manifest.json` records ids and SHA256.

    python scripts/task6z_freeze_packs.py
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
from buildreasonseg_mvp.task6z_field_composition import L3_PROGRAMS, PROGRAM_TO_RELATION  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
OUT = EVAL / "task6z_pack_manifest.json"
SEED = 20260930
DIRECTIONS = ("above", "below", "left", "right")
PAIRED_TARGET = 20
PAIRED_MINIMUM = 12


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def direction_of(program_id: str) -> str:
    return PROGRAM_TO_RELATION[program_id].replace("_of", "")


def tile_of(record: dict) -> str:
    if record.get("tile_id"):
        return str(record["tile_id"])
    native = record.get("native_vector") or {}
    return str(native.get("tile_id") or record.get("image_id"))


def image_path_of(record: dict) -> str:
    from buildreasonseg_mvp.task6n_relation_decoder import WHU_SOURCE_ROOT

    relative = str(record["image_path"]).replace("/", "\\")
    return str(Path(WHU_SOURCE_ROOT) / relative)


def read_split(name: str) -> list[dict]:
    records = []
    with (DATA / f"{name}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("query_type") in L3_PROGRAMS:
                records.append(record)
    return records


def to_sample(record: dict) -> dict:
    native = record["native_vector"]
    reference = native["references"][0]
    target = native["target"]
    program = str(record["query_type"])
    return {
        "sample_id": str(record["sample_id"]),
        "tile_id": tile_of(record),
        "program_id": program,
        "relation": PROGRAM_TO_RELATION[program],
        "direction": direction_of(program),
        "image_path": image_path_of(record),
        "reference_source_feature_id": int(reference["source_feature_id"]),
        "target_source_feature_id": int(target["source_feature_id"]),
        "target_instance_id": int(target["tile_instance_id"]),
        "reference_instance_id": int(reference["tile_instance_id"]),
        "level": int(record.get("level", 3)),
        "split": str(record.get("split", "")),
        "pair_key": tile_of(record),
        "metadata": {
            "reference_source": "oracle_native_gt",
            "component_map_path": str(record["component_map_path"]),
            "candidate_component_ids": list(record.get("candidate_component_ids", [])),
            "target_component_id": int(record.get("target_component_id", 0)),
            "trivial_selection": bool(record.get("trivial_selection", False)),
            "distance_metric": "boundary_distance",
            "directional_alpha": 1.2,
            "directional_tau": 0.04,
            "nearest_margin_px_floor": 2.0,
            "nearest_margin_diag_fraction": 0.005,
            "canonical_operation_order": ["argmax_area", "filter_relation",
                                          "argmin_boundary_distance"],
            "relation_config_version": str(record.get("relation_config_version", "")),
        },
    }


def select_deterministic(records: list[dict], count: int) -> list[dict]:
    return sorted(records, key=lambda record: stable_hash(str(record["sample_id"])))[:count]


def select_diverse(records: list[dict], count: int) -> list[dict]:
    ordered = sorted(records, key=lambda record: stable_hash(str(record["sample_id"])))
    chosen: list[dict] = []
    used: set[str] = set()
    for record in ordered:
        if len(chosen) >= count:
            break
        if tile_of(record) in used:
            continue
        chosen.append(record)
        used.add(tile_of(record))
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
            "Task 6Z frozen L3 direction x nearest pack (oracle largest reference). Program ids are "
            "restricted to the four canonical largest_to_<direction>_to_nearest programs; the canonical "
            "labels and the frozen v0.2 directional/nearest semantics were not regenerated."
        ),
        "task": "6Z", "pack": name, "seed": SEED, "relation_family": "L3 direction x nearest",
        "reference_source": "oracle_native_gt", "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = Counter(record["direction"] for record in records)
    return {
        "name": name, "path": str(path), "records": len(records),
        "by_direction": {direction: counts.get(direction, 0) for direction in DIRECTIONS},
        "unique_tiles": len({tile_of(record) for record in records}),
        "unique_sample_ids": len({record["sample_id"] for record in records}),
        "sha256": sha256_file(path), "bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    train = read_split("train")
    val = read_split("val")
    train_by_direction: dict[str, list[dict]] = defaultdict(list)
    val_by_direction: dict[str, list[dict]] = defaultdict(list)
    for record in train:
        train_by_direction[direction_of(str(record["query_type"]))].append(record)
    for record in val:
        val_by_direction[direction_of(str(record["query_type"]))].append(record)
    availability = {
        "train": {direction: len(train_by_direction[direction]) for direction in DIRECTIONS},
        "val": {direction: len(val_by_direction[direction]) for direction in DIRECTIONS},
    }
    print(f"[6z.pack] availability {availability}", flush=True)
    if any(availability["train"][direction] < 300 for direction in DIRECTIONS):
        write_json(OUT, {"_doc": "Task 6Z section 11.", "task": "6Z",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "insufficient L3 train records", "availability": availability})
        return 2

    packs = {}
    overfit_records = []
    chosen_ids: set[str] = set()
    for direction in DIRECTIONS:
        picked = select_diverse([record for record in train_by_direction[direction]
                                 if record["sample_id"] not in chosen_ids], 5)
        for record in picked:
            chosen_ids.add(record["sample_id"])
        overfit_records.extend(picked)
    packs["z_overfit20"] = write_pack("z_overfit20", [to_sample(record) for record in overfit_records])

    mini_train = []
    for direction in DIRECTIONS:
        mini_train.extend(select_deterministic(train_by_direction[direction], 300))
    packs["z_mini_train_1200"] = write_pack("z_mini_train_1200",
                                            [to_sample(record) for record in mini_train])

    mini_val = []
    for direction in DIRECTIONS:
        mini_val.extend(select_deterministic(val_by_direction[direction], 60))
    packs["z_mini_val_240"] = write_pack("z_mini_val_240", [to_sample(record) for record in mini_val])

    by_tile: dict[str, list[dict]] = defaultdict(list)
    for record in val:
        by_tile[tile_of(record)].append(record)
    pair_candidates = []
    for tile_id, records in by_tile.items():
        if len(records) < 2:
            continue
        for first_index in range(len(records)):
            for second_index in range(first_index + 1, len(records)):
                first, second = records[first_index], records[second_index]
                if first["query_type"] == second["query_type"]:
                    continue
                first_reference = first["native_vector"]["references"][0]["source_feature_id"]
                second_reference = second["native_vector"]["references"][0]["source_feature_id"]
                if first_reference != second_reference:
                    continue
                first_target = first["native_vector"]["target"]["source_feature_id"]
                second_target = second["native_vector"]["target"]["source_feature_id"]
                if first_target == second_target:
                    continue
                pair_candidates.append({
                    "tile_id": tile_id,
                    "pair_id": f"{first['sample_id']}|{second['sample_id']}",
                    "a": to_sample(first), "b": to_sample(second),
                })
    # each record may appear in at most one pair, chosen by stable hash order
    pair_candidates.sort(key=lambda entry: stable_hash(entry["pair_id"]))
    used: set[str] = set()
    unique_pairs = []
    for entry in pair_candidates:
        if entry["a"]["sample_id"] in used or entry["b"]["sample_id"] in used:
            continue
        unique_pairs.append(entry)
        used.add(entry["a"]["sample_id"])
        used.add(entry["b"]["sample_id"])
    if len(unique_pairs) >= PAIRED_TARGET:
        pairs = unique_pairs[:PAIRED_TARGET]
    elif len(unique_pairs) >= PAIRED_MINIMUM:
        pairs = unique_pairs
    else:
        write_json(OUT, {"_doc": "Task 6Z section 13.", "task": "6Z",
                         "verdict": "L3_PAIRED_SET_INSUFFICIENT", "n_pair": len(unique_pairs),
                         "required_minimum": PAIRED_MINIMUM, "availability": availability,
                         "packs": packs})
        print(f"[6z.pack] STOP L3_PAIRED_SET_INSUFFICIENT ({len(unique_pairs)} pairs)", flush=True)
        return 3
    packs["z_paired_val20"] = write_pack("z_paired_val20",
                                         [sample for entry in pairs
                                          for sample in (entry["a"], entry["b"])])

    programs_present = set()
    for name in packs:
        payload = json.loads(Path(packs[name]["path"]).read_text(encoding="utf-8"))
        programs_present.update(record["program_id"] for record in payload["records"])

    payload = {
        "_doc": (
            "Task 6Z sections 9-13. Frozen L3 direction x nearest packs built deterministically (seed "
            "20260930) from the v0.2 train/val splits, restricted to the four canonical "
            "largest_to_<direction>_to_nearest programs with the frozen directional (alpha 1.2, tau 0.04) "
            "and nearest (boundary_distance, margin 2.0 px / 0.005 diag) semantics. No L1/L2, no test, and "
            "no smallest-L3 program (it does not exist in the frozen vocabulary)."
        ),
        "task": "6Z", "stage": "E-freeze-packs", "seed": SEED,
        "source": {"train_jsonl": str(DATA / "train.jsonl"), "val_jsonl": str(DATA / "val.jsonl"),
                   "train_sha256": sha256_file(DATA / "train.jsonl"),
                   "val_sha256": sha256_file(DATA / "val.jsonl"), "dataset_version": "v0.2"},
        "availability": availability, "programs": list(L3_PROGRAMS),
        "directions": list(DIRECTIONS),
        "pack_root": str(PACK_ROOT), "gitignored": True, "packs": packs,
        "paired": {"n_pair": len(pairs), "target": PAIRED_TARGET, "minimum": PAIRED_MINIMUM,
                   "requirements": ["same tile", "same largest reference source feature id",
                                    "two different L3 direction programs",
                                    "two different target source feature ids",
                                    "both canonical valid records"],
                   "one_pair_per_record": True},
        "integrity": {
            "programs_present": sorted(programs_present),
            "only_l3_programs": set(programs_present) <= set(L3_PROGRAMS),
            "l1_records": 0, "l2_records": 0, "test_records": 0,
            "smallest_l3_programs": 0,
            "unique_ids_per_pack": {name: packs[name]["unique_sample_ids"] for name in packs},
        },
        "canonical_semantics": {
            "operation_order": ["argmax_area", "filter_relation", "argmin_boundary_distance"],
            "directional_alpha": 1.2, "directional_tau": 0.04,
            "distance_metric": "boundary_distance",
            "nearest_margin_px_floor": 2.0, "nearest_margin_diag_fraction": 0.005,
            "labels_regenerated": False,
        },
        "verdict": "PACKS_FROZEN", "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    for name in ("z_overfit20", "z_mini_train_1200", "z_mini_val_240", "z_paired_val20"):
        entry = packs[name]
        print(f"[6z.pack] {name}: {entry['records']} records {entry['by_direction']} tiles "
              f"{entry['unique_tiles']}", flush=True)
    print(f"[6z.pack] paired N_pair {len(pairs)} -> PACKS_FROZEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
