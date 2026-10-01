"""Task 7E Parts D-E — construct the untouched L3 validation remainder `E-HoldoutL3`.

Source population (section 6): the full BuildSpatialReason v0.2 **val** split restricted to the four canonical
L3 programs. Expected 936 records (above 224, below 213, left 250, right 249).

Exclusion (section 7): every Z-MiniVal240 record id and every record used as either member of Z-PairedVal20,
removed by `sample_id`. The remainder is `E-HoldoutL3`, is never subsampled, and must have >= 600 records with
>= 120 per program; otherwise STOP `L3_HOLDOUT_REMAINDER_INSUFFICIENT`.

`E-PairedHoldout` (section 8) is built only from `E-HoldoutL3`: same tile, same oracle largest reference
source-feature id, different L3 direction programs, different target source-feature ids, both members inside
the holdout; pairs are sorted by a SHA256 stable key and the first 20 are used
(STOP `L3_HOLDOUT_PAIRED_INSUFFICIENT` below 20).

Nothing is regenerated: the exclusion packs are read byte-for-byte and the record expansion mirrors the frozen
Task 6Z pack builder (`to_sample`), so holdout rows carry exactly the same schema.

    python scripts/task7e_build_holdout.py
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
from scripts.task6z_freeze_packs import L3_PROGRAMS, read_split, to_sample  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
HOLDOUT_ROOT = REPO_ROOT / "artifacts" / "task7e" / "holdout"
OUT = EVAL / "task7e_holdout_manifest.json"
EXPECTED_SOURCE = {"largest_to_above_to_nearest": 224, "largest_to_below_to_nearest": 213,
                   "largest_to_left_of_to_nearest": 250, "largest_to_right_of_to_nearest": 249}
MIN_RECORDS = 600
MIN_PER_PROGRAM = 120
PAIR_COUNT = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_id_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(sample_ids)).encode("utf-8")).hexdigest()


def pack_ids(name: str) -> list[str]:
    records = json.loads((PACK_ROOT / f"{name}.json").read_text(encoding="utf-8"))["records"]
    return [str(record["sample_id"]) for record in records]


def paired_member_ids() -> list[str]:
    return pack_ids("z_paired_val20")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    source_records = read_split("val")
    source_counts = Counter(str(record["query_type"]) for record in source_records)
    source_ok = all(source_counts.get(program, 0) == expected
                    for program, expected in EXPECTED_SOURCE.items())

    mini_ids = set(pack_ids("z_mini_val_240"))
    pair_ids = set(paired_member_ids())
    exclusion = mini_ids | pair_ids

    holdout_records = [record for record in source_records
                       if str(record["sample_id"]) not in exclusion]
    holdout_ids = [str(record["sample_id"]) for record in holdout_records]
    per_program = Counter(str(record["query_type"]) for record in holdout_records)
    per_direction = Counter(str(record["query_type"]).replace("largest_to_", "").replace("_to_nearest", "")
                            for record in holdout_records)

    overlap_mini = sorted(set(holdout_ids) & mini_ids)
    overlap_paired = sorted(set(holdout_ids) & pair_ids)
    test_leak = sorted(str(record["sample_id"]) for record in holdout_records
                       if str(record.get("split", "")) != "val")
    enough_records = len(holdout_records) >= MIN_RECORDS
    enough_per_program = all(per_program.get(program, 0) >= MIN_PER_PROGRAM for program in L3_PROGRAMS)
    remainder_ok = bool(source_ok and not overlap_mini and not overlap_paired and not test_leak
                        and enough_records and enough_per_program)

    # ---------------- E-PairedHoldout
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in holdout_records:
        native = record["native_vector"]
        key = (str(native.get("tile_id") or record.get("image_id")),
               int(native["references"][0]["source_feature_id"]))
        by_key[key].append(record)
    candidates = []
    for (tile_id, reference_id), members in by_key.items():
        for first_index in range(len(members)):
            for second_index in range(first_index + 1, len(members)):
                first, second = members[first_index], members[second_index]
                first_program, second_program = str(first["query_type"]), str(second["query_type"])
                first_target = int(first["native_vector"]["target"]["source_feature_id"])
                second_target = int(second["native_vector"]["target"]["source_feature_id"])
                if first_program == second_program or first_target == second_target:
                    continue
                pair_key = "|".join(sorted([str(first["sample_id"]), str(second["sample_id"])]))
                candidates.append({
                    "pair_id": f"{tile_id}:{stable_hash(pair_key)[:16]}",
                    "stable_key": stable_hash(pair_key), "tile_id": tile_id,
                    "reference_source_feature_id": reference_id,
                    "members": [{"sample_id": str(first["sample_id"]),
                                 "program_id": first_program,
                                 "target_source_feature_id": first_target},
                                {"sample_id": str(second["sample_id"]),
                                 "program_id": second_program,
                                 "target_source_feature_id": second_target}],
                })
    candidates.sort(key=lambda entry: entry["stable_key"])
    pairs = candidates[:PAIR_COUNT]
    paired_ok = len(pairs) >= PAIR_COUNT
    pair_member_ids = {member["sample_id"] for pair in pairs for member in pair["members"]}
    pair_overlap_old = sorted(pair_member_ids & pair_ids)

    HOLDOUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows_path = HOLDOUT_ROOT / "holdout_l3_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for record in holdout_records:
            handle.write(json.dumps(to_sample(record), ensure_ascii=False) + "\n")
    pairs_path = HOLDOUT_ROOT / "holdout_pairs.json"
    pairs_path.write_text(json.dumps({"pairs": pairs}, ensure_ascii=False, indent=1), encoding="utf-8")

    payload = {
        "_doc": ("Task 7E sections 6-8. Untouched L3 validation remainder E-HoldoutL3: the full "
                 "BuildSpatialReason v0.2 val population of the four canonical L3 programs minus every "
                 "Z-MiniVal240 record and every Z-PairedVal20 member (by sample_id). No subsampling, no "
                 "regeneration, no test record, no training."),
        "task": "7E", "stage": "D-holdout-remainder",
        "source": {"split": "val", "dataset": "BuildSpatialReason v0.2",
                   "programs": list(L3_PROGRAMS), "records": len(source_records),
                   "per_program": {program: source_counts.get(program, 0) for program in L3_PROGRAMS},
                   "expected_per_program": EXPECTED_SOURCE, "matches_expected": source_ok},
        "exclusion": {"z_mini_val_240": {"records": len(mini_ids), "path": "z_mini_val_240.json"},
                      "z_paired_val20_members": {"records": len(pair_ids),
                                                 "path": "z_paired_val20.json"},
                      "union": len(exclusion),
                      "excluded_present": len(set(source_records[0]["sample_id"] for _ in [0])
                                             & exclusion) if source_records else 0},
        "holdout": {"records": len(holdout_records),
                    "per_program": {program: per_program.get(program, 0) for program in L3_PROGRAMS},
                    "per_direction": dict(per_direction),
                    "record_id_hash": record_id_hash(holdout_ids),
                    "rows_path": str(rows_path), "gitignored": True,
                    "subsampled": False},
        "checks": {
            "source_total_matches": source_ok,
            "overlap_with_z_mini_val_240": len(overlap_mini),
            "overlap_with_z_paired_val20": len(overlap_paired),
            "test_records": len(test_leak),
            "records_gte_600": enough_records, "min_records": MIN_RECORDS,
            "every_program_gte_120": enough_per_program, "min_per_program": MIN_PER_PROGRAM,
            "remainder_ok": remainder_ok,
        },
        "paired": {
            "name": "E-PairedHoldout", "pairs": len(pairs), "required": PAIR_COUNT,
            "candidate_pairs": len(candidates), "passed": paired_ok,
            "pair_ids": [pair["pair_id"] for pair in pairs],
            "pair_id_hash": record_id_hash([pair["pair_id"] for pair in pairs]),
            "member_id_hash": record_id_hash(sorted(pair_member_ids)),
            "overlap_with_task6z_paired_members": len(pair_overlap_old),
            "pairs_path": str(pairs_path), "gitignored": True,
            "requirements": {"same_tile": True, "same_oracle_reference": True,
                             "different_direction_programs": True, "different_targets": True,
                             "both_members_in_holdout": True},
        },
        "packs_read_only": {"z_mini_val_240_sha256": sha256_file(PACK_ROOT / "z_mini_val_240.json"),
                            "z_paired_val20_sha256": sha256_file(PACK_ROOT / "z_paired_val20.json")},
        "training_performed": False, "test_split_used": False,
        "verdict": ("L3_HOLDOUT_READY" if remainder_ok and paired_ok and not pair_overlap_old
                    else ("L3_HOLDOUT_PAIRED_INSUFFICIENT" if remainder_ok
                          else "L3_HOLDOUT_REMAINDER_INSUFFICIENT")),
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7e.holdout] source {len(source_records)} (expected {sum(EXPECTED_SOURCE.values())}) | "
          f"excluded {len(exclusion)} | holdout {len(holdout_records)} "
          f"{dict(per_program)} | overlapping mini {len(overlap_mini)} paired {len(overlap_paired)} | "
          f"pairs {len(pairs)}/{PAIR_COUNT} (candidates {len(candidates)}, old overlap "
          f"{len(pair_overlap_old)}) -> {payload['verdict']}", flush=True)
    if payload["verdict"] == "L3_HOLDOUT_REMAINDER_INSUFFICIENT":
        return 3
    if payload["verdict"] == "L3_HOLDOUT_PAIRED_INSUFFICIENT":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
