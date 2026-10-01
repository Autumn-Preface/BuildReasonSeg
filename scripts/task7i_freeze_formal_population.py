"""Task 7I Parts B-D — freeze the formal L3 train/val populations and the full-validation counterfactual set.

Reads the Task 7H test lock (requires `LOCKED` / `test_execution_authorized = false`; otherwise STOP
`TEST_LOCK_INCONSISTENT`) and then freezes, from the tracked v0.2 splits only:

* `G-FormalTrain` — all valid v0.2 **train** records of the four L3 programs (expected 1344: 323/347/338/336);
* `G-FormalVal` — all valid v0.2 **val** records of the four L3 programs (expected 936: 250/249/224/213);
* `I-FormalValPairsAll` — every unique unordered formal-val record pair with the same tile, the same oracle
  largest reference, a different canonical direction and a different target, deduplicated by the exact
  `min||max` sample-id key, sorted lexicographically and never subsampled. The pair set is reporting-only and
  is not used for checkpoint selection or early stopping.

Sample-id SHA256 hashes, per-program counts and train/val disjointness are recorded. No test material is read.

Writes `evaluation/task7i_formal_population_manifest.json` and
`evaluation/task7i_formal_val_pair_manifest.json`; expanded rows go to the gitignored `artifacts/task7i/packs/`.

    python scripts/task7i_freeze_formal_population.py
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
PACK_ROOT = REPO_ROOT / "artifacts" / "task7i" / "packs"
OUT_POPULATION = EVAL / "task7i_formal_population_manifest.json"
OUT_PAIRS = EVAL / "task7i_formal_val_pair_manifest.json"
TEST_LOCK = EVAL / "task7h_test_lock.json"
EXPECTED_TRAIN = {"largest_to_left_of_to_nearest": 323, "largest_to_right_of_to_nearest": 347,
                  "largest_to_above_to_nearest": 338, "largest_to_below_to_nearest": 336}
EXPECTED_VAL = {"largest_to_left_of_to_nearest": 250, "largest_to_right_of_to_nearest": 249,
                "largest_to_above_to_nearest": 224, "largest_to_below_to_nearest": 213}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    lock_ok = lock["status"] == "LOCKED" and lock["test_execution_authorized"] is False
    if not lock_ok:
        write_json(OUT_POPULATION, {"_doc": "Task 7I section 4.", "task": "7I",
                                    "verdict": "TEST_LOCK_INCONSISTENT"})
        print("[7i.pop] STOP TEST_LOCK_INCONSISTENT", flush=True)
        return 2

    train_records = read_split("train")
    val_records = read_split("val")
    train_counts = Counter(str(record["query_type"]) for record in train_records)
    val_counts = Counter(str(record["query_type"]) for record in val_records)
    train_ids = sorted(str(record["sample_id"]) for record in train_records)
    val_ids = sorted(str(record["sample_id"]) for record in val_records)
    overlap = sorted(set(train_ids) & set(val_ids))

    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = PACK_ROOT / "formal_train_1344.jsonl"
    val_path = PACK_ROOT / "formal_val_936.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(to_sample(record), ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as handle:
        for record in val_records:
            handle.write(json.dumps(to_sample(record), ensure_ascii=False) + "\n")

    # ---------------- I-FormalValPairsAll (reporting only)
    def reference_id_of(record: dict) -> int:
        return int(record["native_vector"]["references"][0]["source_feature_id"])

    def tile_of(record: dict) -> str:
        return str((record.get("native_vector") or {}).get("tile_id") or record.get("image_id"))

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in val_records:
        groups[(tile_of(record), reference_id_of(record))].append(record)

    pair_keys: dict[str, dict] = {}
    candidates = 0
    for (tile_id, reference_id), members in groups.items():
        for first_index in range(len(members)):
            for second_index in range(first_index + 1, len(members)):
                first, second = members[first_index], members[second_index]
                first_id, second_id = str(first["sample_id"]), str(second["sample_id"])
                if str(first["query_type"]) == str(second["query_type"]):
                    continue
                first_target = int(first["native_vector"]["target"]["source_feature_id"])
                second_target = int(second["native_vector"]["target"]["source_feature_id"])
                if first_target == second_target:
                    continue
                candidates += 1
                key = f"{min(first_id, second_id)}||{max(first_id, second_id)}"
                pair_keys.setdefault(key, {
                    "pair_key": key, "tile_id": tile_id,
                    "reference_source_feature_id": reference_id,
                    "members": [{"sample_id": first_id, "program_id": str(first["query_type"]),
                                 "direction": str(first["query_type"])
                                 .replace("largest_to_", "").replace("_to_nearest", ""),
                                 "target_source_feature_id": first_target},
                                {"sample_id": second_id, "program_id": str(second["query_type"]),
                                 "direction": str(second["query_type"])
                                 .replace("largest_to_", "").replace("_to_nearest", ""),
                                 "target_source_feature_id": second_target}]})
    pairs = [pair_keys[key] for key in sorted(pair_keys)]

    population_ok = (dict(train_counts) == EXPECTED_TRAIN and dict(val_counts) == EXPECTED_VAL
                     and not overlap)
    write_json(OUT_POPULATION, {
        "_doc": ("Task 7I sections 5-8. Frozen formal L3 populations: all valid v0.2 train records (1344) and "
                 "all valid v0.2 val records (936) of the four canonical L3 programs. Checkpoint selection and "
                 "early stopping use the full 936 oracle-reference val records; MiniVal240 is not used. No "
                 "test material was read."),
        "task": "7I", "stage": "C-formal-population",
        "test_lock": {"status": lock["status"],
                      "test_execution_authorized": lock["test_execution_authorized"],
                      "verified_before_training": True, "test_material_read": False},
        "programs": list(L3_PROGRAMS),
        "train": {"records": len(train_records), "per_program": {program: train_counts.get(program, 0)
                                                                 for program in L3_PROGRAMS},
                  "expected_per_program": EXPECTED_TRAIN, "expected_total": 1344,
                  "sample_id_sha256": sha256_text("\n".join(train_ids)),
                  "rows_path": str(train_path), "gitignored": True,
                  "inputs": ["source image", "canonical GT largest reference mask", "canonical direction",
                             "canonical GT target mask (label only, never a feature)"]},
        "val": {"records": len(val_records), "per_program": {program: val_counts.get(program, 0)
                                                             for program in L3_PROGRAMS},
                "expected_per_program": EXPECTED_VAL, "expected_total": 936,
                "sample_id_sha256": sha256_text("\n".join(val_ids)),
                "rows_path": str(val_path), "gitignored": True,
                "selection_protocol": "all 936 records, oracle GT reference, mean target mIoU"},
        "checks": {"train_counts_match": dict(train_counts) == EXPECTED_TRAIN,
                   "val_counts_match": dict(val_counts) == EXPECTED_VAL,
                   "train_val_overlap": len(overlap), "disjoint": not overlap,
                   "test_material_read": False, "population_ok": population_ok},
        "training": "AdamW lr 3e-4, wd 1e-4, batch 8, max 25 epochs, patience 5, bf16 AMP, BCE+Dice, "
                    "seeds 20261001/20261002/20261003 for both Z-B3 and D-B1",
        "verdict": "FORMAL_POPULATION_FROZEN" if population_ok else "FORMAL_POPULATION_MISMATCH",
        "runtime_seconds": round(time.time() - started, 1),
    })
    write_json(OUT_PAIRS, {
        "_doc": ("Task 7I section 9. I-FormalValPairsAll: every unique unordered pair of the 936 formal val "
                 "records with the same tile, the same oracle largest reference, a different canonical "
                 "direction and a different target. Deduplicated by the exact min||max sample-id key, sorted "
                 "lexicographically, never subsampled. Reporting-only: it is not used for checkpoint selection "
                 "or early stopping."),
        "task": "7I", "stage": "D-formal-val-pairs",
        "population": {"val_records": len(val_records), "val_sample_id_sha256": sha256_text("\n".join(val_ids))},
        "candidates_before_dedup": candidates, "pairs": len(pairs),
        "pair_key_hash": sha256_text("\n".join(pair["pair_key"] for pair in pairs)),
        "requirements": {"same_tile": True, "same_reference": True, "different_direction": True,
                         "different_target": True, "subsampled": False, "source": "formal val only"},
        "pairs_list": pairs,
        "used_for_selection": False, "test_split_used": False,
        "verdict": "FORMAL_VAL_PAIRS_FROZEN",
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7i.pop] train {len(train_records)} {dict(train_counts)} | val {len(val_records)} "
          f"{dict(val_counts)} | overlap {len(overlap)} | pairs {len(pairs)} "
          f"(candidates {candidates}) -> {population_ok}", flush=True)
    return 0 if population_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
