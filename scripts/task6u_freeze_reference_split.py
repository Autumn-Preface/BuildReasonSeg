"""Task 6U Part C — freeze the train-only reference calibration split.

Starts from the frozen Task 6P `RefTrainUnique` (825 unique references) and creates two disjoint groups
by unique reference key `(split, tile_id, reference_source_feature_id, reference_family)`:

* **U-Calib200** — exactly 200 references (100 largest + 100 smallest) selected deterministically by
  `sha256(f"{seed}:{key}")`; if a family has fewer than 100 it is filled from the other family.
* **U-RankerTrain** — every remaining RefTrainUnique reference.

No RefValUnique record may enter calibration or ranker training; the test split is never touched.

    python scripts/task6u_freeze_reference_split.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
OUT = EVAL / "task6u_reference_train_split.json"
SEED = 20260930
CALIB_PER_FAMILY = 100
FAMILIES = ("largest", "smallest")


def unique_key(record: dict) -> tuple:
    return (str(record["split"]), str(record["tile_id"]),
            int(record["reference_source_feature_id"]), str(record["reference_family"]))


def key_string(key: tuple) -> str:
    return "|".join(str(part) for part in key)


def key_hash(key: tuple) -> str:
    return hashlib.sha256(f"{SEED}:{key_string(key)}".encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    started = time.time()

    train_records = json.loads((PACK_ROOT / "ref_train_unique.json").read_text(
        encoding="utf-8"))["records"]
    val_records = json.loads((PACK_ROOT / "ref_val_unique.json").read_text(
        encoding="utf-8"))["records"]

    by_family: dict[str, list[dict]] = {family: [] for family in FAMILIES}
    seen: dict[tuple, dict] = {}
    duplicates = 0
    for record in train_records:
        key = unique_key(record)
        if key in seen:
            duplicates += 1
            continue
        seen[key] = record
        by_family[str(record["reference_family"])].append(record)

    calib: list[dict] = []
    for family in FAMILIES:
        ordered = sorted(by_family[family], key=lambda record: key_hash(unique_key(record)))
        calib.extend(ordered[:CALIB_PER_FAMILY])
    deficit = CALIB_PER_FAMILY * len(FAMILIES) - len(calib)
    if deficit > 0:
        # a family had fewer than 100 references: fill deterministically from the other family
        used = {unique_key(record) for record in calib}
        pool = sorted((record for record in train_records if unique_key(record) not in used),
                      key=lambda record: key_hash(unique_key(record)))
        calib.extend(pool[:deficit])

    calib_keys = {unique_key(record) for record in calib}
    ranker_train = [record for record in train_records if unique_key(record) not in calib_keys]
    val_keys = {unique_key(record) for record in val_records}

    overlap_calib_ranker = calib_keys & {unique_key(record) for record in ranker_train}
    overlap_with_val = (calib_keys | {unique_key(record) for record in ranker_train}) & val_keys
    tiles_calib = {str(record["tile_id"]) for record in calib}
    tiles_ranker = {str(record["tile_id"]) for record in ranker_train}
    tiles_val = {str(record["tile_id"]) for record in val_records}

    payload = {
        "_doc": (
            "Task 6U section 6. Train-side calibration split built from the frozen Task 6P "
            "RefTrainUnique by unique reference key (split, tile_id, reference_source_feature_id, "
            "reference_family). U-Calib200 (100 largest + 100 smallest, deterministic by "
            "sha256(seed:key)) is used only to select the frozen proposal configuration; U-RankerTrain "
            "is used only to train ProposalSetRanker v0.1. RefValUnique and the test split never enter "
            "either."
        ),
        "task": "6U", "stage": "C-train-split", "seed": SEED,
        "unique_key": "(split, tile_id, reference_source_feature_id, reference_family)",
        "selection": "sorted by sha256(f\"{seed}:{key}\"), first 100 per family",
        "source": {"pack": "artifacts/task6p/reference_packs/ref_train_unique.json",
                   "records": len(train_records), "unique_keys": len(seen),
                   "duplicate_records_ignored": duplicates},
        "u_calib200": {
            "count": len(calib),
            "by_family": dict(Counter(record["reference_family"] for record in calib)),
            "unique_tiles": len(tiles_calib),
            "keys_sha256": hashlib.sha256(
                "".join(sorted(key_string(unique_key(record)) for record in calib)).encode("utf-8")
            ).hexdigest(),
            "records": calib,
        },
        "u_rankertrain": {
            "count": len(ranker_train),
            "by_family": dict(Counter(record["reference_family"] for record in ranker_train)),
            "unique_tiles": len(tiles_ranker),
            "keys_sha256": hashlib.sha256(
                "".join(sorted(key_string(unique_key(record)) for record in ranker_train))
                .encode("utf-8")).hexdigest(),
        },
        "refval_unique": {"count": len(val_records), "by_family":
                          dict(Counter(record["reference_family"] for record in val_records)),
                          "unique_tiles": len(tiles_val)},
        "requirements": {
            "zero_key_overlap_calib_rankertrain": len(overlap_calib_ranker) == 0,
            "zero_key_overlap_with_refval_unique": len(overlap_with_val) == 0,
            "no_test_split": True,
            "calib_is_train_only": True,
            "rankertrain_is_train_only": True,
            "tile_overlap_calib_ranker": len(tiles_calib & tiles_ranker),
            "tile_overlap_with_refval": len((tiles_calib | tiles_ranker) & tiles_val),
        },
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    payload["verdict"] = "SPLIT_FROZEN" if (
        payload["requirements"]["zero_key_overlap_calib_rankertrain"]
        and payload["requirements"]["zero_key_overlap_with_refval_unique"]
        and payload["u_calib200"]["count"] == 200) else "INVALID_EXPERIMENT"
    write_json(OUT, payload)
    print(f"[6u.split] U-Calib200 {payload['u_calib200']['count']} "
          f"{payload['u_calib200']['by_family']} | U-RankerTrain "
          f"{payload['u_rankertrain']['count']} {payload['u_rankertrain']['by_family']} | "
          f"RefValUnique {len(val_records)} | verdict {payload['verdict']}", flush=True)
    return 0 if payload["verdict"] == "SPLIT_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
