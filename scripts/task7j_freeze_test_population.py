"""Task 7J Parts 2-5 — one-time authorization, checkpoint manifest and frozen final L3 test population.

Writes the ChatGPT authorization artifact **before** any test record is read, verifies the Task 7I test lock
state and all six formal `best.pt` checkpoints against the authoritative Task 7I training artifacts (STOP
`FORMAL_CHECKPOINT_MISMATCH` on any missing/mismatched checkpoint), then freezes:

* every valid BuildSpatialReason v0.2 **test** record of the four canonical L3 programs (no filtering by model
  output, target size, reference quality or success; no post-inference exclusion);
* `I-FinalTestPairsAll` — every unique unordered pair with the same tile, the same oracle largest reference, a
  different direction and a different target, deduplicated by the exact `min||max` sample-id key, sorted
  lexicographically and never subsampled (reporting only).

The historical Task 6M test access is disclosed; the permitted wording is
`final frozen-architecture test evaluation`.

    python scripts/task7j_freeze_test_population.py
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
from scripts.task7i_train import CHECKPOINT_ROOT, SEEDS, sha256_file  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task7j" / "packs"
OUT_AUTH = EVAL / "task7j_test_authorization.json"
OUT_CHECKPOINTS = EVAL / "task7j_checkpoint_manifest.json"
OUT_POPULATION = EVAL / "task7j_test_population_manifest.json"
OUT_PAIRS = EVAL / "task7j_test_pair_manifest.json"
MODELS = ("Z-B3", "D-B1")
TRAINING = {"Z-B3": EVAL / "task7i_training_zb3.json", "D-B1": EVAL / "task7i_training_db1.json"}
LOCK_7I = EVAL / "task7i_test_lock_status.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    # ---------------- Part 2: one-time authorization, written before any test read
    lock = json.loads(LOCK_7I.read_text(encoding="utf-8"))
    authorization = {
        "_doc": ("Task 7J section 2. One-time ChatGPT authorization for the single final frozen-architecture "
                 "test evaluation. Written before any test record is read. No training, no checkpoint "
                 "selection and no architecture change are authorized."),
        "task": "7J", "stage": "2-one-time-authorization",
        "authorized_by": "ChatGPT audit after Task 7I",
        "scope": "Task 7J final frozen-architecture evaluation only",
        "training_allowed": False, "checkpoint_selection_allowed": False,
        "architecture_change_allowed": False, "test_access_authorized": True,
        "historical_test_access_disclosed": True,
        "permitted_test_description": "final frozen-architecture test evaluation",
        "forbidden_description": "untouched test",
        "previous_test_lock": {"path": str(LOCK_7I), "status": lock["status"],
                               "test_execution_authorized": lock["test_execution_authorized"]},
        "previous_lock_was_locked": lock["status"] == "LOCKED"
        and lock["test_execution_authorized"] is False,
        "written_before_test_read": True,
    }
    write_json(OUT_AUTH, authorization)
    if not authorization["previous_lock_was_locked"]:
        print("[7j.freeze] STOP INVALID_EXPERIMENT (previous test lock was not LOCKED)", flush=True)
        return 2

    # ---------------- Part 3: checkpoint manifest
    training = {model: json.loads(path.read_text(encoding="utf-8")) for model, path in TRAINING.items()}
    checkpoints = {}
    mismatches = []
    for model in MODELS:
        for seed in SEEDS:
            expected = next(run for run in training[model]["runs"] if run["seed"] == seed)
            path = CHECKPOINT_ROOT / model / str(seed) / "best.pt"
            entry = {"model": model, "seed": seed, "path": str(path), "exists": path.is_file(),
                     "expected_sha256": expected["checkpoint_best"]["sha256"],
                     "expected_bytes": expected["checkpoint_best"]["bytes"],
                     "selected_epoch": expected["selected_epoch"],
                     "trainable_parameters": expected["trainable_parameters"],
                     "source_artifact": str(TRAINING[model])}
            if path.is_file():
                entry["sha256"] = sha256_file(path)
                entry["bytes"] = path.stat().st_size
                entry["matches"] = (entry["sha256"] == entry["expected_sha256"]
                                    and entry["bytes"] == entry["expected_bytes"])
            else:
                entry["matches"] = False
            checkpoints[f"{model}/{seed}"] = entry
            if not entry["matches"]:
                mismatches.append(f"{model}/{seed}")
    last_pt = [str(CHECKPOINT_ROOT / f"{model}/{seed}" / "last.pt")
               for f in (0,) for model in MODELS for seed in SEEDS
               if (CHECKPOINT_ROOT / model / str(seed) / "last.pt").is_file()]
    write_json(OUT_CHECKPOINTS, {
        "_doc": ("Task 7J section 3. Authoritative manifest of the six Task 7I formal `best.pt` checkpoints, "
                 "verified against evaluation/task7i_training_*.json. `last.pt` is never used and no "
                 "retraining happened."),
        "task": "7J", "stage": "3-checkpoint-manifest",
        "checkpoints": checkpoints, "mismatches": mismatches,
        "all_verified": not mismatches, "last_pt_present_but_unused": last_pt,
        "used_checkpoints": "best.pt only", "training_performed": False,
        "verdict": "FORMAL_CHECKPOINTS_VERIFIED" if not mismatches else "FORMAL_CHECKPOINT_MISMATCH",
    })
    if mismatches:
        write_json(OUT_POPULATION, {"_doc": "Task 7J section 3.", "task": "7J",
                                    "verdict": "FORMAL_CHECKPOINT_MISMATCH",
                                    "mismatches": mismatches})
        print(f"[7j.freeze] STOP FORMAL_CHECKPOINT_MISMATCH ({mismatches})", flush=True)
        return 3

    # ---------------- Part 4: final test population (first authorized test read)
    test_records = read_split("test")
    per_program = Counter(str(record["query_type"]) for record in test_records)
    tiles = sorted({str((record.get("native_vector") or {}).get("tile_id") or record.get("image_id"))
                    for record in test_records})
    sample_ids = sorted(str(record["sample_id"]) for record in test_records)
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    rows_path = PACK_ROOT / "final_l3_test.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for record in test_records:
            handle.write(json.dumps(to_sample(record), ensure_ascii=False) + "\n")
    population_ok = len(test_records) > 0 and all(per_program.get(program, 0) > 0
                                                  for program in L3_PROGRAMS)

    # ---------------- Part 5: pairs
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in test_records:
        native = record["native_vector"]
        key = (str(native.get("tile_id") or record.get("image_id")),
               int(native["references"][0]["source_feature_id"]))
        groups[key].append(record)
    pair_keys: dict[str, dict] = {}
    candidates = 0
    direction_pairs = Counter()
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
                if key not in pair_keys:
                    first_direction = str(first["query_type"]).replace("largest_to_", "") \
                        .replace("_to_nearest", "")
                    second_direction = str(second["query_type"]).replace("largest_to_", "") \
                        .replace("_to_nearest", "")
                    pair_keys[key] = {
                        "pair_key": key, "tile_id": tile_id,
                        "reference_source_feature_id": reference_id,
                        "direction_pair": "|".join(sorted([first_direction, second_direction])),
                        "members": [{"sample_id": first_id,
                                     "program_id": str(first["query_type"]),
                                     "direction": first_direction,
                                     "target_source_feature_id": first_target},
                                    {"sample_id": second_id,
                                     "program_id": str(second["query_type"]),
                                     "direction": second_direction,
                                     "target_source_feature_id": second_target}]}
                    direction_pairs[pair_keys[key]["direction_pair"]] += 1
    pairs = [pair_keys[key] for key in sorted(pair_keys)]

    write_json(OUT_POPULATION, {
        "_doc": ("Task 7J section 4. Frozen final L3 test population: every valid BuildSpatialReason v0.2 test "
                 "record of the four canonical L3 programs. No filtering by model output, target size, "
                 "reference quality or success, and no post-inference exclusion. This is the single authorized "
                 "'final frozen-architecture test evaluation'; the historical Task 6M test access is disclosed."),
        "task": "7J", "stage": "4-final-test-population",
        "authorization": {"path": str(OUT_AUTH), "written_before_test_read": True,
                          "test_access_authorized": True},
        "dataset": {"name": "BuildSpatialReason", "version": "v0.2", "split": "test",
                    "native_vector": "WHU-EA-NativeVector v1.0", "split_view": "scene_disjoint_v1"},
        "programs": list(L3_PROGRAMS),
        "records": len(test_records),
        "per_program": {program: per_program.get(program, 0) for program in L3_PROGRAMS},
        "unique_tiles": len(tiles), "tiles": tiles,
        "sample_id_sha256": sha256_text("\n".join(sample_ids)),
        "rows_path": str(rows_path), "gitignored": True,
        "historical_test_access_disclosure": {
            "historically_accessed": True,
            "when": "Task 6M for an earlier proposal-baseline J4-v2 audit",
            "permitted_description": "final frozen-architecture test evaluation",
            "forbidden_description": "untouched test"},
        "filters_applied": [], "post_inference_exclusion": False,
        "frozen_before_inference": True,
        "verdict": "FINAL_TEST_POPULATION_FROZEN" if population_ok else "FINAL_TEST_POPULATION_INVALID",
    })
    write_json(OUT_PAIRS, {
        "_doc": ("Task 7J section 5. I-FinalTestPairsAll: every unique unordered pair of the frozen final L3 "
                 "test population with the same tile, the same oracle largest reference, a different direction "
                 "and a different target, deduplicated by the exact min||max key, sorted lexicographically and "
                 "never subsampled. Reporting only."),
        "task": "7J", "stage": "5-final-test-pairs",
        "population": {"records": len(test_records), "sample_id_sha256": sha256_text("\n".join(sample_ids))},
        "candidates_before_dedup": candidates, "pairs": len(pairs),
        "direction_pair_counts": dict(direction_pairs),
        "unique_tiles": len({pair["tile_id"] for pair in pairs}),
        "pair_id_sha256": sha256_text("\n".join(pair["pair_key"] for pair in pairs)),
        "requirements": {"same_tile": True, "same_reference": True, "different_direction": True,
                         "different_target": True, "subsampled": False},
        "pairs_list": pairs, "used_for_selection": False,
        "verdict": "FINAL_TEST_PAIRS_FROZEN",
    })
    print(f"[7j.freeze] checkpoints verified {len(checkpoints)} | test records {len(test_records)} "
          f"{dict(per_program)} | tiles {len(tiles)} | pairs {len(pairs)} -> "
          f"{'OK' if population_ok else 'INVALID'}", flush=True)
    return 0 if population_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
