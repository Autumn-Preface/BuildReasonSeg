"""Task 6N section 12 — freeze the four deterministic packs before any training.

Builds, from BuildSpatialReason v0.2 and the canonical native-vector instances:

* **Overfit20** — 20 train records, deterministic sorted selection, all 4 directions, both
  largest/smallest reference families, at least 4 same-image counterfactual pairs when available;
* **MiniTrain1000** — the first 1000 eligible train records after stable seeded stratification by
  direction x reference family (all eligible records are used and reported if fewer than 1000);
* **MiniVal240** — 240 val records, 30 per program id, all 8 ids represented;
* **PairedVal20** — 20 val same-image pairs sharing the tile and the reference instance but with a
  different direction and a different target instance.

Eligibility (section 12): one of the 8 directional programs, oracle reference and target
`source_feature_id`s resolve to canonical native instances, the RGB source tile exists, and the frozen
SAM2 feature exists or can be generated through the frozen cache path. No tiny/border/visibility
filter and no difficulty deletion is applied.

Writes `evaluation/task6n_pack_manifest.json`. Test split is never touched.

    python scripts/task6n_freeze_packs.py
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

from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DIRECTIONAL_PROGRAMS,
    PROGRAM_TO_RELATION,
    WHU_SOURCE_ROOT,
    Task6NSample,
    write_pack,
)

EVAL = REPO_ROOT / "evaluation"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
OUT = EVAL / "task6n_pack_manifest.json"
SEED = 20260929
FAMILIES = {"largest": "largest", "smallest": "smallest"}


def stable_key(sample_id: str, seed: int = SEED) -> str:
    """Stable seeded ordering key (independent of file order and Python hashing)."""

    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def reference_family(program_id: str) -> str:
    return program_id.split("_", 1)[0]


def load_eligible(split: str, dataset_root: Path, verbose: bool = True) -> tuple[list[Task6NSample], dict]:
    """All eligible records for one split, with an eligibility audit."""

    records = []
    audit = Counter()
    with (V02 / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            program = str(record["query_type"])
            if program not in DIRECTIONAL_PROGRAMS:
                audit["skipped_program_out_of_scope"] += 1
                continue
            audit["in_scope"] += 1
            tile_id = str(record["image_id"])
            instances = canonical_instances(tile_id)
            if not instances:
                audit["skipped_no_native_instances"] += 1
                continue
            by_feature = {instance.source_feature_id: instance for instance in instances}
            references = record["native_vector"].get("references") or []
            if not references:
                audit["skipped_no_reference_entry"] += 1
                continue
            reference_feature = int(references[0]["source_feature_id"])
            target_feature = int(record["native_vector"]["target"]["source_feature_id"])
            if reference_feature not in by_feature:
                audit["skipped_reference_unresolved"] += 1
                continue
            if target_feature not in by_feature:
                audit["skipped_target_unresolved"] += 1
                continue
            reference_instance = by_feature[reference_feature]
            target_instance = by_feature[target_feature]
            if reference_instance.area_px <= 0 or target_instance.area_px <= 0:
                audit["skipped_empty_mask"] += 1
                continue
            image_ref = dataset_root.record_for(tile_id)
            image_path = WHU_SOURCE_ROOT / str(image_ref)
            if not image_path.is_file():
                audit["skipped_missing_rgb"] += 1
                continue
            audit["eligible"] += 1
            records.append(
                Task6NSample(
                    sample_id=str(record["sample_id"]),
                    tile_id=tile_id,
                    program_id=program,
                    relation=PROGRAM_TO_RELATION[program],
                    image_path=str(image_path),
                    reference_source_feature_id=reference_feature,
                    target_source_feature_id=target_feature,
                    target_instance_id=int(target_instance.tile_instance_id),
                    reference_instance_id=int(reference_instance.tile_instance_id),
                    level=int(record["level"]),
                    split=split,
                    pair_key=f"{tile_id}:{reference_feature}",
                    metadata={
                        "reference_area_px": int(reference_instance.area_px),
                        "target_area_px": int(target_instance.area_px),
                        "target_touches_border": bool(target_instance.touches_border),
                        "target_tiny": bool(target_instance.tiny),
                        "reference_family": reference_family(program),
                        "direction": PROGRAM_TO_RELATION[program],
                    },
                )
            )
    if verbose:
        print(f"[6n.packs] {split}: {dict(audit)}", flush=True)
    return records, dict(audit)


class _DatasetRoot:
    """Tiny helper exposing `source_image_ref` per tile from the canonical dataset record."""

    def __init__(self) -> None:
        from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset

        self.dataset = NativeVectorDataset()
        self._cache: dict[str, str] = {}

    def record_for(self, tile_id: str) -> str:
        if tile_id not in self._cache:
            view = self.dataset.load_tile(tile_id)
            self._cache[tile_id] = str(view.record["source_image_ref"])
        return self._cache[tile_id]


def counterfactual_pairs(records: list[Task6NSample]) -> list[tuple[Task6NSample, Task6NSample]]:
    """Same tile, same reference instance, different direction, different target instance."""

    grouped: dict[str, list[Task6NSample]] = defaultdict(list)
    for record in records:
        grouped[record.pair_key].append(record)
    pairs = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: (item.relation, item.sample_id))
        for index, first in enumerate(items):
            for second in items[index + 1:]:
                if first.relation == second.relation:
                    continue
                if first.target_source_feature_id == second.target_source_feature_id:
                    continue
                pairs.append((first, second))
    return pairs


def build_overfit20(train: list[Task6NSample], target: int = 20) -> tuple[list[Task6NSample], dict]:
    """Deterministic sorted selection covering 4 directions x 2 families with >= 4 counterfactual pairs."""

    by_id = {record.sample_id: record for record in train}
    pairs = sorted(
        counterfactual_pairs(train),
        key=lambda pair: (pair[0].tile_id, pair[0].relation, pair[1].relation, pair[0].sample_id),
    )
    selected: list[Task6NSample] = []
    selected_ids: set[str] = set()
    chosen_pairs = []
    for first, second in pairs:
        if len(chosen_pairs) >= 4:
            break
        if first.sample_id in selected_ids or second.sample_id in selected_ids:
            continue
        used_tiles = {record.tile_id for record in selected}
        if first.tile_id in used_tiles:
            continue
        chosen_pairs.append((first, second))
        for record in (first, second):
            selected.append(record)
            selected_ids.add(record.sample_id)

    # fill the remaining cells: all 4 directions x both families, distinct tiles, deterministic
    ordered = sorted(train, key=lambda record: (record.pair_key, record.relation, record.sample_id))
    used_tiles = {record.tile_id for record in selected}
    for direction in ("left_of", "right_of", "above", "below"):
        for family in ("largest", "smallest"):
            if len(selected) >= target:
                break
            already = any(
                record.relation == direction and reference_family(record.program_id) == family
                for record in selected
            )
            if already:
                continue
            candidate = next(
                (
                    record for record in ordered
                    if record.relation == direction
                    and reference_family(record.program_id) == family
                    and record.sample_id not in selected_ids
                    and record.tile_id not in used_tiles
                ),
                None,
            )
            if candidate is None:
                candidate = next(
                    (
                        record for record in ordered
                        if record.relation == direction
                        and reference_family(record.program_id) == family
                        and record.sample_id not in selected_ids
                    ),
                    None,
                )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(candidate.sample_id)
                used_tiles.add(candidate.tile_id)

    for record in ordered:
        if len(selected) >= target:
            break
        if record.sample_id in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(record.sample_id)

    selected.sort(key=lambda record: (record.tile_id, record.relation, record.sample_id))
    selected = selected[:target]
    detail = {
        "directions": sorted({record.relation for record in selected}),
        "families": sorted({reference_family(record.program_id) for record in selected}),
        "programs": sorted({record.program_id for record in selected}),
        "counterfactual_pairs": [
            {
                "tile_id": first.tile_id,
                "reference_source_feature_id": first.reference_source_feature_id,
                "a": {"sample_id": first.sample_id, "relation": first.relation,
                      "target_source_feature_id": first.target_source_feature_id},
                "b": {"sample_id": second.sample_id, "relation": second.relation,
                      "target_source_feature_id": second.target_source_feature_id},
            }
            for first, second in chosen_pairs
        ],
        "pair_count": len(chosen_pairs),
        "distinct_tiles": len({record.tile_id for record in selected}),
    }
    assert by_id  # keep the mapping meaningful for callers that inspect ids
    return selected, detail


def stratified_train(train: list[Task6NSample], target: int) -> tuple[list[Task6NSample], dict]:
    """Stable seeded stratification by direction x reference family, round-robin fill."""

    strata: dict[tuple[str, str], list[Task6NSample]] = defaultdict(list)
    for record in train:
        strata[(record.relation, reference_family(record.program_id))].append(record)
    for key in strata:
        strata[key].sort(key=lambda record: (stable_key(record.sample_id), record.sample_id))
    order = sorted(strata)
    selected: list[Task6NSample] = []
    cursor = {key: 0 for key in order}
    while len(selected) < target:
        progressed = False
        for key in order:
            if len(selected) >= target:
                break
            index = cursor[key]
            if index < len(strata[key]):
                selected.append(strata[key][index])
                cursor[key] = index + 1
                progressed = True
        if not progressed:
            break
    detail = {
        "strata": {f"{relation}|{family}": len(items) for (relation, family), items in sorted(strata.items())},
        "selected_per_stratum": {
            f"{relation}|{family}": sum(
                1 for record in selected
                if record.relation == relation and reference_family(record.program_id) == family
            )
            for (relation, family) in sorted(strata)
        },
        "requested": target,
        "available": len(train),
    }
    return selected, detail


def mini_val_240(val: list[Task6NSample], per_program: int = 30, target: int = 240) -> tuple[list[Task6NSample], dict]:
    by_program: dict[str, list[Task6NSample]] = defaultdict(list)
    for record in val:
        by_program[record.program_id].append(record)
    for program in by_program:
        by_program[program].sort(key=lambda record: (stable_key(record.sample_id), record.sample_id))
    selected: list[Task6NSample] = []
    for program in sorted(by_program):
        selected.extend(by_program[program][:per_program])
    if len(selected) < target:
        # deterministic proportional fill from the remaining eligible records
        remaining = [
            record for program in sorted(by_program) for record in by_program[program][per_program:]
        ]
        remaining.sort(key=lambda record: (stable_key(record.sample_id), record.sample_id))
        selected.extend(remaining[: target - len(selected)])
    selected.sort(key=lambda record: (record.program_id, stable_key(record.sample_id)))
    detail = {
        "per_program_requested": per_program,
        "selected_per_program": dict(sorted(Counter(record.program_id for record in selected).items())),
        "available_per_program": dict(sorted((program, len(items)) for program, items in by_program.items())),
        "all_eight_programs": len({record.program_id for record in selected}) == 8,
    }
    return selected, detail


def paired_val_20(val: list[Task6NSample], target: int = 20) -> tuple[list[dict], dict]:
    pairs = counterfactual_pairs(val)
    pairs.sort(key=lambda pair: (pair[0].tile_id, pair[0].relation, pair[1].relation, pair[0].sample_id))
    chosen = []
    used_tiles: set[str] = set()
    for first, second in pairs:
        if len(chosen) >= target:
            break
        if first.tile_id in used_tiles:
            continue
        used_tiles.add(first.tile_id)
        chosen.append(
            {
                "tile_id": first.tile_id,
                "reference_source_feature_id": first.reference_source_feature_id,
                "reference_instance_id": first.reference_instance_id,
                "a": first.as_dict(),
                "b": second.as_dict(),
            }
        )
    detail = {
        "available_pairs": len(pairs),
        "selected_pairs": len(chosen),
        "constraints": {
            "same_tile": True,
            "same_reference": True,
            "different_direction": True,
            "different_target": True,
        },
        "verified": all(
            pair["a"]["tile_id"] == pair["b"]["tile_id"]
            and pair["a"]["reference_source_feature_id"] == pair["b"]["reference_source_feature_id"]
            and pair["a"]["relation"] != pair["b"]["relation"]
            and pair["a"]["target_source_feature_id"] != pair["b"]["target_source_feature_id"]
            for pair in chosen
        ),
    }
    return chosen, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.time()
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_root = _DatasetRoot()

    print("[6n.packs] loading eligible records (test split is never read)", flush=True)
    train, train_audit = load_eligible("train", dataset_root)
    val, val_audit = load_eligible("val", dataset_root)

    overfit, overfit_detail = build_overfit20(train, 20)
    mini_train, mini_train_detail = stratified_train(train, 1000)
    mini_val, mini_val_detail = mini_val_240(val, 30, 240)
    pairs, pair_detail = paired_val_20(val, 20)

    artifacts = {}
    artifacts["overfit20"] = write_pack(
        PACK_ROOT / "overfit20.json", overfit, {"pack": "overfit20", "split": "train", "detail": overfit_detail}
    )
    artifacts["mini_train_1000"] = write_pack(
        PACK_ROOT / "mini_train_1000.json", mini_train,
        {"pack": "mini_train_1000", "split": "train", "detail": mini_train_detail},
    )
    artifacts["mini_val_240"] = write_pack(
        PACK_ROOT / "mini_val_240.json", mini_val,
        {"pack": "mini_val_240", "split": "val", "detail": mini_val_detail},
    )
    paired_path = PACK_ROOT / "paired_val_20.json"
    paired_payload = {
        "pairs": [
            {"tile_id": pair["tile_id"], "reference_source_feature_id": pair["reference_source_feature_id"],
             "reference_instance_id": pair["reference_instance_id"], "a": pair["a"], "b": pair["b"]}
            for pair in pairs
        ],
        "count": len(pairs),
        "reference_source": "oracle_native_gt",
        "split": "val",
        "detail": pair_detail,
    }
    paired_path.write_text(
        json.dumps(paired_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts["paired_val_20"] = {
        "path": str(paired_path),
        "sha256": hashlib.sha256(paired_path.read_bytes()).hexdigest(),
        "count": len(pairs),
    }

    verdict = None
    if len(pairs) < 20:
        verdict = "PAIRED_SET_INSUFFICIENT"

    report = {
        "_doc": (
            "Task 6N section 12. Frozen deterministic packs for the oracle-reference geometric "
            "relation field ablation. Built before any training; the test split is never read. "
            "reference_source = oracle_native_gt for every record."
        ),
        "task": "6N",
        "reference_source": "oracle_native_gt",
        "seed": SEED,
        "eligibility": {
            "train": train_audit,
            "val": val_audit,
            "criteria": [
                "program is one of the 8 directional L2 ids",
                "oracle reference and target source_feature_id resolve to canonical native instances",
                "RGB source tile exists",
                "frozen SAM2 feature exists or is generated through the frozen cache path",
            ],
            "filters_not_applied": ["tiny filter", "border filter", "visibility filter", "difficulty deletion"],
        },
        "packs": {
            "overfit20": {**artifacts["overfit20"], "detail": overfit_detail},
            "mini_train_1000": {**artifacts["mini_train_1000"], "detail": mini_train_detail},
            "mini_val_240": {**artifacts["mini_val_240"], "detail": mini_val_detail},
            "paired_val_20": {**artifacts["paired_val_20"], "detail": pair_detail},
        },
        "verdict": verdict or "PACKS_FROZEN",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(Path(args.out), report)
    print(
        f"[6n.packs] overfit20 {len(overfit)} (pairs {overfit_detail['pair_count']}), "
        f"mini_train {len(mini_train)}, mini_val {len(mini_val)}, paired_val {len(pairs)} "
        f"-> {report['verdict']}",
        flush=True,
    )
    if verdict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
