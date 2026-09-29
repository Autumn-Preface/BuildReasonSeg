"""Task 6P Part D (sections 6-8) — deduplicated reference packs for the ReferenceMaskHead.

Builds, from the frozen Task 6N eligible scope (the same 8 directional L2 programs):

* **RefTrainUnique** / **RefValUnique** — one record per unique
  ``(split, tile_id, reference_source_feature_id, reference_family)`` key, so relations that share a
  reference contribute a single reference-training example;
* **Reference Overfit20** — 20 unique train references, 10 per family when available, distinct
  ``(tile, source_feature_id)``, no duplicate reference mask, with a deterministic fill from the other
  family when one has fewer than 10.

The **target building is never part of the reference packs**: only the reference identity, the image,
the family and the (already frozen) SAM2 feature path are stored. The test split is never read.

Writes `evaluation/task6p_reference_pack_manifest.json` and the packs under
`artifacts/task6p/reference_packs/` (gitignored, regenerable).

    python scripts/task6p_freeze_reference_packs.py
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
from buildreasonseg_mvp.task6n_relation_decoder import DIRECTIONAL_PROGRAMS, read_pack  # noqa: E402
from buildreasonseg_mvp.task6p_reference_head import family_of_program  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
REF_ROOT = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
OUT = EVAL / "task6p_reference_pack_manifest.json"
SEED = 20260929
FAMILIES = ("largest", "smallest")


def stable_key(text: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def reference_record(record) -> dict:
    """A reference-only record: no target identity and no relation id are stored."""

    return {
        "sample_id": record.sample_id,
        "split": record.split,
        "tile_id": record.tile_id,
        "image_path": record.image_path,
        "reference_source_feature_id": int(record.reference_source_feature_id),
        "reference_instance_id": int(record.reference_instance_id),
        "reference_family": family_of_program(record.program_id),
        "source_program_id": record.program_id,
        "reference_source": "oracle_native_gt",
    }


def unique_references(records) -> list[dict]:
    """One record per (split, tile_id, reference_source_feature_id, reference_family)."""

    seen: dict[tuple, dict] = {}
    for record in records:
        if record.program_id not in DIRECTIONAL_PROGRAMS:
            continue
        entry = reference_record(record)
        key = (entry["split"], entry["tile_id"], entry["reference_source_feature_id"],
               entry["reference_family"])
        if key not in seen:
            seen[key] = entry
    ordered = [seen[key] for key in sorted(seen)]
    return ordered


def mask_digest(tile_id: str, source_feature_id: int) -> str:
    instances = canonical_instances(tile_id)
    for instance in instances:
        if instance.source_feature_id == source_feature_id:
            return hashlib.sha256(instance.mask.tobytes()).hexdigest()
    raise KeyError(f"{tile_id}: reference {source_feature_id} not in the canonical native store")


def build_overfit20(train: list[dict], target: int = 20, per_family: int = 10) -> tuple[list[dict], dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for entry in train:
        by_family[entry["reference_family"]].append(entry)
    for family in by_family:
        by_family[family].sort(key=lambda item: (stable_key(f"{item['tile_id']}:"
                                                            f"{item['reference_source_feature_id']}"),
                                                 item["tile_id"]))

    selected: list[dict] = []
    used_keys: set[tuple] = set()
    used_digests: set[str] = set()
    per_family_counts = {}
    for family in FAMILIES:
        picked = 0
        for entry in by_family.get(family, []):
            if picked >= per_family:
                break
            key = (entry["tile_id"], entry["reference_source_feature_id"])
            digest = mask_digest(*key)
            if key in used_keys or digest in used_digests:
                continue
            selected.append({**entry, "mask_sha256": digest})
            used_keys.add(key)
            used_digests.add(digest)
            picked += 1
        per_family_counts[family] = picked

    shortfall = target - len(selected)
    fill_from = []
    if shortfall > 0:
        for family in FAMILIES:
            for entry in by_family.get(family, []):
                key = (entry["tile_id"], entry["reference_source_feature_id"])
                if key in used_keys:
                    continue
                fill_from.append(entry)
        fill_from.sort(key=lambda item: (stable_key(f"{item['tile_id']}:"
                                                    f"{item['reference_source_feature_id']}"),
                                         item["tile_id"]))
        for entry in fill_from:
            if len(selected) >= target:
                break
            key = (entry["tile_id"], entry["reference_source_feature_id"])
            digest = mask_digest(*key)
            if digest in used_digests:
                continue
            selected.append({**entry, "mask_sha256": digest})
            used_keys.add(key)
            used_digests.add(digest)

    selected.sort(key=lambda item: (item["reference_family"], item["tile_id"],
                                    item["reference_source_feature_id"]))
    detail = {
        "requested": target,
        "per_family_requested": per_family,
        "per_family_selected": dict(sorted(Counter(item["reference_family"]
                                                  for item in selected).items())),
        "per_family_available": {family: len(by_family.get(family, [])) for family in FAMILIES},
        "shortfall_filled_from_other_family": bool(shortfall > 0),
        "distinct_tile_feature_keys": len({(item["tile_id"], item["reference_source_feature_id"])
                                           for item in selected}),
        "distinct_masks": len({item["mask_sha256"] for item in selected}),
        "target_included": False,
    }
    return selected, detail


def write_pack(path: Path, records: list[dict], extra: dict) -> dict:
    payload = {"records": records, "count": len(records), "reference_source": "oracle_native_gt", **extra}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": len(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.time()
    REF_ROOT.mkdir(parents=True, exist_ok=True)
    train_records = read_pack(PACK_ROOT / "mini_train_1000.json")
    val_records = read_pack(PACK_ROOT / "mini_val_240.json")

    train_unique = unique_references(train_records)
    val_unique = unique_references(val_records)
    overfit, overfit_detail = build_overfit20(train_unique, 20, 10)

    packs = {
        "ref_train_unique": write_pack(
            REF_ROOT / "ref_train_unique.json", train_unique,
            {"pack": "ref_train_unique", "split": "train",
             "unique_key": "(split, tile_id, reference_source_feature_id, reference_family)"},
        ),
        "ref_val_unique": write_pack(
            REF_ROOT / "ref_val_unique.json", val_unique,
            {"pack": "ref_val_unique", "split": "val",
             "unique_key": "(split, tile_id, reference_source_feature_id, reference_family)"},
        ),
        "reference_overfit20": write_pack(
            REF_ROOT / "reference_overfit20.json", overfit,
            {"pack": "reference_overfit20", "split": "train", "detail": overfit_detail},
        ),
    }

    source_counts = {
        "mini_train_1000_records": len(train_records),
        "mini_val_240_records": len(val_records),
        "ref_train_unique": len(train_unique),
        "ref_val_unique": len(val_unique),
        "dedup_ratio_train": len(train_records) / max(len(train_unique), 1),
    }
    payload = {
        "_doc": (
            "Task 6P sections 6-8. Deduplicated reference packs for the ReferenceMaskHead. Unique key "
            "= (split, tile_id, reference_source_feature_id, reference_family); relations sharing a "
            "reference contribute one record. Reference-only: no target identity, no relation id. The "
            "test split is never read."
        ),
        "task": "6P",
        "stage": "D-reference-packs",
        "reference_source": "oracle_native_gt",
        "seed": SEED,
        "scope_programs": sorted(DIRECTIONAL_PROGRAMS),
        "unique_key": "(split, tile_id, reference_source_feature_id, reference_family)",
        "packs": {name: {**entry, "detail": overfit_detail if name == "reference_overfit20" else None}
                  for name, entry in packs.items()},
        "counts": source_counts,
        "family_counts": {
            "ref_train_unique": dict(sorted(Counter(item["reference_family"]
                                                    for item in train_unique).items())),
            "ref_val_unique": dict(sorted(Counter(item["reference_family"]
                                                  for item in val_unique).items())),
            "reference_overfit20": dict(sorted(Counter(item["reference_family"]
                                                       for item in overfit).items())),
        },
        "target_included_in_reference_packs": False,
        "relation_id_stored_in_reference_packs": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(Path(args.out), payload)
    print(
        f"[6p.refs] train {len(train_records)} -> {len(train_unique)} unique | "
        f"val {len(val_records)} -> {len(val_unique)} unique | overfit20 {len(overfit)} "
        f"({overfit_detail['per_family_selected']})", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
