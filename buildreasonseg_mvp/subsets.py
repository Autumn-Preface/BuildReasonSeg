"""Deterministic Task 6B subset selection.

Task 6B needs three fixed subsets, all decided **before** any training and never
revised using model results:

* a 480-record train mini-set (160 L1 + 160 L2 + 160 nontrivial L3);
* a 120-record validation mini-set (40 + 40 + 40), drawn from the **val split only**;
* a 40-record paired validation probe (20 val images x 2 instructions, same image,
  different query types and different targets).

Selection rules
---------------
* buckets are formed by a per-level key (query type for L1/L2, direction for L3)
  and filled round-robin, so families are balanced as evenly as the data allows;
* within a bucket, candidates are ordered by `sample_id`, so the result is
  independent of file order and of any model behaviour;
* an image is used at most once per subset unless a bucket runs out of fresh
  images, in which case the reuse is counted and reported rather than hidden;
* the test split is never touched.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from . import data as data_mod

#: Requested sizes (Task 6B section 9).
TRAIN_SIZE_BY_LEVEL = {1: 160, 2: 160, 3: 160}
VAL_SIZE_BY_LEVEL = {1: 40, 2: 40, 3: 40}
PAIRED_IMAGES = 20

L1_TYPES = ("leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest")
L3_DIRECTIONS = ("left_of", "right_of", "above", "below")


def is_nontrivial_level3(record: dict) -> bool:
    return int(record["level"]) == 3 and not bool(record.get("trivial_selection", False))


def bucket_key(record: dict) -> str:
    """Balance key: L1/L2 by query type, L3 by direction."""

    level = int(record["level"])
    query_type = record["query_type"]
    if level == 3:
        parts = query_type.split("_to_")
        return parts[1] if len(parts) == 3 else query_type
    return query_type


def _candidates_for_level(records: Sequence[dict], level: int) -> list[dict]:
    if level == 3:
        return [r for r in records if is_nontrivial_level3(r)]
    return [r for r in records if int(r["level"]) == level]


def select_balanced(
    records: Sequence[dict],
    count: int,
    level: int,
    used_images: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Round-robin across buckets, preferring images not already used."""

    used_images = used_images if used_images is not None else set()
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in _candidates_for_level(records, level):
        buckets[bucket_key(record)].append(record)
    for key in buckets:
        buckets[key].sort(key=lambda r: r["sample_id"])

    keys = sorted(buckets)
    if not keys:
        raise RuntimeError(f"no candidates for level {level}")

    chosen: list[dict] = []
    cursor = {key: 0 for key in keys}
    reused_images: list[str] = []

    # pass 1: one sample per image
    while len(chosen) < count:
        progressed = False
        for key in keys:
            if len(chosen) >= count:
                break
            pool = buckets[key]
            while cursor[key] < len(pool):
                candidate = pool[cursor[key]]
                cursor[key] += 1
                if candidate["image_id"] in used_images:
                    continue
                chosen.append(candidate)
                used_images.add(candidate["image_id"])
                progressed = True
                break
        if not progressed:
            break

    # pass 2: allow image reuse, and record it
    if len(chosen) < count:
        for key in keys:
            if len(chosen) >= count:
                break
            pool = buckets[key]
            while cursor[key] < len(pool):
                if len(chosen) >= count:
                    break
                candidate = pool[cursor[key]]
                cursor[key] += 1
                if any(existing["sample_id"] == candidate["sample_id"] for existing in chosen):
                    continue
                chosen.append(candidate)
                reused_images.append(candidate["image_id"])
            if len(chosen) >= count:
                break

    if len(chosen) < count:
        raise RuntimeError(f"level {level}: only {len(chosen)} of {count} records available")
    return chosen[:count], {
        "level": level,
        "requested": count,
        "selected": len(chosen[:count]),
        "buckets": {key: sum(1 for r in chosen[:count] if bucket_key(r) == key) for key in keys},
        "image_reuse_count": len(reused_images),
        "unique_images": len({r["image_id"] for r in chosen[:count]}),
    }


def select_paired_probe(
    records: Sequence[dict],
    images: int = PAIRED_IMAGES,
    used_images: set[str] | None = None,
) -> list[dict]:
    """`images` x 2 records: same image, different query types, different targets."""

    def is_nt3(record: dict) -> bool:
        return is_nontrivial_level3(record)

    by_image: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_image[record["image_id"]].append(record)

    pairs: list[tuple[tuple, list[dict]]] = []
    for image_id in sorted(by_image):
        group = by_image[image_id]
        by_target: dict[int, list[dict]] = defaultdict(list)
        for record in group:
            by_target[int(record["target_component_id"])].append(record)
        targets = sorted(by_target)
        best: tuple[tuple, list[dict]] | None = None
        for i, left_id in enumerate(targets):
            for right_id in targets[i + 1 :]:
                for left in sorted(by_target[left_id], key=lambda r: r["sample_id"]):
                    for right in sorted(by_target[right_id], key=lambda r: r["sample_id"]):
                        pair = [left, right]
                        same_type = left["query_type"] == right["query_type"]
                        levels = {int(left["level"]), int(right["level"])}
                        has_nt3 = any(is_nt3(r) for r in pair)
                        # prefer different query types, then a nontrivial L3, then
                        # different levels, then stable ids
                        key = (
                            same_type,
                            not has_nt3,
                            len(levels) == 1,
                            left["sample_id"],
                            right["sample_id"],
                        )
                        if best is None or key < best[0]:
                            best = (key, pair)
        if best is not None:
            pairs.append(((-int(best[1][0]["level"]), best[1][0]["sample_id"]), best[1]))

    pairs.sort(key=lambda item: item[0])
    used_images = used_images if used_images is not None else set()
    chosen: list[dict] = []
    for _key, pair in pairs:
        if len(chosen) >= images * 2:
            break
        image_id = pair[0]["image_id"]
        if image_id in used_images:
            continue
        if len({int(r["target_component_id"]) for r in pair}) != 2:
            continue
        used_images.add(image_id)
        chosen.extend(sorted(pair, key=lambda r: r["sample_id"]))
    if len(chosen) < images * 2:
        raise RuntimeError(f"only {len(chosen) // 2} usable paired images, need {images}")
    return chosen[: images * 2]


def summarise(records: Sequence[dict], name: str) -> dict:
    levels = Counter(str(r["level"]) for r in records)
    types = Counter(r["query_type"] for r in records)
    images = {r["image_id"] for r in records}
    per_image = Counter(r["image_id"] for r in records)
    return {
        "name": name,
        "n_records": len(records),
        "n_images": len(images),
        "records_per_image_max": max(per_image.values()) if per_image else 0,
        "by_level": dict(sorted(levels.items())),
        "by_query_type": dict(sorted(types.items())),
        "nontrivial_l3": sum(1 for r in records if is_nontrivial_level3(r)),
        "trivial_l3": sum(1 for r in records if int(r["level"]) == 3 and bool(r.get("trivial_selection", False))),
        "sample_ids": [r["sample_id"] for r in records],
    }


def build_all(train_version: str = data_mod.DATASET_VERSION) -> dict:
    """Build every Task 6B subset from the frozen dataset."""

    train_records = data_mod.read_records("train", train_version)
    val_records = data_mod.read_records("val", train_version)

    train_used: set[str] = set()
    train_records_out: list[dict] = []
    train_report: dict[int, dict] = {}
    for level in (1, 2, 3):
        picked, report = select_balanced(train_records, TRAIN_SIZE_BY_LEVEL[level], level, train_used)
        train_records_out.extend(picked)
        train_report[level] = report

    val_used: set[str] = set()
    val_records_out: list[dict] = []
    val_report: dict[int, dict] = {}
    for level in (1, 2, 3):
        picked, report = select_balanced(val_records, VAL_SIZE_BY_LEVEL[level], level, val_used)
        val_records_out.extend(picked)
        val_report[level] = report

    paired = select_paired_probe(val_records, PAIRED_IMAGES)

    train_records_out.sort(key=lambda r: r["sample_id"])
    val_records_out.sort(key=lambda r: r["sample_id"])
    paired = sorted(paired, key=lambda r: (r["image_id"], r["sample_id"]))

    return {
        "_doc": (
            "Task 6B section 9 deterministic subsets. Chosen before any training and never revised "
            "using model results. Test split is not touched."
        ),
        "dataset_version": train_version,
        "train": {
            **summarise(train_records_out, "train_mini_480"),
            "per_level_selection": train_report,
            "source_split": "train",
            "duplicate_sample_ids": len(train_records_out) - len({r["sample_id"] for r in train_records_out}),
        },
        "val": {
            **summarise(val_records_out, "val_mini_120"),
            "per_level_selection": val_report,
            "source_split": "val",
            "duplicate_sample_ids": len(val_records_out) - len({r["sample_id"] for r in val_records_out}),
        },
        "paired_probe": {
            **summarise(paired, "paired_probe_40"),
            "source_split": "val",
            "n_pairs": len(paired) // 2,
            "pairs": [
                {
                    "image_id": paired[index]["image_id"],
                    "a": paired[index]["sample_id"],
                    "b": paired[index + 1]["sample_id"],
                    "a_query_type": paired[index]["query_type"],
                    "b_query_type": paired[index + 1]["query_type"],
                    "a_target": int(paired[index]["target_component_id"]),
                    "b_target": int(paired[index + 1]["target_component_id"]),
                }
                for index in range(0, len(paired) - 1, 2)
            ],
        },
        "sizes": {"train": len(train_records_out), "val": len(val_records_out), "paired": len(paired)},
        "test_split_used": False,
    }


def load_subset(payload: dict, key: str) -> list[str]:
    return list(payload[key]["sample_ids"])
