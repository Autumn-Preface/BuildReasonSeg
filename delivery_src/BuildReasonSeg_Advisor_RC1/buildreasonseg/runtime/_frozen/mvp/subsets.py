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
from pathlib import Path
from typing import Iterable, Sequence

from . import data as data_mod

#: Repository root, used to locate committed subset id files.
REPO_ROOT = Path(__file__).resolve().parents[1]

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


# ---------------------------------------------------------------------------
# Task 6C -- paired counterfactual training subset
# ---------------------------------------------------------------------------

#: (level of instruction A, level of instruction B, how many images)
PAIR_COMPOSITION = ((1, 2, 80), (1, 3, 80), (2, 3, 80))
PAIRED_IMAGES = sum(quota for _a, _b, quota in PAIR_COMPOSITION)
PAIRED_RECORDS = PAIRED_IMAGES * 2
TASK6B_SUBSET_JSON = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"


def eligible_for_pairing(records: Sequence[dict]) -> list[dict]:
    """Level 1, level 2, or nontrivial level 3."""

    return [
        record
        for record in records
        if int(record["level"]) in (1, 2) or is_nontrivial_level3(record)
    ]


def _pair_candidates(by_image: dict[str, list[dict]], level_a: int, level_b: int) -> list[tuple]:
    """All (image_id, rec_a, rec_b) satisfying the Task 6C pair constraints."""

    candidates: list[tuple] = []
    for image_id in sorted(by_image):
        left = sorted(
            (r for r in by_image[image_id] if int(r["level"]) == level_a), key=lambda r: r["sample_id"]
        )
        right = sorted(
            (r for r in by_image[image_id] if int(r["level"]) == level_b), key=lambda r: r["sample_id"]
        )
        for rec_a in left:
            for rec_b in right:
                if int(rec_a["target_component_id"]) == int(rec_b["target_component_id"]):
                    continue
                if rec_a["query_type"] == rec_b["query_type"]:
                    continue
                candidates.append((image_id, rec_a, rec_b))
    candidates.sort(
        key=lambda item: (
            item[1]["query_type"],
            item[2]["query_type"],
            item[0],
            item[1]["sample_id"],
            item[2]["sample_id"],
        )
    )
    return candidates


def build_paired_480(
    records: Sequence[dict], composition=PAIR_COMPOSITION
) -> tuple[list[dict], dict]:
    """Task 6C factor 1 `P`: 240 unique images x 2 instructions.

    Constraints, all checked in the returned audit:

    * every pair has two different target components;
    * every pair has two different query types;
    * no trivial level 3;
    * level counts are exactly 160 / 160 / 160 via the 80 + 80 + 80 composition.

    Selection is deterministic: candidates are ordered by query type, image id and
    sample id, and each step picks the candidate whose query-type usage is currently
    the least used, with that ordering as the tie-break. No model output participates.
    """

    by_image: dict[str, list[dict]] = defaultdict(list)
    for record in eligible_for_pairing(records):
        by_image[record["image_id"]].append(record)

    per_bucket = {
        f"L{level_a}_L{level_b}": _pair_candidates(by_image, level_a, level_b)
        for level_a, level_b, _quota in composition
    }
    eligible_images = {
        name: len({candidate[0] for candidate in candidates})
        for name, candidates in per_bucket.items()
    }
    quotas = {f"L{a}_L{b}": quota for a, b, quota in composition}
    exact_composition_feasible = all(
        eligible_images[name] >= quota for name, quota in quotas.items()
    )

    used_images: set[str] = set()
    chosen: list[dict] = []
    selection_audit: dict[str, dict] = {}
    for level_a, level_b, quota in composition:
        name = f"L{level_a}_L{level_b}"
        candidates = per_bucket[name]
        type_usage_a: Counter = Counter()
        type_usage_b: Counter = Counter()
        picked: list[tuple] = []
        # One slot at a time; each slot re-scores the remaining candidates so the
        # query-type counters stay balanced as selection proceeds.
        while len(picked) < quota:
            best = None
            best_key = None
            for image_id, rec_a, rec_b in candidates:
                if image_id in used_images:
                    continue
                key = (
                    type_usage_a[rec_a["query_type"]],
                    type_usage_b[rec_b["query_type"]],
                    rec_a["query_type"],
                    rec_b["query_type"],
                    image_id,
                    rec_a["sample_id"],
                    rec_b["sample_id"],
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (image_id, rec_a, rec_b)
            if best is None:
                break
            image_id, rec_a, rec_b = best
            used_images.add(image_id)
            type_usage_a[rec_a["query_type"]] += 1
            type_usage_b[rec_b["query_type"]] += 1
            picked.append(best)
            chosen.extend([rec_a, rec_b])
        selection_audit[name] = {
            "level_a": level_a,
            "level_b": level_b,
            "quota": quota,
            "selected_images": len(picked),
            "eligible_images": eligible_images[name],
            "feasible": eligible_images[name] >= quota,
            "query_types_a": dict(sorted(type_usage_a.items())),
            "query_types_b": dict(sorted(type_usage_b.items())),
        }

    chosen.sort(key=lambda r: r["sample_id"])
    pairs: list[dict] = []
    for index in range(0, len(chosen) - 1, 2):
        left, right = chosen[index], chosen[index + 1]
        pairs.append(
            {
                "image_id": left["image_id"],
                "a": left["sample_id"],
                "b": right["sample_id"],
                "a_level": int(left["level"]),
                "b_level": int(right["level"]),
                "a_query_type": left["query_type"],
                "b_query_type": right["query_type"],
                "a_target": int(left["target_component_id"]),
                "b_target": int(right["target_component_id"]),
            }
        )

    levels = Counter(str(r["level"]) for r in chosen)
    per_image = Counter(r["image_id"] for r in chosen)
    audit = {
        "composition_requested": [list(item) for item in composition],
        "composition_feasible_exactly": bool(exact_composition_feasible),
        "eligible_images_per_bucket": eligible_images,
        "selection": selection_audit,
        "n_records": len(chosen),
        "n_images": len(per_image),
        "records_per_image_values": sorted(set(per_image.values())),
        "max_records_per_image": max(per_image.values()) if per_image else 0,
        "by_level": dict(sorted(levels.items())),
        "nontrivial_l3": sum(1 for r in chosen if is_nontrivial_level3(r)),
        "trivial_l3": sum(
            1 for r in chosen if int(r["level"]) == 3 and bool(r.get("trivial_selection", False))
        ),
        "all_pairs_different_targets": all(p["a_target"] != p["b_target"] for p in pairs),
        "all_pairs_different_query_types": all(p["a_query_type"] != p["b_query_type"] for p in pairs),
        "n_pairs": len(pairs),
        "query_type_histogram": dict(sorted(Counter(r["query_type"] for r in chosen).items())),
        "sample_ids": [r["sample_id"] for r in chosen],
        "pairs": pairs,
    }
    return chosen, audit


def _ids_sha256(sample_ids: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_task6c(train_version: str = data_mod.DATASET_VERSION) -> dict:
    """Task 6C subsets: `U` (Task 6B's exact 480) and `P` (240 paired images)."""

    import json

    train_records = data_mod.read_records("train", train_version)
    by_id = {record["sample_id"]: record for record in train_records}

    payload = json.loads(TASK6B_SUBSET_JSON.read_text(encoding="utf-8"))
    u_ids = list(payload["train"]["sample_ids"])
    missing = [sample_id for sample_id in u_ids if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"Task 6B train subset ids missing from the train split: {missing[:5]}")
    u_records = sorted((by_id[sample_id] for sample_id in u_ids), key=lambda r: r["sample_id"])

    p_records, p_audit = build_paired_480(train_records)

    return {
        "_doc": (
            "Task 6C section 5 deterministic subsets. U is Task 6B's exact 480-record train subset "
            "(same sample ids, unchanged); P is a new 240-image x 2-instruction counterfactual subset. "
            "Both come from the train split only; the test split is never touched."
        ),
        "dataset_version": train_version,
        "source_split": "train",
        "test_split_used": False,
        "U": {
            **summarise(u_records, "task6c_U_480_unique_images"),
            "definition": "unpaired / unique-image, identical to the Task 6B training subset",
            "source": "evaluation/task6b_subset_ids.json -> train.sample_ids",
            "source_sample_ids_sha256": _ids_sha256(u_ids),
            "sample_ids": [r["sample_id"] for r in u_records],
        },
        "P": dict(p_audit),
    }
