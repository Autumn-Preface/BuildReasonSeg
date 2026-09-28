"""Task 6L section 18: v0.1.1 vs v0.2 comparison under `legacy_compat_v1`.

Both versions are evaluated on the SAME 4,038 historical positive tiles using the recovered
`legacy_compat_v1` split, so the only differences are the annotation truth (pseudo-instances vs
native instances) and the candidate geometry. The comparison unit is `(tile_id, query_type)`, which
the quotas guarantee is at most one sample per version.

Measured: common queries, target unchanged/changed (mask-overlap matching at IoU 0.5), became
valid/invalid, the weighted answer-change rate and the per-level / per-query-type breakdown. The
result should broadly reconcile with Task 6K.1's ~6.96 % VECTOR<->PSEUDO drift.

    python scripts/task6l_compare_v01_v02.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import annotator as A  # noqa: E402
import thresholds as T  # noqa: E402

from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset  # noqa: E402
from buildreasonseg_mvp.whu_native_vector import (  # noqa: E402
    DATASET_ROOT,
    REASONING_VIEW_ROOT,
    write_json,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6l_v01_vs_v02_comparison.json"
V011 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
LEGACY_VIEW = "legacy_compat_v1"
SPLITS = ("train", "val", "test")

#: frozen v0.1.1 query-type counts, used to weight the answer-change rate
V011_QUERY_WEIGHTS = json.loads((V011 / "manifest.json").read_text(encoding="utf-8"))["sample_counts"]["by_query_type"]


def iou_masks(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left).astype(bool)
    b = np.asarray(right).astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def load_v011_answers() -> dict[tuple[str, str], list[dict]]:
    """All v0.1.1 answers grouped by `(tile_id, query_type)`.

    A group can hold more than one answer (the same program can be generated against different
    references), so the comparison treats each group as a SET of target geometries.
    """

    answers: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for split in SPLITS:
        path = V011 / f"{split}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (str(record["image_id"]), str(record["query_type"]))
                answers[key].append(
                    {
                        "split": split,
                        "level": int(record["level"]),
                        "target_component_id": int(record["target_component_id"]),
                    }
                )
    return dict(answers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    gen_config = _load_v02_config()
    dataset_meta = A.dataset_meta_from_config(gen_config)
    relation_config = T.load_config(gen_config["relations"]["config"])
    dataset = NativeVectorDataset(args.dataset_root)

    view = REASONING_VIEW_ROOT / LEGACY_VIEW
    if not (view / "metadata" / "train.jsonl").is_file():
        print(f"error: legacy reasoning view missing at {view}", file=sys.stderr)
        return 2
    A.IMAGE_METADATA_TEMPLATE = (
        f"artifacts/whu_native_vector/reasoning_view/{LEGACY_VIEW}/metadata/{{split}}.jsonl"
    )

    v011 = load_v011_answers()
    v02: dict[tuple[str, str], list[dict]] = defaultdict(list)
    counter = A.DiscardCounter()
    seen_keys: set[str] = set()
    for split in SPLITS:
        tiles = 0
        for record in _iter_jsonl(view / "metadata" / f"{split}.jsonl"):
            image = A.G.image_geometry_from_record(record)
            for sample in A.generate_for_image(
                image, record, relation_config, gen_config, dataset_meta, counter, seen_keys
            ):
                key = (str(sample["image_id"]), str(sample["query_type"]))
                v02[key].append(
                    {
                        "split": split,
                        "level": int(sample["level"]),
                        "target_component_id": int(sample["target_component_id"]),
                    }
                )
            tiles += 1
        if not args.quiet:
            print(f"[6l.cmp] {split}: {tiles} tiles, {sum(len(v) for k, v in v02.items() if v[0]['split'] == split)} answers",
                  flush=True)
    v02 = dict(v02)

    # ---------------------------------------------------------- compare
    pseudo_maps: dict[str, np.ndarray] = {}
    native_maps: dict[str, np.ndarray] = {}

    def pseudo_mask(image_id: str, component_id: int, split: str) -> np.ndarray | None:
        if image_id not in pseudo_maps:
            path = REPO_ROOT / "datasets" / "whu" / "components" / split / f"{image_id}.png"
            if not path.is_file():
                pseudo_maps[image_id] = np.zeros((0, 0), dtype=np.uint8)
            else:
                import cv2

                pseudo_maps[image_id] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        array = pseudo_maps[image_id]
        if array.size == 0:
            return None
        return array == int(component_id)

    def native_mask(image_id: str, instance_id: int) -> np.ndarray | None:
        if image_id not in native_maps:
            label = dataset.label_map(image_id)
            native_maps[image_id] = label if label is not None else np.zeros((0, 0), dtype=np.uint8)
        array = native_maps[image_id]
        if array.size == 0:
            return None
        return array == int(instance_id)

    common = sorted(set(v011) & set(v02))
    only_v011 = sorted(set(v011) - set(v02))
    only_v02 = sorted(set(v02) - set(v011))

    per_query = defaultdict(lambda: Counter())
    per_level = defaultdict(lambda: Counter())
    unchanged = changed = unmatched_geometry = 0
    group_sizes_011: Counter = Counter()
    group_sizes_02: Counter = Counter()
    for key in common:
        image_id, query_type = key
        left_group = v011[key]
        right_group = v02[key]
        group_sizes_011[len(left_group)] += 1
        group_sizes_02[len(right_group)] += 1
        left_masks = [pseudo_mask(image_id, entry["target_component_id"], entry["split"]) for entry in left_group]
        right_masks = [native_mask(image_id, entry["target_component_id"]) for entry in right_group]
        if any(mask is None for mask in left_masks) or any(mask is None for mask in right_masks):
            unmatched_geometry += 1
            status = "became_invalid"
        else:
            # every v0.1.1 answer must have a distinct v0.2 counterpart at IoU >= 0.5 (and vice versa)
            remaining = list(range(len(right_masks)))
            perfect = True
            for mask in left_masks:
                best = None
                for index in remaining:
                    value = iou_masks(mask, right_masks[index])
                    if best is None or value > best[0]:
                        best = (value, index)
                if best is None or best[0] < 0.5:
                    perfect = False
                    break
                remaining.remove(best[1])
            if perfect and not remaining:
                status = "unchanged"
            else:
                status = "changed"
        if status == "unchanged":
            unchanged += 1
        elif status == "changed":
            changed += 1
        per_query[query_type][status] += 1
        per_level[right_group[0]["level"]][status] += 1

    comparable = unchanged + changed
    weighted_changes = 0.0
    weighted_total = 0.0
    for query_type, counter_for_query in per_query.items():
        comparable_for_query = counter_for_query["unchanged"] + counter_for_query["changed"]
        weight = V011_QUERY_WEIGHTS.get(query_type, 0)
        if comparable_for_query and weight:
            weighted_total += weight
            weighted_changes += weight * (counter_for_query["changed"] / comparable_for_query)
    weighted_rate = float(weighted_changes / weighted_total) if weighted_total else None

    report = {
        "_doc": (
            "Task 6L section 18. v0.1.1 (pseudo-instance truth, random split) vs v0.2 (native-vector "
            "truth) on the identical 4,038 historical tiles under legacy_compat_v1. The comparison "
            "unit is (tile_id, query_type); targets are compared by mask IoU >= 0.5."
        ),
        "task": "6L",
        "scope": {
            "tiles": 4038,
            "split_view": LEGACY_VIEW,
            "v0.1.1_answers": sum(len(v) for v in v011.values()),
            "v0.2_answers": sum(len(v) for v in v02.values()),
            "common_query_groups": len(common),
            "only_v0.1.1_groups": len(only_v011),
            "only_v0.2_groups": len(only_v02),
            "comparison_unit": "(tile_id, query_type) group; every member target must have a distinct counterpart at IoU >= 0.5",
            "group_size_distribution_v011": dict(sorted(group_sizes_011.items())),
            "group_size_distribution_v02": dict(sorted(group_sizes_02.items())),
        },
        "answer_change": {
            "target_unchanged": unchanged,
            "target_changed": changed,
            "comparable": comparable,
            "target_change_rate": float(changed / comparable) if comparable else None,
            "unmatched_geometry": unmatched_geometry,
            "became_valid_in_v02": len(only_v02),
            "became_invalid_in_v02": len(only_v011),
        },
        "weighted_current_query_target_change_rate": weighted_rate,
        "weighting": {
            "method": "per-query-type target-change rate weighted by the frozen v0.1.1 query-type counts",
            "weight_records": int(weighted_total),
        },
        "by_level": {
            str(level): {
                **dict(counter),
                "change_rate": (
                    counter["changed"] / (counter["unchanged"] + counter["changed"])
                    if (counter["unchanged"] + counter["changed"]) else None
                ),
            }
            for level, counter in sorted(per_level.items())
        },
        "by_query_type": {
            query: {
                **dict(counter),
                "change_rate": (
                    counter["changed"] / (counter["unchanged"] + counter["changed"])
                    if (counter["unchanged"] + counter["changed"]) else None
                ),
            }
            for query, counter in sorted(per_query.items())
        },
        "reconciliation": {
            "task6k1_vector_vs_pseudo_weighted_drift": 0.06959,
            "note": (
                "Task 6K.1's 6.96% is a VECTOR-vs-PSEUDO drift over all 20 programs with the frozen "
                "executor; this comparison runs the actual v0.1.1/v0.2 generated datasets and is "
                "expected to land in the same range. Differences come from quota/eligibility "
                "interaction, not from a different relation definition."
            ),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[6l.cmp] common {len(common)}; unchanged {unchanged}; changed {changed}; "
        f"only-v0.1.1 {len(only_v011)}; only-v0.2 {len(only_v02)}; weighted change {weighted_rate}",
        flush=True,
    )
    print(f"[6l.cmp] wrote {OUT.name}", flush=True)
    return 0


def _load_v02_config() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / "configs" / "build_spatial_reason_v0.2.yaml").read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
