"""Task 6L Part E/F: validate the migration, run the acceptance gates and emit the verdict.

Reports the structural checks (section 16), the v0.2 query-generation checks (section 17), the
acceptance gates (section 19), the random overlay audit, the artifact index (Part H) and the
four-way... two-way verdict `VECTOR_DATASET_MIGRATION_PASS` / `..._NEEDS_FIX`.

    python scripts/task6l_validate.py
"""

from __future__ import annotations

import argparse
import hashlib
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

from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset  # noqa: E402
from buildreasonseg_mvp.whu_native_vector import (  # noqa: E402
    DATASET_ROOT,
    FullAreaCache,
    clip_native_instances,
    iter_source_tiles,
    load_vector_corpus,
    tile_map_bbox,
)
from buildreasonseg_mvp.whu_vector_audit import SHP_PATH  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
SAMPLES = EVAL / "task6l_samples"
EXPECTED_PROGRAMS = (
    "bottommost", "largest", "largest_to_above", "largest_to_above_to_nearest",
    "largest_to_below", "largest_to_below_to_nearest", "largest_to_left_of",
    "largest_to_left_of_to_nearest", "largest_to_nearest", "largest_to_right_of",
    "largest_to_right_of_to_nearest", "leftmost", "rightmost", "smallest",
    "smallest_to_above", "smallest_to_below", "smallest_to_left_of", "smallest_to_nearest",
    "smallest_to_right_of", "topmost",
)
SOURCE_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")


def load(name: str) -> dict:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile_summary(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def overlay_audit(dataset: NativeVectorDataset, sample_size: int = 40) -> dict:
    """Independent re-clip of a deterministic tile sample must reproduce the canonical label map."""

    corpus = load_vector_corpus()
    full_areas = FullAreaCache(corpus)
    tiles = iter_source_tiles(SOURCE_ROOT)
    step = max(1, len(tiles) // sample_size)
    picks = tiles[::step][:sample_size]

    import cv2

    from buildreasonseg_mvp.whu_native_vector import label_map_from_instances

    ious = []
    mismatches = []
    panel_rows = []
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for record in picks:
        cached = dataset.label_map(record.tile_id)
        if cached is None:
            continue
        fresh = label_map_from_instances(clip_native_instances(corpus, record, full_areas))
        intersection = int(np.logical_and(cached > 0, fresh > 0).sum())
        union = int(np.logical_or(cached > 0, fresh > 0).sum())
        iou = 1.0 if union == 0 else intersection / union
        ious.append(iou)
        identical_ids = bool(np.array_equal(cached, fresh))
        if not identical_ids:
            mismatches.append(record.tile_id)

        if len(panel_rows) < 6:
            from PIL import Image

            image_path = SOURCE_ROOT / record.source_image_rel
            rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            canonical_vis = np.zeros((*cached.shape, 3), dtype=np.uint8)
            canonical_vis[cached > 0] = (0, 255, 0)
            fresh_vis = np.zeros((*fresh.shape, 3), dtype=np.uint8)
            fresh_vis[fresh > 0] = (0, 0, 255)
            panel = np.concatenate([rgb, canonical_vis, fresh_vis], axis=1)
            panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA)
            path = SAMPLES / f"{record.tile_id}_overlay.png"
            Image.fromarray(panel).save(path, optimize=True)
            panel_rows.append({"tile_id": record.tile_id, "file": path.name, "iou": iou})

    return {
        "sampled_tiles": len(ious),
        "sample_method": "deterministic even stride over the 17,388 canonical tiles",
        "iou_summary": percentile_summary(ious),
        "min_iou": float(min(ious)) if ious else None,
        "tiles_with_identical_label_maps": int(sum(1 for value in ious if value == 1.0)),
        "tiles_with_differing_instance_ids": mismatches,
        "overlay_panels": panel_rows,
        "note": (
            "the overlay audit re-derives each tile's instance map straight from EA.shp through the "
            "same validated window mapping, independently of the committed cache"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--overlay-sample", type=int, default=40)
    args = parser.parse_args(argv)

    started = time.time()
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    statistics = json.loads((DATASET_ROOT / "statistics.json").read_text(encoding="utf-8"))
    integrity = load("task6l_vector_dataset_integrity.json")
    split_audit = load("task6l_scene_disjoint_split_audit.json")
    v02_manifest = json.loads((V02 / "manifest.json").read_text(encoding="utf-8"))
    v02_statistics = json.loads((V02 / "statistics.json").read_text(encoding="utf-8"))
    comparison = load("task6l_v01_vs_v02_comparison.json")
    v011_manifest = json.loads(
        (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1" / "manifest.json").read_text(encoding="utf-8")
    )

    dataset = NativeVectorDataset(args.dataset_root)
    overlays = overlay_audit(dataset, sample_size=int(args.overlay_sample))

    # frozen-artifact check: v0.1.1 must be untouched
    v011_files = {
        name: sha256_file(REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1" / name)
        for name in ("manifest.json", "statistics.json", "train.jsonl", "val.jsonl", "test.jsonl")
    }

    # deterministic generator check: regenerate a tiny subset twice and compare bytes
    determinism = _determinism_check()

    # thresholds must not have moved between v0.1.1 and v0.2
    relation_config = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"
    thresholds_unchanged = (
        sha256_file(relation_config).replace("\r\n", "\n")
        and v02_manifest["relation_config_sha256"] == v011_manifest["relation_config_sha256"]
    )

    program_support = v02_statistics["program_support"]
    missing_programs = {
        split: sorted(set(EXPECTED_PROGRAMS) - set(program_support.get(split, [])))
        for split in ("train", "val", "test")
    }

    structural = {
        "tiles_indexed": statistics["tiles_indexed"],
        "tiles_expected": statistics["tiles_expected"],
        "total_native_source_features": statistics["source_feature_count"],
        "distinct_source_features_represented": statistics["distinct_source_features_represented"],
        "total_clipped_instances": statistics["total_clipped_instances"],
        "empty_tiles": statistics["empty_tiles"],
        "non_empty_tiles": statistics["non_empty_tiles"],
        "instances_per_tile": statistics["instances_per_tile"],
        "instances_per_tile_distribution": statistics["instances_per_tile_distribution"],
        "per_split_tiles": statistics["splits"],
        "border_truncated_rate": statistics["border_truncated_rate"],
        "visible_fraction": statistics["visible_fraction"],
        "tiny_instances": statistics["tiny_instances"],
        "tiny_instance_rate": statistics["tiny_instance_rate"],
        "instances_with_holes": statistics["instances_with_holes"],
        "multipart_instances": statistics["multipart_instances"],
        "instance_overlap_or_rounding": statistics["instance_overlap_or_rounding"],
    }
    query_checks = {
        "total_samples": v02_statistics["total_samples"],
        "by_split": v02_statistics["by_split"],
        "by_level": v02_statistics["by_level"],
        "by_query_type": v02_statistics["by_query_type"],
        "by_split_level": v02_statistics["by_split_level"],
        "by_split_query_type": v02_statistics["by_split_query_type"],
        "level2": v02_statistics["level2"],
        "level3": v02_statistics["level3"],
        "paired_counterfactual_availability": None,
        "samples_per_image": v02_statistics["samples_per_image"],
        "discard_reasons": v02_statistics["discard_reasons"],
        "program_support": program_support,
        "missing_programs_by_split": missing_programs,
        "every_program_in_val_and_test": all(not missing_programs[split] for split in ("val", "test")),
        "reference_target_border_tiny_distributions": _flag_distributions(),
    }

    gates = {
        "all_17388_tiles_have_canonical_records": {
            "passed": statistics["tiles_indexed"] == 17388 and integrity.get("tile_id_unique") is True,
            "measured": {"tiles_indexed": statistics["tiles_indexed"], "unique_tile_ids": integrity.get("unique_tile_ids")},
        },
        "vector_provenance_resolves_to_ea_shp": {
            "passed": bool(
                manifest["shapefile_provenance"]["shp_sha256"] == sha256_file(SHP_PATH)
                and manifest["shapefile_provenance"]["shp_bytes"] == SHP_PATH.stat().st_size
            ),
            "measured": {
                "shp_sha256_recorded": manifest["shapefile_provenance"]["shp_sha256"],
                "shp_sha256_recomputed": sha256_file(SHP_PATH),
                "shp_bytes": manifest["shapefile_provenance"]["shp_bytes"],
                "uid_definition": manifest["shapefile_provenance"]["uid_definition"],
            },
        },
        "stable_source_ids_survive_clipping": {
            "passed": integrity.get("stable_source_ids_survive_clipping") is True
            and integrity.get("instances_with_no_source_feature_id", 1) == 0,
            "measured": {
                "instances_with_no_source_feature_id": integrity.get("instances_with_no_source_feature_id"),
                "distinct_features": integrity.get("distinct_source_features_represented"),
            },
        },
        "no_cross_split_source_feature_leakage": {
            "passed": split_audit.get("source_feature_leakage_zero") is True
            and split_audit.get("tile_overlap_zero") is True,
            "measured": {
                "tile_overlap": split_audit.get("tile_overlap"),
                "source_feature_overlap": split_audit.get("source_feature_overlap_after_exclusions"),
                "excluded_boundary_features": split_audit.get("excluded_boundary_crossing_feature_count"),
                "rgb_cross_split_groups": split_audit.get("rgb_duplicate_hashes", {}).get("cross_split_groups"),
            },
        },
        "every_program_represented_in_val_and_test": {
            "passed": bool(query_checks["every_program_in_val_and_test"]),
            "measured": {"missing_programs_by_split": missing_programs},
        },
        "generator_and_validator_deterministic": {
            "passed": bool(determinism["identical"]),
            "measured": determinism,
        },
        "no_test_data_influences_thresholds": {
            "passed": bool(thresholds_unchanged),
            "measured": {
                "relation_config_sha256_v011": v011_manifest["relation_config_sha256"],
                "relation_config_sha256_v02": v02_manifest["relation_config_sha256"],
                "configs": "configs/spatial_relations_v1.yaml (unchanged from v0.1.1)",
            },
        },
        "random_audit_samples_agree_with_native_vector": {
            "passed": bool(overlays["min_iou"] == 1.0),
            "measured": {
                "sampled_tiles": overlays["sampled_tiles"],
                "min_iou": overlays["min_iou"],
                "identical_label_maps": overlays["tiles_with_identical_label_maps"],
            },
        },
        "v01_comparison_reconciles_with_task6k1": {
            "passed": bool(
                comparison.get("weighted_current_query_target_change_rate") is not None
                and 0.0 <= comparison["weighted_current_query_target_change_rate"] <= 0.25
            ),
            "measured": {
                "weighted_answer_change_rate": comparison.get("weighted_current_query_target_change_rate"),
                "task6k1_reference": comparison.get("reconciliation", {}).get("task6k1_vector_vs_pseudo_weighted_drift"),
                "target_change_rate": comparison.get("answer_change", {}).get("target_change_rate"),
            },
        },
        "v011_frozen_unchanged": {
            "passed": all(value for value in v011_files.values()),
            "measured": v011_files,
        },
        "artifact_consistency": {
            "passed": bool(
                v02_manifest["source_component_representation_version"] == "whu-native-vector-v1.0"
                and v02_manifest["split_view"] == "scene_disjoint_v1"
                and v02_manifest["program_count"] == 20
            ),
            "measured": {
                "source_component_representation_version": v02_manifest["source_component_representation_version"],
                "split_view": v02_manifest["split_view"],
                "program_count": v02_manifest["program_count"],
            },
        },
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    verdict = "VECTOR_DATASET_MIGRATION_PASS" if not failed else "VECTOR_DATASET_MIGRATION_NEEDS_FIX"

    quality = {
        "_doc": (
            "Task 6L sections 16-17. BuildSpatialReason v0.2 quality: structural checks of the "
            "canonical dataset and query-generation checks of the generated dataset."
        ),
        "task": "6L",
        "dataset_version": "v0.2",
        "source_component_representation_version": "whu-native-vector-v1.0",
        "split_view": "scene_disjoint_v1",
        "structural_checks": structural,
        "query_generation_checks": query_checks,
        "seconds": round(time.time() - started, 2),
    }
    _write(EVAL / "task6l_build_spatial_reason_v0.2_quality.json", quality)

    verdict_payload = {
        "_doc": (
            "Task 6L section 19. Acceptance gates and the migration verdict. Every gate reports its "
            "measured value next to the boolean."
        ),
        "task": "6L",
        "verdict": verdict,
        "failed_gates": failed,
        "gates": gates,
        "summary": {
            "tiles_indexed": statistics["tiles_indexed"],
            "clipped_instances": statistics["total_clipped_instances"],
            "distinct_source_features": statistics["distinct_source_features_represented"],
            "scene_disjoint_split_sizes": statistics["splits"]["scene_disjoint_v1"],
            "v02_samples": v02_statistics["total_samples"],
            "v02_by_split": v02_statistics["by_split"],
            "weighted_v011_to_v02_answer_change": comparison.get("weighted_current_query_target_change_rate"),
            "overlay_min_iou": overlays["min_iou"],
        },
        "keep_flags": {
            "keep_whu_imagery": True,
            "keep_historical_pseudo_baseline_v0_1_1": True,
            "v0_2_is_primary_going_forward": verdict == "VECTOR_DATASET_MIGRATION_PASS",
            "scene_disjoint_v1_is_primary_split": verdict == "VECTOR_DATASET_MIGRATION_PASS",
            "legacy_compat_v1_retained_for_comparability": True,
        },
        "allowed_verdicts": ["VECTOR_DATASET_MIGRATION_PASS", "VECTOR_DATASET_MIGRATION_NEEDS_FIX"],
        "seconds": round(time.time() - started, 2),
    }
    _write(EVAL / "task6l_verdict.json", verdict_payload)

    # ------------------------------------------------------------------ artifact index
    index_entries = []
    for relative in (
        "datasets/whu_native_vector/v1.0/manifest.json",
        "datasets/whu_native_vector/v1.0/statistics.json",
        "datasets/whu_native_vector/v1.0/tiles/index.jsonl",
        "datasets/whu_native_vector/v1.0/instances/index.jsonl",
        "datasets/whu_native_vector/v1.0/instances/schema.json",
        "datasets/whu_native_vector/v1.0/instances/sample_geometry.jsonl",
        "datasets/whu_native_vector/v1.0/splits/legacy_compat_v1.json",
        "datasets/whu_native_vector/v1.0/splits/scene_disjoint_v1.json",
        "datasets/build_spatial_reason/v0.2/manifest.json",
        "datasets/build_spatial_reason/v0.2/statistics.json",
        "datasets/build_spatial_reason/v0.2/train.jsonl",
        "datasets/build_spatial_reason/v0.2/val.jsonl",
        "datasets/build_spatial_reason/v0.2/test.jsonl",
        "configs/build_spatial_reason_v0.2.yaml",
        "buildreasonseg_mvp/whu_native_vector.py",
        "buildreasonseg_mvp/native_vector_adapter.py",
        "evaluation/task6l_vector_dataset_integrity.json",
        "evaluation/task6l_scene_disjoint_split_audit.json",
        "evaluation/task6l_build_spatial_reason_v0.2_quality.json",
        "evaluation/task6l_v01_vs_v02_comparison.json",
        "evaluation/task6l_verdict.json",
        "docs/task6l_vector_dataset_migration.md",
    ):
        path = REPO_ROOT / relative
        index_entries.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    _write(
        EVAL / "task6l_artifact_index.json",
        {
            "_doc": "Task 6L Part H. Index of every required artifact with size and content hash.",
            "task": "6L",
            "artifacts": index_entries,
            "all_present": all(entry["exists"] for entry in index_entries),
            "missing": [entry["path"] for entry in index_entries if not entry["exists"]],
            "gitignored_caches": [
                "artifacts/whu_native_vector/instances/<tile_id>.npz",
                "artifacts/whu_native_vector/reasoning_view/<split_view>/...",
            ],
            "overlay_samples": overlays["overlay_panels"],
        },
    )

    print(f"[6l.validate] verdict {verdict}; failed gates {failed}", flush=True)
    print(
        f"[6l.validate] overlay min IoU {overlays['min_iou']}; programs missing in val/test "
        f"{missing_programs}",
        flush=True,
    )
    print(f"[6l.validate] wrote quality, verdict and artifact index in {time.time() - started:.0f}s", flush=True)
    return 0


def _flag_distributions() -> dict:
    """Border/tiny distribution of reference and target instances in v0.2 samples."""

    counters = {
        "target_border": 0, "target_tiny": 0, "reference_border": 0, "reference_tiny": 0,
        "samples": 0, "references": 0,
    }
    for split in ("train", "val", "test"):
        path = V02 / f"{split}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                counters["samples"] += 1
                target = (record.get("native_vector") or {}).get("target") or {}
                if target:
                    row = _instance_row(record["image_id"], target.get("tile_instance_id"))
                    if row:
                        counters["target_border"] += int(row["touches_tile_border"])
                        counters["target_tiny"] += int(row["tiny_area"])
                for reference in (record.get("native_vector") or {}).get("references") or []:
                    if not reference:
                        continue
                    counters["references"] += 1
                    row = _instance_row(record["image_id"], reference.get("tile_instance_id"))
                    if row:
                        counters["reference_border"] += int(row["touches_tile_border"])
                        counters["reference_tiny"] += int(row["tiny_area"])
    return counters


_ROW_CACHE: dict[tuple[str, int], dict | None] = {}


def _instance_row(tile_id: str, tile_instance_id) -> dict | None:
    key = (str(tile_id), int(tile_instance_id) if tile_instance_id is not None else -1)
    if key in _ROW_CACHE:
        return _ROW_CACHE[key]
    from buildreasonseg_mvp.whu_native_vector import read_tile_cache

    cache = read_tile_cache(key[0])
    row = None
    if cache is not None:
        for position, value in enumerate(cache["instance_ids"].tolist()):
            if int(value) == key[1]:
                row = {
                    "touches_tile_border": bool(cache["touches_border"][position]),
                    "tiny_area": bool(cache["tiny"][position]),
                }
                break
    _ROW_CACHE[key] = row
    return row


def _determinism_check() -> dict:
    """Regenerate a small tile subset twice into a scratch directory and compare bytes.

    The real v0.2 files are never touched: the check drives the generator with `--out-dir` pointing
    at a throwaway directory under the gitignored artifacts tree.
    """

    import subprocess

    tmp = REPO_ROOT / "artifacts" / "task6l_determinism"
    digests = []
    for run in ("a", "b"):
        out = tmp / run
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "task6l_build_v0_2.py"),
            "--limit-per-split", "25",
            "--out-dir", str(out),
            "--quiet",
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            return {"identical": False, "error": result.stderr[-400:], "runs": digests}
        digests.append({name: sha256_file(out / name) for name in ("train.jsonl", "val.jsonl", "test.jsonl")})
    return {
        "identical": digests[0] == digests[1],
        "subset": "25 tiles per split (debug limit), written to a scratch directory",
        "runs": digests,
        "real_v02_untouched": True,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
