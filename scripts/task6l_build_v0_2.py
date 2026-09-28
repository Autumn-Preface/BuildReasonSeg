"""Task 6L Part D: generate BuildSpatialReason v0.2 from native-vector instances.

Reuses the frozen v0.1.1 generator code (`spatial_reasoning.annotator` + templates + relations +
thresholds) UNCHANGED; only the annotation truth, the candidate source and the split view change.
Each generated record is enriched with the native-vector provenance the task file requires
(section 15): tile id, source-feature / tile-instance metadata for reference and target, split view,
dataset version, relation config version and visibility policy version.

    python scripts/task6l_build_v0_2.py [--split-view scene_disjoint_v1] [--limit-per-split N]
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
import yaml

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

GEN_CONFIG_PATH = REPO_ROOT / "configs" / "build_spatial_reason_v0.2.yaml"
OUT_DIR = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
EVAL = REPO_ROOT / "evaluation"
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gen_config(path: Path = GEN_CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def enrich_record(record: dict, dataset: NativeVectorDataset, split_view: str) -> dict:
    """Add the native-vector provenance block (metadata only; never model input)."""

    tile_id = str(record["image_id"])
    instances = {row["tile_instance_id"]: row for row in dataset.list_instances(tile_id)}

    def describe(instance_id):
        row = instances.get(int(instance_id))
        if row is None:
            return None
        return {
            "tile_instance_id": int(row["tile_instance_id"]),
            "source_feature_id": int(row["source_feature_id"]),
        }

    record["source_component_representation_version"] = "whu-native-vector-v1.0"
    record["split_view"] = split_view
    record["native_vector"] = {
        "dataset_name": dataset.dataset_name,
        "dataset_version": dataset.dataset_version,
        "tile_id": tile_id,
        "target": describe(record["target_component_id"]),
        "references": [describe(value) for value in record.get("reference_component_ids", [])],
        # candidate/distractor ids are already in the record's own id lists; only the provenance
        # mapping for the reference and target instances is expanded here, to keep the JSONL compact.
        "candidate_count": len(record.get("candidate_component_ids", [])),
        "distractor_count": len(record.get("distractor_component_ids", [])),
        "target_mask": record.get("target_mask"),
    }
    record["target_geometry_ref"] = {
        "dataset": "WHU-EA-NativeVector",
        "dataset_version": dataset.dataset_version,
        "tile_id": tile_id,
        "tile_instance_id": int(record["target_component_id"]),
        "source_feature_id": (describe(record["target_component_id"]) or {}).get("source_feature_id"),
        "geometry_cache": f"artifacts/whu_native_vector/instances/{tile_id}.npz",
    }
    return record


def summarise(samples: list[dict], per_split_images: dict) -> dict:
    by_split = Counter(s["split"] for s in samples)
    by_level = Counter(s["level"] for s in samples)
    by_query_type = Counter(s["query_type"] for s in samples)
    by_split_level = defaultdict(Counter)
    by_split_query_type = defaultdict(Counter)
    for sample in samples:
        by_split_level[sample["split"]][sample["level"]] += 1
        by_split_query_type[sample["split"]][sample["query_type"]] += 1
    level3 = [s for s in samples if s["level"] == 3]
    trivial = sum(1 for s in level3 if s["trivial_selection"])
    level2_nearest = sum(1 for s in samples if s["level"] == 2 and s["query_type"].endswith("_to_nearest"))
    level2_direction = sum(1 for s in samples if s["level"] == 2 and not s["query_type"].endswith("_to_nearest"))
    level2_total = int(by_level.get(2, 0))
    if level2_nearest + level2_direction != level2_total:
        raise AssertionError("level-2 invariant violated")
    per_image = Counter((s["split"], s["image_id"]) for s in samples)
    counts = np.array(list(per_image.values()), dtype=np.float64) if per_image else np.array([0.0])
    return {
        "total_samples": len(samples),
        "by_split": {k: int(v) for k, v in sorted(by_split.items())},
        "by_level": {str(k): int(v) for k, v in sorted(by_level.items())},
        "by_query_type": {k: int(v) for k, v in sorted(by_query_type.items())},
        "by_split_level": {
            split: {str(k): int(v) for k, v in sorted(counter.items())}
            for split, counter in sorted(by_split_level.items())
        },
        "by_split_query_type": {
            split: {k: int(v) for k, v in sorted(counter.items())}
            for split, counter in sorted(by_split_query_type.items())
        },
        "level2": {
            "reference_to_nearest": level2_nearest,
            "reference_to_direction": level2_direction,
            "total_level2": level2_total,
            "invariant_holds": level2_nearest + level2_direction == level2_total,
        },
        "level3": {"trivial": trivial, "nontrivial": len(level3) - trivial},
        "images_with_samples": len(per_image),
        "tiles_considered_by_split": per_split_images,
        "samples_per_image": {
            "min": float(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "p90": float(np.percentile(counts, 90)),
            "max": float(counts.max()),
        },
        "unique_images_by_split": {
            split: len({s["image_id"] for s in samples if s["split"] == split}) for split in SPLITS
        },
        "program_support": {
            split: sorted({s["query_type"] for s in samples if s["split"] == split}) for split in SPLITS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-view", default="scene_disjoint_v1")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    gen_config = load_gen_config()
    dataset_meta = A.dataset_meta_from_config(gen_config)
    relation_config = T.load_config(gen_config["relations"]["config"])
    frozen = gen_config["relations"]["frozen"]
    if abs(relation_config.size_rank.ratio_margin - frozen["ratio_margin"]) > 1e-9:
        print("error: frozen ratio_margin mismatch", file=sys.stderr)
        return 2

    dataset = NativeVectorDataset(args.dataset_root)
    view = REASONING_VIEW_ROOT / args.split_view
    if not (view / "metadata" / "train.jsonl").is_file():
        print(f"error: reasoning view not found under {view}", file=sys.stderr)
        return 2

    # path correctness only: the frozen annotator writes this into every record. `.format(split=...)`
    # is called by the annotator, so the template must contain exactly one `{split}` placeholder.
    A.IMAGE_METADATA_TEMPLATE = (
        f"artifacts/whu_native_vector/reasoning_view/{args.split_view}/metadata/{{split}}.jsonl"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = A.DiscardCounter()
    seen_keys: set[str] = set()
    all_samples: list[dict] = []
    per_split_images: dict[str, int] = {}

    for split in SPLITS:
        samples: list[dict] = []
        tiles = 0
        for record in _iter_metadata(view / "metadata" / f"{split}.jsonl"):
            if args.limit_per_split is not None and tiles >= args.limit_per_split:
                break
            image = A.G.image_geometry_from_record(record)
            # the frozen annotator resolves the label map through the record's component_map path
            for sample in A.generate_for_image(
                image, record, relation_config, gen_config, dataset_meta, counter, seen_keys
            ):
                samples.append(enrich_record(sample, dataset, args.split_view))
            tiles += 1
            if not args.quiet and tiles % 1000 == 0:
                print(f"[6l.v02] {split}: {tiles} tiles, {len(samples)} samples "
                      f"({time.time() - started:.0f}s)", flush=True)
        per_split_images[split] = tiles
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        all_samples.extend(samples)
        print(f"[6l.v02] {split}: {tiles} tiles -> {len(samples)} samples", flush=True)

    stats = summarise(all_samples, per_split_images)
    stats["runtime_seconds"] = round(time.time() - started, 2)
    stats["generation_seed"] = gen_config["generator"]["seed"]
    stats["discard_reasons"] = counter.as_dict()

    generator_paths = [
        REPO_ROOT / "spatial_reasoning" / "annotator.py",
        REPO_ROOT / "spatial_reasoning" / "semantic_policy.py",
        REPO_ROOT / "spatial_reasoning" / "templates.py",
        REPO_ROOT / "spatial_reasoning" / "relations.py",
        REPO_ROOT / "spatial_reasoning" / "geometry.py",
        REPO_ROOT / "spatial_reasoning" / "thresholds.py",
        REPO_ROOT / "spatial_reasoning" / "component_quality.py",
        REPO_ROOT / "scripts" / "task6l_build_v0_2.py",
        REPO_ROOT / "buildreasonseg_mvp" / "native_vector_adapter.py",
    ]
    manifest = {
        "dataset_name": "BuildSpatialReason",
        "dataset_version": dataset_meta["version"],
        "project_name": "BuildReasonSeg",
        "source_dataset": dataset_meta["source_dataset"],
        "source_subset": dataset_meta["source_subset"],
        "source_component_representation_version": "whu-native-vector-v1.0",
        "source_instances": gen_config["source_instances"],
        "split_view": args.split_view,
        "semantic_visibility_policy_version": gen_config["dataset"]["semantic_visibility_policy_version"],
        "relation_config": gen_config["relations"]["config"],
        "relation_config_version": gen_config["relations"]["config_version"],
        "relation_config_sha256": sha256_file(REPO_ROOT / gen_config["relations"]["config"]),
        "frozen_relation_settings": {
            "ratio_margin": relation_config.size_rank.ratio_margin,
            "direction_preset": relation_config.direction.active_candidate,
            "direction_alpha": relation_config.direction.alpha,
            "direction_tau": relation_config.direction.tau,
            "nearest_anchor_non_border": relation_config.eligibility_for("nearest").reject_touches_image_border_anchor,
            "nearest_target_non_border": relation_config.eligibility_for("nearest").reject_touches_image_border,
            "ambiguous_policy": "discard",
        },
        "generator_version": A.GENERATOR_VERSION,
        "generator_code_version": gen_config["generator"]["code_version"],
        "generator_config": "configs/build_spatial_reason_v0.2.yaml",
        "generator_config_sha256": sha256_file(GEN_CONFIG_PATH),
        "generation_seed": gen_config["generator"]["seed"],
        "template_version": gen_config["generator"]["template_version"],
        "generation_source": {
            "file_sha256": {str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
                            for path in generator_paths},
            "hash_definition": "SHA256 of the working-tree file content (LF-normalised by Git on commit)",
        },
        "program_vocabulary_unchanged": True,
        "program_count": 20,
        "sample_counts": {
            "total": stats["total_samples"],
            "by_split": stats["by_split"],
            "by_level": stats["by_level"],
            "by_query_type": stats["by_query_type"],
            "level2": stats["level2"],
            "level3": stats["level3"],
        },
        "discard_reasons": counter.as_dict(),
        "known_limitations": [
            {
                "id": "KL-01",
                "limitation": "instance truth is a manually delineated vector map",
                "detail": (
                    "A 'building region' now maps to a native EA.shp polygon. This is materially "
                    "better than a semantic connected component, but it is still an annotation, not "
                    "a verified physical building inventory."
                ),
            },
            {
                "id": "KL-02",
                "limitation": "annotation migration changes some relation answers",
                "detail": (
                    "Task 6K.1 measured a 6.96% weighted VECTOR-vs-PSEUDO target change on the "
                    "historical tiles; v0.2 is therefore not answer-compatible with v0.1.1, and "
                    "Task 6L section 18 quantifies the delta."
                ),
            },
            {
                "id": "KL-03",
                "limitation": "scene-disjoint split separates scenes, not cities",
                "detail": (
                    "train1/train2/test are three whole rasters of the same East-Asia acquisition. "
                    "This supports scene separation; it does NOT support cross-city or broad "
                    "geographic generalization claims."
                ),
            },
            {
                "id": "KL-04",
                "limitation": "no building-function semantics",
                "detail": "No annotation of use, type, or function exists in the source data.",
            },
            {
                "id": "KL-05",
                "limitation": "border-truncated instances remain",
                "detail": (
                    "Native instances clipped by a tile edge keep their clipped geometry and a "
                    "visible_fraction; eligibility still follows the frozen relation rules."
                ),
            },
            {
                "id": "KL-06",
                "limitation": "language is template-generated rather than human-authored",
                "detail": (
                    "Instructions come from the same deterministic v0.1 template families as v0.1.1 "
                    "so the two versions stay comparable."
                ),
            },
            {
                "id": "KL-07",
                "limitation": "DBF attributes of the source archive are unusable",
                "detail": (
                    "Task 6K.1 proved the shipped DBF is degenerate; identity is "
                    "(EA.shp SHA256, source_feature_id), recorded in the canonical manifest."
                ),
            },
        ],
        "statistics_file": "datasets/build_spatial_reason/v0.2/statistics.json",
    }
    write_json(out_dir / "statistics.json", stats)
    write_json(out_dir / "manifest.json", manifest)
    print(
        f"[6l.v02] total {stats['total_samples']} samples; by split {stats['by_split']}; "
        f"by level {stats['by_level']}",
        flush=True,
    )
    print(f"[6l.v02] wrote v0.2 in {time.time() - started:.0f}s", flush=True)
    return 0


def _iter_metadata(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
