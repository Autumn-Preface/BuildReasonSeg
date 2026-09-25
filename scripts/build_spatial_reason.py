#!/usr/bin/env python
"""CLI: generate the BuildSpatialReason-v0.1 dataset.

    python scripts/build_spatial_reason.py [--splits train val test]
                                           [--limit-per-split N]
                                           [--out-dir PATH] [--quiet]

Writes::

    datasets/build_spatial_reason/v0.1/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── manifest.json
    └── statistics.json

Only repository-relative references are stored. Source images, component maps
and polygons are NOT copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:  # Unix only
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))

import numpy as np  # noqa: E402

import annotator as A  # noqa: E402
import geometry as G  # noqa: E402
import thresholds as T  # noqa: E402

GEN_CONFIG_PATH = _REPO_ROOT / "configs" / "build_spatial_reason_v0.1.yaml"
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_memory_mb() -> float:
    """Best-effort peak resident memory in MB; 0.0 when unavailable.

    ``resource`` is Unix-only, so on Windows the value is read from the Win32
    process counters via ctypes. Failure is not fatal: peak memory is a
    convenience metric, not part of the dataset.
    """

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return usage / 1024.0 if platform.system() != "Darwin" else usage / (1024.0 * 1024.0)

    try:  # pragma: no cover - Windows path
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def load_gen_config(path: Path = GEN_CONFIG_PATH) -> dict:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _relpath(path: Path) -> str:
    """Repository-relative POSIX path; never absolute."""

    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def generate_split(
    split: str,
    dataset_root: Path,
    relation_config: T.RelationConfig,
    gen_config: dict,
    dataset_meta: dict,
    limit: int | None,
    counter: A.DiscardCounter,
    seen_keys: set[str],
    quiet: bool,
) -> list[dict]:
    """Generate all samples for one split, preserving the source image split."""

    samples: list[dict] = []
    n_images = 0
    started = time.time()

    for image_record in G.iter_metadata(dataset_root, split):
        if limit is not None and n_images >= limit:
            break
        image = G.image_geometry_from_record(image_record)
        samples.extend(
            A.generate_for_image(
                image, image_record, relation_config, gen_config, dataset_meta, counter, seen_keys
            )
        )
        n_images += 1
        if not quiet and n_images % 500 == 0:
            print(f"  [{split}] {n_images} images, {len(samples)} samples...", flush=True)

    if not quiet:
        print(
            f"  [{split}] done: {n_images} images, {len(samples)} samples "
            f"in {time.time() - started:.1f}s"
        )
    return samples


def summarise(samples: list[dict]) -> dict:
    """Aggregate counts used by manifest.json and statistics.json."""

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
    nontrivial = len(level3) - trivial

    level2_nearest = sum(1 for s in samples if s["query_type"].endswith("_to_nearest"))
    level2_direction = sum(
        1 for s in samples if s["level"] == 2 and not s["query_type"].endswith("_to_nearest")
    )

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
        },
        "level3": {
            "trivial": trivial,
            "nontrivial": nontrivial,
        },
        "images_with_samples": len(per_image),
        "samples_per_image": {
            "min": float(counts.min()),
            "median": float(np.median(counts)),
            "mean": float(counts.mean()),
            "p90": float(np.percentile(counts, 90)),
            "max": float(counts.max()),
        },
        "unique_images_by_split": {
            split: len({s["image_id"] for s in samples if s["split"] == split})
            for split in SPLITS
        },
    }


def build_manifest(
    stats: dict,
    relation_config: T.RelationConfig,
    relation_config_path: Path,
    generator_paths: list[Path],
    gen_config_path: Path,
    gen_config: dict,
    counter: A.DiscardCounter,
) -> dict:
    """Dataset-level manifest with provenance and known limitations."""

    return {
        "dataset_name": "BuildSpatialReason",
        "dataset_version": "v0.1",
        "project_name": "BuildReasonSeg",
        "source_dataset": "WHU Building Dataset",
        "source_subset": "Satellite dataset II (East Asia)",
        "source_component_representation_version": "v1.0",
        "source_component_metadata": "datasets/whu/metadata/<split>.jsonl",
        "source_component_manifest": "datasets/whu/component_manifest.json",
        "relation_config": _relpath(relation_config_path),
        "relation_config_version": A.RELATION_CONFIG_VERSION,
        "relation_config_sha256": sha256_file(relation_config_path),
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
        "generator_config": _relpath(gen_config_path),
        "generator_config_sha256": sha256_file(gen_config_path),
        "generator_file_sha256": {
            _relpath(p): sha256_file(p) for p in generator_paths
        },
        "generation_seed": gen_config["generator"]["seed"],
        "template_version": gen_config["generator"]["template_version"],
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
                "limitation": "source annotation represents building connected components",
                "detail": (
                    "A 'building region' in an instruction maps to a building CONNECTED "
                    "COMPONENT in the source annotation, not to a verified real-world "
                    "physical building."
                ),
            },
            {
                "id": "KL-02",
                "limitation": "touching physical buildings may already be merged",
                "detail": (
                    "The source raster was a binary semantic mask, so mutually touching "
                    "buildings became a single component before this dataset existed."
                ),
            },
            {
                "id": "KL-03",
                "limitation": "border-truncated components are unreliable for some relations",
                "detail": (
                    "31.72% of components touch the tile edge. They remain eligible for "
                    "tile-relative extremes but are excluded from largest/smallest/nearest."
                ),
            },
            {
                "id": "KL-04",
                "limitation": "no building-function semantics",
                "detail": "No annotation of use, type, or function exists in the source data.",
            },
            {
                "id": "KL-05",
                "limitation": "no overlap / containment relations",
                "detail": (
                    "Components are disjoint by construction, so overlap, contain and "
                    "inside are structurally empty and are not part of v0.1."
                ),
            },
            {
                "id": "KL-06",
                "limitation": "language is template-generated rather than human-authored",
                "detail": (
                    "Instructions are produced deterministically from templates. Wording "
                    "diversity is limited by construction; template selection is recorded "
                    "implicitly through the sample semantic key."
                ),
            },
            {
                "id": "KL-07",
                "limitation": "v0.1 source imagery is WHU, but the schema is not WHU-specific",
                "detail": (
                    "Current v0.1 uses the WHU Building Dataset 'Satellite dataset II "
                    "(East Asia)' subset. The BuildSpatialReason schema, relation "
                    "representation and reasoning program do not assume WHU."
                ),
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--limit-per-split", type=int, default=None, help="debug subset")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gen-config", type=Path, default=GEN_CONFIG_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    gen_config = load_gen_config(args.gen_config)
    dataset_meta = A.dataset_meta_from_config(gen_config)
    relation_config = T.load_config(gen_config["relations"]["config"])

    # Frozen-threshold guard: the generator must not silently disagree with the
    # frozen relation config it claims to use.
    frozen = gen_config["relations"]["frozen"]
    if abs(relation_config.size_rank.ratio_margin - frozen["ratio_margin"]) > 1e-9:
        print(
            f"error: generator config claims ratio_margin={frozen['ratio_margin']} but the "
            f"frozen relation config has {relation_config.size_rank.ratio_margin}",
            file=sys.stderr,
        )
        return 2
    if relation_config.direction.active_candidate != frozen["direction_preset"]:
        print("error: direction preset mismatch between generator and relation config", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (_REPO_ROOT / gen_config["output"]["root"] / gen_config["output"]["version_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = _REPO_ROOT / "datasets" / "whu"
    if not (dataset_root / "metadata" / "train.jsonl").is_file():
        print(f"error: component dataset not found under {dataset_root}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"generator config : {_relpath(args.gen_config)}")
        print(f"relation config  : {_relpath(Path(relation_config.source_path))}")
        print(f"output dir       : {_relpath(out_dir)}")

    counter = A.DiscardCounter()
    seen_keys: set[str] = set()
    all_samples: list[dict] = []
    started = time.time()
    peak_mb = 0.0

    for split in args.splits:
        samples = generate_split(
            split,
            dataset_root,
            relation_config,
            gen_config,
            dataset_meta,
            args.limit_per_split,
            counter,
            seen_keys,
            args.quiet,
        )
        # Write each split immediately so partial runs are still inspectable.
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        all_samples.extend(samples)

        peak_mb = max(peak_mb, _peak_memory_mb())

    elapsed = time.time() - started
    stats = summarise(all_samples)
    stats["runtime_seconds"] = round(elapsed, 2)
    stats["peak_memory_mb"] = round(peak_mb, 1)
    stats["generation_seed"] = gen_config["generator"]["seed"]

    generator_paths = [
        _REPO_ROOT / "spatial_reasoning" / "annotator.py",
        _REPO_ROOT / "spatial_reasoning" / "templates.py",
        _REPO_ROOT / "spatial_reasoning" / "relations.py",
        _REPO_ROOT / "spatial_reasoning" / "geometry.py",
        _REPO_ROOT / "spatial_reasoning" / "thresholds.py",
        _REPO_ROOT / "spatial_reasoning" / "component_quality.py",
        _REPO_ROOT / "scripts" / "build_spatial_reason.py",
    ]
    manifest = build_manifest(
        stats,
        relation_config,
        Path(relation_config.source_path),
        generator_paths,
        args.gen_config,
        gen_config,
        counter,
    )
    manifest["statistics_file"] = _relpath(out_dir / "statistics.json")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not args.quiet:
        print()
        print(f"total samples    : {stats['total_samples']}")
        print(f"by split         : {stats['by_split']}")
        print(f"by level         : {stats['by_level']}")
        print(f"level2           : {stats['level2']}")
        print(f"level3           : {stats['level3']}")
        print(f"samples per image: {stats['samples_per_image']}")
        print(f"discards         : {counter.as_dict()}")
        print(f"runtime          : {elapsed:.1f}s  peak RSS {peak_mb:.0f} MB")
        print(f"output           : {_relpath(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
