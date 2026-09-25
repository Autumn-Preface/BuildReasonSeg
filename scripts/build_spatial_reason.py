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


def sha256_source_file(path: Path) -> str:
    """SHA256 of a *source* file after CRLF -> LF normalisation.

    Provenance hashes must be independent of ``core.autocrlf``: on Windows git
    may materialise a committed LF blob as CRLF in the working tree, which changes
    the raw byte hash without changing the code. Normalising makes a working-tree
    hash comparable with the Git blob hash used by
    ``scripts/split_manifest_provenance.py``.
    """

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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

    # Level-2 decomposition (Task 5B section 10).
    #
    # The v0.1 counter omitted the `level == 2` guard, so
    # `endswith("_to_nearest")` also matched every Level-3 query type
    # (`largest_to_right_of_to_nearest`) and reported 8,700 instead of 4,707.
    # Both halves are now explicitly restricted to level 2.
    level2_nearest = sum(
        1 for s in samples
        if s["level"] == 2 and s["query_type"].endswith("_to_nearest")
    )
    level2_direction = sum(
        1 for s in samples
        if s["level"] == 2 and not s["query_type"].endswith("_to_nearest")
    )

    # Invariant: the two Level-2 subtypes must partition Level 2 exactly.
    level2_total = int(by_level.get(2, 0))
    if level2_nearest + level2_direction != level2_total:
        raise AssertionError(
            "Level-2 invariant violated: "
            f"reference_to_nearest({level2_nearest}) + "
            f"reference_to_direction({level2_direction}) != by_level['2']({level2_total})"
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
            "total_level2": level2_total,
            "invariant_holds": level2_nearest + level2_direction == level2_total,
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


def _git_head() -> str | None:
    """Best-effort current commit hash; ``None`` when unavailable.

    Provenance only: a failure here never blocks generation.
    """

    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


def source_provenance(paths: list[Path], commit: str | None = None) -> dict:
    """Generation-time vs current-source provenance (Task 5.5 section 3.1).

    The two are the SAME at generation time, and that is what is recorded here.
    They diverge later, when maintenance edits touch generation-relevant files;
    ``scripts/split_manifest_provenance.py`` is the housekeeping tool that
    reconstructs the true generation-time bytes from Git history and must never
    silently refresh them.
    """

    block = {"commit": commit or _git_head() or "UNCOMMITTED"}
    block["file_sha256"] = {_relpath(path): sha256_source_file(path) for path in paths}
    block["hash_definition"] = (
        "SHA256 after line-ending normalisation (CRLF -> LF), so the value is independent "
        "of core.autocrlf and equals the committed Git blob hash for a clean checkout."
    )
    return block


def build_manifest(
    stats: dict,
    relation_config: T.RelationConfig,
    relation_config_path: Path,
    generator_paths: list[Path],
    gen_config_path: Path,
    gen_config: dict,
    counter: A.DiscardCounter,
    dataset_meta: dict,
) -> dict:
    """Dataset-level manifest with provenance and known limitations."""

    generation_commit = _git_head()
    #: Recorded once and reused verbatim, so the two blocks can never disagree
    #: about the code that produced this run.
    generation_block = source_provenance(generator_paths, generation_commit)

    return {
        "dataset_name": "BuildSpatialReason",
        "dataset_version": dataset_meta["version"],
        "project_name": "BuildReasonSeg",
        "source_dataset": "WHU Building Dataset",
        "source_subset": "Satellite dataset II (East Asia)",
        "source_component_representation_version": dataset_meta.get(
            "component_representation_version", "v1.0"
        ),
        "semantic_visibility_policy_version": dataset_meta.get(
            "semantic_visibility_policy_version", A.SP.SEMANTIC_VISIBILITY_POLICY_VERSION
        ),
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
        # Task 5.5 section 3.1: generation-time and current-source provenance are
        # separate records. They are identical at generation time; later
        # maintenance edits move only `current_source`.
        "generation_source": generation_block,
        "current_source": {
            "commit": generation_block["commit"],
            "file_sha256": dict(generation_block["file_sha256"]),
        },
        "provenance_note": (
            "generation_source describes the code that produced THIS run and must never be "
            "refreshed afterwards. current_source tracks the repository state and may move as "
            "maintenance edits land. scripts/split_manifest_provenance.py reconstructs the "
            "generation-time bytes from Git history and is enforced by "
            "scripts/check_artifact_consistency.py."
        ),
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
                    "inside are structurally empty and are not part of this dataset version."
                ),
            },
            {
                "id": "KL-06",
                "limitation": "language is template-generated rather than human-authored",
                "detail": (
                    "Instructions are produced deterministically from templates. Wording "
                    "diversity is limited by construction; each sample records an explicit "
                    "template_id so template usage is measurable rather than implicit."
                ),
            },
            {
                "id": "KL-07",
                "limitation": "source imagery is WHU, but the schema is not WHU-specific",
                "detail": (
                    "This version uses the WHU Building Dataset 'Satellite dataset II "
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

    # Provenance set (Task 5C section 9). Every file that can change what the
    # generated samples say or which target they select must be listed, so the
    # manifest can prove which code produced the records. `semantic_policy.py`
    # was added in Task 5C: it decides the semantic universe for every size,
    # nearest and direction query, so it is generation-critical.
    generator_paths = [
        _REPO_ROOT / "spatial_reasoning" / "annotator.py",
        _REPO_ROOT / "spatial_reasoning" / "semantic_policy.py",
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
        dataset_meta,
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
