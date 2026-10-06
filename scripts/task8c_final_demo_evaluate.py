"""Post-hoc Task 8C audit of frozen artifacts; zero model calls and no RC1 writes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "evaluation/task8c_final_demo_v1.json"
REVIEW_ROOT = REPO / "evaluation/task8c_final_demo_v1"
EXPECTED_ORDER = ["right", "left", "above", "below"]
GT_IDS = {"right": (4, 3), "left": (26, 24), "above": (4, 6), "below": (3, 2)}


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_runtime(evidence: dict) -> dict:
    frozen_id = evidence["frozen_runtime_identity"]
    if identity(Path(frozen_id["path"])) != frozen_id:
        raise ValueError("runtime JSON changed after freeze")
    runtime = json.loads(Path(frozen_id["path"]).read_text(encoding="utf-8"))
    if (runtime.get("formal_runner_invocation_count") != 1 or
            runtime.get("case_attempt_order") != EXPECTED_ORDER or
            not runtime.get("all_child_processes_exited") or not runtime.get("output_hashes_frozen") or
            len(runtime.get("cases", [])) != 4):
        raise ValueError("GT embargo: four exited processes and frozen outputs required")
    if runtime["formal_freeze_commit"] != evidence["formal_freeze_commit"]:
        raise ValueError("freeze commit differs")
    for row, locked in zip(runtime["cases"], evidence["candidate_lock"]):
        if any(row[k] != v for k, v in locked.items()):
            raise ValueError("candidate identity differs")
        if identity(Path(row["image"]))["sha256"] != row["image_sha256"]:
            raise ValueError("raster identity differs")
        for artifact in row["artifacts"] + [row["transcript_identity"]]:
            if identity(Path(artifact["path"])) != artifact:
                raise ValueError("output artifact changed after freeze")
        if row["run_root"]:
            actual_paths = {str(p.resolve()) for p in Path(row["run_root"]).rglob("*") if p.is_file()}
            if actual_paths != {a["path"] for a in row["artifacts"]}:
                raise ValueError("run-root file set changed after freeze")
    return runtime


def context_to_global(mask: np.ndarray, context: dict, shape: tuple[int, int]) -> np.ndarray:
    left, top = map(int, context["origin"])
    size = int(context["size"])
    if mask.shape != (size, size):
        raise ValueError("reference context shape differs")
    height, width = shape
    y0, x0, y1, x1 = max(0, top), max(0, left), min(height, top + size), min(width, left + size)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("context has no valid global extent")
    result = np.zeros(shape, dtype=bool)
    result[y0:y1, x0:x1] = mask[y0-top:y1-top, x0-left:x1-left]
    return result


def overlap(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    if prediction.shape != truth.shape:
        raise ValueError("mask alignment differs")
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    total = int(prediction.sum()) + int(truth.sum())
    return (intersection / union if union else 0.0, 2 * intersection / total if total else 0.0)


def best_overlap(prediction: np.ndarray, masks: dict[int, np.ndarray]) -> tuple[int | None, float]:
    best_id, best_iou = None, 0.0
    for instance_id in sorted(masks):
        value, _ = overlap(prediction, masks[instance_id])
        if value > best_iou:
            best_id, best_iou = instance_id, value
    return best_id, best_iou


def identity_audit(reference: np.ndarray, target: np.ndarray, masks: dict[int, np.ndarray],
                   reference_id: int, target_id: int) -> dict:
    ref_best, ref_best_iou = best_overlap(reference, masks)
    target_best, target_best_iou = best_overlap(target, masks)
    ref_iou, _ = overlap(reference, masks[reference_id])
    target_iou, target_dice = overlap(target, masks[target_id])
    ref_match = ref_best == reference_id and ref_best_iou > 0
    target_match = target_best == target_id and target_best_iou > 0
    if ref_match and target_match:
        chain = "CHAIN_IDENTITY_MATCH"
    elif ref_match:
        chain = "TARGET_IDENTITY_MISMATCH"
    elif target_match:
        chain = "REFERENCE_IDENTITY_MISMATCH"
    else:
        chain = "REFERENCE_AND_TARGET_IDENTITY_MISMATCH"
    return {"selected_reference_best_gt_instance_id": ref_best,
            "selected_reference_best_gt_iou": ref_best_iou,
            "selected_reference_iou_with_canonical_gt_reference": ref_iou,
            "reference_identity_best_overlap_match": ref_match,
            "target_iou": target_iou, "target_dice": target_dice,
            "predicted_target_area": int(target.sum()), "gt_target_area": int(masks[target_id].sum()),
            "predicted_target_best_gt_instance_id": target_best,
            "predicted_target_best_gt_iou": target_best_iou,
            "target_identity_best_overlap_match": target_match, "semantic_chain_status": chain}


def load_truth(row: dict) -> tuple[dict[int, np.ndarray], dict]:
    """Read existing native caches only. No regeneration or annotation writes."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from buildreasonseg_mvp.whu_native_vector import read_tile_cache, instance_rings, _rasterize_rings

    samples_path = REPO / "datasets/build_spatial_reason/v0.2/test.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line]
    selected = [r for r in samples if r["sample_id"] == row["sample_id"]]
    if len(selected) != 1:
        raise ValueError("frozen sample missing or duplicated")
    sample = selected[0]
    ref_id, target_id = GT_IDS[row["relation"]]
    if (sample["reference_component_ids"] != [ref_id] or sample["target_component_id"] != target_id or
            sample["query_type"] != row["expected_program"] or sample["split"] != "test" or
            sample["split_view"] != "scene_disjoint_v1"):
        raise ValueError("frozen sample/GT metadata conflict")
    tile_id = str(sample["image_id"])
    index_path = REPO / "datasets/whu_native_vector/v1.0/tiles/index.jsonl"
    tile_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    tiles = [r for r in tile_rows if str(r["tile_id"]) == tile_id]
    if len(tiles) != 1 or Path(tiles[0]["source_image_ref"]).stem != Path(row["image"]).stem:
        raise ValueError("tile/raster identity conflict")
    split_path = REPO / "datasets/whu_native_vector/v1.0/splits/scene_disjoint_v1.json"
    if tile_id not in json.loads(split_path.read_text(encoding="utf-8"))["test"]:
        raise ValueError("tile not in frozen test split")
    cache_path = REPO / "artifacts/whu_native_vector/instances" / (tile_id.replace("/", "_") + ".npz")
    cache = read_tile_cache(tile_id)
    if cache is None or cache["label_map"].shape != (512, 512):
        raise ValueError("native-vector cache missing/alignment conflict")
    # Verify the cache's label map against its native polygon rings, using the original rasterizer.
    reconstructed = np.zeros((512, 512), dtype=cache["label_map"].dtype)
    ids = [int(v) for v in cache["instance_ids"]]
    for instance_id in ids:
        reconstructed[_rasterize_rings(instance_rings(cache, instance_id), 512)] = instance_id
    if not np.array_equal(reconstructed, cache["label_map"]):
        raise ValueError("native-vector rings/label-map conflict")
    masks = {i: reconstructed == i for i in ids}
    if any(i not in masks or not masks[i].any() for i in (ref_id, target_id)):
        raise ValueError("canonical reference/target missing or empty")
    metadata = []
    for instance_id in (ref_id, target_id):
        pos = ids.index(instance_id)
        source_id = int(cache["source_feature_ids"][pos])
        frozen_meta = sample["native_vector"]["target"] if instance_id == target_id else sample["native_vector"]["references"][0]
        if frozen_meta != {"tile_instance_id": instance_id, "source_feature_id": source_id}:
            raise ValueError("native source-feature provenance conflict")
        metadata.append({"tile_instance_id": instance_id, "source_feature_id": source_id,
                         "mask_area": int(masks[instance_id].sum()),
                         "native_clipped_area": int(cache["clipped_area"][pos]),
                         "bbox": cache["bboxes"][pos].tolist()})
    provenance = {"tile_id": tile_id, "canonical_reference_id": ref_id, "canonical_target_id": target_id,
                  "instance_metadata": metadata, "all_gt_instance_count": len(masks),
                  "truth_definition": "Existing Task 6M canonical native instance label map; rings rasterized in original order.",
                  "truth_inputs": [identity(p) for p in (samples_path, index_path, split_path, cache_path)],
                  "native_rings_label_map_equal": True, "model_calls": 0}
    return masks, provenance


def read_binary(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.shape != shape or not set(np.unique(array)).issubset({0, 255}):
        raise ValueError("saved binary mask alignment/values conflict")
    return array > 0


def audit_case(row: dict, masks: dict[int, np.ndarray], provenance: dict) -> tuple[dict, np.ndarray | None]:
    audit = {**provenance, "semantic_chain_status": "NOT_EVALUABLE_MISSING_ARTIFACT"}
    reference = None
    if not row["language_expected_program_executed"]:
        audit["semantic_chain_status"] = "LANGUAGE_FAILED"
    elif row["runtime_status"] != "SUCCESS":
        audit["semantic_chain_status"] = "RUNTIME_FAILED"
    else:
        required = [row.get("mask_path"), row.get("overlay_path"),
                    str(Path(row["diagnostics_path"]) / "reference_context_mask.png") if row.get("diagnostics_path") else None]
        if all(p and Path(p).is_file() for p in required) and row.get("reasoning_context"):
            size = int(row["reasoning_context"]["size"])
            reference = context_to_global(read_binary(Path(required[2]), (size, size)),
                                          row["reasoning_context"], (512, 512))
            target = read_binary(Path(required[0]), (512, 512))
            if int(target.sum()) != row["mask_area"]:
                raise ValueError("saved mask/runtime mask_area conflict")
            audit.update(identity_audit(reference, target, masks,
                                        provenance["canonical_reference_id"], provenance["canonical_target_id"]))
    return audit, reference


def boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:] &= mask[:-1]
    interior[:-1] &= mask[1:]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    interior[0] = interior[-1] = False
    interior[:, 0] = interior[:, -1] = False
    return mask & ~interior


def review_image(row: dict, audit: dict, reference: np.ndarray | None,
                 masks: dict[int, np.ndarray]) -> Image.Image:
    rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
    ref_panel = rgb.copy()
    if reference is not None:
        ref_panel[boundary(reference)] = (255, 0, 255)
    elif row.get("diagnostics_path"):
        preview = Path(row["diagnostics_path"]) / "selected_reference.png"
        if preview.is_file():
            ref_panel = np.asarray(Image.open(preview).convert("RGB"))
    ref_panel[boundary(masks[audit["canonical_reference_id"]])] = (0, 255, 0)
    target_panel = rgb.copy()
    if row.get("overlay_path") and Path(row["overlay_path"]).is_file():
        target_panel = np.array(
            Image.open(row["overlay_path"]).convert("RGB"),
            copy=True,
        )
    target_panel[boundary(masks[audit["canonical_target_id"]])] = (0, 255, 255)
    canvas = Image.new("RGB", (1536, 760), "#15191f")
    for n, panel in enumerate((rgb, ref_panel, target_panel)):
        canvas.paste(Image.fromarray(panel), (512*n, 42))
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 19)
    draw = ImageDraw.Draw(canvas)
    for n, title in enumerate(("Original RGB", "Auto ref: magenta / GT ref: green", "Saved overlay / GT target: cyan")):
        draw.text((512*n+8, 10), title, font=font, fill="white")
    metrics = lambda k: f"{audit[k]:.6f}" if k in audit else "N/A"
    lines = [f"{row['relation'].upper()} | {row['sample_id']}",
             f"expected={row['expected_program']}  initial={row['initial_program']}  suggestion={row['suggested_program']}",
             f"language={row['language_status']}  runtime={row['runtime_status']}  exit={row['exit_code']}  auto ref ID={row['reference_id']}",
             f"GT ref={audit['canonical_reference_id']} / target={audit['canonical_target_id']}  ref IoU={metrics('selected_reference_iou_with_canonical_gt_reference')}  target IoU={metrics('target_iou')}  Dice={metrics('target_dice')}",
             f"Post-hoc: {audit['semantic_chain_status']} (identity match does not imply perfect mask quality)"]
    if row["runtime_status"] != "SUCCESS":
        lines.append(f"RUNTIME FAILED | {row['error_code']} | {str(row['error_reason']).replace(chr(10), ' ')[:115]}")
    for n, line in enumerate(lines):
        draw.text((10, 562+30*n), line, font=font,
                  fill="#ffb3a3" if "FAILED" in line else "white")
    return canvas


def aggregate(cases: list[dict]) -> dict:
    statuses = [r["gt_audit"]["semantic_chain_status"] for r in cases]
    successful = [r for r in cases if r["runtime_status"] == "SUCCESS"]
    evaluated = [r for r in successful if "target_iou" in r["gt_audit"]]
    return {"cases_attempted": len(cases),
            "language_expected_program_executed": sum(r["language_expected_program_executed"] for r in cases),
            "runtime_success": len(successful), "runtime_failed": len(cases)-len(successful),
            "chain_identity_match": statuses.count("CHAIN_IDENTITY_MATCH"),
            "reference_identity_mismatch": statuses.count("REFERENCE_IDENTITY_MISMATCH"),
            "target_identity_mismatch": statuses.count("TARGET_IDENTITY_MISMATCH"),
            "both_identity_mismatch": statuses.count("REFERENCE_AND_TARGET_IDENTITY_MISMATCH"),
            "language_failed": statuses.count("LANGUAGE_FAILED"),
            "not_evaluable": statuses.count("NOT_EVALUABLE_MISSING_ARTIFACT"),
            "mean_target_iou_over_runtime_success": float(np.mean([r["gt_audit"]["target_iou"] for r in evaluated])) if evaluated and len(evaluated) == len(successful) else None,
            "mean_target_dice_over_runtime_success": float(np.mean([r["gt_audit"]["target_dice"] for r in evaluated])) if evaluated and len(evaluated) == len(successful) else None}


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("evaluator accepts no overrides")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    runtime = verify_runtime(evidence)  # First action; no GT access before this succeeds.
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    cases, reviews, artifacts = [], [], []
    for row in runtime["cases"]:
        masks, provenance = load_truth(row)
        audit, reference = audit_case(row, masks, provenance)
        reviewed = review_image(row, audit, reference, masks)
        path = REVIEW_ROOT / f"{row['relation']}_review.png"
        if path.exists():
            raise ValueError("review already exists; preserve existing evidence")
        reviewed.save(path, format="PNG")
        artifacts.append(identity(path))
        reviews.append(reviewed)
        cases.append({**row, "gt_audit": audit, "review_path": str(path)})
    contact = Image.new("RGB", (3072, 1520))
    for n, reviewed in enumerate(reviews):
        contact.paste(reviewed, ((n % 2)*1536, (n // 2)*760))
    path = REVIEW_ROOT / "contact_sheet.png"
    if path.exists():
        raise ValueError("contact sheet already exists")
    contact.save(path, format="PNG")
    artifacts.append(identity(path))
    evidence.update({"status": "GT_AUDIT_COMPLETE", "cases": cases, "aggregate": aggregate(cases),
                     "gt_audit_started_after_runtime_freeze": True, "model_calls_during_gt_audit": 0,
                     "review_artifacts": artifacts})
    with EVIDENCE.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(evidence["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
