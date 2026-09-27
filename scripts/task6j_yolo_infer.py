"""Task 6J section 6-7: read-only YOLO proposal inference.

Runs INSIDE the existing `yolo_sam_env` (Python 3.10, ultralytics 8.x) — this script never
installs anything and never writes into the legacy project. Inputs: the manifest written by
`task6j_yolo_manifest.py`, the frozen model path, the gitignored output directory. For every
image it stores proposals (mask / bbox / centroid / area / confidence) as npz plus a provenance
JSON. Proposal geometry is derived from the model output ONLY; no GT enters this path.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the target env has opencv via ultralytics
    cv2 = None

import torch
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUT_DIR = REPO_ROOT / "artifacts" / "task6j_yolo_proposals"

MODEL_PATH = REPO_ROOT.parent / "WHU_Building_Segment" / "runs" / "segment" / "logs" / "whu_building_v1" / "weights" / "best.pt"
IMGSZ = 640
CONF = 0.25
IOU = 0.7
MAX_DET = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    started = time.time()
    manifest_path = OUT_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest missing; run task6j_yolo_manifest.py first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not MODEL_PATH.is_file():
        raise SystemExit(f"model missing: {MODEL_PATH}")
    model_hash = sha256_file(MODEL_PATH)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))
    model.to(device)

    provenance = {
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "python_version": sys.version,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
        "note": (
            "frozen YOLOv8m-seg-WHU baseline (runs/segment/logs/whu_building_v1/weights/best.pt, "
            "trained 100 epochs on the legacy WHU YOLO dataset). Invoked read-only; the legacy "
            "project and its environment are not modified."
        ),
    }

    results = {}
    for index, (image_id, image_path) in enumerate(sorted(manifest["images"].items()), start=1):
        predictions = model.predict(
            source=image_path,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            retina_masks=True,
            verbose=False,
        )
        result = predictions[0]
        origin_shape = tuple(result.orig_shape)
        boxes = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, conf, cls in zip(xyxy, confs, classes):
                boxes.append(
                    {
                        "bbox_xyxy": [float(v) for v in box],
                        "confidence": float(conf),
                        "class_id": int(cls),
                    }
                )
        masks = []
        if result.masks is not None:
            mask_data = result.masks.data.detach().cpu().numpy()  # [N, H, W] in orig coords
            if mask_data.shape[1:] != origin_shape:
                if cv2 is None:
                    raise SystemExit("opencv required to resize masks")
                resized = np.stack(
                    [
                        cv2.resize(m.astype(np.float32), (origin_shape[1], origin_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                        for m in mask_data
                    ]
                )
                mask_data = resized
            masks = [((m > 0.5).astype(bool)) for m in mask_data]

        np.savez_compressed(
            OUT_DIR / f"{image_id}.npz",
            masks=np.stack(masks) if masks else np.zeros((0, *origin_shape), dtype=bool),
            boxes=np.asarray([[b["bbox_xyxy"][0], b["bbox_xyxy"][1], b["bbox_xyxy"][2], b["bbox_xyxy"][3]] for b in boxes], dtype=np.float32).reshape(-1, 4),
            confidences=np.asarray([b["confidence"] for b in boxes], dtype=np.float32),
            classes=np.asarray([b["class_id"] for b in boxes], dtype=np.int64),
        )
        results[image_id] = {
            "image_path": image_path,
            "orig_shape": list(origin_shape),
            "proposal_count": len(boxes),
            "seconds": round(time.time() - started, 2),
        }
        if index % 20 == 0 or index == len(manifest["images"]):
            print(f"[task6j:yolo] {index}/{len(manifest['images'])} images", flush=True)

    payload = {
        "provenance": provenance,
        "images": results,
        "seconds": round(time.time() - started, 2),
    }
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[task6j:yolo] wrote {len(results)} proposal sets + provenance.json "
          f"(model sha256 {model_hash[:16]}...)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
