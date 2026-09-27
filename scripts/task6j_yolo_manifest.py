"""Write the J1 YOLO inference manifest: unique image ids -> absolute image paths.

Run in the mvp env; the inference itself runs read-only in the existing `yolo_sam_env`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from task6j_common import fixed_validation_material, paired_sample_lists  # noqa: E402

OUT_DIR = REPO_ROOT / "artifacts" / "task6j_yolo_proposals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

val_samples, pairs = fixed_validation_material()
a_samples, b_samples = paired_sample_lists(pairs)
unique = {}
for sample in val_samples + a_samples + b_samples:
    unique.setdefault(sample.image_id, str(data_mod.resolve_repo_path(sample.image_path)))

manifest = {
    "images": {image_id: path for image_id, path in sorted(unique.items())},
    "count": len(unique),
    "samples": len(val_samples) + len(a_samples) + len(b_samples),
}
path = OUT_DIR / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"[task6j:manifest] {len(unique)} images -> {path}")
