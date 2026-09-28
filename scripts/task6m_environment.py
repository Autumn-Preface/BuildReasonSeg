"""Task 6M sections 4-5: environment + provenance manifest.

Records the project-local conda environment, the exact released package versions, the GPU, the
pretrained-weight provenance and the license note. Pins exact versions; nothing is installed here.

    python scripts/task6m_environment.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "evaluation" / "task6m_environment_manifest.json"
ENV_PREFIX = REPO_ROOT / ".conda" / "buildreasonseg-proposal"


def package_versions() -> dict:
    import importlib.metadata as md

    names = (
        "ultralytics", "ultralytics-thop", "torch", "torchvision", "numpy", "opencv-python",
        "pillow", "scipy", "pyyaml", "matplotlib", "polars", "psutil", "requests", "transformers",
        "tokenizers", "safetensors", "accelerate",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except Exception:  # noqa: BLE001
            versions[name] = None
    return versions


def pip_report() -> dict:
    result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
    return {"pip": result.stdout.strip()}


def main() -> int:
    import torch
    import ultralytics

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    manifest.setdefault("_doc", "Task 6M environment and weight provenance.")
    manifest.setdefault("task", "6M")

    manifest["environment"] = {
        "conda_env_prefix": str(ENV_PREFIX),
        "conda_env_gitignored": True,
        "created_by": "conda create --clone .conda/buildreasonseg-mvp (Task 6A env clone) + pip install of the official released ultralytics package",
        "why_not_yolo_sam_env": (
            "the historical yolo_sam_env installs an editable Ultralytics checkout from a local "
            "modified tree; Task 6M section 5 forbids using it for new training"
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "ultralytics_import_path": str(Path(ultralytics.__file__).parent),
        "ultralytics_official_release": "pypi:ultralytics==8.4.164",
        **pip_report(),
    }
    manifest["hardware"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if torch.cuda.is_available() else None,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    manifest["authorized_downloads_used"] = [
        "pypi:ultralytics==8.4.164 (+ declared dependencies)",
        "github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
        "github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt (smoke only)",
    ]
    manifest["unauthorized_downloads"] = []
    manifest["license_note"] = manifest.get(
        "license_note",
        (
            "Ultralytics open-source stack and pre-trained models are distributed under AGPL-3.0, with "
            "enterprise licensing available for proprietary/commercial use. The repository LICENSE is "
            "NOT changed by Task 6M and no commercial-licensing claim is made."
        ),
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"[6m.env] ultralytics {ultralytics.__version__} | python {sys.version.split()[0]} | "
          f"gpu {manifest['hardware']['gpu']} | wrote {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
