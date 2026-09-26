#!/usr/bin/env python
"""Task 6A: download exactly the two authorized model assets.

    python scripts/task6a_download.py [--verify-only]

Downloads:
  * `Qwen/Qwen3-VL-2B-Instruct`            -> local_cache/huggingface/
  * `facebook/sam2.1-hiera-base-plus`      -> local_cache/models/

Every artefact lands inside the repository under `local_cache/`, which is
gitignored. No weight file is ever staged. Nothing else is downloaded: Task 6A
explicitly authorizes the 2B Qwen model and the Base+ SAM2.1 checkpoint only.

Writes `evaluation/task6a_environment.json` with the measured on-disk sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE = REPO_ROOT / "local_cache"
HF_CACHE = LOCAL_CACHE / "huggingface"
MODEL_DIR = LOCAL_CACHE / "models"
ENVIRONMENT_JSON = REPO_ROOT / "evaluation" / "task6a_environment.json"

QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SAM2_REPO_ID = "facebook/sam2.1-hiera-base-plus"
SAM2_FILENAME = "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_SOURCE_DIR = LOCAL_CACHE / "sam2_source"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _configure_cache() -> None:
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def _fix_local_tls() -> dict:
    """Report TLS trust state. Task 6B removed the automatic certifi mutation.

    The Task 6A workaround (appending the Windows ROOT/CA stores to certifi) is no
    longer applied at runtime. The standard bundle is used as-is; if a local
    accelerator is ever reinstalled, `buildreasonseg_mvp.local_env` retains the old
    merge as an explicit `confirm=True` opt-in.
    """

    sys.path.insert(0, str(REPO_ROOT))
    from buildreasonseg_mvp.local_env import certifi_status

    return certifi_status()


def download_qwen() -> dict:
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(
        repo_id=QWEN_MODEL_ID,
        cache_dir=str(HF_CACHE / "hub"),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.py",
            "tokenizer*",
            "preprocessor*",
            "chat_template*",
        ],
        max_workers=4,
    )
    target = Path(path)
    weights = sorted(p.stat().st_size for p in target.glob("*.safetensors"))
    return {
        "model_id": QWEN_MODEL_ID,
        "snapshot_path": str(target),
        "snapshot_path_relative": str(target.relative_to(REPO_ROOT)) if target.is_relative_to(REPO_ROOT) else str(target),
        "total_bytes": _dir_size(target),
        "weight_bytes": sum(weights),
        "weight_files": [p.name for p in sorted(target.glob("*.safetensors"))],
        "download_seconds": round(time.time() - started, 1),
    }


def _urlretrieve(url: str, target: Path) -> int:
    """Plain-urllib download with progress-free streaming.

    Used as a fallback because the local accelerator can strip the
    `X-Repo-Commit` header that `huggingface_hub` requires on the `resolve/`
    redirect, even though the bytes themselves are served correctly.
    """

    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "BuildReasonSeg-Task6A/1.0"})
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
    return written


def download_sam2() -> dict:
    """Fetch the Base+ checkpoint, preferring the Hub API and falling back to urllib."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / SAM2_FILENAME
    started = time.time()
    method = "hf_hub_download"

    if not target.exists():
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=SAM2_REPO_ID,
                filename=SAM2_FILENAME,
                cache_dir=str(HF_CACHE / "hub"),
                local_dir=str(MODEL_DIR),
            )
            target = Path(path)
        except Exception as exc:  # noqa: BLE001
            method = f"urllib_fallback (hf_hub_download failed: {type(exc).__name__})"
            url = f"https://huggingface.co/{SAM2_REPO_ID}/resolve/main/{SAM2_FILENAME}"
            written = _urlretrieve(url, target)
            if written < 1_000_000:
                raise RuntimeError(f"downloaded SAM2 checkpoint looks truncated: {written} bytes")

    return {
        "repo_id": SAM2_REPO_ID,
        "filename": SAM2_FILENAME,
        "path": str(target),
        "path_relative": str(target.relative_to(REPO_ROOT)),
        "bytes": target.stat().st_size,
        "config_name": SAM2_CONFIG_NAME,
        "download_method": method,
        "download_seconds": round(time.time() - started, 1),
    }


def package_versions() -> dict:
    import importlib.metadata as metadata

    names = [
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "tokenizers",
        "qwen-vl-utils",
        "numpy",
        "pillow",
        "opencv-python",
        "pytest",
        "hydra-core",
        "iopath",
        "SAM-2",
        "tqdm",
        "PyYAML",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            out[name] = "NOT_INSTALLED"
    return out


def sam2_source_revision() -> str:
    import subprocess

    if not SAM2_SOURCE_DIR.is_dir():
        return ""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SAM2_SOURCE_DIR, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="do not download; only measure")
    args = parser.parse_args(argv)

    _configure_cache()
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tls_report = _fix_local_tls()

    report: dict = {
        "_doc": "Task 6A section 4.4/4.5/17: dedicated environment and downloaded assets.",
        "task": "6A",
        "repo_root": str(REPO_ROOT),
        "local_tls_trust": tls_report,
        "cache_policy": {
            "HF_HOME": str(HF_CACHE),
            "HF_HUB_CACHE": str(HF_CACHE / "hub"),
            "models_dir": str(MODEL_DIR),
            "gitignored": True,
        },
        "sam2_source": {
            "directory": str(SAM2_SOURCE_DIR),
            "revision": sam2_source_revision(),
            "build_cuda_extension": False,
            "build_cuda_env_note": "installed with SAM2_BUILD_CUDA=0",
            "bytes": _dir_size(SAM2_SOURCE_DIR),
        },
        "packages": package_versions(),
        "python": sys.version.split()[0],
    }

    qwen_snapshot = HF_CACHE / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct"
    if not args.verify_only or qwen_snapshot.exists():
        report["qwen"] = download_qwen() if not args.verify_only else {
            "model_id": QWEN_MODEL_ID,
            "cached_bytes": _dir_size(qwen_snapshot),
        }

    sam_path = MODEL_DIR / SAM2_FILENAME
    if not args.verify_only or sam_path.exists():
        report["sam2_checkpoint"] = download_sam2() if not args.verify_only else {
            "path": str(sam_path),
            "bytes": sam_path.stat().st_size if sam_path.exists() else 0,
        }

    report["local_cache_total_bytes"] = _dir_size(LOCAL_CACHE)

    ENVIRONMENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
