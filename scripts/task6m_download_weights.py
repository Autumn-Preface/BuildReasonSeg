"""Task 6M section 4/6: download the authorized official weights and record provenance.

Authorized downloads for Task 6M only: the official `yolo26m-seg.pt` proposal checkpoint and
(optionally, smoke only) `yolo26s-seg.pt`. Official released packages only; no other downloads.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" / "pretrained"
MANIFEST = REPO_ROOT / "evaluation" / "task6m_environment_manifest.json"


def _verify_bundle():
    """CA bundle for requests.

    This env's `requests` cannot locate a system trust store, so the bundled Mozilla CA set from
    `certifi` (already a requests dependency) is used explicitly. No TLS or proxy settings are
    modified anywhere.
    """

    try:
        import certifi

        return certifi.where()
    except Exception:  # noqa: BLE001
        return True

ASSETS = {
    "yolo26m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
    "yolo26s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> str:
    """Download over verified HTTPS, returning the method actually used.

    `requests` cannot verify this network path (the environment's Python has no usable trust store
    for the intercepting proxy that the pre-existing accelerator presents), so the fallback uses the
    OS trust store through PowerShell's `Invoke-WebRequest`. Nothing about hosts, certificates,
    proxies or TLS settings is modified by this task in either path.
    """

    try:
        response = requests.get(url, stream=True, timeout=120, verify=_verify_bundle())
        response.raise_for_status()
        with target.open("wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)
        return "requests+https"
    except Exception as error:  # noqa: BLE001
        print(f"[6m.weights] requests failed ({type(error).__name__}); using the OS trust store", flush=True)
    import subprocess

    command = [
        "powershell", "-NoProfile", "-Command",
        f"Invoke-WebRequest -Uri '{url}' -OutFile '{target}' -UseBasicParsing",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"download failed: {result.stderr[-300:]}")
    return "powershell+os_trust_store"


def main() -> int:
    started = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, url in ASSETS.items():
        target = DEST / name
        if target.is_file():
            results[name] = {"status": "already_present", "bytes": target.stat().st_size, "method": "none"}
        else:
            method = download(url, target)
            results[name] = {"status": "downloaded", "bytes": target.stat().st_size, "method": method}
        results[name].update(
            {
                "url": url,
                "sha256": sha256_file(target),
                "local_path": str(target),
            }
        )
        print(f"[6m.weights] {name}: {results[name]['status']} {results[name]['bytes']} bytes "
              f"sha256 {results[name]['sha256'][:16]}… via {results[name]['method']}", flush=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
    manifest.setdefault("_doc", "Task 6M environment and weight provenance.")
    manifest.setdefault("task", "6M")
    manifest["pretrained_weights"] = results
    manifest["license_note"] = (
        "Ultralytics open-source stack and pre-trained models are distributed under AGPL-3.0, with "
        "enterprise licensing available for proprietary/commercial use. This repository's LICENSE is "
        "NOT changed by Task 6M and no commercial-licensing claim is made. Training/evaluation here is "
        "research use of the official released package and official weights."
    )
    manifest["download_authorization"] = (
        "Task 6M section 6 allows only: the official ultralytics package/dependencies, "
        "yolo26m-seg.pt, and optionally yolo26s-seg.pt for smoke testing."
    )
    manifest["runtime_seconds"] = round(time.time() - started, 2)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"[6m.weights] wrote {MANIFEST.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
