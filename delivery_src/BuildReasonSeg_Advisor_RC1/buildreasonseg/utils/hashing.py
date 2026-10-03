"""SHA256 helpers used by the model-package contract and `check_setup.py`."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_sha256(path: str | Path, expected: str | None, expected_bytes: int | None = None) -> dict:
    """Return a verification record; `matches` is False for missing files or mismatched digests."""

    target = Path(path)
    record = {"path": str(target), "exists": target.is_file(),
              "expected_sha256": expected, "expected_bytes": expected_bytes}
    if not target.is_file():
        record["matches"] = False
        return record
    record["bytes"] = target.stat().st_size
    record["sha256"] = sha256_file(target)
    checks = []
    if expected is not None:
        checks.append(record["sha256"] == expected)
    if expected_bytes is not None:
        checks.append(record["bytes"] == expected_bytes)
    record["matches"] = all(checks) if checks else True
    return record


__all__ = ["sha256_file", "sha256_text", "verify_sha256"]
