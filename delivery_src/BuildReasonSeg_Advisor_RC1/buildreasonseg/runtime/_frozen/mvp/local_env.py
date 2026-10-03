"""Project-local runtime environment helpers.

Task 6B change: **running ordinary project code no longer mutates TLS trust.**
The default trust store is the standard `certifi` bundle, unmodified.

Historical note (kept deliberately)
-----------------------------------
Task 6A ran on a machine where a local network accelerator (Steam++ / Watt Toolkit)
terminated TLS for some hosts and re-signed them with its own root certificate
("SteamTools Certificate", BeyondDimension). `httpx` / `huggingface_hub` load the
`certifi` bundle rather than the Windows certificate store, so Hub downloads
failed with `CERTIFICATE_VERIFY_FAILED`. Task 6A therefore appended the Windows
ROOT/CA stores to this environment's `certifi` bundle at startup, backing the
original up as `cacert.pem.orig`.

That workaround is **no longer applied automatically**. Task 6B:
* restored the pristine bundle from `cacert.pem.orig`;
* removed the automatic call from the download/runtime paths;
* retains the old merge only as :func:`legacy_merge_windows_roots_into_certifi`,
  an explicit, confirmed opt-in that should be needed only if such an accelerator
  is deliberately reinstalled.

Nothing here edits the Windows certificate store, the registry, drivers, PATH or
any system-wide setting. The opt-in writes only to this project's own conda
environment, which is gitignored.
"""

from __future__ import annotations

import hashlib
import ssl
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CA_DIR = REPO_ROOT / "local_cache" / "ca"
COMBINED_BUNDLE = CA_DIR / "windows-and-certifi.pem"

#: Markers of the accelerator root that Task 6A had to trust.
ACCELERATOR_MARKERS = ("SteamTools", "BeyondDimension")

#: Certificate count shipped by the standard certifi bundle used in Task 6B.
STANDARD_CERTIFICATE_COUNT = 121


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certifi_bundle_path() -> Path:
    import certifi

    return Path(certifi.where())


def certifi_status() -> dict:
    """Report whether the environment's certifi bundle is standard or modified."""

    path = certifi_bundle_path()
    text = path.read_text(encoding="utf-8", errors="replace")
    backup = path.with_suffix(".pem.orig")
    status = {
        "path": str(path),
        "sha256": sha256_file(path),
        "certificate_count": text.count("-----BEGIN CERTIFICATE-----"),
        "contains_accelerator_root": any(marker in text for marker in ACCELERATOR_MARKERS),
        "backup_path": str(backup),
        "backup_exists": backup.is_file(),
    }
    status["backup_sha256"] = sha256_file(backup) if backup.is_file() else None
    status["matches_backup"] = (
        status["backup_sha256"] == status["sha256"] if backup.is_file() else None
    )
    status["standard_certificate_count"] = STANDARD_CERTIFICATE_COUNT
    status["is_standard"] = bool(
        not status["contains_accelerator_root"]
        and status["certificate_count"] == STANDARD_CERTIFICATE_COUNT
        and (status["matches_backup"] is not False)
    )
    return status


def assert_standard_certifi() -> dict:
    """Raise if the bundle carries anything beyond the standard certificate set."""

    status = certifi_status()
    if status["contains_accelerator_root"]:
        raise RuntimeError(
            "certifi bundle contains a local accelerator root; restore it from "
            f"{status['backup_path']}"
        )
    if status["matches_backup"] is False:
        raise RuntimeError(
            "certifi bundle differs from its recorded pristine backup; restore it from "
            f"{status['backup_path']}"
        )
    return status


def restore_standard_certifi() -> dict:
    """Restore the bundle from `cacert.pem.orig` if a backup is present."""

    import shutil

    path = certifi_bundle_path()
    backup = path.with_suffix(".pem.orig")
    result = {"restored": False, "path": str(path), "backup": str(backup)}
    if not backup.is_file():
        result["reason"] = "no pristine backup present"
        result.update({k: v for k, v in certifi_status().items() if k not in result})
        return result

    result["sha256_before"] = sha256_file(path)
    result["backup_sha256"] = sha256_file(backup)
    shutil.copyfile(backup, path)
    result["sha256_after"] = sha256_file(path)
    result["restored"] = result["sha256_after"] == result["backup_sha256"]
    result.update({k: v for k, v in certifi_status().items() if k not in result})
    return result


def _windows_roots_der() -> list[bytes]:
    out: list[bytes] = []
    for store in ("ROOT", "CA"):
        try:
            for der, encoding, _trust in ssl.enum_certificates(store):
                if encoding == "x509_asn":
                    out.append(der)
        except Exception:  # noqa: BLE001 - non-Windows or restricted store
            continue
    return out


def legacy_merge_windows_roots_into_certifi(confirm: bool = False) -> dict:
    """LEGACY, opt-in only: append the Windows trust stores to `certifi`.

    This is the Task 6A behaviour. It is **not** called anywhere in the project any
    more. It exists so a future accelerator-blocked machine has a documented,
    explicit escape hatch instead of an implicit startup side effect.

    Requires ``confirm=True`` so it can never be triggered by accident.
    """

    status = certifi_status()
    if not confirm:
        return {
            "applied": False,
            "reason": "legacy helper requires confirm=True; automatic mutation was removed in Task 6B",
            **status,
        }

    path = certifi_bundle_path()
    roots = _windows_roots_der()
    if not roots:
        return {"applied": False, "reason": "no Windows certificate stores readable", **status}

    current = path.read_bytes()
    backup = path.with_suffix(".pem.orig")
    if not backup.exists():
        backup.write_bytes(current)
    additions = b"".join(ssl.DER_cert_to_PEM_cert(der).encode() for der in roots)
    path.write_bytes(current + b"\n" + additions)

    CA_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_BUNDLE.write_bytes(current + b"\n" + additions)

    return {
        "applied": True,
        "reason": f"legacy merge appended {len(roots)} Windows roots",
        **certifi_status(),
    }
