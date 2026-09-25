"""Project-local runtime environment helpers.

Task 6A runs on a machine where a local network accelerator (Steam++ / Watt
Toolkit) terminates TLS for some hosts and re-signs them with its own root
certificate ("SteamTools Certificate", BeyondDimension). That root is installed
in the Windows certificate store, so `ssl.create_default_context()` trusts it --
but the `certifi` bundle that `httpx` / `huggingface_hub` load by default does
not, which makes Hub downloads fail with `CERTIFICATE_VERIFY_FAILED`.

`ensure_windows_roots_in_certifi()` appends the Windows ROOT and CA stores to the
environment's certifi bundle. This is a **process-local trust configuration**:

* it touches only files inside this project's dedicated conda environment and
  `local_cache/` (both gitignored);
* it changes no system setting, no registry key, no driver, no PATH and no
  system-wide environment variable;
* the original bundle is backed up next to it as `cacert.pem.orig`.

It is a no-op on a machine whose certifi bundle already covers the intercepting
root, and a no-op if the bundle cannot be written.
"""

from __future__ import annotations

import ssl
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CA_DIR = REPO_ROOT / "local_cache" / "ca"
COMBINED_BUNDLE = CA_DIR / "windows-and-certifi.pem"


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


def ensure_windows_roots_in_certifi(verbose: bool = False) -> dict:
    """Append the Windows trust stores to certifi. Returns a small report."""

    report = {"applied": False, "windows_roots": 0, "certifi_path": "", "reason": ""}

    try:
        import certifi
    except Exception as exc:  # noqa: BLE001
        report["reason"] = f"certifi unavailable: {type(exc).__name__}"
        return report

    certifi_path = Path(certifi.where())
    report["certifi_path"] = str(certifi_path)

    roots = _windows_roots_der()
    report["windows_roots"] = len(roots)
    if not roots:
        report["reason"] = "no Windows certificate stores readable (non-Windows host)"
        return report

    try:
        current = certifi_path.read_bytes()
        if b"SteamTools" in current or b"BeyondDimension" in current:
            report["reason"] = "certifi bundle already covers the local accelerator root"
            return report
        additions = b"".join(ssl.DER_cert_to_PEM_cert(der).encode() for der in roots)
        backup = certifi_path.with_suffix(".pem.orig")
        if not backup.exists():
            backup.write_bytes(current)
        certifi_path.write_bytes(current + b"\n" + additions)

        CA_DIR.mkdir(parents=True, exist_ok=True)
        COMBINED_BUNDLE.write_bytes(current + b"\n" + additions)

        report["applied"] = True
        report["reason"] = f"appended {len(roots)} Windows roots to {certifi_path}"
    except Exception as exc:  # noqa: BLE001
        report["reason"] = f"{type(exc).__name__}: {exc}"

    if verbose:
        print(f"local trust setup: {report}")
    return report
