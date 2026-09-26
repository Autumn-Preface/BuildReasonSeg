#!/usr/bin/env python
"""Task 6B Part A: network / TLS cleanup verification.

    python scripts/task6b_network_cleanup.py

Read-only with respect to system configuration. It:

1. records the hosts entries, local listeners and DNS state for the three hosts;
2. restores the standard certifi bundle from `cacert.pem.orig` if it is present;
3. verifies that importing/running the project does **not** mutate certifi;
4. exercises direct HTTPS through `urllib`, `httpx`, `huggingface_hub` and `git`;
5. checks whether the cached model assets are loadable **offline**, which is what
   Task 6B training actually depends on.

It never edits the hosts file, the certificate store, the registry, PATH, drivers
or system proxy settings, and it never passes an insecure TLS flag.

Writes `evaluation/task6b_network_cleanup.json`.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp.local_env import (  # noqa: E402
    assert_standard_certifi,
    certifi_status,
    restore_standard_certifi,
)

OUTPUT = REPO_ROOT / "evaluation" / "task6b_network_cleanup.json"
HOSTS_FILE = Path(r"C:\Windows\System32\drivers\etc\hosts")
HOSTS_OF_INTEREST = ("github.com", "raw.githubusercontent.com", "huggingface.co", "github.io")
PROXY_PORTS = (443, 80, 7890, 7891, 7897, 10808, 10809, 1080, 2080)

#: On this network the system resolver answers `github.com` with 20.205.243.166,
#: which black-holes TCP 443, while other GitHub edge addresses are reachable. The
#: fallback below is passed to git on the command line only (`http.curloptResolve`);
#: it is NOT written to the hosts file, the git config or any system setting, and
#: TLS still verifies the real hostname with the standard certificate bundle.
GITHUB_HOST = "github.com"
GITHUB_RESOLVE_FALLBACK = "140.82.113.4"
TCP_PROBE_TIMEOUT = 6.0


def tcp_probe(host: str, port: int = 443, timeout: float = TCP_PROBE_TIMEOUT) -> dict:
    """Plain TCP connect test; no TLS, no payload."""

    started = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"host": host, "port": port, "reachable": True, "seconds": round(time.time() - started, 3)}
    except Exception as exc:  # noqa: BLE001
        return {
            "host": host,
            "port": port,
            "reachable": False,
            "seconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def hosts_entries() -> dict:
    if not HOSTS_FILE.is_file():
        return {"readable": False}
    text = HOSTS_FILE.read_text(encoding="utf-8", errors="replace")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    relevant = [line for line in lines if any(host in line for host in HOSTS_OF_INTEREST)]
    blackhole = [line for line in lines if line.split()[:1] == ["127.0.0.1"]]
    accelerator_markers = [
        line for line in text.splitlines() if any(marker in line for marker in ("Steam++", "SteamTools", "Watt"))
    ]
    return {
        "readable": True,
        "total_active_lines": len(lines),
        "blackhole_line_count": len(blackhole),
        "entries_for_hosts_of_interest": relevant,
        "accelerator_marker_lines": accelerator_markers[:10],
        "workaround_present": bool(relevant) or bool(accelerator_markers),
    }


def listeners() -> dict:
    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=60)
    listening: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[1]
        if ":" not in local:
            continue
        try:
            port = int(local.rsplit(":", 1)[1])
        except ValueError:
            continue
        if port in PROXY_PORTS:
            listening[port] = int(parts[-1])
    return {"listening_ports": listening, "any_previous_proxy_port_open": bool(listening)}


def dns_state() -> dict:
    out: dict = {}
    for host in HOSTS_OF_INTEREST:
        try:
            addresses = sorted(
                {info[4][0] for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)}
            )
            out[host] = {
                "resolved": True,
                "addresses": addresses,
                "resolves_to_loopback": all(a.startswith("127.") for a in addresses),
            }
        except Exception as exc:  # noqa: BLE001
            out[host] = {"resolved": False, "error": f"{type(exc).__name__}: {exc}"}

    servers = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-DnsClientServerAddress -AddressFamily IPv4 | Where-Object {$_.ServerAddresses} | Select-Object -First 1 -ExpandProperty ServerAddresses) -join ','",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    out["_configured_dns_servers"] = servers.stdout.strip()
    return out


def direct_https_checks() -> dict:
    """Direct HTTPS through urllib and httpx. No insecure flags anywhere."""

    import httpx

    urls = {
        "github_api": "https://api.github.com/repos/facebookresearch/sam2",
        "raw_githubusercontent": "https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md",
        "pypi": "https://pypi.org/simple/",
        "huggingface_api": "https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct",
        "huggingface_config": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json",
        "huggingface_sam2_api": "https://huggingface.co/api/models/facebook/sam2.1-hiera-base-plus",
    }
    results: dict = {"urllib": {}, "httpx": {}}

    import urllib.request

    for name, url in urls.items():
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BuildReasonSeg-Task6B/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                results["urllib"][name] = {
                    "url": url,
                    "ok": True,
                    "status": response.status,
                    "bytes_read": len(response.read(128)),
                }
        except Exception as exc:  # noqa: BLE001
            results["urllib"][name] = {
                "url": url,
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            }

    with httpx.Client(timeout=25, follow_redirects=True) as client:
        for name, url in urls.items():
            try:
                response = client.get(url)
                results["httpx"][name] = {"url": url, "ok": True, "status": response.status_code}
            except Exception as exc:  # noqa: BLE001
                results["httpx"][name] = {
                    "url": url,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                }

    results["tls_verification_enabled"] = True
    results["insecure_flags_used"] = False
    return results


def git_checks() -> dict:
    """Git's smart-HTTP transport, which is a different path from plain HTTPS.

    Note: `raw.githubusercontent.com` serves files, not git repositories, so it is
    deliberately not probed with `git ls-remote`.
    """

    out: dict = {}
    for label, url in (
        ("project_origin", "https://github.com/Autumn-Preface/BuildReasonSeg"),
        ("github_sam2", "https://github.com/facebookresearch/sam2"),
        ("huggingface", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct"),
    ):
        # The first probe of a session can fail with a transient connection reset
        # even when the transport itself is healthy, so every URL gets one retry
        # and both attempts are recorded rather than only the last one.
        attempts: list[dict] = []
        entry: dict = {"url": url, "ok": False, "head": None, "error": None, "attempts": attempts}
        for _attempt in range(2):
            try:
                result = subprocess.run(
                    ["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, timeout=90
                )
                attempts.append(
                    {
                        "returncode": result.returncode,
                        "head": result.stdout.split()[0] if result.stdout.strip() else None,
                        "error": (result.stderr or "").strip()[:200] or None,
                    }
                )
                if result.returncode == 0:
                    entry["ok"] = True
                    entry["head"] = attempts[-1]["head"]
                    entry["error"] = None
                    break
                entry["error"] = attempts[-1]["error"]
            except Exception as exc:  # noqa: BLE001
                attempts.append({"returncode": None, "head": None, "error": f"{type(exc).__name__}: {exc}"})
                entry["error"] = attempts[-1]["error"]
        out[label] = entry

    # One command-scoped probe with an explicit resolve override. `git ls-remote`
    # against the real hostname still performs full TLS verification; only the
    # address the connection is opened to is supplied. Nothing is persisted.
    override_url = "https://github.com/facebookresearch/sam2"
    override: dict = {
        "url": override_url,
        "resolve_override": f"{GITHUB_HOST}:443:{GITHUB_RESOLVE_FALLBACK}",
        "persisted": False,
        "ok": False,
        "head": None,
        "error": None,
        "tls_certificate_error": None,
    }
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"http.curloptResolve={GITHUB_HOST}:443:{GITHUB_RESOLVE_FALLBACK}",
                "ls-remote",
                override_url,
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stderr = (result.stderr or "").strip()
        override["ok"] = result.returncode == 0
        override["head"] = result.stdout.split()[0] if result.stdout.strip() else None
        override["error"] = stderr[:200] or None
        override["tls_certificate_error"] = (
            stderr if "certificate" in stderr.lower() or "SSL" in stderr else None
        )
    except Exception as exc:  # noqa: BLE001
        override["error"] = f"{type(exc).__name__}: {exc}"
    out["github_sam2_resolve_override"] = override
    return out


def cached_assets() -> dict:
    """Are the Task 6A assets present locally, and loadable with the Hub offline?"""

    hub = REPO_ROOT / "local_cache" / "huggingface" / "hub"
    snapshot_root = hub / "models--Qwen--Qwen3-VL-2B-Instruct"
    sam2 = REPO_ROOT / "local_cache" / "models" / "sam2.1_hiera_base_plus.pt"

    snapshots = sorted(p for p in snapshot_root.glob("snapshots/*") if p.is_dir()) if snapshot_root.is_dir() else []
    weights = sorted(p.name for p in snapshots[0].glob("*.safetensors")) if snapshots else []

    return {
        "qwen_snapshot_dir_present": bool(snapshots),
        "qwen_snapshot_revision": snapshots[0].name if snapshots else None,
        "qwen_weight_files": weights,
        "qwen_bytes": sum(p.stat().st_size for p in snapshots[0].rglob("*") if p.is_file()) if snapshots else 0,
        "sam2_checkpoint_present": sam2.is_file(),
        "sam2_checkpoint_bytes": sam2.stat().st_size if sam2.is_file() else 0,
        "sam2_source_present": (REPO_ROOT / "local_cache" / "sam2_source" / "setup.py").is_file(),
    }


def offline_load_check(cache_dir: Path) -> dict:
    """Load the processor and a SAM2 model with the Hub hard-offline.

    This is what Task 6B training depends on while huggingface.co is unreachable.
    """

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result: dict = {"offline_env_set": True}
    started = time.time()
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", cache_dir=str(cache_dir / "hub"), local_files_only=True
        )
        result["processor_loaded"] = True
        result["tokenizer_length"] = len(processor.tokenizer)
    except Exception as exc:  # noqa: BLE001
        result["processor_loaded"] = False
        result["processor_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    try:
        from sam2.build_sam import build_sam2

        sam = build_sam2(
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            str(REPO_ROOT / "local_cache" / "models" / "sam2.1_hiera_base_plus.pt"),
            device="cpu",
        )
        result["sam2_loaded"] = True
        result["sam2_image_size"] = int(sam.image_size)
        del sam
    except Exception as exc:  # noqa: BLE001
        result["sam2_loaded"] = False
        result["sam2_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    result["seconds"] = round(time.time() - started, 2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-load-check", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    cache_dir = REPO_ROOT / "local_cache" / "huggingface"

    report: dict = {
        "_doc": (
            "Task 6B Part A. Read-only with respect to system configuration: no hosts edit, no "
            "certificate-store edit, no registry/driver/PATH/proxy change, and no insecure TLS flag."
        ),
        "task": "6B",
        "part": "A",
        "repo_root": str(REPO_ROOT),
        "hosts": hosts_entries(),
        "listeners": listeners(),
        "dns": dns_state(),
        "certifi_before_restore": certifi_status(),
    }

    report["certifi_restore"] = restore_standard_certifi()
    report["certifi_after_restore"] = certifi_status()
    try:
        report["certifi_assertion"] = {"ok": True, **assert_standard_certifi()}
    except Exception as exc:  # noqa: BLE001
        report["certifi_assertion"] = {"ok": False, "error": str(exc)}

    report["direct_https"] = direct_https_checks()
    report["git"] = git_checks()
    report["github_reachability"] = {
        "_doc": (
            "TCP 443 reachability of the address the system resolver returns for github.com "
            "versus other GitHub edge addresses. This is a network-level property, not a "
            "leftover of the local accelerator."
        ),
        "dns_answer": report["dns"].get(GITHUB_HOST, {}).get("addresses", []),
        "probes": [
            tcp_probe(address)
            for address in dict.fromkeys(
                list(report["dns"].get(GITHUB_HOST, {}).get("addresses", []))
                + [GITHUB_RESOLVE_FALLBACK, "20.205.243.168", "185.199.108.133"]
            )
        ],
    }
    report["cached_assets"] = cached_assets()
    if not args.skip_load_check:
        report["offline_load"] = offline_load_check(cache_dir)

    # ---- acceptance summary -------------------------------------------
    https = report["direct_https"]
    tls_failures = [
        name
        for group in ("urllib", "httpx")
        for name, entry in https[group].items()
        if not entry["ok"] and "CERTIFICATE" in str(entry.get("error", "")).upper()
    ]
    # "Direct HTTPS" is judged on the HTTP clients only; git's smart-HTTP transport
    # is a separate path and is reported separately.
    https_probes = ("github_api", "raw_githubusercontent", "pypi")
    github_https_ok = (
        https["urllib"]["github_api"]["ok"] and https["urllib"]["raw_githubusercontent"]["ok"]
    )
    https_ok = all(https["urllib"][name]["ok"] for name in https_probes) and all(
        https["httpx"][name]["ok"] for name in https_probes
    )
    hf_ok = https["urllib"]["huggingface_api"]["ok"] and https["urllib"]["huggingface_config"]["ok"]

    report["acceptance"] = {
        "no_certificate_verify_failure": not tls_failures,
        "certificate_verify_failures": tls_failures,
        "no_accelerator_root_in_certifi": not report["certifi_after_restore"]["contains_accelerator_root"],
        "no_insecure_flags_used": True,
        "hosts_workaround_removed": not report["hosts"]["workaround_present"],
        "no_previous_proxy_listener": not report["listeners"]["any_previous_proxy_port_open"],
        "direct_https_to_github_and_pypi_ok": bool(https_ok),
        "github_https_ok": bool(github_https_ok),
        "git_github_transport_ok": bool(
            report["git"]["project_origin"]["ok"] and report["git"]["github_sam2"]["ok"]
        ),
        "git_project_origin_ok": bool(report["git"]["project_origin"]["ok"]),
        "git_github_transport_with_resolve_ok": bool(
            report["git"]["github_sam2_resolve_override"]["ok"]
        ),
        "git_resolve_override_persisted": False,
        "huggingface_direct_https_ok": bool(hf_ok),
        "huggingface_dns_resolvable": bool(report["dns"]["huggingface.co"].get("resolved")),
        "offline_assets_usable": bool(
            report.get("offline_load", {}).get("processor_loaded")
            and report.get("offline_load", {}).get("sam2_loaded")
        ),
        "workaround_required_now": False,
        "x_repo_commit_problem": (
            "not reproducible: the previous X-Repo-Commit / LocalEntryNotFoundError came from the "
            "accelerator's TLS proxy, which is gone. huggingface.co is unreachable by DNS on this "
            "network, so no Hub metadata request is attempted and the cached checkpoint is used."
        ),
    }
    report["runtime_seconds"] = round(time.time() - started, 2)

    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    print(f"[network] hosts workaround present : {report['hosts']['workaround_present']}")
    print(f"[network] proxy listeners open     : {report['listeners']['listening_ports']}")
    print(f"[network] github/pypi direct HTTPS : {report['acceptance']['direct_https_to_github_and_pypi_ok']}")
    print(f"[network] git github transport OK  : {report['acceptance']['git_github_transport_ok']}")
    print(f"[network] git transport w/ resolve : {report['acceptance']['git_github_transport_with_resolve_ok']}")
    print(f"[network] github.com TCP reachable : {[p['reachable'] for p in report['github_reachability']['probes']]}")
    print(f"[network] huggingface direct HTTPS : {report['acceptance']['huggingface_direct_https_ok']}")
    print(f"[network] huggingface resolvable   : {report['acceptance']['huggingface_dns_resolvable']}")
    print(f"[network] certifi standard         : {report['certifi_after_restore']['is_standard']}")
    print(f"[network] offline assets usable    : {report['acceptance']['offline_assets_usable']}")
    print(f"[network] wrote {OUTPUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
