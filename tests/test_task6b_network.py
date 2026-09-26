"""Task 6B network / TLS cleanup tests.

Covers, from the Task 6B section 21 list:

 1. normal runtime does not mutate certifi
 2. the Watt-specific helper is not auto-called

Run with pytest, or directly::

    python tests/test_task6b_network.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from buildreasonseg_mvp import local_env  # noqa: E402

MVP_PACKAGE = REPO_ROOT / "buildreasonseg_mvp"
DOWNLOAD_SCRIPTS = sorted((REPO_ROOT / "scripts").glob("task6*.py"))


def _hash() -> str:
    return hashlib.sha256(local_env.certifi_bundle_path().read_bytes()).hexdigest()


def test_importing_and_running_local_env_does_not_touch_certifi():
    require = None  # no pytest import needed for this one
    before = _hash()
    local_env.certifi_status()
    try:
        local_env.assert_standard_certifi()
    except RuntimeError:
        # A non-standard bundle is a legitimate finding, but it must not be
        # *caused* by importing or calling these helpers.
        pass
    local_env.legacy_merge_windows_roots_into_certifi()  # no confirm -> no-op
    after = _hash()
    assert before == after, "certifi was mutated by an ordinary helper call"
    print("  [1/2a] runtime helpers leave certifi untouched OK")


def test_no_module_calls_the_legacy_helper_automatically():
    """Only the definition site and tests may mention the legacy merge."""

    offenders: list[str] = []
    for path in list(MVP_PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        # the definition itself lives in local_env.py
        if path.name == "local_env.py":
            continue
        if "legacy_merge_windows_roots_into_certifi" in text:
            offenders.append(path.name)
    for path in DOWNLOAD_SCRIPTS:
        if "legacy_merge_windows_roots_into_certifi" in path.read_text(encoding="utf-8"):
            offenders.append(f"scripts/{path.name}")
    assert not offenders, f"legacy certifi merge is still called from: {offenders}"

    # the removed Task 6A entry point must not exist anywhere
    for path in list(MVP_PACKAGE.glob("*.py")) + DOWNLOAD_SCRIPTS:
        assert "ensure_windows_roots_in_certifi" not in path.read_text(encoding="utf-8"), path.name
    print("  [2/2b] legacy helper is opt-in only OK")


def test_offline_environment_flags_are_set_by_the_training_entrypoint():
    """Task 6B training must not depend on reaching the Hub."""

    source = (REPO_ROOT / "scripts" / "task6b_train.py").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
    print("  [extra] training entrypoint sets offline Hub flags OK")


def test_no_insecure_tls_flags_in_project_code():
    """No `verify=False`, no `ssl._create_unverified_context`, anywhere."""

    patterns = ("verify=False", "_create_unverified_context", "CERT_NONE", "check_hostname = False")
    for path in list(MVP_PACKAGE.glob("*.py")) + list((REPO_ROOT / "scripts").glob("task6*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert pattern not in text, f"{path.name} contains insecure TLS usage: {pattern}"
    print("  [extra] no insecure TLS flags OK")


def test_network_cleanup_report_exists_and_records_no_workaround():
    report_path = REPO_ROOT / "evaluation" / "task6b_network_cleanup.json"
    if not report_path.is_file():
        print("  [extra] network cleanup report absent; run scripts/task6b_network_cleanup.py")
        return
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["acceptance"]["no_certificate_verify_failure"] is True
    assert report["acceptance"]["no_insecure_flags_used"] is True
    assert report["acceptance"]["hosts_workaround_removed"] is True
    assert report["acceptance"]["no_accelerator_root_in_certifi"] is True
    assert report["certifi_after_restore"]["contains_accelerator_root"] is False
    print("  [extra] network cleanup report consistent OK")


def main() -> int:
    tests = [
        ("1a no certifi mutation", test_importing_and_running_local_env_does_not_touch_certifi),
        ("2b legacy opt-in only", test_no_module_calls_the_legacy_helper_automatically),
        ("offline flags", test_offline_environment_flags_are_set_by_the_training_entrypoint),
        ("no insecure TLS", test_no_insecure_tls_flags_in_project_code),
        ("cleanup report", test_network_cleanup_report_exists_and_records_no_workaround),
    ]
    failures = 0
    for name, function in tests:
        try:
            function()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} task6b network checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
