"""Setup-checker tests (Task 8A section 19.5): good fixture → READY, each broken case → non-zero."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_check(root: Path) -> subprocess.CompletedProcess:
    # Task 8B.2-R2: after `check_setup.py` imports ultralytics/transformers the child writes UTF-8 to the pipe;
    # decode explicitly instead of relying on the machine locale (GBK), which left stdout as None.
    return subprocess.run([sys.executable, str(root / "check_setup.py")], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=str(root))


@pytest.fixture()
def fixture_project(tmp_path: Path) -> Path:
    """A copy of the delivery project with metadata but without the large weights."""

    target = tmp_path / "BuildReasonSeg_Advisor_RC1"
    for name in ("buildreasonseg", "configs", "docs", "tests"):
        shutil.copytree(ROOT / name, target / name)
    for name in ("check_setup.py", "VERSION", "README.md"):
        shutil.copy2(ROOT / name, target / name)
    (target / "model" / "buildreasonseg_advisor").mkdir(parents=True)
    shutil.copy2(ROOT / "model" / "buildreasonseg_advisor" / "model.yaml",
                 target / "model" / "buildreasonseg_advisor" / "model.yaml")
    metadata = json.loads((ROOT / "model" / "buildreasonseg_advisor" / "metadata.json")
                          .read_text(encoding="utf-8"))
    # record the small-file hashes so the fixture can satisfy the checker
    (target / "model" / "buildreasonseg_advisor" / "decoder.pt").write_bytes(b"decoder-fixture")
    (target / "model" / "buildreasonseg_advisor" / "detector.pt").write_bytes(b"detector-fixture")
    from buildreasonseg.utils.hashing import sha256_file

    decoder = target / "model" / "buildreasonseg_advisor" / "decoder.pt"
    detector = target / "model" / "buildreasonseg_advisor" / "detector.pt"
    for asset in metadata["assets"]:
        if asset["role"] == "decoder":
            asset["sha256"] = sha256_file(decoder)
            asset["bytes"] = decoder.stat().st_size
        elif asset["role"] == "detector":
            asset["sha256"] = sha256_file(detector)
            asset["bytes"] = detector.stat().st_size
    (target / "model" / "buildreasonseg_advisor" / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    # model.yaml carries the hashes check_setup verifies: sync it with the fixture weights
    model_yaml = target / "model" / "buildreasonseg_advisor" / "model.yaml"
    yaml_text = model_yaml.read_text(encoding="utf-8")
    yaml_text = yaml_text.replace(
        "9187b133ee4c71ca2750d421d6149bdba1812eabc8c9faacde193036c0db8586",
        sha256_file(decoder)).replace("1117495", str(decoder.stat().st_size))
    yaml_text = yaml_text.replace(
        "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474",
        sha256_file(detector)).replace("54480241", str(detector.stat().st_size))
    model_yaml.write_text(yaml_text, encoding="utf-8")
    # components: create placeholder components with the hashes the checker expects
    sam2 = target / "model" / "components" / "sam2"
    sam2.mkdir(parents=True)
    (sam2 / "sam2.1_hiera_base_plus.pt").write_bytes(b"sam2-fixture")
    (sam2 / "sam2.1_hiera_b+.yaml").write_text("config\n", encoding="utf-8")
    program_head = target / "model" / "components" / "program_head"
    qwen = program_head / "Qwen3-VL-2B-Instruct"
    qwen.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (qwen / name).write_text("{}", encoding="utf-8")
    (program_head / "program_parser_l3_rehearsal_v1.pt").write_bytes(b"program-head-fixture")
    for asset in metadata["assets"]:
        if asset["role"] == "sam2_checkpoint":
            asset["sha256"] = sha256_file(sam2 / "sam2.1_hiera_base_plus.pt")
            asset["bytes"] = (sam2 / "sam2.1_hiera_base_plus.pt").stat().st_size
        elif asset["role"] == "program_head":
            asset["sha256"] = sha256_file(program_head / "program_parser_l3_rehearsal_v1.pt")
            asset["bytes"] = (program_head / "program_parser_l3_rehearsal_v1.pt").stat().st_size
    (target / "model" / "buildreasonseg_advisor" / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in ("datasets", "inference/input", "inference/output/masks",
                 "inference/output/overlays", "inference/output/diagnostics", "runs/train",
                 "runs/eval", "logs"):
        (target / name).mkdir(parents=True, exist_ok=True)
    # Task 8B: the setup checker verifies the full Qwen asset manifest
    import subprocess as _subprocess

    manifest_code = (
        "import sys, json; sys.path.insert(0, r'%s');"
        "from buildreasonseg.runtime import manifest as m;"
        "print(json.dumps(m.write_manifest(m.build_manifest())))" % target)
    completed = _subprocess.run([sys.executable, "-c", manifest_code], capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
    assert completed.returncode == 0, completed.stderr
    return target


def test_good_fixture_is_ready(fixture_project: Path) -> None:
    completed = run_check(fixture_project)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "BuildReasonSeg environment: READY" in completed.stdout


def test_missing_decoder_nonzero(fixture_project: Path) -> None:
    (fixture_project / "model" / "buildreasonseg_advisor" / "decoder.pt").unlink()
    completed = run_check(fixture_project)
    assert completed.returncode != 0
    assert "[MISSING]" in completed.stdout
    assert "NOT READY" in completed.stdout


def test_hash_mismatch_nonzero(fixture_project: Path) -> None:
    (fixture_project / "model" / "buildreasonseg_advisor" / "decoder.pt").write_bytes(b"tampered")
    completed = run_check(fixture_project)
    assert completed.returncode != 0
    assert "[HASH MISMATCH]" in completed.stdout


def test_missing_qwen_component_nonzero(fixture_project: Path) -> None:
    qwen = fixture_project / "model" / "components" / "program_head" / "Qwen3-VL-2B-Instruct"
    shutil.rmtree(qwen)
    completed = run_check(fixture_project)
    assert completed.returncode != 0
    assert "[MISSING]" in completed.stdout


def test_missing_sam2_nonzero(fixture_project: Path) -> None:
    (fixture_project / "model" / "components" / "sam2" / "sam2.1_hiera_base_plus.pt").unlink()
    completed = run_check(fixture_project)
    assert completed.returncode != 0
    assert "[MISSING]" in completed.stdout


def test_invalid_model_yaml_nonzero(fixture_project: Path) -> None:
    (fixture_project / "model" / "buildreasonseg_advisor" / "model.yaml").write_text(
        "name: fixture\nversion: RC1\ndecoder:\n  architecture: Z-B3\n  sha256: abc\n"
        "detector:\n  family: YOLO26m-seg\n  sha256: abc\n", encoding="utf-8")
    completed = run_check(fixture_project)
    assert completed.returncode != 0
    assert "[INVALID]" in completed.stdout


def test_real_project_check_reports_ready() -> None:
    completed = run_check(ROOT)
    assert "BuildReasonSeg environment: READY" in completed.stdout, completed.stdout
    assert completed.returncode == 0


# ------------------------------------------------- Task 8B.2-R2 runtime dependency regressions


def _load_check_setup_module():
    import importlib.util

    module_path = ROOT / "check_setup.py"
    spec = importlib.util.spec_from_file_location("rc1_check_setup_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fresh_report(module):
    return module.Report()


def test_real_runtime_reports_ultralytics_and_ready() -> None:
    completed = run_check(ROOT)
    assert "Ultralytics runtime" in completed.stdout
    assert "8.4.164" in completed.stdout
    assert "BuildReasonSeg environment: READY" in completed.stdout, completed.stdout
    assert completed.returncode == 0


def test_runtime_ultralytics_missing_is_not_ready(monkeypatch, tmp_path: Path) -> None:
    import builtins

    module = _load_check_setup_module()
    report = _fresh_report(module)
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ModuleNotFoundError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    module.check_runtime_dependencies(report, None)
    assert report.ready is False
    assert any("[MISSING] Ultralytics runtime" in line for line in report.lines), report.lines


def test_runtime_ultralytics_wrong_version_is_not_ready(monkeypatch, tmp_path: Path) -> None:
    import builtins
    import types

    module = _load_check_setup_module()
    report = _fresh_report(module)
    real_import = builtins.__import__
    fake = types.ModuleType("ultralytics")
    fake.__version__ = "8.3.0"

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            return fake
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module.check_runtime_dependencies(report, None)
    assert report.ready is False
    assert any("[INVALID] Ultralytics runtime" in line for line in report.lines), report.lines
    assert any("8.4.164" in line for line in report.lines)


def test_portability_wording_no_longer_claims_bundled_runtime() -> None:
    completed = run_check(ROOT)
    assert "source/assets portability" in completed.stdout
    assert "项目可整体移动" not in completed.stdout
    assert "runtime environment" in completed.stdout


def test_runtime_and_provenance_are_distinguished() -> None:
    completed = run_check(ROOT)
    assert "Ultralytics runtime" in completed.stdout
    assert "model provenance" in completed.stdout
    assert "Transformers runtime" in completed.stdout
    check_setup_source = (ROOT / "check_setup.py").read_text(encoding="utf-8")
    assert "def model_provenance_version" in check_setup_source
    assert "REQUIRED_ULTRALYTICS = \"8.4.164\"" in check_setup_source
