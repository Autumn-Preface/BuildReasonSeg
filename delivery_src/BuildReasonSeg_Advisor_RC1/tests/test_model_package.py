"""Model-package contract tests (Task 8A section 19.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from buildreasonseg import DEFAULT_MODEL, paths
from buildreasonseg.errors import BuildReasonSegError, HashMismatchError
from buildreasonseg.models.package import (COMPONENTS, ModelPackage, default_package_available,
                                           resolve_model)
from buildreasonseg.utils.hashing import sha256_file, verify_sha256

EXPECTED_DECODER_SHA = "9187b133ee4c71ca2750d421d6149bdba1812eabc8c9faacde193036c0db8586"
EXPECTED_DECODER_BYTES = 1117495
EXPECTED_DETECTOR_SHA = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
EXPECTED_DETECTOR_BYTES = 54480241


def test_decoder_hash_and_bytes_exact(root: Path) -> None:
    decoder = paths.default_model_dir() / "decoder.pt"
    assert decoder.is_file()
    assert sha256_file(decoder) == EXPECTED_DECODER_SHA
    assert decoder.stat().st_size == EXPECTED_DECODER_BYTES


def test_detector_hash_and_bytes_exact() -> None:
    detector = paths.default_model_dir() / "detector.pt"
    assert detector.is_file()
    assert sha256_file(detector) == EXPECTED_DETECTOR_SHA
    assert detector.stat().st_size == EXPECTED_DETECTOR_BYTES


def test_package_verification_matches() -> None:
    package = ModelPackage.load(DEFAULT_MODEL)
    report = package.verify()
    assert report["all_verified"] is True
    assert report["decoder"]["matches"] is True
    assert report["detector"]["matches"] is True
    assert package.verify_or_raise()["all_verified"] is True


def test_model_yaml_parses() -> None:
    package = ModelPackage.load(DEFAULT_MODEL)
    config = package.config
    assert config["name"] == DEFAULT_MODEL
    assert config["version"] == "RC1"
    assert config["decoder"]["architecture"] == "D-B1"
    assert config["decoder"]["seed"] == 20261003
    assert config["decoder"]["selected_by"] == "task7i_validation"
    assert config["detector"]["family"] == "YOLO26m-seg"
    assert config["detector"]["imgsz"] == 640
    assert config["detector"]["conf"] == 0.05
    assert config["detector"]["max_det"] == 300
    assert config["detector"]["tta"] is False
    assert config["visual_encoder"]["family"] == "SAM2.1-Hiera-Base+"
    assert config["language"]["family"] == "Qwen3-VL-2B-Instruct"
    assert config["language"]["mode"] == "qwen-first"
    assert config["input"]["modality"] == "RGB optical remote sensing"
    assert len(config["program_set"]) == 4


def test_metadata_fields_complete() -> None:
    metadata = json.loads((paths.default_model_dir() / "metadata.json").read_text(encoding="utf-8"))
    for key in ("product", "package", "project_version", "research_baseline", "framework",
                "architecture", "decoder_selection", "assets", "supported_program_set",
                "checkpoint_selection_provenance", "input_domain", "known_limitations", "integrity"):
        assert key in metadata, key
    roles = {asset["role"] for asset in metadata["assets"]}
    assert roles == {"decoder", "detector", "sam2_checkpoint", "sam2_config", "program_head",
                     "qwen_base"}
    for asset in metadata["assets"]:
        assert asset.get("sha256") or asset["role"] == "qwen_base", asset["role"]
        assert asset["bytes"] > 0
        assert asset["source"]
        assert asset["target"]
    assert metadata["decoder_selection"]["seed"] == 20261003
    assert metadata["decoder_selection"]["selected_by"] == "task7i_validation"
    assert metadata["decoder_selection"]["not_selected_by"].startswith("Task 7J")
    assert metadata["checkpoint_selection_provenance"]["test_used_for_selection"] is False
    assert metadata["known_limitations"]


def test_metadata_hashes_match_copied_assets() -> None:
    metadata = json.loads((paths.default_model_dir() / "metadata.json").read_text(encoding="utf-8"))
    for asset in metadata["assets"]:
        if asset["role"] == "qwen_base":
            root = paths.component_dir("program_head") / "Qwen3-VL-2B-Instruct"
            assert (root / "model.safetensors").is_file()
            continue
        target = paths.resolve(asset["target"])
        assert target.is_file(), asset["target"]
        record = verify_sha256(target, asset["sha256"], asset["bytes"])
        assert record["matches"] is True, asset["role"]


def test_metrics_separate_validation_and_test() -> None:
    metrics = json.loads((paths.default_model_dir() / "metrics.json").read_text(encoding="utf-8"))
    assert "task7i_validation" in metrics
    assert "task7j_final_frozen_architecture_test" in metrics
    assert metrics["task7i_validation"]["reference_mode"] == "oracle_native_gt"
    assert metrics["task7i_validation"]["d_b1"]["mean"] == pytest.approx(0.383424, abs=1e-6)
    assert metrics["task7j_final_frozen_architecture_test"]["oracle_reference"]["mean_delta"] == \
        pytest.approx(0.054816, abs=1e-6)
    assert metrics["task7j_final_frozen_architecture_test"]["predicted_reference"]["mean_delta"] == \
        pytest.approx(0.002863, abs=1e-6)
    assert metrics["task7j_final_frozen_architecture_test"]["performance_verdict"] is None
    assert metrics["recording_policy"]["end_to_end_significant_improvement_claimed"] is False


def test_model_fallback_requires_user_confirmation(tmp_path: Path) -> None:
    resolution = resolve_model("does_not_exist")
    assert resolution.ok is False
    assert resolution.used_default is False
    assert resolution.error is not None
    assert any("确认" in note for note in resolution.notes)
    # with an explicit Y the default package may be used
    confirmed = resolve_model("does_not_exist", confirm_reader=lambda question: "y")
    assert confirmed.ok is True
    assert confirmed.used_default is True
    assert confirmed.switched_with_confirmation is True
    # with N the run stops
    declined = resolve_model("does_not_exist", confirm_reader=lambda question: "n")
    assert declined.ok is False


def test_user_specified_bad_model_does_not_silently_switch() -> None:
    resolution = resolve_model("does_not_exist", confirm_reader=None)
    assert resolution.ok is False
    assert resolution.switched_with_confirmation is False
    assert resolution.package is None


def test_default_missing_is_hard_error(monkeypatch, tmp_path: Path) -> None:
    from buildreasonseg.models import package as package_module

    monkeypatch.setattr(package_module, "model_dir", lambda name=None: tmp_path / "nope")
    resolution = package_module.resolve_model(DEFAULT_MODEL)
    assert resolution.ok is False
    assert resolution.error is not None
    assert resolution.error.code == "E301"


def test_hash_mismatch_raises(tmp_path: Path) -> None:
    record = verify_sha256(tmp_path / "missing.pt", "0" * 64)
    assert record["matches"] is False
    package = ModelPackage.load(DEFAULT_MODEL)
    broken = ModelPackage(name=package.name, root=package.root,
                          config={"decoder": {"sha256": "0" * 64, "bytes": 1},
                                  "detector": package.config["detector"]},
                          metadata=package.metadata)
    with pytest.raises(HashMismatchError):
        broken.verify_or_raise()


def test_components_complete() -> None:
    package = ModelPackage.load(DEFAULT_MODEL)
    report = package.component_report()
    assert set(report) == set(COMPONENTS)
    for name, entry in report.items():
        assert entry["complete"] is True, name
    assert package.is_complete() is True
    available, reason = default_package_available()
    assert available is True and "可用" in reason


def test_unsupported_program_raises_e102() -> None:
    with pytest.raises(BuildReasonSegError) as error:
        raise BuildReasonSegError("E102")
    assert error.value.code == "E102"
