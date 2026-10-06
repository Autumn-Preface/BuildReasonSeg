"""CLI contract tests (Task 8A section 19.3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True,
                          text=True, cwd=str(ROOT))


@pytest.mark.parametrize("script", ["predict.py", "train.py", "prepare_dataset.py", "evaluate.py",
                                    "check_setup.py"])
def test_each_cli_help_works(script: str) -> None:
    completed = run_cli(script, "--help")
    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


def test_predict_help_lists_frozen_arguments() -> None:
    completed = run_cli("predict.py", "--help")
    for flag in ("--image", "--input-dir", "--prompt", "--model", "--reference-id",
                 "--inspect-proposals", "--device", "--alpha", "--save-diagnostics",
                 "--no-save-diagnostics", "--confirm-command"):
        assert flag in completed.stdout, flag


def test_predict_image_and_input_dir_are_exclusive() -> None:
    completed = run_cli("predict.py", "--image", "a.tif", "--input-dir", "somewhere",
                        "--prompt", "最大建筑左侧最近的建筑")
    assert completed.returncode == 2
    both_missing = run_cli("predict.py", "--prompt", "最大建筑左侧最近的建筑")
    assert both_missing.returncode == 2


def test_predict_prompt_required_for_normal_inference(tmp_path: Path) -> None:
    image = tmp_path / "tile.png"
    image.write_bytes(b"not-an-image")
    completed = run_cli("predict.py", "--image", str(image))
    assert completed.returncode == 10  # E101 INVALID_COMMAND
    assert "E101" in completed.stderr


def test_predict_inspect_proposals_does_not_require_prompt(tmp_path: Path) -> None:
    """Task 8B: --inspect-proposals runs the real detector path and never asks for --prompt."""

    image = tmp_path / "tile.png"
    image.write_bytes(b"not-an-image")
    completed = run_cli("predict.py", "--image", str(image), "--inspect-proposals")
    # no prompt error is raised (E101); the unreadable fixture fails later in the image loader
    assert "E101" not in completed.stderr
    assert completed.returncode == 20 and "E202" in completed.stderr


def test_predict_defaults_exact(tmp_path: Path) -> None:
    from buildreasonseg import DEFAULT_MODEL
    from buildreasonseg.inference.contract import DEFAULT_ALPHA
    import predict as predict_module

    parser = predict_module.build_parser()
    args = parser.parse_args(["--image", "x.tif", "--prompt", "p"])
    assert args.model == DEFAULT_MODEL
    assert args.device == "auto"
    assert args.alpha == DEFAULT_ALPHA == 0.45
    assert args.save_diagnostics is True
    assert args.reference_id is None
    assert args.inspect_proposals is False
    assert args.config == "configs/inference.yaml"


def test_predict_no_save_diagnostics_flag() -> None:
    import predict as predict_module

    args = predict_module.build_parser().parse_args(
        ["--image", "x.tif", "--prompt", "p", "--no-save-diagnostics"])
    assert args.save_diagnostics is False


def test_predict_missing_image_reports_e201() -> None:
    completed = run_cli("predict.py", "--image", "definitely_missing.tif", "--prompt", "p")
    assert completed.returncode == 20
    assert "E201" in completed.stderr


def test_predict_unsupported_image_type_reports_e203(tmp_path: Path) -> None:
    weird = tmp_path / "tile.xyz"
    weird.write_bytes(b"data")
    completed = run_cli("predict.py", "--image", str(weird), "--prompt", "p")
    assert completed.returncode == 20
    assert "E203" in completed.stderr


def test_train_enum_constraints() -> None:
    bad_stage = run_cli("train.py", "--dataset", "datasets/x", "--name", "n", "--stage", "wrong")
    assert bad_stage.returncode == 2
    bad_init = run_cli("train.py", "--dataset", "datasets/x", "--name", "n", "--init", "wrong")
    assert bad_init.returncode == 2
    import train as train_module

    args = train_module.build_parser().parse_args(
        ["--dataset", "datasets/x", "--name", "n"])
    assert (args.stage, args.init, args.device) == ("all", "default", "auto")
    assert args.config == "configs/train.yaml"


def test_train_missing_dataset_reports_e101() -> None:
    completed = run_cli("train.py", "--dataset", "datasets/definitely_missing", "--name", "n")
    assert completed.returncode == 10
    assert "E101" in completed.stderr


def test_prepare_dataset_format_enum() -> None:
    bad = run_cli("prepare_dataset.py", "--dataset", "datasets/x", "--format", "nope")
    assert bad.returncode == 2
    import prepare_dataset as module

    for fmt in ("coco", "instance-mask", "vector"):
        args = module.build_parser().parse_args(["--dataset", "datasets/x", "--format", fmt])
        assert args.format == fmt
        assert args.overwrite is False


def test_evaluate_split_and_reference_defaults() -> None:
    import evaluate as module

    args = module.build_parser().parse_args(["--dataset", "datasets/x"])
    assert args.split == "val"
    assert args.reference_mode == "both"
    assert args.device == "auto"
    assert args.model == "buildreasonseg_advisor"


def test_evaluate_test_requires_explicit_split() -> None:
    import evaluate as module

    default_args = module.build_parser().parse_args(["--dataset", "datasets/x"])
    assert default_args.split != "test"
    explicit = module.build_parser().parse_args(["--dataset", "datasets/x", "--split", "test"])
    assert explicit.split == "test"
    bad = run_cli("evaluate.py", "--dataset", "datasets/x", "--split", "validation")
    assert bad.returncode == 2


def test_cli_placeholders_return_not_implemented(tmp_path: Path) -> None:
    """Valid requests must return the Task 8A marker with a non-zero exit code, never fake output."""

    dataset = tmp_path / "MyDataset"
    (dataset / "raw").mkdir(parents=True)
    prepare = run_cli("prepare_dataset.py", "--dataset", str(dataset), "--format", "coco")
    assert prepare.returncode == 90
    assert "E900" in prepare.stderr and "NOT_IMPLEMENTED" in prepare.stderr.upper()
    evaluate = run_cli("evaluate.py", "--dataset", str(dataset))
    assert evaluate.returncode == 90
    train = run_cli("train.py", "--dataset", str(dataset), "--name", "rc1_smoke")
    assert train.returncode == 90


def test_error_registry_complete() -> None:
    from buildreasonseg.errors import registry

    codes = registry()
    for code in ("E101", "E102", "E201", "E202", "E203", "E301", "E302", "E303", "E304", "E401",
                 "E402", "E403", "E404", "E501", "E502", "E900", "E901"):
        assert code in codes, code
        assert codes[code]["message"] and codes[code]["suggestion"]

class _FakeResult:
    """Minimal stand-in for a PipelineResult as consumed by predict.py."""

    def __init__(self, payload, ok=True, error_code=None, error_reason=None):
        self.result_payload = payload
        self.ok = ok
        self.error_code = error_code
        self.error_reason = error_reason


def _success_payload():
    from buildreasonseg.runtime.pipeline import success_semantics
    return {
        "status": "SUCCESS",
        **success_semantics(),
        "output_paths": {"mask": "mask.png", "overlay": "overlay.png", "diagnostics": "diag"},
        "reference_id": 123,
        "mask_area": 7,
    }


def _args(**overrides):
    import argparse
    base = dict(input_dir=".", prompt="q", device="cpu", alpha=1.0, save_diagnostics=False,
                reference_id=None, inspect_proposals=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _expected_single_success_lines():
    return [
        "Result       : SUCCESS",
        "Validity     : RUNTIME_STRUCTURAL_ONLY",
        "Semantic     : NOT_EVALUATED",
        "Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.",
        "",
        "Mask         : mask.png",
        "Overlay      : overlay.png",
        "Diagnostics  : diag",
        "Reference ID : 123",
        "Mask area    : 7",
    ]


def test_r1a_single_success_semantics_block(capsys):
    """Single-image SUCCESS prints the four frozen semantic lines plus the artifact lines, exactly once each."""
    import predict
    predict._report_single(_FakeResult(_success_payload()), _args())
    out = capsys.readouterr().out.splitlines()
    expected = _expected_single_success_lines()
    assert out == expected, out
    for line in expected[:4]:
        assert out.count(line) == 1, line


def test_r1a_single_failure_omits_success_semantics(capsys):
    """A failed single-image report keeps only the Result line on stdout and claims no semantics."""
    import predict
    predict._report_single(_FakeResult({"status": "FAILED", "detail": "mask_only_in_padding"}, ok=False,
                                      error_code="E404", error_reason="mask_only_in_padding"), _args())
    captured = capsys.readouterr()
    assert captured.out.count("Result       : FAILED") == 1
    assert captured.out.count("Validity     :") == 0
    assert captured.out.count("Semantic     :") == 0
    assert captured.out.count("Note         :") == 0


def test_r1a_batch_success_annotation_and_runtime_summary(capsys, tmp_path, monkeypatch):
    """_run_batch uses the real _batch_files() discovery path; only predict_one is monkeypatched."""
    import time
    import predict
    for name in ("one.png", "two.png"):
        (tmp_path / name).write_bytes(b"x")
    assert len(predict._batch_files(tmp_path)) == 2
    monkeypatch.setattr(predict, "predict_one", lambda runtime, request: _FakeResult(_success_payload()), raising=True)
    exit_code = predict._run_batch(None, _args(input_dir=tmp_path), "pkg", object(), {}, time.time())
    out = capsys.readouterr().out
    assert out.count("SUCCESS [runtime-only; semantic=NOT_EVALUATED]") == 2, out
    assert out.count("Runtime success: 2") == 1, out
    assert out.count("Failed : 0") == 1, out
    assert exit_code == 0
