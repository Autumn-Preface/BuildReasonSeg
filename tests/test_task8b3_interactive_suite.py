"""Task 8B.3-R2 harness tests for the deterministic interactive driver.

Fake child scripts only: no real models, images, internet, final-test data or external delivery access.
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import task8b3_interactive_suite as driver  # noqa: E402

DRIVER_SOURCE = REPO_ROOT / "scripts" / "task8b3_interactive_suite.py"


def _fake_child(tmp_path: Path, body: str, name: str = "fake_child.py") -> Path:
    script = tmp_path / name
    script.write_text(
        "import sys, time\n"
        "def ask(prompt):\n"
        "    sys.stdout.write(prompt)\n"
        "    sys.stdout.flush()\n"
        "    return sys.stdin.readline().strip()\n"
        + body, encoding="utf-8")
    return script


def _drive(script: Path, expected: str, tmp_path: Path, timeout: float = 20.0) -> dict:
    """Drive a fake child with the real helper (same code path as the six frozen cases)."""

    return driver._run_interactive_process(
        [sys.executable, str(script)], cwd=tmp_path, environment=driver.child_environment(),
        expected_program=expected, case_id=f"TEST_{script.stem}", timeout_seconds=timeout)


# ---------------------------------------------------------------- 1-3, 4: direct prompts


def test_right_expected_right_parse_sends_y(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('[解析] largest -> right_of -> nearest   "
                                   "(largest_to_right_of_to_nearest)')\n"
                                   "print('ANSWER=' + ask('是否按此理解执行？ [Y/N]: '))\n")
    outcome = _drive(script, "largest_to_right_of_to_nearest", tmp_path)
    assert "ANSWER=Y" in outcome["transcript"]
    assert outcome["language_status"] == "DIRECT_CORRECT"


def test_left_expected_left_parse_sends_y(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('[解析] largest -> left_of -> nearest   "
                                   "(largest_to_left_of_to_nearest)')\n"
                                   "print('ANSWER=' + ask('是否按此理解执行？ [Y/N]: '))\n")
    outcome = _drive(script, "largest_to_left_of_to_nearest", tmp_path)
    assert "ANSWER=Y" in outcome["transcript"]


def test_left_expected_right_parse_sends_n(tmp_path: Path) -> None:
    """Test defect A: the helper must use the `expected` argument, never a hard-coded A1 program."""

    script = _fake_child(tmp_path, "print('[解析] largest -> right_of -> nearest   "
                                   "(largest_to_right_of_to_nearest)')\n"
                                   "print('ANSWER=' + ask('是否按此理解执行？ [Y/N]: '))\n")
    outcome = _drive(script, "largest_to_left_of_to_nearest", tmp_path)
    assert "ANSWER=N" in outcome["transcript"]
    assert outcome["language_status"] == "LANGUAGE_ERROR_SUPPORTED_WRONG"


def test_direct_prompt_without_newline_is_detected(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('[解析] largest -> above -> nearest   "
                                   "(largest_to_above_to_nearest)')\n"
                                   "ask('是否按此理解执行？ [Y/N]: ')\n")
    outcome = _drive(script, "largest_to_above_to_nearest", tmp_path)
    assert outcome["decision"] == "Y\n"


# ---------------------------------------------------------------- 5: UTF-8 through binary pipe


def test_chinese_prompt_survives_binary_pipe(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "ask('是否按此理解执行？ [Y/N]: ')\n")
    outcome = _drive(script, "largest_to_right_of_to_nearest", tmp_path)
    assert driver.DIRECT_PROMPT.strip() in outcome["transcript"]


# ---------------------------------------------------------------- 6-7: suggestion prompts


def test_suggestion_above_expected_above_sends_y(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('当前解析结果不属于 RC1 已开放的四类空间推理语义。')\n"
                                   "print('建议程序：largest -> above -> nearest')\n"
                                   "print('ANSWER=' + ask('是否使用建议指令继续？ [Y/N]: '))\n")
    outcome = _drive(script, "largest_to_above_to_nearest", tmp_path)
    assert "ANSWER=Y" in outcome["transcript"]
    assert outcome["language_status"] == "FALLBACK_CORRECT"


def test_suggestion_above_expected_below_sends_n(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('当前解析结果不属于 RC1 已开放的四类空间推理语义。')\n"
                                   "print('建议程序：largest -> above -> nearest')\n"
                                   "print('ANSWER=' + ask('是否使用建议指令继续？ [Y/N]: '))\n")
    outcome = _drive(script, "largest_to_below_to_nearest", tmp_path)
    assert "ANSWER=N" in outcome["transcript"]
    assert outcome["language_status"] == "FALLBACK_WRONG"


# ---------------------------------------------------------------- 8: fallback prompt


def test_fallback_prompt_sends_n(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('语言模型不可用。')\n"
                                   "print('ANSWER=' + ask('是否进入有限兼容模式？ [Y/N]: '))\n")
    outcome = _drive(script, "largest_to_right_of_to_nearest", tmp_path)
    assert "ANSWER=N" in outcome["transcript"]
    assert outcome["language_status"] == "LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST"


# ---------------------------------------------------------------- 9-10


def test_repeated_prompt_is_answered_only_once(tmp_path: Path) -> None:
    """Fake child prints the direct prompt twice before a single stdin read (Task 8B.3-R4A §3.3)."""

    script = _fake_child(tmp_path, "print('[解析] largest -> right_of -> nearest   "
                                   "(largest_to_right_of_to_nearest)')\n"
                                   "sys.stdout.write('是否按此理解执行？ [Y/N]: ')\n"
                                   "sys.stdout.write('是否按此理解执行？ [Y/N]: ')\n"
                                   "sys.stdout.flush()\n"
                                   "answer = sys.stdin.readline().strip()\n"
                                   "print('ANSWER=' + answer)\n")
    outcome = _drive(script, "largest_to_right_of_to_nearest", tmp_path)
    assert outcome["timed_out"] is False
    assert outcome["exit_code"] == 0
    assert "ANSWER=Y" in outcome["transcript"]
    assert outcome["transcript"].count("[driver] direct -> Y (DIRECT_CORRECT)") == 1


def test_no_newline_output_is_preserved(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "sys.stdout.write('TAIL-NO-NEWLINE')\n"
                                   "sys.stdout.flush()\n"
                                   "ask('是否按此理解执行？ [Y/N]: ')\n")
    outcome = _drive(script, "largest_to_right_of_to_nearest", tmp_path)
    assert "TAIL-NO-NEWLINE" in outcome["transcript"]


# ---------------------------------------------------------------- 11: AST-based check (defect B)


def test_no_executable_readline_or_communicate_in_driver() -> None:
    """Task 8B.3-R4A §3.4: walk only the two interaction functions, no docstring scanning."""

    tree = ast.parse(DRIVER_SOURCE.read_text(encoding="utf-8"))
    targets = {"_reader_thread", "_run_interactive_process"}
    found: set[str] = set()
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            found.add(node.name)
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Attribute) and func.attr in {"readline", "communicate"}:
                        forbidden.append(f"{node.name}:{func.attr}")
                    if isinstance(func, ast.Name) and func.id in {"readline", "communicate"}:
                        forbidden.append(f"{node.name}:{func.id}")
    assert found == targets
    assert forbidden == []


def test_changed_outputs_new_path_is_changed() -> None:
    before = {"masks": {}, "overlays": {}, "diagnostics": {}}
    after = {"masks": {"a_mask.png": [10, 1]}, "overlays": {}, "diagnostics": {}}
    assert driver.changed_outputs(before, after)["masks"] == ["a_mask.png"]


def test_changed_outputs_same_metadata_is_unchanged() -> None:
    snapshot = {"masks": {"a_mask.png": [10, 1]}, "overlays": {}, "diagnostics": {}}
    assert driver.changed_outputs(snapshot, snapshot)["masks"] == []


def test_changed_outputs_different_size_is_changed() -> None:
    before = {"masks": {"a_mask.png": [10, 1]}, "overlays": {}, "diagnostics": {}}
    after = {"masks": {"a_mask.png": [12, 1]}, "overlays": {}, "diagnostics": {}}
    assert driver.changed_outputs(before, after)["masks"] == ["a_mask.png"]


def test_changed_outputs_same_size_new_mtime_is_changed() -> None:
    before = {"masks": {"a_mask.png": [10, 1]}, "overlays": {}, "diagnostics": {}}
    after = {"masks": {"a_mask.png": [10, 2]}, "overlays": {}, "diagnostics": {}}
    assert driver.changed_outputs(before, after)["masks"] == ["a_mask.png"]


def test_changed_outputs_overwritten_result_json_is_detected() -> None:
    """A pre-existing diagnostics path whose metadata changed must be reported for this run."""

    before = {"masks": {}, "overlays": {}, "diagnostics": {"A1/result.json": [100, 1]}}
    after = {"masks": {}, "overlays": {}, "diagnostics": {"A1/result.json": [120, 5],
                                                          "A1/maps.npz": [7, 5]}}
    changed = driver.changed_outputs(before, after)
    assert changed["diagnostics"] == ["A1/maps.npz", "A1/result.json"]


# ---------------------------------------------------------------- 12: silent-child timeout


def test_silent_child_timeout_regression(tmp_path: Path) -> None:
    """Fake child emits nothing and sleeps; the driver must time out well under 5 s."""

    script = _fake_child(tmp_path, "time.sleep(60)\n")
    started = time.monotonic()
    outcome = _drive(script, "largest_to_right_of_to_nearest", tmp_path, timeout=0.5)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, elapsed
    assert outcome["timed_out"] is True
    assert outcome["language_status"] == "DRIVER_TIMEOUT"
    assert outcome["process"].poll() is not None  # child not left running


# ---------------------------------------------------------------- 13-14: environment scope


def test_child_environment_overrides_only_three_variables(monkeypatch) -> None:
    monkeypatch.setenv("RC1_SENTINEL_KEEP", "keep-me")
    environment = driver.child_environment()
    assert environment["RC1_SENTINEL_KEEP"] == "keep-me"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONUTF8"] == "1"
    expected = {"PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONUTF8"}
    changed = {key for key in expected if environment.get(key) != os.environ.get(key)}
    assert changed == expected


def test_harness_does_not_set_offline_flags() -> None:
    tree = ast.parse(DRIVER_SOURCE.read_text(encoding="utf-8"))
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    assigned.add(str(target.slice.value))
    assert "HF_HUB_OFFLINE" not in assigned
    assert "TRANSFORMERS_OFFLINE" not in assigned


# ---------------------------------------------------------------- 15: frozen suite


def test_suite_cases_and_prompts_frozen() -> None:
    assert [case[0] for case in driver.CASES] == ["A1", "A2", "A3", "A4", "B1", "B2"]
    assert driver.CASES == [
        ("A1", "inference/input/A1.png", "找出最大的建筑，然后把它右边离它最近的那栋分割出来",
         "largest_to_right_of_to_nearest"),
        ("A2", "inference/input/A2.png", "以面积最大的建筑为参考，分割它左侧最近的建筑",
         "largest_to_left_of_to_nearest"),
        ("A3", "inference/input/A3.png", "以最大建筑为准，分割位于其上方且距离最近的建筑",
         "largest_to_above_to_nearest"),
        ("A4", "inference/input/A4.png", "请分割最大建筑下方距离最近的一栋建筑",
         "largest_to_below_to_nearest"),
        ("B1", "inference/input/B1.tif", "请找出面积最大的建筑，并分割它右边离它最近的那栋楼。",
         "largest_to_right_of_to_nearest"),
        ("B2", "inference/input/B2.tif", "最大建筑物的上面，离它最近的那一栋是什么，分割出来",
         "largest_to_above_to_nearest"),
    ]
    assert driver.CASE_TIMEOUT_SECONDS == 900
    assert driver.DIRECT_PROMPT == "是否按此理解执行？ [Y/N]: "
    assert driver.SUGGESTION_PROMPT == "是否使用建议指令继续？ [Y/N]: "
    assert driver.FALLBACK_PROMPT == "是否进入有限兼容模式？ [Y/N]: "
    assert len(driver.DISPLAY_TO_PROGRAM) == 4
