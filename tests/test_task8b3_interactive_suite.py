"""Task 8B.3-R1 tests for the deterministic interactive driver.

Fake child scripts only: no real models, images, internet, final-test data or external delivery access.
"""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import task8b3_interactive_suite as driver  # noqa: E402


# ---------------------------------------------------------------- helper: fake child


def _fake_child(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_child.py"
    script.write_text(
        "import sys\n"
        "def ask(prompt):\n"
        "    sys.stdout.write(prompt)\n"
        "    sys.stdout.flush()\n"
        "    return sys.stdin.readline().strip()\n"
        + body, encoding="utf-8")
    return script


def _run_fake(script: Path, tmp_path: Path,
              expected: str = "largest_to_right_of_to_nearest") -> tuple[str, str]:
    """Run a fake child exactly like the driver does (binary pipes, incremental UTF-8)."""

    environment = os.environ.copy()
    environment.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
    process = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False,
                               bufsize=0, env=environment)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    transcript = ""
    answers: list[str] = []
    answers.append  # noqa: B018 - keep list handle obvious
    deadline = time.time() + 30
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        text = decoder.decode(chunk)
        buffer += text
        transcript += text
        if driver.DIRECT_PROMPT in buffer and "direct" not in answers:
            answers.append("direct")
            decision = "Y\n" if "largest_to_right_of_to_nearest" in buffer else "N\n"
            process.stdin.write(decision.encode("utf-8"))
            process.stdin.flush()
            transcript += f"[driver] {decision.strip()}\n"
        if time.time() > deadline:
            break
    process.wait(timeout=30)
    return transcript, "".join(answers)


# ---------------------------------------------------------------- tests


def test_direct_prompt_without_newline_is_detected(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('[解析] largest -> right_of -> nearest   "
                                   "(largest_to_right_of_to_nearest)')\n"
                                   "answer = ask('是否按此理解执行？ [Y/N]: ')\n"
                                   "print('ANSWER=' + answer)\n")
    transcript, answered = _run_fake(script, tmp_path)
    assert answered == "direct"
    assert "ANSWER=Y" in transcript


def test_utf8_chinese_prompt_through_pipe(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "ask('是否按此理解执行？ [Y/N]: ')\n")
    transcript, _ = _run_fake(script, tmp_path)
    assert driver.DIRECT_PROMPT.strip() in transcript


def test_correct_direct_program_sends_y(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('[解析] largest -> left_of -> nearest   "
                                   "(largest_to_left_of_to_nearest)')\n"
                                   "print('ANSWER=' + ask('是否按此理解执行？ [Y/N]: '))\n")
    transcript, _ = _run_fake(script, tmp_path)
    assert "ANSWER=Y" in transcript


def test_wrong_direct_program_sends_n(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "print('[解析] largest -> below -> nearest   "
                                   "(largest_to_below_to_nearest)')\n"
                                   "print('ANSWER=' + ask('是否按此理解执行？ [Y/N]: '))\n")
    transcript, _ = _run_fake(script, tmp_path)
    assert "ANSWER=N" in transcript


def test_suggestion_prompt_supported_display_maps_to_program() -> None:
    assert driver.DISPLAY_TO_PROGRAM["largest -> above -> nearest"] == \
        "largest_to_above_to_nearest"
    assert len(driver.DISPLAY_TO_PROGRAM) == 4


def test_fallback_prompt_constant_exact() -> None:
    assert driver.FALLBACK_PROMPT == "是否进入有限兼容模式？ [Y/N]: "
    assert driver.SUGGESTION_PROMPT == "是否使用建议指令继续？ [Y/N]: "


def test_same_prompt_answered_only_once() -> None:
    answered: set[str] = set()

    def answer(kind: str) -> int:
        if kind in answered:
            return 0
        answered.add(kind)
        return 1

    assert answer("direct") == 1
    assert answer("direct") == 0


def test_transcript_complete_without_newline(tmp_path: Path) -> None:
    script = _fake_child(tmp_path, "sys.stdout.write('NO-NEWLINE-TAIL')\n"
                                   "sys.stdout.flush()\n"
                                   "ask('是否按此理解执行？ [Y/N]: ')\n")
    transcript, _ = _run_fake(script, tmp_path)
    assert "NO-NEWLINE-TAIL" in transcript


def test_decision_path_has_no_readline(tmp_path: Path, monkeypatch) -> None:
    source = (REPO_ROOT / "scripts" / "task8b3_interactive_suite.py").read_text(encoding="utf-8")
    assert "readline(" not in source
    assert "for line in" not in source
    assert "communicate(" not in source
    assert "codecs.getincrementaldecoder" in source


def test_suite_cases_exactly_six_in_frozen_order() -> None:
    assert [case[0] for case in driver.CASES] == ["A1", "A2", "A3", "A4", "B1", "B2"]


def test_frozen_prompts_and_programs_match_task_book() -> None:
    expected = [
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
    assert driver.CASES == expected


def test_child_environment_is_deterministic() -> None:
    environment = driver.child_environment()
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONUTF8"] == "1"


def test_prompt_strings_are_exact_product_strings() -> None:
    assert driver.DIRECT_PROMPT == "是否按此理解执行？ [Y/N]: "
    assert driver.PROGRAM_RE.search("[解析] largest -> right_of -> nearest   "
                                    "(largest_to_right_of_to_nearest)").group(1) == \
        "largest_to_right_of_to_nearest"
