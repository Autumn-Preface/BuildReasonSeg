"""Task 8B.3-R1 — deterministic interactive driver for the frozen six-image RC1 Demo suite.

Validation harness only (never shipped to the RC1 delivery). It invokes the **real** external RC1 CLI once per
frozen case and answers the product's Y/N confirmations strictly from the observed parse:

* binary pipes (`text=False`, `bufsize=0`), incremental UTF-8 decoder, no `readline()` / `for line` / `communicate`;
* a prompt is answered only when its exact known substring appears (no trailing newline required), once each;
* per-case wall-clock timeout = 15 minutes, then terminate/kill, classify `DRIVER_TIMEOUT`, keep the transcript,
  re-run `check_setup.py` and continue only if the environment is still READY.

    python scripts/task8b3_interactive_suite.py
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_RC1_ROOT = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
ENV_PYTHON = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda"
                  r"\buildreasonseg-mvp\python.exe")
TRANSCRIPT_DIR = EXTERNAL_RC1_ROOT / "logs" / "task8b3_transcripts"
RESULTS_PATH = EXTERNAL_RC1_ROOT / "logs" / "task8b3_suite_results.json"
CASE_TIMEOUT_SECONDS = 15 * 60

DIRECT_PROMPT = "是否按此理解执行？ [Y/N]: "
SUGGESTION_PROMPT = "是否使用建议指令继续？ [Y/N]: "
FALLBACK_PROMPT = "是否进入有限兼容模式？ [Y/N]: "
PROGRAM_RE = re.compile(r"\[解析\].*?\(([^()\r\n]+)\)")
CONFIDENCE_RE = re.compile(r"置信度\s*([0-9.]+)")
RESULT_RE = re.compile(r"Result\s*:\s*(\w+)")
DISPLAY_TO_PROGRAM = {"largest -> left_of -> nearest": "largest_to_left_of_to_nearest",
                      "largest -> right_of -> nearest": "largest_to_right_of_to_nearest",
                      "largest -> above -> nearest": "largest_to_above_to_nearest",
                      "largest -> below -> nearest": "largest_to_below_to_nearest"}

#: the six frozen cases of Task 8B.3 (test-oracle data; prompts are never altered or retried)
CASES = [
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


def child_environment() -> dict:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def snapshot_outputs() -> dict:
    result = {}
    for sub in ("masks", "overlays", "diagnostics"):
        directory = EXTERNAL_RC1_ROOT / "inference" / "output" / sub
        entries = {}
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    entries[str(path.relative_to(directory)).replace("\\", "/")] = [stat.st_size,
                                                                                   stat.st_mtime_ns]
        result[sub] = entries
    return result


def environment_ready() -> bool:
    completed = subprocess.run([str(ENV_PYTHON), "check_setup.py"], cwd=str(EXTERNAL_RC1_ROOT),
                               capture_output=True, encoding="utf-8", errors="replace")
    return "BuildReasonSeg environment: READY" in (completed.stdout or "")


def run_case(case_id: str, image: str, prompt: str, expected_program: str) -> dict:
    """Run one frozen case through the real CLI; answer Y/N only from the observed program."""

    environment = child_environment()
    before = snapshot_outputs()
    started = time.time()
    process = subprocess.Popen(
        [str(ENV_PYTHON), "predict.py", "--image", image, "--prompt", prompt, "--confirm-command"],
        cwd=str(EXTERNAL_RC1_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=False, bufsize=0, env=environment)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    transcript = ""
    answered: set[str] = set()
    initial_program = None
    confidence = None
    suggested_program = None
    decision = None
    language_status = None
    timed_out = False
    deadline = started + CASE_TIMEOUT_SECONDS

    def answer(kind: str, text: str, status: str) -> None:
        nonlocal decision, language_status, answered
        if kind in answered:
            return
        answered.add(kind)
        decision, language_status = text, status
        transcript_local = f"[driver] {kind} -> {text} ({status})\n"
        del transcript_local
        try:
            process.stdin.write(text.encode("utf-8"))  # type: ignore[union-attr]
            process.stdin.flush()  # type: ignore[union-attr]
        except Exception:
            pass

    while True:
        try:
            chunk = os.read(process.stdout.fileno(), 4096)  # type: ignore[union-attr]
        except OSError:
            break
        if not chunk:
            break
        decoded = decoder.decode(chunk)
        if decoded:
            buffer += decoded
            transcript += decoded
        match = PROGRAM_RE.search(buffer)
        if match and initial_program is None:
            initial_program = match.group(1).strip()
        conf = CONFIDENCE_RE.search(buffer)
        if conf and confidence is None:
            try:
                confidence = float(conf.group(1))
            except ValueError:
                confidence = None
        if SUGGESTION_PROMPT in buffer and "suggestion" not in answered:
            for display, program in DISPLAY_TO_PROGRAM.items():
                if display in buffer:
                    suggested_program = program
                    break
            if suggested_program == expected_program:
                answer("suggestion", "Y\n", "FALLBACK_CORRECT")
            else:
                answer("suggestion", "N\n", "FALLBACK_WRONG")
        if DIRECT_PROMPT in buffer and "direct" not in answered:
            if initial_program == expected_program:
                answer("direct", "Y\n", "DIRECT_CORRECT")
            else:
                answer("direct", "N\n", "LANGUAGE_ERROR_SUPPORTED_WRONG")
        if FALLBACK_PROMPT in buffer and "fallback" not in answered:
            answer("fallback", "N\n", "LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST")
        if time.time() > deadline:
            timed_out = True
            break

    if timed_out:
        process.terminate()
        time.sleep(10)
        if process.poll() is None:
            process.kill()
        language_status = language_status or "DRIVER_TIMEOUT"
    try:
        process.wait(timeout=600)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass

    transcript += decoder.decode(b"", final=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPT_DIR / f"{case_id}.txt").write_text(
        f"$ predict.py --image {image} --prompt {prompt} --confirm-command\n"
        f"driver decision: {decision} | language_status: {language_status}\n"
        f"--- exit {process.returncode} ---\n{transcript}\n", encoding="utf-8")

    after = snapshot_outputs()
    added = {sub: sorted(set(after[sub]) - set(before[sub])) for sub in after}
    result_json = None
    for name in added["diagnostics"]:
        if name.endswith("result.json"):
            result_json = json.loads(
                (EXTERNAL_RC1_ROOT / "inference" / "output" / "diagnostics" / name)
                .read_text(encoding="utf-8"))
    if language_status is None:
        if initial_program == expected_program:
            language_status = "DIRECT_CORRECT"
        elif initial_program in DISPLAY_TO_PROGRAM.values():
            language_status = "LANGUAGE_ERROR_SUPPORTED_WRONG"
        else:
            language_status = "NO_SAFE_SUGGESTION"
    result_line = RESULT_RE.search(transcript)
    return {
        "id": case_id, "file": image, "prompt": prompt, "expected_program": expected_program,
        "initial_program": initial_program, "initial_confidence": confidence,
        "suggested_program": suggested_program, "yn_sent": decision,
        "language_status": language_status, "executed_visual": decision == "Y\n",
        "exit_code": process.returncode,
        "cli_result_line": result_line.group(1) if result_line else None,
        "result_status": None if result_json is None else result_json.get("status"),
        "error_code": None if result_json is None else result_json.get("error_code"),
        "error_reason": None if result_json is None else result_json.get("reason"),
        "reference_id": None if result_json is None else result_json.get("reference_id"),
        "mask_area": None if result_json is None else result_json.get("mask_area"),
        "tile_count": None if result_json is None else result_json.get("tile_count"),
        "raw_proposal_count": None if result_json is None else result_json.get("raw_proposal_count"),
        "merged_proposal_count": None if result_json is None else
        result_json.get("merged_proposal_count"),
        "input_size": None if result_json is None else result_json.get("input_size"),
        "mask_path": None if result_json is None else (result_json.get("output_paths") or {}).get("mask"),
        "overlay_path": None if result_json is None else
        (result_json.get("output_paths") or {}).get("overlay"),
        "diagnostics_path": None if result_json is None else
        (result_json.get("output_paths") or {}).get("diagnostics"),
        "new_artifacts": added, "seconds": round(time.time() - started, 1),
        "timeout": timed_out,
    }


def input_identity() -> list[dict]:
    rows = []
    for case_id, image, _prompt, _expected in CASES:
        path = EXTERNAL_RC1_ROOT / image
        entry: dict = {"id": case_id, "file": image, "exists": path.is_file()}
        if path.is_file():
            from PIL import Image

            with Image.open(path) as handle:
                entry.update({"bytes": path.stat().st_size, "width": handle.width,
                              "height": handle.height, "mode": handle.mode,
                              "channels": len(handle.getbands()),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append(entry)
    return rows


def main() -> int:
    print("[8b3-r1] input identity:", flush=True)
    identities = input_identity()
    for entry in identities:
        print("   ", json.dumps(entry, ensure_ascii=False), flush=True)
    missing = [entry["id"] for entry in identities if not entry["exists"]]
    if missing:
        print("[8b3-r1] STOP: missing inputs", missing, flush=True)
        return 2
    if not environment_ready():
        print("[8b3-r1] STOP: check_setup not READY", flush=True)
        return 3

    results = []
    for case_id, image, prompt, expected in CASES:
        print(f"[8b3-r1] {case_id} starting …", flush=True)
        record = run_case(case_id, image, prompt, expected)
        record["inputs"] = identities
        results.append(record)
        print(f"[8b3-r1] {case_id} language={record['language_status']} yn={record['yn_sent']} "
              f"result={record['result_status']} error={record['error_code']} "
              f"mask={record['mask_area']} tiles={record['tile_count']} "
              f"({record['seconds']}s)", flush=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        if record["timeout"] or not environment_ready():
            print(f"[8b3-r1] environment not healthy after {case_id}; stopping suite", flush=True)
            break
    print("[8b3-r1] suite finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
