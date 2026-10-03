"""Task 8B.3-R2 — deterministic interactive driver for the frozen six-image RC1 Demo suite.

Validation harness only (never shipped to the RC1 delivery). ChatGPT-frozen design:

* binary pipes (`text=False`, `bufsize=0`); a daemon reader thread performs the blocking reads and pushes raw byte
  chunks plus one EOF sentinel into a `queue.Queue`; the main thread consumes it with a bounded wait, owns the
  wall-clock deadline and makes every Y/N decision (`readline`, `for line in pipe` and `communicate` are unused);
* incremental UTF-8 decoding, prompts recognised without requiring a newline, each prompt answered once;
* `child_environment()` copies `os.environ` and overrides exactly three variables;
* per-case wall-clock timeout of 900 s that fires even when the child emits no output at all.

    python scripts/task8b3_interactive_suite.py
"""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_RC1_ROOT = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
ENV_PYTHON = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda"
                  r"\buildreasonseg-mvp\python.exe")
TRANSCRIPT_DIR = EXTERNAL_RC1_ROOT / "logs" / "task8b3_transcripts"
RESULTS_PATH = EXTERNAL_RC1_ROOT / "logs" / "task8b3_suite_results.json"
CASE_TIMEOUT_SECONDS = 900.0
QUEUE_WAIT_SECONDS = 0.10
EOF_SENTINEL = object()

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

#: the six frozen cases (test-oracle data; prompts are never altered or retried)
CASES = [
    ("A1", "inference/input/A1.png", "找出最大的建筑，然后把它右边离它最近的那栋分割出来",
     "largest_to_right_of_to_nearest"),
    ("A2", "inference/input/A2.png", "以面积最大的建筑为参考，分割它左侧最近的建筑",
     "largest_to_left_of_to_nearest"),
    ("A3", "inference/input/A3.png", "以最大建筑为准，分割位于其上方且距离最近的建筑",
     "largest_to_above_of_to_nearest".replace("_above_of_", "_above_to_")),
    ("A4", "inference/input/A4.png", "请分割最大建筑下方距离最近的一栋建筑",
     "largest_to_below_to_nearest"),
    ("B1", "inference/input/B1.tif", "请找出面积最大的建筑，并分割它右边离它最近的那栋楼。",
     "largest_to_right_of_to_nearest"),
    ("B2", "inference/input/B2.tif", "最大建筑物的上面，离它最近的那一栋是什么，分割出来",
     "largest_to_above_to_nearest"),
]


def child_environment() -> dict:
    """Copy the parent environment and override exactly the three frozen variables."""

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    return environment


def _reader_thread(process: subprocess.Popen, channel: "queue.Queue") -> threading.Thread:
    """Blocking binary reads → queue; never decodes and never decides anything."""

    def pump() -> None:
        stream = process.stdout
        try:
            while True:
                chunk = stream.read(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                channel.put(chunk)
        except Exception:
            pass
        finally:
            channel.put(EOF_SENTINEL)

    thread = threading.Thread(target=pump, name="rc1-stdout-reader", daemon=True)
    thread.start()
    return thread


def _run_interactive_process(argv: list[str], *, cwd: Path, environment: dict,
                             expected_program: str, case_id: str,
                             timeout_seconds: float = CASE_TIMEOUT_SECONDS) -> dict:
    """Drive one child process: bounded queue waits, main-thread deadline, frozen Y/N rules.

    Shared by the six frozen cases and by the harness unit tests (fake children).
    """

    started = time.monotonic()
    process = subprocess.Popen(argv, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=False, bufsize=0, env=environment)
    channel: "queue.Queue" = queue.Queue()
    _reader_thread(process, channel)

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    transcript = ""
    answered: set[str] = set()
    initial_program: str | None = None
    confidence: float | None = None
    suggested_program: str | None = None
    decision: str | None = None
    language_status: str | None = None
    timed_out = False
    eof = False
    deadline = started + timeout_seconds

    def answer(kind: str, text: str, status: str) -> None:
        nonlocal decision, language_status
        if kind in answered:
            return
        answered.add(kind)
        decision, language_status = text, status
        transcript_local = f"[driver] {kind} -> {text.strip()} ({status})\n"
        try:
            process.stdin.write(text.encode("utf-8"))  # type: ignore[union-attr]
            process.stdin.flush()  # type: ignore[union-attr]
        except Exception:
            pass
        nonlocal_transcript.append(transcript_local)

    nonlocal_transcript: list[str] = []

    while True:
        try:
            item = channel.get(timeout=QUEUE_WAIT_SECONDS)
        except queue.Empty:
            item = None
        if item is EOF_SENTINEL:
            eof = True
        elif isinstance(item, bytes):
            decoded = decoder.decode(item)
            if decoded:
                buffer += decoded
                transcript += decoded
        if nonlocal_transcript:
            transcript += "".join(nonlocal_transcript)
            nonlocal_transcript.clear()
        if item is not None and not isinstance(item, bytes):
            break
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
            answer("suggestion", "Y\n" if suggested_program == expected_program else "N\n",
                   "FALLBACK_CORRECT" if suggested_program == expected_program else "FALLBACK_WRONG")
        if DIRECT_PROMPT in buffer and "direct" not in answered:
            answer("direct", "Y\n" if initial_program == expected_program else "N\n",
                   "DIRECT_CORRECT" if initial_program == expected_program
                   else "LANGUAGE_ERROR_SUPPORTED_WRONG")
        if FALLBACK_PROMPT in buffer and "fallback" not in answered:
            answer("fallback", "N\n", "LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST")
        if time.monotonic() > deadline and process.poll() is None:
            timed_out = True
            language_status = language_status or "DRIVER_TIMEOUT"
            process.terminate()
            waited = 0.0
            while process.poll() is None and waited < 10.0:
                time.sleep(0.1)
                waited += 0.1
            if process.poll() is None:
                process.kill()
        if eof and process.poll() is not None:
            break
        if timed_out and process.poll() is not None:
            break

    drain_deadline = time.monotonic() + 2.0
    while time.monotonic() < drain_deadline:
        try:
            item = channel.get(timeout=0.05)
        except queue.Empty:
            break
        if isinstance(item, bytes):
            decoded = decoder.decode(item)
            buffer += decoded
            transcript += decoded
    transcript += decoder.decode(b"", final=True)
    try:
        process.wait(timeout=30)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass

    if language_status is None:
        if initial_program == expected_program:
            language_status = "DIRECT_CORRECT"
        elif initial_program in DISPLAY_TO_PROGRAM.values():
            language_status = "LANGUAGE_ERROR_SUPPORTED_WRONG"
        else:
            language_status = "NO_SAFE_SUGGESTION"
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPT_DIR / f"{case_id}.txt").write_text(
        f"driver decision: {decision} | language_status: {language_status}\n"
        f"--- exit {process.returncode} ---\n{transcript}\n", encoding="utf-8")
    return {"decision": decision, "language_status": language_status,
            "initial_program": initial_program, "confidence": confidence,
            "suggested_program": suggested_program, "transcript": transcript,
            "exit_code": process.returncode, "timed_out": timed_out,
            "seconds": round(time.monotonic() - started, 1), "process": process}


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
    before = snapshot_outputs()
    outcome = _run_interactive_process(
        [str(ENV_PYTHON), "predict.py", "--image", image, "--prompt", prompt,
         "--confirm-command"],
        cwd=EXTERNAL_RC1_ROOT, environment=child_environment(),
        expected_program=expected_program, case_id=case_id)
    after = snapshot_outputs()
    added = {sub: sorted(set(after[sub]) - set(before[sub])) for sub in after}
    result_json = None
    for name in added["diagnostics"]:
        if name.endswith("result.json"):
            result_json = json.loads(
                (EXTERNAL_RC1_ROOT / "inference" / "output" / "diagnostics" / name)
                .read_text(encoding="utf-8"))
    result_line = RESULT_RE.search(outcome["transcript"])
    return {
        "id": case_id, "file": image, "prompt": prompt, "expected_program": expected_program,
        "initial_program": outcome["initial_program"],
        "initial_confidence": outcome["confidence"],
        "suggested_program": outcome["suggested_program"],
        "yn_sent": outcome["decision"], "language_status": outcome["language_status"],
        "executed_visual": outcome["decision"] == "Y\n",
        "exit_code": outcome["exit_code"],
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
        "new_artifacts": added, "seconds": outcome["seconds"], "timeout": outcome["timed_out"],
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
    identities = input_identity()
    for entry in identities:
        print("[8b3-r2]", json.dumps(entry, ensure_ascii=False), flush=True)
    if [entry["id"] for entry in identities if not entry["exists"]]:
        print("[8b3-r2] STOP: missing inputs", flush=True)
        return 2
    if not environment_ready():
        print("[8b3-r2] STOP: check_setup not READY", flush=True)
        return 3
    results = []
    for case_id, image, prompt, expected in CASES:
        print(f"[8b3-r2] {case_id} starting …", flush=True)
        record = run_case(case_id, image, prompt, expected)
        record["inputs"] = identities
        results.append(record)
        print(f"[8b3-r2] {case_id} language={record['language_status']} yn={record['yn_sent']} "
              f"result={record['result_status']} error={record['error_code']} "
              f"mask={record['mask_area']} tiles={record['tile_count']} "
              f"({record['seconds']}s)", flush=True)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        if record["timeout"] or not environment_ready():
            print(f"[8b3-r2] environment unhealthy after {case_id}; stopping", flush=True)
            break
    print("[8b3-r2] suite finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
