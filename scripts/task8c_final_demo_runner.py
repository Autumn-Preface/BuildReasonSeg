"""One-shot Task 8C execution harness. No ground-truth access or model-side decisions.

The CLI has no candidate/parameter overrides. The only extra predict flag is
--confirm-command, which enables the Supervisor-required Y/N interaction.
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
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
EVIDENCE = REPO / "evaluation/task8c_final_demo_v1.json"
CASE_TIMEOUT_SECONDS = 900.0
DIRECT_PROMPT = "是否按此理解执行？ [Y/N]: "
SUGGESTION_PROMPT = "是否使用建议指令继续？ [Y/N]: "
FALLBACK_PROMPT = "是否进入有限兼容模式？ [Y/N]: "
DISPLAY_TO_PROGRAM = {
    f"largest -> {direction} -> nearest": f"largest_to_{relation}_to_nearest"
    for direction, relation in (("right_of", "right_of"), ("left_of", "left_of"),
                               ("above", "above"), ("below", "below"))
}
PROGRAM_RE = re.compile(r"\[解析\].*?\(([^()\r\n]+)\)")
CONFIDENCE_RE = re.compile(r"置信度\s*([0-9.]+)")
SUGGESTION_RE = re.compile(r"建议程序：\s*([^\r\n]+)")


@dataclass(frozen=True)
class Candidate:
    relation: str
    sample_id: str
    image: str
    image_sha256: str
    prompt: str
    expected_program: str


IMAGE_BASE = r"C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image"
CANDIDATES = (
    Candidate("right", "buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91",
              IMAGE_BASE + r"\1010.tif", "1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2",
              "最大建筑右侧最近的建筑", "largest_to_right_of_to_nearest"),
    Candidate("left", "buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3",
              IMAGE_BASE + r"\1003.tif", "eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38",
              "最大建筑左侧最近的建筑", "largest_to_left_of_to_nearest"),
    Candidate("above", "buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314",
              IMAGE_BASE + r"\1008.tif", "0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd",
              "最大建筑上方最近的建筑", "largest_to_above_to_nearest"),
    Candidate("below", "buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450",
              IMAGE_BASE + r"\1009.tif", "c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7",
              "最大建筑下方最近的建筑", "largest_to_below_to_nearest"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def inventory(root: Path) -> list[dict]:
    if not root.is_dir():
        raise ValueError(f"missing inventory root: {root}")
    return [{"path": str(p.relative_to(root)).replace("\\", "/"),
             "bytes": p.stat().st_size, "sha256": sha256_file(p),
             "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(root.rglob("*")) if p.is_file()]


def inventory_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_inventory(root: Path, rows: list[dict], *, allow_new: bool = False) -> None:
    actual = inventory(root)
    old = {r["path"]: r for r in rows}
    now = {r["path"]: r for r in actual}
    changed = [p for p in old if now.get(p) != old[p]]
    if changed or (not allow_new and old.keys() != now.keys()):
        raise ValueError(f"inventory changed: {root}: {changed or sorted(now.keys() ^ old.keys())}")


def write_json(path: Path, data: dict, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def formal_argv(candidate: Candidate) -> list[str]:
    return [sys.executable, "-B", str(EXTERNAL / "predict.py"), "--image", candidate.image,
            "--prompt", candidate.prompt, "--confirm-command"]


def decide(kind: str, program: str | None, expected: str) -> tuple[str, str]:
    if kind == "fallback":
        return "N", "LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST"
    correct = program == expected
    if kind == "direct":
        return ("Y", "DIRECT_CORRECT") if correct else ("N", "LANGUAGE_ERROR_SUPPORTED_WRONG")
    if kind == "suggestion":
        return ("Y", "SUGGESTION_CORRECT") if correct else ("N", "SUGGESTION_WRONG")
    raise ValueError(kind)


def drive_child(argv: list[str], cwd: Path, expected: str,
                timeout: float = CASE_TIMEOUT_SECONDS) -> dict:
    """Binary pipe reader owns no writes; the main thread enforces the wall deadline."""
    start = time.monotonic()
    child = subprocess.Popen(argv, cwd=str(cwd), env=os.environ.copy(), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    chunks: queue.Queue = queue.Queue()

    def reader() -> None:
        try:
            while True:
                block = child.stdout.read(4096)
                if not block:
                    break
                chunks.put(block)
        finally:
            chunks.put(None)

    threading.Thread(target=reader, daemon=True).start()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    transcript = ""
    answered: set[str] = set()
    decisions = []
    timed_out = False
    eof = False
    while not eof:
        if not timed_out and time.monotonic() - start >= timeout:
            timed_out = True
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        try:
            block = chunks.get(timeout=0.1)
        except queue.Empty:
            if timed_out and child.poll() is not None:
                # The pipe normally drains immediately; a stuck reader is a harness error.
                if time.monotonic() - start > timeout + 30:
                    raise RuntimeError("pipe did not close after timeout")
            continue
        if block is None:
            transcript += decoder.decode(b"", final=True)
            eof = True
            continue
        transcript += decoder.decode(block)
        for kind, marker in (("direct", DIRECT_PROMPT), ("suggestion", SUGGESTION_PROMPT),
                             ("fallback", FALLBACK_PROMPT)):
            if marker not in transcript or kind in answered or timed_out:
                continue
            initial = PROGRAM_RE.search(transcript)
            suggestion = SUGGESTION_RE.search(transcript)
            program = initial.group(1) if kind == "direct" and initial else None
            if kind == "suggestion" and suggestion:
                program = DISPLAY_TO_PROGRAM.get(suggestion.group(1).strip())
            answer, label = decide(kind, program, expected)
            child.stdin.write((answer + "\n").encode("utf-8"))
            child.stdin.flush()
            answered.add(kind)
            decisions.append({"kind": kind, "observed_program": program, "answer": answer,
                              "language_status": label})
    try:
        code = child.wait(timeout=max(0.001, timeout - (time.monotonic() - start)))
    except subprocess.TimeoutExpired:
        timed_out = True
        child.terminate()
        try:
            code = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait(timeout=10)
    child.stdin.close()
    child.stdout.close()
    initial = PROGRAM_RE.search(transcript)
    confidence = CONFIDENCE_RE.search(transcript)
    suggestion = SUGGESTION_RE.search(transcript)
    status = decisions[-1]["language_status"] if decisions else "LANGUAGE_NOT_CONFIRMED"
    return {"argv": argv, "exit_code": code, "timed_out": timed_out,
            "wall_seconds": time.monotonic() - start, "timeout_seconds": timeout,
            "stdout": transcript, "initial_program": initial.group(1) if initial else None,
            "initial_confidence": float(confidence.group(1)) if confidence else None,
            "suggested_program": DISPLAY_TO_PROGRAM.get(suggestion.group(1).strip()) if suggestion else None,
            "driver_decision": decisions, "language_status": status,
            "language_expected_program_executed": status in ("DIRECT_CORRECT", "SUGGESTION_CORRECT")}


def claim_once(log_root: Path, freeze_commit: str) -> dict:
    """An existing journal prohibits a second invocation, including after a crash."""
    log_root.mkdir(parents=True, exist_ok=True)
    journal = {"schema": "TASK8C_ONE_SHOT_JOURNAL_V1", "formal_freeze_commit": freeze_commit,
               "formal_runner_invocation_count": 1, "case_attempt_order": [],
               "cases": [], "all_child_processes_exited": False}
    write_json(log_root / "invocation.json", journal, exclusive=True)
    return journal


def execute_sequence(candidates: tuple[Candidate, ...], log_root: Path, journal: dict,
                     execute) -> list[dict]:
    """The attempt is persisted before launch; there is deliberately no retry loop."""
    for candidate in candidates:
        if candidate.relation in journal["case_attempt_order"]:
            raise ValueError("attempt already consumed")
        journal["case_attempt_order"].append(candidate.relation)
        write_json(log_root / "invocation.json", journal)
        print(f"TASK8C BEGIN {candidate.relation} (single attempt)", flush=True)
        row = execute(candidate)
        journal["cases"].append(row)
        write_json(log_root / "invocation.json", journal)
        print(f"TASK8C END {candidate.relation}: {row['runtime_status']}", flush=True)
    journal["all_child_processes_exited"] = True
    write_json(log_root / "invocation.json", journal)
    return journal["cases"]


def collect_case(candidate: Candidate) -> dict:
    output = EXTERNAL / "inference/output"
    before = inventory(output)
    roots_before = {p.name for p in output.iterdir()}
    runtime = drive_child(formal_argv(candidate), EXTERNAL, candidate.expected_program)
    verify_inventory(output, before, allow_new=True)
    new_roots = sorted(p for p in output.iterdir() if p.name not in roots_before)
    if len(new_roots) > 1 or any(not p.is_dir() for p in new_roots):
        raise ValueError(f"unexpected allocated roots: {new_roots}")
    root = new_roots[0] if new_roots else None
    if root and not re.fullmatch(re.escape(Path(candidate.image).stem) + r"(?:_\d{3,})?", root.name):
        raise ValueError(f"unexpected allocated root: {root}")
    payload = {}
    if root and (root / "diagnostics/result.json").is_file():
        payload = json.loads((root / "diagnostics/result.json").read_text(encoding="utf-8"))
    paths = payload.get("output_paths", {})
    success = runtime["exit_code"] == 0 and payload.get("status") == "SUCCESS"
    if success and not runtime["language_expected_program_executed"]:
        raise ValueError("pipeline executed without expected-program approval")
    if payload.get("reference_override") or payload.get("reference_mode", "automatic") != "automatic":
        raise ValueError("automatic reference contract violated")
    if not success and root and (any((root / "masks").glob("*")) or any((root / "overlays").glob("*"))):
        raise ValueError("failed run wrote mask/overlay")
    errors = re.findall(r"\bE\d{3}\b", runtime["stdout"])
    if payload.get("confidence") is not None:
        runtime["initial_confidence_display"] = runtime["initial_confidence"]
        runtime["initial_confidence"] = payload["confidence"]
    row = {**asdict(candidate), **runtime, "runtime_status": "SUCCESS" if success else "FAILED",
           "validity_scope": "RUNTIME_STRUCTURAL_ONLY", "semantic_status": "NOT_EVALUATED",
           "error_code": payload.get("error_code") or (errors[-1] if errors else None),
           "error_reason": payload.get("reason") or payload.get("detail") or
               ("wall_timeout" if runtime["timed_out"] else runtime["stdout"][-2500:] if not success else None),
           "run_root": str(root) if root else None,
           "mask_path": paths.get("mask") if success else None,
           "overlay_path": paths.get("overlay") if success else None,
           "diagnostics_path": str(root / "diagnostics") if root else None,
           "runtime_payload": payload,
           "artifacts": [identity(p) for p in sorted(root.rglob("*")) if p.is_file()] if root else []}
    for key in ("raw_proposal_count", "merged_proposal_count", "tile_count", "reference_mode",
                "reference_id", "reference_area", "reference_confidence", "reference_bbox",
                "direction", "reasoning_context", "mask_area", "target_centroid", "timings"):
        row[key] = payload.get(key)
    return row


def git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=REPO,
                          capture_output=True, check=True).stdout


def verify_freeze(evidence: dict) -> str:
    commit = evidence["formal_freeze_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise ValueError("no actual pushed freeze commit")
    committed = json.loads(git_bytes(commit, "evaluation/task8c_final_demo_v1.json"))
    frozen = evidence["formal_freeze"]
    if frozen != committed.get("formal_freeze") or frozen["schema"] != "FORMAL_FINAL_DEMO_FREEZE_V1":
        raise ValueError("freeze differs from committed freeze")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "rev-parse", "origin/eval/task8c-final-demo-v1"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    if head != commit or remote != commit:
        raise ValueError("local/remote tracking HEAD must equal pushed freeze")
    if [asdict(c) for c in CANDIDATES] != frozen["candidate_lock"]:
        raise ValueError("candidate lock differs")
    for row in frozen["harness_identities"]:
        if identity(REPO / row["relative"]) != row["working_file"]:
            raise ValueError(f"harness changed: {row['relative']}")
        if hashlib.sha256(git_bytes(commit, row["relative"])).hexdigest() != row["git_sha256"]:
            raise ValueError("Git harness identity differs")
    if not frozen["dedicated_test_result"]["pass"] or frozen["external_preflight"]["check_setup"]["status"] != "READY":
        raise ValueError("freeze gates not passed")
    gate = frozen["external_preflight"]["source_check"]
    if (gate["checked"], gate["match"], gate["missing"], gate["mismatch"]) != (135, 135, 0, 0):
        raise ValueError("source gate not 135/135")
    for name, relative in (("output", "inference/output"), ("input", "inference/input"),
                           ("model", "model")):
        verify_inventory(EXTERNAL / relative, frozen["inventory_before"][name])
    for row in frozen["source_files"]:
        if identity(EXTERNAL / row["relative"]) != row["identity"]:
            raise ValueError(f"source changed: {row['relative']}")
    if identity(EXTERNAL / "source_manifest.json") != frozen["external_manifest"]:
        raise ValueError("external manifest changed")
    for candidate in CANDIDATES:
        if sha256_file(Path(candidate.image)) != candidate.image_sha256:
            raise ValueError("candidate raster identity changed")
    for key, value in frozen["execution_environment"].items():
        if os.environ.get(key) != value:
            raise ValueError(f"execution environment differs: {key}")
    return commit


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("formal runner accepts no overrides")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    freeze_commit = verify_freeze(evidence)
    log_root = EXTERNAL / "logs/task8c_final_demo_v1"
    journal = claim_once(log_root, freeze_commit)
    cases = execute_sequence(CANDIDATES, log_root, journal, collect_case)
    for row in cases:
        transcript_path = log_root / f"{row['relation']}_stdout.txt"
        with transcript_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(row["stdout"])
        row["transcript_identity"] = identity(transcript_path)
    frozen_runtime = {"schema": "TASK8C_FROZEN_RUNTIME_V1", "formal_freeze_commit": freeze_commit,
                      "formal_runner_invocation_count": 1,
                      "case_attempt_order": journal["case_attempt_order"],
                      "all_child_processes_exited": True, "output_hashes_frozen": True, "cases": cases}
    runtime_path = log_root / "runtime_results.json"
    write_json(runtime_path, frozen_runtime, exclusive=True)
    evidence.update({"status": "FORMAL_RUNTIME_FROZEN", "formal_runner_invocation_count": 1,
                     "case_attempt_order": journal["case_attempt_order"], "cases": cases,
                     "frozen_runtime_identity": identity(runtime_path),
                     "gt_audit_started_after_runtime_freeze": False, "model_calls_during_gt_audit": 0})
    write_json(EVIDENCE, evidence)
    print("TASK8C all four processes exited; runtime JSON and output hashes frozen.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
