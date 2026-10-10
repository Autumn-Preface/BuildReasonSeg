"""Run-isolated read-only trace persistence and off-main-thread preview rendering."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
from PIL import Image

from .render import PreviewRenderer, array_summary, thumbnail

STAGES = ("语言理解", "YOLO proposals", "Automatic Reference", "Reasoning Context",
          "SAM2 Visual Features", "Spatial Reasoning Fields", "Prototype & Similarity",
          "D-B1 Segmentation", "Final Guard & Result")
STATES = {"NOT_EXECUTED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"}
STATE_TEXT = {"NOT_EXECUTED": "未执行", "RUNNING": "进行中", "COMPLETED": "已完成",
              "FAILED": "失败", "CANCELLED": "用户取消"}


def immutable(value):
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): immutable(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return array_summary(value)       # hashes/shape only; never a large feature dump
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclasses.dataclass(frozen=True)
class StageEvent:
    run_id: str
    sequence: int
    stage_id: int
    state: str
    timestamp: str
    elapsed_seconds: float
    metadata: dict
    previews: list
    error_code: str | None = None

    def to_dict(self):
        return dataclasses.asdict(self)


class TraceSession:
    """One session/renderer queue per user request; no model object or call is held here."""
    def __init__(self, directory: Path, events, *, renderer=None):
        self.directory = Path(directory).resolve()
        self.trace = self.directory / "trace"
        self.trace.mkdir(exist_ok=False)
        self.run_id = self.directory.name
        self.events = events
        self.started = time.perf_counter()
        self.renderer = PreviewRenderer() if renderer is None else renderer
        self.queue = queue.Queue(maxsize=3)
        self.stages = {i: "NOT_EXECUTED" for i in range(1, 10)}
        self.latest = {}
        self.records = []
        self.failures = []
        self.engine_run_root = None
        self.runtime_status = "NOT_EXECUTED"
        self.closed = False
        self.lock = threading.Lock()
        self.events.put(("trace_start", {"run_id": self.run_id, "directory": str(self.directory), "reopened": False}))
        self._manifest()
        self.thread = threading.Thread(target=self._render, daemon=False, name="BuildReasonSeg-trace-renderer")
        self.thread.start()

    def _manifest(self):
        manifest = {"schema": "USER_DEMO_TRACE_V1", "run_id": self.run_id,
                    "status": "OBSERVATION_FAILED" if self.failures else ("RECORDING" if not self.closed else "FROZEN"),
                    "runtime_status": self.runtime_status, "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
                    "semantic_status": "NOT_EVALUATED", "engine_run_root": self.engine_run_root,
                    "stages": {str(i): {"name": STAGES[i-1], "state": self.stages[i], "latest_event": self.latest.get(i)} for i in range(1, 10)},
                    "event_count": len(self.records), "observation_failures": list(self.failures),
                    "feature_tensor_saved": False, "model_calls_by_trace": 0}
        target = self.trace / "trace_manifest.json"
        temporary = self.trace / "trace_manifest.pending"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(target)         # only this run's own manifest; never another run or engine file

    def __call__(self, packet):
        self.submit(packet)

    def submit(self, packet):
        if self.closed:
            raise RuntimeError("Trace is already frozen")
        stage, state = int(packet["stage_id"]), packet["state"]
        if stage not in self.stages or state not in STATES:
            raise ValueError("Invalid observation event")
        packet = {**packet, "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "observed_elapsed": round(time.perf_counter() - self.started, 3)}
        self.queue.put(immutable(packet))

    def emit(self, stage, state, metadata=None, arrays=None):
        self.submit({"stage_id": stage, "state": state, "metadata": metadata or {}, "arrays": arrays or {}})

    def observation_failed(self, error, stage=None):
        with self.lock:
            self.failures.append({"stage_id": stage, "error_type": type(error).__name__,
                                  "message": "过程记录未完成，请保留本次记录；不会自动重试。"})
        self.events.put(("trace_error", {"run_id": self.run_id, "stage_id": stage,
                                        "message": "过程观测或记录失败，本次用户交付未完成。", "error_code": "E502"}))

    def _record(self, stage, state, metadata, arrays=None, observed_at=None, observed_elapsed=None):
        if self.stages[stage] in {"FAILED", "CANCELLED"} and state == "COMPLETED":
            raise ValueError("Cannot invent completion after failure")
        if state == "COMPLETED":
            incomplete = [i for i in range(1, stage) if self.stages[i] != "COMPLETED"]
            if incomplete:
                raise ValueError("Out-of-order stage completion")
        sequence = len(self.records) + 1
        images, display = self.renderer.render(stage, arrays or {}, metadata)
        previews = []
        for i, image in enumerate(images):
            name = f"stage_{stage:02d}_event_{sequence:03d}_{i+1}.png"
            target = self.trace / name
            with target.open("xb") as f:
                image.save(f, format="PNG")
            previews.append({"path": name, "bytes": target.stat().st_size,
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        details = plain(metadata)
        details["display"] = plain(display)
        if arrays:
            details["array_snapshot_identities"] = plain(arrays)
        details["render_completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        event = StageEvent(self.run_id, sequence, stage, state, observed_at or details["render_completed_at"],
                           round(time.perf_counter() - self.started, 3) if observed_elapsed is None else observed_elapsed,
                           details, previews, metadata.get("error_code"))
        record = event.to_dict()
        with (self.trace / "stage_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.records.append(record)
        self.stages[stage] = state
        self.latest[stage] = record
        self._manifest()
        self.events.put(("stage_event", {"event": record, "images": [thumbnail(im, (510, 390)) for im in images]}))

    def _render(self):
        while True:
            packet = self.queue.get()
            try:
                if packet is None:
                    return
                stage, state = int(packet["stage_id"]), packet["state"]
                meta = packet.get("metadata", {})
                if stage == 9 and state == "FAILED":
                    running = [i for i in range(1, 9) if self.stages[i] == "RUNNING"]
                    if running:
                        self._record(running[-1], "FAILED", {"error_code": meta.get("error_code"),
                                     "reason": meta.get("reason"), "context": meta.get("context"),
                                     "completed_previous_stages_preserved": True})
                self._record(stage, state, meta, packet.get("arrays", {}), packet.get("observed_at"), packet.get("observed_elapsed"))
            except Exception as error:
                self.observation_failed(error, None if packet is None else packet.get("stage_id"))
            finally:
                self.queue.task_done()

    def finish(self, *, result=None, error=None, engine_run_root=None, cancelled=False):
        self.queue.join()
        self.engine_run_root = engine_run_root
        self.runtime_status = "FAILED" if result is None else result.status
        code = getattr(error, "code", None) or getattr(result, "error_code", None)
        if result is not None and result.status == "SUCCESS" and any(self.stages[i] != "COMPLETED" for i in range(1, 10)):
            self.observation_failed(RuntimeError("Successful runtime lacks required stage observations"))
        if result is None and (code == "E901" or cancelled):
            if self.stages[1] != "COMPLETED":
                self.emit(1, "CANCELLED", {"error_code": code, "subsequent_stages": "NOT_EXECUTED"})
            self.runtime_status = "CANCELLED"
        elif (result is None or result.status != "SUCCESS") and self.stages[9] == "NOT_EXECUTED":
            self.emit(9, "FAILED", {"runtime_status": "FAILED", "error_code": code or "E502",
                       "guard_executed": False, "final_output_valid": False, "semantic_status": "NOT_EVALUATED"})
        self.queue.join()
        self.queue.put(None)
        self.thread.join()
        self.closed = True
        try:
            self._manifest()
        except Exception as error:
            self.observation_failed(error)
        self.events.put(("trace_frozen", {"run_id": self.run_id, "directory": str(self.directory),
                                         "observation_ok": not self.failures}))
        return not self.failures


def reopen_trace(manifest_path: Path, root: Path, events):
    manifest_path = Path(manifest_path).resolve()
    allowed = Path(root).resolve() / "results"
    if not manifest_path.is_relative_to(allowed) or manifest_path.name != "trace_manifest.json":
        raise ValueError("Only this Demo's saved traces may be reopened")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace = manifest_path.parent
    run_id = manifest["run_id"]
    if trace.parent.name != run_id or manifest["schema"] != "USER_DEMO_TRACE_V1":
        raise ValueError("Trace identity conflict")
    records = [json.loads(line) for line in (trace / "stage_events.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != manifest["event_count"]:
        raise ValueError("Trace event inventory conflict")
    prepared = []
    for sequence, record in enumerate(records, 1):
        if record["run_id"] != run_id or record["sequence"] != sequence:
            raise ValueError("Cross-run or event order conflict")
        images = []
        for preview in record["previews"]:
            path = (trace / preview["path"]).resolve()
            if not path.is_relative_to(trace) or path.stat().st_size != preview["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != preview["sha256"]:
                raise ValueError("Trace preview integrity conflict")
            with Image.open(path) as image:
                images.append(thumbnail(image.convert("RGB"), (510, 390)))
        prepared.append({"event": record, "images": images})
    # Emit only after validating the entire requested run, so old/current runs cannot be mixed.
    events.put(("trace_start", {"run_id": run_id, "directory": str(trace.parent), "reopened": True}))
    for item in prepared:
        events.put(("stage_event", item))
    events.put(("trace_frozen", {"run_id": run_id, "directory": str(trace.parent),
                                 "observation_ok": manifest["status"] != "OBSERVATION_FAILED"}))
    return manifest
