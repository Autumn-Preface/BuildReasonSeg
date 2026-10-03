"""Task 6C.5: profiling and system-utilization utilities.

Everything here is opt-in. Importing this module never changes training behaviour,
and `StageTimer` only records — it never synchronizes the device unless it is
constructed with `cuda_events=True`, which is reserved for the small synchronized
stage-profile run (Task 6C.5 section 8).

No new monitoring software is installed: GPU numbers come from `nvidia-smi`, CPU and
memory from `psutil`.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

#: Stage labels required by Task 6C.5 section 8, in pipeline order.
STAGES = (
    "image_io",
    "target_mask_io",
    "qwen_prepare_total",
    "qwen_image_preprocess",
    "qwen_chat_tokenize",
    "cpu_to_gpu",
    "sam_feature_lookup",
    "sam_feature_encode_on_miss",
    "qwen_forward",
    "sam_projection_decode_forward",
    "loss",
    "backward",
    "optimizer_step",
)

NVIDIA_SMI_QUERY = (
    "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm"
)


def percentiles(values: list[float]) -> dict:
    """mean / median / p90 / min / max for a list of samples."""

    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(clean)

    def _pct(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": _pct(0.9),
        "min": ordered[0],
        "max": ordered[-1],
    }


def read_nvidia_smi() -> dict:
    """One GPU sample. Returns zeros/None when nvidia-smi is unavailable."""

    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "error": (result.stderr or "").strip()[:200]}
        parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]

        def number(text: str):
            try:
                return float(text)
            except ValueError:
                return None

        return {
            "available": True,
            "gpu_utilization_percent": number(parts[0]),
            "memory_used_mib": number(parts[1]),
            "memory_total_mib": number(parts[2]),
            "temperature_c": number(parts[3]),
            "power_watts": number(parts[4]),
            "sm_clock_mhz": number(parts[5]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


class UtilizationSampler:
    """Background sampler for GPU and CPU/memory utilization.

    `interval_seconds` defaults to 0.5 s; the spec warns that nvidia-smi polling can
    itself perturb a run, so the sampler records how long each `nvidia-smi` call took
    and the benchmark reports it.
    """

    def __init__(self, interval_seconds: float = 0.5, track_process: bool = True) -> None:
        self.interval_seconds = float(interval_seconds)
        self.track_process = track_process
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self.nvidia_smi_seconds: list[float] = []
        self._process = None
        if track_process:
            try:
                import psutil

                self._process = psutil.Process()
                self._process.cpu_percent(None)
            except Exception:  # noqa: BLE001
                self._process = None

    def _loop(self) -> None:
        import psutil

        while not self._stop.is_set():
            started = time.perf_counter()
            gpu = read_nvidia_smi()
            self.nvidia_smi_seconds.append(time.perf_counter() - started)
            memory = psutil.virtual_memory()
            entry = {
                "t": time.perf_counter(),
                "gpu": gpu,
                "cpu_total_percent": psutil.cpu_percent(None),
                "system_ram_used_gib": round((memory.total - memory.available) / 1024**3, 3),
                "system_ram_percent": memory.percent,
            }
            if self._process is not None:
                try:
                    entry["process_cpu_percent"] = self._process.cpu_percent(None)
                    entry["process_rss_gib"] = round(self._process.memory_info().rss / 1024**3, 3)
                except Exception:  # noqa: BLE001
                    pass
            self.samples.append(entry)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "UtilizationSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        def series(getter):
            values = []
            for sample in self.samples:
                value = getter(sample)
                if value is not None:
                    values.append(value)
            return percentiles(values)

        return {
            "interval_seconds": self.interval_seconds,
            "n_samples": len(self.samples),
            "nvidia_smi_call_seconds": percentiles(self.nvidia_smi_seconds),
            "gpu_utilization_percent": series(lambda s: s["gpu"].get("gpu_utilization_percent")),
            "gpu_memory_used_mib": series(lambda s: s["gpu"].get("memory_used_mib")),
            "gpu_temperature_c": series(lambda s: s["gpu"].get("temperature_c")),
            "gpu_power_watts": series(lambda s: s["gpu"].get("power_watts")),
            "cpu_total_percent": series(lambda s: s.get("cpu_total_percent")),
            "process_cpu_percent": series(lambda s: s.get("process_cpu_percent")),
            "process_rss_gib": series(lambda s: s.get("process_rss_gib")),
            "system_ram_used_gib": series(lambda s: s.get("system_ram_used_gib")),
            "system_ram_percent": series(lambda s: s.get("system_ram_percent")),
        }


@dataclass
class StageProfile:
    """Accumulates per-stage seconds plus optional CUDA-event timings."""

    cuda_events: bool = False
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    cuda_totals: dict[str, float] = field(default_factory=dict)
    step_walls: list[float] = field(default_factory=list)

    def add(self, stage: str, seconds: float, cuda_seconds: float | None = None) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + float(seconds)
        self.counts[stage] = self.counts.get(stage, 0) + 1
        if cuda_seconds is not None:
            self.cuda_totals[stage] = self.cuda_totals.get(stage, 0.0) + float(cuda_seconds)

    @contextmanager
    def stage(self, name: str):
        """Wall-clock stage timer; no synchronize unless CUDA events were requested."""

        if not self.cuda_events:
            started = time.perf_counter()
            try:
                yield
            finally:
                self.add(name, time.perf_counter() - started)
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()
            self.add(name, time.perf_counter() - started, start_event.elapsed_time(end_event) / 1000.0)

    def summary(self) -> dict:
        total = sum(self.totals.values())
        return {
            "cuda_events": self.cuda_events,
            "stage_seconds_total": round(total, 4),
            "stage_seconds": {key: round(value, 4) for key, value in sorted(self.totals.items())},
            "stage_calls": dict(sorted(self.counts.items())),
            "stage_share_percent": {
                key: round(100.0 * value / total, 2) for key, value in sorted(self.totals.items())
            }
            if total
            else {},
            "stage_cuda_seconds": {
                key: round(value, 4) for key, value in sorted(self.cuda_totals.items())
            },
            "step_wall_seconds": percentiles(self.step_walls),
        }
