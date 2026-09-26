"""Task 6C.6 shared harness: variants of the formal batch-1 training step.

The benchmarked step is the **formal training path** (`scripts/task6c_train.py`,
Phase B), not the Task 6C.5 pipeline wrapper:

    batch, image = runtime.prepare(sample)
    features, _  = runtime.features_for(sample, image)
    runtime.train_step(batch, sample.target_mask(), features,
                       optimizer=optimizer,
                       collect_grad_norms=<config>,
                       clip_grad_foreach=<variant>)

That keeps "integrated baseline B0.6" honest: the number reported here is the path
the formal loop actually executes, with `training.collect_grad_norms=false` wired in
by section 1 of the task.

Every variant is measured on the fixed Task 6C.5 benchmark ids (64 records / 32 paired
images, train split only), one sample per optimizer step, after 8 unmeasured warmup
steps and with the SAM2 feature cache warm.

Nothing here changes the model, the loss, the optimizer mathematics, the data or the
sample order. The optional `torch.compile` wrapper and the optimizer/clipping
implementation switches exist only so they can be measured and gated.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.model import decode_mask  # noqa: E402
from buildreasonseg_mvp.perf import StageProfile, UtilizationSampler  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"

#: Section 4: reference value from Task 6C.5, for context only.
TASK6C5_REFERENCE_SAMPLES_PER_SEC = 2.37

#: Task 6C.5's adopted interleaved figures, used as the B0.6 sanity anchor.
TASK6C5_BASELINE_SAMPLES_PER_SEC = 2.152
TASK6C5_ADOPTED_SAMPLES_PER_SEC = 2.373


# ---------------------------------------------------------------- runtime setup


def load_benchmark_samples(limit: int | None = None) -> list:
    """The fixed Task 6C.5 benchmark set: 64 records over 32 paired images, train split."""

    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    ids = payload["record_ids"][:limit] if limit else payload["record_ids"]
    return [data_mod.to_sample(train[sample_id]) for sample_id in ids]


def build_variant_runtime(
    *,
    deterministic_strict: bool | None = None,
    collect_grad_norms: bool | None = None,
    gradient_checkpointing: bool | None = None,
    bridge: str = "centre",
    verbose: bool = False,
):
    """A clean runtime per variant, so no variant inherits another's state.

    Everything is applied through the config, i.e. through the same code path the
    formal runs use. `gradient_checkpointing` in particular is not toggled afterwards:
    `build_runtime` reads it and enables checkpointing on the Qwen tower exactly as it
    does for a formal training run.
    """

    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": bridge}
    if collect_grad_norms is not None:
        cfg.setdefault("training", {})["collect_grad_norms"] = bool(collect_grad_norms)
    if deterministic_strict is not None:
        cfg.setdefault("training", {})["deterministic_strict"] = bool(deterministic_strict)
    if gradient_checkpointing is not None:
        cfg.setdefault("training", {})["gradient_checkpointing"] = bool(gradient_checkpointing)
    return build_runtime(cfg, device="cuda", verbose=verbose)


def collect_grad_norms_setting(runtime) -> bool:
    """Mirror of the formal loop's reader (Task 6C.6 section 1)."""

    return bool(runtime.cfg.get("training", {}).get("collect_grad_norms", True))


# -------------------------------------------------------------------- snapshot


def snapshot_trainable(model) -> dict:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(model, snapshot: dict) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in snapshot:
                parameter.copy_(snapshot[name])


# ------------------------------------------------------------------ optimizer


def make_optimizer(runtime, *, kind: str = "default"):
    """AdamW with the Task 6C recipe, unchanged apart from the reduction implementation.

    `kind` selects the implementation only:

    * ``default``  -- `torch.optim.AdamW` as the formal loop constructs it;
    * ``foreach``  -- ``foreach=True`` (multi-tensor reductions);
    * ``fused``    -- ``fused=True`` (single fused CUDA kernel).

    Learning rates, betas, weight decay and parameter groups are identical in all
    three, because they all come from the same config and the same
    `trainable_parameter_groups` call. The implementation changes the reduction order,
    which is why each one has to be gated on equivalence.
    """

    cfg = runtime.cfg["optimizer"]["phase_b"]
    groups = runtime.model.trainable_parameter_groups(
        lora_lr=float(cfg["lora_lr"]),
        head_lr=float(cfg["token_lr"]),
        weight_decay=float(runtime.cfg["optimizer"]["weight_decay"]),
        decoder_lr=float(cfg.get("decoder_lr", cfg["token_lr"])),
        token_lr=float(cfg["token_lr"]),
    )
    betas = tuple(runtime.cfg["optimizer"]["betas"])
    if kind == "default":
        return torch.optim.AdamW(groups, betas=betas)
    if kind == "foreach":
        return torch.optim.AdamW(groups, betas=betas, foreach=True)
    if kind == "fused":
        return torch.optim.AdamW(groups, betas=betas, fused=True)
    raise ValueError(f"unknown optimizer kind {kind!r}")


def describe_optimizer(runtime, kind: str) -> dict:
    """Which implementation `torch.optim.AdamW` actually resolves to, and why."""

    info: dict[str, Any] = {"requested": kind}
    try:
        optimizer = make_optimizer(runtime, kind=kind)
        info["constructed"] = True
        info["class"] = type(optimizer).__name__
        info["defaults"] = {
            key: (value if isinstance(value, (int, float, bool, str, type(None), tuple)) else str(value))
            for key, value in optimizer.defaults.items()
        }
        info["param_groups"] = len(optimizer.param_groups)
        info["params"] = sum(len(group["params"]) for group in optimizer.param_groups)
        dtypes = sorted({str(p.dtype) for group in optimizer.param_groups for p in group["params"]})
        info["param_dtypes"] = dtypes
    except Exception as exc:  # noqa: BLE001
        info["constructed"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from torch.optim.optimizer import _default_to_fused_or_foreach

        #: Ask torch what it resolves to for exactly the parameters this variant trains.
        params = [p for group in optimizer.param_groups for p in group["params"]]
        fused, foreach = _default_to_fused_or_foreach(params, differentiable=False)
        info["torch_default_resolution"] = {
            "fused": bool(fused),
            "foreach": bool(foreach),
            "interpretation": (
                "fused single-kernel path"
                if fused
                else ("foreach multi-tensor path" if foreach else "per-tensor loop")
            ),
        }
    except Exception as exc:  # noqa: BLE001
        info["torch_default_resolution"] = f"unavailable: {type(exc).__name__}: {exc}"
    return info


# -------------------------------------------------------------------- compile
#
# Task 6C.6 section 6. Scopes C1a / C1b / C1c, several backends and modes. Every
# attempt is recorded, including failures, and nothing falls back silently: a variant
# is marked compiled only if the wrapper was actually installed.


COMPILE_SCOPES = ("qwen", "decoder_path", "decoder_tail", "combined")


def _traceback_tail(limit: int = 25) -> str:
    """Last lines of the current exception, for the failure record."""

    import traceback

    lines = traceback.format_exc().strip().splitlines()
    return "\n".join(lines[-limit:])[:3000]


def _decoder_tail(projection, sam, bridge, seg_hidden, sam_features):
    """C1b (function form): projection + SAM mask decoder, submodules passed in.

    Identical computation to the tail of `BuildReasonSegMvp.forward`.
    """

    projected = projection(seg_hidden)
    decoded = decode_mask(sam, sam_features, projected, multimask_output=False, bridge=bridge)
    return projected, decoded



class _CompiledDecoderPath(torch.nn.Module):
    """C1b: the projection + SAM mask-decoder tail, wrapped so dynamo can see it.

    It is the exact tail of `BuildReasonSegMvp.forward` (same projection, same
    `decode_mask` call, same bridge, same `multimask_output`), so compiling it cannot
    change what is computed. It returns the same two objects the uncompiled tail
    produces, which is what makes the hook below a pure call redirection.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, seg_hidden, sam_features):  # noqa: D102
        projected = self.model.projection(seg_hidden)
        decoded = decode_mask(
            self.model.sam,
            sam_features,
            projected,
            multimask_output=False,
            bridge=self.model.bridge,
        )
        return projected, decoded


def apply_compile(runtime, *, scope: str, backend: str, mode: str | None = None) -> dict:
    """Install a `torch.compile` wrapper for `scope`; return what happened."""

    info: dict[str, Any] = {
        "scope": scope,
        "backend": backend,
        "mode": mode,
        "compiled": False,
        "compile_seconds": None,
        "first_step_seconds": None,
        "error": None,
    }
    kwargs: dict[str, Any] = {"backend": backend}
    if mode is not None:
        kwargs["mode"] = mode
    started = time.perf_counter()
    try:
        if scope == "qwen":
            runtime.model.qwen = torch.compile(runtime.model.qwen, **kwargs)
        elif scope == "decoder_path":
            runtime.model._task6c6_decoder_path = _CompiledDecoderPath(runtime.model)
            runtime.model._task6c6_compiled_decoder = torch.compile(
                runtime.model._task6c6_decoder_path, **kwargs
            )
            _install_decoder_hook(runtime.model)
        elif scope == "decoder_tail":
            # Function-style: the submodules are arguments, so the compiled region does
            # not capture the parent module (which is what makes dynamo trace the model
            # twice and raise "duplicate template name" for the scope above).
            runtime.model._task6c6_compiled_decoder = torch.compile(_decoder_tail, **kwargs)
            _install_decoder_hook(runtime.model)
        elif scope == "combined":
            runtime.model = torch.compile(runtime.model, **kwargs)
        else:
            raise ValueError(f"unknown compile scope {scope!r}")
        info["compiled"] = True
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback_tail"] = _traceback_tail()
    info["compile_seconds"] = round(time.perf_counter() - started, 3)
    return info


def _install_decoder_hook(model) -> None:
    """Route `forward`'s tail through the compiled decoder path.

    Only the call, not the computation, changes: the head (`forward_qwen`) and the
    tail (compiled projection + `decode_mask`) are the same two pieces `forward`
    already ran, in the same order, with the same arguments. `_task6c6_original_forward`
    keeps the unmodified method so the hook is reversible and cannot silently persist.
    """

    if getattr(model, "_task6c6_original_forward", None) is None:
        model._task6c6_original_forward = model.forward

        def forward(batch, sam_features, multimask_output: bool = False):
            from buildreasonseg_mvp.model import MvpForwardOutput  # noqa: PLC0415
            from buildreasonseg_mvp.qwen_seg import forward_qwen  # noqa: PLC0415

            lm_logits, seg_hidden = forward_qwen(model.qwen, batch)
            compiled = model._task6c6_compiled_decoder
            try:
                projected, decoded = compiled(
                    model.projection, model.sam, model.bridge, seg_hidden, sam_features
                )
            except TypeError:
                # `_CompiledDecoderPath` takes (seg_hidden, sam_features) instead.
                projected, decoded = compiled(seg_hidden, sam_features)
            return MvpForwardOutput(
                lm_logits=lm_logits,
                seg_hidden=seg_hidden,
                projected=projected,
                mask_logits=decoded.low_res_logits,
                iou_prediction=decoded.iou_prediction,
                shapes={
                    "sparse_prompt_shape": list(decoded.sparse_prompt_shape),
                    "visual_tokens": batch.visual_tokens,
                    "sequence_length": batch.total_length,
                    "prompt_length": batch.prompt_length,
                    "bridge": model.bridge,
                },
                sparse_prompt=decoded.sparse_prompt,
                prompt_diagnostics=decoded.prompt_diagnostics,
            )

        model.forward = forward


#: Dynamo counters that describe graph capture. Recorded before/after a variant so the
#: recompilation count and graph-break count are measured rather than guessed.
DYNAMO_COUNTER_KEYS = (
    "graph_breaks",
    "unique_graphs",
    "frame_count",
    "guard_failures",
    "calls_captured",
    "recompiles",
)


def dynamo_counters() -> dict:
    try:
        from torch._dynamo.utils import counters

        snapshot = json.loads(json.dumps(counters, default=str))
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    flat: dict[str, int] = {}
    for group, values in (snapshot or {}).items():
        if isinstance(values, dict):
            for key, value in values.items():
                try:
                    flat[f"{group}.{key}"] = int(value)
                except (TypeError, ValueError):
                    continue
    return {"available": True, "counters": flat}


def counter_delta(before: dict, after: dict) -> dict:
    if not before.get("available") or not after.get("available"):
        return {"available": False}
    left, right = before.get("counters", {}), after.get("counters", {})
    keys = sorted(set(left) | set(right))
    delta = {key: right.get(key, 0) - left.get(key, 0) for key in keys}
    interesting = {key: value for key, value in delta.items() if value}
    return {
        "available": True,
        "delta": interesting,
        "graph_breaks": sum(v for k, v in interesting.items() if "graph_break" in k),
        "recompiles": sum(v for k, v in interesting.items() if "recompile" in k),
        "frames": sum(v for k, v in interesting.items() if "frame" in k or "calls" in k),
    }


# ---------------------------------------------------------------- measurement


@dataclass
class VariantSpec:
    """One measured configuration of the formal step."""

    name: str
    description: str = ""
    optimizer_kind: str = "default"
    clip_grad_foreach: bool | None = None
    compile_scope: str | None = None
    compile_backend: str = "inductor"
    compile_mode: str | None = None
    deterministic_strict: bool | None = None
    gradient_checkpointing: bool | None = None
    collect_grad_norms: bool | None = None
    runtime_mode: str = "default"
    #: Section 9: force one already-built SDPA backend for the whole measured loop
    #: ("flash" / "mem_efficient" / "math" / "cudnn"). `None` leaves torch's choice alone.
    sdpa_backend: str | None = None
    notes: list[str] = field(default_factory=list)
    adopted: bool = False
    rejection_reason: str | None = None
    equivalence_status: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "optimizer_kind": self.optimizer_kind,
            "clip_grad_foreach": self.clip_grad_foreach,
            "compile_scope": self.compile_scope,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "deterministic_strict": self.deterministic_strict,
            "gradient_checkpointing": self.gradient_checkpointing,
            "collect_grad_norms": self.collect_grad_norms,
            "runtime_mode": self.runtime_mode,
            "sdpa_backend": self.sdpa_backend,
            "notes": list(self.notes),
            "adopted": self.adopted,
            "rejection_reason": self.rejection_reason,
            "equivalence_status": self.equivalence_status,
        }


def _sdpa_context(backend: str | None):
    """(context manager, error) forcing one SDPA backend, or a no-op when not requested.

    Section 9: only backends this PyTorch already ships are allowed. Nothing is
    installed and no unsupported kernel is forced; if the request cannot be honoured
    the error is recorded and the variant keeps torch's own choice.
    """

    if not backend:
        return contextlib.nullcontext(), None
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }
        if backend not in mapping:
            return contextlib.nullcontext(), f"unknown sdpa backend {backend!r}"
        return sdpa_kernel(mapping[backend]), None
    except Exception as exc:  # noqa: BLE001
        return contextlib.nullcontext(), f"{type(exc).__name__}: {exc}"


def run_variant(
    spec: VariantSpec,
    samples: list,
    *,
    steps: int = 64,
    warmup: int = 8,
    sampler_interval: float = 0.5,
    track_stages: bool = False,
) -> dict:
    """Measure one variant end-to-end. Returns a section-16 style record."""

    entry: dict[str, Any] = {"variant": spec.name, **spec.as_dict()}
    runtime = build_variant_runtime(
        deterministic_strict=spec.deterministic_strict,
        collect_grad_norms=spec.collect_grad_norms,
        gradient_checkpointing=spec.gradient_checkpointing,
    )
    entry["collect_grad_norms_effective"] = collect_grad_norms_setting(runtime)
    entry["gradient_checkpointing_config"] = bool(
        runtime.cfg.get("training", {}).get("gradient_checkpointing", True)
    )
    entry["gradient_checkpointing_enabled"] = bool(
        getattr(getattr(runtime, "qwen", None), "is_gradient_checkpointing", False)
    )

    entry["determinism"] = runtime.reports["determinism"]
    entry["trainable_params"] = runtime.reports["params"]

    if spec.compile_scope:
        entry["compile"] = apply_compile(
            runtime, scope=spec.compile_scope, backend=spec.compile_backend, mode=spec.compile_mode
        )
        if not entry["compile"]["compiled"]:
            entry["status"] = "compile_failed"
            entry["measured_seconds"] = None
            entry["samples_per_sec"] = None
            entry["error"] = entry["compile"]["error"]
            del runtime
            torch.cuda.empty_cache()
            return entry
    else:
        entry["compile"] = {"compiled": False, "scope": None}

    entry["optimizer"] = describe_optimizer(runtime, spec.optimizer_kind)
    if not entry["optimizer"].get("constructed"):
        entry["status"] = "optimizer_failed"
        entry["samples_per_sec"] = None
        del runtime
        torch.cuda.empty_cache()
        return entry

    optimizer = make_optimizer(runtime, kind=spec.optimizer_kind)
    set_seed(int(runtime.cfg["seed"]))

    sequence = [samples[index % len(samples)] for index in range(warmup + steps)]
    profile = StageProfile(cuda_events=False) if track_stages else None

    # Warm the SAM2 CPU feature cache over the measured sequence only, so no variant
    # pays cache-miss encoding inside the measured window.
    for sample in sequence:
        image = sample.image_rgb()
        runtime.features_for(sample, image)
        del image

    counters_before = dynamo_counters()
    step_walls: list[float] = []
    first_step_seconds: float | None = None
    first_step_losses: dict | None = None
    grad_norms_present: list[bool] = []
    clip_norms: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    sdpa_context = _sdpa_context(spec.sdpa_backend)
    entry["sdpa_backend_forced"] = spec.sdpa_backend
    entry["sdpa_backend_error"] = sdpa_context[1]
    try:
        with UtilizationSampler(interval_seconds=sampler_interval) as sampler, sdpa_context[0]:
            started = time.perf_counter()
            for index, sample in enumerate(sequence):
                step_started = time.perf_counter()
                batch, image = runtime.prepare(sample)
                gt_mask = sample.target_mask()
                features, _cached = runtime.features_for(sample, image)
                result = runtime.train_step(
                    batch,
                    gt_mask,
                    features,
                    optimizer=optimizer,
                    timer=profile,
                    collect_grad_norms=collect_grad_norms_setting(runtime),
                    clip_grad_foreach=spec.clip_grad_foreach,
                )
                if index == 0:
                    torch.cuda.synchronize()
                    first_step_seconds = time.perf_counter() - step_started
                    first_step_losses = {
                        key: repr(float(value)) for key, value in (result.get("losses") or {}).items()
                    }
                grad_norms_present.append(bool(result.get("grad_norms")))
                clip_norms.append(float(result.get("grad_clip_total_norm") or 0.0))
                del result, features, batch
                if index >= warmup:
                    step_walls.append(time.perf_counter() - step_started)
            measured_seconds = time.perf_counter() - started
            utilization = sampler.summary()
    except Exception as exc:  # noqa: BLE001
        # Section 6: a candidate that cannot run keeps its failure record. A torch.compile
        # graph that raises on first call must not take the whole group down, and must not
        # be reported as merely "slow".
        entry.update(
            {
                "status": "runtime_failed",
                "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                "traceback_tail": _traceback_tail(),
                "failed_at_step": len(grad_norms_present),
                "steps_measured": 0,
                "samples_per_sec": None,
                "ms_per_sample": None,
                "compile": entry.get("compile"),
            }
        )
        del optimizer, runtime
        torch.cuda.empty_cache()
        return entry
    counters_after = dynamo_counters()

    measured_count = len(step_walls)
    measured_wall = sum(step_walls)
    entry.update(
        {
            "status": "ok",
            "steps_measured": measured_count,
            "warmup_steps": warmup,
            "total_sequence_seconds": round(measured_seconds, 4),
            "measured_seconds": round(measured_wall, 4),
            "ms_per_sample": round(1000.0 * measured_wall / max(1, measured_count), 3),
            "samples_per_sec": round(measured_count / measured_wall, 4) if measured_wall else None,
            "first_step_seconds": round(first_step_seconds, 3) if first_step_seconds else None,
            "first_step_losses": first_step_losses,
            "step_wall_seconds": {
                "n": len(step_walls),
                "mean": round(measured_wall / max(1, len(step_walls)), 6),
                "min": round(min(step_walls), 6) if step_walls else None,
                "max": round(max(step_walls), 6) if step_walls else None,
            },
            "grad_norms_collected": all(grad_norms_present),
            "clip_total_norm_last": round(clip_norms[-1], 6) if clip_norms else None,
            "clip_total_norm_mean": round(sum(clip_norms) / max(1, len(clip_norms)), 6),
            "utilization": utilization,
            "vram": {
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            },
            "dynamo": counter_delta(counters_before, counters_after),
        }
    )
    if profile is not None:
        profile.step_walls = step_walls
        entry["stage_profile"] = profile.summary()

    del optimizer, runtime
    torch.cuda.empty_cache()
    return entry


def primary_metrics(entry: dict) -> dict:
    utilization = entry.get("utilization") or {}
    return {
        "samples_per_sec": entry.get("samples_per_sec"),
        "ms_per_sample": entry.get("ms_per_sample"),
        "gpu_utilization_mean": (utilization.get("gpu_utilization_percent") or {}).get("mean"),
        "cpu_utilization_mean": (utilization.get("cpu_total_percent") or {}).get("mean"),
        "process_rss_gib_mean": (utilization.get("process_rss_gib") or {}).get("mean"),
        "peak_reserved_vram_gib": (entry.get("vram") or {}).get("peak_reserved_gib"),
        "peak_allocated_vram_gib": (entry.get("vram") or {}).get("peak_allocated_gib"),
        "gpu_temperature_c_mean": (utilization.get("gpu_temperature_c") or {}).get("mean"),
        "gpu_power_watts_mean": (utilization.get("gpu_power_watts") or {}).get("mean"),
    }


def bracket_reference(entries: list[dict], reference_prefix: str) -> dict:
    """Reference-bracketed comparison for a sequential group.

    Identical configurations on this laptop differ by up to ~17 % in absolute
    throughput between runs (measured in `task6c6_integrated_baseline.json`), while a
    ratio measured inside one run is stable to well under 1 %. Running the reference
    first *and* last therefore gives both an estimate and an honest uncertainty: a
    candidate whose gain is smaller than the reference drift is not resolvable, and the
    caller must not adopt it on the strength of that number.
    """

    references = [
        entry for entry in entries if entry["variant"].startswith(reference_prefix) and entry.get("samples_per_sec")
    ]
    rates = [entry["samples_per_sec"] for entry in references]
    if not rates:
        return {"available": False}
    first, last = rates[0], rates[-1]
    mean = sum(rates) / len(rates)
    drift = 100.0 * (last / first - 1.0)
    return {
        "available": True,
        "reference_variants": [entry["variant"] for entry in references],
        "reference_samples_per_sec": rates,
        "reference_first": first,
        "reference_last": last,
        "reference_mean": round(mean, 4),
        "reference_drift_percent": round(drift, 3),
        "resolvable_gain_percent_floor": round(abs(drift), 3),
    }


def annotate_speedups(entries: list[dict], bracket: dict, reference_prefix: str = "uncompiled_reference") -> None:
    """Attach speedup and resolvability using an *interpolated* reference.

    Identical configurations on this laptop differ by up to ~19 % in absolute throughput
    across a group (`reference_drift_percent`), which is far larger than the effects the
    candidate audit is looking for. The reference is therefore run first and last, and a
    candidate sitting between them is compared against the linear interpolation of the
    two. The residual uncertainty is half the local reference drift; a candidate whose
    gain is inside that band is marked unresolvable and must not be adopted on it.
    """

    if not bracket.get("available"):
        return
    reference_positions = [
        index
        for index, entry in enumerate(entries)
        if entry["variant"].startswith(reference_prefix) and entry.get("samples_per_sec")
    ]
    if len(reference_positions) < 2:
        return
    for index, entry in enumerate(entries):
        rate = entry.get("samples_per_sec")
        entry["reference_position"] = index
        if entry["variant"].startswith(reference_prefix) or not rate:
            entry["speedup_percent_vs_interpolated_reference"] = None
            entry["gain_resolvable"] = False
            entry["local_reference_uncertainty_percent"] = None
            continue
        before = [position for position in reference_positions if position < index]
        after = [position for position in reference_positions if position > index]
        low_index = before[-1] if before else None
        high_index = after[0] if after else None
        if low_index is not None and high_index is not None:
            low, high = entries[low_index], entries[high_index]
            span = high_index - low_index
            fraction = (index - low_index) / span if span else 0.0
            expected = low["samples_per_sec"] + fraction * (
                high["samples_per_sec"] - low["samples_per_sec"]
            )
            uncertainty = (
                100.0
                * abs(high["samples_per_sec"] - low["samples_per_sec"])
                / 2.0
                / max(1e-9, (high["samples_per_sec"] + low["samples_per_sec"]) / 2.0)
            )
            entry["reference_position_of_bracketing_runs"] = [low_index, high_index]
        else:
            expected = bracket["reference_mean"]
            uncertainty = abs(bracket["reference_drift_percent"])
        entry["speedup_percent_vs_interpolated_reference"] = round(
            100.0 * (rate / expected - 1.0), 3
        )
        entry["local_reference_uncertainty_percent"] = round(uncertainty, 3)
        entry["gain_resolvable"] = bool(
            abs(entry["speedup_percent_vs_interpolated_reference"]) > uncertainty
        )
        entry["gain_resolvable_above_reference_drift"] = entry["gain_resolvable"]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def merge_json(path: Path, key: str, payload) -> None:
    """Merge under `key`, preserving any other keys already in the file."""

    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    existing[key] = payload
    write_json(path, existing)
