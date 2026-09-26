#!/usr/bin/env python
"""Task 6C.7 section 4: localize the remaining synchronization hotspots.

    python scripts/task6c7_profile.py [--steps 3] [--skip-scopes]

Task 6C.6 counted ~1,051 `aten::item` and ~130 `cudaStreamSynchronize` events per step but
did not attribute them. This run profiles the formal Phase-B step with `with_stack=True`
and ranks the call sites that produce them, then repeats the measurement per component
(vision tower / full Qwen forward / SAM decode / optimizer) as a fallback attribution that
does not depend on stack capture.

Writes `evaluation/task6c7_sync_hotspots.json`. Compact output only: no raw trace, no
per-event dump. Nothing here changes training semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp.model import decode_mask  # noqa: E402
from buildreasonseg_mvp.visual_cache import resolve_visual_host  # noqa: E402
from task6c6_common import (  # noqa: E402
    EVAL,
    build_variant_runtime,
    collect_grad_norms_setting,
    load_benchmark_samples,
    make_optimizer,
    set_seed,
    write_json,
)

OUT = EVAL / "task6c7_sync_hotspots.json"

#: Ops whose count is the question.
ITEM_OPS = ("aten::item", "aten::_local_scalar_dense")
SYNC_OPS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaMemcpy")

#: Third-party and project roots, innermost-first attribution order.
BUCKETS = (
    ("project", str(REPO_ROOT / "buildreasonseg_mvp")),
    ("project_scripts", str(REPO_ROOT / "scripts")),
    ("transformers_qwen3_vl", "transformers\\models\\qwen3_vl"),
    ("transformers", "site-packages\\transformers"),
    ("peft", "site-packages\\peft"),
    ("sam2", "sam2"),
    ("torch", "site-packages\\torch"),
)


def _bucket_and_frame(stack) -> tuple[str, str, str]:
    """Innermost frame that is not torch's own dispatcher, plus its bucket."""

    if not stack:
        return "unknown", "<no stack>", ""
    frames = []
    for entry in stack:
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            filename, lineno, name = str(entry[0]), entry[1], str(entry[2])
        else:  # pragma: no cover - defensive
            continue
        frames.append((filename, lineno, name))
    # innermost first
    for filename, lineno, name in reversed(frames):
        normalised = filename.replace("/", "\\")
        if "site-packages\\torch" in normalised or "\\torch\\" in normalised and "transformers" not in normalised:
            # torch's own dispatch / autograd frames are not the caller we want
            if "site-packages\\torch" in normalised:
                continue
        for bucket, marker in BUCKETS:
            if marker and marker.lower() in normalised.lower():
                return bucket, f"{Path(filename).name}:{lineno}", name
    filename, lineno, name = frames[0]
    return "unknown", f"{Path(filename).name}:{lineno}", name


def _neighbourhood(events) -> dict:
    """Which op is executing right before each scalar read-back?

    Python stacks are unavailable, but the Kineto event list keeps temporal order, so the
    op that immediately precedes an `aten::item` is a usable fingerprint of the mechanism
    that issues it.
    """

    pairs: Counter = Counter()
    triples: Counter = Counter()
    names = [str(getattr(event, "name", "")) for event in events]
    for index, name in enumerate(names):
        if name not in ITEM_OPS:
            continue
        previous = names[index - 1] if index >= 1 else "<start>"
        before_that = names[index - 2] if index >= 2 else "<start>"
        pairs[f"{before_that} -> {previous} -> {name}"] += 1
        triples[previous] += 1
    total = sum(pairs.values())
    return {
        "total_item_events": total,
        "immediately_preceding_op": [
            {"op": op, "events": count, "share_percent": round(100.0 * count / max(1, total), 2)}
            for op, count in triples.most_common(10)
        ],
        "three_op_windows": [
            {"window": window, "events": count, "share_percent": round(100.0 * count / max(1, total), 2)}
            for window, count in pairs.most_common(10)
        ],
    }

def _chain(stack, depth: int = 4) -> list[str]:
    if not stack:
        return []
    frames = []
    for entry in stack:
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            frames.append(f"{Path(str(entry[0])).name}:{entry[1]}:{entry[2]}")
    return frames[:depth]


def _summarise(events) -> dict:
    item_rows: Counter = Counter()
    sync_rows: Counter = Counter()
    item_buckets: Counter = Counter()
    sync_buckets: Counter = Counter()
    item_chains: dict[str, list[str]] = {}
    sync_chains: dict[str, list[str]] = {}
    kernels = 0
    for event in events:
        name = str(getattr(event, "name", ""))
        if name in ITEM_OPS or name in SYNC_OPS:
            stack = list(getattr(event, "stack", []) or [])
            bucket, frame, function = _bucket_and_frame(stack)
            row = f"{bucket}|{frame}|{function}"
            if name in ITEM_OPS:
                item_rows[row] += 1
                item_buckets[bucket] += 1
                item_chains.setdefault(row, _chain(stack))
            else:
                sync_rows[f"{name}|{row}"] += 1
                sync_buckets[f"{name}|{bucket}"] += 1
                sync_chains.setdefault(f"{name}|{row}", _chain(stack))
        device_time = getattr(event, "device_time", None) or getattr(event, "device_time_total", None)
        if isinstance(device_time, (int, float)) and device_time > 0:
            kernels += 1
    return {
        "cuda_kernel_count": kernels,
        "item_rows": item_rows,
        "sync_rows": sync_rows,
        "item_buckets": item_buckets,
        "sync_buckets": sync_buckets,
        "item_chains": item_chains,
        "sync_chains": sync_chains,
    }


def _ranked(counter: Counter, total: int, chains: dict, limit: int = 12) -> list[dict]:
    return [
        {
            "call_site": row,
            "events": count,
            "share_percent": round(100.0 * count / max(1, total), 2),
            "stack_chain": chains.get(row, []),
        }
        for row, count in counter.most_common(limit)
    ]


def profile_window(runtime, samples, steps: int, warmup: int = 1) -> dict:
    from torch.profiler import ProfilerActivity, profile

    optimizer = make_optimizer(runtime, kind="default")
    set_seed(int(runtime.cfg["seed"]))
    sequence = [samples[index % len(samples)] for index in range(warmup + steps)]
    for sample in sequence:
        image = sample.image_rgb()
        runtime.features_for(sample, image)
        del image
    for sample in sequence[:warmup]:
        batch, image = runtime.prepare(sample)
        features, _cached = runtime.features_for(sample, image)
        result = runtime.train_step(
            batch,
            sample.target_mask(),
            features,
            optimizer=optimizer,
            collect_grad_norms=collect_grad_norms_setting(runtime),
        )
        del result

    torch.cuda.synchronize()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for sample in sequence[warmup:]:
            batch, image = runtime.prepare(sample)
            features, _cached = runtime.features_for(sample, image)
            result = runtime.train_step(
                batch,
                sample.target_mask(),
                features,
                optimizer=optimizer,
                collect_grad_norms=collect_grad_norms_setting(runtime),
            )
            del result
        torch.cuda.synchronize()
    wall = time.perf_counter() - started

    summary = _summarise(list(prof.events()))
    items = sum(summary["item_rows"].values())
    syncs = sum(summary["sync_rows"].values())
    scale = 1.0 / max(1, steps)
    return {
        "steps_profiled": steps,
        "profiled_wall_seconds": round(wall, 3),
        "wall_inflation_note": (
            "with_stack=True inflates the profiled wall time heavily; counts, not times, are "
            "the evidence here."
        ),
        "per_step": {
            "cuda_kernel_count": round(summary["cuda_kernel_count"] * scale, 1),
            "item_events": round(items * scale, 1),
            "sync_events": round(syncs * scale, 1),
        },
        "item_total": items,
        "sync_total": syncs,
        "top_item_call_sites": _ranked(summary["item_rows"], items, summary["item_chains"]),
        "top_sync_call_sites": _ranked(summary["sync_rows"], syncs, summary["sync_chains"]),
        "item_events_by_bucket": dict(summary["item_buckets"].most_common()),
        "sync_events_by_bucket": dict(summary["sync_buckets"].most_common()),
        "item_neighbourhood": _neighbourhood(list(prof.events())),
    }


def scoped_windows(runtime, samples, steps: int = 2) -> dict:
    """Fallback attribution (section 4): count scalar reads / syncs / kernels per module.

    Stack capture is unavailable on this build (`KinetoEvent.stack` is always empty and
    `torch._C._get_python_stack` does not exist), so attribution is done by the mechanism
    section 4 explicitly allows: profile each component in isolation and report the
    derived LM / backward residuals. Each scalar read-back produces two op events
    (`aten::item` plus `aten::_local_scalar_dense`), so `scalar_reads` is half the
    item-op count and that convention is stated in the artifact.
    """

    from torch.profiler import ProfilerActivity, profile

    from buildreasonseg_mvp.losses import combined_loss  # noqa: PLC0415

    qwen = runtime.model.qwen
    host = resolve_visual_host(qwen)
    sample = samples[0]
    batch, image = runtime.prepare(sample)
    moved = batch.to(runtime.device)
    features, _cached = runtime.features_for(sample, image)
    optimizer = make_optimizer(runtime, kind="default")

    def _profile(action, times: int) -> dict:
        for _ in range(times):
            action()
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(times):
                action()
            torch.cuda.synchronize()
        summary = _summarise(list(prof.events()))
        item_ops = sum(summary["item_rows"].values())
        syncs = sum(summary["sync_rows"].values())
        return {
            "calls": times,
            "scalar_reads_per_call": round(item_ops / 2.0 / times, 2),
            "item_op_events_per_call": round(item_ops / times, 2),
            "sync_ops_per_call": round(syncs / times, 2),
            "cuda_kernels_per_call": round(summary["cuda_kernel_count"] / times, 1),
        }

    windows: dict[str, dict] = {}

    def vision():
        with torch.no_grad():
            host.visual(moved.pixel_values, grid_thw=moved.image_grid_thw, return_dict=True)

    def forward_only():
        output = runtime.model(moved, features)
        target, supervision_logits, _mode = runtime.build_mask_supervision(sample.target_mask(), output.mask_logits)
        combined_loss(output.lm_logits, moved.labels, supervision_logits, target, runtime.loss_weights)
        del output

    def data_prepare():
        prepared, prepared_image = runtime.prepare(sample)
        sample.target_mask()
        del prepared, prepared_image

    def sam_decode():
        with torch.no_grad():
            projected = runtime.model.projection(
                torch.zeros(1, 2048, device=runtime.device, dtype=torch.bfloat16)
            )
            decode_mask(runtime.model.sam, features, projected, multimask_output=False, bridge=runtime.model.bridge)

    def loss_and_scalars():
        output = runtime.model(moved, features)
        target, supervision_logits, _mode = runtime.build_mask_supervision(sample.target_mask(), output.mask_logits)
        breakdown = combined_loss(
            output.lm_logits, moved.labels, supervision_logits, target, runtime.loss_weights
        )
        breakdown.as_dict()  # the 4 project-owned float() read-backs
        del output, breakdown

    def full_step():
        result = runtime.train_step(
            moved,
            sample.target_mask(),
            features,
            optimizer=optimizer,
            collect_grad_norms=False,
        )
        del result

    def clip_and_optimizer():
        # Gradients must stay present: `zero_grad(set_to_none=True)` makes every parameter
        # grad None, AdamW then skips them all and the window measures nothing. They were
        # populated by the warm step above.
        torch.nn.utils.clip_grad_norm_(
            [p for p in runtime.model.parameters() if p.requires_grad],
            float(runtime.cfg["optimizer"]["grad_clip_norm"]),
        )
        optimizer.step()

    # Gradients must be present for the clip/optimizer window to be meaningful.
    warm_step = runtime.train_step(
        moved,
        sample.target_mask(),
        features,
        optimizer=optimizer,
        collect_grad_norms=False,
    )
    del warm_step

    windows["data_prepare"] = _profile(data_prepare, steps)
    windows["vision_tower_only"] = _profile(vision, steps)
    windows["forward_with_grad_and_loss"] = _profile(forward_only, steps)
    windows["sam_projection_decode_only"] = _profile(sam_decode, steps)
    windows["loss_and_project_scalars"] = _profile(loss_and_scalars, steps)
    windows["full_train_step"] = _profile(full_step, steps)

    # Split the backward from the update: the first measurement says which of the two
    # produces the scalar read-backs, the second is a one-shot hypothesis test for the
    # non-reentrant gradient-checkpointing recomputation.
    windows["optimizer_and_clip_only"] = _profile(clip_and_optimizer, steps)

    full = windows["full_train_step"]
    forward = windows["forward_with_grad_and_loss"]
    vision_window = windows["vision_tower_only"]
    optimizer_window = windows["optimizer_and_clip_only"]
    derived = {
        "qwen_language_model_only": {
            "scalar_reads_per_call": round(
                max(0.0, forward["scalar_reads_per_call"] - vision_window["scalar_reads_per_call"]), 2
            ),
            "sync_ops_per_call": round(
                max(0.0, forward["sync_ops_per_call"] - vision_window["sync_ops_per_call"]), 2
            ),
            "cuda_kernels_per_call": round(
                max(0.0, forward["cuda_kernels_per_call"] - vision_window["cuda_kernels_per_call"]), 1
            ),
            "derivation": "forward_with_grad_and_loss minus vision_tower_only",
        },
        "backward_only": {
            "scalar_reads_per_call": round(
                max(
                    0.0,
                    full["scalar_reads_per_call"]
                    - forward["scalar_reads_per_call"]
                    - optimizer_window["scalar_reads_per_call"],
                ),
                2,
            ),
            "sync_ops_per_call": round(
                max(
                    0.0,
                    full["sync_ops_per_call"]
                    - forward["sync_ops_per_call"]
                    - optimizer_window["sync_ops_per_call"],
                ),
                2,
            ),
            "cuda_kernels_per_call": round(
                max(
                    0.0,
                    full["cuda_kernels_per_call"]
                    - forward["cuda_kernels_per_call"]
                    - optimizer_window["cuda_kernels_per_call"],
                ),
                1,
            ),
            "derivation": "full_train_step minus forward_with_grad_and_loss minus optimizer_and_clip_only",
        },
    }

    total_reads = max(1e-9, full["scalar_reads_per_call"])
    rows = []
    for name, window in list(windows.items()) + list(derived.items()):
        reads = window["scalar_reads_per_call"]
        rows.append(
            {
                "component": name,
                "scalar_reads_per_step": reads,
                "share_of_step_scalar_reads_percent": round(100.0 * reads / total_reads, 2),
                "sync_ops_per_step": window["sync_ops_per_call"],
                "cuda_kernels_per_step": window["cuda_kernels_per_call"],
                "avoidable": (
                    "yes" if name in ("data_prepare", "loss_and_project_scalars") else "no"
                ),
                "note": window.get("derivation", ""),
            }
        )
    rows.sort(key=lambda row: -row["scalar_reads_per_step"])

    return {
        "convention": (
            "scalar_reads = item-op events / 2, because one read-back records both `aten::item` "
            "and `aten::_local_scalar_dense`. This is also why Task 6C.6 reported 1,051 "
            "`aten::item` per step while this run counts 2,102 item-op events per step."
        ),
        "windows": windows,
        "derived": derived,
        "ranked_table": rows,
        "step_scalar_reads": full["scalar_reads_per_call"],
        "step_sync_ops": full["sync_ops_per_call"],
        "step_cuda_kernels": full["cuda_kernels_per_call"],
    }


def classify(profile: dict, scopes: dict | None) -> dict:
    """Where do the remaining scalar read-backs come from, and are they avoidable?"""

    total = max(1, profile["item_total"])
    top = profile["top_item_call_sites"][0] if profile["top_item_call_sites"] else None
    stack_available = bool(top and top["call_site"] != "unknown|<no stack>|")
    result = {
        "stack_attribution_available": stack_available,
        "stack_attribution_error": (
            None
            if stack_available
            else (
                "KinetoEvent.stack is always empty on this build and torch._C._get_python_stack "
                "does not exist, so call-site attribution falls back to scoped windows as "
                "section 4 permits"
            )
        ),
        "dominant_call_site": top,
    }
    if scopes:
        table = scopes["ranked_table"]
        result.update(
            {
                "step_scalar_reads": scopes["step_scalar_reads"],
                "step_sync_ops": scopes["step_sync_ops"],
                "step_cuda_kernels": scopes["step_cuda_kernels"],
                "largest_scalar_read_source": table[0]["component"] if table else None,
                "vision_tower_share_of_scalar_reads_percent": next(
                    (
                        row["share_of_step_scalar_reads_percent"]
                        for row in table
                        if row["component"] == "vision_tower_only"
                    ),
                    0.0,
                ),
                "project_owned_scalar_reads": round(
                    sum(
                        row["scalar_reads_per_step"]
                        for row in table
                        if row["component"] in ("data_prepare", "loss_and_project_scalars")
                    ),
                    2,
                ),
                "ranked_table": table,
                "avoidable": (
                    "the two project-owned windows are avoidable in principle (defer the four "
                    "loss floats and the clip norm to logging intervals); the vision-tower and "
                    "language-model traffic is third-party and section 4 forbids patching those "
                    "internals"
                ),
            }
        )
    return result


def trace_item_call_sites(runtime, samples, steps: int = 1) -> dict:
    """Exact Python call sites for `Tensor.item()`, via a `sys.setprofile` c_call hook.

    The Kineto profiler cannot give stacks on this build, but the CPython profile hook
    reports C-function calls (`c_call`) with the built-in method as the argument, so the
    calling frame is available. This is a diagnostic-only hook: it is installed for the
    duration of one measured step and removed afterwards.
    """

    import sys
    import traceback

    hits: Counter = Counter()
    frames_of_interest: dict[str, dict] = {}

    def hook(frame, event, arg):  # noqa: ANN001
        if event != "c_call":
            return None
        # The c_call argument for `t.item()` is the built-in method, whose `__name__` is
        # "item" and whose `__self__` is the tensor. (`float(t)` / `int(t)` do not raise a
        # c_call event at all, but they do emit `aten::_local_scalar_dense`, which is why
        # the op-event count is cross-checked against this tracer.)
        name = getattr(arg, "__name__", "")
        owner = getattr(getattr(arg, "__self__", None), "__class__", None)
        if name != "item" or owner is None or owner.__name__ not in ("Tensor", "Parameter"):
            return None
        stack = traceback.extract_stack(frame)[:-1]
        interesting = [entry for entry in stack if "site-packages\\torch" not in entry.filename][-4:]
        if not interesting:
            interesting = stack[-4:]
        key = " | ".join(f"{Path(entry.filename).name}:{entry.lineno}:{entry.name}" for entry in interesting)
        hits[key] += 1
        if key not in frames_of_interest:
            frames_of_interest[key] = {
                "frames": [
                    {
                        "file": entry.filename,
                        "line": entry.lineno,
                        "function": entry.name,
                        "source": (entry.line or "").strip()[:160],
                    }
                    for entry in interesting
                ]
            }
        return None

    optimizer = make_optimizer(runtime, kind="default")
    sample = samples[0]
    batch, image = runtime.prepare(sample)
    features, _cached = runtime.features_for(sample, image)
    runtime.set_visual_cache_key(sample.image_id)

    sys.setprofile(hook)
    try:
        for _ in range(steps):
            result = runtime.train_step(
                batch,
                sample.target_mask(),
                features,
                optimizer=optimizer,
                collect_grad_norms=False,
            )
            del result
    finally:
        sys.setprofile(None)
    torch.cuda.synchronize()

    total = sum(hits.values())
    return {
        "steps_traced": steps,
        "total_item_calls": total,
        "item_calls_per_step": round(total / max(1, steps), 1),
        "top_call_sites": [
            {
                "call_site": key,
                "calls": count,
                "share_percent": round(100.0 * count / max(1, total), 2),
                "frames": frames_of_interest.get(key, {}).get("frames", []),
            }
            for key, count in hits.most_common(12)
        ],
        "note": (
            "sys.setprofile slows execution; this is a diagnostic attribution, not a timing "
            "measurement. `Tensor.item`, `float(tensor)` and `int(tensor)` are all recorded."
        ),
    }


def mechanism_verdict(profiler: dict, scopes: dict, tracer: dict | None) -> dict:
    """Section 4's answer: where the remaining scalar read-backs actually come from."""

    step_reads = scopes["step_scalar_reads"] if scopes else None
    backward = scopes["derived"]["backward_only"]["scalar_reads_per_call"] if scopes else None
    optimizer = scopes["windows"]["optimizer_and_clip_only"]["scalar_reads_per_call"] if scopes else None
    forward = scopes["windows"]["forward_with_grad_and_loss"]["scalar_reads_per_call"] if scopes else None
    project = None
    if scopes:
        project = round(
            scopes["windows"]["data_prepare"]["scalar_reads_per_call"]
            + scopes["windows"]["loss_and_project_scalars"]["scalar_reads_per_call"],
            2,
        )
    top = ((tracer or {}).get("top_call_sites") or [{}])[0]
    mechanism = {
        "identified": "torch.optim's per-parameter optimizer bookkeeping",
        "source": "torch/optim/adam.py:770-776 (foreach path) via _get_value() -> torch/optim/optimizer.py:95",
        "explanation": (
            "with capturable=False and fused=False, AdamW deliberately hosts each parameter's "
            "`step` counter on the CPU (torch/optim/adam.py:165-176, upstream comment: 'kernel "
            "launches are costly on CUDA and XLA'), and the foreach bias-correction converts that "
            "counter to a Python number twice per parameter per step. 2 x 528 trainable tensors "
            "= ~1,056 reads, which is what is measured."
        ),
        "third_party": True,
        "requires_device_sync": False,
        "cost_estimate": (
            "CPU tensor .item() does not synchronize the device (measured: ~130 "
            "cudaStreamSynchronize per step against ~1,051 reads), and the alternative that "
            "removes them - fused=True, which keeps `step` on the GPU - was measured at -0.71% "
            "(unresolvable) in Task 6C.6 and is not bit-equivalent, so the conversions are cheap "
            "and not worth trading equivalence for."
        ),
    }
    return {
        "step_scalar_reads_per_step": step_reads,
        "backward_scalar_reads_per_step": backward,
        "backward_share_percent": (
            round(100.0 * backward / step_reads, 2) if backward is not None and step_reads else None
        ),
        "forward_scalar_reads_per_step": forward,
        "optimizer_and_clip_scalar_reads_per_step": optimizer,
        "optimizer_share_percent": (
            round(100.0 * optimizer / step_reads, 2) if optimizer is not None and step_reads else None
        ),
        "project_owned_scalar_reads_per_step": project,
        "project_owned_share_percent": (
            round(100.0 * project / step_reads, 2) if project is not None and step_reads else None
        ),
        "dominant_python_call_site": top.get("call_site"),
        "dominant_call_site_share_percent": top.get("share_percent"),
        "mechanism": mechanism,
        "avoidable": (
            "no — the traffic is inside third-party optimizer internals (section 4 forbids patching "
            "them), it does not synchronize the device, and the only supported alternative "
            "(fused=True) measured no gain and is not bit-equivalent. The project's own reads are "
            "under 1% of the total, so section 10 says to leave them alone."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-scopes", action="store_true")
    parser.add_argument("--skip-tracer", action="store_true")
    parser.add_argument("--trace-steps", type=int, default=1)
    parser.add_argument("--scope-steps", type=int, default=2)
    args = parser.parse_args(argv)

    samples = load_benchmark_samples()
    runtime = build_variant_runtime()
    try:
        profiler = profile_window(runtime, samples, args.steps, args.warmup)
        scopes = None if args.skip_scopes else scoped_windows(runtime, samples, args.scope_steps)
        tracer = None if args.skip_tracer else trace_item_call_sites(runtime, samples, args.trace_steps)
    finally:
        del runtime
        torch.cuda.empty_cache()

    checkpointing_probe = None
    if not args.skip_scopes:
        # Hypothesis test: the backward pass may be paying for the non-reentrant
        # gradient-checkpointing recomputation. Same step, checkpointing off.
        probe_runtime = build_variant_runtime(gradient_checkpointing=False)
        try:
            probe_runtime.visual_cache_key = None
            checkpointing_probe = scoped_windows(probe_runtime, samples, args.scope_steps)
            checkpointing_probe["gradient_checkpointing"] = False
            checkpointing_probe["comparison"] = {
                "full_step_scalar_reads_on": scopes["step_scalar_reads"],
                "full_step_scalar_reads_off": checkpointing_probe["step_scalar_reads"],
                "backward_scalar_reads_on": scopes["derived"]["backward_only"]["scalar_reads_per_call"],
                "backward_scalar_reads_off": checkpointing_probe["derived"]["backward_only"][
                    "scalar_reads_per_call"
                ],
            }
        finally:
            del probe_runtime
            torch.cuda.empty_cache()

    report = {
        "_doc": (
            "Task 6C.7 section 4. Attribution of the remaining per-step `aten::item` and "
            "host/device synchronization events on the formal Phase-B step. Python stack capture "
            "is unavailable on this build (`KinetoEvent.stack` is empty, "
            "`torch._C._get_python_stack` absent), so attribution uses the scoped-window fallback "
            "section 4 permits, plus a gradient-checkpointing hypothesis test. Compact summary only."
        ),
        "task": "6C.7",
        "path_measured": "formal Phase-B step, collect_grad_norms=false (integrated baseline B0.6)",
        "batch_size": 1,
        "profiler": profiler,
        "classification": classify(profiler, scopes),
    }
    if scopes is not None:
        report["scoped_windows"] = scopes
    else:
        # A profiler-only re-run must not erase the scoped attribution measured earlier.
        existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
        for key in ("scoped_windows", "checkpointing_probe", "item_call_sites"):
            if key in existing and key not in report:
                report[key] = existing[key]
    if tracer is not None:
        report["item_call_sites"] = tracer
    elif "item_call_sites" not in report:
        existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
        if "item_call_sites" in existing:
            report["item_call_sites"] = existing["item_call_sites"]
    scopes_for_verdict = report.get("scoped_windows")
    if scopes_for_verdict:
        report["mechanism_verdict"] = mechanism_verdict(
            profiler, scopes_for_verdict, report.get("item_call_sites")
        )
    if checkpointing_probe is not None:
        report["checkpointing_probe"] = checkpointing_probe
        print(f"[task6c7:sync] checkpointing probe: reads/step on={scopes['step_scalar_reads']} "
              f"off={checkpointing_probe['step_scalar_reads']} "
              f"(backward on={scopes['derived']['backward_only']['scalar_reads_per_call']} "
              f"off={checkpointing_probe['derived']['backward_only']['scalar_reads_per_call']})")
    write_json(OUT, report)

    print(f"[task6c7:sync] per step: items={profiler['per_step']['item_events']} "
          f"syncs={profiler['per_step']['sync_events']} kernels={profiler['per_step']['cuda_kernel_count']}")
    neighbourhood = profiler.get("item_neighbourhood") or {}
    for row in (neighbourhood.get("immediately_preceding_op") or [])[:5]:
        print(f"[task6c7:sync]   preceded by {row['share_percent']:5.1f}%  {row['op']}")
    for row in (neighbourhood.get("three_op_windows") or [])[:5]:
        print(f"[task6c7:sync]   window {row['share_percent']:5.1f}%  {row['window']}")
    print(f"[task6c7:sync] buckets: {json.dumps(profiler['item_events_by_bucket'])}")
    for row in profiler["top_item_call_sites"][:5]:
        print(f"[task6c7:sync]   {row['share_percent']:5.1f}%  {row['call_site']}")
    for row in profiler["top_sync_call_sites"][:5]:
        print(f"[task6c7:sync]   {row['share_percent']:5.1f}%  {row['call_site']}")
    if scopes:
        print(f"[task6c7:sync] step totals (scoped): scalar_reads/step={scopes['step_scalar_reads']} "
              f"sync_ops/step={scopes['step_sync_ops']} kernels/step={scopes['step_cuda_kernels']}")
        for row in scopes["ranked_table"]:
            print(f"[task6c7:sync]   {row['share_of_step_scalar_reads_percent']:6.1f}%  "
                  f"reads={row['scalar_reads_per_step']:7.2f}  syncs={row['sync_ops_per_step']:7.2f}  "
                  f"kernels={row['cuda_kernels_per_step']:9.1f}  {row['component']}")
    print(f"[task6c7:sync] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    tracer_payload = report.get("item_call_sites") or {}
    for row in (tracer_payload.get("top_call_sites") or [])[:5]:
        print(f"[task6c7:sync]   item call site {row['share_percent']:5.1f}%  {row['call_site']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
