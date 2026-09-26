#!/usr/bin/env python
"""Task 6C section 4.1: make `training.deterministic` real and prove what it buys.

    python scripts/task6c_determinism.py

Runs a 2-sample forward/backward (+ one optimizer step) **twice, in two separate
processes** with the same seed, and compares the losses, the gradient digest and the
post-step trainable-state digest. Cross-process comparison is the point: Task 6B's
Phase B was reproducible within a process but not across processes.

Strict `torch.use_deterministic_algorithms(True)` is tried first. If an operation on
the training path has no deterministic implementation, the exact error is recorded
and the run falls back to `warn_only=True`, with bit reproducibility explicitly NOT
claimed.

Writes `evaluation/task6c_determinism.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6c_determinism.json"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"


def _once(mode: str, out_path: Path) -> int:
    import torch

    from buildreasonseg_mvp import data as data_mod
    from buildreasonseg_mvp.determinism import gradients_fingerprint, trainable_state_fingerprint
    from buildreasonseg_mvp.runtime import build_runtime, load_config, set_phase_trainables

    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}
    runtime = build_runtime(cfg, device="cuda", verbose=False, deterministic_strict=(mode == "strict"))
    set_phase_trainables(runtime.model, "B")

    optimizer = torch.optim.AdamW(
        runtime.model.trainable_parameter_groups(
            lora_lr=1e-4, head_lr=3e-4, weight_decay=0.01, decoder_lr=3e-4, token_lr=3e-4
        ),
        betas=(0.9, 0.999),
    )

    payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    sample_ids = payload["U"]["sample_ids"][:2]

    before = trainable_state_fingerprint(runtime.model)
    losses: list[dict] = []
    error = None
    try:
        for sample_id in sample_ids:
            sample = data_mod.to_sample(train[sample_id])
            batch, image = runtime.prepare(sample)
            features, _cached = runtime.features_for(sample, image)
            result = runtime.train_step(batch, sample.target_mask(), features, optimizer=None)
            losses.append({key: repr(float(value)) for key, value in result["losses"].items()})
        grads = gradients_fingerprint([p for p in runtime.model.parameters() if p.requires_grad])
        optimizer.step()
        after = trainable_state_fingerprint(runtime.model)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        grads = None
        after = None

    out_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "determinism_report": runtime.reports["determinism"],
                "sample_ids": sample_ids,
                "losses": losses,
                "gradients": grads,
                "initial_fingerprint": before["sha256"],
                "post_step_fingerprint": after["sha256"] if after else None,
                "error": error,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mode": mode, "error": error, "losses": losses}, ensure_ascii=False))
    return 0 if error is None else 3


def _run_child(mode: str, out_path: Path) -> dict:
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--once", "--mode", mode, "--out", str(out_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    payload["returncode"] = proc.returncode
    payload["seconds"] = round(time.time() - started, 1)
    if proc.returncode != 0:
        payload.setdefault("stderr_tail", (proc.stderr or "")[-2000:])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--mode", default="strict", choices=("strict", "warn_only"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.once:
        return _once(args.mode, Path(args.out))

    work = REPO_ROOT / "artifacts" / "task6c_determinism"
    work.mkdir(parents=True, exist_ok=True)

    first = _run_child("strict", work / "strict_run1.json")
    strict_supported = first.get("error") is None
    strict_error = first.get("error")
    mode = "strict" if strict_supported else "warn_only"

    if strict_supported:
        second = _run_child("strict", work / "strict_run2.json")
    else:
        print(f"[determinism] strict mode failed: {strict_error}", flush=True)
        print("[determinism] falling back to torch.use_deterministic_algorithms(True, warn_only=True)", flush=True)
        first = _run_child("warn_only", work / "warn_run1.json")
        second = _run_child("warn_only", work / "warn_run2.json")

    def compare(key: str):
        return first.get(key) == second.get(key)

    losses_identical = compare("losses")
    grads_identical = compare("gradients")
    params_identical = compare("post_step_fingerprint")
    bit_reproducible = bool(losses_identical and grads_identical and params_identical)

    report = {
        "_doc": (
            "Task 6C section 4.1 determinism evidence. Two independent processes, same seed, "
            "2-sample forward/backward plus one optimizer step. `training.deterministic` is now "
            "consumed by build_runtime (it was a dead flag in Task 6B)."
        ),
        "task": "6C",
        "config": str(CONFIG.relative_to(REPO_ROOT)),
        "strict_requested": True,
        "strict_supported": strict_supported,
        "strict_error": strict_error,
        "effective_mode": mode,
        "use_deterministic_algorithms_warn_only": mode == "warn_only",
        "measure": {
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_manual_seed": True,
            "torch_cuda_manual_seed_all": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": first.get("determinism_report", {}).get("cublas_workspace_config"),
        },
        "losses_identical": losses_identical,
        "gradients_identical": grads_identical,
        "post_step_parameters_identical": params_identical,
        "cross_process_bit_reproducible": bit_reproducible,
        "bit_reproducibility_claimed": bool(bit_reproducible and mode == "strict"),
        "run1": first,
        "run2": second,
        "seconds": round(first.get("seconds", 0) + second.get("seconds", 0), 1),
    }
    work.mkdir(parents=True, exist_ok=True)
    (EVAL).mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[determinism] strict supported: {strict_supported}")
    print(f"[determinism] losses identical: {losses_identical}  gradients identical: {grads_identical}  "
          f"post-step params identical: {params_identical}")
    print(f"[determinism] cross-process bit reproducible: {bit_reproducible} "
          f"(claimed: {report['bit_reproducibility_claimed']})")
    print(f"[determinism] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
