#!/usr/bin/env python
"""Task 6D.1 sections 2 and 14: scheduler-horizon audit.

    python scripts/task6d1_scheduler_audit.py

Records, side by side:

* the **defective** Task 6D horizon (the scheduler was constructed with one epoch but two
  epochs were run, so the cosine factor reached its terminal value at the end of epoch 1
  and epoch 2 trained at LR ~0);
* the **corrected** horizon (the real optimizer-step budget), checkable independently of
  the run;
* the **observed** per-group LR at the five checkpoints required by section 2, taken from
  `evaluation/task6d1_g0_corrected.json` when the corrective G0-R has run.

Writes `evaluation/task6d1_scheduler_audit.json`. No GPU work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp.runtime import load_config  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
OUT = EVAL / "task6d1_scheduler_audit.json"
G0_CORRECTED = EVAL / "task6d1_g0_corrected.json"
G0_ORIGINAL = EVAL / "task6d_g0.json"


def factor(step: int, horizon: int, warmup_steps: int) -> float:
    """The project's scheduler factor, reproduced exactly."""

    if warmup_steps and step < warmup_steps:
        return max(1e-3, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, horizon - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _verify_observed(report: dict) -> dict:
    """Recompute the section-2 LR checks from the recorded trace.

    Reading the trace rather than trusting a stored flag keeps the audit authoritative even if the
    training report used an older criterion. "Terminal" is a ratio (the cosine factor at the final
    step is ~2.8e-06 of peak), not an absolute zero.
    """

    audit = report.get("lr_audit") or {}
    trace = audit.get("trace") or []
    checkpoint_epoch2 = (audit.get("checkpoints") or {}).get("start_of_epoch_2") or {}
    if not trace:
        return {"available": False, "reason": "the corrected run recorded no LR trace"}
    peak = max((max(record["lrs"]) for record in trace if record["lrs"]), default=0.0)
    final = trace[-1]
    final_max = max(final["lrs"]) if final["lrs"] else 0.0
    total = int((report.get("scheduler_horizon") or {}).get("total_optimizer_steps") or 0)
    return {
        "available": True,
        "trace_points": len(trace),
        "peak_lr_over_run": peak,
        "first_step_lrs": (trace[0]["lrs"] if trace else None),
        "epoch_2_start_lrs": checkpoint_epoch2.get("lrs"),
        "final_step": final["global_step"],
        "final_lrs": final["lrs"],
        "final_lr_over_peak_ratio": (final_max / peak) if peak else None,
        "epoch_2_initial_lr_nonzero": bool(
            checkpoint_epoch2 and any(value > 0.0 for value in checkpoint_epoch2["lrs"])
        ),
        "terminal_at_final_scheduled_step": bool(
            total and final["global_step"] == total and peak > 0 and (final_max / peak) <= 1e-5
        ),
        "mid_run_is_not_terminal": bool(
            len(trace) > 2 and max(trace[len(trace) // 2]["lrs"]) > 0.1 * peak
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=480)
    args = parser.parse_args(argv)

    cfg = load_config(CONFIG)
    lora_lr = float(cfg["optimizer"]["phase_b"]["lora_lr"])
    token_lr = float(cfg["optimizer"]["phase_b"]["token_lr"])
    decoder_lr = float(cfg["optimizer"]["phase_b"].get("decoder_lr", token_lr))
    warmup_steps = int(cfg["optimizer"]["phase_b"]["warmup_steps"])
    steps_per_epoch = args.steps_per_epoch
    epochs = args.epochs
    total = epochs * steps_per_epoch

    def lrs(factor_value: float) -> dict:
        return {
            "lora": lora_lr * factor_value,
            "trainable_token": token_lr * factor_value,
            "decoder_and_grounding_head": decoder_lr * factor_value,
        }

    checkpoints = {
        "first_optimizer_step": (0, "step 0 (before the first scheduler.step)"),
        "after_warmup": (warmup_steps, f"step {warmup_steps}"),
        "end_of_epoch_1": (steps_per_epoch - 1, f"step {steps_per_epoch - 1}"),
        "start_of_epoch_2": (steps_per_epoch, f"step {steps_per_epoch}"),
        "final_optimizer_step": (total - 1, f"step {total - 1}"),
    }

    defective = {
        name: {"global_step": step, "label": label, "factor": factor(step, steps_per_epoch, warmup_steps),
               "lrs": lrs(factor(step, steps_per_epoch, warmup_steps))}
        for name, (step, label) in checkpoints.items()
    }
    corrected = {
        name: {"global_step": step, "label": label, "factor": factor(step, total, warmup_steps),
               "lrs": lrs(factor(step, total, warmup_steps))}
        for name, (step, label) in checkpoints.items()
    }

    original_report = json.loads(G0_ORIGINAL.read_text(encoding="utf-8")) if G0_ORIGINAL.is_file() else {}
    corrected_report = (
        json.loads(G0_CORRECTED.read_text(encoding="utf-8")) if G0_CORRECTED.is_file() else {}
    )

    report = {
        "_doc": (
            "Task 6D.1 section 2. Scheduler-horizon audit for the Task 6D G0 run and its corrective "
            "rerun. The defective horizon gave epoch 2 an LR of exactly zero; the corrected horizon "
            "spreads the cosine over the real step budget. The observed block is filled from the "
            "corrective run when it exists."
        ),
        "task": "6D.1",
        "config": {
            "lora_lr": lora_lr,
            "token_lr": token_lr,
            "decoder_lr": decoder_lr,
            "warmup_steps": warmup_steps,
            "lr_schedule": cfg["optimizer"]["phase_b"]["lr_schedule"],
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
        },
        "defect": {
            "summary": (
                "scripts/task6d_train.py built the AdamW scheduler with `steps_per_epoch` as its "
                "horizon and then ran two epochs. The cosine factor therefore reached its terminal "
                "value at the end of epoch 1, and epoch 2 trained with LR exactly 0."
            ),
            "horizon_used": steps_per_epoch,
            "steps_actually_run": total,
            "consequences": [
                "epoch 1 decayed to ~1e-5 of peak before its own end, so only the first few hundred "
                "steps had a meaningful learning rate",
                "epoch 2 performed no optimisation at all, which is why the Task 6D epoch-1 and "
                "epoch-2 validation metrics were bit-identical",
            ],
            "checkpoints": defective,
        },
        "correction": {
            "summary": (
                "The scheduler is constructed with the actual optimizer-step budget "
                "(epochs x steps_per_epoch, or the truncated smoke budget), so the terminal factor "
                "is reached only at the final step of the last epoch."
            ),
            "horizon_used": total,
            "checkpoints": corrected,
        },
        "verification": {
            "epoch_2_initial_factor_defective": defective["start_of_epoch_2"]["factor"],
            "epoch_2_initial_factor_corrected": corrected["start_of_epoch_2"]["factor"],
            "terminal_at_final_step_corrected": corrected["final_optimizer_step"]["factor"],
            "epoch_2_initial_lr_nonzero_after_fix": bool(
                any(value > 0.0 for value in corrected["start_of_epoch_2"]["lrs"].values())
            ),
        },
        "observed_original_run": {
            "report": "evaluation/task6d_g0.json",
            "scheduler_horizon": original_report.get("scheduler_horizon"),
            "lr_audit": original_report.get("lr_audit"),
            "note": (
                "Task 6D's report predates the LR trace, so its LRs are reconstructed analytically "
                "in `defect.checkpoints` from the same scheduler formula."
            ),
        },
        "observed_corrected_run": {
            "report": "evaluation/task6d1_g0_corrected.json",
            "scheduler_horizon": corrected_report.get("scheduler_horizon"),
            "lr_audit": corrected_report.get("lr_audit"),
            "available": bool(corrected_report),
            "verification": _verify_observed(corrected_report),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "[task6d1:audit] defective horizon "
        f"{steps_per_epoch}: epoch-2 start factor {defective['start_of_epoch_2']['factor']:.3e}, "
        f"end-of-epoch-1 factor {defective['end_of_epoch_1']['factor']:.3e}"
    )
    print(
        f"[task6d1:audit] corrected horizon {total}: epoch-2 start factor "
        f"{corrected['start_of_epoch_2']['factor']:.4f}, final factor "
        f"{corrected['final_optimizer_step']['factor']:.3e}"
    )
    if corrected_report:
        audit = corrected_report.get("lr_audit") or {}
        print(f"[task6d1:audit] observed epoch-2 start LR: "
              f"{(audit.get('checkpoints', {}).get('start_of_epoch_2') or {}).get('lrs')}")
    print(f"[task6d1:audit] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
