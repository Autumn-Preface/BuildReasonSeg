"""Task 6R section 9 — deterministic GRCL v0.1 gradient audit and directional sanity check.

Gates (failure means `GRCL_IMPLEMENTATION_INVALID`):

* at least 8 deterministic soft target-logit tensors, all four relations represented, non-symmetric
  reference masks, backprop of `L_GRCL`;
* the loss is finite, the target-logit gradient exists, is finite, and has L1 norm > 1e-8;
* directional sanity: moving a synthetic target centroid farther into the correct relation must not
  increase GRCL, and moving it across the reference to the wrong side must increase GRCL.

Writes `evaluation/task6r_grcl_audit.json`.

    python scripts/task6r_grcl_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.grcl_directional import (  # noqa: E402
    ALPHA,
    EPS,
    LAMBDA_GRCL,
    RELATIONS,
    TAU,
    grcl_directional,
)
from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6r_grcl_audit.json"
SIZE = (64, 64)
MIN_GRADIENT_L1 = 1e-8


def reference_mask(size=SIZE, centre=(0.30, 0.30), sigma=0.10) -> torch.Tensor:
    """Non-symmetric deterministic reference blob (never centred)."""

    height, width = size
    rows = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    blob = torch.exp(-(((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * sigma * sigma)))
    return (blob > 0.5).float()


def blob_logits(size=SIZE, centre=(0.7, 0.7), sigma=0.10, scale=10.0) -> torch.Tensor:
    """Soft logits whose sigmoid mass is concentrated at `centre` (differentiable, not thresholded).

    The plateau is `+scale` and the background floor `-scale`, so the uniform floor contributes a
    negligible fraction of the probability mass and the soft centroid lands on `centre` (a shallow
    blob would be dragged toward the image centre by its own background).
    """

    height, width = size
    rows = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    distance = ((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * sigma * sigma)
    return (2.0 * scale * torch.exp(-distance) - scale).view(1, 1, height, width)


def directional_axis(relation: str, centre: tuple[float, float], delta: float) -> tuple[float, float]:
    """Move a synthetic target centre along the relation's primary axis by `delta`."""

    cx, cy = centre
    if relation == "left_of":
        return cx - delta, cy
    if relation == "right_of":
        return cx + delta, cy
    if relation == "above":
        return cx, cy - delta
    return cx, cy + delta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.time()
    rows = []
    for index in range(8):
        relation = RELATIONS[index % len(RELATIONS)]
        centre_ref = (0.30 + 0.05 * (index % 4), 0.30 + 0.04 * (index % 3))
        reference = reference_mask(centre=centre_ref, sigma=0.08 + 0.01 * (index % 3))
        # the synthetic target is deliberately placed on the WRONG side of the reference so that both
        # hinge terms are active; otherwise the loss is saturated at zero and the gradient is trivially
        # zero, which would make the audit vacuous.
        target_centre = directional_axis(relation, centre_ref, -0.06 - 0.01 * (index % 2))
        logits = blob_logits(centre=target_centre, sigma=0.10).clone().requires_grad_(True)
        report = grcl_directional(logits, reference, relation)
        report.loss.backward()
        gradient = logits.grad
        finite = bool(torch.isfinite(report.loss))
        gradient_finite = gradient is not None and bool(torch.isfinite(gradient).all())
        l1 = float(gradient.abs().sum()) if gradient is not None else 0.0
        reference_binary = reference > 0.5
        non_symmetric = bool(
            not torch.equal(reference_binary, torch.flip(reference_binary, dims=[-1]))
            or not torch.equal(reference_binary, torch.flip(reference_binary, dims=[-2]))
        )
        margin_term = float(report.margin.mean().detach())
        axis_term = float(report.axis.mean().detach())
        rows.append(
            {
                "index": index,
                "relation": relation,
                "target_centre": list(target_centre),
                "reference_centre": list(centre_ref),
                "loss_finite": finite,
                "loss": float(report.loss.detach()),
                "margin_term": margin_term,
                "axis_term": axis_term,
                "hinge_active": bool(margin_term > 0.0 and axis_term > 0.0),
                "gradient_exists": gradient is not None,
                "gradient_finite": gradient_finite,
                "gradient_l1": l1,
                "reference_non_symmetric": non_symmetric,
                "passed": bool(finite and gradient is not None and gradient_finite
                               and l1 > MIN_GRADIENT_L1 and non_symmetric
                               and margin_term > 0.0 and axis_term > 0.0),
            }
        )

    # ---------------- directional sanity per relation
    sanity = []
    for relation in RELATIONS:
        reference = reference_mask(centre=(0.5, 0.5), sigma=0.10)
        reference_tensor = reference
        base_centre = (0.5, 0.5)
        distances = (0.10, 0.20, 0.30)
        losses = []
        for distance in distances:
            centre = directional_axis(relation, base_centre, distance)
            logits = blob_logits(centre=centre, sigma=0.10)
            losses.append(float(grcl_directional(logits, reference_tensor, relation).loss))
        # farther into the correct relation must not increase GRCL
        monotone = losses[0] >= losses[1] >= losses[2] - 1e-9
        # crossing to the wrong side must increase GRCL
        wrong_centre = directional_axis(relation, base_centre, -0.25)
        wrong = float(grcl_directional(blob_logits(centre=wrong_centre, sigma=0.10),
                                       reference_tensor, relation).loss)
        correct = losses[-1]
        sanity.append(
            {
                "relation": relation,
                "losses_at_distances": losses,
                "monotone_non_increasing": bool(monotone),
                "wrong_side_loss": wrong,
                "correct_side_loss": correct,
                "wrong_side_higher": bool(wrong > correct),
                "passed": bool(monotone and wrong > correct),
            }
        )

    gradient_ok = all(row["passed"] for row in rows)
    sanity_ok = all(row["passed"] for row in sanity)
    passed = bool(gradient_ok and sanity_ok and len(rows) >= 8)
    payload = {
        "_doc": (
            "Task 6R section 9. Deterministic GRCL v0.1 audit: gradient existence/finiteness/L1 on "
            "non-binary soft target logits with non-symmetric reference masks across all four "
            "relations, plus the numerical directional sanity check."
        ),
        "task": "6R",
        "stage": "C-grcl-audit",
        "constants": {"alpha": ALPHA, "tau": TAU, "eps": EPS, "lambda_grcl": LAMBDA_GRCL},
        "size": list(SIZE),
        "module": str(REPO_ROOT / "buildreasonseg_mvp" / "grcl_directional.py"),
        "module_sha256": hashlib.sha256(
            (REPO_ROOT / "buildreasonseg_mvp" / "grcl_directional.py").read_bytes()
        ).hexdigest(),
        "min_gradient_l1": MIN_GRADIENT_L1,
        "gradient_rows": rows,
        "directional_sanity": sanity,
        "gradient_check_passed": gradient_ok,
        "directional_sanity_passed": sanity_ok,
        "verdict": "GRCL_VALID" if passed else "GRCL_IMPLEMENTATION_INVALID",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(Path(args.out), payload)
    print(
        f"[6r.audit] gradient L1 min {min(row['gradient_l1'] for row in rows):.3e} "
        f"(> {MIN_GRADIENT_L1:.0e}); directional sanity {sanity_ok} -> {payload['verdict']}",
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
