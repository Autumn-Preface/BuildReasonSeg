"""Task 6S — directional natural-language end-to-end chain.

The exact primary chain (Task 6S section 1), all modules frozen:

```text
natural-language instruction
    -> frozen Qwen3-VL-2B ProgramHead            (text only; 20 canonical program ids)
    -> canonical program id
    -> deterministic decomposition: reference_family + relation
    -> frozen Task 6Q YOLO26m-seg proposal reference resolver
    -> predicted reference mask
    -> GeometricRelationField v0.2
    -> P_rel
    -> frozen SAM2.1 Hiera Base+ visual feature + P_rel + relation embedding
    -> frozen Task 6O B3 target decoder
    -> target mask
```

**No GRCL v0.1 in this chain** (Task 6R kept only as a negative ablation). No oracle reference mask and
no annotation/record input: the inference path receives only an image, a natural-language prompt and the
frozen checkpoints/configs. The proposal system grounds the *reference* only — the final target comes
from the dense B3 visual decoder, never from a deterministic target proposal.

Exit codes: `0` answered, `3` reference abstention, `4` out-of-domain prompt (before ProgramHead and
before any downstream model), `5` valid-but-unsupported canonical program (before proposal/SAM2/B3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]

TILE_SIZE = 512
FIELD_SIZE = (64, 64)
TARGET_SIZE = (512, 512)

EXIT_ANSWERED = 0
EXIT_REFERENCE_ABSTENTION = 3
EXIT_UNSUPPORTED_INSTRUCTION = 4
EXIT_UNSUPPORTED_DIRECTIONAL = 5

STATUS_OK = "ok"
STATUS_REFERENCE_ABSTENTION = "reference_abstention"
STATUS_UNSUPPORTED_INSTRUCTION = "unsupported_instruction"
STATUS_UNSUPPORTED_DIRECTIONAL = "unsupported_directional_program"

#: Task 6S section 9 — exactly eight supported programs with their exact decomposition (section 10).
PROGRAM_DECOMPOSITION: dict[str, tuple[str, str]] = {
    "largest_to_left_of": ("largest", "left_of"),
    "largest_to_right_of": ("largest", "right_of"),
    "largest_to_above": ("largest", "above"),
    "largest_to_below": ("largest", "below"),
    "smallest_to_left_of": ("smallest", "left_of"),
    "smallest_to_right_of": ("smallest", "right_of"),
    "smallest_to_above": ("smallest", "above"),
    "smallest_to_below": ("smallest", "below"),
}
SUPPORTED_PROGRAMS: tuple[str, ...] = tuple(PROGRAM_DECOMPOSITION)
RELATION_ORDER = ("left_of", "right_of", "above", "below")

#: Frozen assets (Task 6S sections 5-7).
PROGRAM_HEAD_CHECKPOINT_SHA256 = "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"
DEFAULT_PROGRAM_HEAD_CANDIDATES = (
    Path("artifacts") / "checkpoints" / "task6m" / "program_parser_v02_best.pt",
    Path("artifacts") / "checkpoints" / "task6j" / "j2_best.pt",
)
#: Task 6T hardened ProgramHead (semantic hardening); preferred by the CLI when present.
HARDENED_PROGRAM_HEAD = Path("artifacts") / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_program_head_checkpoint(explicit: Path | None = None) -> tuple[Path | None, list[dict]]:
    """Resolve the frozen ProgramHead.

    With an explicit path (added by Task 6T for the hardened checkpoint) that file is used as-is and its
    hash is reported. Without an explicit path the Task 6S baseline preference is preserved unchanged: the
    candidate whose SHA256 matches the Task 6S expected frozen hash.
    """

    if explicit is not None:
        candidate = Path(explicit)
        present = candidate.is_file()
        digest = sha256_file(candidate) if present else None
        report = [{"path": str(candidate), "present": present, "sha256": digest,
                   "matches_expected": digest == PROGRAM_HEAD_CHECKPOINT_SHA256,
                   "role": "explicit"}]
        return (candidate if present else None), report

    candidates = [REPO_ROOT / candidate for candidate in DEFAULT_PROGRAM_HEAD_CANDIDATES]
    report = []
    for candidate in candidates:
        present = candidate.is_file()
        digest = sha256_file(candidate) if present else None
        report.append({"path": str(candidate), "present": present, "sha256": digest,
                       "matches_expected": digest == PROGRAM_HEAD_CHECKPOINT_SHA256,
                       "role": "task6s_baseline_candidate"})
        if present and digest == PROGRAM_HEAD_CHECKPOINT_SHA256:
            return candidate, report
    return None, report


def default_program_head_checkpoint() -> tuple[Path | None, list[dict]]:
    """Task 6T section 9: the hardened ProgramHead when present, else the Task 6S baseline."""

    hardened = REPO_ROOT / HARDENED_PROGRAM_HEAD
    if hardened.is_file():
        digest = sha256_file(hardened)
        return hardened, [{
            "path": str(hardened), "present": True, "sha256": digest,
            "matches_expected": digest == PROGRAM_HEAD_CHECKPOINT_SHA256,
            "role": "task6t_hardened_default",
            "note": "Task 6T hardened ProgramHead used as the directional CLI default",
        }]
    return resolve_program_head_checkpoint()


def decompose_program(program_id: str) -> tuple[str, str] | None:
    """Exact program -> (reference_family, relation); `None` when outside the 6S scope."""

    return PROGRAM_DECOMPOSITION.get(str(program_id))


@dataclass
class ChainModels:
    """Loaded frozen modules plus the resolved checkpoint provenance."""

    parser_runtime: object
    proposal_model: object
    target_model: object
    feature_store: object
    device: str
    parser_checkpoint: Path | None = None
    proposal_checkpoint: Path | None = None
    target_checkpoint: Path | None = None
    provenance: dict = dataclass_field(default_factory=dict)


def parse_instruction(runtime, prompt: str, program_ids: tuple[str, ...]) -> dict:
    """Text-only ProgramHead classification through the frozen runtime inference path.

    Uses `runtime.predict` (the exact frozen text-only entry point already used by the Task 6M
    structured CLI) and additionally reports the top-1 softmax confidence from the same batch.
    """

    import torch

    program = runtime.predict(prompt)
    confidence = None
    try:
        batch = runtime.build_batch([prompt], [program_ids[0]]).to(runtime.device)
        with torch.no_grad():
            logits, _hidden = runtime.forward(batch)
        probabilities = torch.softmax(logits.float(), dim=-1)[0]
        confidence = float(probabilities.max().item())
    except Exception:  # noqa: BLE001 - confidence is reporting-only and must never break inference
        confidence = None
    return {
        "program": program,
        "confidence": confidence,
        "model_input": "instruction text only",
        "image_tokens": False,
    }


def select_reference_mask(proposals, family: str):
    """Frozen Task 6Q deterministic resolver (unchanged eligibility/ranking)."""

    from buildreasonseg_mvp.task6q_reference_resolver import select_reference

    return select_reference(proposals, family)


@torch.no_grad()
def run_directional_chain(
    models: ChainModels,
    image_path: Path,
    prompt: str,
    *,
    program_ids: tuple[str, ...],
    domain_gate,
    want_visuals: bool = False,
    timing: dict | None = None,
) -> dict:
    """Run the full primary chain for one image + one natural-language prompt.

    Returns a result dict with `status`, `exit_code`, the parser/domain-gate trace, the reference
    resolution detail, the field statistics, the predicted target mask and a runtime breakdown.
    """

    from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02
    from buildreasonseg_mvp.task6q_reference_resolver import run_frozen_proposals

    timing = timing if timing is not None else {}
    started = time.perf_counter()
    image_path = Path(image_path)

    # ---------------- 1. deterministic domain gate (before ProgramHead and before any model)
    gate = domain_gate(prompt)
    if not gate.get("supported", False):
        return {
            "status": STATUS_UNSUPPORTED_INSTRUCTION,
            "exit_code": EXIT_UNSUPPORTED_INSTRUCTION,
            "prompt": prompt,
            "parsed_program": None,
            "reference_family": None,
            "relation": None,
            "domain_gate": gate,
            "target_mask": None,
            "ground_truth_used": False,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }

    # ---------------- 2. frozen ProgramHead (text only)
    mark = time.perf_counter()
    parsed = parse_instruction(models.parser_runtime, prompt, program_ids)
    timing["parser"] = round(time.perf_counter() - mark, 3)

    # ---------------- 3. deterministic decomposition and scope check
    decomposition = decompose_program(parsed["program"])
    if decomposition is None:
        return {
            "status": STATUS_UNSUPPORTED_DIRECTIONAL,
            "exit_code": EXIT_UNSUPPORTED_DIRECTIONAL,
            "prompt": prompt,
            "parsed_program": parsed["program"],
            "parser": parsed,
            "reference_family": None,
            "relation": None,
            "domain_gate": gate,
            "target_mask": None,
            "ground_truth_used": False,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
    family, relation = decomposition

    # ---------------- 4. frozen proposal reference resolver
    mark = time.perf_counter()
    proposals = run_frozen_proposals(models.proposal_model, image_path,
                                     device=models.provenance.get("proposal_device", "0"))
    selection = select_reference_mask(proposals, family)
    timing["proposal_reference"] = round(time.perf_counter() - mark, 3)
    if selection.abstained:
        return {
            "status": STATUS_REFERENCE_ABSTENTION,
            "exit_code": EXIT_REFERENCE_ABSTENTION,
            "prompt": prompt,
            "parsed_program": parsed["program"],
            "parser": parsed,
            "reference_family": family,
            "relation": relation,
            "domain_gate": gate,
            "proposal_count": len(proposals),
            "eligible_reference_proposals": selection.proposals_eligible,
            "reference_abstained": True,
            "reference_abstention_reason": selection.reason,
            "target_mask": None,
            "ground_truth_used": False,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
    reference_mask = selection.mask

    # ---------------- 5. GeometricRelationField v0.2
    mark = time.perf_counter()
    reference_tensor = torch.as_tensor(reference_mask.astype(np.float32))
    field = geometric_relation_field_v02(reference_tensor, relation, FIELD_SIZE)
    timing["field"] = round(time.perf_counter() - mark, 3)

    # ---------------- 6. frozen SAM2 visual feature (computed on the fly when not cached)
    mark = time.perf_counter()
    visual = models.feature_store.get(image_path.stem, image_path).float().unsqueeze(0)
    timing["sam2_feature"] = round(time.perf_counter() - mark, 3)
    visual = visual.to(models.device)

    # ---------------- 7. frozen Task 6O B3 target decoder
    mark = time.perf_counter()
    index = torch.as_tensor([RELATION_ORDER.index(relation)], dtype=torch.long,
                            device=models.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=str(models.device) != "cpu"):
        logits = models.target_model(visual, index, None, field.to(models.device))
    upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
    target_mask = (upsampled[0, 0] > 0.0).cpu().numpy()
    timing["target_decoder"] = round(time.perf_counter() - mark, 3)
    timing["total"] = round(time.perf_counter() - started, 3)

    result = {
        "status": STATUS_OK,
        "exit_code": EXIT_ANSWERED,
        "prompt": prompt,
        "parsed_program": parsed["program"],
        "parser": parsed,
        "reference_family": family,
        "relation": relation,
        "domain_gate": gate,
        "proposal_count": len(proposals),
        "eligible_reference_proposals": selection.proposals_eligible,
        "selected_reference_proposal": selection.proposal.as_dict() if selection.proposal else None,
        "reference_abstained": False,
        "reference_abstention_reason": None,
        "relation_field": {
            "min": float(field.min()), "max": float(field.max()), "mean": float(field.mean()),
        },
        "target_mask": target_mask,
        "target_positive_pixels": int(target_mask.sum()),
        "runtime": timing,
        "ground_truth_used": False,
        "runtime_seconds": timing["total"],
    }
    if want_visuals:
        result["reference_mask"] = reference_mask
        result["field_array"] = field[0, 0].detach().cpu().numpy()
        result["image_path"] = str(image_path)
    return result


__all__ = [
    "ChainModels",
    "DEFAULT_PROGRAM_HEAD_CANDIDATES",
    "EXIT_ANSWERED",
    "EXIT_REFERENCE_ABSTENTION",
    "EXIT_UNSUPPORTED_DIRECTIONAL",
    "EXIT_UNSUPPORTED_INSTRUCTION",
    "FIELD_SIZE",
    "HARDENED_PROGRAM_HEAD",
    "PROGRAM_DECOMPOSITION",
    "PROGRAM_HEAD_CHECKPOINT_SHA256",
    "STATUS_OK",
    "STATUS_REFERENCE_ABSTENTION",
    "STATUS_UNSUPPORTED_DIRECTIONAL",
    "STATUS_UNSUPPORTED_INSTRUCTION",
    "SUPPORTED_PROGRAMS",
    "TARGET_SIZE",
    "decompose_program",
    "default_program_head_checkpoint",
    "parse_instruction",
    "resolve_program_head_checkpoint",
    "run_directional_chain",
    "select_reference_mask",
    "sha256_file",
]
