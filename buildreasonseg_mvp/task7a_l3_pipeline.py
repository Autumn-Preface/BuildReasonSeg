"""Task 7A — L3 predicted-reference + natural-language integration pipeline (inference only).

No training anywhere in this module. Every model/config is loaded frozen:

* U-C1 YOLO26m-seg proposals (`imgsz 640 / conf 0.05 / max_det 300 / default NMS / no TTA / no tiling`);
* the exact Task 6Q deterministic `largest` reference resolver (border/extent eligibility, maximum
  predicted mask area, higher YOLO confidence, then lower original proposal index) — never a ranker,
  quality estimator or SAM2 refinement;
* GeometricRelationField v0.2 (`P_dir`) and NearestBoundaryField v0.1 (`P_near`), untouched;
* the frozen SAM2.1 Hiera Base+ image feature and the frozen Task 6Z Z-B3 L3 decoder.

Exactly four supported L3 programs (section 6), decomposed without any learned step:

```text
largest_to_left_of_to_nearest  -> family=largest, direction=left_of,  terminal=nearest
largest_to_right_of_to_nearest -> family=largest, direction=right_of, terminal=nearest
largest_to_above_to_nearest    -> family=largest, direction=above,    terminal=nearest
largest_to_below_to_nearest    -> family=largest, direction=below,    terminal=nearest
```

The pipeline object is also the single inference entry point used by `predict_buildreasonseg_l3.py`.
Ground truth is never an input; it is only used offline for scoring.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
    read_rgb_tile,
)
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference  # noqa: E402
from buildreasonseg_mvp.task6z_field_composition import (  # noqa: E402
    PROGRAM_TO_RELATION,
    record_fields,
)
from buildreasonseg_mvp.task6z_l3_decoder import L3TargetDecoder  # noqa: E402

#: Exactly the four supported L3 programs (section 6); everything else is out of scope.
SUPPORTED_L3_PROGRAMS: tuple[str, ...] = (
    "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
    "largest_to_above_to_nearest",
    "largest_to_below_to_nearest",
)
PROGRAM_DECOMPOSITION: dict[str, dict[str, str]] = {
    program: {"family": "largest", "direction": PROGRAM_TO_RELATION[program], "terminal": "nearest"}
    for program in SUPPORTED_L3_PROGRAMS
}
REFERENCE_FAMILY = "largest"
U_C1_CONFIG = {"id": "U-C1", "imgsz": 640, "conf": 0.05, "max_det": 300, "nms": "default",
               "tta": False, "tiling": False}
GT_FREE_PIPELINE = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ProposalBundle:
    proposals: list
    eligible: list
    selection: dict | None
    abstention_reason: str | None


@dataclass
class L3StageTiming:
    proposals_seconds: float = 0.0
    reference_seconds: float = 0.0
    field_seconds: float = 0.0
    feature_seconds: float = 0.0
    decoder_seconds: float = 0.0
    total_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class L3Pipeline:
    """Frozen Task 7A inference pipeline for the four supported L3 programs."""

    def __init__(self, *, device: str = "cuda", yolo_device: str = "0",
                 proposal_checkpoint: Path | None = None,
                 target_checkpoint: Path | None = None,
                 expected_target_sha256: str | None = None,
                 feature_root: Path | None = None):
        from scripts.task6u_common import CONFIGS, PROPOSAL_CHECKPOINT, proposals_for_tile
        from task6n_train import FEATURE_ROOT

        self._proposals_for_tile = proposals_for_tile
        self.config = CONFIGS["U-C1"]
        self.proposal_checkpoint = Path(proposal_checkpoint or PROPOSAL_CHECKPOINT)
        self.device = device
        self.yolo_device = yolo_device
        self.feature_root = Path(feature_root or FEATURE_ROOT)

        target_path = Path(target_checkpoint) if target_checkpoint else default_target_checkpoint()
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        self.target_checkpoint = target_path
        self.target_sha256 = sha256_file(target_path)
        if expected_target_sha256 and self.target_sha256 != expected_target_sha256:
            raise ValueError(f"Z-B3 checkpoint hash mismatch: {self.target_sha256} != "
                             f"{expected_target_sha256}")

        payload = torch.load(target_path, map_location=device, weights_only=False)
        if payload.get("variant") != "Z-B3":
            raise ValueError(f"checkpoint variant is {payload.get('variant')!r}, expected 'Z-B3'")
        self.target_variant = str(payload["variant"])
        self.target_model = L3TargetDecoder(self.target_variant).to(device)
        self.target_model.load_state_dict(payload["state_dict"])
        self.target_model.eval()

        self._yolo = None
        self.encoder, self.encoder_report = load_frozen_sam2_encoder(device=device)
        self.store = FrozenFeatureStore(self.feature_root, encoder=self.encoder, device=device)
        self._proposal_cache: dict[str, list] = {}

    # ------------------------------------------------------------------ proposals / reference

    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO

            self._yolo = YOLO(str(self.proposal_checkpoint))
        return self._yolo

    def proposals(self, tile_id: str, image_path: Path) -> list:
        if tile_id not in self._proposal_cache:
            proposals, _run = self._proposals_for_tile(self.yolo(), self.config, tile_id,
                                                       Path(image_path), self.yolo_device,
                                                       use_cache=True)
            self._proposal_cache[tile_id] = proposals
        return self._proposal_cache[tile_id]

    def resolve_reference(self, proposals: list, family: str = REFERENCE_FAMILY) -> ProposalBundle:
        """Exact Task 6Q deterministic family resolver (no learned module)."""

        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        if not proposals:
            return ProposalBundle(proposals=proposals, eligible=eligible, selection=None,
                                  abstention_reason="no_proposals")
        if not eligible:
            return ProposalBundle(proposals=proposals, eligible=eligible, selection=None,
                                  abstention_reason="no_eligible_proposals")
        selection = select_reference(proposals, family)
        if selection.abstained:
            return ProposalBundle(proposals=proposals, eligible=eligible, selection=None,
                                  abstention_reason=selection.reason)
        return ProposalBundle(proposals=proposals, eligible=eligible,
                              selection={"mask": selection.proposal.mask,
                                         "index": int(selection.proposal.index),
                                         "confidence": float(selection.proposal.confidence),
                                         "area_px": int(selection.proposal.area_px),
                                         "bbox_xyxy": [float(value)
                                                       for value in selection.proposal.bbox_xyxy],
                                         "reason": selection.reason},
                              abstention_reason=None)

    # ------------------------------------------------------------------ target prediction

    @torch.no_grad()
    def predict_target(self, image_path: Path, tile_id: str, program_id: str,
                       reference_mask: np.ndarray) -> dict:
        import time

        timing = L3StageTiming()
        started = time.perf_counter()
        fields = record_fields(np.asarray(reference_mask, dtype=bool), program_id)
        timing.field_seconds = time.perf_counter() - started
        started = time.perf_counter()
        visual = self.store.get(tile_id, image_path).float().to(self.device)
        timing.feature_seconds = time.perf_counter() - started
        started = time.perf_counter()
        relation = PROGRAM_TO_RELATION[program_id]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device != "cpu"):
            logits = self.target_model(visual[None], [relation],
                                       fields["P_dir_64"][None, None].to(self.device),
                                       fields["P_near_64"][None, None].to(self.device), None)
        upsampled = torch.nn.functional.interpolate(logits.float(), size=(512, 512),
                                                    mode="bilinear", align_corners=False)
        timing.decoder_seconds = time.perf_counter() - started
        mask = (upsampled > 0.0)[0, 0].cpu().numpy()
        return {"mask": mask, "fields": fields, "timing": timing,
                "logits_positive_ratio": float((upsampled > 0.0).float().mean())}

    # ------------------------------------------------------------------ full inference

    def infer(self, *, image_path: Path, tile_id: str, program_id: str) -> dict:
        """One full L3 inference from a canonical program id (no GT anywhere)."""

        import time

        overall = time.perf_counter()
        if program_id not in SUPPORTED_L3_PROGRAMS:
            return {"status": "unsupported_l3_program", "parsed_program": program_id,
                    "supported": False, "abstained": False, "mask": None,
                    "ground_truth_used": False}
        started = time.perf_counter()
        proposals = self.proposals(tile_id, image_path)
        proposals_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bundle = self.resolve_reference(proposals)
        reference_seconds = time.perf_counter() - started
        result = {
            "status": "ok", "parsed_program": program_id, "supported": True,
            "decomposition": PROGRAM_DECOMPOSITION[program_id],
            "proposal_count": len(proposals), "eligible_count": len(bundle.eligible),
            "abstained": bundle.selection is None,
            "abstention_reason": bundle.abstention_reason,
            "selection": bundle.selection, "mask": None,
            "ground_truth_used": False,
        }
        if bundle.selection is None:
            result["status"] = "abstained"
            result["reference_mask"] = None
            result["fields"] = None
            result["timing"] = {"proposals_seconds": proposals_seconds,
                                "reference_seconds": reference_seconds,
                                "total_seconds": time.perf_counter() - overall}
            return result
        reference_mask = np.asarray(bundle.selection["mask"], dtype=bool)
        prediction = self.predict_target(image_path, tile_id, program_id, reference_mask)
        prediction["timing"].proposals_seconds = proposals_seconds
        prediction["timing"].reference_seconds = reference_seconds
        prediction["timing"].total_seconds = time.perf_counter() - overall
        result.update({"mask": prediction["mask"], "reference_mask": reference_mask,
                       "fields": prediction["fields"], "timing": prediction["timing"],
                       "logits_positive_ratio": prediction["logits_positive_ratio"]})
        return result


def default_target_checkpoint() -> Path:
    return REPO_ROOT / "artifacts" / "checkpoints" / "task6z" / "zb3_minitrain1200.pt"


def default_parser_checkpoint() -> Path:
    """Frozen Task 6T hardened ProgramHead (Task 7A integration checkpoint)."""

    return REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"


def task7b_parser_checkpoint() -> Path:
    """Task 7B L3 compositional-semantic hardened ProgramHead (if it exists)."""

    return REPO_ROOT / "artifacts" / "checkpoints" / "task7b" / "program_parser_l3_hardened_v1.pt"


def task7c_parser_checkpoint() -> Path:
    """Task 7C 20-class rehearsal ProgramHead (if it exists)."""

    return REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"


def default_l3_parser_checkpoint() -> Path:
    """The L3 CLI's default parser: the latest hardening checkpoint whose canonical gates passed.

    Task 7A's `default_parser_checkpoint()` is deliberately left untouched so the frozen Task 7A/7B
    integration artifacts keep referring to the Task 6T checkpoint they were measured with. Task 7B
    section 20 and Task 7C section 24 both allow the CLI default to move to a hardened checkpoint **only
    if that task's canonical gates pass**, so this helper consults the frozen verdicts newest-first.
    """

    for verdict_name, candidate in (("task7c_verdict.json", task7c_parser_checkpoint()),
                                    ("task7b_verdict.json", task7b_parser_checkpoint())):
        if not candidate.is_file():
            continue
        verdict_path = REPO_ROOT / "evaluation" / verdict_name
        if not verdict_path.is_file():
            continue
        try:
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - defensive
            continue
        if verdict.get("canonical_gates_passed"):
            return candidate
    return default_parser_checkpoint()


def pipeline_report() -> dict:
    from buildreasonseg_mvp.task6z_field_composition import field_report

    return {
        "task": "7A",
        "supported_programs": list(SUPPORTED_L3_PROGRAMS),
        "program_decomposition": {program: dict(value)
                                  for program, value in PROGRAM_DECOMPOSITION.items()},
        "reference_family": REFERENCE_FAMILY,
        "u_c1_config": dict(U_C1_CONFIG),
        "fields": field_report(),
        "training_performed": False,
        "ground_truth_used_in_inference": False,
        "ranker_or_quality_or_refinement_used": False,
        "attention_or_transformer_or_gnn": False,
        "grcl": False,
    }


__all__ = [
    "GT_FREE_PIPELINE",
    "L3Pipeline",
    "L3StageTiming",
    "PROGRAM_DECOMPOSITION",
    "ProposalBundle",
    "REFERENCE_FAMILY",
    "SUPPORTED_L3_PROGRAMS",
    "U_C1_CONFIG",
    "default_parser_checkpoint",
    "default_target_checkpoint",
    "pipeline_report",
    "read_rgb_tile",
    "sha256_file",
]
