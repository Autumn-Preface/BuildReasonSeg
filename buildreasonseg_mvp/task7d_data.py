"""Task 7D — shared data/batch helpers for the global-competition variants.

Reuses the exact frozen Task 6Z packs (Z-Overfit20, Z-MiniTrain1200, Z-MiniVal240, Z-PairedVal20), the frozen
oracle reference masks, the frozen `P_dir_64` / `P_near_64` fields and the frozen SAM2 features.
"""

from __future__ import annotations

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

from buildreasonseg_mvp.task7d_global_competition_decoder import TRAINABLE_VARIANTS  # noqa: E402

PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
PACK_MANIFEST = REPO_ROOT / "evaluation" / "task6z_pack_manifest.json"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7d"
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")


@dataclass
class CompetitionBatch:
    visual: torch.Tensor
    directional: torch.Tensor
    nearest: torch.Tensor
    relations: list[str]
    target: torch.Tensor
    sample_ids: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    tile_ids: list[str] = field(default_factory=list)
    reference_masks: list[np.ndarray] = field(default_factory=list)

    @property
    def target_size(self) -> tuple[int, int]:
        """Canonical 512x512 target resolution (used by the frozen Task 6N metric helpers)."""

        return (int(self.target.shape[-2]), int(self.target.shape[-1]))

    @property
    def programs_for_metrics(self) -> list[str]:
        return self.programs


def load_pack(name: str) -> list[dict]:
    return json.loads((PACK_ROOT / f"{name}.json").read_text(encoding="utf-8"))["records"]


def image_path_of(record: dict) -> Path:
    path = Path(record["image_path"])
    return path if path.is_absolute() else REPO_ROOT / path


def build_batch(records: list[dict], store, masks, device: str,
                precomputed: dict[str, dict]) -> CompetitionBatch:
    from buildreasonseg_mvp.task6z_field_composition import PROGRAM_TO_RELATION, record_fields

    visuals, directionals, nearests, targets, reference_masks = [], [], [], [], []
    for record in records:
        sample_id = record["sample_id"]
        entry = precomputed.get(sample_id)
        reference_mask = np.asarray(masks.mask(record["tile_id"],
                                              record["reference_source_feature_id"]), dtype=bool)
        if entry is None:
            fields = record_fields(reference_mask, record["program_id"])
            entry = {"P_dir_64": fields["P_dir_64"], "P_near_64": fields["P_near_64"]}
            precomputed[sample_id] = entry
        directionals.append(entry["P_dir_64"][None])
        nearests.append(entry["P_near_64"][None])
        target = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                            dtype=np.float32)
        targets.append(torch.as_tensor(target))
        reference_masks.append(reference_mask)
        visuals.append(store.get(record["tile_id"], image_path_of(record)).float())
    return CompetitionBatch(
        visual=torch.stack(visuals).to(device),
        directional=torch.stack(directionals).to(device),
        nearest=torch.stack(nearests).to(device),
        relations=[PROGRAM_TO_RELATION[record["program_id"]] for record in records],
        target=torch.stack(targets).to(device),
        sample_ids=[record["sample_id"] for record in records],
        programs=[record["program_id"] for record in records],
        tile_ids=[record["tile_id"] for record in records],
        reference_masks=reference_masks,
        )


def forward_variant(model, batch: CompetitionBatch):
    return model(batch.visual, batch.relations, batch.directional, batch.nearest)


__all__ = ["CHECKPOINT_ROOT", "CompetitionBatch", "L3_PROGRAMS", "PACK_MANIFEST", "PACK_ROOT",
           "TRAINABLE_VARIANTS", "build_batch", "forward_variant", "image_path_of", "load_pack"]
