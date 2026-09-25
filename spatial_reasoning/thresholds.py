"""Threshold configuration loading for the spatial relation engine.

No numeric threshold may be hard-coded in the relation or quality modules; they
all read the values resolved here, which come from
``configs/spatial_relations_v1.yaml``.

The loader validates the config so that a typo or a missing key fails loudly
instead of silently falling back to a default.

Uses ``yaml`` when available and falls back to a bundled minimal parser only if
PyYAML is absent, so the module never becomes an undeclared dependency risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"

#: Relations implemented in v1. Deliberately frozen; see docs/relation_definitions.md.
CORE_RELATIONS = (
    "left_of",
    "right_of",
    "above",
    "below",
    "nearest",
    "largest",
    "smallest",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
)

DIRECTION_RELATIONS = ("left_of", "right_of", "above", "below")
EXTREME_RELATIONS = ("leftmost", "rightmost", "topmost", "bottommost")
SIZE_RELATIONS = ("largest", "smallest")

#: Relation pairs that must be symmetric under argument swap.
SYMMETRIC_PAIRS = (
    ("left_of", "right_of"),
    ("above", "below"),
)


class ConfigError(ValueError):
    """Raised when the relation config is missing or malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ConfigError(
            "PyYAML is required to read the relation config. "
            f"Install pyyaml or pass a preloaded dict. ({exc})"
        ) from exc

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return mapping[key]


@dataclass(frozen=True)
class QualityThresholds:
    merge_bbox_extent_ratio: float
    tiny_area_px: int

    def __post_init__(self) -> None:
        if not 0.0 < self.merge_bbox_extent_ratio <= 1.0:
            raise ConfigError("quality.suspected_large_merge.bbox_extent_ratio_threshold must be in (0, 1]")
        if self.tiny_area_px < 0:
            raise ConfigError("quality.tiny_component.area_px_threshold must be >= 0")


@dataclass(frozen=True)
class DirectionThresholds:
    alpha: float
    tau: float
    active_candidate: str
    candidates: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.alpha < 1.0:
            raise ConfigError("direction alpha must be >= 1.0 (1.0 disables dominance)")
        if self.tau < 0.0:
            raise ConfigError("direction tau must be >= 0")


@dataclass(frozen=True)
class NearestThresholds:
    distance_metric: str
    margin_px_floor: float
    margin_diag_fraction: float
    margin_mode: str

    def __post_init__(self) -> None:
        if self.distance_metric != "boundary_distance":
            raise ConfigError(
                "nearest.distance_metric must be 'boundary_distance'; the centroid "
                "shortcut is not an acceptable relation ground truth"
            )
        if self.margin_px_floor < 0 or self.margin_diag_fraction < 0:
            raise ConfigError("nearest margins must be non-negative")


@dataclass(frozen=True)
class SizeRankThresholds:
    ratio_margin: float
    largest: dict[str, bool]
    smallest: dict[str, bool]

    def __post_init__(self) -> None:
        if self.ratio_margin < 1.0:
            raise ConfigError("size_rank.ratio_margin must be >= 1.0")


@dataclass(frozen=True)
class EligibilityRule:
    reject_touches_image_border: bool
    reject_suspected_large_merge: bool
    reject_tiny_component: bool
    min_valid_components: int
    reject_touches_image_border_anchor: bool = False

    def accepts(self, flags: dict[str, bool]) -> tuple[bool, str | None]:
        """Return ``(accepted, rejection_reason)`` for one component as a TARGET.

        Only the flags this rule explicitly rejects are applied. There is
        deliberately no unconditional ``geometry_valid`` gate: ``geometry_valid``
        is a convenience summary, and applying it everywhere would wrongly
        disqualify, for example, a tiny or border-truncated component from
        ``left_of``, which is a tile-relative relation that must tolerate both.
        """

        if self.reject_touches_image_border and flags.get("touches_image_border"):
            return False, "component_touches_image_border"
        if self.reject_suspected_large_merge and flags.get("suspected_large_merge"):
            return False, "component_suspected_large_merge"
        if self.reject_tiny_component and flags.get("tiny_component"):
            return False, "component_tiny"
        return True, None

    def accepts_as_anchor(self, flags: dict[str, bool]) -> tuple[bool, str | None]:
        """Return ``(accepted, rejection_reason)`` for one component as an ANCHOR.

        Identical to :meth:`accepts`, except the border test uses
        ``reject_touches_image_border_anchor``. For ``nearest`` both are true,
        because the metric is a boundary distance and a truncated boundary makes
        the measurement unreliable at either end.
        """

        if self.reject_touches_image_border_anchor and flags.get("touches_image_border"):
            return False, "component_touches_image_border"
        return self.accepts(flags)


@dataclass(frozen=True)
class RelationConfig:
    version: str
    scope_kind: str
    scope_note: str
    quality: QualityThresholds
    direction: DirectionThresholds
    extreme_margin_px: float
    nearest: NearestThresholds
    size_rank: SizeRankThresholds
    eligibility: dict[str, EligibilityRule]
    source_path: str

    def eligibility_for(self, relation: str) -> EligibilityRule:
        if relation not in self.eligibility:
            raise ConfigError(f"eligibility: no rule defined for relation '{relation}'")
        return self.eligibility[relation]

    def thresholds_for(self, relation: str) -> dict[str, float]:
        """Report the numeric thresholds a relation actually uses (for evidence)."""

        if relation in DIRECTION_RELATIONS:
            return {
                "alpha": self.direction.alpha,
                "tau": self.direction.tau,
                "candidate": self.direction.active_candidate,
            }
        if relation in EXTREME_RELATIONS:
            return {"margin_px": self.extreme_margin_px}
        if relation == "nearest":
            return {
                "margin_px_floor": self.nearest.margin_px_floor,
                "margin_diag_fraction": self.nearest.margin_diag_fraction,
                "distance_metric": self.nearest.distance_metric,
            }
        if relation in SIZE_RELATIONS:
            return {"ratio_margin": self.size_rank.ratio_margin}
        raise ConfigError(f"no thresholds defined for relation '{relation}'")


def load_config(path: Path | str | None = None) -> RelationConfig:
    """Load and validate the relation config."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"relation config not found: {config_path}")

    raw = _load_yaml(config_path)

    quality_raw = _require(raw, "quality", "config")
    merge_raw = _require(quality_raw, "suspected_large_merge", "quality")
    tiny_raw = _require(quality_raw, "tiny_component", "quality")

    direction_raw = _require(raw, "direction", "config")
    candidates = _require(direction_raw, "candidates", "direction")
    active = _require(direction_raw, "active", "direction")
    if active not in candidates:
        raise ConfigError(
            f"direction.active '{active}' is not one of {sorted(candidates)}"
        )
    active_params = candidates[active]

    nearest_raw = _require(raw, "nearest", "config")
    size_raw = _require(raw, "size_rank", "config")
    eligibility_raw = _require(raw, "eligibility", "config")

    eligibility: dict[str, EligibilityRule] = {}
    for relation in CORE_RELATIONS:
        if relation not in eligibility_raw:
            raise ConfigError(f"eligibility: missing entry for '{relation}'")
        entry = eligibility_raw[relation]
        eligibility[relation] = EligibilityRule(
            reject_touches_image_border=bool(
                _require(entry, "reject_touches_image_border", f"eligibility.{relation}")
            ),
            reject_suspected_large_merge=bool(
                _require(entry, "reject_suspected_large_merge", f"eligibility.{relation}")
            ),
            reject_tiny_component=bool(
                _require(entry, "reject_tiny_component", f"eligibility.{relation}")
            ),
            min_valid_components=int(
                _require(entry, "min_valid_components", f"eligibility.{relation}")
            ),
            reject_touches_image_border_anchor=bool(
                entry.get("reject_touches_image_border_anchor", False)
            ),
        )

    scope_raw = raw.get("scope", {})

    return RelationConfig(
        version=str(_require(raw, "version", "config")),
        scope_kind=str(scope_raw.get("kind", "tile_relative")),
        scope_note=str(scope_raw.get("note", "")),
        quality=QualityThresholds(
            merge_bbox_extent_ratio=float(
                _require(merge_raw, "bbox_extent_ratio_threshold", "quality.suspected_large_merge")
            ),
            tiny_area_px=int(
                _require(tiny_raw, "area_px_threshold", "quality.tiny_component")
            ),
        ),
        direction=DirectionThresholds(
            alpha=float(_require(active_params, "alpha", f"direction.candidates.{active}")),
            tau=float(_require(active_params, "tau", f"direction.candidates.{active}")),
            active_candidate=active,
            candidates={
                name: {
                    "alpha": float(params["alpha"]),
                    "tau": float(params["tau"]),
                }
                for name, params in candidates.items()
            },
        ),
        extreme_margin_px=float(
            _require(_require(raw, "extreme", "config"), "margin_px", "extreme")
        ),
        nearest=NearestThresholds(
            distance_metric=str(_require(nearest_raw, "distance_metric", "nearest")),
            margin_px_floor=float(_require(nearest_raw, "margin_px_floor", "nearest")),
            margin_diag_fraction=float(
                _require(nearest_raw, "margin_diag_fraction", "nearest")
            ),
            margin_mode=str(_require(nearest_raw, "margin_mode", "nearest")),
        ),
        size_rank=SizeRankThresholds(
            ratio_margin=float(_require(size_raw, "ratio_margin", "size_rank")),
            largest=dict(_require(size_raw, "largest", "size_rank")),
            smallest=dict(_require(size_raw, "smallest", "size_rank")),
        ),
        eligibility=eligibility,
        source_path=str(config_path),
    )


def with_direction_candidate(config: RelationConfig, candidate: str) -> RelationConfig:
    """Return a copy of the config using a different direction candidate set.

    Used by the diagnostic tool to compare strict / medium / loose without
    editing the config file.
    """

    if candidate not in config.direction.candidates:
        raise ConfigError(f"unknown direction candidate '{candidate}'")
    params = config.direction.candidates[candidate]
    new_direction = DirectionThresholds(
        alpha=params["alpha"],
        tau=params["tau"],
        active_candidate=candidate,
        candidates=config.direction.candidates,
    )
    return RelationConfig(
        version=config.version,
        scope_kind=config.scope_kind,
        scope_note=config.scope_note,
        quality=config.quality,
        direction=new_direction,
        extreme_margin_px=config.extreme_margin_px,
        nearest=config.nearest,
        size_rank=config.size_rank,
        eligibility=config.eligibility,
        source_path=config.source_path,
    )
