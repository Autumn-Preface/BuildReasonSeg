"""Diagnostics contract: what a future run must be able to emit and where it goes.

Task 8A freezes the artifact names and the output layout so the CLI, the model package and the reports agree.
`--no-save-diagnostics` disables every one of them; nothing is emitted when diagnostics are off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from buildreasonseg import paths

DIAGNOSTIC_ARTIFACTS = (
    "proposals.json",
    "reference_selection.json",
    "relation_fields.npz",
    "command_trace.json",
    "timing.json",
)
LARGE_IMAGE_PARAMETERS_FROZEN = False  # tile size / overlap / merge thresholds are defined in a later task


@dataclass
class DiagnosticsRequest:
    enabled: bool = True
    output_dir: str | None = None
    artifacts: list[str] = field(default_factory=lambda: list(DIAGNOSTIC_ARTIFACTS))

    def directory(self) -> str:
        return self.output_dir or str(paths.inference_dir() / "output" / "diagnostics")

    def describe(self) -> dict:
        return {"enabled": self.enabled, "directory": self.directory(),
                "artifacts": list(self.artifacts) if self.enabled else [],
                "large_image_parameters_frozen": LARGE_IMAGE_PARAMETERS_FROZEN}


__all__ = ["DIAGNOSTIC_ARTIFACTS", "LARGE_IMAGE_PARAMETERS_FROZEN", "DiagnosticsRequest"]
