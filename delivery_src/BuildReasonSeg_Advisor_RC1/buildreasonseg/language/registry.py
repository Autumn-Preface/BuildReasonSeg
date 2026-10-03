"""Parsed-program schema and the supported-program registry (Task 8A section 7).

The schema is the only object that may travel from the language front end to the hard validator. No heuristic
regular-expression result may ever be presented as a Qwen parse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: The four programs RC1 is formally allowed to execute.
SUPPORTED_PROGRAMS: tuple[str, ...] = (
    "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
    "largest_to_above_to_nearest",
    "largest_to_below_to_nearest",
)

#: Programs that never enter the execution path in RC1 (recorded so their status is explicit).
KNOWN_UNSUPPORTED_PROGRAMS: tuple[str, ...] = (
    "smallest_to_left_of_to_nearest",
    "smallest_to_right_of_to_nearest",
    "smallest_to_above_to_nearest",
    "smallest_to_below_to_nearest",
)

PROGRAM_FAMILIES = {"largest": ("left_of", "right_of", "above", "below"),
                    "smallest": ("left_of", "right_of", "above", "below")}

PARSE_SOURCES = ("qwen_program_head", "deterministic_fallback", "user_supplied")

REFERENCE_SELECTION = ("deterministic_largest", "user_supplied_reference_id")


@dataclass
class ParsedProgram:
    """The frozen language-front-end -> validator contract."""

    program: str
    source: str
    confidence: float | None = None
    raw_text: str = ""
    normalized_text: str = ""
    reference_family: str | None = None
    direction: str | None = None
    terminal: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in PARSE_SOURCES:
            raise ValueError(f"unknown parse source: {self.source!r}")
        if self.reference_family is None and self.program.startswith("largest"):
            object.__setattr__(self, "reference_family", "largest")
        if self.reference_family is None and self.program.startswith("smallest"):
            object.__setattr__(self, "reference_family", "smallest")
        if self.terminal is None and self.program.endswith("_to_nearest"):
            object.__setattr__(self, "terminal", "nearest")
        if self.direction is None:
            for direction in ("left_of", "right_of", "above", "below"):
                if f"_to_{direction}" in self.program:
                    object.__setattr__(self, "direction", direction)
                    break

    @property
    def is_supported(self) -> bool:
        return self.program in SUPPORTED_PROGRAMS

    @property
    def is_qwen_parsed(self) -> bool:
        return self.source == "qwen_program_head"

    def to_dict(self) -> dict:
        return asdict(self)


def registry() -> dict:
    return {
        "supported_programs": list(SUPPORTED_PROGRAMS),
        "known_unsupported_programs": list(KNOWN_UNSUPPORTED_PROGRAMS),
        "reference_families": {key: list(value) for key, value in PROGRAM_FAMILIES.items()},
        "parse_sources": list(PARSE_SOURCES),
        "reference_selection": list(REFERENCE_SELECTION),
        "executable_program_count": len(SUPPORTED_PROGRAMS),
    }


__all__ = ["KNOWN_UNSUPPORTED_PROGRAMS", "PARSE_SOURCES", "PROGRAM_FAMILIES",
           "REFERENCE_SELECTION", "SUPPORTED_PROGRAMS", "ParsedProgram", "registry"]
