"""Task 6T — build the frozen parser-hardening packs (before training) and the augmented training set.

Stages (run in this order):

* `--stage packs`   — writes `evaluation/task6t_parser_minimal_pairs.json` (>= 48 contrast prompts) and
  `evaluation/task6t_parser_stress_v1.json` (>= 120 prompts, every one of the 20 canonical classes,
  >= 3 Chinese + 3 English per class). Both are **frozen before training** and use held-out template and
  lexical forms that the training grammar never generates.
* `--stage augment` — writes `evaluation/task6t_parser_train_augmentation_spec.json` and the local
  augmented dataset under `artifacts/task6t/parser_train_augmented/` (gitignored).
* `--stage leakage` — writes `evaluation/task6t_parser_leakage_audit.json`: exact UTF-8 and normalized
  prompt comparison of every training prompt against MiniVal240 queries, PairedVal20 queries, the exact
  Task 6S fixed-24 paraphrase list, the Task 6T minimal pairs and the Task 6T stress prompts. Requires
  zero overlap.

The training grammar deliberately excludes the exact Task 6S fixed-24 strings, and any residual
collision is removed by a deterministic pre-training filter so that the frozen audit is exactly 0.

    python scripts/task6t_build_parser_data.py --stage packs
    python scripts/task6t_build_parser_data.py --stage augment
    python scripts/task6t_build_parser_data.py --stage leakage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
AUGMENT_ROOT = REPO_ROOT / "artifacts" / "task6t" / "parser_train_augmented"
OUT_SPEC = EVAL / "task6t_parser_train_augmentation_spec.json"
OUT_MINIMAL = EVAL / "task6t_parser_minimal_pairs.json"
OUT_STRESS = EVAL / "task6t_parser_stress_v1.json"
OUT_LEAKAGE = EVAL / "task6t_parser_leakage_audit.json"
SEED = 20260930
AUGMENTED_PER_PROGRAM_PER_LANGUAGE = 45

PROGRAMS: tuple[str, ...] = tuple(EXPECTED_QUERY_TYPES)
DIRECTIONS: dict[str, tuple[str, str]] = {
    "left_of": ("左侧", "to the left of"),
    "right_of": ("右侧", "to the right of"),
    "above": ("上方", "above"),
    "below": ("下方", "below"),
}

# --------------------------------------------------------------------------- Task 6S frozen prompts

FIXED24: tuple[str, ...] = (
    "分割面积最大的建筑物左侧的建筑物。", "找出最大建筑左边的建筑物。",
    "segment the building to the left of the largest building",
    "分割面积最大的建筑物右侧的建筑物。", "找出最大建筑右边的建筑物。",
    "segment the building to the right of the largest building",
    "分割面积最大的建筑物上方的建筑物。", "找出最大建筑上面的建筑物。",
    "segment the building above the largest building",
    "分割面积最大的建筑物下方的建筑物。", "找出最大建筑下面的建筑物。",
    "segment the building below the largest building",
    "分割面积最小的建筑物左侧的建筑物。", "找出最小建筑左边的建筑物。",
    "segment the building to the left of the smallest building",
    "分割面积最小的建筑物右侧的建筑物。", "找出最小建筑右边的建筑物。",
    "segment the building to the right of the smallest building",
    "分割面积最小的建筑物上方的建筑物。", "找出最小建筑上面的建筑物。",
    "segment the building above the smallest building",
    "分割面积最小的建筑物下方的建筑物。", "找出最小建筑下面的建筑物。",
    "segment the building below the smallest building",
)
CONTROLS: tuple[str, ...] = (
    "Write a poem about the sea.", "今天天气怎么样？", "检测道路。", "",
    "分割面积最大的建筑物。", "分割最左侧的建筑物。",
    "分割面积最大的建筑物右侧最近的建筑物。",
    "segment the building nearest to the right of the largest building",
)

# --------------------------------------------------------------------------- normalization / hashing


def normalize_prompt(prompt: str) -> str:
    """Strip whitespace, lowercase English, normalise common punctuation (leakage comparison only)."""

    text = prompt.strip().lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    for source, target in (("，", ","), ("。", "."), ("？", "?"), ("！", "!"), ("、", ","),
                           ("：", ":"), ("；", ";"), ("（", "("), ("）", ")"), ("“", '"'),
                           ("”", '"'), ("'", "'"), ("’", "'"), ("　", "")):
        text = text.replace(source, target)
    text = re.sub(r"[.。,，!！?？;；:：]+$", "", text)
    return text


# --------------------------------------------------------------------------- EVAL-side (held-out) forms

EVAL_ZH_VERBS = ("请圈出", "请标注", "请勾勒出", "标注", "请圈出", "请标注", "请勾勒", "标出")
EVAL_ZH_NOUN = "建筑区域"
EVAL_ZH_SHORT_NOUN = "建筑"
#: `{ref}` is the reference phrase, `{dir}` the Chinese direction word, `{side}` the English side noun.
#: Every template always substitutes the direction so the prompt stays semantically unambiguous.
EVAL_ZH_CONNECTIVE = ("位于{ref}{dir}的", "{ref}{dir}的", "处于{ref}{dir}的", "坐落在{ref}{dir}的",
                      "位于{ref}{dir}的", "{ref}{dir}的", "处于{ref}{dir}的", "坐落在{ref}{dir}的")
EVAL_EN_VERBS = ("Please delineate", "Please demarcate", "Delineate", "Please delineate",
                 "Please demarcate", "Delineate", "Please delineate and mark",
                 "Delineate and mark")
EVAL_EN_NOUN = "building region"
EVAL_EN_CONNECTIVE = ("lying {dir} the {ref}", "on the {side} side of the {ref}",
                      "situated {dir} the {ref}", "that lies {dir} the {ref}",
                      "{dir} the {ref}", "on the {side} side of the {ref}",
                      "lying {dir} the {ref}", "situated {dir} the {ref}")

#: held-out lexical synonyms used in a minority of stress prompts (never generated by the training
#: grammar): zh 占地最大/占地最小 + 楼房, en most extensive / least extensive + structure.
EVAL_ZH_HELDOUT_SUP = ("占地面积最大的", "占地面积最小的")
EVAL_ZH_HELDOUT_NOUN = "楼房区域"
EVAL_EN_HELDOUT_SUP = ("most extensive", "least extensive")
EVAL_EN_HELDOUT_NOUN = "structure"

EVAL_DIRECTION = {
    "left_of": {"zh": ("左侧", "左边"), "en": ("to the left of", "left of"), "side": "left"},
    "right_of": {"zh": ("右侧", "右边"), "en": ("to the right of", "right of"), "side": "right"},
    "above": {"zh": ("上方", "上面"), "en": ("above", "over"), "side": "upper"},
    "below": {"zh": ("下方", "下面"), "en": ("below", "under"), "side": "lower"},
}
EVAL_NEAREST_ZH = ("最近的", "距离最近的", "最靠近的", "边界距离最小的")
EVAL_NEAREST_EN = ("nearest", "closest", "with the shortest boundary distance",
                   "with the smallest boundary distance")

L1_ZH = {"leftmost": ("最左侧的", "最左边的"), "rightmost": ("最右侧的", "最右边的"),
         "topmost": ("最上方的", "最上面的"), "bottommost": ("最下方的", "最下面的")}
L1_EN = {"leftmost": "furthest left", "rightmost": "furthest right", "topmost": "highest",
         "bottommost": "lowest"}


def eval_prompt(program: str, language: str, form: int) -> str:
    """Held-out evaluation prompt for one canonical program (form 0-7, deterministic).

    Forms 0-3 are used by the stress pack, forms 4-7 by the minimal-pair pack, so every
    (program, language, form) combination is unique inside each pack.
    """

    zh = language == "zh"
    verb = EVAL_ZH_VERBS[form] if zh else EVAL_EN_VERBS[form]
    noun = EVAL_ZH_NOUN if zh else EVAL_EN_NOUN
    short = EVAL_ZH_SHORT_NOUN if zh else "building"
    if form == 3 and zh:
        noun = EVAL_ZH_HELDOUT_NOUN
    if form == 3 and not zh:
        noun = EVAL_EN_HELDOUT_NOUN

    def superlative(family: str, held_out: bool = False) -> str:
        if held_out:
            return EVAL_ZH_HELDOUT_SUP[0 if family == "largest" else 1] if zh \
                else EVAL_EN_HELDOUT_SUP[0 if family == "largest" else 1]
        return ("面积最大的" if family == "largest" else "面积最小的") if zh \
            else ("largest" if family == "largest" else "smallest")

    if program in L1_ZH:
        if zh:
            return f"{verb}{L1_ZH[program][form % 2]}{noun}。"
        return f"{verb} the {L1_EN[program]} {noun}."
    if program in ("largest", "smallest"):
        return f"{verb}{superlative(program)}的{noun}。" if zh \
            else f"{verb} the {superlative(program)} {noun}."
    if program in ("largest_to_nearest", "smallest_to_nearest"):
        family = "largest" if program.startswith("largest") else "smallest"
        if zh:
            return (f"以{superlative(family)}的{noun}为参照，{verb}"
                    f"{EVAL_NEAREST_ZH[form % 4]}另一{noun}。")
        return (f"Using the {superlative(family)} {noun} as reference, {verb.lower()} the other "
                f"{noun} {EVAL_NEAREST_EN[form % 4]}.")

    parts = program.split("_to_")
    family = parts[0]
    rest = "_to_".join(parts[1:])
    nearest = rest.endswith("_to_nearest")
    direction = rest[:-len("_to_nearest")] if nearest else rest
    zh_dir = EVAL_DIRECTION[direction]["zh"][form % 2]
    en_dir = EVAL_DIRECTION[direction]["en"][form % 2]
    en_side = EVAL_DIRECTION[direction]["side"]
    ref = f"{superlative(family)}的{noun}" if zh else f"{superlative(family)} {noun}"
    if not nearest:
        if zh:
            return f"{verb}{EVAL_ZH_CONNECTIVE[form].format(ref=ref, dir=zh_dir)}{noun}。"
        return f"{verb} the {noun} {EVAL_EN_CONNECTIVE[form].format(dir=en_dir, side=en_side, ref=ref)}."
    # direction + nearest
    if zh:
        return (f"先确定{ref}，再从它{zh_dir}的{noun}中选出"
                f"{EVAL_NEAREST_ZH[form % 4]}一个并{verb.replace('请', '')}。")
    return (f"First locate the {ref}, then from the {noun} {en_dir} it select and "
            f"{verb.lower().replace('please ', '')} the one {EVAL_NEAREST_EN[form % 4]}.")


# --------------------------------------------------------------------------- minimal pairs / stress


def validate_pack(rows: list[dict], name: str) -> dict:
    """A prompt may map to exactly one expected program, and direction programs must name a direction.

    This catches construction bugs such as a missing direction word (which would make otherwise
    distinct programs share a single ambiguous prompt string).
    """

    by_prompt: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_prompt[row["prompt"]].add(row["expected_program"])
    ambiguous = {prompt: sorted(programs) for prompt, programs in by_prompt.items()
                 if len(programs) > 1}
    direction_words = {"left_of": ("左侧", "左边", "left of", "left side", "left-hand"),
                       "right_of": ("右侧", "右边", "right of", "right side", "right-hand"),
                       "above": ("上方", "上面", "above", "on top", "upper", "over"),
                       "below": ("下方", "下面", "below", "under", "lower", "beneath")}
    missing_direction = []
    for row in rows:
        program = row["expected_program"]
        if "_to_" not in program:
            continue
        rest = "_to_".join(program.split("_to_")[1:])
        if rest.endswith("_to_nearest"):
            rest = rest[:-len("_to_nearest")]
        if rest in direction_words and not any(
                word in row["prompt"] for word in direction_words[rest]):
            missing_direction.append({"prompt": row["prompt"], "program": program})
    if ambiguous or missing_direction:
        raise SystemExit(
            f"error: {name} pack is semantically invalid: {len(ambiguous)} ambiguous prompts, "
            f"{len(missing_direction)} direction prompts without a direction word"
        )
    return {"prompts": len(rows), "unique_prompts": len(by_prompt),
            "ambiguous_prompts": 0, "direction_prompts_without_direction": 0}


def build_minimal_pairs() -> dict:
    """Section 6.2 — >= 48 paired contrasts across groups A-D (forms 4-5, held out from stress)."""

    prompts = []

    def add(program: str, language: str, form: int, group: str, note: str) -> None:
        prompts.append({
            "prompt": eval_prompt(program, language, form),
            "language": language,
            "expected_program": program,
            "contrast_group": group,
            "contrast_note": note,
        })

    # A. largest vs smallest, for each direction
    for direction in DIRECTIONS:
        for language in ("zh", "en"):
            add(f"largest_to_{direction}", language, 4, "A_largest_vs_smallest",
                f"largest_to_{direction} vs smallest_to_{direction}")
            add(f"smallest_to_{direction}", language, 4, "A_largest_vs_smallest",
                f"largest_to_{direction} vs smallest_to_{direction}")
    # B. direction-only vs direction+nearest
    for direction in DIRECTIONS:
        for language in ("zh", "en"):
            add(f"largest_to_{direction}", language, 5, "B_direction_vs_direction_nearest",
                f"largest_to_{direction} vs largest_to_{direction}_to_nearest")
            add(f"largest_to_{direction}_to_nearest", language, 5,
                "B_direction_vs_direction_nearest",
                f"largest_to_{direction} vs largest_to_{direction}_to_nearest")
    # C. L1 vs L2
    for direction in DIRECTIONS:
        for language in ("zh", "en"):
            add("largest", language, 6, "C_L1_vs_L2", f"largest vs largest_to_{direction}")
            add(f"largest_to_{direction}", language, 6, "C_L1_vs_L2",
                f"largest vs largest_to_{direction}")
            add("smallest", language, 6, "C_L1_vs_L2", f"smallest vs smallest_to_{direction}")
            add(f"smallest_to_{direction}", language, 6, "C_L1_vs_L2",
                f"smallest vs smallest_to_{direction}")
    # D. simple nearest vs direction nearest
    for language in ("zh", "en"):
        add("largest_to_nearest", language, 7, "D_nearest_vs_direction_nearest",
            "largest_to_nearest vs largest_to_<dir>_to_nearest")
        add("smallest_to_nearest", language, 7, "D_nearest_vs_direction_nearest",
            "smallest_to_nearest vs smallest_to_<dir>_to_nearest")
        for direction in DIRECTIONS:
            add(f"largest_to_{direction}_to_nearest", language, 7,
                "D_nearest_vs_direction_nearest",
                "largest_to_nearest vs largest_to_<dir>_to_nearest")

    counts = Counter(row["expected_program"] for row in prompts)
    groups = Counter(row["contrast_group"] for row in prompts)
    validation = validate_pack(prompts, "minimal_pairs")
    return {
        "_doc": (
            "Task 6T section 6.2. Frozen semantic minimal-pair pack, created BEFORE training and using "
            "held-out template/lexical forms that the training augmentation grammar never generates. "
            "Contrast groups: A largest vs smallest per direction, B direction-only vs "
            "direction+nearest, C L1 vs L2, D simple-nearest vs direction-nearest."
        ),
        "task": "6T", "stage": "frozen-minimal-pairs",
        "frozen_before_training": True,
        "generator": "scripts/task6t_build_parser_data.py::build_minimal_pairs",
        "prompt_count": len(prompts),
        "languages": dict(Counter(row["language"] for row in prompts)),
        "programs_covered": len(counts),
        "contrast_groups": dict(groups),
        "prompts_per_program": dict(sorted(counts.items())),
        "validation": validation,
        "prompts": prompts,
    }


def build_stress() -> dict:
    """Section 6.3 — >= 120 held-out stress prompts, all 20 classes, >= 6 per class (>= 3 + 3)."""

    prompts = []
    for program in PROGRAMS:
        for language in ("zh", "en"):
            for form in range(4):
                prompts.append({
                    "prompt": eval_prompt(program, language, form),
                    "language": language,
                    "expected_program": program,
                    "length": "short" if form % 2 == 0 else "long",
                })
    counts = Counter(row["expected_program"] for row in prompts)
    per_language = Counter((row["expected_program"], row["language"]) for row in prompts)
    validation = validate_pack(prompts, "stress_v1")
    return {
        "_doc": (
            "Task 6T section 6.3. Frozen held-out stress pack, created BEFORE training; never extended "
            "after observing model failures. Every canonical class is represented by 4 Chinese and 4 "
            "English prompts using held-out template/lexical forms, including a minority of held-out "
            "synonyms (占地面积最大/占地面积最小/楼房区域; most extensive/least extensive/structure)."
        ),
        "task": "6T", "stage": "frozen-stress-v1",
        "frozen_before_training": True,
        "generator": "scripts/task6t_build_parser_data.py::build_stress",
        "prompt_count": len(prompts),
        "languages": dict(Counter(row["language"] for row in prompts)),
        "programs_covered": len(counts),
        "minimum_per_class": min(counts.values()),
        "minimum_per_class_per_language": min(per_language.values()),
        "all_classes_represented": len(counts) == len(PROGRAMS),
        "validation": validation,
        "prompts": prompts,
    }


# --------------------------------------------------------------------------- TRAIN-side augmentation

TRAIN_ZH_VERBS = ("请分割", "请标出", "分割", "标出", "请找出", "找出")
TRAIN_ZH_LONG_VERBS = TRAIN_ZH_VERBS
TRAIN_ZH_SHORT_VERBS = ("请分割", "分割", "请标出", "标出")  # never "找出" (fixed-24 collision guard)
TRAIN_ZH_NOUNS = ("建筑区域", "建筑物", "建筑", "楼房", "房屋")
TRAIN_ZH_SHORT_NOUNS = ("建筑", "建筑物", "楼房")
TRAIN_ZH_SUP_LONG = {"largest": "面积最大的", "smallest": "面积最小的"}
TRAIN_ZH_SUP_SHORT = {"largest": "最大", "smallest": "最小"}
TRAIN_ZH_DIRS = {"left_of": ("左侧", "左边"), "right_of": ("右侧", "右边"),
                 "above": ("上方", "上面"), "below": ("下方", "下面")}
TRAIN_ZH_NEAREST = ("最近的", "距离最近的", "最靠近的")
TRAIN_EN_VERBS = ("Please segment", "Segment", "Please outline", "Outline", "Mark", "Please mark")
TRAIN_EN_NOUNS = ("building region", "building", "structure", "building structure")
TRAIN_EN_SUP = {"largest": ("largest", "greatest-area"), "smallest": ("smallest", "least-area")}
TRAIN_EN_DIRS = {"left_of": ("to the left of", "left of"), "right_of": ("to the right of", "right of"),
                 "above": ("above", "over"), "below": ("below", "under")}
TRAIN_EN_NEAREST = ("nearest", "closest", "with the smallest boundary distance to it")
TRAIN_EN_L1 = {"leftmost": ("furthest left", "leftmost"), "rightmost": ("furthest right", "rightmost"),
               "topmost": ("highest", "topmost"), "bottommost": ("lowest", "bottommost")}
TRAIN_ZH_L1 = {"leftmost": ("最左侧的", "最左边的"), "rightmost": ("最右侧的", "最右边的"),
               "topmost": ("最上方的", "最上面的"), "bottommost": ("最下方的", "最下面的")}


def build_augmentation() -> tuple[list[dict], dict]:
    """Deterministic paraphrase augmentation over the 20 canonical programs (both languages)."""

    rows: list[dict] = []

    def add(prompt: str, program: str, language: str, template: str) -> None:
        rows.append({"prompt": prompt, "program": program, "language": language,
                     "template": template, "source": "augmented"})

    for program in PROGRAMS:
        zh_rows: list[tuple[str, str]] = []
        en_rows: list[tuple[str, str]] = []
        if program in TRAIN_ZH_L1:
            for verb in TRAIN_ZH_VERBS:
                for superlative in TRAIN_ZH_L1[program]:
                    for noun in TRAIN_ZH_NOUNS:
                        zh_rows.append((f"{verb}{superlative}{noun}。", "l1"))
            for verb in TRAIN_EN_VERBS:
                for superlative in TRAIN_EN_L1[program]:
                    for noun in TRAIN_EN_NOUNS:
                        en_rows.append((f"{verb} the {superlative} {noun}.", "l1"))
        elif program in ("largest", "smallest"):
            for verb in TRAIN_ZH_VERBS:
                for noun in TRAIN_ZH_NOUNS:
                    zh_rows.append((f"{verb}{TRAIN_ZH_SUP_LONG[program]}{noun}。", "l1_long"))
                    zh_rows.append((f"{verb}{TRAIN_ZH_SUP_SHORT[program]}{noun}。", "l1_short"))
            for verb in TRAIN_EN_VERBS:
                for noun in TRAIN_EN_NOUNS:
                    for superlative in TRAIN_EN_SUP[program]:
                        en_rows.append((f"{verb} the {superlative} {noun}.", "l1_long"))
        elif program in ("largest_to_nearest", "smallest_to_nearest"):
            family = "largest" if program.startswith("largest") else "smallest"
            for verb in TRAIN_ZH_VERBS:
                for noun in TRAIN_ZH_NOUNS:
                    for nearest in TRAIN_ZH_NEAREST:
                        zh_rows.append((f"以{TRAIN_ZH_SUP_LONG[family]}{noun}为参照，"
                                        f"{verb}与它{nearest}的另一{noun}。", "l2_nearest"))
                        zh_rows.append((f"先确定{TRAIN_ZH_SUP_LONG[family]}{noun}，再{verb}"
                                        f"{nearest}的另一{noun}。", "l2_nearest"))
            for verb in TRAIN_EN_VERBS:
                for noun in TRAIN_EN_NOUNS:
                    for nearest in TRAIN_EN_NEAREST:
                        en_rows.append((f"Using the {TRAIN_EN_SUP[family][0]} {noun} as reference, "
                                        f"{verb.lower()} the other {noun} {nearest}.", "l2_nearest"))
                        en_rows.append((f"First determine the {TRAIN_EN_SUP[family][0]} {noun}, then "
                                        f"{verb.lower()} the other {noun} {nearest}.", "l2_nearest"))
        else:
            parts = program.split("_to_")
            family = parts[0]
            rest = "_to_".join(parts[1:])
            nearest = rest.endswith("_to_nearest")
            direction = rest[:-len("_to_nearest")] if nearest else rest
            # long reference form (面积最大的建筑区域) and the short form (最大建筑)
            for verb in TRAIN_ZH_LONG_VERBS:
                for noun in TRAIN_ZH_NOUNS:
                    for zh_dir in TRAIN_ZH_DIRS[direction]:
                        if nearest:
                            for nearest_word in TRAIN_ZH_NEAREST:
                                zh_rows.append((f"先确定{TRAIN_ZH_SUP_LONG[family]}{noun}，再从它"
                                                f"{zh_dir}的{noun}中选出{nearest_word}一个并"
                                                f"{verb.replace('请', '')}。", "l3_nearest"))
                        else:
                            zh_rows.append((f"{verb}位于{TRAIN_ZH_SUP_LONG[family]}{noun}{zh_dir}的"
                                            f"{noun}。", "l2_direction"))
            for verb in TRAIN_ZH_SHORT_VERBS:
                for noun in TRAIN_ZH_SHORT_NOUNS:
                    for zh_dir in TRAIN_ZH_DIRS[direction]:
                        if not nearest:
                            zh_rows.append((f"{verb}{TRAIN_ZH_SUP_SHORT[family]}{noun}{zh_dir}的"
                                            f"{noun}。", "l2_direction_short"))
            for verb in TRAIN_EN_VERBS:
                for noun in TRAIN_EN_NOUNS:
                    for en_dir in TRAIN_EN_DIRS[direction]:
                        if nearest:
                            for nearest_word in TRAIN_EN_NEAREST:
                                en_rows.append((f"First locate the {TRAIN_EN_SUP[family][0]} {noun}, "
                                                f"then from the {noun} {en_dir} it {verb.lower()} the "
                                                f"one {nearest_word}.", "l3_nearest"))
                        else:
                            en_rows.append((f"{verb} the {noun} {en_dir} the "
                                            f"{TRAIN_EN_SUP[family][0]} {noun}.", "l2_direction"))
                            en_rows.append((f"{verb} the {noun} {en_dir} the "
                                            f"{TRAIN_EN_SUP[family][1]} {noun}.", "l2_direction"))
        zh_rows = sorted(set(zh_rows))[:AUGMENTED_PER_PROGRAM_PER_LANGUAGE]
        en_rows = sorted(set(en_rows))[:AUGMENTED_PER_PROGRAM_PER_LANGUAGE]
        for prompt, template in zh_rows:
            add(prompt, program, "zh", template)
        for prompt, template in en_rows:
            add(prompt, program, "en", template)

    spec = {
        "_doc": (
            "Task 6T section 5.2. Tracked grammar specification for the deterministic paraphrase "
            "augmentation of the ProgramHead training data. Only BuildSpatialReason v0.2 TRAIN split "
            "instruction text and this grammar (derived from train-split/program semantics) are used "
            "for optimisation. The grammar covers reference-family contrasts (largest vs smallest), "
            "level contrasts (L1 / direction / nearest / direction+nearest), all four directions and "
            "lexical variation in both languages. It never reproduces an exact evaluation string; a "
            "deterministic pre-training filter removes any residual collision."
        ),
        "task": "6T", "seed": SEED,
        "cap_per_program_per_language": AUGMENTED_PER_PROGRAM_PER_LANGUAGE,
        "generated_prompts": len(rows),
        "generated_by_language": dict(Counter(row["language"] for row in rows)),
        "generated_by_program": dict(sorted(Counter(row["program"] for row in rows).items())),
        "generated_by_template": dict(sorted(Counter(row["template"] for row in rows).items())),
        "inventories": {
            "zh_verbs": list(TRAIN_ZH_VERBS), "zh_short_verbs": list(TRAIN_ZH_SHORT_VERBS),
            "zh_nouns": list(TRAIN_ZH_NOUNS), "zh_short_nouns": list(TRAIN_ZH_SHORT_NOUNS),
            "zh_superlatives_long": TRAIN_ZH_SUP_LONG, "zh_superlatives_short": TRAIN_ZH_SUP_SHORT,
            "zh_directions": {key: list(value) for key, value in TRAIN_ZH_DIRS.items()},
            "zh_nearest": list(TRAIN_ZH_NEAREST), "zh_l1": TRAIN_ZH_L1,
            "en_verbs": list(TRAIN_EN_VERBS), "en_nouns": list(TRAIN_EN_NOUNS),
            "en_superlatives": {key: list(value) for key, value in TRAIN_EN_SUP.items()},
            "en_directions": {key: list(value) for key, value in TRAIN_EN_DIRS.items()},
            "en_nearest": list(TRAIN_EN_NEAREST), "en_l1": TRAIN_EN_L1,
        },
        "contrast_coverage": {
            "reference_family": "largest vs smallest in L1, L2, L2-nearest and L3-nearest forms",
            "level": ["largest", "largest_to_direction", "largest_to_nearest",
                      "largest_to_direction_to_nearest"],
            "direction": list(DIRECTIONS),
            "lexical_variation": {
                "zh": ["面积最大/最大/最大的", "面积最小/最小/最小的", "左侧/左边", "右侧/右边",
                       "上方/上面", "下方/下面", "最近/距离最近/最靠近"],
                "en": ["largest/greatest-area", "smallest/least-area", "to the left of/left of",
                       "to the right of/right of", "above/over", "below/under",
                       "nearest/closest"],
            },
        },
        "forbidden_shortcuts": [
            "keyword remapping", "regex/class-id override", "hard-coded handling of specific failures",
            "deterministic largest/smallest post-correction", "bypassing ProgramHead",
        ],
    }
    return rows, spec


# --------------------------------------------------------------------------- leakage + filter


def frozen_eval_prompts() -> dict[str, list[str]]:
    """Every frozen evaluation prompt that training text must not overlap with."""

    groups: dict[str, list[str]] = {"task6s_fixed24": list(FIXED24)}
    minimal = json.loads(OUT_MINIMAL.read_text(encoding="utf-8"))
    groups["task6t_minimal_pairs"] = [row["prompt"] for row in minimal["prompts"]]
    stress = json.loads(OUT_STRESS.read_text(encoding="utf-8"))
    groups["task6t_stress_v1"] = [row["prompt"] for row in stress["prompts"]]
    mini_val = json.loads((PACK_ROOT / "mini_val_240.json").read_text(encoding="utf-8"))
    instructions = {}
    with (V02 / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[record["sample_id"]] = record
    groups["minival240_queries"] = [instructions[sample["sample_id"]]["instruction_en"]
                                    for sample in mini_val["records"]]
    paired = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))
    paired_prompts = []
    for pair in paired["pairs"]:
        for side in ("a", "b"):
            paired_prompts.append(instructions[pair[side]["sample_id"]]["instruction_en"])
    groups["pairedval20_queries"] = paired_prompts
    return groups


def filter_collisions(rows: list[dict], groups: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    exact = {prompt for prompts in groups.values() for prompt in prompts}
    normalized = {normalize_prompt(prompt) for prompt in exact}
    kept, dropped = [], []
    for row in rows:
        prompt = row["prompt"]
        if prompt in exact or normalize_prompt(prompt) in normalized:
            dropped.append(row)
        else:
            kept.append(row)
    return kept, dropped


def load_train_records() -> list[dict]:
    records = []
    with (V02 / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append({"prompt": str(record["instruction_en"]), "program": record["query_type"],
                            "language": "en", "template": "v02_train",
                            "source": "build_spatial_reason_v02_train"})
            records.append({"prompt": str(record["instruction_zh"]), "program": record["query_type"],
                            "language": "zh", "template": "v02_train",
                            "source": "build_spatial_reason_v02_train"})
    return records


def run_packs(args) -> int:
    minimal = build_minimal_pairs()
    stress = build_stress()
    write_json(OUT_MINIMAL, minimal)
    write_json(OUT_STRESS, stress)
    print(f"[6t.data] minimal pairs {minimal['prompt_count']} prompts, groups "
          f"{minimal['contrast_groups']}", flush=True)
    print(f"[6t.data] stress v1 {stress['prompt_count']} prompts, classes "
          f"{stress['programs_covered']}, min/class {stress['minimum_per_class']}, "
          f"min/class/language {stress['minimum_per_class_per_language']}", flush=True)
    return 0


def run_augment(args) -> int:
    started = time.time()
    groups = frozen_eval_prompts()
    augmented, dropped_augmented = filter_collisions(build_augmentation()[0], groups)
    _, spec = build_augmentation()
    original = load_train_records()
    train_kept, dropped_train = filter_collisions(original, groups)

    AUGMENT_ROOT.mkdir(parents=True, exist_ok=True)
    augmented_path = AUGMENT_ROOT / "parser_train_augmented.jsonl"
    with augmented_path.open("w", encoding="utf-8") as handle:
        for row in augmented:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    combined_path = AUGMENT_ROOT / "parser_train_combined.jsonl"
    with combined_path.open("w", encoding="utf-8") as handle:
        for row in train_kept + augmented:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    spec.update({
        "outputs": {
            "augmented": str(augmented_path),
            "combined": str(combined_path),
            "augmented_is_gitignored": True,
        },
        "counts": {
            "build_spatial_reason_v02_train_records": len(original) // 2,
            "v02_train_examples_both_languages": len(original),
            "v02_train_examples_kept": len(train_kept),
            "augmented_generated": len(augmented) + len(dropped_augmented),
            "augmented_kept": len(augmented),
            "combined": len(train_kept) + len(augmented),
        },
        "deterministic_collision_filter": {
            "dropped_train_examples": len(dropped_train),
            "dropped_augmented": len(dropped_augmented),
            "dropped_train_examples_sample": [row["prompt"] for row in dropped_train[:5]],
            "dropped_augmented_sample": [row["prompt"] for row in dropped_augmented[:5]],
            "reason": (
                "exact or normalized overlap with a frozen evaluation prompt; removed before training "
                "so that the frozen leakage audit is exactly zero"
            ),
        },
        "runtime_seconds": round(time.time() - started, 1),
    })
    write_json(OUT_SPEC, spec)
    print(f"[6t.data] augmented kept {len(augmented)} (dropped {len(dropped_augmented)}); v0.2 train "
          f"kept {len(train_kept)} (dropped {len(dropped_train)}); combined "
          f"{len(train_kept) + len(augmented)}", flush=True)
    return 0


def run_leakage(args) -> int:
    groups = frozen_eval_prompts()
    combined_path = AUGMENT_ROOT / "parser_train_combined.jsonl"
    train_rows = [json.loads(line) for line in combined_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    exact_index: dict[str, list[str]] = defaultdict(list)
    normalized_index: dict[str, list[str]] = defaultdict(list)
    for group, prompts in groups.items():
        for prompt in prompts:
            exact_index[prompt].append(group)
            normalized_index[normalize_prompt(prompt)].append(group)

    per_group = {}
    exact_hits, normalized_hits = [], []
    for row in train_rows:
        prompt = row["prompt"]
        if prompt in exact_index:
            exact_hits.append({"prompt": prompt, "groups": exact_index[prompt]})
        normalized = normalize_prompt(prompt)
        if normalized in normalized_index:
            normalized_hits.append({"prompt": prompt, "normalized": normalized,
                                    "groups": normalized_index[normalized]})
    for group in groups:
        prompts = groups[group]
        per_group[group] = {
            "prompts": len(prompts),
            "exact_overlaps": sum(1 for hit in exact_hits if group in hit["groups"]),
            "normalized_overlaps": sum(1 for hit in normalized_hits if group in hit["groups"]),
        }
    payload = {
        "_doc": (
            "Task 6T section 5.3. Exact UTF-8 and normalized-string leakage audit of every training "
            "prompt against MiniVal240 queries, PairedVal20 queries, the exact Task 6S fixed-24 "
            "paraphrase list and the Task 6T frozen minimal-pair and stress packs. Zero overlap is "
            "required; the audit runs on the final combined training set that is actually trained on."
        ),
        "task": "6T", "stage": "leakage-audit",
        "training_prompts": len(train_rows),
        "training_sources": dict(Counter(row["source"] for row in train_rows)),
        "evaluation_groups": {group: len(prompts) for group, prompts in groups.items()},
        "normalization": "strip whitespace, lowercase ASCII, normalise CJK/ASCII punctuation, trim "
                         "trailing sentence punctuation",
        "per_group": per_group,
        "exact_overlap_count": len(exact_hits),
        "normalized_overlap_count": len(normalized_hits),
        "exact_overlaps": exact_hits[:20],
        "normalized_overlaps": normalized_hits[:20],
        "verdict": "NO_LEAKAGE" if not exact_hits and not normalized_hits else "INVALID_EXPERIMENT",
    }
    write_json(OUT_LEAKAGE, payload)
    print(f"[6t.data] leakage exact {len(exact_hits)} normalized {len(normalized_hits)} over "
          f"{len(train_rows)} training prompts -> {payload['verdict']}", flush=True)
    return 0 if payload["verdict"] == "NO_LEAKAGE" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("packs", "augment", "leakage"), required=True)
    args = parser.parse_args(argv)
    if args.stage == "packs":
        return run_packs(args)
    return run_augment(args) if args.stage == "augment" else run_leakage(args)


if __name__ == "__main__":
    raise SystemExit(main())
