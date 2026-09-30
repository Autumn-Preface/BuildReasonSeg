"""Task 7B Part C-D — freeze the evaluation packs, build the training data, audit leakage.

Everything in Part C is written **before** any training run:

* `evaluation/task7b_compositional_minimal_pairs.json` — exactly 96 prompts in four contrast families
  (M1 L2-direction vs L3 direction+nearest = 48, M2 L1-largest vs L2-direction = 16,
  M3 nearest-only vs direction+nearest = 16, M4 largest vs smallest in L2 = 16);
* `evaluation/task7b_l3_stress_v1.json` — exactly 192 prompts with the section-6 coverage;
* `evaluation/task7b_eval_prompt_manifest.json` — pre-training SHA256 manifest of every evaluation source.

Part D then builds the training data from the BuildSpatialReason v0.2 train split plus exactly 4,800
deterministic train-only compositional prompts (section 9 allocation), removes every original train record
whose text matches an evaluation prompt exactly or after normalization (section 8), and requires zero exact
and zero normalized overlap with every evaluation source (section 11).

The evaluation lexical pools are reserved: they are never emitted by the training grammar, and no Task 7A
fixed24 string is reproduced.

    python scripts/task7b_build_parser_data.py
"""

from __future__ import annotations

import argparse
import hashlib
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

EVAL = REPO_ROOT / "evaluation"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
AUGMENT_ROOT = REPO_ROOT / "artifacts" / "task7b" / "parser_train_augmented"
OUT_MINIMAL = EVAL / "task7b_compositional_minimal_pairs.json"
OUT_STRESS = EVAL / "task7b_l3_stress_v1.json"
OUT_MANIFEST = EVAL / "task7b_eval_prompt_manifest.json"
OUT_SPEC = EVAL / "task7b_train_augmentation_spec.json"
OUT_LEAKAGE = EVAL / "task7b_parser_leakage_audit.json"
TASK7A_FIXED24 = EVAL / "task7a_l3_paraphrase_pack.json"
SEED = 20261001
DIRECTIONS = ("left", "right", "above", "below")
#: Canonical programme-id fragments: left/right carry `_of`, above/below do not.
DIRECTION_SUFFIX = {"left": "left_of", "right": "right_of", "above": "above", "below": "below"}
L3_PROGRAMS = tuple(f"largest_to_{DIRECTION_SUFFIX[direction]}_to_nearest"
                    for direction in DIRECTIONS)
L2_LARGEST = tuple(f"largest_to_{DIRECTION_SUFFIX[direction]}" for direction in DIRECTIONS)
L2_SMALLEST = tuple(f"smallest_to_{DIRECTION_SUFFIX[direction]}" for direction in DIRECTIONS)
L3_BY_DIRECTION = {direction: f"largest_to_{DIRECTION_SUFFIX[direction]}_to_nearest"
                   for direction in DIRECTIONS}
L2_LARGEST_BY_DIRECTION = {direction: f"largest_to_{DIRECTION_SUFFIX[direction]}"
                           for direction in DIRECTIONS}
L2_SMALLEST_BY_DIRECTION = {direction: f"smallest_to_{DIRECTION_SUFFIX[direction]}"
                            for direction in DIRECTIONS}
DIRECTION_FROM_SUFFIX = {suffix: direction for direction, suffix in DIRECTION_SUFFIX.items()}
EXPECTED_PROGRAM_IDS = (
    "leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest", "largest_to_nearest",
    "smallest_to_nearest",
    "largest_to_above", "largest_to_below", "largest_to_left_of", "largest_to_right_of",
    "smallest_to_above", "smallest_to_below", "smallest_to_left_of", "smallest_to_right_of",
    "largest_to_above_to_nearest", "largest_to_below_to_nearest", "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
)
#: Section 9 allocation: exactly 4,800 synthetic training prompts.
AUGMENT_ALLOCATION = {
    **{program: 600 for program in L3_PROGRAMS},
    **{program: 300 for program in L2_LARGEST},
    "largest_to_nearest": 400, "smallest_to_nearest": 200,
    **{program: 150 for program in L2_SMALLEST},
}
AUGMENT_TOTAL = sum(AUGMENT_ALLOCATION.values())

# ---------------------------------------------------------------- reserved evaluation lexical pools
# These forms are reserved for evaluation only; the training grammar never emits them.
EVAL_ZH_DIRECTION = {"left": "左侧", "right": "右侧", "above": "上面", "below": "下面"}
EVAL_EN_DIRECTION = {"left": "left", "right": "right", "above": "above", "below": "below"}
EVAL_ZH_REFERENCE = "最大建筑"
EVAL_EN_REFERENCE = "the largest building"
EVAL_ZH_NEAREST = ("最近的", "距离最近的", "最靠近的")
EVAL_EN_NEAREST = ("nearest", "closest", "with the minimum distance", "having the smallest gap")
EVAL_ZH_NOUN = ("建筑", "楼房", "建筑区域")
EVAL_EN_NOUN = ("structure", "plot", "building region", "footprint")
EVAL_ZH_VERB = ("请分割", "标出", "请圈定", "勾勒出")
EVAL_EN_VERB = ("delineate", "outline", "mark", "segment")

# ---------------------------------------------------------------- training grammar pools (section 10)
TRAIN_ZH_LARGEST = ("面积最大的", "规模最大的", "占地最大的")
TRAIN_ZH_SMALLEST = ("面积最小的", "规模最小的", "占地最小的")
TRAIN_ZH_DIRECTION = {"left": ("左侧", "位于左侧"), "right": ("右侧", "位于右侧"),
                      "above": ("上方", "位于上方"), "below": ("下方", "位于下方")}
TRAIN_ZH_NEAREST = ("距离最近的", "与其距离最近的", "最靠近该方向区域的")
TRAIN_ZH_NOUN = ("建筑物", "建筑体", "地块")
TRAIN_ZH_VERB = ("请分割", "请标注", "请提取", "标注")
TRAIN_EN_LARGEST = ("the largest", "the greatest-area", "the building with the greatest footprint")
TRAIN_EN_SMALLEST = ("the smallest", "the least-area", "the building with the smallest footprint")
TRAIN_EN_DIRECTION = {"left": ("to the left of",), "right": ("to the right of",),
                      "above": ("above",), "below": ("below",)}
TRAIN_EN_NEAREST = ("the nearest", "the closest", "with the minimum distance")
TRAIN_EN_NOUN = ("building", "structure", "parcel")
TRAIN_EN_VERB = ("segment", "label", "extract", "mark")


def normalize_prompt(prompt: str) -> str:
    """Unicode strip, lowercase, collapse whitespace, normalize punctuation, strip terminal marks."""

    text = prompt.strip().lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    for source, target in (("，", ","), ("。", "."), ("？", "?"), ("！", "!"), ("、", ","),
                           ("：", ":"), ("；", ";"), ("（", "("), ("）", ")"), ("“", '"'),
                           ("”", '"'), ("‘", "'"), ("’", "'"), ("　", "")):
        text = text.replace(source, target)
    text = re.sub(r"[.,!?;:]+$", "", text)
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- Part C: minimal pairs (96)


def build_minimal_pairs() -> list[dict]:
    rows: list[dict] = []

    def add(contrast: str, language: str, direction: str | None, prompt: str, program: str) -> None:
        rows.append({"id": f"M{len(rows) + 1:03d}", "contrast_group": contrast,
                     "language": language, "direction": direction, "prompt": prompt,
                     "expected_program": program})

    # M1 — L2 direction vs L3 direction+nearest: 4 directions x (3 zh + 3 en) matched pairs = 48
    zh_m1 = (
        ("请分割{ref}{dir}的建筑。", "请分割{ref}{dir}{near}建筑。"),
        ("标出{ref}{dir}的楼房。", "标出{ref}{dir}{near}楼房。"),
        ("请圈定{ref}{dir}的建筑区域。", "请圈定{ref}{dir}{near}建筑区域。"),
    )
    en_m1 = (
        ("delineate the structure {dir} {ref}", "delineate the structure {near} {dir} {ref}"),
        ("outline the plot on the {side} side of {ref}",
         "outline the plot on the {side} side of {ref} {near}"),
        ("mark the region lying {dir} {ref}",
         "mark the region lying {dir} {ref} {near}"),
    )
    for direction in DIRECTIONS:
        for zh_l2, zh_l3 in zh_m1:
            for template, program in ((zh_l2, L2_LARGEST_BY_DIRECTION[direction]),
                                      (zh_l3, L3_BY_DIRECTION[direction])):
                add("M1", "zh", direction,
                    template.format(ref=EVAL_ZH_REFERENCE, dir=EVAL_ZH_DIRECTION[direction],
                                    near=EVAL_ZH_NEAREST[0]), program)
        for en_l2, en_l3 in en_m1:
            for template, program in ((en_l2, L2_LARGEST_BY_DIRECTION[direction]),
                                      (en_l3, L3_BY_DIRECTION[direction])):
                text = template.format(ref=EVAL_EN_REFERENCE, dir=EVAL_EN_DIRECTION[direction],
                                       side=EVAL_EN_DIRECTION[direction],
                                       near=EVAL_EN_NEAREST[0])
                add("M1", "en", direction, text[0].upper() + text[1:], program)

    # M2 — L1 largest vs L2 direction: 4 directions x (1 zh pair + 1 en pair) = 16
    for direction in DIRECTIONS:
        add("M2", "zh", direction, "请分割最大建筑。", "largest")
        add("M2", "zh", direction,
            f"请分割{EVAL_ZH_REFERENCE}{EVAL_ZH_DIRECTION[direction]}的建筑。",
            L2_LARGEST_BY_DIRECTION[direction])
        add("M2", "en", direction, "Delineate the largest building.", "largest")
        add("M2", "en", direction,
            f"Delineate the structure {EVAL_EN_DIRECTION[direction]} {EVAL_EN_REFERENCE}.",
            L2_LARGEST_BY_DIRECTION[direction])

    # M3 — nearest-only vs direction+nearest: 4 directions x (1 zh pair + 1 en pair) = 16
    for direction in DIRECTIONS:
        add("M3", "zh", direction, "请分割距离最大建筑最近的建筑。", "largest_to_nearest")
        add("M3", "zh", direction,
            f"请分割{EVAL_ZH_REFERENCE}{EVAL_ZH_DIRECTION[direction]}{EVAL_ZH_NEAREST[0]}建筑。",
            L3_BY_DIRECTION[direction])
        add("M3", "en", direction,
            "Delineate the structure nearest to the largest building.", "largest_to_nearest")
        add("M3", "en", direction,
            f"Delineate the structure nearest to the {EVAL_EN_DIRECTION[direction]} of "
            f"{EVAL_EN_REFERENCE}.", L3_BY_DIRECTION[direction])

    # M4 — largest vs smallest family in L2: 4 directions x (1 zh pair + 1 en pair) = 16
    for direction in DIRECTIONS:
        add("M4", "zh", direction,
            f"请分割{EVAL_ZH_REFERENCE}{EVAL_ZH_DIRECTION[direction]}的建筑。",
            L2_LARGEST_BY_DIRECTION[direction])
        add("M4", "zh", direction,
            f"请分割{EVAL_ZH_REFERENCE.replace('最大', '最小')}{EVAL_ZH_DIRECTION[direction]}的建筑。",
            L2_SMALLEST_BY_DIRECTION[direction])
        add("M4", "en", direction,
            f"Delineate the structure {EVAL_EN_DIRECTION[direction]} {EVAL_EN_REFERENCE}.",
            L2_LARGEST_BY_DIRECTION[direction])
        add("M4", "en", direction,
            f"Delineate the structure {EVAL_EN_DIRECTION[direction]} the smallest building.",
            L2_SMALLEST_BY_DIRECTION[direction])
    return rows


# ---------------------------------------------------------------- Part C: stress pack (192)


def build_stress() -> list[dict]:
    rows: list[dict] = []

    def add(program: str, language: str, prompt: str, form: str) -> None:
        rows.append({"id": f"S{len(rows) + 1:03d}", "expected_program": program,
                     "language": language, "form": form, "prompt": prompt})

    # 4 L3 classes x 24 (12 Chinese + 12 English) = 96
    zh_structures = (
        ("short", "请分割{ref}{dir}{near}{noun}。"),
        ("medium", "请先找到图中的{ref}，然后分割它{dir}{near}{noun}。"),
        ("long", "请定位画面中的{ref}，再从中挑选出位于它{dir}、并且{near}{noun}并完成分割。"),
        ("order_v2", "{ref}{dir}{near}{noun}是哪一个？请分割出来。"),
        ("order_v3", "在{ref}{dir}的方向上，{noun}里{near}的是哪一个？请标出。"),
        ("order_v4", "标出与{ref}在{dir}方向上{near}{noun}。"),
    )
    en_structures = (
        ("short", "Delineate the {noun} {near} to the {dir} of {ref}."),
        ("medium", "First locate {ref} in the tile, then delineate the {noun} {near} to its {dir} "
                   "side."),
        ("long", "After identifying {ref}, delineate the {noun} that lies {dir} of it and is {near}."),
        ("order_v2", "Which {noun} is {near} to the {dir} of {ref}? Delineate it."),
        ("order_v3", "On the {dir} side of {ref}, mark the {noun} that is {near}."),
        ("order_v4", "Outline the {noun} {dir} {ref} that is {near}."),
    )
    for direction in DIRECTIONS:
        program = L3_BY_DIRECTION[direction]
        for index in range(12):
            form, template = zh_structures[index % len(zh_structures)]
            near = EVAL_ZH_NEAREST[(index // len(zh_structures)) % len(EVAL_ZH_NEAREST)]
            noun = EVAL_ZH_NOUN[index % len(EVAL_ZH_NOUN)]
            add(program, "zh",
                template.format(ref=EVAL_ZH_REFERENCE, dir=EVAL_ZH_DIRECTION[direction], near=near,
                                noun=noun), f"{form}_{index}")
        for index in range(12):
            form, template = en_structures[index % len(en_structures)]
            near = EVAL_EN_NEAREST[(index // len(en_structures)) % len(EVAL_EN_NEAREST)]
            noun = EVAL_EN_NOUN[index % len(EVAL_EN_NOUN)]
            prompt = template.format(ref=EVAL_EN_REFERENCE, dir=EVAL_EN_DIRECTION[direction],
                                     near=near, noun=noun)
            add(program, "en", prompt[0].upper() + prompt[1:], f"{form}_{index}")

    # corresponding 4 L2 direction classes x 12 each = 48
    zh_l2_structures = (
        ("l2_short", "请分割{ref}{dir}的{noun}。"),
        ("l2_medium", "标出位于{ref}{dir}的一栋{noun}。"),
        ("l2_long", "请圈定{ref}{dir}的那一片{noun}区域。"),
    )
    en_l2_structures = (
        ("l2_short", "Delineate the {noun} {dir} {ref}."),
        ("l2_medium", "Outline a {noun} on the {dir} side of {ref}."),
        ("l2_long", "Mark the {noun} region that lies {dir} {ref}."),
    )
    for direction in DIRECTIONS:
        program = L2_LARGEST_BY_DIRECTION[direction]
        for index in range(6):
            form, template = zh_l2_structures[index % len(zh_l2_structures)]
            noun = EVAL_ZH_NOUN[index % len(EVAL_ZH_NOUN)]
            add(program, "zh",
                template.format(ref=EVAL_ZH_REFERENCE, dir=EVAL_ZH_DIRECTION[direction], noun=noun),
                f"{form}_{index}")
        for index in range(6):
            form, template = en_l2_structures[index % len(en_l2_structures)]
            noun = EVAL_EN_NOUN[index % len(EVAL_EN_NOUN)]
            prompt = template.format(ref=EVAL_EN_REFERENCE, dir=EVAL_EN_DIRECTION[direction],
                                     noun=noun)
            add(program, "en", prompt[0].upper() + prompt[1:], f"{form}_{index}")

    # largest_to_nearest = 16 (8 Chinese + 8 English)
    zh_nearest = (
        "请分割距离{ref}{near_marker}{noun}。",
        "标出与{ref}{near_marker}{noun}。",
        "请圈定{near_marker}的{noun}，参考物是{ref}。",
        "找出{ref}周边距离最小的{noun}并分割出来。",
    )
    en_nearest = (
        "Delineate the {noun} {near_marker} {ref}.",
        "Outline the {noun} that stands {near_marker} {ref}.",
        "Mark the {noun} with the smallest gap to {ref}.",
        "Which {noun} is {near_marker} {ref}? Delineate it.",
    )
    for index in range(4):
        for marker in (EVAL_ZH_NEAREST[index % len(EVAL_ZH_NEAREST)],):
            add("largest_to_nearest", "zh",
                zh_nearest[index].format(ref=EVAL_ZH_REFERENCE, noun=EVAL_ZH_NOUN[0],
                                         near_marker=marker), "nearest_zh")
            add("largest_to_nearest", "zh",
                zh_nearest[index].format(ref=EVAL_ZH_REFERENCE, noun=EVAL_ZH_NOUN[1],
                                         near_marker=marker), "nearest_zh_v2")
        marker = EVAL_EN_NEAREST[index % len(EVAL_EN_NEAREST)]
        for noun in (EVAL_EN_NOUN[0], EVAL_EN_NOUN[1]):
            prompt = en_nearest[index].format(ref=EVAL_EN_REFERENCE, noun=noun, near_marker=marker)
            add("largest_to_nearest", "en", prompt[0].upper() + prompt[1:], "nearest_en")

    # smallest_to_nearest = 8
    for index in range(4):
        add("smallest_to_nearest", "zh",
            zh_nearest[index].format(ref="最小建筑", noun=EVAL_ZH_NOUN[0],
                                     near_marker=EVAL_ZH_NEAREST[index % len(EVAL_ZH_NEAREST)]),
            "nearest_zh")
        prompt = en_nearest[index].format(ref="the smallest building", noun=EVAL_EN_NOUN[0],
                                          near_marker=EVAL_EN_NEAREST[index % len(EVAL_EN_NEAREST)])
        add("smallest_to_nearest", "en", prompt[0].upper() + prompt[1:], "nearest_en")

    # largest = 8, smallest = 8
    zh_l1 = ("请分割{ref}。", "标出{ref}。", "请圈定{ref}所在的{noun}。", "找出{ref}并分割。")
    en_l1 = ("Delineate {ref}.", "Outline {ref}.", "Mark the {noun} of {ref}.",
             "Locate and delineate {ref}.")
    for index in range(4):
        add("largest", "zh", zh_l1[index].format(ref=EVAL_ZH_REFERENCE, noun=EVAL_ZH_NOUN[0]), "l1")
        prompt = en_l1[index].format(ref=EVAL_EN_REFERENCE, noun=EVAL_EN_NOUN[0])
        add("largest", "en", prompt[0].upper() + prompt[1:], "l1")
    for index in range(4):
        add("smallest", "zh",
            zh_l1[index].format(ref="最小建筑", noun=EVAL_ZH_NOUN[0]), "l1")
        prompt = en_l1[index].format(ref="the smallest building", noun=EVAL_EN_NOUN[0])
        add("smallest", "en", prompt[0].upper() + prompt[1:], "l1")

    # leftmost / rightmost / topmost / bottommost total = 8
    extremes = (("leftmost", "最左侧的{noun}", "left-most"), ("rightmost", "最右侧的{noun}",
                                                              "right-most"),
                ("topmost", "最上方的{noun}", "top-most"), ("bottommost", "最下方的{noun}",
                                                            "bottom-most"))
    for program, phrase, english in extremes:
        add(program, "zh", f"请分割图中{phrase.format(noun=EVAL_ZH_NOUN[0])}。", "extreme")
        add(program, "en", f"Delineate the {english} {EVAL_EN_NOUN[0]} in the tile.", "extreme")
    return rows


# ---------------------------------------------------------------- Part D: training data


def build_augmentations() -> list[dict]:
    rows: list[dict] = []
    for program, count in AUGMENT_ALLOCATION.items():
        language_split = {"zh": count // 2, "en": count - count // 2}
        if program in L3_PROGRAMS:
            suffix = program[len("largest_to_"):-len("_to_nearest")]
            direction = DIRECTION_FROM_SUFFIX[suffix]
            for language, amount in language_split.items():
                for index in range(amount):
                    rows.append(_l3_row(program, direction, language, index))
        elif program in L2_LARGEST or program in L2_SMALLEST:
            family = "largest" if program.startswith("largest") else "smallest"
            suffix = program.split("_to_", 1)[1]
            direction = DIRECTION_FROM_SUFFIX[suffix]
            for language, amount in language_split.items():
                for index in range(amount):
                    rows.append(_l2_row(program, family, direction, language, index))
        elif program in ("largest_to_nearest", "smallest_to_nearest"):
            family = "largest" if program.startswith("largest") else "smallest"
            for language, amount in language_split.items():
                for index in range(amount):
                    rows.append(_nearest_row(program, family, language, index))
        else:  # pragma: no cover - allocation table is fixed
            raise ValueError(f"unexpected augmentation program {program}")
    return rows


def _l3_row(program: str, direction: str, language: str, index: int) -> dict:
    if language == "zh":
        verb = TRAIN_ZH_VERB[index % len(TRAIN_ZH_VERB)]
        reference = TRAIN_ZH_LARGEST[index % len(TRAIN_ZH_LARGEST)]
        directional = TRAIN_ZH_DIRECTION[direction][index % len(TRAIN_ZH_DIRECTION[direction])]
        nearest = TRAIN_ZH_NEAREST[index % len(TRAIN_ZH_NEAREST)]
        noun = TRAIN_ZH_NOUN[index % len(TRAIN_ZH_NOUN)]
        forms = (f"{verb}{reference}建筑物，再分割它{directional}{nearest}{noun}。",
                 f"{verb}位于{reference}建筑物{directional}且{nearest}{noun}。",
                 f"在{reference}建筑物{directional}的方向上，{verb}{nearest}{noun}。")
        prompt = forms[index % len(forms)]
    else:
        verb = TRAIN_EN_VERB[index % len(TRAIN_EN_VERB)]
        reference = TRAIN_EN_LARGEST[index % len(TRAIN_EN_LARGEST)]
        directional = TRAIN_EN_DIRECTION[direction][0]
        nearest = TRAIN_EN_NEAREST[index % len(TRAIN_EN_NEAREST)]
        noun = TRAIN_EN_NOUN[index % len(TRAIN_EN_NOUN)]
        forms = (f"{verb} {reference} building, then {verb} the {noun} {directional} it that is "
                 f"{nearest}.",
                 f"{verb} the {noun} {nearest} {directional} {reference} building.",
                 f"{verb} the {noun} that lies {directional} {reference} building and is {nearest}.")
        prompt = forms[index % len(forms)][0].upper() + forms[index % len(forms)][1:]
    return {"prompt": prompt, "program": program, "language": language,
            "source": "task7b_augmentation", "template_index": index}


def _l2_row(program: str, family: str, direction: str, language: str, index: int) -> dict:
    if language == "zh":
        verb = TRAIN_ZH_VERB[index % len(TRAIN_ZH_VERB)]
        reference = (TRAIN_ZH_LARGEST if family == "largest" else TRAIN_ZH_SMALLEST)[
            index % len(TRAIN_ZH_LARGEST)]
        directional = TRAIN_ZH_DIRECTION[direction][index % len(TRAIN_ZH_DIRECTION[direction])]
        noun = TRAIN_ZH_NOUN[index % len(TRAIN_ZH_NOUN)]
        forms = (f"{verb}{reference}{noun}{directional}的那一栋。",
                 f"{verb}位于{reference}{noun}{directional}的{noun}。",
                 f"在{reference}{noun}{directional}，{verb}对应的一栋{noun}。")
        prompt = forms[index % len(forms)]
    else:
        verb = TRAIN_EN_VERB[index % len(TRAIN_EN_VERB)]
        reference = (TRAIN_EN_LARGEST if family == "largest" else TRAIN_EN_SMALLEST)[
            index % len(TRAIN_EN_LARGEST)]
        directional = TRAIN_EN_DIRECTION[direction][0]
        noun = TRAIN_EN_NOUN[index % len(TRAIN_EN_NOUN)]
        forms = (f"{verb} the {noun} {directional} {reference} building.",
                 f"{verb} the {noun} that lies {directional} {reference} building.",
                 f"{verb} the {noun} on the {direction} of {reference} building.")
        prompt = forms[index % len(forms)]
        prompt = prompt[0].upper() + prompt[1:]
    return {"prompt": prompt, "program": program, "language": language,
            "source": "task7b_augmentation", "template_index": index}


def _nearest_row(program: str, family: str, language: str, index: int) -> dict:
    if language == "zh":
        verb = TRAIN_ZH_VERB[index % len(TRAIN_ZH_VERB)]
        reference = (TRAIN_ZH_LARGEST if family == "largest" else TRAIN_ZH_SMALLEST)[
            index % len(TRAIN_ZH_LARGEST)]
        nearest = TRAIN_ZH_NEAREST[index % len(TRAIN_ZH_NEAREST)]
        noun = TRAIN_ZH_NOUN[index % len(TRAIN_ZH_NOUN)]
        forms = (f"{verb}{reference}建筑物{nearest}{noun}。",
                 f"{verb}与{reference}建筑物{nearest}{noun}。",
                 f"{verb}到{reference}建筑物为止{nearest}{noun}。")
        prompt = forms[index % len(forms)]
    else:
        verb = TRAIN_EN_VERB[index % len(TRAIN_EN_VERB)]
        reference = (TRAIN_EN_LARGEST if family == "largest" else TRAIN_EN_SMALLEST)[
            index % len(TRAIN_EN_LARGEST)]
        nearest = TRAIN_EN_NEAREST[index % len(TRAIN_EN_NEAREST)]
        noun = TRAIN_EN_NOUN[index % len(TRAIN_EN_NOUN)]
        forms = (f"{verb} the {noun} {nearest} to {reference} building.",
                 f"{verb} the {noun} that is {nearest} to {reference} building.",
                 f"{verb} the {noun} {nearest} from {reference} building.")
        prompt = forms[index % len(forms)]
        prompt = prompt[0].upper() + prompt[1:]
    return {"prompt": prompt, "program": program, "language": language,
            "source": "task7b_augmentation", "template_index": index}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    # ---------------- freeze the evaluation packs first
    minimal = build_minimal_pairs()
    stress = build_stress()
    fixed24 = json.loads(TASK7A_FIXED24.read_text(encoding="utf-8"))
    fixed24_prompts = [entry["text"] for entry in fixed24["prompts"]]
    fixed24_normalized = {normalize_prompt(text) for text in fixed24_prompts}

    assert len(minimal) == 96, len(minimal)
    assert len(stress) == 192, len(stress)
    for row in minimal:
        assert normalize_prompt(row["prompt"]) not in fixed24_normalized, row
    for row in stress:
        assert normalize_prompt(row["prompt"]) not in fixed24_normalized, row

    minimal_by_group = Counter(row["contrast_group"] for row in minimal)
    write_json(OUT_MINIMAL, {
        "_doc": ("Task 7B section 5. Frozen 96-prompt compositional minimal-pair pack (four contrast "
                 "families: M1 L2 direction vs L3 direction+nearest, M2 L1 largest vs L2 direction, "
                 "M3 nearest-only vs direction+nearest, M4 largest vs smallest family in L2). Frozen "
                 "before training; the lexical pool is reserved from the training grammar and no Task 7A "
                 "fixed24 string is reproduced."),
        "task": "7B", "stage": "C-minimal-pairs", "prompt_count": len(minimal),
        "by_contrast_group": dict(minimal_by_group),
        "by_language": dict(Counter(row["language"] for row in minimal)),
        "programs": sorted({row["expected_program"] for row in minimal}),
        "rows": minimal,
    })
    stress_counts = Counter(row["expected_program"] for row in stress)
    stress_l3 = {program: sum(1 for row in stress if row["expected_program"] == program)
                 for program in L3_PROGRAMS}
    stress_l3_language = {f"{program}|{language}": sum(
        1 for row in stress if row["expected_program"] == program and row["language"] == language)
        for program in L3_PROGRAMS for language in ("zh", "en")}
    extremes = ("leftmost", "rightmost", "topmost", "bottommost")
    coverage = {
        "l3_classes_total": sum(stress_l3.values()),
        "l2_direction_total": sum(stress_counts.get(program, 0) for program in L2_LARGEST),
        "largest_to_nearest": stress_counts.get("largest_to_nearest", 0),
        "smallest_to_nearest": stress_counts.get("smallest_to_nearest", 0),
        "largest": stress_counts.get("largest", 0),
        "smallest": stress_counts.get("smallest", 0),
        "extreme_classes_total": sum(stress_counts.get(program, 0) for program in extremes),
    }
    write_json(OUT_STRESS, {
        "_doc": ("Task 7B section 6. Frozen 192-prompt held-out L3 stress pack: 4 L3 classes x 24 "
                 "(12 Chinese + 12 English, short/medium/long phrasing, nearest/closest and "
                 "最近/距离最近/最靠近 variants, word-order variants), 4 L2 direction classes x 12, "
                 "largest_to_nearest 16, smallest_to_nearest 8, largest 8, smallest 8 and the four "
                 "extreme classes 8 in total. No Task 7A fixed24 string is reproduced."),
        "task": "7B", "stage": "C-stress-v1", "prompt_count": len(stress),
        "coverage": coverage, "expected_total": 192,
        "by_program": {program: stress_counts.get(program, 0) for program in EXPECTED_PROGRAM_IDS
                       if stress_counts.get(program, 0)},
        "l3_by_language": stress_l3_language,
        "by_language": dict(Counter(row["language"] for row in stress)),
        "by_form": dict(Counter(row["form"] for row in stress)),
        "rows": stress,
    })

    # ---------------- pre-training evaluation manifest
    def pack_hash(path: Path) -> str:
        return sha256_file(path) if path.is_file() else ""

    manifest = {
        "_doc": ("Task 7B section 6. Pre-training hash manifest of every evaluation source. Written "
                 "before any optimization so no evaluation prompt can be adjusted after seeing results."),
        "task": "7B", "stage": "C-eval-manifest", "frozen_before_training": True,
        "sources": {
            "minimal_pairs": {"path": str(OUT_MINIMAL), "prompts": len(minimal),
                              "sha256": pack_hash(OUT_MINIMAL)},
            "stress_v1": {"path": str(OUT_STRESS), "prompts": len(stress),
                          "sha256": pack_hash(OUT_STRESS)},
            "task7a_fixed24": {"path": str(TASK7A_FIXED24), "prompts": len(fixed24_prompts),
                               "sha256": pack_hash(TASK7A_FIXED24)},
            "z_minival240": {"path": str(REPO_ROOT / "artifacts" / "task6z" / "packs"
                                         / "z_mini_val_240.json"),
                             "prompts": 240,
                             "sha256": pack_hash(REPO_ROOT / "artifacts" / "task6z" / "packs"
                                                 / "z_mini_val_240.json")},
            "z_paired_val20": {"path": str(REPO_ROOT / "artifacts" / "task6z" / "packs"
                                           / "z_paired_val20.json"),
                               "prompts": 40,
                               "sha256": pack_hash(REPO_ROOT / "artifacts" / "task6z" / "packs"
                                                   / "z_paired_val20.json")},
            "v02_val": {"path": str(DATA / "val.jsonl"), "sha256": pack_hash(DATA / "val.jsonl")},
            "v02_train": {"path": str(DATA / "train.jsonl"), "sha256": pack_hash(DATA / "train.jsonl")},
        },
        "test_split_used": False,
    }
    write_json(OUT_MANIFEST, manifest)
    print(f"[7b.data] minimal pairs {len(minimal)} {dict(minimal_by_group)} | stress {len(stress)} "
          f"coverage {coverage}", flush=True)

    # ---------------- Part D: exclusion set, cleaned train, augmentation, leakage
    exclusion_exact: dict[str, str] = {}
    exclusion_normalized: dict[str, str] = {}

    def register(text: str, source: str) -> None:
        exclusion_exact.setdefault(text, source)
        exclusion_normalized.setdefault(normalize_prompt(text), source)

    train_records = []
    with (DATA / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            train_records.append(record)
    val_records = []
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            val_records.append(json.loads(line))
    for record in val_records:
        for key in ("instruction_en", "instruction_zh"):
            if record.get(key):
                register(str(record[key]), f"v02_val:{record['sample_id']}")
    for name, path in (("z_minival240", REPO_ROOT / "artifacts" / "task6z" / "packs"
                        / "z_mini_val_240.json"),
                       ("z_paired_val20", REPO_ROOT / "artifacts" / "task6z" / "packs"
                        / "z_paired_val20.json")):
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            register(str(record["sample_id"]), f"{name}:sample_id")
    for entry in fixed24["prompts"]:
        register(str(entry["text"]), f"task7a_fixed24:{entry['id']}")
    for row in minimal:
        register(row["prompt"], f"task7b_minimal:{row['id']}")
    for row in stress:
        register(row["prompt"], f"task7b_stress:{row['id']}")

    cleaned_rows, dropped = [], []
    for record in train_records:
        prompts = [str(record.get("instruction_en") or ""), str(record.get("instruction_zh") or "")]
        if any(prompt in exclusion_exact for prompt in prompts if prompt) or \
                any(normalize_prompt(prompt) in exclusion_normalized for prompt in prompts if prompt):
            dropped.append({"sample_id": record["sample_id"],
                            "reason": "exact_or_normalized_eval_overlap"})
            continue
        for language, key in (("en", "instruction_en"), ("zh", "instruction_zh")):
            if record.get(key):
                cleaned_rows.append({"prompt": str(record[key]), "program": str(record["query_type"]),
                                     "language": language, "source": "v02_train",
                                     "sample_id": str(record["sample_id"])})
    augmentations = build_augmentations()
    assert len(augmentations) == AUGMENT_TOTAL == 4800, len(augmentations)

    dupes = Counter(normalize_prompt(row["prompt"]) for row in augmentations)
    repeated = {key: value for key, value in dupes.items() if value > 1}
    augmentation_exact = {row["prompt"] for row in augmentations}
    augmentation_normalized = {normalize_prompt(row["prompt"]) for row in augmentations}
    augmentation_program_conflicts = sorted(
        {normalize_prompt(row["prompt"]) for row in augmentations
         if len({entry["program"] for entry in augmentations
                 if normalize_prompt(entry["prompt"]) == normalize_prompt(row["prompt"])}) > 1})

    exact_overlap = sorted(augmentation_exact & set(exclusion_exact))
    normalized_overlap = sorted(augmentation_normalized & set(exclusion_normalized))
    train_exact_overlap = sorted({row["prompt"] for row in cleaned_rows} & set(exclusion_exact))
    train_normalized_overlap = sorted({normalize_prompt(row["prompt"]) for row in cleaned_rows}
                                      & set(exclusion_normalized))

    AUGMENT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_path = AUGMENT_ROOT / "parser_train_combined.jsonl"
    with combined_path.open("w", encoding="utf-8") as handle:
        for row in cleaned_rows + augmentations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    spec = {
        "_doc": ("Task 7B sections 7-10. Training-data specification: BuildSpatialReason v0.2 train "
                 "instruction text plus exactly 4,800 deterministic train-only compositional prompts with "
                 "the section-9 allocation. The training grammar uses only the section-10 TRAIN lexical "
                 "pools; the evaluation lexical forms (Task 7A fixed24 phrases and the Task 7B minimal "
                 "pair/stress pools) are reserved and never emitted."),
        "task": "7B", "stage": "D-training-data", "seed": SEED,
        "sources": ["v02_train", "task7b_augmentation"],
        "allocation": AUGMENT_ALLOCATION, "augmentation_total": AUGMENT_TOTAL,
        "cleaned_train_rows": len(cleaned_rows), "dropped_train_records": len(dropped),
        "combined_rows": len(cleaned_rows) + len(augmentations),
        "combined_path": str(combined_path), "gitignored": True,
        "train_lexical_pool": {
            "zh_largest": list(TRAIN_ZH_LARGEST), "zh_smallest": list(TRAIN_ZH_SMALLEST),
            "zh_direction": {key: list(value) for key, value in TRAIN_ZH_DIRECTION.items()},
            "zh_nearest": list(TRAIN_ZH_NEAREST), "zh_noun": list(TRAIN_ZH_NOUN),
            "en_largest": list(TRAIN_EN_LARGEST), "en_smallest": list(TRAIN_EN_SMALLEST),
            "en_direction": {key: list(value) for key, value in TRAIN_EN_DIRECTION.items()},
            "en_nearest": list(TRAIN_EN_NEAREST), "en_noun": list(TRAIN_EN_NOUN),
        },
        "reserved_for_evaluation": {
            "zh": ["最大建筑左边最近", "最大建筑右边最近", "最大建筑上方最近", "最大建筑下方最近"],
            "en": ["closest building to the left", "closest building to the right",
                   "nearest building above", "closest building below"],
            "eval_zh_reference": EVAL_ZH_REFERENCE, "eval_en_reference": EVAL_EN_REFERENCE,
            "eval_zh_direction": dict(EVAL_ZH_DIRECTION), "eval_en_direction": dict(EVAL_EN_DIRECTION),
        },
        "augmentation_hygiene": {
            "unique_normalized_prompts": len(dupes), "repeated_normalized_prompts": len(repeated),
            "normalized_program_conflicts": len(augmentation_program_conflicts),
            "duplicate_examples": list(repeated)[:5],
            "conflict_examples": augmentation_program_conflicts[:5],
        },
        "task7a_fixed24_used_in_training": False,
        "val_or_test_used_for_optimization": False,
    }
    write_json(OUT_SPEC, spec)

    leakage = {
        "_doc": ("Task 7B section 11. Pre-training leakage audit: exact and normalized overlap between "
                 "the training text (cleaned v0.2 train plus every Task 7B augmentation) and every "
                 "evaluation source must both be zero."),
        "task": "7B", "stage": "D-leakage-audit",
        "normalization": ("unicode strip, lowercase English, collapse whitespace, normalize common "
                          "punctuation, strip terminal punctuation"),
        "evaluation_sources": {
            "v02_val_records": len(val_records),
            "z_minival240": 240, "z_paired_val20_members": 40,
            "task7a_fixed24": len(fixed24_prompts),
            "task7b_minimal_pairs": len(minimal), "task7b_stress_v1": len(stress),
        },
        "exclusion_prompts": {"exact": len(exclusion_exact), "normalized": len(exclusion_normalized)},
        "cleaned_train": {"rows": len(cleaned_rows), "dropped_records": len(dropped),
                          "dropped": dropped[:20], "dropped_total": len(dropped)},
        "augmentation": {"rows": len(augmentations), "exact_overlap": len(exact_overlap),
                         "normalized_overlap": len(normalized_overlap),
                         "exact_overlap_examples": exact_overlap[:5],
                         "normalized_overlap_examples": normalized_overlap[:5]},
        "cleaned_train_overlap": {"exact": len(train_exact_overlap),
                                  "normalized": len(train_normalized_overlap),
                                  "examples": (train_exact_overlap + train_normalized_overlap)[:5]},
        "task7a_fixed24_in_training": sorted({row["prompt"] for row in cleaned_rows + augmentations}
                                             & set(fixed24_prompts))[:5],
        "verdict": "LEAKAGE_FREE" if not (exact_overlap or normalized_overlap
                                          or train_exact_overlap or train_normalized_overlap) else
                   "INVALID_EXPERIMENT",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_LEAKAGE, leakage)
    print(f"[7b.data] cleaned train {len(cleaned_rows)} (dropped {len(dropped)}), augmentations "
          f"{len(augmentations)}, exact overlap {len(exact_overlap)}, normalized overlap "
          f"{len(normalized_overlap)} -> {leakage['verdict']}", flush=True)
    return 0 if leakage["verdict"] == "LEAKAGE_FREE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
