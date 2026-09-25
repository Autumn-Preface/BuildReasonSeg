"""Bilingual instruction and reasoning templates for BuildSpatialReason.

Every template pair carries the SAME ``template_id`` in Chinese and English, so
``instruction_zh`` and ``instruction_en`` are guaranteed to express the same
semantic query. A test asserts this parity.

Template selection is deterministic
-----------------------------------
A template is chosen by ``SHA256(semantic_key + "|" + template_id)`` hashed to a
bucket. Python's builtin ``hash()`` is never used: it is salted per process and
would make regeneration non-reproducible.

Wording rules
-------------
- "建筑区域" / "building region" is the preferred Chinese/English noun. Both
  refer to a **building connected component** in the source annotation, not to a
  guaranteed real-world physical building. See
  ``docs/build_spatial_reason_v0.1.md``.
- No architecture-specific tokens appear anywhere. ``[SEG]`` and ``[REF]`` belong
  to a future MLLM conversation format, not to dataset semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

TEMPLATE_VERSION = "v0.1"


@dataclass(frozen=True)
class TemplatePair:
    """One semantic template in both languages."""

    template_id: str

    zh: str
    en: str


# --------------------------------------------------------------------------
# Level 1 -- direct spatial grounding
# --------------------------------------------------------------------------
#
# `{noun}` is filled with the component count when relevant, otherwise the
# templates are self-contained.

LEVEL1_TEMPLATES: dict[str, tuple[TemplatePair, ...]] = {
    "leftmost": (
        TemplatePair("l1.leftmost.a", "分割图像中最左侧的建筑区域。", "Segment the leftmost building region in the image."),
        TemplatePair("l1.leftmost.b", "请标出画面中最左边的那栋建筑区域。", "Please outline the building region furthest to the left."),
        TemplatePair("l1.leftmost.c", "找出图中位置最靠左的建筑区域并分割。", "Locate and segment the building region positioned furthest left."),
    ),
    "rightmost": (
        TemplatePair("l1.rightmost.a", "分割图像中最右侧的建筑区域。", "Segment the rightmost building region in the image."),
        TemplatePair("l1.rightmost.b", "请标出画面中最右边的那栋建筑区域。", "Please outline the building region furthest to the right."),
        TemplatePair("l1.rightmost.c", "找出图中位置最靠右的建筑区域并分割。", "Locate and segment the building region positioned furthest right."),
    ),
    "topmost": (
        TemplatePair("l1.topmost.a", "分割图像中最上方的建筑区域。", "Segment the topmost building region in the image."),
        TemplatePair("l1.topmost.b", "请标出画面中最靠上的那栋建筑区域。", "Please outline the building region nearest the top of the image."),
        TemplatePair("l1.topmost.c", "找出图中位置最高的建筑区域并分割。", "Locate and segment the building region positioned highest."),
    ),
    "bottommost": (
        TemplatePair("l1.bottommost.a", "分割图像中最下方的建筑区域。", "Segment the bottommost building region in the image."),
        TemplatePair("l1.bottommost.b", "请标出画面中最靠下的那栋建筑区域。", "Please outline the building region nearest the bottom of the image."),
        TemplatePair("l1.bottommost.c", "找出图中位置最低的建筑区域并分割。", "Locate and segment the building region positioned lowest."),
    ),
    "largest": (
        TemplatePair("l1.largest.a", "分割图像中面积最大的建筑区域。", "Segment the largest building region in the image."),
        TemplatePair("l1.largest.b", "请标出画面中占地面积最大的建筑区域。", "Please outline the building region with the greatest area."),
        TemplatePair("l1.largest.c", "找出图中面积最大的建筑区域并分割。", "Locate and segment the building region with the largest area."),
    ),
    "smallest": (
        TemplatePair("l1.smallest.a", "分割图像中面积最小的建筑区域。", "Segment the smallest building region in the image."),
        TemplatePair("l1.smallest.b", "请标出画面中占地面积最小的建筑区域。", "Please outline the building region with the smallest area."),
        TemplatePair("l1.smallest.c", "找出图中面积最小的建筑区域并分割。", "Locate and segment the building region with the smallest area."),
    ),
}


# --------------------------------------------------------------------------
# Level 2 -- reference-based reasoning
# --------------------------------------------------------------------------

LEVEL2_NEAREST_TEMPLATES: dict[str, tuple[TemplatePair, ...]] = {
    "largest": (
        TemplatePair(
            "l2.nearest.largest.a",
            "找到面积最大的建筑区域，并分割距离它最近的另一建筑区域。",
            "Find the largest building region and segment the other building region nearest to it.",
        ),
        TemplatePair(
            "l2.nearest.largest.b",
            "以面积最大的建筑区域为参照，分割与它边界距离最近的另一建筑区域。",
            "Using the largest building region as reference, segment the other building region with the smallest boundary distance to it.",
        ),
        TemplatePair(
            "l2.nearest.largest.c",
            "先确定图中面积最大的建筑区域，再分割离它最近的另一建筑区域。",
            "First determine the largest building region, then segment the other building region closest to it.",
        ),
    ),
    "smallest": (
        TemplatePair(
            "l2.nearest.smallest.a",
            "找到面积最小的建筑区域，并分割距离它最近的另一建筑区域。",
            "Find the smallest building region and segment the other building region nearest to it.",
        ),
        TemplatePair(
            "l2.nearest.smallest.b",
            "以面积最小的建筑区域为参照，分割与它边界距离最近的另一建筑区域。",
            "Using the smallest building region as reference, segment the other building region with the smallest boundary distance to it.",
        ),
        TemplatePair(
            "l2.nearest.smallest.c",
            "先确定图中面积最小的建筑区域，再分割离它最近的另一建筑区域。",
            "First determine the smallest building region, then segment the other building region closest to it.",
        ),
    ),
}

#: Directional Level-2 wording. ``{ref}`` is the reference phrase, e.g.
#: "面积最大的建筑区域" / "the largest building region".
LEVEL2_DIRECTION_TEMPLATES: dict[str, tuple[TemplatePair, ...]] = {
    "right_of": (
        TemplatePair(
            "l2.dir.right_of.a",
            "分割位于{ref}右侧的建筑区域。",
            "Segment the building region to the right of {ref}.",
        ),
        TemplatePair(
            "l2.dir.right_of.b",
            "请标出{ref}右侧的那栋建筑区域。",
            "Please outline the building region on the right side of {ref}.",
        ),
        TemplatePair(
            "l2.dir.right_of.c",
            "找出{ref}右边唯一的建筑区域并分割。",
            "Locate and segment the single building region to the right of {ref}.",
        ),
    ),
    "left_of": (
        TemplatePair(
            "l2.dir.left_of.a",
            "分割位于{ref}左侧的建筑区域。",
            "Segment the building region to the left of {ref}.",
        ),
        TemplatePair(
            "l2.dir.left_of.b",
            "请标出{ref}左侧的那栋建筑区域。",
            "Please outline the building region on the left side of {ref}.",
        ),
        TemplatePair(
            "l2.dir.left_of.c",
            "找出{ref}左边唯一的建筑区域并分割。",
            "Locate and segment the single building region to the left of {ref}.",
        ),
    ),
    "above": (
        TemplatePair(
            "l2.dir.above.a",
            "分割位于{ref}上方的建筑区域。",
            "Segment the building region above {ref}.",
        ),
        TemplatePair(
            "l2.dir.above.b",
            "请标出{ref}上方的那栋建筑区域。",
            "Please outline the building region on the upper side of {ref}.",
        ),
        TemplatePair(
            "l2.dir.above.c",
            "找出{ref}上方唯一的建筑区域并分割。",
            "Locate and segment the single building region above {ref}.",
        ),
    ),
    "below": (
        TemplatePair(
            "l2.dir.below.a",
            "分割位于{ref}下方的建筑区域。",
            "Segment the building region below {ref}.",
        ),
        TemplatePair(
            "l2.dir.below.b",
            "请标出{ref}下方的那栋建筑区域。",
            "Please outline the building region on the lower side of {ref}.",
        ),
        TemplatePair(
            "l2.dir.below.c",
            "找出{ref}下方唯一的建筑区域并分割。",
            "Locate and segment the single building region below {ref}.",
        ),
    ),
}

#: Reference phrases used inside Level-2 / Level-3 instructions.
REFERENCE_PHRASE: dict[str, dict[str, str]] = {
    "largest": {"zh": "面积最大的建筑区域", "en": "the largest building region"},
    "smallest": {"zh": "面积最小的建筑区域", "en": "the smallest building region"},
    "leftmost": {"zh": "最左侧的建筑区域", "en": "the leftmost building region"},
    "rightmost": {"zh": "最右侧的建筑区域", "en": "the rightmost building region"},
    "topmost": {"zh": "最上方的建筑区域", "en": "the topmost building region"},
    "bottommost": {"zh": "最下方的建筑区域", "en": "the bottommost building region"},
}

#: Direction word used inside Level-3 instructions.
DIRECTION_PHRASE: dict[str, dict[str, str]] = {
    "right_of": {"zh": "右侧", "en": "right"},
    "left_of": {"zh": "左侧", "en": "left"},
    "above": {"zh": "上方", "en": "above"},
    "below": {"zh": "下方", "en": "below"},
}

#: Prepositional form of the direction, used where the template places the
#: direction AFTER the reference ("its {dir}"). A bare adverb would produce
#: ungrammatical output such as "to its below", so Level-3 templates use this
#: set instead.
DIRECTION_PREPOSITION: dict[str, dict[str, str]] = {
    "right_of": {"zh": "右侧", "en": "on its right side"},
    "left_of": {"zh": "左侧", "en": "on its left side"},
    "above": {"zh": "上方", "en": "above it"},
    "below": {"zh": "下方", "en": "below it"},
}

#: Role-based phrasing for ID-FREE reasoning prose.
#:
#: ``OPERATION_PHRASE`` above is used where the sentence reads
#: "X is component N" (id-bearing, v0.1 only). These phrases say
#: "identify the X" instead, so the prose never exposes an annotation id.
ROLE_PHRASE: dict[str, dict[str, str]] = {
    "argmax_area": {"zh": "面积最大的建筑区域", "en": "the largest building region"},
    "argmin_area": {"zh": "面积最小的建筑区域", "en": "the smallest building region"},
    "argmin_centroid_x": {"zh": "最左侧的建筑区域", "en": "the leftmost building region"},
    "argmax_centroid_x": {"zh": "最右侧的建筑区域", "en": "the rightmost building region"},
    "argmin_centroid_y": {"zh": "最上方的建筑区域", "en": "the topmost building region"},
    "argmax_centroid_y": {"zh": "最下方的建筑区域", "en": "the bottommost building region"},
}

#: Direction phrase that reads correctly AFTER the reference region is already
#: established, used in ID-FREE reasoning prose. The value is a complete
#: adverbial, so the sentence stays grammatical in every direction.
DIRECTION_RELATIVE_TO_REFERENCE: dict[str, dict[str, str]] = {
    "right_of": {"zh": "位于参考区域右侧", "en": "on the right side of the reference region"},
    "left_of": {"zh": "位于参考区域左侧", "en": "on the left side of the reference region"},
    "above": {"zh": "位于参考区域上方", "en": "above the reference region"},
    "below": {"zh": "位于参考区域下方", "en": "below the reference region"},
}

#: Full prepositional phrase, used where the direction is inserted BEFORE the
#: reference ("the candidates {dir} of component N"). The value already contains
#: "of", so the template must not add another one.
DIRECTION_RELATIVE_OF: dict[str, dict[str, str]] = {
    "right_of": {"zh": "右侧", "en": "to the right of"},
    "left_of": {"zh": "左侧", "en": "to the left of"},
    "above": {"zh": "上方", "en": "above"},
    "below": {"zh": "下方", "en": "below"},
}


# --------------------------------------------------------------------------
# Level 3 -- multi-hop: reference -> direction -> nearest within filter
# --------------------------------------------------------------------------

LEVEL3_TEMPLATES: tuple[TemplatePair, ...] = (
    TemplatePair(
        "l3.chain.a",
        "先找到{ref}，在位于它{dir}的建筑区域中，分割距离它最近的一个。",
        "First locate {ref}. Among the building regions {dir}, segment the one nearest to it.",
    ),
    TemplatePair(
        "l3.chain.b",
        "以{ref}为参照，在它{dir}的所有建筑区域里，分割与它边界距离最小的那个。",
        "Using {ref} as reference, among all building regions {dir}, segment the one with the smallest boundary distance to it.",
    ),
    TemplatePair(
        "l3.chain.c",
        "先确定{ref}，再从其{dir}的建筑区域中选出与它最近的一个并分割。",
        "First determine {ref}, then from the building regions {dir} select and segment the closest one.",
    ),
)


# --------------------------------------------------------------------------
# Deterministic selection
# --------------------------------------------------------------------------


def select_template(
    templates: tuple[TemplatePair, ...],
    semantic_key: str,
) -> TemplatePair:
    """Pick one template deterministically from a family.

    The bucket is ``SHA256(semantic_key + "|" + template_id)`` modulo the family
    size. Including the ``template_id`` in the digest keeps the choice spread
    across the family instead of always resolving to the first member.

    Python's builtin ``hash()`` is deliberately not used: it is randomised per
    process and would break reproducibility.
    """

    import hashlib

    if not templates:
        raise ValueError("empty template family")

    digest = hashlib.sha256(f"{semantic_key}|{TEMPLATE_VERSION}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % len(templates)
    return templates[bucket]


# --------------------------------------------------------------------------
# Natural-language reasoning (generated FROM structured steps)
# --------------------------------------------------------------------------


def join_components_zh(ids: list[int]) -> str:
    if not ids:
        return "（无）"
    if len(ids) == 1:
        return f"component {ids[0]}"
    return "、".join(f"component {i}" for i in ids[:-1]) + f" 和 component {ids[-1]}"

def join_components_en(ids: list[int]) -> str:
    if not ids:
        return "(none)"
    if len(ids) == 1:
        return f"component {ids[0]}"
    if len(ids) == 2:
        return f"component {ids[0]} and component {ids[1]}"
    return ", ".join(f"component {i}" for i in ids[:-1]) + f", and component {ids[-1]}"


#: Human-readable phrasing for each structured operation.
#:
#: `zh` is a bare noun phrase used after "是 ...", so it needs no article.
#: `en` is a bare noun phrase used after "is ...", likewise.
OPERATION_PHRASE: dict[str, dict[str, str]] = {
    "argmax_area": {"zh": "面积最大的建筑区域", "en": "largest building region"},
    "argmin_area": {"zh": "面积最小的建筑区域", "en": "smallest building region"},
    "argmin_centroid_x": {"zh": "最左侧的建筑区域", "en": "leftmost building region"},
    "argmax_centroid_x": {"zh": "最右侧的建筑区域", "en": "rightmost building region"},
    "argmin_centroid_y": {"zh": "最上方的建筑区域", "en": "topmost building region"},
    "argmax_centroid_y": {"zh": "最下方的建筑区域", "en": "bottommost building region"},
}
