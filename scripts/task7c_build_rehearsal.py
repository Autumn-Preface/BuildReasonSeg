"""Task 7C Parts C-G — reused-evaluation manifest, 8,000-prompt rehearsal set and leakage audit.

Part C: hash the already-frozen evaluation sources without regenerating anything.
Part E-G: generate exactly 8,000 **unique normalized** rehearsal prompts covering all 20 canonical classes
(four L3 classes 800 each = 400 Chinese + 400 English; the other sixteen classes 300 each = 150 + 150),
using only the section-10/11 TRAIN lexical pools, then require zero exact and zero normalized overlap with
the union of every evaluation prompt and zero normalized duplicates inside the rehearsal set.

**No original BuildSpatialReason v0.2 train instruction string is used**: v0.2 train text is read only to
verify the class-id schema.

    python scripts/task7c_build_rehearsal.py
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
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
REHEARSAL_ROOT = REPO_ROOT / "artifacts" / "task7c" / "parser_rehearsal"
OUT_REUSED = EVAL / "task7c_reused_eval_manifest.json"
OUT_SPEC = EVAL / "task7c_rehearsal_spec.json"
OUT_AUDIT = EVAL / "task7c_training_data_audit.json"
TASK7B_MINIMAL = EVAL / "task7b_compositional_minimal_pairs.json"
TASK7B_STRESS = EVAL / "task7b_l3_stress_v1.json"
TASK7A_FIXED24 = EVAL / "task7a_l3_paraphrase_pack.json"
TASK7B_SCOPE = EVAL / "task7b_scope_safety.json"
SEED = 20261001

L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")
L2_LARGEST = ("largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below")
L2_SMALLEST = ("smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below")
EXTREMES = ("leftmost", "rightmost", "topmost", "bottommost")
OTHER_CLASSES = ("leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest",
                 "largest_to_nearest", "smallest_to_nearest", *L2_LARGEST, *L2_SMALLEST)
ALL_CLASSES = (*OTHER_CLASSES, *L3_PROGRAMS)
DIRECTION_SUFFIX = {"left_of": "left", "right_of": "right", "above": "above", "below": "below"}
L3_BY_DIRECTION = {direction: f"largest_to_{suffix}_to_nearest"
                   for suffix, direction in DIRECTION_SUFFIX.items()}
L2_LARGEST_BY_DIRECTION = {direction: f"largest_to_{suffix}"
                           for suffix, direction in DIRECTION_SUFFIX.items()}
L2_SMALLEST_BY_DIRECTION = {direction: f"smallest_to_{suffix}"
                            for suffix, direction in DIRECTION_SUFFIX.items()}
ALLOCATION = {**{program: 800 for program in L3_PROGRAMS},
              **{program: 300 for program in OTHER_CLASSES}}
TOTAL = sum(ALLOCATION.values())

# ---------------------------------------------------------------- section 10/11 TRAIN pools
ZH_VERB = ("请分割", "请标注", "请提取", "标出", "请勾勒出", "圈出")
ZH_VERB_PLAIN = ("分割", "标注", "提取", "标出", "勾勒出", "圈出")
ZH_PREFIX = ("请", "请在图中", "请在该影像中", "请在整幅图上", "请在画面中", "请直接")
ZH_LARGEST_REF = ("面积最大的建筑", "占地最大的建筑")
ZH_SMALLEST_REF = ("面积最小的建筑", "占地最小的建筑")
ZH_DIRECTION = {
    "left_of": ("位于其左侧的", "处在其左方的"),
    "right_of": ("位于其右侧的", "处在其右方的"),
    "above": ("位于其上方的", "处在其上侧的"),
    "below": ("位于其下方的", "处在其下侧的"),
}
ZH_NEAREST = ("其中距离最近的", "其中与参考建筑间距最小的", "其中最接近参考建筑的")
ZH_NEAREST_ONLY = ("与其距离最近的", "与其间距最小的")
ZH_NOUN = ("建筑物", "楼房", "地块", "建筑体", "房屋", "建筑目标")
ZH_EXTREME = {"leftmost": "最靠左的", "rightmost": "最靠右的", "topmost": "位置最高的",
              "bottommost": "位置最低的"}

EN_VERB = ("Segment", "Label", "Extract", "Mark", "Outline", "Delineate")
EN_PREFIX = ("", "In the tile, ", "In this image, ", "On this map, ", "Now ", "Please ")
EN_LARGEST_REF = ("the building with the largest footprint", "the greatest-area building")
EN_SMALLEST_REF = ("the building with the smallest footprint", "the least-area building")
EN_DIRECTION = {
    "left_of": ("located on its left side", "situated to its left"),
    "right_of": ("located on its right side", "situated to its right"),
    "above": ("located above it", "situated on its upper side"),
    "below": ("located below it", "situated on its lower side"),
}
EN_NEAREST = ("the one with the minimum separation", "the one nearest to the reference",
              "the closest one among them")
EN_NEAREST_ONLY = ("the building with the minimum separation from it",
                   "the building nearest to it")
EN_NOUN = ("building", "structure", "parcel", "block", "building region", "target building")
EN_EXTREME = {"leftmost": ("the farthest-left building", "the leftmost building"),
              "rightmost": ("the farthest-right building", "the rightmost building"),
              "topmost": ("the uppermost building", "the topmost building"),
              "bottommost": ("the lowermost building", "the bottommost building")}
EN_LARGEST_EXTREME = "the building with the greatest footprint"
EN_SMALLEST_EXTREME = "the building with the least footprint"


def _english(verb: str, prefix: str, rest: str) -> str:
    """`prefix + verb + rest` with the verb lowercased after a prefix ending the clause."""

    if prefix in ("", "Now "):
        return f"{prefix}{verb} {rest}"
    lowered = verb if verb.startswith(tuple("AEOU")) is False and False else verb
    if prefix == "Please ":
        return f"{prefix}{verb.lower()} {rest}"
    return f"{prefix}{verb.lower()} {rest}"


def normalize_prompt(prompt: str) -> str:
    text = prompt.strip().lower()
    text = re.sub(r"[\s\u3000]+", "", text)
    for source, target in (("，", ","), ("。", "."), ("？", "?"), ("！", "!"), ("、", ","),
                           ("：", ":"), ("；", ";"), ("（", "("), ("）", ")"), ("“", '"'),
                           ("”", '"'), ("‘", "'"), ("’", "'"), ("　", "")):
        text = text.replace(source, target)
    return re.sub(r"[.,!?;:]+$", "", text)


def _extreme_phrases(program: str, language: str) -> tuple[str, ...]:
    if language == "en":
        return tuple(EN_EXTREME[program])
    return (ZH_EXTREME[program],)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _suffix_of(direction: str) -> str:
    """Map the short direction name to the canonical programme-id suffix used by the pools."""

    return {"left": "left_of", "right": "right_of", "above": "above", "below": "below"}[direction]


def program_to_direction(program: str) -> str:
    """Short direction name for an L2 (`..._to_left_of`) or L3 (`..._to_left_of_to_nearest`) program."""

    stem = program[: -len("_to_nearest")] if program.endswith("_to_nearest") else program
    for suffix, direction in DIRECTION_SUFFIX.items():
        if stem.endswith(f"_{suffix}"):
            return direction
    raise ValueError(program)


# ---------------------------------------------------------------- candidate generation


def zh_candidates(program: str) -> list[str]:
    """Deterministic Chinese candidate pool for one class (order = generation order)."""

    candidates: list[str] = []
    if program in L3_PROGRAMS:
        direction = program_to_direction(program)
        for reference in ZH_LARGEST_REF:
            for directional in ZH_DIRECTION[_suffix_of(direction)]:
                for nearest in ZH_NEAREST:
                    for noun in ZH_NOUN:
                        for verb in ZH_VERB:
                            candidates.extend([
                                f"{verb}{reference}，再分割它{directional}{nearest}{noun}。",
                                f"{verb}{reference}{directional}{noun}，{nearest}那一栋。",
                                f"先确定{reference}，然后{verb}{directional}{nearest}{noun}。",
                                f"{verb}{noun}：参考为{reference}，位置{directional}，{nearest}。",
                                f"在{reference}{directional}范围内，{verb}{nearest}{noun}。",
                            ])
                        for prefix in ZH_PREFIX:
                            for verb in ZH_VERB_PLAIN:
                                candidates.append(
                                    f"{prefix}{verb}{reference}{directional}{nearest}{noun}。")
    elif program in L2_LARGEST or program in L2_SMALLEST:
        family = ZH_LARGEST_REF if program.startswith("largest") else ZH_SMALLEST_REF
        direction = program_to_direction(program)
        for reference in family:
            for directional in ZH_DIRECTION[_suffix_of(direction)]:
                for noun in ZH_NOUN:
                    for verb in ZH_VERB:
                        candidates.extend([
                            f"{verb}{reference}{directional}{noun}。",
                            f"{verb}{reference}{directional}那一栋{noun}。",
                            f"先确定{reference}，然后{verb}{directional}{noun}。",
                            f"{verb}{noun}，其位置{directional}{reference}。",
                        ])
                    for prefix in ZH_PREFIX:
                        for verb in ZH_VERB_PLAIN:
                            candidates.append(f"{prefix}{verb}{reference}{directional}{noun}。")
    elif program in ("largest_to_nearest", "smallest_to_nearest"):
        family = ZH_LARGEST_REF if program.startswith("largest") else ZH_SMALLEST_REF
        for reference in family:
            for nearest in ZH_NEAREST_ONLY:
                for noun in ZH_NOUN:
                    for verb in ZH_VERB:
                        candidates.extend([
                            f"{verb}{reference}{nearest}{noun}。",
                            f"{verb}{reference}附近的{noun}，{nearest}那一栋。",
                            f"先确定{reference}，然后{verb}{nearest}{noun}。",
                        ])
                    for prefix in ZH_PREFIX:
                        for verb in ZH_VERB_PLAIN:
                            candidates.append(f"{prefix}{verb}{reference}{nearest}{noun}。")
    elif program in EXTREMES:
        extreme = ZH_EXTREME[program]
        for noun in ZH_NOUN:
            for verb in ZH_VERB:
                candidates.extend([
                    f"{verb}图中{extreme}{noun}。",
                    f"{verb}{extreme}那一栋{noun}。",
                    f"{verb}整幅影像里{extreme}的{noun}。",
                ])
            for prefix in ZH_PREFIX:
                for verb in ZH_VERB_PLAIN:
                    candidates.extend([
                        f"{prefix}{verb}图中{extreme}{noun}。",
                        f"{prefix}{verb}整幅影像里{extreme}的{noun}。",
                        f"{prefix}{verb}该瓦片中{extreme}的{noun}。",
                    ])
    elif program in ("largest", "smallest"):
        family = ZH_LARGEST_REF if program == "largest" else ZH_SMALLEST_REF
        for reference in family:
            for noun in ZH_NOUN:
                for verb in ZH_VERB:
                    candidates.extend([
                        f"{verb}{reference}。",
                        f"{verb}{reference}所在的{noun}。",
                        f"{verb}图中{reference}对应的{noun}。",
                    ])
                for prefix in ZH_PREFIX:
                    for verb in ZH_VERB_PLAIN:
                        candidates.extend([
                            f"{prefix}{verb}{reference}。",
                            f"{prefix}{verb}图中{reference}所在的{noun}。",
                            f"{prefix}{verb}该影像里{reference}的{noun}。",
                        ])
    else:  # pragma: no cover - allocation table is fixed
        raise ValueError(program)
    return candidates


def en_candidates(program: str) -> list[str]:
    candidates: list[str] = []
    if program in L3_PROGRAMS:
        direction = program_to_direction(program)
        for reference in EN_LARGEST_REF:
            for directional in EN_DIRECTION[_suffix_of(direction)]:
                for nearest in EN_NEAREST:
                    for noun in EN_NOUN:
                        for verb in EN_VERB:
                            candidates.extend([
                                _english(verb, "", f"{reference}, then segment the {noun} "
                                                   f"{directional} it that is {nearest}."),
                                _english(verb, "", f"the {noun} {directional} {reference} that is "
                                                   f"{nearest}."),
                                _english(verb, "", f"the {noun} that lies {directional} {reference} "
                                                   f"and is {nearest}."),
                                _english(verb, "", f"the {nearest} {noun} {directional} {reference}."),
                            ])
                        for prefix in EN_PREFIX:
                            candidates.append(_english("segment", prefix,
                                                       f"the {noun} {directional} {reference} that "
                                                       f"is {nearest}."))
    elif program in L2_LARGEST or program in L2_SMALLEST:
        family = EN_LARGEST_REF if program.startswith("largest") else EN_SMALLEST_REF
        direction = program_to_direction(program)
        for reference in family:
            for directional in EN_DIRECTION[_suffix_of(direction)]:
                for noun in EN_NOUN:
                    for verb in EN_VERB:
                        candidates.extend([
                            _english(verb, "", f"the {noun} {directional} {reference}."),
                            _english(verb, "", f"the {noun} that is {directional} {reference}."),
                            _english(verb, "", f"{reference}, then segment the {noun} {directional} "
                                               f"it."),
                        ])
                    for prefix in EN_PREFIX:
                        candidates.append(_english("segment", prefix,
                                                   f"the {noun} {directional} {reference}."))
    elif program in ("largest_to_nearest", "smallest_to_nearest"):
        family = EN_LARGEST_REF if program.startswith("largest") else EN_SMALLEST_REF
        for reference in family:
            for nearest in EN_NEAREST_ONLY:
                for noun in EN_NOUN:
                    for verb in EN_VERB:
                        candidates.extend([
                            _english(verb, "", f"the {noun} that is {nearest}."),
                            _english(verb, "", f"the {noun} {nearest}."),
                            _english(verb, "", f"{reference}, then segment the {noun} nearest to it."),
                        ])
                    for prefix in EN_PREFIX:
                        candidates.append(_english("segment", prefix,
                                                   f"the {noun} {nearest}."))
    elif program in EXTREMES:
        for extreme in _extreme_phrases(program, "en"):
            for noun in EN_NOUN:
                for verb in EN_VERB:
                    candidates.extend([
                        _english(verb, "", f"{extreme} in the tile."),
                        _english(verb, "", f"the {noun} that is {extreme}."),
                        _english(verb, "", f"the {noun} matching {extreme}."),
                    ])
                for prefix in EN_PREFIX:
                    candidates.extend([
                        _english("segment", prefix, f"{extreme} in the tile."),
                        _english("segment", prefix, f"the {noun} that is {extreme}."),
                        _english("segment", prefix, f"the {noun} matching {extreme}."),
                    ])
    elif program in ("largest", "smallest"):
        phrases = (EN_LARGEST_REF, (EN_LARGEST_EXTREME,)) if program == "largest" \
            else (EN_SMALLEST_REF, (EN_SMALLEST_EXTREME,))
        for group in phrases:
            for phrase in group:
                for noun in EN_NOUN:
                    for verb in EN_VERB:
                        candidates.extend([
                            _english(verb, "", f"{phrase}."),
                            _english(verb, "", f"the {noun} of {phrase}."),
                        ])
                    for prefix in EN_PREFIX:
                        candidates.extend([
                            _english("segment", prefix, f"{phrase}."),
                            _english("segment", prefix, f"the {noun} of {phrase}."),
                        ])
    else:  # pragma: no cover
        raise ValueError(program)
    return candidates


def evaluation_prompts() -> dict[str, list[str]]:
    """Every frozen evaluation prompt, grouped by source (never modified)."""

    sources: dict[str, list[str]] = {}
    val_prompts = []
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for key in ("instruction_en", "instruction_zh"):
                if record.get(key):
                    val_prompts.append(str(record[key]))
    sources["v02_val"] = val_prompts
    instructions = {}
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[str(record["sample_id"])] = record
    for name, pack in (("z_minival240", "z_mini_val_240.json"),
                       ("z_paired_val20", "z_paired_val20.json")):
        records = json.loads((PACK_ROOT / pack).read_text(encoding="utf-8"))["records"]
        sources[name] = [str(instructions[record["sample_id"]]["instruction_en"])
                         for record in records]
    sources["task7a_fixed24"] = [str(entry["text"]) for entry
                                 in json.loads(TASK7A_FIXED24.read_text(encoding="utf-8"))["prompts"]]
    sources["task7b_minimal96"] = [str(row["prompt"]) for row
                                   in json.loads(TASK7B_MINIMAL.read_text(encoding="utf-8"))["rows"]]
    sources["task7b_stress192"] = [str(row["prompt"]) for row
                                   in json.loads(TASK7B_STRESS.read_text(encoding="utf-8"))["rows"]]
    scope = json.loads(TASK7B_SCOPE.read_text(encoding="utf-8"))
    sources["task7b_scope_controls"] = [str(entry["prompt"])
                                        for entry in scope["out_of_scope_controls"]] + \
        [str(entry["prompt"]) for entry in scope["ood_controls"]]
    return sources


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    # ---------------- Part C: reused-evaluation manifest (no regeneration)
    sources = evaluation_prompts()
    manifest = {
        "_doc": ("Task 7C section 5. Hash manifest of every already-frozen evaluation source. Nothing is "
                 "regenerated, edited or extended; these are the exact Task 4/6Z/7A/7B artifacts."),
        "task": "7C", "stage": "C-reused-eval-manifest", "regenerated": False,
        "sources": {name: {"prompts": len(prompts),
                           "sha256_texts": hashlib.sha256(
                               "\n".join(prompts).encode("utf-8")).hexdigest()}
                    for name, prompts in sorted(sources.items())},
        "files": {"task7a_fixed24": {"path": str(TASK7A_FIXED24),
                                     "sha256": sha256_file(TASK7A_FIXED24)},
                  "task7b_minimal96": {"path": str(TASK7B_MINIMAL),
                                       "sha256": sha256_file(TASK7B_MINIMAL)},
                  "task7b_stress192": {"path": str(TASK7B_STRESS),
                                       "sha256": sha256_file(TASK7B_STRESS)},
                  "task7b_scope_controls": {"path": str(TASK7B_SCOPE),
                                            "sha256": sha256_file(TASK7B_SCOPE)},
                  "z_minival240": {"path": str(PACK_ROOT / "z_mini_val_240.json"),
                                   "sha256": sha256_file(PACK_ROOT / "z_mini_val_240.json")},
                  "z_paired_val20": {"path": str(PACK_ROOT / "z_paired_val20.json"),
                                     "sha256": sha256_file(PACK_ROOT / "z_paired_val20.json")},
                  "v02_val": {"path": str(DATA / "val.jsonl"),
                              "sha256": sha256_file(DATA / "val.jsonl")}},
        "test_split_used": False,
    }
    write_json(OUT_REUSED, manifest)

    # schema check only: the class ids come from the canonical v0.2 schema, never from train text
    train_classes = set()
    with (DATA / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            train_classes.add(str(json.loads(line)["query_type"]))
    assert train_classes == set(ALL_CLASSES), sorted(train_classes ^ set(ALL_CLASSES))

    exclusion_exact = {prompt for prompts in sources.values() for prompt in prompts}
    exclusion_normalized = {normalize_prompt(prompt) for prompt in exclusion_exact}

    # ---------------- Part E: the 8,000-prompt rehearsal set
    rows: list[dict] = []
    used_normalized: set[str] = set()
    generator = zh_candidates, en_candidates
    for program in ALL_CLASSES:
        target = ALLOCATION[program]
        for language, per_language in (("zh", target // 2), ("en", target - target // 2)):
            produced = 0
            pool = zh_candidates(program) if language == "zh" else en_candidates(program)
            for prompt in pool:
                if produced >= per_language:
                    break
                text = prompt if language == "zh" else prompt
                normalized = normalize_prompt(text)
                if normalized in exclusion_normalized or normalized in used_normalized:
                    continue
                used_normalized.add(normalized)
                rows.append({"prompt": text, "program": program, "language": language,
                             "source": "task7c_rehearsal"})
                produced += 1
            if produced < per_language:
                raise SystemExit(f"pool exhausted for {program}/{language}: {produced}/{per_language}")
    assert len(rows) == TOTAL == 8000, len(rows)

    REHEARSAL_ROOT.mkdir(parents=True, exist_ok=True)
    rows_path = REHEARSAL_ROOT / "parser_rehearsal_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_class = Counter(row["program"] for row in rows)
    by_class_language = Counter(f"{row['program']}|{row['language']}" for row in rows)
    spec = {
        "_doc": ("Task 7C sections 7-11. Tracked generation spec: exactly 8,000 unique normalized "
                 "all-class rehearsal prompts — the four L3 classes 800 each (400 Chinese + 400 English) "
                 "and the other sixteen classes 300 each (150 + 150) — built only from the section-10/11 "
                 "TRAIN lexical pools. Original BuildSpatialReason v0.2 train instruction strings are "
                 "never used for optimization; the train file was read only for the class-id schema."),
        "task": "7C", "stage": "E-rehearsal-spec", "seed": SEED,
        "total_rows": len(rows), "allocation": ALLOCATION,
        "by_class": {program: by_class.get(program, 0) for program in ALL_CLASSES},
        "by_class_language": {f"{program}|{language}": by_class_language.get(f"{program}|{language}", 0)
                              for program in ALL_CLASSES for language in ("zh", "en")},
        "l3_classes": list(L3_PROGRAMS), "other_classes": list(OTHER_CLASSES),
        "zh_pools": {"verb": list(ZH_VERB), "largest_ref": list(ZH_LARGEST_REF),
                     "smallest_ref": list(ZH_SMALLEST_REF),
                     "direction": {key: list(value) for key, value in ZH_DIRECTION.items()},
                     "nearest": list(ZH_NEAREST), "nearest_only": list(ZH_NEAREST_ONLY),
                     "noun": list(ZH_NOUN), "extreme": dict(ZH_EXTREME)},
        "en_pools": {"verb": list(EN_VERB), "largest_ref": list(EN_LARGEST_REF),
                     "smallest_ref": list(EN_SMALLEST_REF),
                     "direction": {key: list(value) for key, value in EN_DIRECTION.items()},
                     "nearest": list(EN_NEAREST), "nearest_only": list(EN_NEAREST_ONLY),
                     "noun": list(EN_NOUN), "extreme": dict(EN_EXTREME)},
        "structural_stem_sharing": ("L2 and L3 templates share the `verb + reference + direction + noun` "
                                    "stems; the L3 forms insert one of the nearest phrases, so a "
                                    "style cue cannot separate them"),
        "rows_path": str(rows_path), "gitignored": True,
        "original_v02_train_text_used_for_optimization": False,
    }
    write_json(OUT_SPEC, spec)

    # ---------------- Part G: leakage audit
    exact_overlap = sorted({row["prompt"] for row in rows} & exclusion_exact)
    normalized_overlap = sorted(used_normalized & exclusion_normalized
                                if False else {normalize_prompt(row["prompt"]) for row in rows}
                                & exclusion_normalized)
    duplicates = {key: value for key, value in
                  Counter(normalize_prompt(row["prompt"]) for row in rows).items() if value > 1}
    expected_class_language = {f"{program}|{language}":
                               (ALLOCATION[program] // 2 if language == "zh"
                                else ALLOCATION[program] - ALLOCATION[program] // 2)
                               for program in ALL_CLASSES for language in ("zh", "en")}
    counts_ok = all(by_class_language.get(key, 0) == value
                    for key, value in expected_class_language.items())
    audit = {
        "_doc": ("Task 7C section 12. Pre-training leakage and construction audit of the 8,000-prompt "
                 "rehearsal set: exact and normalized overlap with the union of every evaluation prompt "
                 "must both be zero, there may be no normalized duplicate inside the rehearsal set, and "
                 "the per-class and zh/en counts must be exact."),
        "task": "7C", "stage": "G-leakage-audit",
        "normalization": ("unicode strip, lowercase English, collapse whitespace, normalize common "
                          "punctuation, strip terminal punctuation"),
        "evaluation_union": {"sources": len(sources), "exact": len(exclusion_exact),
                             "normalized": len(exclusion_normalized)},
        "rehearsal": {"rows": len(rows), "unique_normalized": len(used_normalized),
                      "exact_overlap": len(exact_overlap),
                      "normalized_overlap": len(normalized_overlap),
                      "normalized_duplicates": len(duplicates),
                      "exact_overlap_examples": exact_overlap[:5],
                      "normalized_overlap_examples": normalized_overlap[:5],
                      "duplicate_examples": list(duplicates)[:5],
                      "per_class_counts_exact": counts_ok,
                      "by_class": {program: by_class.get(program, 0) for program in ALL_CLASSES},
                      "by_class_language": {key: by_class_language.get(key, 0)
                                            for key in expected_class_language}},
        "original_v02_train_text_used_for_optimization": False,
        "test_split_used": False,
        "verdict": "REHEARSAL_CLEAN" if (not exact_overlap and not normalized_overlap
                                         and not duplicates and counts_ok
                                         and len(rows) == 8000) else "INVALID_EXPERIMENT",
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_AUDIT, audit)
    print(f"[7c.data] rehearsal rows {len(rows)} unique {len(used_normalized)} | exact overlap "
          f"{len(exact_overlap)} normalized overlap {len(normalized_overlap)} duplicates "
          f"{len(duplicates)} counts_ok {counts_ok} -> {audit['verdict']}", flush=True)
    return 0 if audit["verdict"] == "REHEARSAL_CLEAN" else 3


if __name__ == "__main__":
    raise SystemExit(main())
