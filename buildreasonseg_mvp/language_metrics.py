"""Language / reasoning metrics for Task 6B.

The dataset's reasoning text is deterministic and template-generated, so a
normalized exact match is a meaningful measurement rather than a proxy: two
different renderings of the same operation chain differ in wording, not in
meaning, and the template set is finite.

Definitions used here
---------------------
``normalise``
    NFKC-normalise, drop all whitespace and punctuation, lowercase. This makes
    the comparison robust to the spacing/punctuation differences that the chat
    template and the tokenizer introduce.
``reasoning_exact_match``
    normalised generated reasoning (everything before ``[SEG]``) equals the
    normalised expected ``reasoning_zh``.
``operation_chain_accuracy``
    the normalised generated reasoning maps to the **expected query type** through
    a lookup built from the dataset's own template renderings. Every rendering of a
    query type encodes the same operation chain, so a hit means the model produced
    the right chain even if it picked a different accepted template.

No external LLM judge is used anywhere.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable, Sequence

SEG = "[SEG]"

_PUNCTUATION = re.compile(r"[\s\u3000,，。.、;；:：!！?？\"'“”‘’()（）\[\]【】<>《》—\-_/\\|~`@#$%^&*+=]+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _PUNCTUATION.sub("", text)
    return text.strip().lower()


def reasoning_part(text: str) -> str:
    """Everything before the first `[SEG]`."""

    index = text.find(SEG)
    return text[:index] if index >= 0 else text


def count_seg(text: str) -> int:
    return text.count(SEG)


def exact_match(generated: str, expected: str) -> bool:
    return bool(normalise(reasoning_part(generated))) and normalise(reasoning_part(generated)) == normalise(expected)


def char_similarity(generated: str, expected: str) -> float:
    a = normalise(reasoning_part(generated))
    b = normalise(expected)
    if not a and not b:
        return 1.0
    return float(SequenceMatcher(None, a, b).ratio())


def build_reasoning_lookup(records: Iterable[dict]) -> dict[str, set[str]]:
    """normalised reasoning -> the query types that legitimately produce it."""

    lookup: dict[str, set[str]] = defaultdict(set)
    for record in records:
        lookup[normalise(record["reasoning_zh"])].add(record["query_type"])
    return dict(lookup)


def operation_chain_correct(generated: str, expected_query_type: str, lookup: dict[str, set[str]]) -> bool:
    """Did the generated reasoning come from the right operation chain?"""

    key = normalise(reasoning_part(generated))
    if not key:
        return False
    return expected_query_type in lookup.get(key, set())


def perplexity(cross_entropy: float) -> float:
    try:
        return float(math.exp(min(cross_entropy, 50.0)))
    except OverflowError:  # pragma: no cover - guarded by the clamp above
        return float("inf")


def summarise_language(entries: Sequence[dict]) -> dict:
    """Aggregate per-record language measurements into report metrics."""

    if not entries:
        return {"count": 0}
    count = len(entries)
    exact = sum(1 for e in entries if e["reasoning_exact_match"])
    chain = sum(1 for e in entries if e["operation_chain_correct"])
    similarities = [e["reasoning_char_similarity"] for e in entries]
    reasoning_acc = [e["reasoning_token_accuracy"] for e in entries if e.get("reasoning_token_accuracy") is not None]
    return {
        "count": count,
        "mean_lm_ce": sum(e["lm_ce"] for e in entries) / count,
        "perplexity": perplexity(sum(e["lm_ce"] for e in entries) / count),
        "mean_assistant_token_accuracy": sum(e["assistant_token_accuracy"] for e in entries) / count,
        "mean_reasoning_token_accuracy": (sum(reasoning_acc) / len(reasoning_acc)) if reasoning_acc else None,
        "seg_token_accuracy": sum(1 for e in entries if e["seg_token_correct"]) / count,
        "reasoning_exact_match": exact / count,
        "operation_chain_accuracy": chain / count,
        "mean_reasoning_char_similarity": sum(similarities) / count,
    }
