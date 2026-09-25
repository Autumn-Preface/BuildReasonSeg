# Spatial Relation Definitions (v1)

The frozen core relation set for BuildSpatialReason, and why each relation is
defined the way it is.

Generated thresholds live in `configs/spatial_relations_v1.yaml`. No numeric
threshold is hard-coded in `spatial_reasoning/*.py`. Measured dataset statistics
are in `docs/relation_statistics_v1.md`.

---

## FROZEN PREDICATE CONVENTION

> **`relation(subject, object)` always means: the SUBJECT satisfies the relation
> with respect to the OBJECT.**

There is exactly one natural-language reading for every directional predicate:

| Call | Unique natural-language reading | Geometric meaning |
|---|---|---|
| `left_of(A, B)` | **A is left of B** | `cx_A < cx_B` |
| `right_of(A, B)` | **A is right of B** | `cx_A > cx_B` |
| `above(A, B)` | **A is above B** | `cy_A < cy_B` |
| `below(A, B)` | **A is below B** | `cy_A > cy_B` |

(Image coordinates: x grows right, **y grows downward**, so a smaller y is
higher in the image.)

This convention is **frozen**. It must not be changed, re-interpreted, or worked
around by:

- the dataset generator (Task 4);
- any reasoning chain built on top of it;
- MLLM training or prompting;
- evaluation code;
- the Spatial Consistency Loss.

An earlier revision of this codebase used the **inverted** convention
(`left_of(X, Y)` meant "Y is left of X"), which contradicted standard predicate
semantics. That revision is superseded. Do not document around it: the code now
implements the convention above, and
`tests/test_relations.py::test_predicate_semantics_left_right_absolute_coordinates`
pins it with absolute coordinates so it cannot silently regress.

### Self-describing evidence

Every `RelationResult.evidence` contains the absolute centroids and the explicit
subject-relative offset, so the sign convention is readable from the evidence
alone without relying on an implicit agreement:

```jsonc
{
  "subject_centroid_x": 100.0,
  "object_centroid_x": 300.0,
  "subject_minus_object_dx": -200.0,     // negative -> subject is on the LEFT
  "relationship": "subject is left of object"
}
```

### Offset helpers

`geometry.subject_to_object_delta(subject, object)` returns
`object - subject`. `geometry.subject_minus_object_delta(subject, object)`
returns `subject - object`, and the relation engine uses **this** one so that a
negative dominant-axis value reads directly as "subject is on the lower side".

The older ambiguous name `centroid_delta` has been removed from the direction
path entirely.

### Predicate templates

`relations.PREDICATE_TEMPLATE` and `relations.describe_relation` hold the
single source of truth mapping each relation to its sentence. The future
instruction generator should use them, so that wording and logic cannot drift
apart. `test_round_trip_semantic_mapping` converts a relation to text and then
checks the geometry predicate that the text denotes — all four relations, both
directions.

---

## Scope: this is tile-relative spatial reasoning

> **v1 reasoning is TILE-RELATIVE.** Every direction and extreme relation is
> interpreted **within a single 512x512 image tile**, not in world coordinates.

This distinction is not cosmetic. 31.72% of components touch the image border,
because the WHU tiles were cropped from larger scenes and buildings are cut by
the crop. That means:

| Kind of relation | Example | Border-truncated component |
|---|---|---|
| **Tile-relative** | `leftmost`, `left_of` | **Eligible.** A building clipped by the left tile edge is still legitimately the leftmost component of this tile. |
| **Physical magnitude** | `largest`, `smallest`, `nearest` | **Not eligible.** A clipped building's area is only a lower bound on its true area, so its size rank is unreliable. |

This is why there is **no single global `geometry_valid` flag applied to every
relation**. Each relation carries its own eligibility rule. Applying one flag
uniformly would either throw away usable tile-relative ground truth or admit
unreliable magnitude ground truth.

---

## Relations NOT implemented in v1

| Relation | Reason |
|---|---|
| `overlap` | **Structurally impossible.** Component maps come from disjoint external contours; measured 0 polygon conflicts across 4038 images. |
| `contain` | Same: no component is inside another. |
| `inside` | Same. |
| `adjacent_to` | **No dependable ground truth.** Mutually touching buildings were already merged into one component upstream, so strict-touch adjacency is indistinguishable from merge. |
| `near` / `far` | **Uncalibrated scale threshold.** Requires a validated scale definition; deferred. |

These are excluded deliberately, not forgotten. `tests/test_relations.py`
asserts they stay out of the frozen set.

---

## Argument order convention

See the **FROZEN PREDICATE CONVENTION** section above. In short:
`relation(subject, object)` means the subject satisfies the relation with respect
to the object, so `left_of(A, B)` reads as "A is left of B".

`tests/test_relations.py` pins this behaviour with absolute coordinates and with
a natural-language round-trip, in both argument orders, so it cannot drift.

---

## Directional relations

`left_of` · `right_of` · `above` · `below`

A directional relation holds only when **all three** conditions hold:

1. **Direction** — the subject-relative dominant-axis offset has the required
   sign (see the frozen convention above).
2. **Axis dominance** — `|dominant| >= alpha * |other|`, so a near-diagonal pair
   is not claimed as a clean left/right or above/below relation.
3. **Minimum normalized margin** — `|dominant_normalized| >= tau`, so the
   separation is not a rounding artifact.

`margin` is reported as the normalized absolute dominant-axis offset;
`axis_ratio` is reported as evidence. Rejections carry an auditable reason:
`wrong_direction`, `axis_not_dominant`, or `margin_too_small`.

### Presets are monotonic

`strict -> medium -> loose` relaxes **both** `alpha` and `tau`, so

```
Valid(strict)  ⊆  Valid(medium)  ⊆  Valid(loose)
```

| preset | alpha | tau |
|---|---|---|
| `strict` | 2.0 | 0.06 |
| **`medium` (active)** | **1.2** | **0.04** |
| `loose` | 1.0 | 0.02 |

The earlier preset set was **not** monotonic (strict had a smaller `tau` than
medium), which made the names misleading. The subset property is now asserted
twice: on synthetic pairs covering clean / diagonal / small-margin geometry, and
on real image pairs from the dataset.

Active candidate set: **`medium`**.

---

## Extreme relations

`leftmost` · `rightmost` · `topmost` · `bottommost`

Rank components by centroid on the relevant axis. Rank 1 and rank 2 must differ
by at least `extreme.margin_px` (4 px); otherwise the result is **ambiguous**
rather than being silently resolved in favour of rank 1.

Ties in the ranking break by `component_id`, so the ordering never depends on
input order or on an unstable sort.

---

## `nearest`

```
nearest(anchor) -> unique target component
```

Rules, in order:

1. Both the **anchor** and every **candidate** are drawn from the `nearest`
   eligible set (no border truncation, no suspected merge, not tiny).
2. Distance is **`boundary_distance`**, i.e. the minimum gap between the two
   components' pixel sets. The centroid-distance shortcut is **not** used: it
   misranks elongated and L-shaped buildings, which is exactly the geometry this
   dataset is full of. Config validation rejects
   `distance_metric: centroid_distance` outright.
3. Rank 1 and rank 2 must clear both an absolute floor (`margin_px_floor`, 2 px)
   and a normalized fraction of the image diagonal
   (`margin_diag_fraction`, 0.005). Otherwise the result is **ambiguous**.

### Border policy (FROZEN)

> **`nearest` anchor: `touches_image_border == false`
> `nearest` target: `touches_image_border == false`**

`nearest` is the only relation whose metric is a **boundary** distance. A
component clipped by the tile edge has an **incomplete boundary**, so both the
distance measured from it and the ordering of distances around it are
unreliable. Accepting a border anchor would produce ground truth that looks
plausible but is not.

An earlier revision allowed an ineligible anchor (reasoning that a human may ask
about a clipped building). That was inconsistent with the fact that the *metric
itself* is unreliable for such a component, and it is superseded.

**This does not change the tile-relative policy.** `leftmost`, `rightmost`,
`topmost`, `bottommost` (and the four directional predicates) still accept
border components, because "leftmost within this tile" is a legitimate
tile-relative question. Only the magnitude/distance relations exclude them.

`tests/test_relations.py` asserts all three parts: border anchor rejected, border
target rejected, and extremes unaffected.

**A single admissible candidate is VALID, not ambiguous.** With exactly one
candidate the answer is unique — there is no runner-up and no margin to apply.
Ambiguity in this relation means the distance margin is too small. Measured on
the dataset: 89.5% of anchors have two or more candidates, 6.8% have exactly
one, and 3.8% have none.

---

## Size relations

`largest` · `smallest`

Separation is always expressed as **`larger_area / smaller_area >= ratio_margin`**,
so the same threshold means the same thing for both relations.
(Comparing rank1/rank2 directly would make the ratio below 1 for `smallest` and
the relation could never be valid — this was a real bug, now covered by
`test_smallest_ratio_uses_larger_over_smaller`.)

### `ratio_margin` = 1.10

Chosen on the principle **supervision reliability > sample count**. Measured
over all 4038 images:

| ratio | largest valid | smallest valid | largest ambiguous | smallest ambiguous |
|---|---|---|---|---|
| 1.05 | 2965 | 3085 | 491 | 369 |
| **1.10** | **2566** | **2769** | **890** | **685** |
| 1.15 | 2242 | 2523 | 1214 | 931 |

1.05 admits pairs whose areas differ by only 5%. The **representation
discrepancy** between rasterized area and continuous polygon area has a median of
about 6% for small components (see `docs/data_representation.md` section 3). That
is a deterministic rasterization / discretization discrepancy, **not random
measurement noise**. A 5% margin therefore sits **inside that discrepancy**: the
rank order between rank 1 and rank 2 would not be dependable ground truth. 1.10
moves the decision boundary above it.

The cost is about 13% fewer valid samples and roughly 1.8x more samples
classified as ambiguous. That is the intended trade: ambiguous samples are
**discarded, never guessed**.

**Eligibility is deliberately asymmetric:**

| Relation | Excludes border-truncated | Excludes suspected merge | Excludes tiny |
|---|---|---|---|
| `largest` | yes | yes | **no** |
| `smallest` | yes | yes | **yes** |

Reasoning: a tiny component can hardly be the largest, so excluding it would
only shrink the candidate pool for no benefit. But a tiny component absolutely
must not be reported as "the smallest building" — that would select
rasterization artifact rather than a building.

---

## Component quality flags

| Flag | Definition | Nature |
|---|---|---|
| `touches_image_border` | Component reaches the tile edge (from metadata; not re-estimated) | Factual |
| `suspected_large_merge` | `bbox_extent_ratio > 0.20` | **Conservative heuristic, NOT merge ground truth** |
| `tiny_component` | `area_px < 150` | Threshold on the measured area distribution |
| `geometry_valid` | NOT border AND NOT merge AND NOT tiny | Convenience summary only |

`geometry_valid` is **not** applied uniformly to all relations — see the scope
section above. Use per-relation eligibility.

Merge heuristic calibration: the `bbox_extent_ratio` distribution has
p99 = 0.0969, p99.5 = 0.1280, p99.8 = 0.1959, p99.9 = 0.2483, max = 0.6178. The
0.20 cut sits in the p99.8 tail and flags 70 of 36926 components (0.19%), i.e.
it targets the extreme tail rather than a broad quantile.

Tiny threshold calibration: `area_px` has p0.5 = 88, p1 = 105, p2 = 136,
p5 = 226. The 150 cut flags 1063 of 36926 components (2.88%).

---

## Structured results and auditable rejection reasons

Every relation evaluation returns a `RelationResult` rather than a bare boolean:

```python
RelationResult(
    valid: bool,
    relation: str,
    subject: int | None,
    object: int | None,
    ambiguous: bool,
    reason: str | None,
    score: float | None,
    margin: float | None,
    evidence: dict,
)
```

Status vocabulary:

| Reason | Meaning |
|---|---|
| `valid` | Relation holds unambiguously |
| `ambiguous` | Geometrically real but too close to a decision boundary |
| `wrong_direction` | Dominant-axis sign is opposed |
| `axis_not_dominant` | Pair is too diagonal for a clean axis claim |
| `margin_too_small` | Below the normalized minimum margin |
| `invalid_component` | A participant fails this relation's eligibility |
| `too_few_eligible_candidates` | Not enough eligible components to decide |
| `anchor_not_eligible` | Anchor is degenerate (zero area) |
| `no_such_component` | Unknown component id |

This is what lets a downstream validator audit *why* a candidate relation was
accepted, discarded, or marked ambiguous, instead of merely counting survivors.

---

## Summary

| Relation | Family | Key rule | Border eligible |
|---|---|---|---|
| `left_of` | direction | subject is LEFT of object; sign + dominance + margin | yes |
| `right_of` | direction | subject is RIGHT of object; sign + dominance + margin | yes |
| `above` | direction | subject is ABOVE object; sign + dominance + margin | yes |
| `below` | direction | subject is BELOW object; sign + dominance + margin | yes |
| `leftmost` | extreme | rank margin >= 4 px | yes |
| `rightmost` | extreme | rank margin >= 4 px | yes |
| `topmost` | extreme | rank margin >= 4 px | yes |
| `bottommost` | extreme | rank margin >= 4 px | yes |
| `nearest` | distance | boundary gap + dual margin; **anchor and target both non-border** | **no** |
| `largest` | size | larger/smaller >= 1.10 | no |
| `smallest` | size | larger/smaller >= 1.10, tiny excluded | no |
