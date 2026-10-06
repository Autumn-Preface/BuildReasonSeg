"""Task 8B tests: image loader, tiling, merge, reasoning context, outputs (CPU only, no model runtime)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.runtime import detector, imageio
from buildreasonseg.runtime.context import (
    ANCHOR_FRACTIONS,
    anchor_pixel,
    context_to_global,
    direction_satisfied,
    plan_context,
    round_half_up,
)
from buildreasonseg.runtime.detector import (
    GlobalProposal,
    eligible,
    merge_proposals,
    plan_tiles,
    select_reference,
)
from buildreasonseg.runtime.outputs import allocate_outputs, allocate_run_suffix


# ---------------------------------------------------------------- image loader


def _write(array: np.ndarray, path: Path, mode: str | None = None) -> Path:
    Image.fromarray(array, mode=mode).save(path)
    return path


def test_uint8_rgb(tmp_path: Path) -> None:
    path = _write(np.zeros((16, 16, 3), dtype=np.uint8), tmp_path / "a.png")
    loaded = imageio.load_image(path)
    assert loaded.rgb.shape == (16, 16, 3) and loaded.dtype == "uint8" and loaded.channels == 3


def test_uint16_rgb_fixed_linear_scaling(tmp_path: Path, monkeypatch) -> None:
    """PIL cannot write 3-channel uint16, so the synthetic TIFF read is injected."""

    array = np.zeros((8, 8, 3), dtype=np.uint16)
    array[..., 0] = 65535
    array[..., 1] = 32768
    path = tmp_path / "b.tif"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(imageio, "_read_raw", lambda target: (array, "RGB", np.dtype("uint16")))
    loaded = imageio.load_image(path)
    assert loaded.dtype == "uint16"
    assert loaded.rgb[0, 0, 0] == 255
    assert loaded.rgb[0, 0, 1] == round(32768 * 255 / 65535)
    assert any("uint16" in note for note in loaded.notes)


def test_rgba_drops_alpha(tmp_path: Path) -> None:
    array = np.zeros((8, 8, 4), dtype=np.uint8)
    array[..., 3] = 255
    path = _write(array, tmp_path / "c.png")
    loaded = imageio.load_image(path)
    assert loaded.channels == 3
    assert any("RGBA" in note for note in loaded.notes)


def test_grayscale_rejected(tmp_path: Path) -> None:
    path = _write(np.zeros((8, 8), dtype=np.uint8), tmp_path / "d.png", mode="L")
    with pytest.raises(BuildReasonSegError) as error:
        imageio.load_image(path)
    assert error.value.code == "E203"


def test_two_and_six_channel_rejected(tmp_path: Path, monkeypatch) -> None:
    for channels in (2, 6):
        path = tmp_path / f"ch{channels}.png"
        path.write_bytes(b"synthetic")
        array = np.zeros((8, 8, channels), dtype=np.uint8)
        monkeypatch.setattr(imageio, "_read_raw",
                            lambda target, a=array: (a, "RGB", np.dtype("uint8")))
        with pytest.raises(BuildReasonSegError) as error:
            imageio.load_image(path)
        assert error.value.code == "E203"


def test_extension_and_missing_file(tmp_path: Path) -> None:
    weird = tmp_path / "a.xyz"
    weird.write_bytes(b"x")
    with pytest.raises(BuildReasonSegError) as error:
        imageio.load_image(weird)
    assert error.value.code == "E203"
    with pytest.raises(BuildReasonSegError) as error:
        imageio.load_image(tmp_path / "missing.png")
    assert error.value.code == "E201"


def test_pad_to_512_is_reflection_not_resize() -> None:
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    padded, padding = imageio.pad_to_512(rgb)
    assert padded.shape == (512, 512, 3)
    assert np.array_equal(padded[:4, :6], rgb)
    assert padding["bottom"] == 508 and padding["right"] == 506


# ---------------------------------------------------------------- tiling


def test_plan_tiles_small_image_is_single_padded_tile() -> None:
    windows = plan_tiles(300, 200)
    assert len(windows) == 1
    assert windows[0].top == 0 and windows[0].left == 0


def test_plan_tiles_overlap_and_exact_coverage() -> None:
    windows = plan_tiles(1024, 1024)
    tops = sorted({window.top for window in windows})
    lefts = sorted({window.left for window in windows})
    assert tops == [0, 384, 512] and lefts == [0, 384, 512]
    assert detector.TILE_OVERLAP == 128 and detector.TILE_STRIDE == 384
    assert max(tops) + 512 == 1024 and max(lefts) + 512 == 1024


def test_plan_tiles_edge_clamped() -> None:
    windows = plan_tiles(1000, 700)
    assert max(window.top for window in windows) + 512 == 700
    assert max(window.left for window in windows) + 512 == 1000


def test_extract_tile_reflection_padding_mapping() -> None:
    rgb = np.zeros((900, 900, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    inside = detector.TileWindow("t", top=300, left=300, size=512, pad_top=0, pad_bottom=0,
                                pad_left=0, pad_right=0)
    tile, padding = detector.extract_tile(rgb, inside)
    assert tile.shape == (512, 512, 3) and padding["applied"] is False
    assert np.array_equal(tile, rgb[300:812, 300:812])

    outside = detector.TileWindow("t2", top=500, left=500, size=512, pad_top=0, pad_bottom=0,
                                  pad_left=0, pad_right=0)
    padded_tile, padded = detector.extract_tile(rgb, outside)
    assert padded_tile.shape == (512, 512, 3) and padded["applied"] is True
    assert padded["bottom"] == 112 and padded["right"] == 112


# ---------------------------------------------------------------- merge


def _proposal(mask: np.ndarray, confidence: float, tile: str, raw_index: int, tile_index: int,
              proposal_id: int = -1) -> GlobalProposal:
    box = detector.bbox_of(mask)
    return GlobalProposal(proposal_id=proposal_id, source_tile_id=tile, tile_index=tile_index,
                          confidence=confidence, mask_crop=np.ascontiguousarray(mask[box[0]:box[2] + 1, box[1]:box[3] + 1], dtype=bool),
                          global_bbox=box,
                          mask_area=int(mask.sum()),
                          touches_image_border=detector.touches_original_border(mask),
                          border_clearance=detector.border_clearance(mask),
                          centroid=detector.centroid_of(mask), raw_index=raw_index,
                          image_size=mask.shape)


def _to_full(proposal, size: int = 512) -> np.ndarray:
    """Rebuild the full-frame mask from the compact representation (test-only helper)."""

    full = np.zeros((size, size), dtype=bool)
    top, left, bottom, right = proposal.global_bbox
    full[top:bottom + 1, left:right + 1] = proposal.mask_crop
    return full


def _entry(mask: np.ndarray, confidence: float, tile: str, tile_index: int, raw_index: int,
           image_size: tuple[int, int] = (512, 512)) -> dict:
    """Compact accumulated-entry builder used by the merge tests (test-local)."""

    compact = detector._compact_mask(mask, top=0, left=0)
    assert compact is not None
    mask_crop, global_bbox = compact
    return {"source_tile_id": tile, "tile_index": tile_index, "confidence": confidence,
            "mask_crop": mask_crop, "global_bbox": global_bbox, "image_size": image_size,
            "raw_index": raw_index}


def test_merge_duplicate_iou_threshold_and_no_union() -> None:
    first = np.zeros((512, 512), dtype=bool)
    first[100:140, 100:140] = True           # area 1600
    second = np.zeros((512, 512), dtype=bool)
    second[104:144, 104:144] = True          # same area 1600, IoU ≈ 0.78 → duplicate
    third = np.zeros((512, 512), dtype=bool)
    third[300:340, 300:340] = True           # area 1600 but far away → separate
    entries = [_entry(first, 0.5, "t1", 0, 0),
               _entry(second, 0.9, "t2", 1, 1),
               _entry(third, 0.7, "t3", 2, 2)]
    merged, groups = merge_proposals(entries)
    assert len(merged) == 2
    assert detector.iou_of(first, second) >= detector.DUPLICATE_IOU
    # the winner is one of the originals (equal area → higher confidence), never a union
    duplicate_group = [proposal for proposal in merged
                       if np.array_equal(_to_full(proposal), first)
                       or np.array_equal(_to_full(proposal), second)]
    assert len(duplicate_group) == 1
    assert duplicate_group[0].confidence == pytest.approx(0.9)
    # "no union" means the kept mask is bit-identical to one of the raw inputs
    assert duplicate_group[0].mask_area in (int(first.sum()), int(second.sum()))
    assert not any(proposal.mask_area > max(int(first.sum()), int(second.sum()))
                   for proposal in merged)
    assert sorted(proposal.proposal_id for proposal in merged) == [0, 1]


def test_merge_priority_prefers_non_border_then_area() -> None:
    border = np.zeros((512, 512), dtype=bool)
    border[0:60, 200:260] = True             # touches border, larger area
    inner = np.zeros((512, 512), dtype=bool)
    inner[250:290, 250:290] = True           # no border, smaller
    overlapped = inner.copy()
    overlapped[252:292, 252:292] = True
    entries = [_entry(border, 0.9, "t1", 0, 0),
               _entry(inner, 0.4, "t2", 1, 1),
               _entry(overlapped, 0.6, "t3", 2, 2)]
    merged, _groups = merge_proposals(entries)
    ids = {proposal.source_tile_id for proposal in merged}
    assert "t2" in ids or "t3" in ids            # the non-border duplicate wins its group
    winner = [proposal for proposal in merged if proposal.source_tile_id in ("t2", "t3")][0]
    assert winner.touches_image_border is False


def test_stable_ids_by_centroid_then_area() -> None:
    top = np.zeros((512, 512), dtype=bool)
    top[10:30, 10:30] = True
    bottom = np.zeros((512, 512), dtype=bool)
    bottom[400:440, 400:440] = True
    entries = [_entry(bottom, 0.5, "t2", 1, 0),
               _entry(top, 0.5, "t1", 0, 0)]
    merged, _ = merge_proposals(entries)
    assert merged[0].centroid[0] < merged[1].centroid[0]
    assert [proposal.proposal_id for proposal in merged] == [0, 1]


def test_eligibility_rules() -> None:
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:140, 100:140] = True
    proposal = _proposal(mask, 0.9, "t", 0, 0)
    assert eligible(proposal) is True

    big = np.zeros((512, 512), dtype=bool)
    big[10:400, 10:400] = True                     # extent ratio > 0.20
    assert eligible(_proposal(big, 0.9, "t", 0, 0)) is False

    bordered = np.zeros((512, 512), dtype=bool)
    bordered[0:30, 100:130] = True
    assert eligible(_proposal(bordered, 0.9, "t", 0, 0)) is False

    tiny = np.zeros((512, 512), dtype=bool)
    tiny[100:110, 100:110] = True
    assert eligible(_proposal(tiny, 0.9, "t", 0, 0), family="largest") is True
    assert eligible(_proposal(tiny, 0.9, "t", 0, 0), family="smallest") is False


def test_select_reference_tie_break() -> None:
    first = np.zeros((512, 512), dtype=bool)
    first[100:130, 100:130] = True
    second = first.copy()
    entries = [_entry(first, 0.5, "t1", 0, 0),
               _entry(second, 0.9, "t9", 1, 0)]
    merged, _ = merge_proposals(entries)
    selected = select_reference(merged)
    assert selected is not None
    assert selected.mask_area == int(first.sum())


def _reference_rect(height: int, width: int, confidence: float, proposal_id: int,
                    *, top: int = 100, left: int = 100) -> GlobalProposal:
    mask = np.zeros((512, 512), dtype=bool)
    mask[top:top + height, left:left + width] = True
    return _proposal(mask, confidence, f"r{proposal_id}", proposal_id, proposal_id,
                     proposal_id=proposal_id)


def test_largest_extent_dominance_exception_selects_strictly_dominant_candidate() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    dominant = _reference_rect(120, 120, 0.61, 2, top=250, left=250)
    assert eligible(baseline, family="largest") is True
    assert eligible(dominant, family="largest") is False
    assert detector.eligible_proposals([baseline, dominant], family="largest") == [baseline]
    assert select_reference([baseline, dominant], family="largest") is dominant


def test_largest_extent_exception_rejects_lower_confidence_candidate() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    candidate = _reference_rect(120, 120, 0.59, 2, top=250, left=250)
    assert select_reference([baseline, candidate], family="largest") is baseline


def test_largest_extent_exception_requires_strict_confidence_gain() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    candidate = _reference_rect(120, 120, 0.60, 2, top=250, left=250)
    assert select_reference([baseline, candidate], family="largest") is baseline


def test_largest_extent_exception_requires_strict_area_gain() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    same_area = _reference_rect(40, 160, 0.95, 2, top=250, left=250)
    assert baseline.mask_area == same_area.mask_area
    assert eligible(same_area, family="largest") is False
    assert select_reference([baseline, same_area], family="largest") is baseline


def test_largest_extent_exception_never_admits_border_proposal() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    border = _reference_rect(120, 120, 0.95, 2, top=0, left=250)
    assert border.touches_image_border is True
    assert select_reference([baseline, border], family="largest") is baseline


def test_largest_extent_exception_requires_frozen_baseline() -> None:
    only_extent_violation = _reference_rect(120, 120, 0.95, 1)
    assert eligible(only_extent_violation, family="largest") is False
    assert select_reference([only_extent_violation], family="largest") is None


def test_largest_extent_exception_keeps_production_area_order() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    exception_a = _reference_rect(120, 120, 0.70, 2, top=250, left=50)
    exception_b = _reference_rect(130, 130, 0.61, 3, top=250, left=250)
    assert select_reference([baseline, exception_a, exception_b], family="largest") is exception_b


def test_largest_extent_exception_does_not_change_smallest_family() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    extent_violation = _reference_rect(120, 120, 0.95, 2, top=250, left=250)
    assert eligible(extent_violation, family="smallest") is False
    assert select_reference([baseline, extent_violation], family="smallest") is baseline


# ---------------------------------------------------------------- reasoning context


def test_anchor_pixels_exact() -> None:
    assert ANCHOR_FRACTIONS["right_of"] == (0.30, 0.50)
    assert anchor_pixel("right_of") == (round_half_up(0.30 * 512), 256)
    assert anchor_pixel("left_of")[0] == round_half_up(0.70 * 512)
    assert anchor_pixel("above")[1] == round_half_up(0.70 * 512)
    assert anchor_pixel("below")[1] == round_half_up(0.30 * 512)
    assert round_half_up(153.6) == 154 and round_half_up(153.4) == 153


def test_plan_context_places_reference_at_anchor() -> None:
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:540, 500:540] = True
    proposal = _proposal(mask, 0.9, "t", 0, 0)
    for direction in ("left_of", "right_of", "above", "below"):
        context = plan_context(proposal, direction, size=512)
        anchor_x, anchor_y = anchor_pixel(direction)
        assert abs((proposal.centroid[1] - context.left) - anchor_x) <= 1
        assert abs((proposal.centroid[0] - context.top) - anchor_y) <= 1


def test_reference_too_large_raises_e404() -> None:
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[0:500, 0:500] = True
    proposal = _proposal(mask, 0.9, "t", 0, 0)
    with pytest.raises(BuildReasonSegError) as error:
        plan_context(proposal, "right_of")
    assert error.value.code == "E404"
    assert error.value.context["reason"] == "reference_too_large_for_rc1_context"


def test_context_out_of_image_uses_reflection_and_maps_back() -> None:
    mask = np.zeros((300, 300), dtype=bool)
    mask[10:40, 10:40] = True
    proposal = _proposal(mask, 0.9, "t", 0, 0)
    context = plan_context(proposal, "right_of")
    assert context.top < 0 or context.left < 0
    context_mask = np.zeros((512, 512), dtype=bool)
    context_mask[256, 256] = True
    full, padding = context_to_global(context_mask, context, (300, 300))
    assert full.shape == (300, 300)
    assert padding["applied"] is True
    assert full.sum() == 1


def test_direction_satisfied_hard_check() -> None:
    reference = (100.0, 100.0)
    assert direction_satisfied((100.0, 130.0), reference, "right_of") is True
    assert direction_satisfied((100.0, 70.0), reference, "right_of") is False
    assert direction_satisfied((70.0, 100.0), reference, "above") is True
    assert direction_satisfied((130.0, 100.0), reference, "below") is True
    assert direction_satisfied((100.0, 130.0), reference, "left_of") is False


# ---------------------------------------------------------------- outputs


def test_output_filename_suffix_never_overwrites(tmp_path: Path, monkeypatch) -> None:
    stem = "example"
    mask_dir = tmp_path
    assert allocate_run_suffix(mask_dir, stem) == 0
    (mask_dir / f"{stem}_mask.png").write_bytes(b"x")
    assert allocate_run_suffix(mask_dir, stem) == 1
    (mask_dir / f"{stem}_mask_001.png").write_bytes(b"x")
    assert allocate_run_suffix(mask_dir, stem) == 2


def test_mask_and_overlay_sizes_and_values(tmp_path: Path, monkeypatch) -> None:
    from buildreasonseg import paths

    monkeypatch.setattr(paths, "inference_dir", lambda: tmp_path)
    rgb = np.full((40, 60, 3), 100, dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:20, 10:20] = True
    outputs = allocate_outputs(Path("example.png"))
    info = __import__("buildreasonseg.runtime.outputs", fromlist=["x"]).save_final_outputs(
        outputs, rgb, mask, alpha=0.5)
    with Image.open(info["mask"]) as handle:
        array = np.asarray(handle)
    assert array.shape == (40, 60)
    assert set(np.unique(array)).issubset({0, 255})
    with Image.open(info["overlay"]) as handle:
        overlay = np.asarray(handle)
    assert overlay.shape == (40, 60, 3)
    assert overlay[15, 15, 0] > rgb[15, 15, 0]          # red overlay applied
    assert np.array_equal(overlay[0, 0], rgb[0, 0])     # untouched outside the mask


def test_alpha_bounds() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    for bad in (0.0, -0.2, 1.5):
        with pytest.raises(BuildReasonSegError):
            imageio.overlay(rgb, mask, alpha=bad)
    assert imageio.overlay(rgb, mask, alpha=1.0) is not None


def test_diagnostics_required_fields(tmp_path: Path) -> None:
    from buildreasonseg.runtime import outputs as outputs_module

    outputs = outputs_module.SampleOutputs(stem="s", suffix_index=0,
                                           diagnostics_dir=str(tmp_path / "s"))
    payload = {"prompt": "p", "parsed": {"program": "largest_to_left_of_to_nearest"},
               "proposals": {"count": 0}, "status": "FAILED"}
    outputs_module.save_diagnostics(outputs, payload, maps={"P_dir": np.zeros((4, 4), np.float32)})
    directory = Path(outputs.diagnostics_dir)
    for name in ("prompt.txt", "parsed_program.json", "proposals.json", "result.json", "maps.npz"):
        assert (directory / name).is_file(), name
    saved = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED"
    with np.load(directory / "maps.npz") as arrays:
        assert "P_dir" in arrays


# ------------------------------------------------- Task 8B.3-M1A.2B compact-proposal contract


def _compact_proposal(mask: np.ndarray, *, top: int, left: int, image_size: tuple[int, int],
                      confidence: float = 0.5, tile: str = "t", raw_index: int = 0,
                      proposal_id: int = -1) -> detector.GlobalProposal:
    compact = detector._compact_mask(mask, top=top, left=left)
    assert compact is not None
    crop, bbox = compact
    return detector.GlobalProposal(
        proposal_id=proposal_id, source_tile_id=tile, tile_index=0, confidence=confidence,
        mask_crop=crop, global_bbox=bbox, mask_area=int(crop.sum()),
        touches_image_border=detector._crop_touches_image_border(crop, bbox, image_size),
        border_clearance=detector._crop_border_clearance(crop, bbox, image_size),
        centroid=detector._crop_centroid(crop, bbox), raw_index=raw_index, image_size=image_size)


def test_compact_geometry_is_tight_and_global() -> None:
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:20, 5:15] = True
    proposal = _compact_proposal(mask, top=100, left=200, image_size=(400, 500))
    top, left, bottom, right = proposal.global_bbox
    assert (top, left, bottom, right) == (110, 205, 119, 214)
    assert proposal.mask_crop.shape == (10, 10)
    assert proposal.mask_crop.all()
    assert proposal.mask_area == 100
    assert proposal.centroid == (114.5, 209.5)
    assert np.array_equal(_to_full(proposal, 400)[110:120, 205:215], proposal.mask_crop)


def test_proposal_iou_equals_legacy_full_frame_iou() -> None:
    first = np.zeros((200, 200), dtype=bool)
    first[20:60, 20:60] = True
    second = np.zeros((200, 200), dtype=bool)
    second[40:80, 40:80] = True
    disjoint = np.zeros((200, 200), dtype=bool)
    disjoint[150:170, 150:170] = True
    a = _compact_proposal(first, top=0, left=0, image_size=(200, 200))
    b = _compact_proposal(second, top=0, left=0, image_size=(200, 200))
    c = _compact_proposal(disjoint, top=0, left=0, image_size=(200, 200))
    assert detector.proposal_iou(a, b) == pytest.approx(detector.iou_of(first, second), abs=1e-12)
    assert detector.proposal_iou(a, c) == 0.0
    assert detector.proposal_iou(a, a) == pytest.approx(1.0, abs=1e-12)


def test_merge_equivalence_with_compact_masks() -> None:
    """Two independent equal-area 40x40 overlapping masks; a third disjoint equal-area mask stays separate."""

    first = np.zeros((512, 512), dtype=bool)
    first[100:140, 100:140] = True            # area 1600
    second = np.zeros((512, 512), dtype=bool)
    second[104:144, 104:144] = True           # independent equal-area mask, IoU ≈ 0.78 (> 0.50)
    third = np.zeros((512, 512), dtype=bool)
    third[300:340, 300:340] = True            # equal area 1600 but disjoint

    entries = [_entry(first, 0.5, "t1", 0, 0), _entry(second, 0.9, "t2", 1, 0),
               _entry(third, 0.7, "t3", 2, 0)]
    assert all(entry["mask_crop"].sum() == 1600 for entry in entries)

    merged, _groups = detector.merge_proposals(entries)
    assert len(merged) == 2
    assert [p.proposal_id for p in merged] == [0, 1]
    assert sorted(p.mask_area for p in merged) == [1600, 1600]
    duplicate = [p for p in merged
                 if np.array_equal(_to_full(p), first) or np.array_equal(_to_full(p), second)]
    assert len(duplicate) == 1
    assert duplicate[0].confidence == pytest.approx(0.9)
    assert not any(p.mask_area != 1600 for p in merged)


def test_reference_context_equivalence_with_legacy_full_frame() -> None:
    from buildreasonseg.runtime.context import CONTEXT_SIZE
    from buildreasonseg.runtime.core import reference_mask_from_proposal

    size = (300, 320)
    full = np.zeros(size, dtype=bool)
    full[40:70, 50:90] = True
    proposal = _compact_proposal(full, top=0, left=0, image_size=size)

    class Ctx:
        def __init__(self, top: int, left: int) -> None:
            self.top, self.left, self.size = top, left, CONTEXT_SIZE

    for context_top, context_left in ((40, 50), (-20, -15), (10, 10), (250, 270)):
        context = Ctx(context_top, context_left)
        expected = np.zeros((CONTEXT_SIZE, CONTEXT_SIZE), dtype=bool)
        valid_top = max(0, context_top)
        valid_left = max(0, context_left)
        valid_bottom = min(size[0], context_top + CONTEXT_SIZE)
        valid_right = min(size[1], context_left + CONTEXT_SIZE)
        if valid_bottom > valid_top and valid_right > valid_left:
            expected[valid_top - context_top:valid_top - context_top + (valid_bottom - valid_top),
                     valid_left - context_left:valid_left - context_left + (valid_right - valid_left)] = \
                full[valid_top:valid_bottom, valid_left:valid_right]
        assert np.array_equal(reference_mask_from_proposal(proposal, context), expected)


def test_preview_uses_crop_not_full_frame_mask() -> None:
    from buildreasonseg.runtime import outputs as outputs_module

    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:30, 10:30] = True
    proposal = _compact_proposal(mask, top=100, left=120, image_size=(256, 256), proposal_id=3)
    preview = outputs_module.proposals_preview_image(rgb, [proposal], selected_id=3)
    assert preview.shape == (256, 256, 3)
    assert preview[100:160, 120:180].sum() > 0
    assert preview[:100, :120].sum() == 0


def test_serialization_contract_has_no_mask_payload() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:15, 5:15] = True
    proposal = _compact_proposal(mask, top=10, left=20, image_size=(100, 100))
    payload = proposal.to_dict()
    assert "mask_crop" not in payload and "global_mask" not in payload
    assert payload["global_bbox"] == [15, 25, 24, 34]
    assert payload["mask_area"] == 100
    assert isinstance(payload["bbox_extent_ratio"], float)


def test_large_image_path_never_allocates_full_frame_bool(monkeypatch) -> None:
    """Fake (5000,5000,3) image, one tile window, two independent overlapping 50x60 detections.

    Keeps a 5000x5000 bool `np.zeros` guard, does **not** monkeypatch `logical_and`/`logical_or`, and instead wraps
    `detector.proposal_iou()` to assert that neither input carries a full-frame crop.
    """

    real_zeros = np.zeros

    def zero_guard(*args, **kwargs):
        shape = args[0] if args else kwargs.get("shape")
        if shape is not None and tuple(shape) == (5000, 5000) and kwargs.get("dtype") is bool:
            raise AssertionError("full-frame 5000x5000 bool allocation attempted")
        return real_zeros(*args, **kwargs)

    class FakeRgb:
        shape = (5000, 5000, 3)

    class FakeWindow:
        source_tile_id = "tile_0000_r000_c000"
        top, left, size = 0, 0, 512

    block = real_zeros((512, 512), dtype=bool)
    block[100:150, 200:260] = True                      # 50 x 60 = 3000
    detections = [{"index": 0, "confidence": 0.9, "mask": block, "box": [200, 100, 260, 150]},
                  {"index": 1, "confidence": 0.8, "mask": block.copy(),
                   "box": [200, 100, 260, 150]}]

    monkeypatch.setattr(detector, "plan_tiles", lambda width, height, **kwargs: [FakeWindow()])
    monkeypatch.setattr(detector, "extract_tile",
                        lambda rgb, window: (real_zeros((512, 512, 3), dtype=np.uint8),
                                             {"applied": False}))
    monkeypatch.setattr(detector.DetectorRuntime, "detect_tile", lambda self, tile_rgb: detections)

    calls: list[tuple] = []
    real_iou = detector.proposal_iou

    def counting_iou(first, second):                    # noqa: ANN001 - test double
        for proposal in (first, second):
            assert tuple(proposal.mask_crop.shape) != (5000, 5000)
            assert proposal.mask_crop.size < 5000 * 5000
            assert proposal.mask_area == 3000
        calls.append((first.global_bbox, second.global_bbox))
        return real_iou(first, second)

    monkeypatch.setattr(detector, "proposal_iou", counting_iou)
    monkeypatch.setattr(detector.np, "zeros", zero_guard)

    runtime = detector.DetectorRuntime.__new__(detector.DetectorRuntime)
    runtime.device, runtime.imgsz, runtime.conf, runtime.max_det = "cpu", 640, 0.05, 300
    runtime.checkpoint, runtime._model, runtime.load_seconds = Path("unused.pt"), None, None

    assert zero_guard((10, 10), dtype=bool).shape == (10, 10)      # guard delegates correctly
    result = runtime.detect_global(FakeRgb())
    assert result["raw_count"] == 2
    assert len(calls) == 1, calls
    assert len(result["merged"]) == 1
    assert result["merged"][0].mask_area == 3000
    assert result["merged"][0].mask_crop.shape == (50, 60)
    assert tuple(result["merged"][0].mask_crop.shape) != (5000, 5000)

def test_success_semantics_contract_exact():
    """The public machine-readable SUCCESS contract is exactly this dictionary."""
    from buildreasonseg.runtime.pipeline import success_semantics
    assert success_semantics() == {
        "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
        "semantic_status": "NOT_EVALUATED",
        "semantic_note": "SUCCESS means the RC1 runtime completed and passed its current structural checks; semantic target correctness is not established.",
    }


def test_pipeline_result_ok_is_status_compatibility():
    """PipelineResult.ok is exactly (status == "SUCCESS") and nothing else."""
    import dataclasses
    from buildreasonseg.runtime.pipeline import PipelineResult
    required = {field.name for field in dataclasses.fields(PipelineResult)
                if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING}
    payload = {"status": "SUCCESS"}

    def build(status):
        kwargs = {name: payload if name == "result_payload" else None for name in required}
        kwargs["status"] = status
        if "result_payload" not in kwargs:
            kwargs["result_payload"] = payload
        return PipelineResult(**kwargs)

    success = build("SUCCESS")
    failed = build("FAILED")
    other = build("PARTIAL")
    assert success.ok is True and success.status == "SUCCESS"
    assert failed.ok is False and failed.status == "FAILED"
    assert other.ok is False and other.status == "PARTIAL"
    for result in (success, failed, other):
        assert result.ok == (result.status == "SUCCESS")
