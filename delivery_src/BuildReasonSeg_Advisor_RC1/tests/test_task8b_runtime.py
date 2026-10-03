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
                          confidence=confidence, global_mask=mask, global_bbox=box,
                          mask_area=int(mask.sum()),
                          touches_image_border=detector.touches_original_border(mask),
                          border_clearance=detector.border_clearance(mask),
                          centroid=detector.centroid_of(mask), raw_index=raw_index)


def test_merge_duplicate_iou_threshold_and_no_union() -> None:
    first = np.zeros((512, 512), dtype=bool)
    first[100:140, 100:140] = True
    second = first.copy()
    second[104:144, 104:144] = True          # IoU well above 0.50 → duplicate
    third = np.zeros((512, 512), dtype=bool)
    third[300:340, 300:340] = True           # far away → separate
    entries = [{"mask": first, "confidence": 0.5, "source_tile_id": "t1", "tile_index": 0,
                "raw_index": 0},
               {"mask": second, "confidence": 0.9, "source_tile_id": "t2", "tile_index": 1,
                "raw_index": 1},
               {"mask": third, "confidence": 0.7, "source_tile_id": "t3", "tile_index": 2,
                "raw_index": 2}]
    merged, groups = merge_proposals(entries)
    assert len(merged) == 2
    assert detector.iou_of(first, second) >= detector.DUPLICATE_IOU
    # the winner is one of the originals (equal area → higher confidence), never a union
    duplicate_group = [proposal for proposal in merged
                       if np.array_equal(proposal.global_mask, first)
                       or np.array_equal(proposal.global_mask, second)]
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
    entries = [{"mask": border, "confidence": 0.9, "source_tile_id": "t1", "tile_index": 0,
                "raw_index": 0},
               {"mask": inner, "confidence": 0.4, "source_tile_id": "t2", "tile_index": 1,
                "raw_index": 1},
               {"mask": overlapped, "confidence": 0.6, "source_tile_id": "t3", "tile_index": 2,
                "raw_index": 2}]
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
    entries = [{"mask": bottom, "confidence": 0.5, "source_tile_id": "t2", "tile_index": 1,
                "raw_index": 0},
               {"mask": top, "confidence": 0.5, "source_tile_id": "t1", "tile_index": 0, "raw_index": 0}]
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
    entries = [{"mask": first, "confidence": 0.5, "source_tile_id": "t1", "tile_index": 0,
                "raw_index": 0},
               {"mask": second, "confidence": 0.9, "source_tile_id": "t9", "tile_index": 1,
                "raw_index": 0}]
    merged, _ = merge_proposals(entries)
    selected = select_reference(merged)
    assert selected is not None
    assert selected.mask_area == int(first.sum())


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
