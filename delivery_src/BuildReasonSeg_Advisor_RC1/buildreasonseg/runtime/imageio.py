"""Image loading / validation contract for the RC1 runtime (Task 8B section 9).

Accepted: 3-channel RGB, 4-channel RGBA (alpha dropped), uint8 and uint16 (fixed linear 0→0 / 65535→255).
Rejected with E203: grayscale, 2-channel, >4-channel, SAR / raw multispectral, uninterpretable TIFF.
No histogram or percentile stretching is ever applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from buildreasonseg.errors import BuildReasonSegError

ACCEPTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
UINT16_MAX = 65535


@dataclass
class LoadedImage:
    rgb: np.ndarray                      # uint8 (H, W, 3)
    path: Path
    width: int
    height: int
    dtype: str
    channels: int
    original_mode: str
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {"input_path": str(self.path), "input_size": [self.width, self.height],
                "input_dtype": self.dtype, "input_channels": self.channels,
                "input_mode": self.original_mode, "notes": list(self.notes)}


def _read_raw(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as error:  # pragma: no cover - Pillow is a hard dependency
        raise BuildReasonSegError("E502", detail=f"无法导入 Pillow: {error}") from error
    try:
        with Image.open(path) as handle:
            handle.load()
            return np.asarray(handle), handle.mode, np.asarray(handle).dtype
    except BuildReasonSegError:
        raise
    except Exception as error:
        raise BuildReasonSegError("E202", detail=f"影像无法读取: {path.name} ({error})") from error


def _to_uint8(array: np.ndarray, dtype: np.dtype, mode: str) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    if dtype == np.uint8:
        return array, notes
    if dtype == np.uint16:
        scaled = np.rint(array.astype(np.float64) * (255.0 / UINT16_MAX)).clip(0, 255).astype(np.uint8)
        notes.append("uint16→uint8 固定线性缩放 (0→0, 65535→255)")
        return scaled, notes
    raise BuildReasonSegError(
        "E203",
        detail=f"不支持的影像 dtype: {dtype}（仅支持 uint8 / uint16 RGB）")


def load_image(path: str | Path) -> LoadedImage:
    target = Path(path)
    if not target.is_file():
        raise BuildReasonSegError("E201", detail=f"影像不存在: {target}")
    if target.suffix.lower() not in ACCEPTED_SUFFIXES:
        raise BuildReasonSegError(
            "E203",
            detail=f"不支持的扩展名 {target.suffix}（支持 {', '.join(ACCEPTED_SUFFIXES)}）")

    array, mode, dtype = _read_raw(target)
    notes: list[str] = []
    if array.ndim == 2:
        raise BuildReasonSegError("E203", detail=f"灰度影像是单通道，RC1 仅支持 RGB 光学影像 (mode={mode})")
    if array.ndim != 3:
        raise BuildReasonSegError("E203", detail=f"无法解释的影像维度: {array.shape}")
    channels = int(array.shape[2])
    if channels == 4:
        array = array[:, :, :3]
        notes.append("RGBA→RGB（去掉 alpha 通道）")
    elif channels != 3:
        raise BuildReasonSegError(
            "E203",
            detail=f"{channels} 通道影像不受支持（SAR / 原始多光谱 / 未处理多波段均不在 RC1 输入域）")
    rgb, scale_notes = _to_uint8(np.ascontiguousarray(array), dtype, mode)
    notes.extend(scale_notes)
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise BuildReasonSegError("E202", detail=f"影像尺寸为空: {target}")
    return LoadedImage(rgb=rgb, path=target, width=int(rgb.shape[1]), height=int(rgb.shape[0]),
                       dtype=str(dtype), channels=3, original_mode=str(mode), notes=notes)


def reflection_pad(array: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """Reflection padding (mode='reflect'); no content resize ever happens."""

    if top == bottom == left == right == 0:
        return array
    pad_width = [(top, bottom), (left, right)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pad_width, mode="reflect")


def pad_to_512(rgb: np.ndarray, size: int = 512) -> tuple[np.ndarray, dict]:
    """Section 11.1: `<=512` images are reflection-padded to `size`, never resized."""

    height, width = rgb.shape[:2]
    if height > size or width > size:
        raise ValueError(f"pad_to_512 requires both dimensions <= {size}, got {(width, height)}")
    bottom = size - height
    right = size - width
    padded = reflection_pad(rgb, 0, bottom, 0, right)
    return padded, {"top": 0, "bottom": bottom, "left": 0, "right": right,
                    "mode": "reflect", "applied": bool(bottom or right)}


def crop_with_reflection(rgb: np.ndarray, top: int, left: int, size: int = 512) -> tuple[np.ndarray, dict]:
    """Crop `size x size` starting at (top, left), reflection-padding whatever falls outside."""

    height, width = rgb.shape[:2]
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, (top + size) - height)
    pad_right = max(0, (left + size) - width)
    source_top = max(0, top)
    source_left = max(0, left)
    source_bottom = min(height, top + size)
    source_right = min(width, left + size)
    window = rgb[source_top:source_bottom, source_left:source_right]
    padded = reflection_pad(window, pad_top, pad_bottom, pad_left, pad_right)
    if padded.shape[0] != size or padded.shape[1] != size:
        # a degenerate 1-pixel image can still under-fill after reflection padding
        padded = np.pad(padded, [(0, max(0, size - padded.shape[0])),
                                 (0, max(0, size - padded.shape[1])), (0, 0)], mode="edge")
    padding = {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right,
               "origin_top": int(top), "origin_left": int(left), "mode": "reflect",
               "applied": bool(pad_top or pad_bottom or pad_left or pad_right)}
    return np.ascontiguousarray(padded[:size, :size]), padding


def resize_preview(rgb: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
    """Aspect-preserving preview for diagnostics only (section 24.1); never touches the algorithm."""

    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return rgb
    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize(new_size, Image.BILINEAR))


def save_png(path: str | Path, array: np.ndarray) -> str:
    """Save an array as PNG. uint8 2-D → grayscale, uint8 3-D → RGB, bool → 0/255."""

    from PIL import Image

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype == bool:
        payload = (array.astype(np.uint8) * 255)
    elif array.dtype != np.uint8:
        payload = np.clip(array, 0, 255).astype(np.uint8)
    else:
        payload = array
    Image.fromarray(payload).save(target)
    return str(target)


def overlay(rgb: np.ndarray, mask: np.ndarray, *, alpha: float = 0.45,
            color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Translucent colour overlay; mask geometry is never altered (section 23.2)."""

    if not 0.0 < alpha <= 1.0:
        raise BuildReasonSegError("E101", detail=f"--alpha 必须在 (0, 1] 区间内，收到 {alpha}")
    base = rgb.astype(np.float32)
    layer = np.zeros_like(base)
    layer[:, :, 0], layer[:, :, 1], layer[:, :, 2] = color
    selector = mask.astype(bool)[:, :, None]
    blended = np.where(selector, base * (1.0 - alpha) + layer * alpha, base)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


__all__ = ["ACCEPTED_SUFFIXES", "LoadedImage", "UINT16_MAX", "crop_with_reflection", "load_image",
           "overlay", "pad_to_512", "reflection_pad", "resize_preview", "save_png"]
