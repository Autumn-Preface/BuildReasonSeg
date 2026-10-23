"""Bounded display-only previews. Never writes into a scientific array."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_PREVIEW = (960, 720)
PALETTE = ((43, 139, 214), (237, 136, 45), (50, 172, 106), (190, 86, 193),
           (231, 88, 108), (52, 173, 185), (174, 160, 43), (117, 105, 221))


def font(size=16):
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def array_summary(array):
    a = np.asarray(array)
    finite = a[np.isfinite(a)] if a.dtype.kind in "fc" else a.reshape(-1)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "raw_min": float(finite.min()) if finite.size else None,
            "raw_max": float(finite.max()) if finite.size else None}


def thumbnail(image, bounds=MAX_PREVIEW):
    result = image.copy()
    result.thumbnail(bounds, Image.Resampling.LANCZOS)
    return result


def rgb_image(array):
    return Image.fromarray(np.asarray(array), "RGB")


def caption(image, text):
    image = thumbnail(image)
    lines = text.splitlines()
    output = Image.new("RGB", (max(image.width, 340), image.height + 26 * len(lines) + 12), "white")
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    for i, line in enumerate(lines):
        draw.text((8, image.height + 4 + 26 * i), line, font=font(), fill="#20364b")
    return output


def proposal_preview(rgb, proposals, selected_id=None):
    image = thumbnail(rgb_image(rgb))
    original_height, original_width = np.asarray(rgb).shape[:2]
    sx, sy = image.width / original_width, image.height / original_height
    draw = ImageDraw.Draw(image)
    ids = []
    for proposal in proposals:              # every merged candidate, without eligibility/display filtering
        pid = int(proposal["proposal_id"])
        ids.append(pid)
        top, left, bottom, right = proposal["bbox"]
        box = (round(left * sx), round(top * sy), max(round((right + 1) * sx), round(left * sx) + 1),
               max(round((bottom + 1) * sy), round(top * sy) + 1))
        mask = Image.fromarray(np.asarray(proposal["mask_crop"], dtype=np.uint8) * 255)
        mask = mask.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST)
        color = PALETTE[pid % len(PALETTE)]
        color_layer = Image.new("RGB", mask.size, color)
        alpha = mask.point(lambda p: int(p * .28))
        image.paste(color_layer, (box[0], box[1]), alpha)
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=color, width=1)
        if pid == selected_id:
            raw = np.asarray(mask) > 0
            edge = boundary(raw)
            edge_image = Image.fromarray(edge.astype(np.uint8) * 255)
            image.paste(Image.new("RGB", mask.size, (255, 222, 44)), (box[0], box[1]), edge_image)
            draw.rectangle(box, outline="#ffde2c", width=3)
        draw.text((max(0, box[0]), max(0, box[1])), str(pid), font=font(15), fill="white",
                  stroke_width=2, stroke_fill="#17324d")
    return image, ids


def boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    inside = mask.copy()
    inside[1:] &= mask[:-1]
    inside[:-1] &= mask[1:]
    inside[:, 1:] &= mask[:, :-1]
    inside[:, :-1] &= mask[:, 1:]
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    return mask & ~inside


def reference_boundary(image, reference):
    result = image.copy()
    edge = Image.fromarray(boundary(reference).astype(np.uint8) * 255)
    edge = edge.resize(result.size, Image.Resampling.NEAREST)
    result.paste(Image.new("RGB", result.size, (255, 225, 35)), (0, 0), edge)
    return result


def heatmap(array, name, reference=None, fixed_range=None):
    a = np.asarray(array).squeeze()
    if a.ndim != 2:
        raise ValueError("Preview heatmap must be a real 2-D map")
    summary = array_summary(a)
    low, high = fixed_range or (summary["raw_min"], summary["raw_max"])
    if low is None or high is None:
        raise ValueError("No finite values available for display")
    # This temporary display mapping never flows back into the model.
    mapped = np.zeros(a.shape, dtype=np.float32) if high == low else (a.astype(np.float64) - low) / (high - low)
    mapped = np.clip(np.nan_to_num(mapped, nan=0, posinf=1, neginf=0), 0, 1)
    colors = np.stack((255 * mapped, 210 * (1 - np.abs(2 * mapped - 1)), 255 * (1 - mapped)), axis=-1).astype(np.uint8)
    image = Image.fromarray(colors).resize((320, 320), Image.Resampling.NEAREST)
    if reference is not None:
        image = reference_boundary(image, reference)
    bar_values = np.linspace(0, 1, 320)
    bar = np.stack((255 * bar_values, 210 * (1 - np.abs(2 * bar_values - 1)), 255 * (1 - bar_values)), axis=-1).astype(np.uint8)
    output = Image.new("RGB", (340, 415), "white")
    output.paste(image, (10, 27))
    output.paste(Image.fromarray(np.repeat(bar[None], 12, axis=0)), (10, 354))
    draw = ImageDraw.Draw(output)
    draw.text((10, 3), name, font=font(), fill="#17324d")
    draw.text((10, 372), f"color: [{low:.5g}, {high:.5g}]", font=font(14), fill="#20364b")
    draw.text((10, 392), f"raw: [{summary['raw_min']:.5g}, {summary['raw_max']:.5g}]", font=font(14), fill="#20364b")
    return output, {**summary, "display_range": [low, high],
                    "display_normalization": "fixed range" if fixed_range else "per-map min/max; constant maps use zero color",
                    "color_is_not_raw_probability": True}


def grid(images, columns=2):
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    result = Image.new("RGB", (width * columns, height * ((len(images) + columns - 1) // columns)), "#e9eff4")
    for i, image in enumerate(images):
        result.paste(image, ((i % columns) * width, (i // columns) * height))
    return result


class PreviewRenderer:
    """One renderer per run; keeps only bounded display images and one 512 context."""
    def __init__(self):
        self.proposal_image = None
        self.proposals = None
        self.original_shape = None
        self.reference = None
        self.context_rgb = None

    def render(self, stage, arrays, metadata):
        images, details = [], {}
        if stage == 2 and "rgb" in arrays:
            rgb = arrays["rgb"]
            image, ids = proposal_preview(rgb, arrays.get("proposals", ()))
            self.proposal_image = image.copy()
            # No full input image retained by the renderer after this event.
            self.original_shape = np.asarray(rgb).shape[:2]
            self.proposals = tuple({"proposal_id": p["proposal_id"], "bbox": p["bbox"]} for p in arrays.get("proposals", ()))
            details["displayed_proposal_ids"] = ids
            images.append(caption(image, f"YOLO merged proposals: {len(ids)} / raw: {metadata.get('raw_count')}\n显示缩放仅用于预览；全部 merged 候选保留"))
        elif stage == 3 and "reference_mask_crop" in arrays and self.proposal_image is not None:
            image = self.proposal_image.copy()
            top, left, bottom, right = metadata["bbox"]
            h, w = self.original_shape
            box = (round(left * image.width / w), round(top * image.height / h),
                   max(round((right + 1) * image.width / w), round(left * image.width / w) + 1),
                   max(round((bottom + 1) * image.height / h), round(top * image.height / h) + 1))
            edge = Image.fromarray(boundary(arrays["reference_mask_crop"]).astype(np.uint8) * 255).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST)
            image.paste(Image.new("RGB", edge.size, (255, 225, 35)), (box[0], box[1]), edge)
            ImageDraw.Draw(image).rectangle(box, outline="#ffe123", width=3)
            details["highlighted_reference_id"] = metadata["selected_id"]
            images.append(caption(image, f"Automatic Reference ID: {metadata['selected_id']}\n黄色边界为实际自动选择；不可手动改写"))
        elif stage == 4:
            if "context_rgb" in arrays:
                self.context_rgb = np.array(arrays["context_rgb"], copy=True)
            if "reference_mask" in arrays:
                self.reference = np.array(arrays["reference_mask"], copy=True)
            context = metadata.get("context", {})
            if self.proposal_image is not None and context.get("origin") is not None:
                image = self.proposal_image.copy()
                x, y = context["origin"]
                size = context["size"]
                h, w = self.original_shape
                box = (round(x * image.width / w), round(y * image.height / h),
                       round((x + size) * image.width / w), round((y + size) * image.height / h))
                ImageDraw.Draw(image).rectangle(box, outline="#f0257a", width=3)
                images.append(caption(image, f"Actual context origin={context['origin']}, size={size}\n实际计算坐标；未重新裁剪推测"))
            if self.context_rgb is not None:
                image = rgb_image(self.context_rgb)
                if self.reference is not None:
                    image = reference_boundary(image, self.reference)
                images.append(caption(image, "真实传入核心链的 RGB / 黄色 Reference 边界\nPadding: " + str(context.get("padding"))))
        elif stage == 6:
            tiles = []
            for name in ("P_dir", "P_near", "W", "A"):
                if name in arrays:
                    tile, info = heatmap(arrays[name], name, self.reference, (0., 1.) if name != "A" else None)
                    details[name] = info
                else:
                    tile = Image.new("RGB", (340, 415), "white")
                    ImageDraw.Draw(tile).text((20, 25), name + " 尚未产生", font=font(), fill="#667a89")
                tiles.append(tile)
            images.append(caption(grid(tiles), "颜色仅为显示映射；黄色为 Reference 边界\nA 使用原始实际 attention；颜色不是原始概率"))
        elif stage == 7 and "C" in arrays:
            image, info = heatmap(arrays["C"], "C / visual cosine similarity", self.reference, (-1., 1.))
            details["C"] = info
            images.append(caption(image, "C 是视觉余弦相似度，不是目标正确概率\n对应同一次运行的真实 context"))
        elif stage == 8 and "mask" in arrays:
            probability, info = heatmap(arrays["probability"], "D-B1 probability (sigmoid)", self.reference, (0., 1.))
            details["probability"] = info
            if "logits" in arrays:
                details["logits"] = array_summary(arrays["logits"])
            mask = np.asarray(arrays["mask"], dtype=bool)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")
            mask_image = caption(mask_image, "D-B1 binary candidate / unvalidated")
            tiles = [probability, mask_image]
            if self.context_rgb is not None:
                image = rgb_image(self.context_rgb)
                overlay = Image.new("RGB", image.size, (235, 74, 93))
                image.paste(overlay, (0, 0), Image.fromarray(mask.astype(np.uint8) * 90))
                tiles.append(caption(image, "Context 候选区域 / 非最终接受结果"))
            images.append(caption(grid(tiles), "D-B1 已生成候选 Mask，但尚未通过最终结构检查。"))
        elif stage == 9 and metadata.get("runtime_status") == "SUCCESS":
            for key in ("overlay_path", "mask_path"):
                path = metadata.get(key)
                if path:
                    with Image.open(path) as image:
                        images.append(caption(image.convert("RGB"), "最终结构检查通过 / 未验证目标身份\n" + ("Overlay" if key == "overlay_path" else "Mask")))
        return [thumbnail(image, MAX_PREVIEW) for image in images], details
