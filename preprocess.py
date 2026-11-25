"""Module 2 - Image validation and preprocessing.

Checks the uploaded file is a real image, measures quality (size, blur,
brightness) and returns a resized copy plus a list of quality warnings.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_FILE_MB = 10


@dataclass
class ImageCheck:
    ok: bool
    image: Image.Image | None = None
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def quality(self) -> str:
        """'good', 'fair' or 'poor' based on how many warnings were raised."""
        if not self.ok:
            return "poor"
        if len(self.warnings) == 0:
            return "good"
        return "fair" if len(self.warnings) == 1 else "poor"


def load_image(data: bytes | Image.Image) -> ImageCheck:
    """Validate raw bytes (from an upload) or a PIL image (from the dataset)."""
    if isinstance(data, Image.Image):
        img = data
    else:
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            return ImageCheck(ok=False, error=f"File is larger than {MAX_FILE_MB} MB.")
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()                      # detects corrupt files
            img = Image.open(io.BytesIO(data))  # re-open after verify
        except (UnidentifiedImageError, OSError, SyntaxError):
            return ImageCheck(ok=False, error="This file is not a readable image (corrupt or unsupported format).")
        if img.format and img.format.upper() not in ALLOWED_FORMATS:
            return ImageCheck(ok=False, error=f"Unsupported format {img.format}. Use JPG, PNG, WEBP or BMP.")
    img = img.convert("RGB")
    return assess_quality(img)


def assess_quality(img: Image.Image) -> ImageCheck:
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    warnings = []
    if min(h, w) < config.MIN_SIDE_PX:
        warnings.append(f"Image is very small ({w}x{h} px); details may be lost.")
    if blur < config.BLUR_THRESHOLD:
        warnings.append("Image looks blurry.")
    if brightness < config.DARK_THRESHOLD:
        warnings.append("Image is very dark.")
    elif brightness > config.BRIGHT_THRESHOLD:
        warnings.append("Image is very bright / washed out (possible glare).")

    return ImageCheck(
        ok=True,
        image=resize_for_model(img),
        warnings=warnings,
        metrics={"width": w, "height": h, "blur_score": round(blur, 1), "brightness": round(brightness, 1)},
    )


def resize_for_model(img: Image.Image, max_side: int = 768, min_side: int = 224) -> Image.Image:
    """Keep aspect ratio; shrink big photos and enlarge tiny dataset crops."""
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
    elif max(w, h) < min_side:
        scale = min_side / max(w, h)
    if scale != 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
