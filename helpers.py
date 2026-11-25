"""Synthetic test images (no internet needed)."""
import io

from PIL import Image, ImageDraw, ImageFilter


def no_entry(size=200) -> Image.Image:
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    m = size // 20
    d.ellipse([m, m, size - m, size - m], fill=(200, 20, 30), outline=(0, 0, 0), width=3)
    d.rectangle([size // 5, size // 2 - size // 13, size - size // 5, size // 2 + size // 13], fill=(255, 255, 255))
    return img


def blurry(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=8))


def to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
