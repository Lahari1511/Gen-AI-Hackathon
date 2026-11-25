"""Module 1b - Hugging Face dataset access.

Loads the Indian traffic sign dataset, finds the image and label columns
automatically, and gives random samples (image + true label) for the demo
and for evaluation.
"""
from __future__ import annotations

import random
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def load(split: str = "train"):
    from datasets import load_dataset
    return load_dataset(config.HF_DATASET, split=split)


def columns(ds) -> tuple[str, str]:
    """Return (image_column, label_column) by looking at the feature types."""
    from datasets import ClassLabel, Image as HFImage
    img_col = next((k for k, v in ds.features.items() if isinstance(v, HFImage)), None)
    lab_col = next((k for k, v in ds.features.items() if isinstance(v, ClassLabel)), None)
    if img_col is None or lab_col is None:
        raise ValueError(f"Could not find image/label columns in: {ds.features}")
    return img_col, lab_col


def label_names(ds=None) -> list[str]:
    ds = ds or load()
    _, lab = columns(ds)
    return list(ds.features[lab].names)


def describe() -> dict:
    """Small summary for the README / Review 2 (rows, classes, columns)."""
    ds = load()
    img, lab = columns(ds)
    names = label_names(ds)
    counts = {}
    for y in ds[lab]:
        counts[names[y]] = counts.get(names[y], 0) + 1
    return {"rows": len(ds), "image_column": img, "label_column": lab,
            "num_classes": len(names), "class_counts": dict(sorted(counts.items(), key=lambda x: -x[1]))}


def sample(n: int = 1, seed: int | None = None) -> list[tuple]:
    """Return n random (PIL image, true label name) pairs."""
    ds = load()
    img, lab = columns(ds)
    names = label_names(ds)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), k=min(n, len(ds)))
    return [(ds[i][img].convert("RGB"), names[ds[i][lab]]) for i in idx]
