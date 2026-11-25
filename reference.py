"""Module 4 - Knowledge verification (grounding).

Looks up a predicted sign name in the curated reference list and returns
its verified meaning, with a match score so we know how confident the match is.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from . import config


@dataclass
class RefMatch:
    found: bool
    score: float            # 0.0 - 1.0
    entry: dict | None = None
    query: str = ""

    @property
    def name(self) -> str:
        return self.entry["name"] if self.entry else ""


def normalise(text: str) -> str:
    text = (text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(sign|signs|ahead)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_reference() -> list[dict]:
    with open(config.REFERENCE_FILE, encoding="utf-8") as f:
        return json.load(f)["signs"]


def all_sign_names() -> list[str]:
    return [s["name"] for s in load_reference()]


def lookup(name: str, min_score: float = 0.72) -> RefMatch:
    """Find the best reference entry for a sign name (exact, alias or fuzzy)."""
    q = normalise(name)
    if not q:
        return RefMatch(found=False, score=0.0, query=name)
    best, best_score = None, 0.0
    for entry in load_reference():
        for cand in [entry["name"], *entry.get("aliases", [])]:
            c = normalise(cand)
            if not c:
                continue
            if c == q:
                return RefMatch(True, 1.0, entry, name)
            score = SequenceMatcher(None, q, c).ratio()
            # numbers matter for speed limits: "speed limit 50" vs "speed limit 30"
            if re.search(r"\d", q) and re.sub(r"\D", "", q) == re.sub(r"\D", "", c) and c.split()[0] == q.split()[0]:
                score = max(score, 0.95)
            if c in q or q in c:
                score = max(score, 0.85)
            if score > best_score:
                best, best_score = entry, score
    return RefMatch(best_score >= min_score, round(best_score, 2), best if best_score >= min_score else None, name)


def speed_value(name: str) -> str | None:
    m = re.search(r"(\d{2,3})", name or "")
    return m.group(1) if m else None
