"""Module 7 - Safety guardrails.

1. Input check: refuse questions that ask for live driving help.
2. Output check: catch explanations that sound like real-time commands.
"""
from __future__ import annotations

import re

REFUSAL_MESSAGE = (
    "This tool is for learning only and cannot guide you while driving. "
    "Please focus on the road and follow the official signs, traffic police and local traffic laws."
)

# Phrases that show the user wants real-time help while on the road
_DRIVING_INTENT = [
    r"\b(i am|i'm|im|we are|we're)\s+(driving|riding|on the road|at the signal|behind the wheel)\b",
    r"\b(right now|at the moment|currently)\b.*\b(turn|stop|go|overtake|park|drive|cross|speed)\b",
    r"\b(should|can|may|must)\s+i\s+(turn|stop|go|overtake|park|cross|speed|proceed|enter|take)\b",
    r"\bwhat (should|do) i do\b.*\b(now|here|this signal|this junction|road)\b",
    r"\b(tell|guide|navigate)\s+me\b.*\b(drive|route|turn|road|way)\b",
    r"\bis it safe (for me )?to\b",
]

# Phrases in the output that sound like live commands to the reader
_COMMANDS = [
    r"\byou\s+(should|must|need to|have to|can|may)\s+(now\s+)?(stop|turn|slow|speed|overtake|brake|go|proceed|accelerate|change lanes?|park|enter|cross)\b",
    r"(^|[.!]\s+)(slow down|speed up|turn (left|right) now|overtake|brake|accelerate|go ahead|proceed now|stop now|stop immediately)\b",
    r"\b(right now|immediately|at once)\b",
    r"\bit is (now )?safe (for you )?to\b",
    r"\bfollow (this|my) (route|direction)\b",
]


def driving_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _DRIVING_INTENT)


def command_violations(text: str) -> list[str]:
    t = (text or "").lower()
    hits = []
    for p in _COMMANDS:
        m = re.search(p, t, flags=re.M)
        if m:
            hits.append(m.group(0).strip(" .!"))
    return hits
