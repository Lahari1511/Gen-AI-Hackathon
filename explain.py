"""Module 5/6 - Explanation generation with Gemini (LLM).

Builds a grounded prompt, calls Gemini, checks the answer has the four
sections and no driving commands, retries once, and falls back to a safe
reference-only explanation if Gemini is unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config, prompts, safety
from .jsonutil import extract_json

SECTIONS = ("meaning", "purpose", "importance", "uncertainty")

IMPORTANCE_BY_CATEGORY = {
    "Mandatory / Regulatory": "Regulatory signs are rules that road users are expected to follow. Respecting them keeps traffic orderly and helps prevent conflicts between vehicles and people.",
    "Mandatory / Compulsory": "Compulsory signs tell road users which direction or action is required. They keep traffic moving in an organised and predictable way.",
    "Cautionary / Warning": "Warning signs give road users early notice of a hazard ahead, so there is time to prepare and react safely.",
    "Informatory": "Informatory signs help road users find facilities and plan ahead, which reduces confusion on the road.",
}


@dataclass
class Explanation:
    ok: bool
    sections: dict = field(default_factory=dict)
    source: str = ""          # "gemini", "reference-fallback"
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""
    violations: list[str] = field(default_factory=list)


def validate(data: dict | None) -> str:
    """Return '' if valid, otherwise the reason it is invalid."""
    if not isinstance(data, dict):
        return "answer was not a JSON object"
    missing = [k for k in SECTIONS if not str(data.get(k, "")).strip()]
    if missing:
        return f"missing or empty sections: {', '.join(missing)}"
    too_long = [k for k in SECTIONS if len(str(data[k])) > 700]
    if too_long:
        return f"sections too long: {', '.join(too_long)}"
    hits = safety.command_violations(" ".join(str(data[k]) for k in SECTIONS))
    if hits:
        return f"contains live driving instructions: {hits}"
    return ""


def fallback(ref_entry: dict, confidence: str, notes: list[str]) -> dict:
    """Deterministic explanation straight from the verified reference (no LLM)."""
    return {
        "meaning": ref_entry["meaning"],
        "purpose": ref_entry["purpose"],
        "importance": IMPORTANCE_BY_CATEGORY.get(ref_entry["category"], "Understanding road signs helps all road users share the road safely."),
        "uncertainty": uncertainty_text(confidence, notes),
    }


def uncertainty_text(confidence: str, notes: list[str]) -> str:
    base = {"high": "The recognition looks reliable.",
            "medium": "The recognition is reasonably likely but not certain.",
            "low": "The recognition is uncertain."}[confidence]
    return base + (" " + " ".join(notes) if notes else "") + " Always confirm with official sources."


def call_gemini(system: str, user: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0.2, max_output_tokens=1500,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # no hidden "thinking" tokens -> full answer
        ),
    )
    return resp.text or ""


def explain(sign_name: str, description: str, ref_entry: dict, confidence: str,
            notes: list[str], llm=None, max_attempts: int = 2) -> Explanation:
    """Generate the 4-section explanation. `llm` can be swapped for testing."""
    t = time.time()
    llm = llm or call_gemini
    if not config.GEMINI_API_KEY and llm is call_gemini:
        return Explanation(True, fallback(ref_entry, confidence, notes), "reference-fallback", 0,
                           time.time() - t, "GEMINI_API_KEY missing - showing reference explanation.")
    user = prompts.USER_V2.format(
        sign_name=sign_name, description=description or "not available",
        category=ref_entry["category"], shape_colour=ref_entry.get("shape_colour", ""),
        meaning=ref_entry["meaning"], purpose=ref_entry["purpose"],
        confidence=confidence, notes="; ".join(notes) if notes else "none",
    )
    reason, last_err, violations = "", "", []
    for attempt in range(1, max_attempts + 1):
        prompt = user if not reason else user + prompts.RETRY_SUFFIX.format(reason=reason)
        try:
            raw = llm(prompts.SYSTEM_V2, prompt)
        except Exception as e:  # network, bad key, quota (429), server error
            last_err = f"Gemini error: {e}"
            break
        data = extract_json(raw)
        reason = validate(data)
        if not reason:
            sections = {k: str(data[k]).strip() for k in SECTIONS}
            return Explanation(True, sections, "gemini", attempt, time.time() - t)
        if "driving instructions" in reason:
            violations = safety.command_violations(raw)
        last_err = f"Invalid Gemini answer: {reason}"
    # Safe fallback: verified reference text only
    return Explanation(True, fallback(ref_entry, confidence, notes), "reference-fallback",
                       attempt, time.time() - t, last_err, violations)
