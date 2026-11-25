"""The full pipeline that connects all modules.

image -> validation -> Vision AI (Qwen2.5-VL) -> reference check -> confidence
      -> Gemini explanation (4 sections) -> safety check -> result
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from . import explain as explain_mod
from . import preprocess, reference, safety, vision

CONF_SCORE = {"high": 3, "medium": 2, "low": 1}
SCORE_CONF = {3: "high", 2: "medium", 1: "low"}


@dataclass
class Result:
    status: str                      # explained | uncertain | no_sign | refused | error
    message: str = ""
    sign_name: str = ""
    category: str = ""
    confidence: str = "low"
    notes: list[str] = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    alternatives: list[dict] = field(default_factory=list)
    explanation_source: str = ""
    label_match: str = "No label given"
    image_check: preprocess.ImageCheck | None = None
    vision: vision.VisionResult | None = None
    seconds: float = 0.0


def same_sign(a: str, b: str) -> bool:
    """True if two names refer to the same sign (handles aliases / small spelling differences)."""
    if not a or not b:
        return False
    if reference.normalise(a) == reference.normalise(b):
        return True
    ra, rb = reference.lookup(a), reference.lookup(b)
    if ra.found and rb.found and ra.name == rb.name:
        # speed limits must also match on the number
        va, vb = reference.speed_value(a), reference.speed_value(b)
        return va == vb if (va or vb) else True
    return False


def combine_confidence(vlm_conf: str, check: preprocess.ImageCheck, ref_found: bool,
                       label_match: str, vlm_issues: list[str]) -> tuple[str, list[str]]:
    score, notes = CONF_SCORE.get(vlm_conf, 1), []
    if check.warnings:
        score -= 1 if check.quality == "fair" else 2
        notes += check.warnings
    if vlm_issues:
        score -= 1
        notes.append("Vision model noted: " + ", ".join(vlm_issues) + ".")
    if not ref_found:
        score -= 1
        notes.append("Sign not found in the verified reference list.")
    if label_match == "Mismatch":
        score -= 1
        notes.append("The label you entered does not match what the image shows.")
    score = max(1, min(3, score))
    if check.warnings and score == 3:   # never 'high' if image quality is questionable
        score = 2
    return SCORE_CONF[score], notes


def run(image: bytes | Image.Image, user_label: str = "", question: str = "",
        backend: str | None = None, candidates: list[str] | None = None,
        llm=None) -> Result:
    t0 = time.time()

    # 1. Safety: refuse live-driving questions before doing anything else
    if question and safety.driving_intent(question):
        return Result(status="refused", message=safety.REFUSAL_MESSAGE, seconds=time.time() - t0)

    # 2. Validate and check image quality
    check = preprocess.load_image(image)
    if not check.ok:
        return Result(status="error", message=check.error, image_check=check, seconds=time.time() - t0)

    # 3. Recognise the sign
    cands = candidates or reference.all_sign_names()
    if backend == "label":
        if not user_label:
            return Result(status="error", message="Label-only mode needs a class label.", image_check=check)
        vis = vision.VisionResult(ok=True, is_sign=True, sign_name=user_label, confidence="medium",
                                  description="(label-only mode: image not analysed)", backend="label")
    else:
        vis = vision.recognise(check.image, cands, backend)
    if not vis.ok:
        return Result(status="error", message=vis.error, image_check=check, vision=vis, seconds=time.time() - t0)
    if not vis.is_sign:
        return Result(status="no_sign", message="No traffic sign was detected in this image, so no explanation is given.",
                      image_check=check, vision=vis, seconds=time.time() - t0)

    # 4. Check against the verified reference and the user's label
    ref = reference.lookup(vis.sign_name)
    label_match = "No label given"
    if user_label and backend != "label":
        label_match = "Match" if same_sign(user_label, vis.sign_name) else "Mismatch"

    # 5. Combine signals into one confidence level
    conf, notes = combine_confidence(vis.confidence, check, ref.found, label_match, vis.issues)
    alternatives = []
    for alt in vis.alternatives:
        r = reference.lookup(alt)
        if r.found and r.name != ref.name:
            alternatives.append({"name": r.name, "meaning": r.entry["meaning"]})

    base = dict(sign_name=vis.sign_name, confidence=conf, notes=notes, alternatives=alternatives,
                label_match=label_match, image_check=check, vision=vis)

    # 6. Low confidence or unverified sign -> do NOT generate, say "cannot confirm"
    if conf == "low" or not ref.found:
        if ref.found:
            alternatives.insert(0, {"name": ref.name, "meaning": ref.entry["meaning"]})
        return Result(status="uncertain", seconds=time.time() - t0,
                      message="Cannot confirm this sign reliably. Possible meanings are shown for learning only. "
                              "Try a clearer, closer photo.", category=ref.entry["category"] if ref.found else "", **base)

    # 7. Generate the grounded 4-section explanation
    kwargs = {"llm": llm} if llm else {}
    exp = explain_mod.explain(ref.name if not reference.speed_value(vis.sign_name) else vis.sign_name,
                              vis.description, ref.entry, conf, notes, **kwargs)
    msg = exp.error if exp.source == "reference-fallback" and exp.error else ""
    return Result(status="explained", message=msg, category=ref.entry["category"], sections=exp.sections,
                  explanation_source=exp.source, seconds=time.time() - t0, **base)
