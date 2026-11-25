from PIL import Image

from src import preprocess, reference, safety
from tests.helpers import blurry, no_entry, to_bytes


# ---------- Image validation ----------
def test_corrupt_file_rejected():
    res = preprocess.load_image(b"this is not an image")
    assert not res.ok and "not a readable image" in res.error


def test_clear_image_has_no_warnings():
    res = preprocess.load_image(to_bytes(no_entry()))
    assert res.ok and res.warnings == [] and res.quality == "good"


def test_blurry_image_flagged():
    res = preprocess.load_image(to_bytes(blurry(no_entry())))
    assert any("blurry" in w for w in res.warnings)


def test_dark_and_tiny_image_flagged():
    res = preprocess.load_image(Image.new("RGB", (20, 20), (5, 5, 5)))
    assert res.quality == "poor"
    assert any("small" in w for w in res.warnings) and any("dark" in w for w in res.warnings)


def test_tiny_image_upscaled_for_model():
    res = preprocess.load_image(no_entry(30))
    assert max(res.image.size) >= 224


# ---------- Reference lookup ----------
def test_reference_exact_alias_and_fuzzy():
    assert reference.lookup("No Entry").name == "No Entry"
    assert reference.lookup("no honking").name == "Horn Prohibited"
    assert reference.lookup("speed limit 50").name == "Speed Limit"
    assert reference.lookup("zebra crossing").name == "Pedestrian Crossing"


def test_reference_unknown_not_found():
    assert not reference.lookup("purple dragon").found


# ---------- Safety ----------
def test_driving_questions_refused():
    for q in ["I'm driving, can I turn now?", "Should I overtake here?", "Is it safe to go?",
              "what should i do now at this junction"]:
        assert safety.driving_intent(q), q


def test_learning_questions_allowed():
    for q in ["What does this sign mean?", "Why is this sign used near schools?", ""]:
        assert not safety.driving_intent(q), q


def test_command_detection():
    assert safety.command_violations("You should slow down right now.")
    assert safety.command_violations("It is safe to overtake here.")
    assert not safety.command_violations("This sign tells drivers that entry is not allowed from this direction.")
    assert not safety.command_violations("Drivers are expected to stop at the line.")
