"""Phase 1 (capture): FormFactor model + deterministic shape extractor.

Pins the extractor's precision bias -- hardware briefs are full of shape-like
words that are NOT shapes (star ground, hex inverter, heart rate, Nth round),
and those must never be misread as an outline request.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kicraft.design.models import FormFactor, IntentSlot
from kicraft.design.synthesis.form_factor import extract_form_factor


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def test_formfactor_defaults_to_rect():
    assert FormFactor().shape == "rect"


def test_formfactor_normalizes_shape_token():
    assert FormFactor(shape="  Circle ").shape == "circle"
    assert FormFactor(shape="").shape == "rect"


def test_formfactor_allows_unknown_named_shape():
    # Lenient: a novel/library shape name must not brick the intent commit.
    assert FormFactor(shape="snowman").shape == "snowman"


def test_formfactor_rejects_negative_dimensions():
    with pytest.raises(ValidationError):
        FormFactor(shape="rounded_rect", corner_radius_mm=-1.0)


def test_formfactor_forbids_extra_fields():
    with pytest.raises(ValidationError):
        FormFactor(shape="circle", radius=5)  # type: ignore[call-arg]


def test_intentslot_form_factor_optional_and_parsed():
    assert IntentSlot(goal="x").form_factor is None
    slot = IntentSlot(goal="x", form_factor={"shape": "circle", "size_mm": 50})
    assert slot.form_factor is not None
    assert slot.form_factor.shape == "circle"
    assert slot.form_factor.size_mm == 50.0


# --------------------------------------------------------------------------- #
# Extractor — positives
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,shape",
    [
        ("A circular LED ring", "circle"),
        ("round coaster temperature sensor board", "circle"),
        ("make it round", "circle"),
        ("the board should be round", "circle"),
        ("disc-shaped proximity sensor", "circle"),
        ("hexagonal LED badge", "hexagon"),
        ("a hexagon board with addressable LEDs", "hexagon"),
        ("octagonal coaster", "octagon"),
        ("pentagon outline", "pentagon"),
        ("triangular breakout", "triangle"),
        ("snowman ornament with warm-white LEDs", "snowman"),
        ("heart-shaped valentine badge", "heart"),
        ("star-shaped pcb", "star"),
        ("gear-shaped fidget board", "gear"),
        ("dev board with rounded corners", "rounded_rect"),
        ("rounded rectangle outline", "rounded_rect"),
        ("board with filleted corners", "rounded_rect"),
        ("chamfered corners on the card edge", "chamfered_rect"),
        ("a chamfered rectangle board", "chamfered_rect"),
        ("beveled corners", "chamfered_rect"),
    ],
)
def test_extractor_positive(text, shape):
    ff = extract_form_factor(text)
    assert ff is not None, text
    assert ff.shape == shape
    assert ff.note  # provenance phrase recorded


# --------------------------------------------------------------------------- #
# Extractor — negatives (must stay rectangular / None)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "",
        "a 4-layer rectangular board",
        "ground plane with stitching vias",
        "star ground topology for the power board",
        "star topology clock distribution",
        "74HC04 hex inverter on a 2-layer board",
        "optical heart rate monitor board",
        "second round of prototyping for the board",
        "the third round for the board layout",
        "round up the ADC readings",
        "an ESP32-S3 plant monitor with soil moisture",
    ],
)
def test_extractor_negative(text):
    assert extract_form_factor(text) is None, text


# --------------------------------------------------------------------------- #
# Extractor — size
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text,size",
    [
        ("round board Ø40", 40.0),
        ("circular coaster, 50 mm diameter", 50.0),
        ("a 25mm diameter round sensor", 25.0),
        ("circular badge, 60mm wide", 60.0),
    ],
)
def test_extractor_size(text, size):
    ff = extract_form_factor(text)
    assert ff is not None
    assert ff.size_mm == size


def test_extractor_no_size_is_none():
    ff = extract_form_factor("a circular LED ring board")
    assert ff is not None
    assert ff.size_mm is None


# --------------------------------------------------------------------------- #
# Intent stage-commit wiring (the deterministic safety net)
# --------------------------------------------------------------------------- #

import json  # noqa: E402

from kicraft.design.cli_app import main  # noqa: E402


def _commit_intent(tmp_path, capsys, intent: dict, brief: str | None = None):
    """Drive `stage-commit intent` against a production-shaped project dir
    (``<proj>/.kicraft/state.json`` + ``<proj>/brief.txt``) and return
    ``(rc, payload, written_state)``."""
    kdir = tmp_path / ".kicraft"
    kdir.mkdir(parents=True, exist_ok=True)
    state_path = kdir / "state.json"
    if brief is not None:
        (tmp_path / "brief.txt").write_text(brief)
    slot = tmp_path / "intent_slot.json"
    slot.write_text(json.dumps(intent))
    rc = main(
        [
            "stage-commit", "intent",
            "--slot-file", str(slot),
            "--project-stem", "FF_TEST",
            "--no-archive",
            str(state_path),
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out) if out.strip() else {}
    written = json.loads(state_path.read_text())
    return rc, payload, written


def _intent(goal: str, **extra) -> dict:
    return {
        "goal": goal,
        "constraints": [],
        "named_parts": [],
        "assumptions": [],
        "inferred_expertise": "intermediate",
        **extra,
    }


def test_intent_commit_captures_round_shape(tmp_path, capsys):
    rc, payload, written = _commit_intent(
        tmp_path, capsys,
        _intent("A round coaster temperature sensor board, 50 mm diameter."),
    )
    assert rc == 0, payload
    assert payload.get("form_factor") == "circle"
    assert written["intent"]["form_factor"]["shape"] == "circle"
    assert written["intent"]["form_factor"]["size_mm"] == 50.0


def test_intent_commit_rectangular_leaves_form_factor_none(tmp_path, capsys):
    rc, payload, written = _commit_intent(
        tmp_path, capsys,
        _intent("An ESP32-S3 plant monitor with soil moisture and BME280."),
    )
    assert rc == 0, payload
    assert "form_factor" not in payload
    assert written["intent"]["form_factor"] is None


def test_intent_commit_preserves_explicit_shape(tmp_path, capsys):
    # The model set a shape plain text wouldn't reveal; the extractor must defer.
    rc, payload, written = _commit_intent(
        tmp_path, capsys,
        _intent("A small badge board.", form_factor={"shape": "hexagon"}),
    )
    assert rc == 0, payload
    assert written["intent"]["form_factor"]["shape"] == "hexagon"


def test_intent_commit_reads_brief_when_goal_drops_shape(tmp_path, capsys):
    rc, payload, written = _commit_intent(
        tmp_path, capsys,
        _intent("An LED ornament with warm-white LEDs."),
        brief="Make me a snowman-shaped LED ornament for the tree.",
    )
    assert rc == 0, payload
    assert written["intent"]["form_factor"]["shape"] == "snowman"
