"""Guards for the landing-page onboarding content (kicraft.server.examples).

These lists drive the animated placeholder, the "Surprise me" button, and the
suggestion chips on the web app's landing page. The web wiring assumes both lists
are populated and that every chip carries a label plus a full prompt, so a typo
that empties or malforms them would silently break the inspiration UI.
"""
from __future__ import annotations

from kicraft.server.examples import CHIP_PROMPTS, EXAMPLE_PROMPTS


def test_example_prompts_nonempty_strings():
    assert EXAMPLE_PROMPTS, "EXAMPLE_PROMPTS must not be empty"
    for p in EXAMPLE_PROMPTS:
        assert isinstance(p, str) and p.strip(), f"bad example prompt: {p!r}"


def test_chip_prompts_have_label_and_prompt():
    assert CHIP_PROMPTS, "CHIP_PROMPTS must not be empty"
    for c in CHIP_PROMPTS:
        assert set(("label", "prompt")) <= c.keys(), f"chip missing keys: {c!r}"
        assert c["label"].strip(), f"chip has empty label: {c!r}"
        assert c["prompt"].strip(), f"chip has empty prompt: {c!r}"
