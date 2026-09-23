"""Guards for the landing-page onboarding content (kicraft.server.examples).

``EXAMPLE_PROMPTS`` drives the animated placeholder. The "Surprise me" button
composes a fresh brief per click with ``generate_brief(seed)``; these tests pin
that generator's contract (deterministic, novel, well-formed, inside the frozen
design envelope) and that it does not quietly fall back to a saved brief.
"""
from __future__ import annotations

import re
import string

from kicraft.server.examples import (
    BRIEF_SLOTS,
    BRIEF_TEMPLATES,
    BRIEF_TRAILING,
    EXAMPLE_PROMPTS,
    generate_brief,
    indefinite,
)
from kicraft.tuning.benchmark import briefs as selfeval_briefs

_SEEDS = range(500)


def test_example_prompts_nonempty_strings():
    assert EXAMPLE_PROMPTS, "EXAMPLE_PROMPTS must not be empty"
    for p in EXAMPLE_PROMPTS:
        assert isinstance(p, str) and p.strip(), f"bad example prompt: {p!r}"


def test_templates_only_use_known_slots():
    """A typo'd field name would raise at click time, in front of a user."""
    for _family, templates in BRIEF_TEMPLATES:
        assert templates, f"empty template group: {_family!r}"
    for template in [t for _f, ts in BRIEF_TEMPLATES for t in ts] + list(BRIEF_TRAILING):
        for _literal, field, _spec, _conv in string.Formatter().parse(template):
            if field is None:
                continue
            assert field in BRIEF_SLOTS, f"unknown slot {field!r} in {template!r}"


def test_generated_briefs_are_well_formed_and_novel():
    generated = [generate_brief(s) for s in _SEEDS]
    for brief in generated:
        assert isinstance(brief, str) and brief.strip(), f"blank brief: {brief!r}"
        assert "\n" not in brief, f"brief must be one line: {brief!r}"
        assert brief[:1] == brief[:1].upper(), f"brief must start a sentence: {brief!r}"
        assert "  " not in brief, f"brief has collapsed spacing: {brief!r}"
    # The button's guarantee is click-to-click novelty: consecutive seeds (the
    # persistent counter hands out consecutive ones) must never repeat.
    for a, b in zip(generated, generated[1:]):
        assert a != b, f"consecutive seeds repeated a brief: {a!r}"
    # And the generated stream must stay overwhelmingly distinct rather than
    # cycling a handful of phrases.
    assert len(set(generated)) >= 490, f"only {len(set(generated))} distinct of {len(generated)}"
    # Nothing is a saved corpus brief (the entire point of the change).
    assert not set(generated) & set(selfeval_briefs())
    assert not set(generated) & set(EXAMPLE_PROMPTS)


def test_generate_brief_is_deterministic_per_seed():
    for seed in range(50):
        assert generate_brief(seed) == generate_brief(seed)


def test_generated_briefs_stay_inside_the_frozen_design_envelope():
    """Surprises must remain reasonable asks: <=24 V, <=3 A, <=100 x 100 mm."""
    for value in BRIEF_SLOTS["supply"]:
        for volts in (int(m) for m in re.findall(r"(\d+)\s*V", value)):
            assert volts <= 24, f"supply exceeds the envelope: {value!r}"
    for amps in BRIEF_SLOTS["amps"]:
        assert float(amps) <= 3.0, f"rating exceeds the envelope: {amps!r}"
    for size in BRIEF_SLOTS["size"]:
        width, height = (float(v) for v in re.findall(r"\d+", size)[:2])
        assert width <= 100 and height <= 100, f"size exceeds the envelope: {size!r}"


def test_indefinite_article_follows_pronunciation():
    cases = {
        "USB-C 5 V input": "a USB-C 5 V input",
        "UART header": "a UART header",
        "SPI header": "an SPI header",
        "I2C header": "an I2C header",
        "RGB status LED": "an RGB status LED",
        "LED matrix": "an LED matrix",
        "MCP6002": "an MCP6002",
        "LM358": "an LM358",
        "RP2040": "an RP2040",
        "STM32G0": "an STM32G0",
        "ULN2003 darlington array": "a ULN2003 darlington array",
        "ESP32-C3 module": "an ESP32-C3 module",
        "CH32V003": "a CH32V003",
        "18 V DC input": "an 18 V DC input",
        "24 V DC screw terminal": "a 24 V DC screw terminal",
        "temperature": "a temperature",
        "ambient-light": "an ambient-light",
    }
    for phrase, expected in cases.items():
        assert indefinite(phrase) == expected


def test_every_slot_value_gets_a_valid_article():
    for name, values in BRIEF_SLOTS.items():
        assert values, f"empty slot pool: {name!r}"
        for value in values:
            assert indefinite(value) in (f"a {value}", f"an {value}")
