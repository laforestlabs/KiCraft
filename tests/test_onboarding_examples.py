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


#: One regex per requirement a brief may state. A brief that states one twice is not a harder
#: ask, it is a malformed one -- and the generator used to hand those out.
_REQUIREMENT_SUBJECTS = (
    r"enable jumper",
    r"reverse[- ]polarity",
    r"reset button",
    r"two-layer",
    r"four-layer",
    r"on one edge",
    r"power LED",
    r"\bunder \d+ ?x ?\d+ mm\b",
)


def test_no_brief_states_the_same_requirement_twice():
    """A surprise must not ask for the same thing twice.

    The first live click of the 2026-09-24 session (seed 25) read "...a power LED, and an enable
    jumper. Add an enable jumper." -- the template's own requirement plus the same one appended by
    `BRIEF_TRAILING`, which collided on 129 of 3000 seeds. Two slots could also draw one subject
    between them (seed 257: "a 3 W power LED, and a power LED").
    """
    for seed in _SEEDS:
        brief = generate_brief(seed)
        for subject in _REQUIREMENT_SUBJECTS:
            assert len(re.findall(subject, brief, flags=re.I)) <= 1, (seed, subject, brief)


def _mcu_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold().replace(" module", ""))


def test_every_mcu_the_generator_names_has_a_reviewed_carrier():
    """A surprise that names an MCU the reviewed library cannot build is unwinnable by construction.

    The class these parts answer (`microcontroller`) HAS reviewed coverage, so the BOM requires a
    reviewed carrier for it: no ordering code, no board. Live run KC-YMEWZV (seed 28, ATtiny1604)
    offered the right code on all six attempts and could never satisfy the demand, because no
    record carried that part — two of this generator's six MCUs had no carrier at all. This holds
    the generator's own vocabulary to the library it feeds, and checks the carrier is *emittable*:
    a group built from the record's own symbol/footprint/ordering code must answer the demand.
    """
    from kicraft.design.part_identity import reviewed_parts_for_feature
    from kicraft.server.stage_contracts import BomComponentGroup
    from kicraft.server.stage_work_units import _group_has_physical_feature

    carriers = reviewed_parts_for_feature("microcontroller")
    assert carriers, "the microcontroller class has no reviewed carrier at all"
    for value in BRIEF_SLOTS["mcu"]:
        token = _mcu_token(value)
        matches = [
            part
            for part in carriers
            if token in re.sub(r"[^a-z0-9]", "", part.identity)
            or token in re.sub(r"[^a-z0-9]", "", part.family)
        ]
        assert matches, f"no reviewed microcontroller carrier for the generator's {value!r}"
        for part in matches:
            group = BomComponentGroup(
                id="mcu",
                reference_prefix="U",
                quantity=1,
                sheet="MCU",
                value=part.identity.upper(),
                mpn=part.identity.upper(),
                symbol=part.symbol,
                footprint=part.footprint,
            )
            assert _group_has_physical_feature(group, "microcontroller"), (
                f"{part.identity} resolves but does not answer the demand it exists for"
            )
