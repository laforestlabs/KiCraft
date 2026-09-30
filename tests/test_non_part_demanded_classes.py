"""A demanded class that is not a part class is classified or refused, never demanded as a part.

Measured input (scan of the saved states, 2026-09-29): the highest-occurrence demanded `physical`
classes the reviewed library cannot carry were not part categories at all -- board outlines and
shapes (`snowman-shaped-board` 38, `star-shaped-board-outline` 8), mechanical holes (`hang-hole`
24), the board's own copper (`copper-heatsink-area` 12, `thermal-via-copper-pour` 8), castellations
(`castellated-gpio` 24), package descriptors (`qfn-56-package` 38, `lqfp-48-package` 11), region
labels (`base-led-section` 6), net-level connections (`vbus-connection` 8), dielectric properties
(`isolation-barrier` 4), switchable networks named by their function
(`switchable-can-termination` 6) and absences (`no-microcontroller` 6). Each one reached the parts
stage as a BOM demand no placed part can satisfy (`E_PHYSICAL_REALIZATION`), so the run failed
after burning its repair rounds on a reading the compiler could make itself.

The two dispositions this file pins:

* a class that states a board fact or an absence is RETYPED to the row the typed vocabulary has
  for it (`fabrication`, `negative`) by both normalizers the pipeline runs -- the pre-diagnosis
  completion and the commit-time source normalizer, which must agree;
* a class that is neither a part class nor a board fact is REFUSED with the named code
  `intent_obligation_class_unrealizable`, whose reading and repair now name the exact reading
  (a package descriptor, a region label, a connection, ...) instead of one generic sentence.

Every rule must hold in BOTH directions: a genuine part class -- a reviewed one, a longer spelling
of one, a category the library has never covered, or a class carried by a reviewed record that
merely contains one of the same words (`thermal-pad` on the AONR21357, `mounting-hole`) -- is
untouched, and the off-board power source keeps the refusal it has always had.
"""

from __future__ import annotations

import pytest

from kicraft.design.part_identity import (
    absent_class_of_negation,
    board_fabrication_feature,
    negation_names_a_reviewed_class,
)
from kicraft.design.stage_semantics import complete_intent_classification, diagnose_stage
from kicraft.server.stage_contracts import normalize_non_part_obligations

BRIEF = "A board that asks for these classes."

#: (demanded class, the fabrication feature the row must carry). The feature is the class's own
#: canonical spelling: it is the board's, so no part classifier reads it again, and keeping the
#: writer's wording preserves what the brief asked for.
BOARD_FACTS = [
    # board outline and shape spellings
    ("snowman-shaped-board", "snowman-shaped-board"),
    ("round-pcb", "round-pcb"),
    ("circular-pcb", "circular-pcb"),
    ("star-shaped-board-outline", "star-shaped-board-outline"),
    ("rounded-rectangular-pcb-outline", "rounded-rectangular-pcb-outline"),
    ("chamfered-rectangular-pcb", "chamfered-rectangular-pcb"),
    ("arduino-uno-shield-outline", "arduino-uno-shield-outline"),
    # a mechanical hole the board carries
    ("hang-hole", "hang-hole"),
    # the board's own copper
    ("copper-heatsink-area", "copper-heatsink-area"),
    ("heatsink-copper-area", "heatsink-copper-area"),
    ("pcb-copper-heatsink-area", "pcb-copper-heatsink-area"),
    ("thermal-via-copper-pour", "thermal-via-copper-pour"),
    ("thermal-via", "thermal-via"),
    # the board's own edge plating
    ("castellated-gpio", "castellated-gpio"),
    ("castellated-gpio-pads", "castellated-gpio-pads"),
]

#: (demanded class, the class it forbids).
ABSENCES = [
    ("no-microcontroller", "microcontroller"),
    ("no-edge-connector", "edge-connector"),
    ("no-edge-connectors", "edge-connector"),
    ("non-isolated-dc-dc-converter", "isolated-dc-dc-converter"),
]

#: Part classes no rule may touch, with the reason each one is a part.
REAL_PART_CLASSES = [
    # a reviewed record carries each of these two, one word included in the board-fact vocabulary
    "mounting-hole",  # kicad-mounting-hole-3.2mm-m3
    "thermal-pad",  # the AONR21357's own land
    # a reviewed class
    "pin-header",
    "header",
    "screw-terminal",
    "coin-cell-holder",
    "battery-connector",
    # a class only a registered lowerer builds
    "resistor-ladder",
    # a genuine new category the reviewed library does not cover yet
    "gps-module",
    "air-quality-sensor",
    # a small, honest coverage gap
    "jumper",
    # a class that NAMES A PACKAGE beside its device is still a part class
    "dip-switch",
    "soic-8-op-amp",
    # a class that merely CONTAINS a shape word is not an outline
    "heart-rate-sensor-pcb",
    "gear-motor-board",
    "led-ring-pcb",
]

#: (demanded class, the reading the refusal must name, the repair it must state).
REFUSALS = [
    ("qfn-56-package", "package or footprint descriptor", "package"),
    ("lqfp-48-package", "package or footprint descriptor", "package"),
    ("lqfp-48", "package or footprint descriptor", "package"),
    ("base-led-section", "region or section label", "label"),
    ("warm-white-leds-in-body-section", "region or section label", "label"),
    ("vbus-connection", "net-level connection", "wiring"),
    ("cc-signal-connection", "net-level connection", "wiring"),
    ("isolation-barrier", "dielectric or clearance property", "constraint"),
    ("galvanic-isolation", "dielectric or clearance property", "constraint"),
    ("switchable-can-termination", "network named by its function", "resistor"),
    ("switchable-termination-network", "network named by its function", "resistor"),
    ("i2c-interface", "interface, bus or board format", "constraint"),
    ("arduino-uno-format-board", "interface, bus or board format", "constraint"),
]


def _physical(component_class: str, index: int = 0) -> dict:
    return {
        "kind": "physical",
        "original_obligation_id": f"obligation_{index}",
        "component_class": component_class,
    }


def _codes(candidate: dict) -> list[str]:
    return [
        finding.code
        for finding in diagnose_stage("intent", brief=BRIEF, upstream_state={}, candidate=candidate)
    ]


@pytest.mark.parametrize("demanded,feature", BOARD_FACTS)
def test_a_board_fact_is_retyped_to_fabrication_by_both_normalizers(demanded, feature):
    """An outline, hole, castellation or copper area is the board's, so it is never a part demand."""
    candidate = {"obligations": [_physical(demanded)]}
    expected = {
        "kind": "fabrication",
        "original_obligation_id": "obligation_0",
        "feature": feature,
    }

    assert board_fabrication_feature(demanded) == feature
    assert complete_intent_classification(BRIEF, candidate)["obligations"] == [expected]
    # The commit-time normalizer must make the same reading, or the row that is committed would
    # disagree with the candidate the writer was diagnosed against.
    assert normalize_non_part_obligations(candidate)["obligations"] == [expected]
    # The input is untouched.
    assert candidate["obligations"] == [_physical(demanded)]


@pytest.mark.parametrize("demanded,absent", ABSENCES)
def test_an_absence_is_retyped_to_a_negative_row(demanded, absent):
    """No BOM line implements an absence, so the demand becomes the `negative` row it states."""
    candidate = {"obligations": [_physical(demanded)]}
    expected = {
        "kind": "negative",
        "original_obligation_id": "obligation_0",
        "absent_class": absent,
    }

    assert absent_class_of_negation(demanded) == absent
    assert complete_intent_classification(BRIEF, candidate)["obligations"] == [expected]
    assert normalize_non_part_obligations(candidate)["obligations"] == [expected]


@pytest.mark.parametrize("demanded", REAL_PART_CLASSES)
def test_a_real_part_class_is_never_retyped_or_refused(demanded):
    """The rules must not reject a part: not a reviewed one, not a new category, not a rename."""
    candidate = {"obligations": [_physical(demanded)]}

    assert board_fabrication_feature(demanded) is None
    assert absent_class_of_negation(demanded) is None
    assert complete_intent_classification(BRIEF, candidate)["obligations"] == [_physical(demanded)]
    assert normalize_non_part_obligations(candidate)["obligations"] == [_physical(demanded)]
    assert _codes(candidate) == []


@pytest.mark.parametrize("demanded,reading,repair", REFUSALS)
def test_a_non_part_wording_is_refused_with_its_own_reading_and_repair(demanded, reading, repair):
    """The refusal must name WHAT the wording is and the concrete move, not blame a spelling."""
    candidate = {"obligations": [_physical(demanded)]}
    findings = [
        finding
        for finding in diagnose_stage(
            "intent", brief=BRIEF, upstream_state={}, candidate=candidate
        )
        if finding.code == "intent_obligation_class_unrealizable"
    ]

    assert len(findings) == 1
    assert findings[0].severity == "repair_required"
    assert reading in findings[0].message
    evidence = " ".join(findings[0].evidence)
    assert "not a part class" in evidence
    assert repair in evidence
    # It is refused, never silently retyped onto another class.
    assert complete_intent_classification(BRIEF, candidate)["obligations"] == [
        _physical(demanded)
    ]


def test_a_negation_of_an_unknown_class_is_refused_not_silently_converted():
    """`not-gate-ic` reads as a negation, but "gate-ic" is no class: a NOT gate is a part.

    The pipeline converts only an absence of a class it can read; anything else is the writer's
    to restate, and the refusal carries both repairs.
    """
    demanded = "not-gate-ic"
    candidate = {"obligations": [_physical(demanded)]}

    assert absent_class_of_negation(demanded) == "gate-ic"
    assert not negation_names_a_reviewed_class("gate-ic")
    assert complete_intent_classification(BRIEF, candidate)["obligations"] == [
        _physical(demanded)
    ]
    assert normalize_non_part_obligations(candidate)["obligations"] == [_physical(demanded)]
    findings = [
        finding
        for finding in diagnose_stage(
            "intent", brief=BRIEF, upstream_state={}, candidate=candidate
        )
        if finding.code == "intent_obligation_class_unrealizable"
    ]
    assert len(findings) == 1
    assert "restate the class plainly" in " ".join(findings[0].evidence)


def test_a_bare_package_is_refused_but_a_package_named_beside_a_device_is_not():
    """`lqfp-48` is the same row as `lqfp-48-package`; `soic-8-op-amp` names a part."""
    from kicraft.design.part_identity import package_descriptor_class

    for package_only in ("lqfp-48", "qfn-56", "to-220", "sot23-5", "soic-8"):
        assert package_descriptor_class(package_only), package_only
        assert _codes({"obligations": [_physical(package_only)]}) == [
            "intent_obligation_class_unrealizable"
        ], package_only

    for named_part in ("soic-8-op-amp", "dip-switch", "to-can", "pin-header", "crystal"):
        assert not package_descriptor_class(named_part), named_part


def test_a_castellated_part_spelling_keeps_its_reviewed_rename():
    """`castellated-pin-header` carries the same word as an edge fact and is still a part class.

    It is a longer spelling of the reviewed `pin-header`, so the spelling guard keeps the class
    out of the board-fact reading and the existing rename still applies.
    """
    from kicraft.design.stage_semantics import complete_class_spellings

    demanded = "castellated-pin-header"
    candidate = {"obligations": [_physical(demanded)], "assumptions": []}

    assert board_fabrication_feature(demanded) is None
    assert normalize_non_part_obligations(candidate)["obligations"] == [_physical(demanded)]
    renamed = complete_class_spellings(candidate)["obligations"][0]["component_class"]
    assert renamed in {"pin-header", "header"}
    evidence = " ".join(
        item
        for finding in diagnose_stage(
            "intent", brief=BRIEF, upstream_state={}, candidate=candidate
        )
        if finding.code == "intent_obligation_class_unrealizable"
        for item in finding.evidence
    )
    assert "not a part class" not in evidence


def test_a_negation_inside_a_larger_demand_is_refused_not_retyped():
    """`passive-rc-filter-no-microcontroller` is a function AND an absence: the writer splits it."""
    demanded = "passive-rc-filter-no-microcontroller"
    candidate = {"obligations": [_physical(demanded)]}

    assert absent_class_of_negation(demanded) is None
    assert complete_intent_classification(BRIEF, candidate)["obligations"] == [
        _physical(demanded)
    ]
    findings = [
        finding
        for finding in diagnose_stage(
            "intent", brief=BRIEF, upstream_state={}, candidate=candidate
        )
        if finding.code == "intent_obligation_class_unrealizable"
    ]
    assert len(findings) == 1
    assert "split the row" in " ".join(findings[0].evidence)


def test_the_off_board_cell_keeps_its_holder_distinction():
    """`coin-cell-battery` stays the refused off-board source; the holder stays a placed part."""
    battery = {"obligations": [_physical("coin-cell-battery")]}
    holder = {"obligations": [_physical("coin-cell-holder")]}

    # The cell is neither retyped nor renamed: the board carries the holder, and the honest
    # repair is the writer's to make (demand the mate, or keep the source in the goal).
    assert board_fabrication_feature("coin-cell-battery") is None
    assert absent_class_of_negation("coin-cell-battery") is None
    assert complete_intent_classification(BRIEF, battery)["obligations"] == [
        _physical("coin-cell-battery")
    ]
    findings = [
        finding
        for finding in diagnose_stage(
            "intent", brief=BRIEF, upstream_state={}, candidate=battery
        )
        if finding.code == "intent_obligation_class_unrealizable"
    ]
    assert len(findings) == 1
    assert "off-board power source" in " ".join(findings[0].evidence)
    assert "battery" in " ".join(findings[0].evidence)

    # The mate the board does place is a realizable class and raises nothing.
    assert _codes(holder) == []
    assert complete_intent_classification(BRIEF, holder)["obligations"] == [
        _physical("coin-cell-holder")
    ]


def test_a_mixed_candidate_keeps_the_two_normalizers_in_step():
    """The completion and the commit-time normalizer read every row the same way."""
    candidate = {
        "obligations": [
            _physical("snowman-shaped-board", 0),
            _physical("no-microcontroller", 1),
            _physical("qfn-56-package", 2),
            _physical("pin-header", 3),
        ]
    }
    expected = [
        {
            "kind": "fabrication",
            "original_obligation_id": "obligation_0",
            "feature": "snowman-shaped-board",
        },
        {
            "kind": "negative",
            "original_obligation_id": "obligation_1",
            "absent_class": "microcontroller",
        },
        _physical("qfn-56-package", 2),
        _physical("pin-header", 3),
    ]

    assert complete_intent_classification(BRIEF, candidate)["obligations"] == expected
    assert normalize_non_part_obligations(candidate)["obligations"] == expected
