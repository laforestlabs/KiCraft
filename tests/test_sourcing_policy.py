"""What to do about a part that is real but out of stock (owner's rule, 2026-09-26).

*Make a valid board at the end, even if one or two of the parts are out of stock.* A part the brief
names by its own order code is the owner's — keep it (ask them when someone is watching, keep it
when the run is unattended). A part the pipeline chose for a class the brief names is the
pipeline's to replace: an in-stock carrier of the **same family and same package**, so no pin,
footprint or wire the design already states moves. A different package is a different board and is
named, never applied.

Hermetic: the reviewed inventory is the real one, and stock is a lambda.
"""
from __future__ import annotations

from types import SimpleNamespace

from kicraft.design import sourcing_policy
from kicraft.design.part_identity import reviewed_part


class _Inventory:
    """Stands in for the reviewed inventory so a case can be built without one existing."""

    def __init__(self, *rows):
        self.rows = rows


def _record(identity, family, footprint, contacts=("1", "2"), lcsc="C1"):
    return SimpleNamespace(
        identity=identity, family=family, footprint=footprint, contacts=tuple(contacts),
        lcsc=lcsc, is_portable_candidate=True, bundle=identity, package=footprint,
    )


def _decide(monkeypatch, *, identity, family, named=(), brief="", stock_ids=(), inventory=()):
    monkeypatch.setattr(
        "kicraft.design.part_identity.reviewed_inventory", lambda: tuple(inventory)
    )
    monkeypatch.setattr(
        "kicraft.design.part_identity.reviewed_part",
        lambda name: next(
            (row for row in inventory if row.identity == str(name).casefold()), None
        ),
    )
    return sourcing_policy.decide(
        identity=identity,
        mpn=identity.upper(),
        family=family,
        shortfall="out of stock at the lcsc.com retail storefront (0 available, min buy 5)",
        named_parts=named,
        brief=brief,
        in_stock=lambda row: getattr(row, "identity", "") in stock_ids,
    )


def test_a_part_the_brief_names_is_kept_and_a_person_is_asked(monkeypatch):
    """The owner may hold stock or have a second source, so the choice is theirs."""
    decision = _decide(
        monkeypatch,
        identity="ch32v003j4m6",
        family="ch32v003",
        named=("CH32V003J4M6",),
        inventory=(_record("ch32v003j4m6", "ch32v003", "pkg:SOP-8"),),
    )
    assert decision.action == sourcing_policy.KEEP
    assert decision.named_in_brief is True and decision.asked is True
    assert "the brief names this part, so it is kept" in decision.reason


def test_a_family_name_is_not_the_part_number(monkeypatch):
    """'a CH32V003 development board' names a family; the SOP-8 order code is the pipeline's pick."""
    decision = _decide(
        monkeypatch,
        identity="ch32v003j4m6",
        family="ch32v003",
        named=("CH32V003", "JST-XH"),
        brief="A CH32V003 development board with a barrel jack.",
        inventory=(_record("ch32v003j4m6", "ch32v003", "pkg:SOP-8"),),
    )
    assert decision.named_in_brief is False and decision.asked is False
    assert decision.action == sourcing_policy.KEEP  # nothing else shares the package


def test_an_unnamed_part_is_swapped_for_an_in_stock_carrier_of_the_same_package(monkeypatch):
    """Same family keeps the device, same package keeps every pin the design already binds."""
    current = _record("led-a", "led-0603", "vendor-a:LED_0603_1608Metric", lcsc="C1")
    sibling = _record("led-b", "led-0603", "vendor-b:LED_0603_1608Metric", lcsc="C2")
    decision = _decide(
        monkeypatch,
        identity="led-a",
        family="led-0603",
        inventory=(current, sibling),
        stock_ids=("led-b",),
    )
    assert decision.action == sourcing_policy.SWAP
    assert decision.alternative is sibling
    assert "the same package, in stock — takes its place (defaulted)" in decision.reason


def test_a_different_package_is_named_but_never_swapped_silently(monkeypatch):
    """A SOP-16 for a SOP-8 moves the pins: that is a design decision, and the reason says so."""
    current = _record("mcu-a", "mcu", "vendor:SOP-8_L4.9-W3.8", lcsc="C1")
    wider = _record("mcu-b", "mcu", "vendor:SOIC-16_L9.9-W3.9", contacts=tuple("1234567890abcdef"), lcsc="C2")
    decision = _decide(
        monkeypatch,
        identity="mcu-a",
        family="mcu",
        inventory=(current, wider),
        stock_ids=("mcu-b",),
    )
    assert decision.action == sourcing_policy.KEEP and decision.alternative is None
    assert "different packages (mcu-b)" in decision.reason
    assert "the pins and is a design decision" in decision.reason


def test_the_real_led_carriers_are_not_swappable_across_packages():
    """The reviewed 0603 LED and the vendored 0805 one share a class but not a package."""
    small = reviewed_part("kicad-led-0603") or reviewed_part("ltst-c190kgkt")
    big = reviewed_part("e6c0805wway1uda(1.1t m)")
    assert small is not None and big is not None
    assert sourcing_policy.same_package(small, big) is False
    assert sourcing_policy.same_package(small, small) is True


def test_the_question_keeps_the_part_first_and_names_what_is_in_stock(monkeypatch):
    """The ask: build with the part you named, or take an in-stock variation."""
    monkeypatch.setattr(
        "kicraft.design.part_identity.reviewed_inventory",
        lambda: (
            _record("led-a", "led-0603", "a:LED_0603_1608Metric", lcsc="C1"),
            _record("led-b", "led-0603", "b:LED_0603_1608Metric", lcsc="C2"),
            _record("mcu-b", "led-0603", "c:SOIC-16", contacts=("1", "2", "3"), lcsc="C3"),
        ),
    )
    monkeypatch.setattr(
        "kicraft.design.part_identity.reviewed_part",
        lambda name: _record(str(name), "led-0603", "a:LED_0603_1608Metric", lcsc="C1"),
    )
    monkeypatch.setattr(
        sourcing_policy, "_reviewed_record_for",
        lambda row: _record("led-a", "led-0603", "a:LED_0603_1608Metric", lcsc="C1"),
    )
    def stock(cid, _record=None):
        return cid != "C1", "out of stock at retail (0 available)"

    questions = sourcing_policy.stock_questions(
        [{"ref": "D1", "mpn": "LED-A", "symbol": "a:LED", "sourcing_note": "LCSC C1"}],
        named_parts=("LED-A",),
        brief="",
        stock=stock,
    )

    assert len(questions) == 1
    question = questions[0]
    assert question["blocking"] is True
    assert question["options"][0] == "Keep LED-A as designed"
    assert any("led-b" in option and "in stock" in option for option in question["options"])
    assert any("different package" in option for option in question["options"])


def test_no_question_for_a_part_the_brief_does_not_name():
    asked = sourcing_policy.stock_questions(
        [{"ref": "D1", "mpn": "LED-A", "symbol": "a:LED"}],
        named_parts=("SOMETHING-ELSE",),
        brief="A board.",
        stock=lambda cid, _record=None: (False, "dry"),
    )
    assert asked == []


def test_the_parts_stage_asks_with_the_briefs_own_named_parts(monkeypatch):
    """The runtime hook: the question is built from the committed intent's named parts and the
    brief, and comes back bounded the way the stage parks questions."""
    from kicraft.server import stage_runtime

    seen: dict = {}

    def fake_stock_questions(parts, *, named_parts=(), brief="", **_kwargs):
        seen["parts"] = list(parts)
        seen["named_parts"] = tuple(named_parts)
        seen["brief"] = brief
        return [
            {
                "text": "LED-A is out of stock. Build with it, or change it?",
                "blocking": True,
                "options": ["Keep LED-A as designed", "Use LED-B (in stock)"],
            }
        ]

    monkeypatch.setattr(sourcing_policy, "stock_questions", fake_stock_questions)
    candidate = {"parts": [{"ref": "D1", "mpn": "LED-A"}], "connections": []}
    questions = stage_runtime._bom_stock_questions(
        candidate, {"intent": {"named_parts": ["LED-A"]}}, "A board with a power LED."
    )

    assert seen["named_parts"] == ("LED-A",)
    assert seen["brief"] == "A board with a power LED."
    assert [q["text"] for q in questions] == ["LED-A is out of stock. Build with it, or change it?"]
    assert questions[0]["blocking"] is True and questions[0]["options"][0].startswith("Keep")
    # Fail-soft: no parts, or an auditor that blows up, never blocks a draft.
    assert stage_runtime._bom_stock_questions({"connections": []}, {}, "") == []

    def boom(*_args, **_kwargs):
        raise RuntimeError("storefront down")

    monkeypatch.setattr(sourcing_policy, "stock_questions", boom)
    assert stage_runtime._bom_stock_questions(candidate, {}, "") == []
