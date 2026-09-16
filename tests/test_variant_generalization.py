"""Generalization cases: small reviewed variations of supported parameters,
connector counts, and interface compositions.

These are deliberately NOT part of the 34-brief corpus.  The corpus asks "does
the pipeline realize these specific designs"; these cases ask "does it keep
working when a supported knob moves", so a design that only works at one exact
value is visible as a failure here rather than as a silent corpus gap.
"""
from __future__ import annotations

import pytest

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.recipes import registered_recipes
from kicraft.design.recipes.models import RecipeSelection, ResolvedRecipeSelection
from kicraft.design.recipes.registry import get_recipe
from kicraft.design.recipes import expand_recipe
from kicraft.eval import artifact_evidence


def _ws2812_expansion(quantity: int):
    selection = RecipeSelection(
        recipe="ws2812-output@1", instance=f"px{quantity}", sheets={"interface": "A"},
        parameters={"quantity": quantity},
        port_bindings={"vdd": "+5V", "gnd": "GND", "data_in": "PIXEL_DATA"},
    )
    return expand_recipe(selection)


@pytest.mark.parametrize("quantity", [1, 2, 6, 24])
def test_pixel_chain_scales_to_every_supported_quantity(quantity):
    """A cascade must scale: N pixels chained, one no-connect at the end."""
    expansion = _ws2812_expansion(quantity)
    refs = [part.ref for part in expansion.parts if part.recipe_role == "led"]
    assert len(refs) == quantity
    assert {row.ref for row in expansion.no_connect_pins} == {refs[-1]}


@pytest.mark.parametrize(
    ("symbol", "footprint", "contacts", "mismatched"),
    [
        ("Connector_Generic:Conn_01x02", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", 2,
         "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"),
        ("Connector_Generic:Conn_01x05", "Connector_PinHeader_2.54mm:PinHeader_1x05_P2.54mm_Vertical", 5,
         "Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical"),
        ("Connector_Generic:Conn_02x04", "Connector_PinHeader_2.54mm:PinHeader_2x04_P2.54mm_Vertical", 8,
         "Connector_PinHeader_2.54mm:PinHeader_2x05_P2.54mm_Vertical"),
        ("Connector_Generic:Conn_02x10_Odd_Even", "Connector_PinHeader_2.54mm:PinHeader_2x10_P2.54mm_Vertical", 20,
         "Connector_PinHeader_2.54mm:PinHeader_2x09_P2.54mm_Vertical"),
    ],
)
def test_header_connector_count_variations_resolve_exactly(symbol, footprint, contacts, mismatched):
    """Each requested connector count must resolve to that count's own land pattern."""
    record = physical_inventory_record(mpn=None, symbol=symbol, footprint=footprint)
    assert record is not None and len(record.contacts) == contacts
    # A count mismatch is never silently accepted.
    assert physical_inventory_record(mpn=None, symbol=symbol, footprint=mismatched) is None


def test_three_same_part_addresses_are_distinct_on_one_bus():
    """Interface composition: three identical ADCs need three distinct straps."""
    definition = get_recipe("ads1115-i2c-adc@1")
    sheets = {role: "A" for role in definition.required_sheet_roles}
    nets = []
    for strap in ("gnd", "vdd", "sda"):
        selection = RecipeSelection(
            recipe=definition.recipe, instance=f"adc_{strap}", sheets=sheets,
            parameters={"address_strap": strap},
            port_bindings={
                "vdd": "+3V3", "gnd": "GND", "sda": "SDA", "scl": "SCL",
                "io0": "A0", "io1": "A1", "io2": "A2", "io3": "A3",
            },
            requirement_ids=["adc"],
        )
        expansion = expand_recipe(selection)
        nets.append(next(own.net for own in expansion.ownership.pins if own.pin == "1" and own.ref.startswith("U")))
    assert nets == ["GND", "+3V3", "SDA"]
    assert len(set(nets)) == 3


def test_class_counter_reports_varied_connector_and_part_counts():
    """Counts must follow the variation, including a zero for an absent class."""
    from collections import Counter

    parts = [
        {"ref": "J1", "symbol": "Connector_Generic:Conn_01x02",
         "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"},
        {"ref": "J2", "symbol": "Connector_Generic:Conn_01x05",
         "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x05_P2.54mm_Vertical"},
        {"ref": "U1", "symbol": "usb-c-16p:TYPE-C-31-M-12",
         "footprint": "usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1", "mpn": "TYPE-C-31-M-12"},
    ]
    counts: Counter[str] = Counter()
    for part in parts:
        counts.update(artifact_evidence._classify_part(part))
    assert counts["usb_c_receptacle"] == 1
    assert counts["microcontroller"] == 0
