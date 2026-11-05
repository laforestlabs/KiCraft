"""Manufacturer land-pattern and exact identity for the 2.2 uH buck inductor."""

import pcbnew

from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.footprint_library import load_footprint
from kicraft.design.synthesis.symbol_pinout import lookup_pins


SYMBOL = "cjiang-fxl0630-2r2-m:FXL0630-2R2-M"
FOOTPRINT = "cjiang-fxl0630-2r2-m:FXL0630_Cjiang_Recommended"


def test_exact_inductor_symbol_and_manufacturer_recommended_land_pattern():
    pins = lookup_pins(SYMBOL)["pins"]
    assert {pin["number"]: pin["electrical_type"] for pin in pins} == {
        "1": "passive", "2": "passive"
    }
    footprint, _ = load_footprint(pcbnew, "cjiang-fxl0630-2r2-m", "FXL0630_Cjiang_Recommended")
    pads = {
        pad.GetNumber(): (pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y),
                          pcbnew.ToMM(pad.GetSize().x), pcbnew.ToMM(pad.GetSize().y))
        for pad in footprint.Pads()
    }
    assert pads == {"1": (-3.025, 0.0, 2.35, 3.5), "2": (3.025, 0.0, 2.35, 3.5)}


def test_exact_inductor_identity_does_not_claim_old_fifteen_microhenry_bundle():
    record = physical_inventory_record(mpn="FXL0630-2R2-M", symbol=SYMBOL, footprint=FOOTPRINT)
    assert record is not None and record.lcsc == "C167218"
    assert physical_inventory_record(
        mpn="FXL0630-2R2-M", symbol="Device:L",
        footprint="inductor-15uh-3a:IND-SMD_L7.0-W6.6_FXL0630",
    ) is None
