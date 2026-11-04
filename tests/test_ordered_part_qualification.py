"""Stock and matching pin numbers cannot qualify an incompatible ordered package."""
from copy import deepcopy

import pytest

from kicraft.design.models import BOM, BomPart, NetConnection, PinEndpoint
from kicraft.design.part_identity import physical_inventory_record
from kicraft.design.synthesis.kicad_pcb_stub import write_empty_pcb
from kicraft.eval.artifact_evidence import (
    _board_facts,
    _bom_parts_digest,
    _ordered_package_is_qualified,
    _reconcile_inventory,
    _state_facts,
)


def _native_facts(tmp_path, *, mpn=None, symbol="Device:C", footprint="Capacitor_SMD:C_0603_1608Metric", ref="C1"):
    pytest.importorskip("pcbnew")
    bom = BOM(
        parts=[BomPart(
            ref=ref,
            value="10uF" if mpn == "CL31B106KBHNNNE" else "100nF" if ref == "C1" else "4.7uH",
            symbol=symbol, footprint=footprint, mpn=mpn, sheet="MAIN",
        )],
        connections=[
            NetConnection(net_name=f"N{pin}", sheet="MAIN", endpoints=[PinEndpoint(ref=ref, pin=pin)])
            for pin in ("1", "2")
        ],
    )
    state = {"bom": bom.model_dump(mode="json")}
    path = write_empty_pcb(tmp_path, "PACKAGE_QUALIFICATION", bom)
    facts = _state_facts(state)
    assert _board_facts(path, facts) is not None
    return facts


def _receipt(facts, *, catalog_mpn="STANDARD-100NF-0603", package="0603", ref="C1"):
    return {
        "schema_version": 1,
        "parts_digest": _bom_parts_digest(facts["all_bom_parts"]),
        "parts": [{
            "ref": ref, "verdict": "pass", "catalog_mpn": catalog_mpn,
            "manufacturer": "Fixture",
            "exact_lcsc": {"CL31B106KBHNNNE": "C89632", "CL05A104KA5NNNC": "C100072"}.get(catalog_mpn, "C14663"),
            "package": package, "assembly_stock": 1000, "retail_stock": 1000,
        }],
    }


@pytest.mark.parametrize("package,qualified", [("0603", True), ("1206", False), ("", False), ("SMD", False)])
def test_generic_standard_chip_requires_the_ordered_standard_case(tmp_path, package, qualified):
    facts = _native_facts(tmp_path)
    facts["sourcing_validation"] = _receipt(facts, package=package)
    _reconcile_inventory(facts)
    assert bool(facts["part_inventory"]) is qualified
    assert facts["unqualified_source_refs"] == ([] if qualified else ["C1"])
    assert {row["footprint_pad"] for row in facts["pin_mapping"]} == {"C1.1", "C1.2"}


def test_explicit_unknown_mpn_cannot_fall_back_to_the_generic_case(tmp_path):
    facts = _native_facts(tmp_path, mpn="UNREVIEWED-100NF-0603")
    facts["sourcing_validation"] = _receipt(facts, catalog_mpn="UNREVIEWED-100NF-0603")
    _reconcile_inventory(facts)
    assert facts["part_inventory"] == []
    assert facts["unqualified_source_refs"] == ["C1"]


def test_reviewed_bom_part_requires_the_same_reviewed_catalog_order(tmp_path):
    facts = _native_facts(
        tmp_path, mpn="CL31B106KBHNNNE",
        footprint="Capacitor_SMD:C_1206_3216Metric",
    )
    intact = deepcopy(facts)
    intact["sourcing_validation"] = _receipt(intact, catalog_mpn="CL31B106KBHNNNE", package="1206")
    _reconcile_inventory(intact)
    assert intact["unqualified_source_refs"] == []
    assert intact["part_inventory"][0]["mpn"] == "CL31B106KBHNNNE"
    facts["sourcing_validation"] = _receipt(facts, catalog_mpn="CL05A104KA5NNNC", package="0402")
    _reconcile_inventory(facts)
    assert facts["part_inventory"] == []
    assert facts["unqualified_source_refs"] == ["C1"]


@pytest.mark.parametrize("defect", ["missing_row", "stale_digest", "nonpassing_row"])
def test_package_proof_does_not_rescue_an_unbound_stock_receipt(tmp_path, defect):
    facts = _native_facts(tmp_path)
    receipt = _receipt(facts)
    if defect == "missing_row":
        receipt["parts"] = []
    elif defect == "stale_digest":
        receipt["parts_digest"] = "stale"
    else:
        receipt["parts"][0]["verdict"] = "blocked"
    facts["sourcing_validation"] = receipt
    _reconcile_inventory(facts)
    assert facts["part_inventory"] == []
    assert facts["unqualified_source_refs"] == ["C1"]


def test_two_inductor_contacts_do_not_qualify_an_unreviewed_manufacturer_land(tmp_path):
    facts = _native_facts(
        tmp_path, ref="L1", symbol="Device:L",
        footprint="Inductor_SMD:L_6.3x6.3_H3",
    )
    facts["sourcing_validation"] = _receipt(
        facts, ref="L1", catalog_mpn="CKCSA4020-4.7uH/M", package="4x4mm",
    )
    _reconcile_inventory(facts)
    assert {row["footprint_pad"] for row in facts["pin_mapping"]} == {"L1.1", "L1.2"}
    assert facts["part_inventory"] == []
    assert facts["unqualified_source_refs"] == ["L1"]


def test_identical_printed_order_codes_do_not_merge_manufacturers():
    part = {
        "mpn": "USBLC6-2SC6",
        "symbol": "usblc6-2sc6:USBLC6-2SC6_C2687116",
        "footprint": "usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95-LS2.8-BL",
    }
    record = physical_inventory_record(**part)
    row = {"catalog_mpn": "USBLC6-2SC6", "exact_lcsc": "C2687116", "package": "SOT-23-6"}
    assert _ordered_package_is_qualified(part, record, row)
    row["exact_lcsc"] = "C7519"
    assert not _ordered_package_is_qualified(part, record, row)


@pytest.mark.parametrize(
    "mpn,code,symbol,footprint",
    [
        ("CL10A105KB8NNNC", "C15849", "Device:C", "Capacitor_SMD:C_0603_1608Metric"),
        ("PTFR0603B680RP9", "C351634", "Device:R", "Resistor_SMD:R_0603_1608Metric"),
        ("0603WAF1003T5E", "C25803", "Device:R", "Resistor_SMD:R_0603_1608Metric"),
        ("KT-0603R", "C2286", "Device:LED", "LED_SMD:LED_0603_1608Metric"),
    ],
)
@pytest.mark.parametrize("defect", [None, "wrong_code", "wrong_model", "wrong_package", "wrong_land"])
def test_selected_generic_part_requires_matching_catalog_order(mpn, code, symbol, footprint, defect):
    part = {"ref": "X1", "mpn": None, "symbol": symbol, "footprint": footprint}
    facts = {
        "bom_parts": [part],
        "all_bom_parts": [part],
        "board_footprints": {"X1": footprint},
        "board_pad_observations": [
            {"footprint_pad": "X1.1"}, {"footprint_pad": "X1.2"},
        ],
    }
    row = {
        "ref": "X1", "verdict": "pass", "catalog_mpn": mpn,
        "exact_lcsc": code, "package": "0603", "manufacturer": "catalog",
    }
    if defect == "wrong_code":
        row["exact_lcsc"] = "C999999999"
    elif defect == "wrong_model":
        row["catalog_mpn"] = "UNREVIEWED-ORDER"
    elif defect == "wrong_package":
        row["package"] = "0805"
    elif defect == "wrong_land":
        part["footprint"] = "Other:Land"
        facts["board_footprints"]["X1"] = "Other:Land"
    facts["sourcing_validation"] = {
        "schema_version": 1, "parts_digest": _bom_parts_digest([part]), "parts": [row],
    }
    _reconcile_inventory(facts)
    if defect is None:
        assert facts["unqualified_source_refs"] == []
        assert facts["part_inventory"][0]["rated_limits"]
        assert facts["part_identities"][0]["identity"] == mpn.casefold()
    else:
        assert facts["part_inventory"] == []
