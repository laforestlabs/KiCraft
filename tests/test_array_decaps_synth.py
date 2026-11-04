"""Tests for the synthesis-stage per-LED-decap thinning rule."""
from __future__ import annotations


import pytest
from kicraft.design.models import (
    BOM,
    Architecture,
    CircuitRequirement,
    ArraySpec,
    BomPart,
    NetConnection,
    PinEndpoint,
    RecipeOwnershipManifest,
    RecipeSelection,
    Sheet,
)
from kicraft.design.synthesis.array_decaps import (
    drop_decap_only_arrays,
    normalize_array_decaps,
    isolate_array_sheets,
    isolate_opposite_edge_connectors,
)


def _led(ref: str) -> BomPart:
    return BomPart(ref=ref, value="WS2812B", symbol="LED:WS2812B",
                   footprint="LED:WS2812B-1313", sheet="LED ARRAY")


def _cap(ref: str) -> BomPart:
    return BomPart(ref=ref, value="100nF", symbol="Device:C",
                   footprint="Capacitor_SMD:C_0603_1608Metric", sheet="LED ARRAY")


def _build_bom(n_leds: int, n_caps: int) -> BOM:
    parts = [_led(f"D{i}") for i in range(1, n_leds + 1)]
    parts += [_cap(f"C{i}") for i in range(1, n_caps + 1)]
    # daisy-chain data nets keep LEDs OUT of the decap predicate
    conns = []
    for i in range(1, n_leds + 1):
        conns.append(NetConnection(net_name=f"DATA{i}", sheet="LED ARRAY",
                                    endpoints=[PinEndpoint(ref=f"D{i}", pin="2"),
                                               PinEndpoint(ref=f"D{i}", pin="4")]))
    # every part hangs on the global +5V / GND rails
    p5 = [PinEndpoint(ref=f"D{i}", pin="1") for i in range(1, n_leds + 1)]
    gnd = [PinEndpoint(ref=f"D{i}", pin="3") for i in range(1, n_leds + 1)]
    for i in range(1, n_caps + 1):
        p5.append(PinEndpoint(ref=f"C{i}", pin="1"))
        gnd.append(PinEndpoint(ref=f"C{i}", pin="2"))
    conns.append(NetConnection(net_name="+5V", sheet="LED ARRAY", endpoints=p5))
    conns.append(NetConnection(net_name="GND", sheet="LED ARRAY", endpoints=gnd))
    arrays = [ArraySpec(refs=[f"D{i}" for i in range(1, n_leds + 1)],
                        rows=1, cols=n_leds)]
    return BOM(parts=parts, connections=conns, arrays=arrays)


def test_low_current_array_thinned_to_bulk() -> None:
    bom = _build_bom(n_leds=4, n_caps=4)  # 4*60 = 240 mA < 3000 -> thin
    dropped = normalize_array_decaps(bom)
    assert dropped == ["C3", "C4"], "keep the first 2 caps, drop the rest"
    remaining_caps = [p.ref for p in bom.parts if p.ref.startswith("C")]
    assert remaining_caps == ["C1", "C2"]
    # dropped caps fully scrubbed from the net list -> BOM still re-validates
    BOM.model_validate(bom.model_dump())
    cap_endpoints = {ep.ref for c in bom.connections for ep in c.endpoints
                     if ep.ref.startswith("C")}
    assert cap_endpoints == {"C1", "C2"}
    assert any("thinned" in a for a in bom.assumptions), "decision recorded"


def test_high_current_array_keeps_per_led_decaps() -> None:
    bom = _build_bom(n_leds=51, n_caps=51)  # 51*60 = 3060 mA >= 3000 -> keep all
    dropped = normalize_array_decaps(bom)
    assert dropped == []
    assert len([p for p in bom.parts if p.ref.startswith("C")]) == 51
    assert bom.assumptions == []


def test_threshold_boundary_3a() -> None:
    # 50 LEDs = 3000 mA is exactly the >= threshold -> keep all per-LED.
    keep = _build_bom(n_leds=50, n_caps=50)
    assert normalize_array_decaps(keep) == []
    assert len([p for p in keep.parts if p.ref.startswith("C")]) == 50
    # 49 LEDs = 2940 mA < 3000 -> thin to 2 bulk caps.
    thin = _build_bom(n_leds=49, n_caps=49)
    assert normalize_array_decaps(thin) == [f"C{i}" for i in range(3, 50)]
    assert [p.ref for p in thin.parts if p.ref.startswith("C")] == ["C1", "C2"]


def test_no_decaps_no_change() -> None:
    bom = _build_bom(n_leds=4, n_caps=0)  # bulk-only board (no per-LED caps)
    assert normalize_array_decaps(bom) == []
    assert len(bom.parts) == 4


def test_idempotent() -> None:
    bom = _build_bom(n_leds=4, n_caps=4)
    normalize_array_decaps(bom)
    again = normalize_array_decaps(bom)  # already at keep=2 -> nothing more
    assert again == []


def test_signal_resistor_not_dropped() -> None:
    # a series DATA resistor (signal nets) is NOT a decap and must survive
    bom = _build_bom(n_leds=4, n_caps=4)
    bom.parts.append(BomPart(ref="R1", value="330", symbol="Device:R",
                             footprint="Resistor_SMD:R_0603_1608Metric",
                             sheet="LED ARRAY"))
    bom.connections.append(NetConnection(
        net_name="DATA_SER", sheet="LED ARRAY",
        endpoints=[PinEndpoint(ref="R1", pin="1"), PinEndpoint(ref="R1", pin="2")]))
    dropped = normalize_array_decaps(bom)
    assert "R1" not in dropped


def _add_cap_array(bom: BOM, n_caps: int) -> None:
    """Declare C1..Cn as their own grid -- the spurious decap-array the BOM stage
    sometimes emits alongside the real LED array (KC-NZXXEE)."""
    bom.arrays.append(ArraySpec(refs=[f"C{i}" for i in range(1, n_caps + 1)],
                                rows=1, cols=n_caps))


def test_decap_only_array_dropped() -> None:
    # LED array + a spurious cap array over the per-LED decaps on the same sheet.
    bom = _build_bom(n_leds=10, n_caps=10)
    _add_cap_array(bom, 10)
    assert len(bom.arrays) == 2
    dropped = drop_decap_only_arrays(bom)
    assert dropped == [[f"C{i}" for i in range(1, 11)]], "the cap array is dropped"
    assert [a.refs[0] for a in bom.arrays] == ["D1"], "only the LED array survives"
    assert any("decoupling caps" in a for a in bom.assumptions), "decision recorded"
    # the caps themselves are NOT removed -- they become array companions
    assert len([p for p in bom.parts if p.ref.startswith("C")]) == 10


def test_real_second_array_not_dropped() -> None:
    # A second array of SIGNAL parts (series data resistors) is a genuine grid.
    bom = _build_bom(n_leds=6, n_caps=0)
    bom.parts += [BomPart(ref=f"R{i}", value="330", symbol="Device:R",
                          footprint="Resistor_SMD:R_0603_1608Metric",
                          sheet="LED ARRAY") for i in range(1, 7)]
    for i in range(1, 7):  # each resistor carries a non-power DATA net -> not a decap
        bom.connections.append(NetConnection(
            net_name=f"SER{i}", sheet="LED ARRAY",
            endpoints=[PinEndpoint(ref=f"R{i}", pin="1"),
                       PinEndpoint(ref=f"R{i}", pin="2")]))
    bom.arrays.append(ArraySpec(refs=[f"R{i}" for i in range(1, 7)], rows=1, cols=6))
    assert drop_decap_only_arrays(bom) == []
    assert len(bom.arrays) == 2


def test_sole_decap_array_kept() -> None:
    # A decap-only array with NO sibling real array has nothing to companion --
    # leave it alone rather than strip its grid hint. (Needs >=2 specs to even be
    # considered, so pair it with a second decap array on a different sheet.)
    parts = [_cap(f"C{i}") for i in range(1, 9)]  # all on sheet "LED ARRAY"
    conns = [
        NetConnection(net_name="+5V", sheet="LED ARRAY",
                      endpoints=[PinEndpoint(ref=f"C{i}", pin="1") for i in range(1, 9)]),
        NetConnection(net_name="GND", sheet="LED ARRAY",
                      endpoints=[PinEndpoint(ref=f"C{i}", pin="2") for i in range(1, 9)]),
    ]
    arrays = [ArraySpec(refs=[f"C{i}" for i in range(1, 5)], rows=2, cols=2),
              ArraySpec(refs=[f"C{i}" for i in range(5, 9)], rows=2, cols=2)]
    bom = BOM(parts=parts, connections=conns, arrays=arrays)
    # both specs are decap-only and neither is "real", so nothing is a companion
    assert drop_decap_only_arrays(bom) == []
    assert len(bom.arrays) == 2


def test_single_array_is_noop() -> None:
    bom = _build_bom(n_leds=6, n_caps=6)  # only the LED array spec present
    assert drop_decap_only_arrays(bom) == []
    assert len(bom.arrays) == 1


def test_power_resistor_is_not_a_decap_or_thinned() -> None:
    bom = _build_bom(n_leds=4, n_caps=2)
    bom.parts.append(BomPart(ref="R1", value="1k", symbol="Device:R",
                             footprint="Resistor_SMD:R_0603_1608Metric",
                             sheet="LED ARRAY"))
    bom.connections[-2].endpoints.append(PinEndpoint(ref="R1", pin="1"))
    bom.connections[-1].endpoints.append(PinEndpoint(ref="R1", pin="2"))
    assert normalize_array_decaps(bom) == []
    assert any(p.ref == "R1" for p in bom.parts)
    assert sum(ep.ref == "R1" for c in bom.connections for ep in c.endpoints) == 2


def test_distinct_and_sourced_caps_are_preserved() -> None:
    bom = _build_bom(n_leds=4, n_caps=4)
    bom.parts[-2].value = "100uF"
    bom.parts[-1].mpn = "CL10A104KB8NNNC"
    assert normalize_array_decaps(bom) == []
    assert {p.ref for p in bom.parts if p.ref.startswith("C")} == {"C1", "C2", "C3", "C4"}


def test_non_led_array_does_not_trigger_led_thinning() -> None:
    bom = _build_bom(n_leds=4, n_caps=4)
    for part in bom.parts:
        if part.ref.startswith("D"):
            part.symbol = "Device:R"
    assert normalize_array_decaps(bom) == []
    assert len(bom.parts) == 8


def test_array_sheet_split_refuses_to_split_a_requirement() -> None:
    bom = _build_bom(n_leds=4, n_caps=0)
    bom.parts[0].lowering_requirement_id = "shared"
    bom.parts.append(BomPart(
        ref="J1", value="header", symbol="Connector:Conn_01x02",
        footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        sheet="LED ARRAY", lowering_requirement_id="shared",
    ))
    arch = Architecture(sheets=[Sheet(name="LED ARRAY", stem="LED_ARRAY", function="test")],
                        power_nets=[], inter_sheet_nets=[],
                        requirements=[CircuitRequirement(
                            id="shared", sheet="LED ARRAY", role="connector", family="pin-header")])
    before_bom, before_arch = bom.model_dump(), arch.model_dump()
    with pytest.raises(ValueError, match="cannot split requirement"):
        isolate_array_sheets(bom, arch)
    assert bom.model_dump() == before_bom
    assert arch.model_dump() == before_arch


def test_opposite_edge_split_moves_requirement_with_owned_connector() -> None:
    parts = [
        BomPart(ref=f"J{i}", value="header", symbol="Connector:Conn_01x02",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                sheet="MAIN", lowering_requirement_id="header" if i == 1 else None)
        for i in (1, 2)
    ]
    bom = BOM(parts=parts, component_zones={"J1": {"edge": "top"}, "J2": {"edge": "bottom"}})
    arch = Architecture(sheets=[Sheet(name="MAIN", stem="MAIN", function="test")],
                        power_nets=[], inter_sheet_nets=[],
                        requirements=[CircuitRequirement(
                            id="header", sheet="MAIN", role="connector", family="pin-header")])
    assert isolate_opposite_edge_connectors(bom, arch, verbose=False) == ["J1"]
    assert arch.requirements[0].sheet == bom.parts[0].sheet != "MAIN"


def test_opposite_edge_split_refuses_shared_recipe_without_mutation() -> None:
    parts = [
        BomPart(ref=f"J{i}", value="header", symbol="Connector:Conn_01x02",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                sheet="MAIN")
        for i in (1, 2)
    ]
    bom = BOM(
        parts=parts,
        component_zones={"J1": {"edge": "top"}, "J2": {"edge": "bottom"}},
        recipe_ownership=[RecipeOwnershipManifest(
            recipe="header@1", instance="headers", refs=["J1", "J2"])],
    )
    arch = Architecture(
        sheets=[Sheet(name="MAIN", stem="MAIN", function="test")],
        power_nets=[], inter_sheet_nets=[],
        recipe_selections=[RecipeSelection(recipe="header@1", instance="headers",
                                           sheets={"interface": "MAIN"})],
    )
    before_bom, before_arch = bom.model_dump(), arch.model_dump()
    with pytest.raises(ValueError, match="cannot split recipe"):
        isolate_opposite_edge_connectors(bom, arch, verbose=False)
    assert bom.model_dump() == before_bom
    assert arch.model_dump() == before_arch
