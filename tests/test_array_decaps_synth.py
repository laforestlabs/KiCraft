"""Tests for the synthesis-stage per-LED-decap thinning rule."""
from __future__ import annotations

from kicraft.design.models import (
    BOM,
    ArraySpec,
    BomPart,
    NetConnection,
    PinEndpoint,
)
from kicraft.design.synthesis.array_decaps import normalize_array_decaps


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
    bom = _build_bom(n_leds=4, n_caps=4)  # 4*60 = 240 mA < 500 -> thin
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
    bom = _build_bom(n_leds=25, n_caps=25)  # 25*60 = 1500 mA >= 500 -> keep all
    dropped = normalize_array_decaps(bom)
    assert dropped == []
    assert len([p for p in bom.parts if p.ref.startswith("C")]) == 25
    assert bom.assumptions == []


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
    assert any(p.ref == "R1" for p in bom.parts)
