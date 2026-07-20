"""Tests for the deterministic LED-array decap package backstop."""
from __future__ import annotations

from kicraft.design.models import (
    BOM,
    ArraySpec,
    BomPart,
    NetConnection,
    PinEndpoint,
)
from kicraft.design.synthesis.array_decap_footprints import (
    _led_cap_package,
    downsize_array_decap_footprints,
)

C0603 = "Capacitor_SMD:C_0603_1608Metric"
C0402 = "Capacitor_SMD:C_0402_1005Metric"
C0805 = "Capacitor_SMD:C_0805_2012Metric"

LED_SMALL = "ws2812b-1313:LED-SMD_4P-L1.3-W1.3-P0.80_WS2812E-1313"   # 1.3 mm -> 0402
LED_LARGE = "ws2812b:LED-SMD_4P-L5.0-W5.0-TL_WS2812B-B"              # 5.0 mm, no token -> default


def _build_bom(*, n_leds: int, cap_fp: str, led_fp: str, n_caps: int | None = None,
               cap_mpn: str | None = None) -> BOM:
    n_caps = n_leds if n_caps is None else n_caps
    parts = [BomPart(ref=f"D{i}", value="WS2812B", symbol="LED:WS2812B",
                     footprint=led_fp, sheet="LED ARRAY") for i in range(1, n_leds + 1)]
    parts += [BomPart(ref=f"C{i}", value="100nF", symbol="Device:C",
                      footprint=cap_fp, sheet="LED ARRAY", mpn=cap_mpn)
              for i in range(1, n_caps + 1)]
    conns = []
    for i in range(1, n_leds + 1):
        conns.append(NetConnection(net_name=f"DATA{i}", sheet="LED ARRAY",
                                   endpoints=[PinEndpoint(ref=f"D{i}", pin="2"),
                                              PinEndpoint(ref=f"D{i}", pin="4")]))
    p5 = [PinEndpoint(ref=f"D{i}", pin="1") for i in range(1, n_leds + 1)]
    gnd = [PinEndpoint(ref=f"D{i}", pin="3") for i in range(1, n_leds + 1)]
    for i in range(1, n_caps + 1):
        p5.append(PinEndpoint(ref=f"C{i}", pin="1"))
        gnd.append(PinEndpoint(ref=f"C{i}", pin="2"))
    conns.append(NetConnection(net_name="+5V", sheet="LED ARRAY", endpoints=p5))
    conns.append(NetConnection(net_name="GND", sheet="LED ARRAY", endpoints=gnd))
    arrays = [ArraySpec(refs=[f"D{i}" for i in range(1, n_leds + 1)], rows=1, cols=n_leds)]
    return BOM(parts=parts, connections=conns, arrays=arrays)


def _caps(bom: BOM) -> dict[str, str]:
    return {p.ref: p.footprint for p in bom.parts if p.ref.startswith("C")}


def test_led_cap_package_token_match() -> None:
    assert _led_cap_package([LED_SMALL]) == "0402"
    assert _led_cap_package(["LED:WS2812B-1313"]) == "0402"
    assert _led_cap_package(["ws2812b-2020:LED-SMD_4P-L2.0-W2.0-TL_WS2812B-2020"]) == "0402"
    assert _led_cap_package([LED_LARGE]) == "0603"          # 5 mm, no token -> default
    assert _led_cap_package(["LED_SMD:LED_5050_5050Metric"]) == "0603"
    assert _led_cap_package([""]) == "0603"                 # unrecognised -> default
    # mixed array: the smallest LED's package wins
    assert _led_cap_package(["...5050...", "...1313..."]) == "0402"


def test_small_led_downsizes_0603_to_0402() -> None:
    bom = _build_bom(n_leds=4, cap_fp=C0603, led_fp=LED_SMALL)
    changed = downsize_array_decap_footprints(bom)
    assert changed == ["C1", "C2", "C3", "C4"]
    assert all(fp == C0402 for fp in _caps(bom).values())
    assert any("resized 0603 -> 0402" in a for a in bom.assumptions)


def test_small_led_downsizes_0805_to_0402() -> None:
    bom = _build_bom(n_leds=3, cap_fp=C0805, led_fp=LED_SMALL)
    downsize_array_decap_footprints(bom)
    assert all(fp == C0402 for fp in _caps(bom).values())


def test_large_led_downsizes_0805_to_0603() -> None:
    bom = _build_bom(n_leds=3, cap_fp=C0805, led_fp=LED_LARGE)
    downsize_array_decap_footprints(bom)
    assert all(fp == C0603 for fp in _caps(bom).values())


def test_large_led_already_0603_is_noop() -> None:
    bom = _build_bom(n_leds=3, cap_fp=C0603, led_fp=LED_LARGE)
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == C0603 for fp in _caps(bom).values())


def test_never_enlarges() -> None:
    # 0402 caps on a large-LED array (target 0603) must stay 0402 -- never grow.
    bom = _build_bom(n_leds=3, cap_fp=C0402, led_fp=LED_LARGE)
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == C0402 for fp in _caps(bom).values())


def test_idempotent() -> None:
    bom = _build_bom(n_leds=3, cap_fp=C0805, led_fp=LED_SMALL)
    downsize_array_decap_footprints(bom)
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == C0402 for fp in _caps(bom).values())


def test_mpn_cap_untouched() -> None:
    # An MPN means a deliberately-chosen real part -> package preserved.
    bom = _build_bom(n_leds=3, cap_fp=C0805, led_fp=LED_SMALL, cap_mpn="CL10A104KB8NNNC")
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == C0805 for fp in _caps(bom).values())


def test_vendored_cap_untouched() -> None:
    bom = _build_bom(n_leds=3, cap_fp="my_caps:C_special_3D", led_fp=LED_SMALL)
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == "my_caps:C_special_3D" for fp in _caps(bom).values())


def test_handsolder_variant_untouched() -> None:
    # Suffixed variants are excluded by the stock regex (oversized courtyard).
    fp = "Capacitor_SMD:C_0805_2012Metric_Pad1.18x1.45mm_HandSolder"
    bom = _build_bom(n_leds=3, cap_fp=fp, led_fp=LED_SMALL)
    assert downsize_array_decap_footprints(bom) == []
    assert all(v == fp for v in _caps(bom).values())


def test_no_arrays_is_noop() -> None:
    bom = _build_bom(n_leds=3, cap_fp=C0805, led_fp=LED_SMALL)
    bom.arrays = []
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == C0805 for fp in _caps(bom).values())


def test_cap_on_non_array_sheet_untouched() -> None:
    bom = _build_bom(n_leds=3, cap_fp=C0805, led_fp=LED_SMALL)
    # a decap on a different sheet, with no array on it
    bom.parts.append(BomPart(ref="C99", value="100nF", symbol="Device:C",
                             footprint=C0805, sheet="POWER"))
    bom.connections.append(NetConnection(
        net_name="+5V_PWR", sheet="POWER",
        endpoints=[PinEndpoint(ref="C99", pin="1")]))
    bom.connections.append(NetConnection(
        net_name="GND_PWR", sheet="POWER",
        endpoints=[PinEndpoint(ref="C99", pin="2")]))
    downsize_array_decap_footprints(bom)
    assert _caps(bom)["C99"] == C0805        # untouched (no array on its sheet)
    assert all(_caps(bom)[f"C{i}"] == C0402 for i in range(1, 4))  # array caps still resized


def test_unknown_larger_package_left_alone() -> None:
    # 2026-07-19 review §4.6: a size code OUTSIDE the rank table (2010, 2512)
    # used to default to rank 99 and get force-shrunk -- inverting the
    # downsize-only contract. Unknown packages must be left untouched.
    c2010 = "Capacitor_SMD:C_2010_5025Metric"
    bom = _build_bom(n_leds=3, cap_fp=c2010, led_fp=LED_LARGE)
    assert downsize_array_decap_footprints(bom) == []
    assert all(fp == c2010 for fp in _caps(bom).values())
