"""Tests for kicraft.design.synthesis.sch_geometry.

Pure-math tests plus a kicad-cli regression that LOCKS the transform to
KiCad's own renderer: place a Device:R at each rotation, draw a stub out of
each pin in the returned exit direction, and assert KiCad reports zero
``wire_dangling`` — i.e. the predicted pin position is where KiCad draws it.
If KiCad ever changes its rotation convention, this test catches it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kicraft.design.synthesis.sch_geometry import (
    opposite,
    pin_abs_position,
    pin_exit_direction,
    rotate_vec,
    rotation_for_exit,
    step,
)


def test_rotate_vec_ccw() -> None:
    assert rotate_vec(1.0, 0.0, 0) == (1.0, 0.0)
    assert rotate_vec(1.0, 0.0, 90) == (0.0, 1.0)
    assert rotate_vec(1.0, 0.0, 180) == (-1.0, 0.0)
    assert rotate_vec(1.0, 0.0, 270) == (0.0, -1.0)


def test_pin_abs_position_flips_y() -> None:
    pin = {"position": {"x": 0.0, "y": 3.81}, "orientation": 270}
    # rotation 0: +y-up lib coord becomes -y (up) on the sheet.
    assert pin_abs_position(10.0, 20.0, 0, pin) == (10.0, 20.0 - 3.81)
    # rotation 180: the pin swings to the other side.
    assert pin_abs_position(10.0, 20.0, 180, pin) == (10.0, 20.0 + 3.81)


def test_pin_exit_direction_rotations() -> None:
    # A resistor's pin-1 (orientation 270) exits "up" unrotated, and rotates
    # with the body.
    pin = {"position": {"x": 0.0, "y": 3.81}, "orientation": 270}
    assert pin_exit_direction(0, pin) == "up"
    assert pin_exit_direction(90, pin) == "left"
    assert pin_exit_direction(180, pin) == "down"
    assert pin_exit_direction(270, pin) == "right"


def test_rotation_for_exit_inverts_exit_direction() -> None:
    pin = {"position": {"x": 0.0, "y": 3.81}, "orientation": 270}
    for want in ("up", "down", "left", "right"):
        rot = rotation_for_exit(pin, want)
        assert pin_exit_direction(rot, pin) == want


def test_opposite_and_step() -> None:
    assert opposite("left") == "right"
    assert step(0.0, 0.0, "down", 2.54) == (0.0, 2.54)
    assert step(0.0, 0.0, "up", 2.54) == (0.0, -2.54)


def test_hierarchical_and_local_label_text_do_not_overlap():
    from kicraft.design.synthesis.router import (
        HierLabelPlacement, NetLabel, RoutedSheet, WireSegment,
        _label_rect, _rects_overlap, _resolve_label_collisions,
    )

    routed = RoutedSheet(
        wires=[WireSegment(0, 0, 2.54, 0), WireSegment(0, 1.27, 2.54, 1.27)],
        labels=[NetLabel("LOCAL_LONG_NAME", 2.54, 0)],
        hier_labels=[HierLabelPlacement("HIER_LONG_NAME", "input", 2.54, 1.27)],
    )
    _resolve_label_collisions(routed, [], [])
    local, hier = routed.labels[0], routed.hier_labels[0]
    assert not _rects_overlap(
        _label_rect(local.text, local.x_mm, local.y_mm, local.angle_deg),
        _label_rect(hier.name, hier.x_mm, hier.y_mm, hier.angle_deg),
    )
    assert (local.x_mm, local.y_mm) == (2.54, 0)
    assert (hier.x_mm, hier.y_mm) == (2.54, 1.27)
    assert local.text == "LOCAL_LONG_NAME" and hier.name == "HIER_LONG_NAME"


def test_label_fanout_keeps_connected_wire_and_clear_text():
    from kicraft.design.synthesis.router import (
        NetLabel, RoutedSheet, WireSegment, _label_rect, _rects_overlap,
        _resolve_label_collisions,
    )

    routed = RoutedSheet(
        wires=[WireSegment(-2.54, 0, 0, 0)],
        labels=[NetLabel("SIGNAL_NAME", 0, 0), NetLabel("X" * 200, 0, 1.27)],
    )
    body = (-15, -1, -5, 1)
    _resolve_label_collisions(routed, [body], [])
    first, other = routed.labels
    assert not _rects_overlap(
        _label_rect(first.text, first.x_mm, first.y_mm, first.angle_deg),
        _label_rect(other.text, other.x_mm, other.y_mm, other.angle_deg),
    )
    assert len(routed.wires) == 3
    assert (routed.wires[1].x1_mm, routed.wires[1].y1_mm) == (0, 0)
    assert (routed.wires[-1].x2_mm, routed.wires[-1].y2_mm) == (first.x_mm, first.y_mm)


def test_capacitor_and_ground_clear_vendor_module_graphics():
    from kicraft.design.models import BOM, BomPart, NetConnection, PinEndpoint, Sheet
    from kicraft.design.synthesis.placement import place_sheet
    from kicraft.design.synthesis.sch_geometry import placed_graphics_bounds

    sheet = Sheet(name="MCU", stem="MCU", function="controller")
    parts = [
        BomPart(ref="U1", value="ESP32-S3-WROOM-1-N16R8",
                symbol="esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1",
                footprint="esp32-s3-wroom-1-n16r8:WIRELM-SMD_ESP32-S3-WROOM-1", sheet="MCU"),
        BomPart(ref="C1", value="100nF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric", sheet="MCU"),
    ]
    bom = BOM(parts=parts, connections=[
        NetConnection(net_name="+3V3", sheet="MCU", endpoints=[
            PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="C1", pin="1")]),
        NetConnection(net_name="GND", sheet="MCU", endpoints=[
            PinEndpoint(ref="U1", pin="1"), PinEndpoint(ref="C1", pin="2")]),
    ])
    placed = {part.ref: part for part in place_sheet(sheet, parts, bom)}
    module = placed["U1"]
    body = placed_graphics_bounds(parts[0].symbol, 1, module.x_mm, module.y_mm, 0)
    assert body is not None
    assert placed["C1"].y_mm + 8.0 < body[1]


def test_rotated_power_value_avoids_adjacent_hierarchical_label():
    from kicraft.design.synthesis.emitter import _emit_power_symbol, _fit_leaf_page
    from kicraft.design.synthesis.router import (
        HierLabelPlacement, PowerSymbol, RoutedSheet, WireSegment,
        _label_rect, _place_power_values, _power_value_rect, _rects_overlap,
        _resolve_label_collisions,
    )

    routed = RoutedSheet(
        wires=[WireSegment(-2.54, 0, 0, 0), WireSegment(0, 3, 2.54, 3)],
        power_symbols=[PowerSymbol("power:GND", 0, 0, 90)],
        hier_labels=[HierLabelPlacement("HUB75_ENABLE", "output", 0, 3, 0)],
    )
    original_power_wire = routed.wires[0]
    obstacles = _place_power_values(routed, [])
    _resolve_label_collisions(routed, obstacles, [])
    power, label = routed.power_symbols[0], routed.hier_labels[0]
    assert routed.wires[0] == original_power_wire
    assert not _rects_overlap(
        _power_value_rect(power),
        _label_rect(label.name, label.x_mm, label.y_mm, label.angle_deg),
    )
    assert power.value_x_mm is not None and power.value_y_mm is not None
    emitted = _emit_power_symbol(power, "leaf", "DEMO", "ground")
    assert (f'(at {power.value_x_mm:g} {power.value_y_mm:g} 270)' in emitted)
    _, fitted, _ = _fit_leaf_page([], [], routed)
    fitted_power, fitted_label = fitted.power_symbols[0], fitted.hier_labels[0]
    assert not _rects_overlap(
        _power_value_rect(fitted_power),
        _label_rect(fitted_label.name, fitted_label.x_mm,
                    fitted_label.y_mm, fitted_label.angle_deg),
    )

# ---------- kicad-cli regression: transform matches KiCad ----------

from kicraft.design.synthesis.symbol_library import (  # noqa: E402
    DEFAULT_KICAD_SYMBOL_DIR,
    build_lib_symbols_block,
)
from kicraft.design.synthesis.symbol_pinout import lookup_pins  # noqa: E402

_NEEDS_KICAD = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir() or shutil.which("kicad-cli") is None,
    reason="needs KiCad symbols + kicad-cli",
)


@_NEEDS_KICAD
def test_rotated_pin_geometry_matches_kicad(tmp_path: Path) -> None:
    pins = lookup_pins("Device:R")["pins"]
    lib = build_lib_symbols_block([("Device", "R")])
    blocks, extra = [], []
    n = 0
    for i, a in enumerate((0, 90, 180, 270)):
        ox, oy = 50.8 + i * 25.4, 63.5  # all on the 1.27 mm grid
        blocks.append(
            f'\t(symbol (lib_id "Device:R") (at {ox} {oy} {a}) (unit 1) '
            f'(in_bom yes) (on_board yes) (uuid "u{a}")\n'
            f'\t\t(property "Reference" "R{a}" (at {ox + 5} {oy} 0) '
            f"(effects (font (size 1.27 1.27))))\n"
            f'\t\t(property "Value" "1k" (at {ox + 5} {oy + 2} 0) '
            f"(effects (font (size 1.27 1.27))))\n"
            f'\t\t(instances (project "P" (path "/l" (reference "R{a}") (unit 1)))))'
        )
        for pin in pins:
            px, py = pin_abs_position(ox, oy, a, pin)
            d = pin_exit_direction(a, pin)
            ex, ey = step(px, py, d, 2.54)
            extra.append(
                f'\t(wire (pts (xy {px} {py}) (xy {ex} {ey})) '
                f'(stroke (width 0) (type default)) (uuid "w{n}"))')
            extra.append(
                f'\t(label "N{n}" (at {ex} {ey} 0) '
                f'(effects (font (size 1.27 1.27))) (uuid "lab{n}"))')
            n += 1
    leaf = (
        '(kicad_sch (version 20250114) (generator "eeschema") '
        '(generator_version "9.0") (uuid "l") (paper "A4")\n\n'
        + lib + "\n\n" + "\n".join(blocks) + "\n" + "\n".join(extra)
        + '\n\n\t(sheet_instances (path "/" (page "1")))\n)\n'
    )
    sch = tmp_path / "ROT.kicad_sch"
    sch.write_text(leaf)
    rpt = tmp_path / "rot.json"
    subprocess.run(
        ["kicad-cli", "sch", "erc", "--format", "json", "--output", str(rpt), str(sch)],
        capture_output=True, text=True, timeout=60,
    )
    report = json.loads(rpt.read_text())
    dangling = [
        v for sheet in report.get("sheets", []) for v in sheet.get("violations", [])
        if v.get("type") == "wire_dangling"
    ]
    assert dangling == [], f"transform mismatch: {dangling}"
