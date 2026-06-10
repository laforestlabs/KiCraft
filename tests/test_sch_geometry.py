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
