"""Shared schematic geometry: where a symbol's pins land after rotation.

Both ``placement`` (deciding where to put parts) and ``router`` (drawing
wires to their pins) need the SAME answer to "given a symbol placed at
``(ox, oy, rot)``, where is pin P and which way does its wire exit?".
Keeping that math in one place is what lets the placer rotate a passive
and the router still find its pins.

Coordinate systems
------------------
* KiCad *library* symbols use math convention: +x right, +y UP. A pin's
  ``position`` is its connection point (the outer tip wires attach to);
  its ``orientation`` (0/90/180/270) is the angle the pin BODY extends
  into the symbol, so the wire leaves in the OPPOSITE direction.
* KiCad *schematic* sheets use +x right, +y DOWN.
* A symbol instance ``(at ox oy rot)`` rotates the library graphic CCW by
  ``rot`` (in the +y-up frame), then the sheet flips y.

These two functions were verified against ``kicad-cli sch erc`` for every
rotation: place a part at ``rot``, draw a stub out of each pin in the
returned exit direction, and KiCad reports zero ``wire_dangling`` — i.e.
the stub really lands on the pin. See tests/test_sch_geometry.py.
"""
from __future__ import annotations

ROTATIONS = (0, 90, 180, 270)

# Unit step (schematic +y down) for each exit direction.
DIR_VEC: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
}

_OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}

# (orientation + rot) % 360 -> wire exit direction in schematic coords.
# orientation 0 => body +x, wire exits -x => "left"; 90 => "down";
# 180 => "right"; 270 => "up". (The y flip turns lib +y-up into sheet down.)
_EXIT_BY_ORIENT = {0: "left", 90: "down", 180: "right", 270: "up"}


def opposite(direction: str) -> str:
    return _OPPOSITE[direction]


def rotate_vec(px: float, py: float, rot: int) -> tuple[float, float]:
    """Rotate a library-frame vector CCW by ``rot`` (+y up)."""
    r = rot % 360
    if r == 0:
        return (px, py)
    if r == 90:
        return (-py, px)
    if r == 180:
        return (-px, -py)
    if r == 270:
        return (py, -px)
    raise ValueError(f"rotation must be a multiple of 90, got {rot}")


def pin_abs_position(
    origin_x: float, origin_y: float, rot: int, pin: dict
) -> tuple[float, float]:
    """Absolute schematic (x, y) of ``pin`` for a symbol at (origin, rot)."""
    rx, ry = rotate_vec(pin["position"]["x"], pin["position"]["y"], rot)
    return (origin_x + rx, origin_y - ry)


def pin_exit_direction(rot: int, pin: dict) -> str:
    """Direction the wire leaves ``pin`` (schematic coords) at rotation ``rot``."""
    o = (int(pin.get("orientation", 0)) + rot) % 360
    return _EXIT_BY_ORIENT.get(o, "right")


def step(x: float, y: float, direction: str, dist: float) -> tuple[float, float]:
    """Move ``dist`` mm from (x, y) in ``direction``."""
    dx, dy = DIR_VEC[direction]
    return (x + dx * dist, y + dy * dist)


def rotation_for_exit(pin: dict, want_dir: str) -> int:
    """Rotation that makes ``pin`` exit toward ``want_dir`` (or 0 if none)."""
    for r in ROTATIONS:
        if pin_exit_direction(r, pin) == want_dir:
            return r
    return 0
