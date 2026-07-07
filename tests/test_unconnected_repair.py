"""Unit tests for the C1 signal unconnected repair's DRC-report parsing and
path candidates (the geometric layer needs no pcbnew)."""

from __future__ import annotations

from kicraft.autoplacer.brain.unconnected_repair import (
    _candidate_paths,
    _parse_unconnected_edges,
)

# Real shapes from run_09 / run_13 of logs/self_eval/20260706T224451Z.
_REPORT = """\
** Found 3 unconnected pads **
[unconnected_items]: Missing connection between items
    Local override; error
    @(123.5788 mm, 114.2167 mm): Pad A6 [USB_D+] of J1 on F.Cu
    @(142.4313 mm, 102.4121 mm): Pad 33 [USB_D+] of U2 on F.Cu
[unconnected_items]: Missing connection between items
    Local override; error
    @(147.5492 mm, 106.3944 mm): Track [BUTTON_GPIO] on F.Cu, length 1.1857 mm
    @(144.7332 mm, 118.3694 mm): Pad AC9 [BUTTON_GPIO] of U1 on F.Cu
[unconnected_items]: Missing connection between items
    Local override; error
    @(145.7332 mm, 118.3695 mm): Track [nRESET] on F.Cu, length 0.2500 mm
    @(147.6699 mm, 132.1643 mm): PTH pad 5 [nRESET] of J1
[shorting_items]: Shorting items
    @(1.0 mm, 1.0 mm): Track [A] on F.Cu
    @(1.0 mm, 1.0 mm): Track [B] on F.Cu
"""


def test_parses_pad_track_and_pth_endpoints():
    edges = _parse_unconnected_edges(_REPORT)
    assert len(edges) == 3  # the shorting_items block is not an edge

    usb = edges[0]
    assert (usb[0].net, usb[0].ref, usb[0].pad) == ("USB_D+", "J1", "A6")
    assert usb[0].layers == {"F.Cu"}
    assert (usb[1].ref, usb[1].pad) == ("U2", "33")

    button = edges[1]
    assert button[0].ref is None and button[0].pad is None  # track endpoint
    assert button[0].xy == (147.5492, 106.3944)
    assert (button[1].ref, button[1].pad) == ("U1", "AC9")

    nreset = edges[2]
    assert (nreset[1].ref, nreset[1].pad) == ("J1", "5")
    assert nreset[1].layers == {"F.Cu", "B.Cu"}  # PTH: no layer suffix


def test_candidate_paths_shapes():
    paths = _candidate_paths((0.0, 0.0), (12.0, 9.0), None)
    assert paths[0] == [(12.0, 9.0)]                     # straight first
    assert [(0.0, 9.0), (12.0, 9.0)] in paths            # L-bend 1
    assert [(12.0, 0.0), (12.0, 9.0)] in paths           # L-bend 2
    assert any(len(p) == 3 for p in paths)               # doglegs present
    assert all(p[-1] == (12.0, 9.0) for p in paths)      # all end on target


def test_candidate_paths_escape_prefix():
    esc = (1.0, -1.0)
    paths = _candidate_paths((0.0, 0.0), (10.0, 5.0), esc)
    assert all(p[0] == esc for p in paths)
    assert all(p[-1] == (10.0, 5.0) for p in paths)
