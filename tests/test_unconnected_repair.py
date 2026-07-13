"""Unit tests for the C1 signal unconnected repair's DRC-report parsing and
path candidates (the geometric layer needs no pcbnew)."""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.unconnected_repair import (
    _candidate_paths,
    _parse_unconnected_edges,
    _path_clear,
    _seg_seg_dist_mm,
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


def test_seg_seg_dist_crossing_parallel_and_apart():
    # Proper crossing -> 0.
    assert _seg_seg_dist_mm((0, 0), (10, 0), (5, -5), (5, 5)) == 0.0
    # Parallel, 2mm apart.
    assert _seg_seg_dist_mm((0, 0), (10, 0), (0, 2), (10, 2)) == pytest.approx(2.0)
    # Endpoint-to-endpoint.
    assert _seg_seg_dist_mm((0, 0), (1, 0), (4, 4), (9, 4)) == pytest.approx(5.0)


def test_path_clear_prunes_crossing_and_forgives_endpoints():
    # One foreign F.Cu track crossing the corridor at x=10.
    obstacles = [("F.Cu", "GND", 10.0, -20.0, 10.0, 20.0, 0.25)]
    kw = dict(layer="F.Cu", net="SIG", margin_mm=0.23, src=(0.0, 0.0), tgt=(20.0, 0.0))
    # Straight path crosses it -> pruned.
    assert not _path_clear(obstacles, [(0.0, 0.0), (20.0, 0.0)], **kw)
    # A detour around its end (track spans y +/-20) stays clear.
    assert _path_clear(
        obstacles, [(0.0, 0.0), (5.0, 25.0), (15.0, 25.0), (20.0, 0.0)], **kw
    )
    # Same-net copper is never an obstacle.
    assert _path_clear(
        [("F.Cu", "SIG", 10.0, -20.0, 10.0, 20.0, 0.25)],
        [(0.0, 0.0), (20.0, 0.0)], **kw,
    )
    # Other-layer copper is not an obstacle (no via in the chain model).
    assert _path_clear(
        [("B.Cu", "GND", 10.0, -20.0, 10.0, 20.0, 0.25)],
        [(0.0, 0.0), (20.0, 0.0)], **kw,
    )
    # An obstacle hugging the TARGET pad is forgiven (endpoint carve-out):
    # the tie must be allowed to approach copper at both ends.
    assert _path_clear(
        [("F.Cu", "GND", 19.5, -0.5, 19.5, 0.5, 0.25)],
        [(0.0, 0.0), (20.0, 0.0)], **kw,
    )


def test_repair_ties_dogleg_around_partial_wall(tmp_path):
    """POSITIVE CONTROL for the candidate pre-filter: an edge that IS
    winnable (straight corridor walled by a foreign track, wide detour open)
    must still get tied. Guards against the pre-filter over-pruning -- on
    genuinely-blocked boards it returns 0 survivors (20260713 probe), and
    this test is the proof that winnable candidates survive the screen and
    reach the stamping guards."""
    import shutil as _shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not _shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable (repair edges come from real DRC)")
    from kicraft.autoplacer.brain.unconnected_repair import (
        repair_unconnected_signals,
    )

    mm = pcbnew.FromMM
    path = str(tmp_path / "wall.kicad_pcb")
    board = pcbnew.NewBoard(path)
    for name in ("SIG", "GND"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    for ref, x in (("J1", 5.0), ("J2", 20.0)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(net("SIG"))
        fp.Add(pad)

    # Foreign wall: a GND track across the straight corridor (y 10..20 at
    # x=12) -- straight and small-offset doglegs collide, the +/-7.5mm
    # dogleg clears it with >2mm to spare.
    wall = pcbnew.PCB_TRACK(board)
    wall.SetStart(pcbnew.VECTOR2I(mm(12.0), mm(10.0)))
    wall.SetEnd(pcbnew.VECTOR2I(mm(12.0), mm(20.0)))
    wall.SetWidth(mm(0.5))
    wall.SetLayer(pcbnew.F_Cu)
    wall.SetNet(net("GND"))
    board.Add(wall)
    board.Save(path)

    s = repair_unconnected_signals(path, {})

    assert s["edges"] == 1
    assert s["tied"] == 1, f"winnable edge not tied: {s}"
    assert s["skipped"] == []
    assert s["pruned"] > 0, "the walled straight/near candidates must be pruned"


def test_wrapper_inline_script_matches_repair_return_contract(
    tmp_path, monkeypatch, capsys
):
    """Execute the wrapper's REAL inline pcbnew script against the repair's
    documented return contract. A key the repair doesn't return raises
    KeyError AFTER the repair has mutated the board, and the wrapper's
    except-restore then silently byte-reverts it -- which turned the whole
    pass into a no-op on all 9 rc7 boards of the 20260710 self-eval batch
    (the caller printed s['nets']/s['stranded']/s['unresolved'] against a
    {edges, tied, skipped} contract)."""
    from kicraft.autoplacer import freerouting_runner as fr
    from kicraft.autoplacer.brain import unconnected_repair as ur
    from kicraft.cli import _compose_route as cr

    pcb = tmp_path / "parent_routed.kicad_pcb"
    pcb.write_text("(kicad_pcb repaired)\n", encoding="utf-8")

    calls: dict[str, bool] = {}

    def fake_repair(path, cfg):
        calls["repair"] = True
        return {"edges": 1, "tied": 1, "skipped": [], "pruned": 0}  # real contract

    monkeypatch.setattr(ur, "repair_unconnected_signals", fake_repair)
    # Run the inline script in-process so a contract mismatch raises here.
    monkeypatch.setattr(fr, "_run_pcbnew_script", lambda script: exec(script, {}))
    improved = {"drc": {"unconnected": 0, "shorts": 0}}
    monkeypatch.setattr(fr, "validate_routed_board", lambda *a, **k: improved)

    out = cr._attempt_signal_unconnected_repair(
        pcb, {}, {"drc": {"unconnected": 1, "shorts": 0}}
    )

    assert calls.get("repair"), "inline script never invoked the repair"
    # KEPT (returned the improved revalidation), not silently reverted.
    assert out == improved
    assert "signal unconnected repair:" in capsys.readouterr().out
    assert not (tmp_path / "parent_routed.kicad_pcb.pre_signal_repair").exists()
