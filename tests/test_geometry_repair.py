"""Illegal-geometry remediation (self-eval 2026-07-20 N2).

Rip DRC-illegal / outline-escaped copper, re-close the opens, keep only a
clean result. The functional test exercises the outline-escape branch with
real pcbnew; the wrapper tests execute the REAL inline subprocess scripts
in-process against the rip's documented return contract (the signal-repair
wrapper once no-op'd for a whole batch on exactly such a mismatch).
"""
from __future__ import annotations

import pytest


def _board_with_escaped_track(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    mm = pcbnew.FromMM
    board = pcbnew.CreateEmptyBoard()

    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(mm(100), mm(100)))
    rect.SetEnd(pcbnew.VECTOR2I(mm(130), mm(120)))
    rect.SetLayer(pcbnew.Edge_Cuts)
    board.Add(rect)

    def track(x1, y1, x2, y2):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        t.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        t.SetWidth(mm(0.25))
        t.SetLayer(pcbnew.F_Cu)
        board.Add(t)
        return t

    track(105, 110, 115, 110)          # fully inside -- must survive
    track(125, 110, 140, 110)          # escapes 10mm past the right edge
    out = tmp_path / "escaped.kicad_pcb"
    pcbnew.SaveBoard(str(out), board)
    return out


def test_rip_removes_escaped_copper_keeps_inside(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.brain.geometry_repair import rip_illegal_copper

    pcb = _board_with_escaped_track(tmp_path)
    s = rip_illegal_copper(str(pcb))
    assert s["ripped"] == 1
    assert s["over_cap"] == 0

    board = pcbnew.LoadBoard(str(pcb))
    tracks = [t for t in board.GetTracks()]
    assert len(tracks) == 1
    assert pcbnew.ToMM(tracks[0].GetEnd().x) < 130.0


def test_rip_bails_out_over_cap(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.brain.geometry_repair import rip_illegal_copper

    pcb = _board_with_escaped_track(tmp_path)
    before = pcb.read_bytes()
    s = rip_illegal_copper(str(pcb), {"geometry_repair_max_rips": 0})
    assert s["ripped"] == 0
    assert s["over_cap"] == 1
    assert pcb.read_bytes() == before  # bail leaves the board untouched


def test_geometry_wrapper_inline_scripts_match_return_contracts(
    tmp_path, monkeypatch, capsys
):
    """Execute the wrapper's REAL inline pcbnew scripts in-process: a key
    mismatch against {ripped, over_cap, nets, skipped} (or the signal
    repair's contract) raises AFTER the board mutated, and the wrapper's
    except-restore silently byte-reverts -- the no-op failure mode the
    signal wrapper shipped for a whole batch."""
    from kicraft.autoplacer import routing_board as fr
    from kicraft.autoplacer.brain import geometry_repair as gr
    from kicraft.autoplacer.brain import unconnected_repair as ur
    from kicraft.cli import _compose_route as cr

    pcb = tmp_path / "parent_routed.kicad_pcb"
    pcb.write_text("(kicad_pcb original)\n", encoding="utf-8")

    calls: dict[str, bool] = {}

    def fake_rip(path, cfg):
        calls["rip"] = True
        return {"ripped": 2, "over_cap": 0, "nets": ["SIG1"], "skipped": []}

    def fake_sig(path, cfg):
        calls["sig"] = True
        return {"edges": 1, "tied": 1, "skipped": [], "pruned": 0}

    monkeypatch.setattr(gr, "rip_illegal_copper", fake_rip)
    monkeypatch.setattr(ur, "repair_unconnected_signals", fake_sig)
    monkeypatch.setattr(fr, "run_pcbnew_script", lambda script: exec(script, {}))
    improved = {"drc": {"unconnected": 0, "shorts": 0}}
    monkeypatch.setattr(fr, "validate_routed_board", lambda *a, **k: improved)

    out = cr._attempt_illegal_geometry_repair(
        pcb,
        {},
        {
            "drc": {"unconnected": 2, "shorts": 0},
            "obviously_illegal_routed_geometry": True,
        },
    )

    assert calls.get("rip"), "inline script never invoked the rip"
    assert calls.get("sig"), "post-rip signal repair never ran"
    assert out == improved  # KEPT
    assert "illegal geometry rip:" in capsys.readouterr().out
    assert not (tmp_path / "parent_routed.kicad_pcb.pre_geometry_repair").exists()
    assert not (tmp_path / "parent_routed.kicad_pcb.geometry_repair.json").exists()


def test_geometry_wrapper_reverts_when_flags_persist(tmp_path, monkeypatch):
    """If the re-validate still flags illegal geometry, the mutated board is
    byte-restored and the original validation returned."""
    from kicraft.autoplacer import routing_board as fr
    from kicraft.autoplacer.brain import geometry_repair as gr
    from kicraft.autoplacer.brain import unconnected_repair as ur
    from kicraft.cli import _compose_route as cr

    pcb = tmp_path / "parent_routed.kicad_pcb"
    original = "(kicad_pcb original)\n"
    pcb.write_text(original, encoding="utf-8")

    def fake_rip(path, cfg):
        from pathlib import Path

        Path(path).write_text("(kicad_pcb mutated)\n", encoding="utf-8")
        return {"ripped": 1, "over_cap": 0, "nets": [], "skipped": []}

    monkeypatch.setattr(gr, "rip_illegal_copper", fake_rip)
    monkeypatch.setattr(
        ur,
        "repair_unconnected_signals",
        lambda path, cfg: {"edges": 0, "tied": 0, "skipped": [], "pruned": 0},
    )
    monkeypatch.setattr(fr, "run_pcbnew_script", lambda script: exec(script, {}))
    still_bad = {
        "drc": {"unconnected": 2, "shorts": 0},
        "obviously_illegal_routed_geometry": True,
    }
    monkeypatch.setattr(fr, "validate_routed_board", lambda *a, **k: still_bad)

    before = {
        "drc": {"unconnected": 2, "shorts": 0},
        "obviously_illegal_routed_geometry": True,
    }
    out = cr._attempt_illegal_geometry_repair(pcb, {}, before)

    assert out == before  # original validation stands
    assert pcb.read_text(encoding="utf-8") == original  # byte-restored
    assert not (tmp_path / "parent_routed.kicad_pcb.pre_geometry_repair").exists()
