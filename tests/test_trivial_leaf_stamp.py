"""Regression tests for the trivial-leaf stamp path (no routable on-leaf nets).

KC-V8YWN8 (2026-07-02): commit 4d359f0 added ``render_intermediate`` to the
diagnostics gate inside ``_stamp_trivial_leaf``, but the variable was a local
of ``route_local_subcircuit`` -- every trivial leaf (screw terminal, battery
holder, any sheet whose nets each have a single on-leaf pad) raised NameError,
was rejected all rounds as ``routing_exception``, never serialized a result,
and the auto-pin safety net then failed the whole build. These tests drive the
real dispatch through ``route_local_subcircuit`` so a scope regression in the
trivial path can never ship silently again.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain import leaf_routing


class _StubAdapter:
    """Stands in for KiCadAdapter: 'stamps' by writing a minimal board file."""

    def __init__(self, source_pcb, config=None):
        self.source_pcb = source_pcb
        self.config = config

    def stamp_subcircuit_board(self, board, *, output_path, **kwargs):
        Path(output_path).write_text("(kicad_pcb (version 20240108))\n")


def _trivial_extraction(tmp_path: Path) -> SimpleNamespace:
    """An extraction whose every net has < 2 on-leaf pads (trivial leaf)."""
    local_state = SimpleNamespace(
        nets={"VOUT": SimpleNamespace(pad_refs=["J3.1"]),
              "GND": SimpleNamespace(pad_refs=["J3.2"])},
        components={},
        traces=[],
        vias=[],
        silkscreen=[],
        board_outline=None,
        board_width=20.0,
        board_height=20.0,
    )
    sch = tmp_path / "leaf.kicad_sch"
    sch.write_text("(kicad_sch)\n")
    return SimpleNamespace(
        subcircuit=SimpleNamespace(schematic_path=str(sch), id="leaf-under-test"),
        local_state=local_state,
        internal_net_names=[],
        notes=[],
    )


@pytest.fixture
def stubbed_leaf_routing(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    monkeypatch.setattr(
        leaf_routing, "resolve_artifact_paths",
        lambda root, sid: SimpleNamespace(artifact_dir=str(artifact_dir)),
    )
    monkeypatch.setattr(
        leaf_routing, "repair_leaf_placement_legality",
        lambda extraction, comps, cfg: (copy.deepcopy(comps), {"resolved": True}),
    )
    monkeypatch.setattr(leaf_routing, "KiCadAdapter", _StubAdapter)
    monkeypatch.setattr(leaf_routing, "_outline_around_geometry",
                        lambda comps, cfg: None)
    monkeypatch.setattr(leaf_routing, "_silk_for_leaf",
                        lambda extraction, comps, cfg: [])
    diag_calls: list[dict] = []

    def _fake_diag(**kwargs):
        diag_calls.append(kwargs)
        return {"stubbed": True}

    monkeypatch.setattr(leaf_routing, "generate_leaf_diagnostic_artifacts",
                        _fake_diag)
    return artifact_dir, diag_calls


def test_trivial_leaf_routes_via_real_dispatch(stubbed_leaf_routing, tmp_path):
    """route_local_subcircuit must complete the trivial path end-to-end,
    including the per-round diagnostics gate (the 4d359f0 NameError line)."""
    artifact_dir, diag_calls = stubbed_leaf_routing
    source_pcb = tmp_path / "seed.kicad_pcb"
    source_pcb.write_text("(kicad_pcb)\n")
    cfg = {
        "pcb_path": str(source_pcb),
        # Defaults render_intermediate=True so the diagnostics branch (where
        # the NameError lived) is actually evaluated and taken.
    }
    extraction = _trivial_extraction(tmp_path)

    routing, timing = leaf_routing.route_local_subcircuit(
        extraction,
        solved_components={},
        cfg=cfg,
        generate_diagnostics=True,
        round_index=0,
    )

    assert routing["reason"] == "no_internal_nets"
    assert routing["failed"] is False
    assert routing["validation"]["accepted"] is True
    assert Path(routing["routed_board_path"]).exists()
    assert (artifact_dir / "round_0000_leaf_routed.kicad_pcb").exists()
    # The diagnostics branch ran (proves render_intermediate resolved).
    assert routing["render_diagnostics"] == {"stubbed": True}
    assert diag_calls, "diagnostics gate was never evaluated"


def test_trivial_leaf_render_intermediate_off_skips_diagnostics(
    stubbed_leaf_routing, tmp_path
):
    """Headless builds set subcircuit_render_intermediate=False; the trivial
    path must honor the flag instead of crashing or rendering anyway."""
    artifact_dir, diag_calls = stubbed_leaf_routing
    source_pcb = tmp_path / "seed.kicad_pcb"
    source_pcb.write_text("(kicad_pcb)\n")
    cfg = {
        "pcb_path": str(source_pcb),
        "subcircuit_render_intermediate": False,
    }
    extraction = _trivial_extraction(tmp_path)

    routing, _ = leaf_routing.route_local_subcircuit(
        extraction,
        solved_components={},
        cfg=cfg,
        generate_diagnostics=True,
        round_index=1,
    )

    assert routing["reason"] == "no_internal_nets"
    assert routing["render_diagnostics"]["skipped"] is True
    assert not diag_calls
