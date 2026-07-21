"""Power-first parent routing: power nets get their own freerouting phase.

freerouting 1.9.0 routes in board item-list order with no net priority, so
the wide power-class nets are structurally last-in-practice and end up walled
off by thin-net copper (KC-ZRAUR7: VBUS split in two islands 18 mm apart on a
55%-empty board, rc7 unc=1 through three rounds). `_route_parent_board` now
runs a phase-1 route with ONLY the power nets connectable (every other net's
DSN pins emptied via `freerouting_route_only_nets`; pads and locked wiring
stay obstacles), adopts the result, and lets the normal route lock the power
copper like leaf copper.

These guard the orchestration: phase 1 carries the right config and runs
before the main route, its output is adopted as the main route's input, any
phase-1 failure falls through to the single-phase flow, and the
`parent_power_first` kill switch works.
"""
from __future__ import annotations

from types import SimpleNamespace

import kicraft.autoplacer.brain.gnd_pour as gnd_pour
import kicraft.autoplacer.freerouting_runner as fr
import kicraft.cli._compose_route as cr

_P1_MARK = "(kicad_pcb phase1-power-routed)\n"


def _setup(monkeypatch, route_fn):
    monkeypatch.setattr(fr, "strip_net_copper", lambda *a, **k: None)
    monkeypatch.setattr(gnd_pour, "pour_gnd_planes", lambda *a, **k: None)
    monkeypatch.setattr(
        gnd_pour, "add_gnd_pour_and_thermal_vias", lambda *a, **k: None
    )
    monkeypatch.setattr(fr, "route_with_freerouting", route_fn)


def _state():
    return SimpleNamespace(
        composition=SimpleNamespace(inferred_interconnect_nets={"VBUS": 1}),
        component_count=10,
    )


def _cfg(**over):
    cfg = {
        "freerouting_jar": "unused-stub.jar",
        "gnd_zone_net": "GND",
        "shield_tie_enabled": False,
        "power_nets": ["GND", "VBUS", "VOUT_1"],
    }
    cfg.update(over)
    return cfg


def test_power_first_phase_runs_first_and_output_is_adopted(
    monkeypatch, tmp_path
):
    calls: list[dict] = []

    def _fake_route(*, kicad_pcb_path, output_path, jar_path, config):
        calls.append({
            "config": dict(config),
            "input_text": open(kicad_pcb_path).read(),
        })
        if config.get("freerouting_route_only_nets"):
            with open(output_path, "w") as f:
                f.write(_P1_MARK)      # phase 1 "routes" power and succeeds
            return {"ok": True}
        raise RuntimeError("stop after capturing the main route's input")

    _setup(monkeypatch, _fake_route)
    stamped = tmp_path / "parent_pre_freerouting.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n", encoding="utf-8")

    cr._route_parent_board(stamped, _state(), tmp_path, _cfg())

    only = [c for c in calls if c["config"].get("freerouting_route_only_nets")]
    assert only and calls[0] is only[0], (
        "the power-only phase must be freerouting's FIRST invocation -- "
        "running it later forfeits the structural priority that is its point"
    )
    # Power nets minus GND: GND is stripped/poured out-of-band, never routed.
    assert calls[0]["config"]["freerouting_route_only_nets"] == [
        "VBUS", "VOUT_1"
    ]
    assert calls[0]["config"]["freerouting_clear_zones"] is False

    # The main route must consume phase 1's output (power copper aboard, to
    # be locked by the DSN export), not the pre-phase stamped board.
    mains = [
        c for c in calls if not c["config"].get("freerouting_route_only_nets")
    ]
    assert mains and mains[0]["input_text"] == _P1_MARK, (
        "main route ran on the pre-phase board: phase-1 power copper was "
        "dropped instead of adopted"
    )


def test_power_first_failure_falls_through_to_single_phase(
    monkeypatch, tmp_path
):
    calls: list[dict] = []

    def _fake_route(*, kicad_pcb_path, output_path, jar_path, config):
        calls.append({
            "config": dict(config),
            "input_text": open(kicad_pcb_path).read(),
        })
        raise RuntimeError("freerouting fell over")

    _setup(monkeypatch, _fake_route)
    stamped = tmp_path / "parent_pre_freerouting.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n", encoding="utf-8")

    result = cr._route_parent_board(stamped, _state(), tmp_path, _cfg())

    # Phase 1 failed -> the normal attempt + GND-skip fallback still ran, on
    # the ORIGINAL stamped board (nothing adopted), and the failure was
    # recorded for parent_pipeline.json instead of vanishing.
    assert len(calls) == 3
    assert calls[1]["input_text"] == "(kicad_pcb stamped)\n"
    assert not calls[1]["config"].get("freerouting_route_only_nets")
    assert result["failed"] is True
    assert "failed" in result["freerouting_stats"]["power_first"]


def test_power_first_kill_switch(monkeypatch, tmp_path):
    calls: list[dict] = []

    def _fake_route(*, kicad_pcb_path, output_path, jar_path, config):
        calls.append(dict(config))
        raise RuntimeError("stop")

    _setup(monkeypatch, _fake_route)
    stamped = tmp_path / "parent_pre_freerouting.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n", encoding="utf-8")

    cr._route_parent_board(
        stamped, _state(), tmp_path, _cfg(parent_power_first=False)
    )

    assert calls and all(
        not c.get("freerouting_route_only_nets") for c in calls
    ), "parent_power_first=False must disable the power-only phase entirely"
