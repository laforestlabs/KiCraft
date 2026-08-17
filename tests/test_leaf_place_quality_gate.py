"""Place-quality gate + seed-bbox suppression (dense-soc plan P2).

Both levers trade routing attempts for wall clock, so both must be provably
unable to leave a leaf with NO routed board -- that would turn a best-effort
compose into a hard leaf failure.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    PlacementScore,
    Point,
    SolveRoundResult,
)
from kicraft.cli import solve_subcircuits as ss


class _FakeSolver:
    last_grid_stats = {"slots_total": 4, "guard": "accept_score"}


def _comp(ref: str, x: float, net: str) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, 10.0), rotation=0.0, layer=Layer.FRONT,
        width_mm=2.0, height_mm=1.0, kind="passive", body_center=Point(x, 10.0),
        pads=[Pad(ref=ref, pad_id="1", pos=Point(x, 10.0), net=net, layer=Layer.FRONT)],
    )


def _anchor() -> Component:
    return Component(
        ref="U1", value="", pos=Point(0.0, 10.0), rotation=0.0, layer=Layer.FRONT,
        width_mm=6.0, height_mm=6.0, kind="ic", body_center=Point(0.0, 10.0),
        pads=[
            Pad(ref="U1", pad_id="1", pos=Point(0.0, 10.0), net="SIG", layer=Layer.FRONT),
            Pad(ref="U1", pad_id="2", pos=Point(1.0, 10.0), net="VCC", layer=Layer.FRONT),
        ],
    )


def test_diagnostics_report_honest_millimetres():
    comps = {"U1": _anchor(), "C1": _comp("C1", 30.0, "SIG")}
    diag = ss._placement_diagnostics(_FakeSolver(), comps, {})
    assert diag["median_pin_mm"] == pytest.approx(30.0, abs=0.1)
    assert diag["max_pin_mm"] == pytest.approx(30.0, abs=0.1)
    assert diag["worst_pins"][0]["ref"] == "C1"
    assert diag["grid"]["slots_total"] == 4


def test_no_anchors_no_gate():
    # A leaf with nothing scorable (anchor-less passive array) must not be gated.
    comps = {"C1": _comp("C1", 30.0, "SIG"), "C2": _comp("C2", 34.0, "SIG")}
    diag = ss._placement_diagnostics(_FakeSolver(), comps, {})
    assert "median_pin_mm" not in diag


def test_gate_only_fires_for_adjacency_failures():
    assert ss._only_adjacency_failures([]) is True
    assert ss._only_adjacency_failures(["no_unconnected"]) is True
    assert ss._only_adjacency_failures(["no_unconnected", "place_quality_gate"]) is True
    # a short or a courtyard overlap is not this gate's business
    assert ss._only_adjacency_failures(["no_unconnected", "no_shorts"]) is False


def _round(idx: int, median: float, routed: bool) -> SolveRoundResult:
    return SolveRoundResult(
        round_index=idx, seed=idx, score=50.0 if routed else float("-inf"),
        placement=PlacementScore(), components={},
        routing={"failed": not routed, "reason": "" if routed else "place_quality_gate",
                 "validation": {"accepted": routed}},
        routed=routed,
        placement_diagnostics={"median_pin_mm": median},
    )


class _FakeNode:
    def __init__(self, definition):
        self.definition = definition
        self.id = definition.id


def _leaf_ladder(monkeypatch, medians, tmp_path):
    """Run the ladder with a solver whose rounds have the given medians; every
    round 'fails' on no_unconnected. Returns the list of (round, gate_arg)."""
    from test_content_canvas import _leaf_and_state

    leaf, state = _leaf_and_state(n_parts=4)
    seen: list[tuple[int, float | None]] = []

    def fake_round(extraction, cfg, seed, round_index, route,
                   place_quality_best_mm=None):
        seen.append((round_index, place_quality_best_mm))
        median = medians[min(len(seen) - 1, len(medians) - 1)]
        gated = (
            place_quality_best_mm is not None
            and median > 4.0
            and median >= place_quality_best_mm
        )
        r = _round(round_index, median, routed=not gated)
        if not gated:
            r.routing = {
                "failed": False, "reason": "",
                "routed_board_path": str(tmp_path / "x.kicad_pcb"),
                "validation": {
                    "accepted": False,
                    "drc": {"unconnected": 1, "unconnected_nets": ["SIG"]},
                },
                "_trace_segments": [], "_via_objects": [],
            }
            r.routed = True
            r.score = 1.0
        return r

    monkeypatch.setattr(ss, "_solve_one_round", fake_round)
    monkeypatch.setattr(
        ss, "_attempt_leaf_size_reduction",
        lambda extraction, best, _cfg: (extraction, best, {"attempted": False}),
    )
    (tmp_path / "x.kicad_pcb").write_text("(kicad_pcb)")
    solved = ss._solve_leaf_subcircuit(
        node=_FakeNode(leaf), full_state=state,
        cfg={"leaf_canvas_mode": "content", "leaf_acceptance_max_unconnected": 0},
        rounds=2, base_seed=1, route=True, experiment_round=1,
    )
    return solved, seen


def test_last_round_of_the_ladder_always_routes(monkeypatch, tmp_path):
    # Every round is far off the threshold: the gate must still leave the final
    # round ungated, so the leaf composes best-effort instead of hard-failing.
    solved, seen = _leaf_ladder(monkeypatch, [30.0], tmp_path)
    assert seen, "the ladder must have run rounds"
    assert seen[-1][1] is None, "the last round must be ungated"
    assert solved.best_round is not None



class _RoundSolver:
    def __init__(self, state, cfg, seed):
        self.state = state
        self._edge_pinned_groups = []

    def solve(self):
        return self.state.components


def _round_with_route_error(monkeypatch, error):
    state = BoardState(components={"U1": _anchor()}, nets={})
    extraction = SimpleNamespace(
        local_state=state,
        internal_net_names={"SIG"},
    )
    monkeypatch.setattr(ss, "PlacementSolver", _RoundSolver)
    monkeypatch.setattr(
        ss,
        "_repair_leaf_placement_legality",
        lambda extraction, components, cfg: (components, {"resolved": True}),
    )
    monkeypatch.setattr(
        ss,
        "_score_local_components",
        lambda state, components, cfg: PlacementScore(),
    )
    monkeypatch.setattr(
        ss,
        "_placement_diagnostics",
        lambda solver, components, cfg: {},
    )

    def fail_route(*args, **kwargs):
        raise error

    monkeypatch.setattr(ss, "_route_local_subcircuit", fail_route)
    return ss._solve_one_round(
        extraction=extraction,
        cfg={
            "routing_backend": "kicad-routing-tools",
            "connector_edge_companion_clearance_mm": 0,
        },
        seed=1,
        round_index=0,
        route=True,
    )


def test_krt_backend_unavailable_reraises(monkeypatch):
    error = ss.RoutingBackendUnavailableError("native module unavailable")
    with pytest.raises(ss.RoutingBackendUnavailableError) as caught:
        _round_with_route_error(monkeypatch, error)
    assert caught.value is error


def test_generic_krt_route_failure_has_truthful_router_label(monkeypatch):
    result = _round_with_route_error(monkeypatch, RuntimeError("route failed"))
    assert result.routed is False
    assert result.routing["reason"] == "routing_exception"
    assert result.routing["router"] == "kicad-routing-tools"