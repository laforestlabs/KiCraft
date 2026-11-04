from types import SimpleNamespace

import pytest

from kicraft.design.synthesis import router, validation
from kicraft.design.synthesis.placement import PlacedPart


def _part(ref, symbol="Fake:Part", value="1"):
    return SimpleNamespace(ref=ref, symbol=symbol, value=value, sheet="S")


def _connection(name, *refs):
    return SimpleNamespace(net_name=name, sheet="S", endpoints=[
        SimpleNamespace(ref=ref, pin="1") for ref in refs
    ])


def _route(monkeypatch, parts, placements, connections, *, flag_nets=frozenset()):
    monkeypatch.setattr(router, "lookup_pins", lambda *a, **k: {"pins": [{
        "number": "1", "position": {"x": 0, "y": 0},
        "orientation": 180, "unit": 1, "length": 2.54,
    }]})
    bom = SimpleNamespace(parts=parts, connections=connections, no_connect_pins=[])
    architecture = SimpleNamespace(inter_sheet_nets=[])
    return router.route_sheet("S", "S", placements, bom, architecture, flag_nets)


def _placed(ref, x, y):
    return PlacedPart(ref, x, y, 0, None, "free")


def test_signal_stub_retreats_from_foreign_unwired_pin(monkeypatch):
    routed = _route(monkeypatch, [_part("U1"), _part("U2")],
                    [_placed("U1", 0, 0), _placed("U2", 2.54, 0)],
                    [_connection("DATA", "U1")])
    assert len(routed.wires) == 1
    assert routed.wires[0].x2_mm == pytest.approx(1.27)


def test_label_collision_keeps_anchor_on_own_wire():
    wire = router.WireSegment(0, 0, 0, 2.54)
    routed = router.RoutedSheet(labels=[router.NetLabel("OTHER", 0, 2.54)])
    x, y = router._free_label_anchor("DATA", (0, 2.54), [wire], routed)
    assert router._pt_on_axis_seg(x, y, 0, 0, 0, 2.54)
    assert (x, y) != (0, 2.54)
    with pytest.raises(ValueError):
        router._free_label_anchor("DATA", (0, 2.54), [], routed)


def test_flag_uses_own_component_not_nearby_foreign_wire(monkeypatch):
    def add_foreign_wire(routed, *_):
        routed.wires.append(router.WireSegment(5, 0.2, 5, 1.2))
    monkeypatch.setattr(router, "_resolve_label_collisions", add_foreign_wire)
    routed = _route(monkeypatch, [_part("U1"), _part("U2")],
                    [_placed("U1", 0, 0), _placed("U2", 10, 0)],
                    [_connection("DATA", "U1", "U2")], flag_nets=frozenset({"DATA"}))
    flag = next(p for p in routed.power_symbols if p.lib_id == "power:PWR_FLAG")
    assert (flag.x_mm, flag.y_mm) in {(0, 0), (10, 0)}


def test_schematic_instance_scanning_ignores_parentheses_in_strings(tmp_path):
    (tmp_path / "S.kicad_sch").write_text(
        '(kicad_sch (lib_symbols (symbol "Device:R" '
        '(property "Description" "example)") (property "Reference" "R1") '
        '(property "Footprint" ""))) '
        '(symbol (lib_id "Device:R") (property "Reference" "R2") '
        '(property "Value" "also ) literal") (property "Footprint" "Lib:R")))'
    )
    result = validation.check_footprints_nonempty(tmp_path)
    assert result.ok, result.offenders


def test_erc_does_not_accept_or_delete_stale_report(tmp_path, monkeypatch):
    (tmp_path / "S.kicad_sch").write_text("schematic")
    stale = tmp_path / "S_erc.rpt"
    stale.write_text('{"sheets": []}')
    monkeypatch.setattr(validation.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stderr="export failed"))
    result = validation.check_erc(tmp_path, "S")
    assert not result.ok and result.report_path is None
    assert stale.read_text() == '{"sheets": []}'


def test_erc_rejects_malformed_report(tmp_path, monkeypatch):
    (tmp_path / "S.kicad_sch").write_text("schematic")
    def run(args, **kwargs):
        from pathlib import Path
        Path(args[args.index("--output") + 1]).write_text("not JSON")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(validation.subprocess, "run", run)
    result = validation.check_erc(tmp_path, "S")
    assert not result.ok and result.report_path is None


def test_erc_points_to_current_report_without_overwriting_old(tmp_path, monkeypatch):
    (tmp_path / "S.kicad_sch").write_text("schematic")
    stale = tmp_path / "S_erc.rpt"
    stale.write_text('{"sheets": []}')
    current = '{"sheets": [{"path": "/", "violations": [{"severity": "error", "description": "bad"}]}]}'
    def run(args, **kwargs):
        from pathlib import Path
        Path(args[args.index("--output") + 1]).write_text(current)
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(validation.subprocess, "run", run)
    result = validation.check_erc(tmp_path, "S")
    assert not result.ok
    assert result.report_path and result.report_path != stale.name
    assert (tmp_path / result.report_path).read_text() == current
    assert stale.read_text() == '{"sheets": []}'


def test_netlist_does_not_accept_or_delete_stale_export(tmp_path, monkeypatch):
    (tmp_path / "S.kicad_sch").write_text("schematic")
    stale = tmp_path / "S_netlist_check.net"
    stale.write_text("previous output")
    monkeypatch.setattr(validation.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stderr="export failed"))
    assert not validation.check_netlist_faithfulness(tmp_path, "S", None).ok
    assert stale.read_text() == "previous output"


def test_zero_capacitor_in_series_crossover_is_a_defect_not_a_crash():
    requirement = SimpleNamespace(
        id="filter", family="passive-crossover", sheet="S",
        ports={"input": "IN", "low": "LOW", "high": "HIGH"},
        obligations=[SimpleNamespace(kind="quantitative", quantity="frequency",
                                    value=1000, unit="Hz", relation="equal"),
                     SimpleNamespace(kind="quantitative", quantity="impedance",
                                    value=8, unit="ohm", relation="equal")],
    )
    parts = [_part("C1", value="0uF"), _part("C2", value="1uF")]
    connections = [
        SimpleNamespace(net_name=net, endpoints=[SimpleNamespace(ref=ref, pin=pin)])
        for ref, pin, net in [("C1", "1", "IN"), ("C1", "2", "MID"),
                              ("C2", "1", "MID"), ("C2", "2", "HIGH")]
    ]
    result = validation.check_typed_passive_crossover_values(
        SimpleNamespace(requirements=[requirement]),
        SimpleNamespace(parts=parts, connections=connections),
    )
    assert not result.ok
    assert any("high-pass capacitors" in item for item in result.offenders)
