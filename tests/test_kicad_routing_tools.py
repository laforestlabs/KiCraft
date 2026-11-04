import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.autoplacer import kicad_routing_tools as rb


def _write_project(board):
    from kicraft.design.synthesis.kicad_pro import DEFAULT_NETCLASS, DEFAULT_RULES

    project = board.with_suffix(".kicad_pro")
    project.write_text(json.dumps({
        "board": {"design_settings": {"rules": dict(DEFAULT_RULES)}},
        "net_settings": {"classes": [dict(DEFAULT_NETCLASS)]},
    }))
    return project


def test_adaptive_routing_cannot_undershoot_project_geometry(tmp_path):
    project = _write_project(tmp_path / "input.kicad_pcb")
    body = json.loads(project.read_text())
    rules = body["board"]["design_settings"]["rules"]
    rules.update(min_via_annular_width=0.2, min_hole_clearance=0.4)
    body["net_settings"]["classes"].append({"name": "Power", "clearance": 0.4})
    project.write_text(json.dumps(body))
    floors = rb._project_routing_floors(project, {})
    ring = (floors["via_diameter"] - floors["via_drill"]) / 2
    assert ring >= rules["min_via_annular_width"]
    assert floors["clearance"] + ring >= rules["min_hole_clearance"]
    assert all(
        floors["clearance"] >= netclass["clearance"]
        for netclass in body["net_settings"]["classes"]
    )
    assert floors["via_diameter"] >= rules["min_via_diameter"]


def test_router_clearance_ceiling_cannot_weaken_a_power_class(tmp_path):
    project = _write_project(tmp_path / "input.kicad_pcb")
    body = json.loads(project.read_text())
    body["net_settings"]["classes"].append({"name": "Power", "clearance": 0.4})
    project.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        rb._project_routing_floors(project, {"kicad_routing_tools_clearance_mm": 0.2})


def test_preflight_requires_configured_checkout():
    with pytest.raises(rb.KicadRoutingToolsUnavailableError, match="path is unset"):
        rb.preflight_kicad_routing_tools({})




def test_parent_routes_stamped_board_once_with_krt(monkeypatch, tmp_path):
    import kicraft.autoplacer.kicad_routing_tools as krt
    import kicraft.autoplacer.routing_board as board_utils
    import kicraft.cli._compose_route as compose_route

    stamped = tmp_path / "parent_stamped.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n")
    events = []

    def fake_route(source, output, config):
        events.append((source, output, dict(config)))
        Path(output).write_text("(kicad_pcb routed)\n")
        return {"backend": "kicad-routing-tools", "returncode": 0}

    monkeypatch.setattr(krt, "route_with_kicad_routing_tools", fake_route)
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda p: {"traces": [], "vias": []})
    monkeypatch.setattr(board_utils, "validate_routed_board", lambda *a, **k: {"accepted": True, "drc": {}})
    state = SimpleNamespace(composition=SimpleNamespace())
    cfg = {
        "gnd_zone_net": "",
        "power_plane_enabled": False,
        "signal_unconnected_repair_enabled": False,
        "illegal_geometry_repair_enabled": False,
    }
    result = compose_route._route_parent_board(stamped, state, tmp_path, cfg)
    assert len(events) == 1
    assert events[0][0] == str(stamped)
    assert result["backend"] == "kicad-routing-tools"
    assert result["routing_stats"]["returncode"] == 0
    assert list(result).count("routing_stats") == 1


def test_parent_route_uses_parent_timeout_budget(monkeypatch, tmp_path):
    """The parent route gets its own (longer) KRT slice, not the leaf cap.

    A parent's in-run recovery ladder (rip -> #103 reconciliation) does not fit
    the 120 s leaf cap; production runs were SIGTERMed mid-reconciliation.
    """
    import kicraft.autoplacer.kicad_routing_tools as krt
    import kicraft.autoplacer.routing_board as board_utils
    import kicraft.cli._compose_route as compose_route

    stamped = tmp_path / "parent_stamped.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n")
    seen: list[dict] = []

    def fake_route(source, output, config):
        seen.append(dict(config))
        Path(output).write_text("(kicad_pcb routed)\n")
        return {"backend": "kicad-routing-tools", "returncode": 0}

    monkeypatch.setattr(krt, "route_with_kicad_routing_tools", fake_route)
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda p: {"traces": [], "vias": []})
    monkeypatch.setattr(
        board_utils, "validate_routed_board", lambda *a, **k: {"accepted": True, "drc": {}}
    )
    state = SimpleNamespace(composition=SimpleNamespace())
    base = {
        "gnd_zone_net": "",
        "power_plane_enabled": False,
        "signal_unconnected_repair_enabled": False,
        "illegal_geometry_repair_enabled": False,
    }

    compose_route._route_parent_board(
        stamped,
        state,
        tmp_path,
        {**base, "kicad_routing_tools_timeout_s": 120,
         "kicad_routing_tools_parent_timeout_s": 420},
    )
    assert seen[-1]["kicad_routing_tools_timeout_s"] == 420

    # Callers that only know the leaf cap keep their value.
    compose_route._route_parent_board(
        stamped, state, tmp_path, {**base, "kicad_routing_tools_timeout_s": 90}
    )
    assert seen[-1]["kicad_routing_tools_timeout_s"] == 90


def _runtime(root: Path) -> dict[str, str]:
    return {
        "backend": "kicad-routing-tools",
        "root": str(root),
        "python": sys.executable,
        "version": rb.KICAD_ROUTING_TOOLS_VERSION,
        "commit": rb.KICAD_ROUTING_TOOLS_COMMIT,
        "native_version": "0.20.1",
    }


def _trace() -> dict[str, object]:
    return {
        "start_x": 1.0,
        "start_y": 2.0,
        "end_x": 3.0,
        "end_y": 4.0,
        "layer": "FRONT",
        "width": 0.2,
    }


def _via() -> dict[str, float]:
    return {"x": 2.0, "y": 3.0, "drill": 0.3, "size": 0.6}


def _copper(*, present: bool = True) -> dict[str, object]:
    return {
        "traces": [_trace()] if present else [],
        "vias": [_via()] if present else [],
        "trace_count": 1 if present else 0,
        "via_count": 1 if present else 0,
        "total_length_mm": 2.0 if present else 0.0,
    }


def test_krt_preflight_observes_runtime_and_caches_success(monkeypatch, tmp_path):
    root = tmp_path / "krt"
    (root / "py_router").mkdir(parents=True)
    (root / "py_router" / "route.py").write_text("# route\n")
    (root / "VERSION").write_text("0.20.2\n")
    (root / ".git").mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{rb.KICAD_ROUTING_TOOLS_COMMIT}\n",
                stderr="",
            )
        assert command[0] == "/fake/krt-venv/bin/python"
        assert kwargs["cwd"] == root.resolve()
        return SimpleNamespace(returncode=0, stdout="0.20.1\n", stderr="")

    monkeypatch.setattr(rb.shutil, "which", lambda _: "/fake/krt-venv/bin/python")
    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    rb._KRT_PREFLIGHT_CACHE.clear()
    cfg = {
                "kicad_routing_tools_path": str(root),
        "kicad_routing_tools_python": "/configured/python",
    }

    first = rb.preflight_kicad_routing_tools(cfg)
    second = rb.preflight_kicad_routing_tools(cfg)

    assert first == second
    assert first["version"] == "0.20.2"
    assert first["commit"] == rb.KICAD_ROUTING_TOOLS_COMMIT
    assert first["native_version"] == "0.20.1"
    assert len(calls) == 2


def test_krt_preflight_failures_are_not_cached(monkeypatch, tmp_path):
    root = tmp_path / "krt"
    (root / "py_router").mkdir(parents=True)
    (root / "py_router" / "route.py").write_text("# route\n")
    (root / "VERSION").write_text("0.20.2\n")
    (root / ".git").mkdir()
    startup_calls = 0

    def fake_run(command, **kwargs):
        nonlocal startup_calls
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{rb.KICAD_ROUTING_TOOLS_COMMIT}\n",
                stderr="",
            )
        startup_calls += 1
        raise subprocess.CalledProcessError(
            1, command, stderr="native module unavailable"
        )

    monkeypatch.setattr(rb.shutil, "which", lambda _: "/fake/krt-venv/bin/python")
    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    rb._KRT_PREFLIGHT_CACHE.clear()
    cfg = {
                "kicad_routing_tools_path": str(root),
    }

    for _ in range(2):
        with pytest.raises(rb.KicadRoutingToolsUnavailableError, match="startup checks"):
            rb.preflight_kicad_routing_tools(cfg)
    assert startup_calls == 2




def test_krt_route_rejects_same_input_and_output(tmp_path):
    board = tmp_path / "same.kicad_pcb"
    board.write_text("board\n")
    with pytest.raises(ValueError, match="must be distinct"):
        rb.route_with_kicad_routing_tools(
            str(board),
            str(board),
            {},
        )


def test_krt_route_requires_project_rules_before_launch(monkeypatch, tmp_path):
    import kicraft.autoplacer.routing_board as board_utils

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    launched = False

    def fail_launch(*args, **kwargs):
        nonlocal launched
        launched = True
        pytest.fail("route launched without project rules")

    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", fail_launch)
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda _: _copper())

    with pytest.raises(
        rb.KicadRoutingToolsUnavailableError,
        match=r"^KiCadRoutingTools requires a sibling \.kicad_pro",
    ):
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
            },
        )
    assert launched is False


@pytest.mark.parametrize(
    ("returncode", "create_output", "match"),
    [(7, True, r"failed \(rc=7\)"), (0, False, r"failed \(rc=0\)")],
)
def test_krt_route_keeps_nonzero_and_no_output_failures(
    monkeypatch, tmp_path, returncode, create_output, match
):
    import kicraft.autoplacer.routing_board as board_utils

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    _write_project(input_board)

    class FakeProcess:
        pid = 456

        def __init__(self, command, **kwargs):
            self.returncode = returncode

        def communicate(self, timeout=None):
            if create_output:
                output_board.write_text("diagnostic output\n")
            return ("stdout", "stderr")

    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda _: _copper())

    with pytest.raises(RuntimeError, match=match):
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
            },
        )


def test_krt_route_timeout_behavior_is_unchanged(monkeypatch, tmp_path):
    import kicraft.autoplacer.routing_board as board_utils

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    _write_project(input_board)
    signals = []

    class FakeProcess:
        returncode = -15
        pid = 789

        def __init__(self, command, **kwargs):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("krt", timeout)
            return ("partial stdout", "partial stderr")

    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(rb.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda _: _copper())

    with pytest.raises(RuntimeError, match="timed out after 1s"):
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
                "kicad_routing_tools_timeout_s": 1,
            },
        )
    assert signals == [(789, signal.SIGTERM)]
    assert output_board.with_suffix(".router.stdout.log").read_text() == "partial stdout"
    assert output_board.with_suffix(".router.stderr.log").read_text() == "partial stderr"


def test_krt_route_rejects_missing_input_copper(monkeypatch, tmp_path):
    import kicraft.autoplacer.routing_board as board_utils

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    original_rules = _write_project(input_board).read_bytes()

    def fake_import(path):
        return _copper(present=Path(path).resolve() == input_board.resolve())

    class FakeProcess:
        returncode = 0
        pid = 999

        def __init__(self, command, **kwargs):
            pass

        def communicate(self, timeout=None):
            output_board.write_text("routed but corrupt\n")
            return ('JSON_SUMMARY: {"successful": 1}\n', "")

    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(board_utils, "import_routed_copper", fake_import)

    with pytest.raises(rb.RoutingCopperPreservationError) as caught:
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
            },
        )
    assert caught.value.stats["preserved_existing_copper"] is False
    preservation = caught.value.stats["input_copper_preservation"]
    assert preservation["traces"]["missing_count"] == 1
    assert preservation["vias"]["missing_count"] == 1
    assert output_board.is_file()
    assert output_board.with_suffix(".kicad_pro").read_bytes() == original_rules


def _trace_span(x: float) -> dict[str, object]:
    return {
        "start_x": x,
        "start_y": 2.0,
        "end_x": x + 2.0,
        "end_y": 4.0,
        "layer": "FRONT",
        "width": 0.2,
    }


def _run_bridge_with_missing_input_trace(monkeypatch, tmp_path, stdout):
    import kicraft.autoplacer.routing_board as board_utils

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    _write_project(input_board)
    kept, pruned = _trace_span(1.0), _trace_span(9.0)

    def fake_import(path):
        is_input = Path(path).resolve() == input_board.resolve()
        return {
            "traces": [kept, pruned] if is_input else [kept],
            "vias": [_via()],
            "trace_count": 2 if is_input else 1,
            "via_count": 1,
            "total_length_mm": 4.0,
        }

    class FakeProcess:
        returncode = 0
        pid = 321

        def __init__(self, command, **kwargs):
            pass

        def communicate(self, timeout=None):
            output_board.write_text("routed\n")
            return (stdout, "")

    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(board_utils, "import_routed_copper", fake_import)
    return input_board, output_board, root


def test_krt_route_accepts_declared_unloaded_input_prune(monkeypatch, tmp_path):
    """A removal the router DECLARES as unloaded cleanup is not copper loss.

    The dead-end/orphan-island sweep strips original-file segments that reach no
    pad and no continuation even under --keep-input-copper; refusing those turned
    fully routed parents (brief 2, project 1121) into route failures.
    """
    input_board, output_board, root = _run_bridge_with_missing_input_trace(
        monkeypatch,
        tmp_path,
        "Removed 1 dead-end input segment(s) from the output\n"
        'JSON_SUMMARY: {"successful": 1}\n',
    )
    stats = rb.route_with_kicad_routing_tools(
        str(input_board),
        str(output_board),
        {"kicad_routing_tools_path": str(root)},
    )
    assert stats["copper_preservation_declared_prune"] is True
    assert stats["pruned_unloaded_input_traces"] == 1
    assert stats["preserved_existing_copper"] is False


def test_krt_route_rejects_undeclared_missing_input_copper(monkeypatch, tmp_path):
    """Same missing trace, no declared cleanup: still a custody failure."""
    input_board, output_board, root = _run_bridge_with_missing_input_trace(
        monkeypatch,
        tmp_path,
        'JSON_SUMMARY: {"successful": 1}\n',
    )
    with pytest.raises(rb.RoutingCopperPreservationError) as caught:
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {"kicad_routing_tools_path": str(root)},
        )
    assert caught.value.stats["input_copper_preservation"]["traces"]["missing_count"] == 1
    assert "declared dead-end removals=0" in str(caught.value)


def test_npth_rule_area_staging_preserves_source_and_is_idempotent(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    footprint_lib = (
        Path(__file__).resolve().parents[1]
        / "kicraft"
        / "parts_library"
        / "usb-c-16p"
        / "usb-c-16p.pretty"
    )
    source = tmp_path / "pre_route.kicad_pcb"
    output = tmp_path / "routed.kicad_pcb"
    board = pcbnew.NewBoard(str(source))
    connector = pcbnew.FootprintLoad(
        str(footprint_lib), "USB-C_SMD-TYPE-C-31-M-12_1"
    )
    assert connector is not None
    board.Add(connector)
    connector.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(20), pcbnew.FromMM(20)))
    connector.SetOrientationDegrees(90)
    connector.Flip(connector.GetPosition(), False)
    slot_lib = tmp_path / "slot.pretty"
    slot_lib.mkdir()
    (slot_lib / "Slot.kicad_mod").write_text(
        """(footprint "Slot"
  (version 20240108)
  (generator pcbnew)
  (layer "F.Cu")
  (pad "" np_thru_hole oval
    (at 0 0)
    (size 1 3)
    (drill oval 1 3)
    (layers "*.Cu" "*.Mask"))
)
"""
    )
    slot = pcbnew.FootprintLoad(str(slot_lib), "Slot")
    assert slot is not None
    board.Add(slot)
    slot.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(30), pcbnew.FromMM(20)))
    slot.SetOrientationDegrees(90)
    slot.Flip(slot.GetPosition(), False)
    board.Save(str(source))
    project = _write_project(source)
    project_data = json.loads(project.read_text())
    project_data["board"]["design_settings"]["rules"]["min_hole_clearance"] = 0.25
    project.write_text(json.dumps(project_data))
    source_bytes = source.read_bytes()

    floors = rb._project_routing_floors(project, {})
    staged, summary = rb._stage_npth_keepouts(
        source,
        output,
        project,
        hole_clearance_mm=0.25,
        router_clearance_mm=floors["clearance"],
    )

    assert source.read_bytes() == source_bytes
    assert staged.with_suffix(".kicad_pro").read_bytes() == project.read_bytes()
    assert staged != source
    assert summary is not None
    assert summary["holes"] == 3
    assert summary["zones"] == 6
    assert summary["margin_mm"] == pytest.approx(0.25 - floors["clearance"])

    staged_board = pcbnew.LoadBoard(str(staged))
    npth_pads = [
        pad
        for footprint in staged_board.Footprints()
        for pad in footprint.Pads()
        if pad.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH
    ]
    keepouts = [
        zone
        for zone in staged_board.Zones()
        if zone.GetIsRuleArea() and zone.GetZoneName() == rb._NPTH_KEEPOUT_ZONE_NAME
    ]
    assert len(keepouts) == len(npth_pads) * 2
    for pad in npth_pads:
        point = pad.GetPosition()
        matching = [
            zone
            for zone in keepouts
            if (
                zone.GetBoundingBox().GetLeft() <= point.x <= zone.GetBoundingBox().GetRight()
                and zone.GetBoundingBox().GetTop() <= point.y <= zone.GetBoundingBox().GetBottom()
            )
        ]
        assert len(matching) == 2
        drill = pad.GetDrillSize()
        minor_radius = min(pcbnew.ToMM(drill.x), pcbnew.ToMM(drill.y)) / 2
        major_diameter = max(pcbnew.ToMM(drill.x), pcbnew.ToMM(drill.y))
        for zone in matching:
            bounds = zone.GetBoundingBox()
            width = pcbnew.ToMM(bounds.GetWidth())
            height = pcbnew.ToMM(bounds.GetHeight())
            assert min(width, height) / 2 >= minor_radius + summary["margin_mm"]
            assert max(width, height) >= major_diameter + 2 * summary["margin_mm"]

    repeat = rb._stamp_npth_keepouts(
        staged,
        hole_clearance_mm=0.25,
        router_clearance_mm=floors["clearance"],
    )
    assert repeat["zones"] == len(keepouts)
    restamped = pcbnew.LoadBoard(str(staged))
    assert sum(
        zone.GetIsRuleArea() and zone.GetZoneName() == rb._NPTH_KEEPOUT_ZONE_NAME
        for zone in restamped.Zones()
    ) == len(keepouts)

    # The shipped board carries only authored geometry: the router-side areas
    # are removed once the routed copper already satisfies the project rule.
    rb._strip_npth_keepouts(staged)
    stripped = pcbnew.LoadBoard(str(staged))
    assert not [
        zone
        for zone in stripped.Zones()
        if zone.GetIsRuleArea() and zone.GetZoneName() == rb._NPTH_KEEPOUT_ZONE_NAME
    ]
    assert sum(len(footprint.Pads()) for footprint in stripped.Footprints()) == sum(
        len(footprint.Pads()) for footprint in restamped.Footprints()
    )


def test_assembly_pad_keepouts_reach_pinned_router(tmp_path):
    """Mask edge + off-centre drill, same-net pads and reviewed thermal lands."""
    pcbnew = pytest.importorskip("pcbnew")
    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not python.is_file() or not (root / "py_router" / "obstacle_map.py").is_file():
        pytest.skip("pinned KiCadRoutingTools runtime not installed")
    source = tmp_path / "source.kicad_pcb"
    source.write_text("""(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal)
          (36 "B.SilkS" user "b.silkscreen") (37 "F.SilkS" user "f.silkscreen")
          (38 "B.Mask" user) (39 "F.Mask" user))
  (setup (pad_to_mask_clearance 0.05))
  (net 0 "") (net 1 "SAME")
  (footprint "Test:Ordinary" (layer "F.Cu") (at 10 10)
    (property "Reference" "R1" (at 0 0) (layer "F.SilkS"))
    (property "Value" "R" (at 0 0) (layer "F.Fab"))
    (pad "1" smd rect (at 0 0) (size 1 1)
      (layers "F.Cu" "F.Mask") (net 1 "SAME")))
  (footprint "Test:BULETM-SMD_ESPRESSIF_ESP32-S3-MINI-1U-N8" (layer "F.Cu") (at 20 10)
    (property "Reference" "U1" (at 0 0) (layer "F.SilkS"))
    (property "Value" "ESP32-S3-MINI-1U-N8" (at 0 0) (layer "F.Fab"))
    (pad "61" smd rect (at 0 0) (size 1.2 1.2)
      (layers "F.Cu" "F.Mask") (net 1 "SAME")))
  (zone (net 0) (net_name "") (layer "F.Cu") (name "authored")
    (hatch edge 0.5) (keepout (tracks not_allowed) (vias not_allowed)
      (pads allowed) (copperpour allowed) (footprints allowed))
    (polygon (pts (xy 30 10) (xy 31 10) (xy 31 11) (xy 30 11)))))
""")
    original = source.read_bytes()
    staged = tmp_path / "staged.kicad_pcb"
    staged.write_bytes(original)
    assert rb._board_has_assembly_pads(staged)
    zone_name = rb._PAD_VIA_KEEPOUT_ZONE_NAME + " test"
    summary = rb._stamp_pad_via_keepouts(staged, zone_name)
    assert summary == {"pads": 1, "zones": 1, "authorized_thermal_pads": 1}
    assert source.read_bytes() == original

    # Use the pinned parser and its real obstacle-map rasterizer, not a
    # reimplementation of its interpretation of KiCad rule areas.
    probe = """
import sys
from types import SimpleNamespace
sys.path[:0] = [sys.argv[1] + '/py_router', sys.argv[1] + '/py_tools']
from kicad_parser import extract_keepouts
from obstacle_map import add_rule_area_keepout_obstacles
from routing_config import GridRouteConfig
areas = extract_keepouts(open(sys.argv[2]).read())
assert len(areas) == 2
assert any(a['tracks_allowed'] and not a['vias_allowed'] for a in areas)
class Capture:
    def __init__(self):
        self.vias = set()
        self.tracks = set()
    def add_blocked_vias_batch(self, cells):
        self.vias.update(map(tuple, cells.tolist()))
    def add_blocked_cells_batch(self, cells):
        self.tracks.update(map(tuple, cells.tolist()))
c = Capture()
pcb = SimpleNamespace(board_info=SimpleNamespace(keepouts=areas, board_bounds=None))
cfg = GridRouteConfig(grid_step=0.05, via_size=0.6, via_drill=0.3, clearance=0.15)
add_rule_area_keepout_obstacles(c, pcb, cfg)
assert (200, 200) in c.vias       # own-net pad centre
assert (213, 200) in c.vias       # x=10.65: drill intersects mask at x=10.55
assert not any(x == 200 and y == 200 for x, y, *_ in c.tracks)
assert (400, 200) not in c.vias   # explicitly authorised thermal pad
assert (600, 200) in c.vias       # authored keepout survives
"""
    subprocess.run([str(python), "-c", probe, str(root), str(staged)], check=True)
    rb._strip_pad_via_keepouts(staged, zone_name)
    board = pcbnew.LoadBoard(str(staged))
    assert [z.GetZoneName() for z in board.Zones() if z.GetIsRuleArea()] == ["authored"]
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(10.65), pcbnew.FromMM(10)))
    via.SetWidth(pcbnew.FromMM(0.6))
    via.SetDrill(pcbnew.FromMM(0.3))
    via.SetNetCode(1)
    board.Add(via)
    board.Save(str(staged))
    with pytest.raises(RuntimeError, match="Via drill intersects assembly mask"):
        rb._reject_vias_in_assembly_openings(staged)


def test_krt_command_passes_configured_grid_step(tmp_path):
    """The escape knob reaches KRT only when configured (default = router's 0.1)."""
    board = tmp_path / "in.kicad_pcb"
    board.write_text("(kicad_pcb)\n")
    base = {
        "kicad_routing_tools_path": str(tmp_path / "krt"),
        "kicad_routing_tools_layers": ["F.Cu", "B.Cu"],
    }

    cmd = rb._krt_command(
        str(board),
        str(tmp_path / "out.kicad_pcb"),
        {**base, "kicad_routing_tools_grid_step_mm": 0.05},
        tmp_path / "fab.txt",
    )
    assert "--grid-step" in cmd
    assert cmd[cmd.index("--grid-step") + 1] == "0.05"

    default_cmd = rb._krt_command(
        str(board), str(tmp_path / "out.kicad_pcb"), base, tmp_path / "fab.txt"
    )
    assert "--grid-step" not in default_cmd


def test_default_config_leaves_grid_step_to_the_router():
    from kicraft.autoplacer.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kicad_routing_tools_grid_step_mm"] is None


def test_grid_step_env_override_parses_and_rejects_junk(monkeypatch):
    """The escape knob is operable from the environment, and a typo fails loudly."""
    from kicraft.autoplacer.config import _optional_env_float

    monkeypatch.delenv("KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM", raising=False)
    assert _optional_env_float("KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM") is None
    monkeypatch.setenv("KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM", "0.05")
    assert _optional_env_float("KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM") == 0.05
    monkeypatch.setenv("KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM", "bogus")
    with pytest.raises(ValueError, match="must be a number"):
        _optional_env_float("KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM")


def test_route_timeout_knobs_are_env_overridable(monkeypatch):
    from kicraft.autoplacer import config as cfgmod

    assert cfgmod._optional_env_float("NOPE", 300) == 300
    monkeypatch.setenv("NOPE", "0")
    assert cfgmod._optional_env_float("NOPE", 300) == 0.0


def test_parent_route_does_not_inherit_the_leaf_grid_step(monkeypatch, tmp_path):
    """A leaf-tuned routing grid must not be forced on the (much larger) parent.

    Driving 0.05 globally in the 2026-10-02 experiment made the parent route
    exceed a 600 s cap; the parent's grid is its own key.
    """
    import kicraft.autoplacer.kicad_routing_tools as krt
    import kicraft.autoplacer.routing_board as board_utils
    import kicraft.cli._compose_route as compose_route

    stamped = tmp_path / "parent_stamped.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n")
    seen: list[dict] = []

    def fake_route(source, output, config):
        seen.append(dict(config))
        Path(output).write_text("(kicad_pcb routed)\n")
        return {"backend": "kicad-routing-tools", "returncode": 0}

    monkeypatch.setattr(krt, "route_with_kicad_routing_tools", fake_route)
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda p: {"traces": [], "vias": []})
    monkeypatch.setattr(
        board_utils, "validate_routed_board", lambda *a, **k: {"accepted": True, "drc": {}}
    )
    state = SimpleNamespace(composition=SimpleNamespace())
    base = {
        "gnd_zone_net": "",
        "power_plane_enabled": False,
        "signal_unconnected_repair_enabled": False,
        "illegal_geometry_repair_enabled": False,
        "kicad_routing_tools_grid_step_mm": 0.05,
    }

    compose_route._route_parent_board(stamped, state, tmp_path, dict(base))
    assert seen[-1]["kicad_routing_tools_grid_step_mm"] is None

    compose_route._route_parent_board(
        stamped, state, tmp_path, {**base, "kicad_routing_tools_parent_grid_step_mm": 0.1}
    )
    assert seen[-1]["kicad_routing_tools_grid_step_mm"] == 0.1
