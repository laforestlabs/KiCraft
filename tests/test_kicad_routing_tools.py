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


def test_two_ounce_profile_reaches_native_rules_and_router(tmp_path):
    from kicraft.autoplacer.fab_profile import (
        fab_floors, fanout_via, require_two_ounce_project_rules,
        stamp_fab_floors_into_pro,
    )
    from kicraft.design.synthesis.kicad_pro import write_kicad_pro

    cfg = {"thermal_requirements": [{"outer_copper_oz": 2.0}],
           "fab_capability": {"min_track_mm": 0.1, "min_clearance_mm": 0.1}}
    project = write_kicad_pro(tmp_path, "B", SimpleNamespace(power_nets=[]), fab_config=cfg)
    body = json.loads(project.read_text())
    rules = body["board"]["design_settings"]["rules"]
    assert rules["min_track_width"] == rules["min_clearance"] == 0.17
    assert all(c["clearance"] >= 0.17 for c in body["net_settings"]["classes"])
    assert fab_floors(cfg) == {"track_mm": 0.17, "clearance_mm": 0.17}
    assert fanout_via(cfg) == (0.36, 0.15)
    require_two_ounce_project_rules(project, cfg)
    routing = rb._project_routing_floors(project, cfg)
    assert routing["track_width"] >= 0.17 and routing["clearance"] >= 0.17

    rules["min_track_width"] = 0.127
    body["net_settings"]["classes"][0]["clearance"] = 0.153
    project.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="below"):
        require_two_ounce_project_rules(project, cfg)
    assert stamp_fab_floors_into_pro(project, cfg)
    require_two_ounce_project_rules(project, cfg)
    assert fab_floors({})["track_mm"] == 0.127


def test_preflight_requires_configured_checkout():
    with pytest.raises(rb.KicadRoutingToolsUnavailableError, match="path is unset"):
        rb.preflight_kicad_routing_tools({})








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
    monkeypatch.setattr(rb, "_lock_input_copper_nets", lambda _: [])
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda _: _copper())

    with pytest.raises(RuntimeError, match=match):
        rb._route_krt_pass(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
            },
            nets=["*"], track_width=0.2,
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
    monkeypatch.setattr(rb, "_lock_input_copper_nets", lambda _: [])
    monkeypatch.setattr(rb.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(board_utils, "import_routed_copper", lambda _: _copper())

    with pytest.raises(RuntimeError, match="timed out after 1s"):
        rb._route_krt_pass(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
                "kicad_routing_tools_timeout_s": 1,
            },
            nets=["*"], track_width=0.2,
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
    monkeypatch.setattr(rb, "_lock_input_copper_nets", lambda _: [])
    monkeypatch.setattr(board_utils, "import_routed_copper", fake_import)

    with pytest.raises(rb.RoutingCopperPreservationError) as caught:
        rb._route_krt_pass(
            str(input_board),
            str(output_board),
            {
                                "kicad_routing_tools_path": str(root),
            },
            nets=["*"], track_width=0.2,
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
    monkeypatch.setattr(rb, "_lock_input_copper_nets", lambda _: [])
    monkeypatch.setattr(board_utils, "import_routed_copper", fake_import)
    return input_board, output_board, root


def test_krt_route_rejects_declared_but_unproven_input_prune(monkeypatch, tmp_path):
    """KRT also calls ripped-net removals dead ends; the label proves nothing."""
    input_board, output_board, root = _run_bridge_with_missing_input_trace(
        monkeypatch,
        tmp_path,
        "Removed 1 dead-end input segment(s) from the output\n"
        'JSON_SUMMARY: {"successful": 1}\n',
    )
    with pytest.raises(rb.RoutingCopperPreservationError) as caught:
        rb._route_krt_pass(
            str(input_board),
            str(output_board),
            {"kicad_routing_tools_path": str(root)},
            nets=["*"], track_width=0.2,
        )
    assert caught.value.stats["input_copper_preservation"]["traces"]["missing_count"] == 1


def test_krt_route_rejects_undeclared_missing_input_copper(monkeypatch, tmp_path):
    """Same missing trace, no declared cleanup: still a custody failure."""
    input_board, output_board, root = _run_bridge_with_missing_input_trace(
        monkeypatch,
        tmp_path,
        'JSON_SUMMARY: {"successful": 1}\n',
    )
    with pytest.raises(rb.RoutingCopperPreservationError) as caught:
        rb._route_krt_pass(
            str(input_board),
            str(output_board),
            {"kicad_routing_tools_path": str(root)},
            nets=["*"], track_width=0.2,
        )
    assert caught.value.stats["input_copper_preservation"]["traces"]["missing_count"] == 1
    assert "declared dead-end removals=0" in str(caught.value)


@pytest.mark.parametrize("failure", ["lost_prior_copper", "timeout", "nonzero"])
def test_width_pass_failure_never_publishes_partial_board(monkeypatch, tmp_path, failure):
    import kicraft.autoplacer.routing_board as board_utils

    root = tmp_path / "krt"
    root.mkdir()
    source = tmp_path / "input.kicad_pcb"
    output = tmp_path / "output.kicad_pcb"
    source.write_text("input")
    _write_project(source)
    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb, "_width_groups", lambda *args: [(0.5, ["PWR"]), (0.2, ["SIG"])])
    monkeypatch.setattr(rb, "_lock_input_copper_nets", lambda _: [])
    monkeypatch.setattr(rb.os, "killpg", lambda *args: None)

    def copper(path):
        name = Path(path).name
        traces = [_trace_span(1)]
        if "pass-01" in name:
            traces.append(_trace_span(9))
        return {"traces": traces, "vias": []}

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            self.path = Path(command[3])
            self.second = "pass-02" in self.path.name
            self.returncode = 7 if self.second and failure == "nonzero" else 0
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            self.path.write_text("diagnostic")
            if self.second and failure == "timeout" and self.calls == 1:
                raise subprocess.TimeoutExpired("route", timeout)
            return ('JSON_SUMMARY: {"successful": 1, "failed": 0}\n', "")

    monkeypatch.setattr(board_utils, "import_routed_copper", copper)
    monkeypatch.setattr(rb.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError):
        rb.route_with_kicad_routing_tools(
            str(source), str(output), {"kicad_routing_tools_path": str(root)}
        )
    assert not output.exists()
    assert (tmp_path / "output.krt-pass-01.kicad_pcb").exists()
    assert (tmp_path / "output.krt-pass-02.kicad_pcb").exists()
    assert (tmp_path / "output.krt-pass-02.router.stdout.log").exists()


def test_width_groups_use_router_classes_and_reject_ambiguous_names(tmp_path):
    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not python.is_file() or not (root / "py_router" / "list_nets.py").is_file():
        pytest.skip("pinned KiCadRoutingTools runtime not installed")
    board = tmp_path / "board.kicad_pcb"
    board.write_text('''(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (net 0 "") (net 1 "PWR") (net 2 "SIG") (net 3 "/sheet/PWR"))
''')
    project = _write_project(board)
    data = json.loads(project.read_text())
    data["net_settings"]["classes"].append({"name": "Power", "track_width": 0.5})
    data["net_settings"]["netclass_assignments"] = {"PWR": "Power", "/sheet/PWR": "Power"}
    project.write_text(json.dumps(data))
    runtime = dict(_runtime(root), python=str(python))
    assert rb._width_groups(board, runtime, 0.25, 30) == [
        (0.5, ["/sheet/PWR", "PWR"]), (0.25, ["SIG"]),
    ]
    data["net_settings"]["netclass_assignments"] = {"PWR": "Power"}
    project.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="ambiguous net selection"):
        rb._width_groups(board, runtime, 0.25, 30)


def test_ground_last_uses_native_selection_and_preserves_class_widths(tmp_path):
    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not python.is_file() or not (root / "py_router" / "list_nets.py").is_file():
        pytest.skip("pinned KiCadRoutingTools runtime not installed")
    board = tmp_path / "board.kicad_pcb"
    board.write_text('''(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (net 0 "") (net 1 "PWR") (net 2 "GND") (net 3 "SIG")
  (net 4 "AGND") (net 5 "GND_SENSE"))
''')
    project = _write_project(board)
    data = json.loads(project.read_text())
    data["net_settings"]["classes"].append({"name": "Power", "track_width": 0.5})
    data["net_settings"]["netclass_assignments"] = {
        "PWR": "Power", "GND": "Power", "AGND": "Power",
    }
    project.write_text(json.dumps(data))
    runtime = dict(_runtime(root), python=str(python))
    config = {"local_bypasses": [{"ground_return": {
        "owner_ref": "U1", "owner_pin": "1", "member_pin": "2", "net": "AGND",
    }}]}
    assert rb._width_groups(board, runtime, 0.25, 30, config) == [
        (0.5, ["PWR"]), (0.25, ["GND_SENSE", "SIG"]), (0.5, ["AGND", "GND"]),
    ]
    assert rb._width_groups(board, runtime, 0.25, 30, {"gnd_zone_net": ""}) == [
        (0.5, ["AGND", "GND", "PWR"]), (0.25, ["GND_SENSE", "SIG"]),
    ]

    # GND expands to both names in the pinned CLI. Splitting only the
    # configured plane from its namesake would route that other net twice.
    board.write_text(board.read_text().replace(
        '(net 5 "GND_SENSE")', '(net 5 "GND_SENSE") (net 6 "/sheet/GND")',
    ))
    data["net_settings"]["netclass_assignments"]["/sheet/GND"] = "Power"
    project.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="ambiguous net selection"):
        rb._width_groups(board, runtime, 0.25, 30, config)
    assert rb._width_groups(board, runtime, 0.25, 30, {
        **config,
        "local_bypasses": config["local_bypasses"] + [{"ground_return": {
            "owner_ref": "U2", "owner_pin": "1", "member_pin": "2", "net": "/sheet/GND",
        }}],
    }) == [
        (0.5, ["PWR"]), (0.25, ["GND_SENSE", "SIG"]),
        (0.5, ["/sheet/GND", "AGND", "GND"]),
    ]


def test_width_passes_share_one_deadline(monkeypatch, tmp_path):
    root = tmp_path / "krt"
    root.mkdir()
    source = tmp_path / "input.kicad_pcb"
    output = tmp_path / "output.kicad_pcb"
    source.write_text("input")
    _write_project(source)
    clock = [0.0]
    monkeypatch.setattr(rb.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rb, "preflight_kicad_routing_tools", lambda _: _runtime(root))
    monkeypatch.setattr(rb, "_width_groups", lambda *args: [(0.5, ["PWR"]), (0.2, ["SIG"])])

    def pass_route(source_path, output_path, config, **kwargs):
        Path(output_path).write_text("first pass")
        clock[0] += 11
        return {}

    monkeypatch.setattr(rb, "_route_krt_pass", pass_route)
    with pytest.raises(rb.KicadRoutingToolsTimeoutError, match="before pass 2"):
        rb.route_with_kicad_routing_tools(
            str(source), str(output), {"kicad_routing_tools_timeout_s": 10}
        )
    assert not output.exists()
    assert (tmp_path / "output.krt-pass-01.kicad_pcb").read_text() == "first pass"


def test_temporary_copper_locks_protect_krt_nets_and_preserve_authored_locks(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not python.is_file() or not (root / "py_router" / "protected_nets.py").is_file():
        pytest.skip("pinned KiCadRoutingTools runtime not installed")
    board = tmp_path / "input.kicad_pcb"
    board.write_text("""(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (net 0 "") (net 1 "GND") (net 2 "SIG")
  (footprint "Test:P1" (layer "F.Cu") (at 10 10)
    (property "Reference" "P1" (at 0 0) (layer "F.SilkS"))
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask") (net 1 "GND")))
  (footprint "Test:P2" (layer "F.Cu") (at 20 10)
    (property "Reference" "P2" (at 0 0) (layer "F.SilkS"))
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu" "F.Mask") (net 1 "GND")))
  (segment (start 10 10) (end 11 10) (width 0.2) (layer "F.Cu")
    (net 1) (uuid "11111111-1111-4111-8111-111111111111"))
  (segment (start 10 12) (end 11 12) (width 0.2) (layer "F.Cu")
    (locked yes) (net 2) (uuid "22222222-2222-4222-8222-222222222222")))
""")
    original = board.read_bytes()
    staged = tmp_path / "staged.kicad_pcb"
    staged.write_bytes(original)
    temporary = rb._lock_input_copper_nets(staged)
    assert temporary == ["11111111-1111-4111-8111-111111111111"]
    probe = """
import sys
sys.path[:0] = [sys.argv[1] + '/py_router', sys.argv[1] + '/py_tools']
from kicad_parser import parse_kicad_pcb
from protected_nets import protection_map, filter_rippable_names
board = parse_kicad_pcb(sys.argv[2])
protected = protection_map(board, sys.argv[2])
assert protected['GND'] == protected['SIG'] == 'locked'
assert filter_rippable_names(['GND', 'SIG'], protected,
                             override_patterns=['GND', 'SIG']) == []
from routing_common import filter_already_routed
from routing_config import GridRouteConfig
to_route, _ = filter_already_routed(board, [('GND', 1)], GridRouteConfig(),
                                    fragment_gate=True)
assert ('GND', 1) in to_route
"""
    subprocess.run([str(python), "-c", probe, str(root), str(staged)], check=True)
    rb._restore_input_copper_locks(staged, temporary)
    loaded = pcbnew.LoadBoard(str(staged))
    tracks = {t.m_Uuid.AsString(): t.IsLocked() for t in loaded.GetTracks()}
    assert tracks == {
        "11111111-1111-4111-8111-111111111111": False,
        "22222222-2222-4222-8222-222222222222": True,
    }
    assert board.read_bytes() == original


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
    rb._strip_router_keepouts(staged, zone_name)
    board = pcbnew.LoadBoard(str(staged))
    assert [z.GetZoneName() for z in board.Zones() if z.GetIsRuleArea()] == ["authored"]
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(10.65), pcbnew.FromMM(10)))
    via.SetWidth(pcbnew.FromMM(0.6))
    via.SetDrill(pcbnew.FromMM(0.3))
    via.SetNetCode(1)
    board.Add(via)
    board.Save(str(staged))
    with pytest.raises(RuntimeError, match="Drill intersects assembly mask or paste"):
        rb._reject_drills_in_assembly_openings(staged)


def test_through_hole_mask_keepouts_retain_both_face_coordinates(tmp_path):
    """Both mask contours must outlive the temporary KiCad shape owners."""
    pcbnew = pytest.importorskip("pcbnew")
    source = tmp_path / "source.kicad_pcb"
    source.write_text("""(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal)
          (36 "B.SilkS" user "b.silkscreen") (37 "F.SilkS" user "f.silkscreen")
          (38 "B.Mask" user) (39 "F.Mask" user))
  (setup (pad_to_mask_clearance 0.05))
  (net 0 "")
  (footprint "Test:Connector" (layer "F.Cu") (at 10 10 37)
    (property "Reference" "J1" (at 0 0) (layer "F.SilkS"))
    (property "Value" "Connector" (at 0 0) (layer "F.Fab"))
    (pad "1" thru_hole oval (at 0 0 37) (size 2 3) (drill 0.8)
      (layers "*.Cu" "*.Mask"))
    (pad "2" thru_hole rect (at 5 0 37) (size 2 3) (drill 0.8)
      (layers "*.Cu" "*.Mask"))))
""")

    def ring_key(points):
        # KiCad may change a closed contour's starting vertex or winding.
        return min(
            tuple(sequence[index:] + sequence[:index])
            for sequence in (points, list(reversed(points)))
            for index in range(len(sequence))
        )

    board = pcbnew.LoadBoard(str(source))
    expected = []
    for footprint in board.Footprints():
        for pad in footprint.Pads():
            for copper, mask in ((pcbnew.F_Cu, pcbnew.F_Mask), (pcbnew.B_Cu, pcbnew.B_Mask)):
                shape = pcbnew.SHAPE_POLY_SET()
                pad.TransformShapeToPolygon(
                    shape, copper, max(0, pad.GetSolderMaskExpansion(mask)),
                    pcbnew.FromMM(0.001), pcbnew.ERROR_OUTSIDE,
                )
                for index in range(shape.OutlineCount()):
                    contour = shape.Outline(index)
                    coordinates = []
                    for point_index in range(contour.PointCount()):
                        point = contour.CPoint(point_index)
                        coordinates.append((point.x, point.y))
                    expected.append(ring_key(coordinates))

    original = source.read_bytes()
    staged = tmp_path / "staged.kicad_pcb"
    staged.write_bytes(original)
    zone_name = rb._PAD_VIA_KEEPOUT_ZONE_NAME + " dual-face"
    rb._stamp_pad_via_keepouts(staged, zone_name)
    routed = pcbnew.LoadBoard(str(staged))
    actual = []
    for zone in routed.Zones():
        if not zone.GetIsRuleArea() or zone.GetZoneName() != zone_name:
            continue
        assert zone.GetDoNotAllowVias() and not zone.GetDoNotAllowTracks()
        contour = zone.Outline().Outline(0)
        coordinates = []
        for index in range(contour.PointCount()):
            point = contour.CPoint(index)
            coordinates.append((point.x, point.y))
        actual.append(ring_key(coordinates))
    assert sorted(actual) == sorted(expected)
    assert source.read_bytes() == original


def test_assembly_drill_guard_rejects_footprint_hole_in_smd_land(tmp_path):
    pytest.importorskip("pcbnew")
    board = tmp_path / "thermal-hole.kicad_pcb"
    board.write_text("""(kicad_pcb (version 20240108) (generator pcbnew)
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal)
          (36 "B.SilkS" user "b.silkscreen") (37 "F.SilkS" user "f.silkscreen")
          (38 "B.Mask" user) (39 "F.Mask" user) (35 "F.Paste" user))
  (net 0 "") (net 1 "GND")
  (footprint "Test:Driver" (layer "F.Cu") (at 10 10)
    (property "Reference" "U1" (at 0 0) (layer "F.SilkS"))
    (property "Value" "Driver" (at 0 0) (layer "F.Fab"))
    (pad "17" smd rect (at 0 0) (size 2.74 2.74)
      (layers "F.Cu" "F.Paste" "F.Mask") (net 1 "GND"))
    (pad "17" thru_hole circle (at 0.5 0.5) (size 0.61 0.61) (drill 0.3)
      (layers "*.Cu" "*.Mask") (net 1 "GND"))))
""")
    with pytest.raises(RuntimeError, match="U1.17 drill -> U1.17"):
        rb._reject_drills_in_assembly_openings(board)


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




def test_local_ground_reservations_protect_only_native_back_fill(tmp_path):
    """Parent signals cannot cut a local return, or lose an existing pad window."""
    import shutil

    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.fab_profile import stamp_fab_floors_into_pro
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not (root / "py_router/kicad_parser.py").is_file() or not python.is_file():
        pytest.skip("pinned router is not installed")
    source = tmp_path / "local_plane.kicad_pcb"
    board = pcbnew.NewBoard(str(source))
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    signal_net = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(ground)
    board.Add(signal_net)
    mm = pcbnew.FromMM
    for start, end in (
        ((1, 1), (20, 1)), ((20, 1), (20, 20)),
        ((20, 20), (1, 20)), ((1, 20), (1, 1)),
    ):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(pcbnew.VECTOR2I(*(mm(v) for v in start)))
        edge.SetEnd(pcbnew.VECTOR2I(*(mm(v) for v in end)))
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetWidth(mm(0.05))
        board.Add(edge)
    for ref, number, xy, net, pth in (
        ("U1", "1", (10, 10), ground, False),
        ("C1", "2", (14, 10), ground, False),
        ("J1", "1", (12, 12), signal_net, True),
    ):
        footprint = pcbnew.FOOTPRINT(board)
        footprint.SetReference(ref)
        board.Add(footprint)
        pad = pcbnew.PAD(footprint)
        pad.SetNumber(number)
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        pad.SetSize(pcbnew.VECTOR2I(mm(1), mm(1)))
        pad.SetPosition(pcbnew.VECTOR2I(*(mm(v) for v in xy)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH if pth else pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.PTHMask() if pth else pcbnew.PAD.SMDMask())
        if pth:
            pad.SetDrillSize(pcbnew.VECTOR2I(mm(0.3), mm(0.3)))
        footprint.Add(pad)
        pad.SetNet(net)
    for x in (10, 14):
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(mm(x), mm(11.5)))
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetWidth(mm(0.6))
        via.SetDrill(mm(0.3))
        via.SetNet(ground)
        board.Add(via)
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(pcbnew.VECTOR2I(mm(x), mm(10)))
        track.SetEnd(pcbnew.VECTOR2I(mm(x), mm(11.5)))
        track.SetLayer(pcbnew.F_Cu)
        track.SetWidth(mm(0.2))
        track.SetNet(ground)
        board.Add(track)
    pcbnew.SaveBoard(str(source), board)
    cfg = {
        "local_bypasses": [{
            "owner_ref": "U1", "bypass_ref": "C1",
            "ground_return": {"net": "GND", "owner_ref": "U1", "owner_pin": "1",
                              "member_pin": "2", "max_via_distance_mm": 3},
        }],
        "thermal_requirements": [{
            "owner_ref": "U1", "ground_pins": ["1"], "outer_copper_oz": 2,
            "local_radius_mm": 2,
        }],
    }
    project = _write_project(source)
    stamp_fab_floors_into_pro(project, cfg)
    staged = tmp_path / "staged.kicad_pcb"
    shutil.copy2(source, staged)
    shutil.copy2(project, staged.with_suffix(".kicad_pro"))
    before = source.read_bytes()
    name = "KiCraft local ground reservation test"
    rb._stamp_local_ground_keepouts(staged, cfg, name, router_clearance_mm=0.17)
    assert source.read_bytes() == before
    assert not audit_local_copper(pcbnew.LoadBoard(str(staged)), cfg["local_bypasses"])
    probe = """
import sys
from types import SimpleNamespace
sys.path[:0] = [sys.argv[1] + '/py_router', sys.argv[1] + '/py_tools']
from kicad_parser import extract_keepouts
from obstacle_map import add_rule_area_keepout_obstacles
from routing_config import GridRouteConfig
areas = extract_keepouts(open(sys.argv[2]).read())
class Capture:
    def __init__(self):
        self.tracks, self.vias = set(), set()
    def add_blocked_cells_batch(self, rows):
        self.tracks.update(tuple(int(v) for v in row) for row in rows)
    def add_blocked_vias_batch(self, rows):
        self.vias.update(tuple(int(v) for v in row) for row in rows)
c = Capture()
pcb = SimpleNamespace(board_info=SimpleNamespace(keepouts=areas, board_bounds=None))
cfg = GridRouteConfig(grid_step=0.05, layers=['F.Cu','B.Cu'], track_width=0.2,
                      via_size=0.6, via_drill=0.3, clearance=0.17)
add_rule_area_keepout_obstacles(c, pcb, cfg)
# A vertical crossing between the via anchors must intersect the reserved return.
assert any((240, y, 1) in c.tracks for y in range(20, 381))
assert all((240, y, 0) not in c.tracks for y in range(20, 381))
# The signal PTH and its entire downward escape remain usable.
assert all((240, y, 1) not in c.tracks for y in range(240, 301))
assert (240, 240) not in c.vias
assert (80, 80, 1) not in c.tracks # No whole-board B.Cu reservation.
"""
    subprocess.run([str(python), "-c", probe, str(root), str(staged)], check=True)
    rb._strip_router_keepouts(staged, name)
    delivered = pcbnew.LoadBoard(str(staged))
    assert not any(zone.GetIsRuleArea() for zone in delivered.Zones())
    assert not audit_local_copper(delivered, cfg["local_bypasses"])


def _neck_ground_board(tmp_path, center):
    """A native filled dumbbell, with a real foreign-net hole in one room."""
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    mm = pcbnew.FromMM
    point = lambda x, y: pcbnew.VECTOR2I(mm(x), mm(y))
    board = pcbnew.NewBoard(str(tmp_path / "neck.kicad_pcb"))
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    signal = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(ground)
    board.Add(signal)
    for a, b in zip(((0, 0), (20, 0), (20, 20), (0, 20)),
                    ((20, 0), (20, 20), (0, 20), (0, 0))):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(point(*a))
        edge.SetEnd(point(*b))
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetWidth(mm(0.05))
        board.Add(edge)
    for ref, number, x in (("U1", "1", 5), ("C1", "2", 15)):
        footprint = pcbnew.FOOTPRINT(board)
        footprint.SetReference(ref)
        board.Add(footprint)
        pad = pcbnew.PAD(footprint)
        pad.SetNumber(number)
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        pad.SetSize(point(1, 1))
        pad.SetPosition(point(x, center))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        footprint.Add(pad)
        pad.SetNet(ground)
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(point(x + 1, center))
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetWidth(mm(0.6))
        via.SetDrill(mm(0.3))
        via.SetNet(ground)
        board.Add(via)
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(point(x, center))
        track.SetEnd(point(x + 1, center))
        track.SetLayer(pcbnew.F_Cu)
        track.SetWidth(mm(0.2))
        track.SetNet(ground)
        board.Add(track)
    footprint = pcbnew.FOOTPRINT(board)
    footprint.SetReference("J1")
    board.Add(footprint)
    pad = pcbnew.PAD(footprint)
    pad.SetNumber("1")
    pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    pad.SetSize(point(1, 1))
    pad.SetPosition(point(5, center + 1.5))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetLayerSet(pcbnew.PAD.PTHMask())
    pad.SetDrillSize(point(0.3, 0.3))
    footprint.Add(pad)
    pad.SetNet(signal)

    zone = pcbnew.ZONE(board)
    zone.SetNet(ground)
    zone.SetLayer(pcbnew.B_Cu)
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    zone.SetLocalClearance(mm(0.30))
    zone.SetMinThickness(mm(0.25))
    zone.Outline().NewOutline()
    for x, y in ((2, 3), (8, 3), (8, center - 0.15),
                 (12, center - 0.15), (12, 3), (18, 3),
                 (18, 17), (12, 17), (12, center + 0.15),
                 (8, center + 0.15), (8, 17), (2, 17)):
        zone.Outline().Append(mm(x), mm(y))
    board.Add(zone)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    spec = {"owner_ref": "U1", "bypass_ref": "C1",
            "ground_return": {"net": "GND", "owner_ref": "U1", "owner_pin": "1",
                              "member_pin": "2", "max_via_distance_mm": 3}}
    assert not audit_local_copper(board, [spec])
    assert not zone.GetFilledPolysList(pcbnew.B_Cu).Contains(point(5, center + 1.5))
    return board, spec, signal


def test_leaf_prepares_ground_before_signal_copper_and_strips_only_its_mask(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.brain.leaf_routing import (
        _finish_leaf_ground_planes, _prepare_leaf_ground_reservations, _validate_leaf_input,
    )
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, _ = _neck_ground_board(tmp_path, 10.125)
    for track in list(board.GetTracks()):
        board.Delete(track)
    for zone in list(board.Zones()):
        board.Delete(zone)
    authored = pcbnew.ZONE(board)
    authored.SetLayer(pcbnew.F_Cu)
    authored.SetIsRuleArea(True)
    authored.SetZoneName("authored keepout")
    authored.SetDoNotAllowTracks(True)
    authored.Outline().NewOutline()
    for x, y in ((1, 1), (2, 1), (2, 2), (1, 2)):
        authored.Outline().Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    board.Add(authored)
    path = tmp_path / "leaf.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    project = _write_project(path)
    rules = project.read_bytes()
    cfg = {"local_bypasses": [spec], "gnd_zone_net": "GND",
           "thermal_requirements": [{"owner_ref": "ABSENT", "outer_copper_oz": 2.0}]}

    marker, pins = _prepare_leaf_ground_reservations(path, path, cfg)
    assert pins == {("U1", "1"), ("C1", "2")}
    prepared = pcbnew.LoadBoard(str(path))
    masks = [zone for zone in prepared.Zones() if zone.GetZoneName() == marker]
    assert masks and all(zone.GetIsRuleArea() for zone in masks)
    ground_zones = [zone for zone in prepared.Zones()
                    if not zone.GetIsRuleArea() and zone.GetNetname() == "GND"]
    assert ground_zones and all(zone.GetLocalClearance() >= pcbnew.FromMM(0.17)
                                for zone in ground_zones)
    assert not audit_local_copper(prepared, [spec])
    assert sum(isinstance(track, pcbnew.PCB_VIA) for track in prepared.GetTracks()) >= 2
    assert project.read_bytes() == rules

    # Signal escapes are stamped after the original fill. The input DRC must
    # refill its mask-free copy, not report these stale GND polygons as shorts.
    signal = pcbnew.PCB_TRACK(prepared)
    signal.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(10), pcbnew.FromMM(5)))
    signal.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(12), pcbnew.FromMM(5)))
    signal.SetWidth(pcbnew.FromMM(0.2))
    signal.SetLayer(pcbnew.F_Cu)
    signal.SetNet(prepared.FindNet("SIG"))
    prepared.Add(signal)
    pcbnew.SaveBoard(str(path), prepared)
    project.write_bytes(rules)
    before_validation = path.read_bytes()
    validation = _validate_leaf_input(path, marker, cfg=cfg)
    assert validation["drc"]["clearance"] == 0
    assert path.read_bytes() == before_validation
    assert project.read_bytes() == rules

    rb._strip_router_keepouts(path, marker)
    _finish_leaf_ground_planes(path, cfg)
    finished = pcbnew.LoadBoard(str(path))
    assert not any(zone.GetZoneName() == marker for zone in finished.Zones())
    assert any(zone.GetZoneName() == "authored keepout" for zone in finished.Zones())
    assert not audit_local_copper(finished, [spec])
    assert project.read_bytes() == rules


def _neck_router_probe(board_path, mode):
    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not (root / "py_router/kicad_parser.py").is_file() or not python.is_file():
        pytest.skip("pinned router is not installed")
    probe = """
import sys
from types import SimpleNamespace
sys.path[:0] = [sys.argv[1] + '/py_router', sys.argv[1] + '/py_tools']
from kicad_parser import extract_keepouts
from obstacle_map import add_rule_area_keepout_obstacles
from routing_config import GridRouteConfig
class Capture:
    def __init__(self):
        self.tracks, self.vias = set(), set()
    def add_blocked_cells_batch(self, rows):
        self.tracks.update(tuple(int(v) for v in row) for row in rows)
    def add_blocked_vias_batch(self, rows):
        self.vias.update(tuple(int(v) for v in row) for row in rows)
c = Capture()
areas = extract_keepouts(open(sys.argv[2]).read())
pcb = SimpleNamespace(board_info=SimpleNamespace(keepouts=areas, board_bounds=None))
cfg = GridRouteConfig(grid_step=0.025, layers=['F.Cu', 'B.Cu'], track_width=0.2,
                      via_size=0.6, via_drill=0.3, clearance=0.17)
add_rule_area_keepout_obstacles(c, pcb, cfg)
if sys.argv[3] == 'offgrid':
    # Every transverse path through the only neck must encounter the witness.
    assert any((400, y, 1) in c.tracks for y in range(380, 430))
    assert all((400, y, 0) not in c.tracks for y in range(380, 430))
else:
    # Both 0.20 mm signal tracks would erode the source fill into the neck.
    for y in (389, 421):  # centre y=10.125 +/- 0.400 mm
        assert any((x, y, 1) in c.tracks for x in range(340, 461))
    # A real PTH hole remains open, including for via placement.
    assert (200, 465, 1) not in c.tracks
    assert (200, 465) not in c.vias
"""
    subprocess.run([str(python), "-c", probe, str(root), str(board_path), mode], check=True)


def test_ground_witness_detours_around_native_hole(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_local_ground_planes
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, _ = _neck_ground_board(tmp_path, 10.125)
    mm = pcbnew.FromMM
    point = lambda x, y: pcbnew.VECTOR2I(mm(x), mm(y))
    pad = next(iter(board.FindFootprintByReference("J1").Pads()))
    pad.SetPosition(point(10, 10.125))
    for zone in list(board.Zones()):
        board.Delete(zone)
    zone = pcbnew.ZONE(board)
    zone.SetNet(board.FindNet("GND"))
    zone.SetLayer(pcbnew.B_Cu)
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    zone.SetLocalClearance(mm(0.30))
    zone.SetMinThickness(mm(0.25))
    zone.Outline().NewOutline()
    for x, y in ((2, 3), (18, 3), (18, 17), (2, 17)):
        zone.Outline().Append(mm(x), mm(y))
    board.Add(zone)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert not zone.GetFilledPolysList(pcbnew.B_Cu).Contains(point(10, 10.125))
    assert not audit_local_copper(board, [spec])

    result = reserve_local_ground_planes(board, [spec], "hole detour", router_clearance_mm=0.17)
    masks = [z for z in board.Zones() if z.GetZoneName() == "hole detour"]
    assert result["zones"] and masks
    assert all(not mask.Outline().Contains(point(10, 10.125)) for mask in masks)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert not audit_local_copper(board, [spec])


def test_off_grid_native_neck_gets_a_protected_ground_witness(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_local_ground_planes
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, _ = _neck_ground_board(tmp_path, 10.151)
    result = reserve_local_ground_planes(board, [spec], "neck protection",
                                         router_clearance_mm=0.17)
    assert result["zones"] > 0
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert not audit_local_copper(board, [spec])
    path = tmp_path / "offgrid.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    _neck_router_probe(path, "offgrid")


def test_native_ground_polygon_decomposes_touching_contours_exactly():
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import _native_polygon
    from shapely.geometry import Point

    shape = pcbnew.SHAPE_POLY_SET()
    shape.NewOutline()
    # Two regions touch at one vertex, like a native fractured fill.  The
    # second region must not be discarded just because GEOS rejects the ring.
    for x, y in ((0, 0), (1000000, 0), (1000000, 1000000), (0, 1000000),
                 (0, 0), (-100000, -100000), (0, -100000)):
        shape.Append(x, y)
    polygon = _native_polygon(shape, 0)
    assert polygon.is_valid
    assert polygon.geom_type == "MultiPolygon"
    assert len(polygon.geoms) == 2
    assert polygon.area == abs(shape.Area())
    assert polygon.covers(Point(-50000, -75000))


def test_native_ground_polygon_rejects_crossed_contour():
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import _native_polygon

    shape = pcbnew.SHAPE_POLY_SET()
    shape.NewOutline()
    for x, y in ((0, 0), (1000000, 1000000), (0, 1000000), (1000000, 0)):
        shape.Append(x, y)
    with pytest.raises(ValueError, match="native polygon conversion is invalid"):
        _native_polygon(shape, 0)


def test_global_ground_witness_covers_distinct_same_number_pads(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_connected_ground

    board, _, _ = _neck_ground_board(tmp_path, 10.125)
    owner = board.FindFootprintByReference("U1")
    other = board.FindFootprintByReference("C1")
    pad = pcbnew.PAD(owner)
    pad.SetNumber("1")
    pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(1), pcbnew.FromMM(1)))
    pad.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(15), pcbnew.FromMM(10.125)))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    pad.SetLayerSet(pcbnew.PAD.SMDMask())
    owner.Add(pad)
    pad.SetNet(board.FindNet("GND"))
    board.Delete(other)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())

    result = reserve_connected_ground(
        board, "GND", "global ground witness", width_mm=0.17,
        router_clearance_mm=0.17,
    )
    assert result["pads"] == 2
    path = tmp_path / "global-ground.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    _neck_router_probe(path, "clearance")


def test_global_ground_witness_rejects_physically_split_fill(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_connected_ground
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, signal = _neck_ground_board(tmp_path, 10.125)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(10), pcbnew.FromMM(8)))
    track.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(10), pcbnew.FromMM(12)))
    track.SetWidth(pcbnew.FromMM(0.2))
    track.SetLayer(pcbnew.B_Cu)
    track.SetNet(signal)
    board.Add(track)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert audit_local_copper(board, [spec])
    with pytest.raises(ValueError, match="no complete native-filled witness forest"):
        reserve_connected_ground(board, "GND", "global ground witness",
                                 width_mm=0.17, router_clearance_mm=0.17)


@pytest.mark.parametrize("plane_proof", [True, False])
def test_ground_connectivity_requires_actual_plane_proof_or_routing(tmp_path, plane_proof):
    import hashlib
    import subprocess

    pcbnew = pytest.importorskip("pcbnew")
    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not (root / "py_router/kicad_parser.py").is_file() or not python.is_file():
        pytest.skip("pinned KiCadRoutingTools runtime unavailable")
    source, output = tmp_path / "source.kicad_pcb", tmp_path / "output.kicad_pcb"
    board = pcbnew.NewBoard(str(source))
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(ground)
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    board.Add(fp)
    mm = pcbnew.FromMM
    for number, x in (("1", 5), ("2", 8)):
        pad = pcbnew.PAD(fp)
        pad.SetNumber(number)
        pad.SetSize(pcbnew.VECTOR2I(mm(1), mm(1)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(10)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        fp.Add(pad)
        pad.SetNet(ground)
    stub = pcbnew.PCB_TRACK(board)
    stub.SetStart(pcbnew.VECTOR2I(mm(5), mm(10)))
    stub.SetEnd(pcbnew.VECTOR2I(mm(5.5), mm(10)))
    stub.SetWidth(mm(0.25))
    stub.SetLayer(pcbnew.F_Cu)
    stub.SetNet(ground)
    board.Add(stub)
    corners = ((0, 0), (20, 0), (20, 20), (0, 20), (0, 0))
    for a, b in zip(corners, corners[1:]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(pcbnew.VECTOR2I(*(mm(v) for v in a)))
        edge.SetEnd(pcbnew.VECTOR2I(*(mm(v) for v in b)))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    board.Save(str(source))
    _write_project(source)
    spec = {"owner_ref": "J1", "bypass_ref": "J1",
            "ground_return": {"net": "GND", "owner_ref": "J1", "owner_pin": "1",
                              "member_pin": "2", "max_via_distance_mm": 3}}
    result = rb.route_with_kicad_routing_tools(str(source), str(output), {
        "kicad_routing_tools_path": str(root), "kicad_routing_tools_python": str(python),
        "kicad_routing_tools_timeout_s": 60, "gnd_zone_net": "GND",
        "routing_local_ground_reservations": plane_proof, "local_bypasses": [spec],
    })
    finished = pcbnew.LoadBoard(str(output))
    assert any(
        item.GetNetname() == "GND" and item.GetLayer() == pcbnew.F_Cu
        and item.GetWidth() == mm(0.25)
        and {(item.GetStart().x, item.GetStart().y), (item.GetEnd().x, item.GetEnd().y)}
            == {(mm(5), mm(10)), (mm(5.5), mm(10))}
        for item in finished.GetTracks() if isinstance(item, pcbnew.PCB_TRACK)
    )
    if plane_proof:
        plan = result["local_ground_reservation_plan"]
        assert hashlib.sha256(Path(plan["board_path"]).read_bytes()).hexdigest() == plan["sha256"]
    report = tmp_path / "drc.json"
    subprocess.run(["kicad-cli", "pcb", "drc", "--format", "json",
                    "--output", str(report), str(output)], check=True, capture_output=True)
    drc = json.loads(report.read_text())
    assert not [row for row in drc["violations"] if row["severity"] == "error"]
    assert drc["unconnected_items"] == []


def test_global_ground_forest_retains_zone_minimum_after_legal_signal_routes(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_connected_ground
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, signal = _neck_ground_board(tmp_path, 10.125)
    router_clearance, signal_width = 0.17, 0.2
    reserve_connected_ground(
        board, "GND", "minimum-thickness-forest", width_mm=0.17,
        router_clearance_mm=router_clearance,
    )
    masks = [zone for zone in board.Zones()
             if zone.GetZoneName() == "minimum-thickness-forest"]
    neck = pcbnew.SHAPE_POLY_SET()
    neck.NewOutline()
    for x, y in ((8.25, 0), (11.75, 0), (11.75, 20), (8.25, 20)):
        neck.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    reserved = pcbnew.SHAPE_POLY_SET()
    for zone in masks:
        clipped = pcbnew.SHAPE_POLY_SET(zone.Outline())
        clipped.BooleanIntersection(neck)
        reserved.BooleanAdd(clipped)
    assert reserved.OutlineCount()
    bounds = reserved.BBox()
    distance = pcbnew.FromMM(router_clearance + signal_width / 2 + 0.005)
    for y in (bounds.GetTop() - distance, bounds.GetBottom() + distance):
        start = pcbnew.VECTOR2I(pcbnew.FromMM(8.25), y)
        end = pcbnew.VECTOR2I(pcbnew.FromMM(11.75), y)
        segment = pcbnew.SEG(start, end)
        assert all(not zone.Outline().Collide(
            segment, pcbnew.FromMM(router_clearance + signal_width / 2),
        ) for zone in masks)
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(start)
        track.SetEnd(end)
        track.SetLayer(pcbnew.B_Cu)
        track.SetWidth(pcbnew.FromMM(signal_width))
        track.SetNet(signal)
        board.Add(track)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert not audit_local_copper(board, [spec])


def test_zone_clearance_compensates_corridor_masks(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_local_ground_planes
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, _ = _neck_ground_board(tmp_path, 10.125)
    reserve_local_ground_planes(
        board, [spec], "clearance protection", router_clearance_mm=0.17,
    )
    assert not audit_local_copper(board, [spec])
    path = tmp_path / "clearance.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    _neck_router_probe(path, "clearance")

    # Without the masks, these legal-at-0.17 tracks narrow the native 0.30
    # neck below its 0.25 mm minimum and disconnect the actual return on refill.
    exposed, exposed_spec, signal = _neck_ground_board(tmp_path, 10.125)
    for y in (9.725, 10.525):
        track = pcbnew.PCB_TRACK(exposed)
        track.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(8), pcbnew.FromMM(y)))
        track.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(12), pcbnew.FromMM(y)))
        track.SetLayer(pcbnew.B_Cu)
        track.SetWidth(pcbnew.FromMM(0.2))
        track.SetNet(signal)
        exposed.Add(track)
    exposed.BuildConnectivity()
    pcbnew.ZONE_FILLER(exposed).Fill(exposed.Zones())
    assert any("ground return" in error for error in audit_local_copper(exposed, [exposed_spec]))


def test_front_ground_witness_when_back_fill_is_split(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.ground_reservations import reserve_local_ground_planes
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    board, spec, _ = _neck_ground_board(tmp_path, 10.125)
    mm = pcbnew.FromMM
    ground = board.FindNet("GND")
    front = pcbnew.ZONE(board)
    front.SetNet(ground)
    front.SetLayer(pcbnew.F_Cu)
    front.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    front.SetLocalClearance(mm(0.30))
    front.SetMinThickness(mm(0.25))
    front.Outline().NewOutline()
    for x, y in ((2, 3), (18, 3), (18, 17), (2, 17)):
        front.Outline().Append(mm(x), mm(y))
    board.Add(front)

    split = pcbnew.ZONE(board)
    split.SetLayer(pcbnew.B_Cu)
    split.SetIsRuleArea(True)
    split.SetZoneName("authored B copper split")
    split.SetDoNotAllowTracks(False)
    split.SetDoNotAllowVias(False)
    split.SetDoNotAllowPads(False)
    split.SetDoNotAllowCopperPour(True)
    split.Outline().NewOutline()
    for x, y in ((9.8, 0), (10.2, 0), (10.2, 20), (9.8, 20)):
        split.Outline().Append(mm(x), mm(y))
    board.Add(split)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert not audit_local_copper(board, [spec])
    back = next(z for z in board.Zones() if not z.GetIsRuleArea() and z.IsOnLayer(pcbnew.B_Cu))
    assert back.GetFilledPolysList(pcbnew.B_Cu).OutlineCount() >= 2

    name = "front ground witness"
    reserve_local_ground_planes(board, [spec], name, router_clearance_mm=0.17)
    masks = [z for z in board.Zones() if z.GetZoneName() == name]
    assert all(z.IsOnLayer(pcbnew.F_Cu) and not z.IsOnLayer(pcbnew.B_Cu) for z in masks)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert not audit_local_copper(board, [spec])

    root = Path("/home/kicraft/KiCadRoutingTools")
    python = Path("/home/kicraft/krt-venv/bin/python")
    if not (root / "py_router/kicad_parser.py").is_file() or not python.is_file():
        pytest.skip("pinned router is not installed")
    path = tmp_path / "front-witness.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    probe = """
import sys
from types import SimpleNamespace
sys.path[:0] = [sys.argv[1] + '/py_router', sys.argv[1] + '/py_tools']
from kicad_parser import extract_keepouts
from obstacle_map import add_rule_area_keepout_obstacles
from routing_config import GridRouteConfig
class Capture:
    def __init__(self):
        self.tracks, self.vias = set(), set()
    def add_blocked_cells_batch(self, rows):
        self.tracks.update(tuple(int(v) for v in row) for row in rows)
    def add_blocked_vias_batch(self, rows):
        self.vias.update(tuple(int(v) for v in row) for row in rows)
c = Capture()
areas = extract_keepouts(open(sys.argv[2]).read())
pcb = SimpleNamespace(board_info=SimpleNamespace(keepouts=areas, board_bounds=None))
cfg = GridRouteConfig(grid_step=0.025, layers=['F.Cu', 'B.Cu'], track_width=0.2,
                      via_size=0.6, via_drill=0.3, clearance=0.17)
add_rule_area_keepout_obstacles(c, pcb, cfg)
assert any((400, y, 0) in c.tracks for y in range(120, 681))
assert all((400, y, 1) not in c.tracks for y in range(120, 681))
assert (160, 160, 0) not in c.tracks
"""
    subprocess.run([str(python), "-c", probe, str(root), str(path)], check=True)
