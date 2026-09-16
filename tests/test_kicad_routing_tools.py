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


def test_krt_preflight_uses_environment_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("KICRAFT_KICAD_ROUTING_TOOLS_PATH", "/tmp/KiCadRoutingTools")
    monkeypatch.setenv("KICRAFT_KICAD_ROUTING_TOOLS_PYTHON", "/tmp/krt-venv/bin/python")

    result = rb.preflight_kicad_routing_tools({
                "kicad_routing_tools_path": "",
        "kicad_routing_tools_python": "",
    })


    assert result["root"] == "/tmp/KiCadRoutingTools"
    assert result["python"] == "/tmp/krt-venv/bin/python"

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
