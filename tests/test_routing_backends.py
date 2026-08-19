import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.autoplacer import routing_backends as rb


def test_krt_is_default():
    assert rb.routing_backend({}) == "kicad-routing-tools"


def test_backend_aliases_and_invalid_name():
    assert rb.routing_backend({"routing_backend": "krt"}) == "kicad-routing-tools"
    with pytest.raises(ValueError, match="Unknown routing_backend"):
        rb.routing_backend({"routing_backend": "unknown"})


def test_krt_command_preserves_rules_and_existing_copper(tmp_path):
    root = tmp_path / "krt"
    (root / "py_router").mkdir(parents=True)
    (root / "py_router" / "route.py").write_text("# stub\n")
    cfg = {
        "routing_backend": "kicad-routing-tools",
        "kicad_routing_tools_path": str(root),
        "signal_width_mm": 0.25,
        "via_size_mm": 0.7,
        "via_drill_mm": 0.35,
    }
    cmd = rb._krt_command("input.kicad_pcb", "output.kicad_pcb", cfg)
    assert "--keep-input-copper" in cmd
    assert "--no-fix-drc-settings" in cmd
    assert "--force-reroute" not in cmd
    assert "--rip-existing-nets" not in cmd
    assert cmd[cmd.index("--nets") + 1] == "*"
    for option in ("--track-width", "--via-size", "--via-drill", "--clearance"):
        assert option not in cmd


def test_dispatches_to_krt_without_freerouting(monkeypatch, tmp_path):
    called = {}

    def fake(source, output, config):
        called.update(source=source, output=output, config=config)
        return {"backend": "kicad-routing-tools"}

    monkeypatch.setattr(rb, "route_with_kicad_routing_tools", fake)
    cfg = {"routing_backend": "kicad-routing-tools"}
    result = rb.route_board("in.kicad_pcb", "out.kicad_pcb", cfg)
    assert result["backend"] == "kicad-routing-tools"
    assert called["source"] == "in.kicad_pcb"


def test_preflight_requires_configured_checkout():
    with pytest.raises(rb.RoutingBackendUnavailableError, match="path is unset"):
        rb.preflight_routing_backend({"routing_backend": "kicad-routing-tools"})


def test_krt_preflight_uses_environment_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("KICRAFT_KICAD_ROUTING_TOOLS_PATH", "/tmp/KiCadRoutingTools")
    monkeypatch.setenv("KICRAFT_KICAD_ROUTING_TOOLS_PYTHON", "/tmp/krt-venv/bin/python")

    result = rb.preflight_routing_backend({
        "routing_backend": "kicad-routing-tools",
        "kicad_routing_tools_path": "",
        "kicad_routing_tools_python": "",
    })


    assert result["root"] == "/tmp/KiCadRoutingTools"
    assert result["python"] == "/tmp/krt-venv/bin/python"

def test_parent_krt_path_bypasses_freerouting_workarounds(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import kicraft.autoplacer.freerouting_runner as fr
    import kicraft.autoplacer.routing_backends as backends
    import kicraft.autoplacer.brain.gnd_pour as gnd_pour
    import kicraft.cli._compose_route as compose_route

    stamped = tmp_path / "parent_stamped.kicad_pcb"
    stamped.write_text("(kicad_pcb stamped)\n")
    events = []

    def fake_route(source, output, config, *, jar_path=None):
        events.append((source, dict(config)))
        Path(output).write_text("(kicad_pcb routed)\n")
        return {"backend": "kicad-routing-tools", "returncode": 0}

    monkeypatch.setattr(backends, "route_board", fake_route)
    monkeypatch.setattr(fr, "strip_net_copper", lambda *a, **k: pytest.fail("FR GND strip ran"))
    monkeypatch.setattr(
        fr,
        "route_with_freerouting",
        lambda *a, **k: pytest.fail("FreeRouting route ran"),
    )
    monkeypatch.setattr(
        gnd_pour,
        "add_gnd_pour_and_thermal_vias",
        lambda *a, **k: pytest.fail("FreeRouting pre-route GND pour ran"),
    )
    monkeypatch.setattr(fr, "import_routed_copper", lambda p: {"traces": [], "vias": []})
    monkeypatch.setattr(fr, "validate_routed_board", lambda *a, **k: {"accepted": True, "drc": {}})
    state = SimpleNamespace(
        composition=SimpleNamespace(inferred_interconnect_nets={"A": 1}),
        component_count=2,
    )
    cfg = {
        "routing_backend": "kicad-routing-tools",
        "gnd_zone_net": "",
        "power_plane_enabled": False,
        "signal_unconnected_repair_enabled": False,
        "illegal_geometry_repair_enabled": False,
    }
    result = compose_route._route_parent_board(stamped, state, tmp_path, cfg)
    assert len(events) == 1
    assert result["backend"] == "kicad-routing-tools"
    assert result["routing_stats"]["returncode"] == 0
    assert result["routing_stats"]["backend"] == "kicad-routing-tools"


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
        "routing_backend": "kicad-routing-tools",
        "kicad_routing_tools_path": str(root),
        "kicad_routing_tools_python": "/configured/python",
    }

    first = rb.preflight_routing_backend(cfg)
    second = rb.preflight_routing_backend(cfg)

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
        "routing_backend": "kicad-routing-tools",
        "kicad_routing_tools_path": str(root),
    }

    for _ in range(2):
        with pytest.raises(rb.RoutingBackendUnavailableError, match="startup checks"):
            rb.preflight_routing_backend(cfg)
    assert startup_calls == 2


def test_krt_route_process_boundary_preserves_rules_and_summaries(
    monkeypatch, tmp_path
):
    import kicraft.autoplacer.freerouting_runner as fr

    root = tmp_path / "krt"
    root.mkdir()
    rules_board = tmp_path / "rules" / "authoritative.kicad_pcb"
    rules_board.parent.mkdir()
    rules_board.with_suffix(".kicad_pro").write_text("project rules\n")
    rules_board.with_suffix(".kicad_dru").write_text("custom rules\n")
    route_dir = tmp_path / "route"
    route_dir.mkdir()
    input_board = route_dir / "input.kicad_pcb"
    output_board = route_dir / "output.kicad_pcb"
    input_board.write_text("input\n")
    output_board.write_text("stale output\n")
    events = []
    observed = {}

    def fake_preflight(_config):
        events.append("preflight")
        return _runtime(root)

    def fake_import(path):
        assert Path(path) in (input_board.resolve(), output_board.resolve())
        return _copper()

    class FakeProcess:
        returncode = 0
        pid = 123

        def __init__(self, command, **kwargs):
            events.append("launch")
            assert events == ["preflight", "launch"]
            assert not output_board.exists()
            assert input_board.with_suffix(".kicad_pro").read_text() == "project rules\n"
            assert input_board.with_suffix(".kicad_dru").read_text() == "custom rules\n"
            observed["command"] = command
            observed["env"] = kwargs["env"]

        def communicate(self, timeout=None):
            output_board.write_text("new routed output\n")
            return (
                'JSON_SUMMARY: {"successful": 8, "failed": 2, '
                '"total_vias": 3, "total_time": 1.5}\n'
                'JSON_SUMMARY: {"successful": 1, "failed": 0}\n',
                "diagnostic stderr",
            )

    monkeypatch.setattr(rb, "preflight_routing_backend", fake_preflight)
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(fr, "import_routed_copper", fake_import)
    stats = rb.route_with_kicad_routing_tools(
        str(input_board),
        str(output_board),
        {
            "routing_backend": "kicad-routing-tools",
            "kicad_routing_tools_path": str(root),
            "pcb_path": str(rules_board),
        },
    )

    assert observed["env"]["KICAD_RIP_PREEXISTING"] == "0"
    assert observed["env"]["KICAD_PLANE_FINALIZE"] == "0"
    assert "KICAD_FINALIZE_RIP" not in observed["env"]
    assert "--keep-input-copper" in observed["command"]
    assert not input_board.with_suffix(".kicad_pro").exists()
    assert not input_board.with_suffix(".kicad_dru").exists()
    assert output_board.read_text() == "new routed output\n"
    assert output_board.with_suffix(".kicad_pro").read_text() == "project rules\n"
    assert output_board.with_suffix(".kicad_dru").read_text() == "custom rules\n"
    assert len(stats["json_summaries"]) == 2
    assert stats["successful_nets"] == 8
    assert stats["failed_nets"] == 2
    assert stats["preserved_existing_copper"] is True
    assert stats["input_copper_preservation"]["traces"]["missing_count"] == 0
    assert stats["input_copper_preservation"]["vias"]["missing_count"] == 0
    assert stats["_raw_stderr"] == "diagnostic stderr"


def test_krt_route_rejects_same_input_and_output(tmp_path):
    board = tmp_path / "same.kicad_pcb"
    board.write_text("board\n")
    with pytest.raises(ValueError, match="must be distinct"):
        rb.route_with_kicad_routing_tools(
            str(board),
            str(board),
            {"routing_backend": "kicad-routing-tools"},
        )


def test_krt_route_requires_project_rules_before_launch(monkeypatch, tmp_path):
    import kicraft.autoplacer.freerouting_runner as fr

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

    monkeypatch.setattr(rb, "preflight_routing_backend", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", fail_launch)
    monkeypatch.setattr(fr, "import_routed_copper", lambda _: _copper())

    with pytest.raises(
        rb.RoutingBackendUnavailableError,
        match=r"^KiCadRoutingTools requires a sibling \.kicad_pro",
    ):
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                "routing_backend": "kicad-routing-tools",
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
    import kicraft.autoplacer.freerouting_runner as fr

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    input_board.with_suffix(".kicad_pro").write_text("rules\n")

    class FakeProcess:
        pid = 456

        def __init__(self, command, **kwargs):
            self.returncode = returncode

        def communicate(self, timeout=None):
            if create_output:
                output_board.write_text("diagnostic output\n")
            return ("stdout", "stderr")

    monkeypatch.setattr(rb, "preflight_routing_backend", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(fr, "import_routed_copper", lambda _: _copper())

    with pytest.raises(RuntimeError, match=match):
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                "routing_backend": "kicad-routing-tools",
                "kicad_routing_tools_path": str(root),
            },
        )


def test_krt_route_timeout_behavior_is_unchanged(monkeypatch, tmp_path):
    import kicraft.autoplacer.freerouting_runner as fr

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    input_board.with_suffix(".kicad_pro").write_text("rules\n")
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

    monkeypatch.setattr(rb, "preflight_routing_backend", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(rb.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(fr, "import_routed_copper", lambda _: _copper())

    with pytest.raises(RuntimeError, match="timed out after 1s"):
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                "routing_backend": "kicad-routing-tools",
                "kicad_routing_tools_path": str(root),
                "kicad_routing_tools_timeout_s": 1,
            },
        )
    assert signals == [(789, signal.SIGTERM)]


def test_krt_route_rejects_missing_input_copper(monkeypatch, tmp_path):
    import kicraft.autoplacer.freerouting_runner as fr

    root = tmp_path / "krt"
    root.mkdir()
    input_board = tmp_path / "input.kicad_pcb"
    output_board = tmp_path / "output.kicad_pcb"
    input_board.write_text("input\n")
    input_board.with_suffix(".kicad_pro").write_text("authoritative rules\n")

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

    monkeypatch.setattr(rb, "preflight_routing_backend", lambda _: _runtime(root))
    monkeypatch.setattr(rb.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(fr, "import_routed_copper", fake_import)

    with pytest.raises(rb.RoutingCopperPreservationError) as caught:
        rb.route_with_kicad_routing_tools(
            str(input_board),
            str(output_board),
            {
                "routing_backend": "kicad-routing-tools",
                "kicad_routing_tools_path": str(root),
            },
        )
    assert caught.value.stats["preserved_existing_copper"] is False
    preservation = caught.value.stats["input_copper_preservation"]
    assert preservation["traces"]["missing_count"] == 1
    assert preservation["vias"]["missing_count"] == 1
    assert output_board.is_file()
    assert output_board.with_suffix(".kicad_pro").read_text() == "authoritative rules\n"
