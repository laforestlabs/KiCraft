"""kicraft manual-route: argument gating + the failure-safe promote tail.

The promote tail (`_promote_verify_fab`, shared with `build`) must never
clobber the last good ``<stem>.kicad_pcb``: a failed verify gate restores
the backed-up board. Driven with monkeypatched verify/align/export seams
(no pcbnew, no router)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kicraft.design.cli_app as cli_app


# ---- subcommand registration -------------------------------------------------


def test_manual_route_registered_and_gates_missing_layout(tmp_path, capsys):
    """Without a saved manual layout the command exits 3 with a clear
    message (and without one it must not touch the build slot)."""
    from kicraft.design.models import ConversationState
    st = ConversationState(project_stem="WIDGET")
    state_path = tmp_path / "state.json"
    state_path.write_text(st.model_dump_json(), encoding="utf-8")

    rc = cli_app.main(["manual-route", str(state_path), str(tmp_path / "generated")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "manual layout" in err or "BOM" in err


# ---- promote tail -------------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    pd = tmp_path / "WIDGET"
    pd.mkdir()
    (pd / "WIDGET.kicad_pcb").write_text("GOOD BOARD", encoding="utf-8")
    routed_dir = pd / ".experiments" / "subcircuits" / "x"
    routed_dir.mkdir(parents=True)
    (routed_dir / "parent_routed.kicad_pcb").write_text("NEW ROUTE", encoding="utf-8")
    return pd


def _state_stub():
    from kicraft.design.models import BOM, BomPart, ConversationState
    return ConversationState(
        project_stem="WIDGET",
        bom=BOM(parts=[BomPart(ref="R1", value="10k", sheet="MAIN",
                               symbol="Device:R",
                               footprint="Resistor_SMD:R_0402_1005Metric")]),
    )


def _artifacts_stub(pd: Path):
    from kicraft.design.models import ArtifactPaths
    return ArtifactPaths(
        project_dir=pd, project_stem="WIDGET",
        root_sch=pd / "WIDGET.kicad_sch", leaf_schs=[],
        kicad_pro=pd / "WIDGET.kicad_pro",
        autoplacer_json=pd / "WIDGET_autoplacer.json",
    )


def test_failed_gate_restores_previous_board(project, tmp_path, monkeypatch):
    pcb = project / "WIDGET.kicad_pcb"
    monkeypatch.setattr(cli_app, "_align_project_clearance_to_routing",
                        lambda *a, **k: None)
    monkeypatch.setattr(cli_app, "_verify_routed_board", lambda p: {
        "ok": False, "shorts": 3, "unconnected": 1, "reasons": ["shorts"],
        "tracks": {}})
    state_path = tmp_path / "state.json"
    rc = cli_app._promote_verify_fab(
        _state_stub(), state_path, _artifacts_stub(project), "WIDGET",
        project, pcb)
    assert rc == 7
    assert pcb.read_text(encoding="utf-8") == "GOOD BOARD", (
        "failed candidate must not clobber the previous good board")
    assert not pcb.with_name(pcb.name + ".prev").exists()


def test_failed_gate_with_no_previous_board_leaves_none(project, tmp_path,
                                                        monkeypatch):
    pcb = project / "WIDGET.kicad_pcb"
    pcb.unlink()  # no prior board (first promote)
    monkeypatch.setattr(cli_app, "_align_project_clearance_to_routing",
                        lambda *a, **k: None)
    monkeypatch.setattr(cli_app, "_verify_routed_board", lambda p: {
        "ok": False, "shorts": 1, "unconnected": 0, "reasons": [], "tracks": {}})
    rc = cli_app._promote_verify_fab(
        _state_stub(), tmp_path / "state.json", _artifacts_stub(project),
        "WIDGET", project, pcb)
    assert rc == 7
    assert not pcb.exists()


def test_passing_gate_promotes_and_exports(project, tmp_path, monkeypatch):
    pcb = project / "WIDGET.kicad_pcb"
    seen = {}
    monkeypatch.setattr(cli_app, "_align_project_clearance_to_routing",
                        lambda *a, **k: None)
    monkeypatch.setattr(cli_app, "_verify_routed_board", lambda p: {
        "ok": True, "shorts": 0, "unconnected": 0, "reasons": [],
        "tracks": {"traces": 12, "vias": 2}})

    def fake_export(pcb_path, project_dir, stem, bom_parts=None):
        seen["bom_parts"] = bom_parts
        zip_path = Path(project_dir) / "fab.zip"
        zip_path.write_text("zip", encoding="utf-8")
        return {"zip": str(zip_path), "files": ["a.gbr"], "fab_dir": project_dir,
                "bom_csv": None}

    import kicraft.design.synthesis.fab_export as fab_export
    monkeypatch.setattr(fab_export, "export_fab", fake_export)

    state = _state_stub()
    state_path = tmp_path / "state.json"
    rc = cli_app._promote_verify_fab(
        state, state_path, _artifacts_stub(project), "WIDGET", project, pcb,
        done_label="MANUAL ROUTE COMPLETE")
    assert rc == 0
    assert pcb.read_text(encoding="utf-8") == "NEW ROUTE"
    assert not pcb.with_name(pcb.name + ".prev").exists()
    assert seen["bom_parts"][0]["ref"] == "R1"
    # Artifacts persisted onto the state file.
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["artifacts"]["routed_pcb"].endswith("WIDGET.kicad_pcb")
    assert persisted["artifacts"]["fab_zip"].endswith("fab.zip")


def test_cmd_manual_route_end_to_end_with_stubbed_router(tmp_path, monkeypatch):
    """The full subcommand glue: state loading, stem-dir resolution,
    compose invocation (stubbed: drops a routed parent), then the real
    promote tail with stubbed verify/export. No FreeRouting, no pcbnew."""
    import subprocess as subprocess_mod

    from kicraft.design.models import BOM, BomPart, ConversationState

    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")  # slot gate off

    ws = tmp_path
    state = ConversationState(
        project_stem="WIDGET",
        bom=BOM(parts=[BomPart(ref="R1", value="10k", sheet="MAIN",
                               symbol="Device:R",
                               footprint="Resistor_SMD:R_0402_1005Metric")]),
    )
    state_path = ws / "state.json"
    state_path.write_text(state.model_dump_json(), encoding="utf-8")
    pd = ws / "generated" / "WIDGET"
    manual_dir = pd / ".experiments" / "manual"
    manual_dir.mkdir(parents=True)
    (pd / "WIDGET.kicad_pcb").write_text("SEED BOARD", encoding="utf-8")
    (manual_dir / "manual_layout.json").write_text("{}", encoding="utf-8")

    seen = {}

    def fake_run(cmd, cwd=None, **kw):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        routed_dir = pd / ".experiments" / "subcircuits" / "x"
        routed_dir.mkdir(parents=True, exist_ok=True)
        (routed_dir / "parent_routed.kicad_pcb").write_text(
            "ROUTED", encoding="utf-8")
        return subprocess_mod.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess_mod, "run", fake_run)
    monkeypatch.setattr(cli_app, "_align_project_clearance_to_routing",
                        lambda *a, **k: None)
    monkeypatch.setattr(cli_app, "_verify_routed_board", lambda p: {
        "ok": True, "shorts": 0, "unconnected": 0, "reasons": [],
        "tracks": {"traces": 5, "vias": 0}})

    import kicraft.design.synthesis.fab_export as fab_export

    def fake_export(pcb_path, project_dir, stem, bom_parts=None):
        zp = Path(project_dir) / "fab.zip"
        zp.write_text("zip", encoding="utf-8")
        return {"zip": str(zp), "files": ["a.gbr"], "fab_dir": project_dir,
                "bom_csv": None}

    monkeypatch.setattr(fab_export, "export_fab", fake_export)

    rc = cli_app.main(["manual-route", str(state_path), str(ws / "generated")])
    assert rc == 0
    # Compose was invoked with the manual layout against the project dir.
    cmd = seen["cmd"]
    assert "kicraft.cli.compose_subcircuits" in cmd
    assert "--manual-layout" in cmd and "--route" in cmd
    assert str(manual_dir / "manual_layout.json") in cmd
    assert seen["cwd"] == str(pd)
    # The routed candidate was promoted over the seed board.
    assert (pd / "WIDGET.kicad_pcb").read_text(encoding="utf-8") == "ROUTED"
