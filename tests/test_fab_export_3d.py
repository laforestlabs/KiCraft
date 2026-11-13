"""Tests for the fab 3D outputs (STEP export + assembled-board render).

``_run`` is replaced by a recorder that fabricates kicad-cli outputs, so
these stay hermetic: no kicad-cli, no GL. The contract under test: 3D
outputs land in the zip when they succeed, never fail the export when they
do not, and a stale render can never leak into a new zip.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import kicraft.design.synthesis.fab_export as fe


class _Runner:
    """Stands in for fab_export._run; writes whatever '-o' names."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = lambda kind, cmd: False

    def __call__(self, cmd, timeout=None):
        offset = 2 if cmd[0] == "xvfb-run" else 0
        kind = cmd[offset + 2]
        if kind == "export":
            kind = cmd[offset + 3]
        self.calls.append((kind, cmd))
        if self.fail(kind, cmd):
            raise RuntimeError(f"fake {kind} failure")
        out = Path(cmd[cmd.index("-o") + 1])
        if kind in ("gerbers", "drill"):
            out.mkdir(parents=True, exist_ok=True)
            (out / f"dummy.{kind}").write_text("x")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(kind)

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


@pytest.fixture
def runner(monkeypatch) -> _Runner:
    r = _Runner()
    monkeypatch.setattr(fe, "_run", r)
    return r


@pytest.fixture
def pcb(tmp_path: Path) -> Path:
    p = tmp_path / "WIDGET.kicad_pcb"
    p.write_text("(kicad_pcb)")
    return p


def test_3d_outputs_exported_and_zipped(runner, pcb, tmp_path):
    out = fe.export_fab(str(pcb), str(tmp_path), "WIDGET")
    assert out["step"] == str(tmp_path / "fab" / "WIDGET.step")
    assert out["board_3d_png"] == str(tmp_path / "fab" / "board_3d.png")
    assert "WIDGET.step" in out["files"]
    assert "board_3d.png" in out["files"]
    with zipfile.ZipFile(out["zip"]) as zf:
        names = zf.namelist()
    assert "WIDGET.step" in names and "board_3d.png" in names
    # The render ran once, no xvfb fallback needed.
    assert runner.kinds().count("render") == 1


def test_3d_failures_never_fail_the_export(runner, pcb, tmp_path, capsys):
    runner.fail = lambda kind, cmd: kind in ("step", "render")
    out = fe.export_fab(str(pcb), str(tmp_path), "WIDGET")
    assert out["step"] is None
    assert out["board_3d_png"] is None
    assert Path(out["zip"]).is_file()  # gerber package still produced
    err = capsys.readouterr().err
    assert "STEP export failed" in err
    assert "3D render failed" in err


def test_stale_render_never_resurrects_into_zip(runner, pcb, tmp_path):
    fab = tmp_path / "fab"
    fab.mkdir()
    (fab / "board_3d.png").write_text("stale image")
    runner.fail = lambda kind, cmd: kind == "render"
    out = fe.export_fab(str(pcb), str(tmp_path), "WIDGET")
    assert out["board_3d_png"] is None
    assert not (fab / "board_3d.png").exists()
    with zipfile.ZipFile(out["zip"]) as zf:
        assert "board_3d.png" not in zf.namelist()


def test_render_falls_back_to_xvfb(runner, pcb, tmp_path, monkeypatch):
    monkeypatch.setattr(fe.shutil, "which", lambda name: "/usr/bin/xvfb-run")
    runner.fail = lambda kind, cmd: kind == "render" and cmd[0] != "xvfb-run"
    out = fe.export_fab(str(pcb), str(tmp_path), "WIDGET")
    assert out["board_3d_png"] is not None
    render_cmds = [cmd for kind, cmd in runner.calls if kind == "render"]
    assert len(render_cmds) == 2
    assert render_cmds[1][0] == "xvfb-run"


def test_include_3d_false_skips_both(runner, pcb, tmp_path):
    out = fe.export_fab(str(pcb), str(tmp_path), "WIDGET", include_3d=False)
    assert out["step"] is None and out["board_3d_png"] is None
    assert "step" not in runner.kinds()
    assert "render" not in runner.kinds()


@pytest.mark.parametrize("failed_kind,filename", [
    ("step", "WIDGET.step"),
    ("render", "board_3d.png"),
])
def test_partial_failed_3d_output_is_not_certified(
    runner, pcb, tmp_path, monkeypatch, failed_kind, filename
):
    def write_then_fail(cmd, timeout=None):
        runner(cmd, timeout=timeout)
        kind = cmd[3] if cmd[2] == "export" else cmd[2]
        if kind == failed_kind:
            raise RuntimeError("converter left a partial file")

    monkeypatch.setattr(fe, "_run", write_then_fail)
    monkeypatch.setattr(fe.shutil, "which", lambda name: None)
    result = fe.export_fab(str(pcb), str(tmp_path), "WIDGET")
    assert result["step" if failed_kind == "step" else "board_3d_png"] is None
    assert filename not in result["files"]
    with zipfile.ZipFile(result["zip"]) as archive:
        assert filename not in archive.namelist()


@pytest.mark.parametrize("copper_mm,success", [(0.07, True), (0.035, False)])
def test_two_ounce_fab_job_is_checked_and_instructions_are_packaged(
    runner, pcb, tmp_path, monkeypatch, copper_mm, success
):
    import json
    from types import SimpleNamespace
    from kicraft.design.synthesis.kicad_pro import write_kicad_pro
    from kicraft.autoplacer import routing_board

    write_kicad_pro(
        tmp_path, "WIDGET", SimpleNamespace(power_nets=[]),
        fab_config={"thermal_requirements": [{"outer_copper_oz": 2}]},
    )

    monkeypatch.setattr(fe, "has_two_ounce_outer_stackup", lambda path: True)
    monkeypatch.setattr(routing_board, "require_copper_min_width", lambda *args: None)
    monkeypatch.setattr(routing_board, "require_two_ounce_holes", lambda *args: None)
    monkeypatch.setattr(routing_board, "run_kicad_cli_drc", lambda *args: {
        "ran": True, "returncode": 0, "track_width": 0, "clearance": 0, "report_text": "DRC report",
    })

    def run_with_job(cmd, timeout=None):
        runner(cmd, timeout=timeout)
        if cmd[2:4] == ["export", "gerbers"]:
            out = Path(cmd[cmd.index("-o") + 1])
            (out / "WIDGET-job.gbrjob").write_text(json.dumps({
                "MaterialStackup": [
                    {"Type": "Copper", "Name": name, "Thickness": copper_mm}
                    for name in ("F.Cu", "B.Cu")
                ]
            }))

    monkeypatch.setattr(fe, "_run", run_with_job)
    if not success:
        with pytest.raises(RuntimeError, match="0.070 mm"):
            fe.export_fab(str(pcb), str(tmp_path), "WIDGET", include_3d=False)
        return
    result = fe.export_fab(str(pcb), str(tmp_path), "WIDGET", include_3d=False)
    with zipfile.ZipFile(result["zip"]) as archive:
        assert b"0.070 mm finished" in archive.read("fabrication_instructions.txt")
        assert b"0.170 mm" in archive.read("fabrication_instructions.txt")
        assert "WIDGET-job.gbrjob" in archive.namelist()


def test_two_ounce_fab_rejects_one_ounce_project_rules(runner, pcb, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kicraft.design.synthesis.kicad_pro import write_kicad_pro

    write_kicad_pro(tmp_path, "WIDGET", SimpleNamespace(power_nets=[]))
    monkeypatch.setattr(fe, "has_two_ounce_outer_stackup", lambda path: True)
    with pytest.raises(ValueError, match="below"):
        fe.export_fab(str(pcb), str(tmp_path), "WIDGET", include_3d=False)
    assert not runner.calls


def test_two_ounce_export_rejects_actual_narrow_native_track(tmp_path, runner):
    from types import SimpleNamespace
    from kicraft.design.synthesis.kicad_pro import write_kicad_pro
    from kicraft.pcb_layers import set_two_ounce_outer_stackup

    pcbnew = pytest.importorskip("pcbnew")
    path = tmp_path / "WIDGET.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    net = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(net)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(5), pcbnew.FromMM(5)))
    track.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(10), pcbnew.FromMM(5)))
    track.SetWidth(pcbnew.FromMM(0.127))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(net)
    board.Add(track)
    board.Save(str(path))
    write_kicad_pro(
        tmp_path, "WIDGET", SimpleNamespace(power_nets=[]),
        fab_config={"thermal_requirements": [{"outer_copper_oz": 2}]},
    )
    set_two_ounce_outer_stackup(path)
    with pytest.raises(ValueError, match="routed copper width 0.127"):
        fe.export_fab(str(path), str(tmp_path), "WIDGET", include_3d=False)
    assert not runner.calls


@pytest.mark.parametrize("land,drill,shape,passes", [
    ((1.524, 1.524), (1.09982, 1.09982), "circle", False),
    ((1.2, 2.0), (0.8, 1.6), "oval", False),
    ((1.62, 1.62), (1.10, 1.10), "circle", True),
    ((1.32, 2.12), (0.8, 1.6), "oval", True),
])
def test_two_ounce_native_component_annulus_and_fanout_via(tmp_path, land, drill, shape, passes):
    from kicraft.autoplacer.routing_board import require_two_ounce_holes

    pcbnew = pytest.importorskip("pcbnew")
    path = tmp_path / "board.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    board.Add(fp)
    pad = pcbnew.PAD(fp)
    pad.SetNumber("1")
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE if shape == "circle" else pcbnew.PAD_SHAPE_OVAL)
    pad.SetSize(pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in land)))
    pad.SetDrillSize(pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in drill)))
    if shape == "oval":
        pad.SetDrillShape(pcbnew.PAD_DRILL_SHAPE_OBLONG)
    pad.SetLayerSet(pcbnew.LSET.AllCuMask())
    fp.Add(pad)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(5), 0))
    via.SetWidth(pcbnew.FromMM(0.36))
    via.SetDrill(pcbnew.FromMM(0.15))
    board.Add(via)
    board.Save(str(path))
    if passes:
        require_two_ounce_holes(path)
    else:
        with pytest.raises(ValueError, match="component PTH annulus"):
            require_two_ounce_holes(path)


def test_two_ounce_native_hole_gap_rejects_close_vias(tmp_path):
    from kicraft.autoplacer.routing_board import require_two_ounce_holes

    pcbnew = pytest.importorskip("pcbnew")
    path = tmp_path / "board.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for x in (0, 0.34):
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), 0))
        via.SetWidth(pcbnew.FromMM(0.36))
        via.SetDrill(pcbnew.FromMM(0.15))
        board.Add(via)
    board.Save(str(path))
    with pytest.raises(ValueError, match="hole-to-hole gap"):
        require_two_ounce_holes(path)


def test_two_ounce_export_rejects_actual_undersized_component_land(tmp_path, runner):
    from types import SimpleNamespace
    from kicraft.design.synthesis.kicad_pro import write_kicad_pro
    from kicraft.pcb_layers import set_two_ounce_outer_stackup

    pcbnew = pytest.importorskip("pcbnew")
    path = tmp_path / "WIDGET.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J2")
    board.Add(fp)
    pad = pcbnew.PAD(fp)
    pad.SetNumber("1")
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(1.524), pcbnew.FromMM(1.524)))
    pad.SetDrillSize(pcbnew.VECTOR2I(pcbnew.FromMM(1.09982), pcbnew.FromMM(1.09982)))
    pad.SetLayerSet(pcbnew.LSET.AllCuMask())
    fp.Add(pad)
    board.Save(str(path))
    write_kicad_pro(tmp_path, "WIDGET", SimpleNamespace(power_nets=[]),
                    fab_config={"thermal_requirements": [{"outer_copper_oz": 2}]})
    set_two_ounce_outer_stackup(path)
    with pytest.raises(ValueError, match="component PTH annulus"):
        fe.export_fab(str(path), str(tmp_path), "WIDGET", include_3d=False)
    assert not runner.calls
