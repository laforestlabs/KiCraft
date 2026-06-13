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
