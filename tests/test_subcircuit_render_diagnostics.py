from pathlib import Path
from types import SimpleNamespace

from kicraft.autoplacer.brain import subcircuit_render_diagnostics as diagnostics


def test_build_leaf_contact_sheet_creates_nested_output(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    source.write_bytes(b"png")
    output = tmp_path / "nested" / "contact-sheet.png"
    commands: list[list[str]] = []

    monkeypatch.setattr(
        diagnostics.shutil,
        "which",
        lambda name: "/usr/bin/montage" if name == "montage" else None,
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"contact-sheet")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(diagnostics.subprocess, "run", fake_run)

    result = diagnostics.build_leaf_contact_sheet([source], output)

    assert result["created"] is True
    assert result["errors"] == []
    assert commands == [[
        "montage",
        str(source),
        "-background",
        "white",
        "-tile",
        "2x2",
        "-geometry",
        "+8+8",
        str(output),
    ]]
    assert output.read_bytes() == b"contact-sheet"
