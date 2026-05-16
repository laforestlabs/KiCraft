"""Emit an empty `<PROJECT>.kicad_pcb` stub.

The contract doc §1 lists this file as OPTIONAL ("KiCraft will create one
if absent"), but in practice `solve-subcircuits` and `autoexperiment`
require the PCB to exist before they start. We ship the empty stub so the
downstream pipeline can run immediately on synthesized output without
manual fiddling.

Uses `pcbnew.NewBoard(path)` — the canonical API for creating a fresh
empty board file, matching whatever the installed KiCad version expects.
"""
from __future__ import annotations

from pathlib import Path


def write_empty_pcb(project_dir: Path, project_stem: str) -> Path:
    """Create `<project_stem>.kicad_pcb` as an empty pcbnew board. Returns path."""
    import pcbnew  # noqa: WPS433 — local import keeps non-pcbnew callers (tests) clean

    out = project_dir / f"{project_stem}.kicad_pcb"
    pcbnew.NewBoard(str(out))
    return out
