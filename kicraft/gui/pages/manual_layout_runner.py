"""Desktop-only helpers for the Manual Layout page.

Everything host-agnostic (leaf discovery, layout persistence, the
compose/stamp/route subprocess runner) moved to
``kicraft.layout_editor``; only the pcbnew launcher stays here because
it assumes a local KiCad install next to the GUI process.
"""

from __future__ import annotations

from pathlib import Path


def open_in_pcbnew(pcb_path: Path) -> None:
    """Launch pcbnew on the given board, detached from the GUI process.

    pcbnew is the right binary for opening a .kicad_pcb directly --
    `kicad <file>` invokes the project manager which only handles
    .kicad_pro. We Popen with start_new_session so killing the GUI
    doesn't take pcbnew with it.
    """
    import subprocess

    subprocess.Popen(
        ["pcbnew", str(pcb_path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
