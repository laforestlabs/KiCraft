"""`_set_board_clearance_um`: make a routed board declare the (lowered) clearance
it was actually routed to, so KiCad DRC validates against the same rule the
fine-pitch FreeRouting lower used.

Root cause this guards (2026-06-05): `_resolve_fine_pitch_rule` lowers the
autorouter clearance globally (e.g. 0.2 -> 0.153 mm) so traces can escape a dense
pad field (USB-C, fine-pitch ICs), but the board kept its 0.2 mm default
clearance, so every trace routed tighter than 0.2 mm was a DRC clearance
violation -> the geometry acceptance gate rejected an otherwise fab-clean board
(BMP280 fixture, web-default quality=good, build exit 7)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from kicraft.autoplacer.freerouting_runner import _set_board_clearance_um

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pcbnew") is None, reason="pcbnew not available"
)


def _mk_board(path, clearance_mm: float) -> None:
    """Create a board whose default netclass clearance is `clearance_mm`."""
    script = (
        "import pcbnew\n"
        f"b = pcbnew.NewBoard({str(path)!r})\n"
        "b.GetDesignSettings().m_NetSettings.GetDefaultNetclass()."
        f"SetClearance(pcbnew.FromMM({clearance_mm}))\n"
        f"b.Save({str(path)!r})\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True,
                   capture_output=True, text=True)


def _read_clearance_mm(path) -> float:
    script = (
        "import pcbnew\n"
        f"b = pcbnew.LoadBoard({str(path)!r})\n"
        "print(pcbnew.ToMM(b.GetDesignSettings().m_NetSettings."
        "GetDefaultNetclass().GetClearance()))\n"
    )
    r = subprocess.run([sys.executable, "-c", script], check=True,
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def test_lowers_default_netclass_to_routed_clearance(tmp_path):
    p = tmp_path / "board.kicad_pcb"
    _mk_board(p, 0.2)
    _set_board_clearance_um(str(p), 153)  # routed at 0.153 mm
    assert abs(_read_clearance_mm(p) - 0.153) < 1e-4


def test_never_widens_clearance(tmp_path):
    # Called with a value WIDER than the board's rule: leave it (min semantics),
    # so an intentionally tighter class is never relaxed.
    p = tmp_path / "board.kicad_pcb"
    _mk_board(p, 0.153)
    _set_board_clearance_um(str(p), 300)
    assert abs(_read_clearance_mm(p) - 0.153) < 1e-4


def test_missing_board_is_non_fatal(tmp_path):
    # Best-effort: a bad path must not raise (routing would keep the old rule).
    _set_board_clearance_um(str(tmp_path / "does_not_exist.kicad_pcb"), 153)
