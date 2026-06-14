"""Part 3 acceptance metric: connector_edge_gap.

Fast unit tests pin the outward-gap arithmetic for every edge (flush / overhang
/ inboard). The gated integration test composes the committed fixture and
records the live measurement -- it is the harness that found the real
stranding: J1/J2 land flush, but the top-zoned switch SW1 is ~9mm INBOARD (the
90/270 convention bug, proven + documented on parent_adapter._rotated; its fix
is gated on parent-stamp robustness). When that fix lands, SW1 flips to flush
and the xfail here is removed.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kicraft.autoplacer.brain.connector_edge_gap import (
    EdgeGap,
    connector_edge_gaps,
    edge_gap_mm,
    stranded,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "replay_workspace" / "USB_PD_TRIGGER"
)
# board (x0,y0,x1,y1) = left,top,right,bottom in KiCad Y-down
BOARD = (0.0, 0.0, 20.0, 10.0)


@pytest.mark.parametrize(
    "edge, court, expected",
    [
        # flush: courtyard edge exactly on the board edge
        ("left", (0.0, 2.0, 5.0, 8.0), 0.0),
        ("right", (15.0, 2.0, 20.0, 8.0), 0.0),
        ("top", (2.0, 0.0, 8.0, 5.0), 0.0),
        ("bottom", (2.0, 5.0, 8.0, 10.0), 0.0),
        # overhang: courtyard past the board edge (positive)
        ("left", (-1.5, 2.0, 5.0, 8.0), 1.5),
        ("right", (15.0, 2.0, 22.0, 8.0), 2.0),
        # inboard / stranded: courtyard pulled in from the edge (negative)
        ("top", (2.0, 3.0, 8.0, 6.0), -3.0),
        ("bottom", (2.0, 4.0, 8.0, 9.0), -1.0),
    ],
)
def test_edge_gap_arithmetic(edge, court, expected):
    assert edge_gap_mm(edge, BOARD, court) == pytest.approx(expected)


def test_edge_gap_rejects_bad_edge():
    with pytest.raises(ValueError):
        edge_gap_mm("middle", BOARD, BOARD)


def test_stranded_filters_failures():
    gaps = [
        EdgeGap("J1", "left", 0.4, True),
        EdgeGap("SW1", "top", -9.2, False),
        EdgeGap("J2", "right", 0.4, True),
    ]
    assert [g.ref for g in stranded(gaps)] == ["SW1"]


# ---- gated integration: live measurement on the committed fixture -----------


def _compose_and_measure(tmp_path: Path) -> dict[str, EdgeGap]:
    dest = tmp_path / "USB_PD_TRIGGER"
    shutil.copytree(FIXTURE, dest)
    real = str(dest.resolve())
    for jf in (dest / ".experiments").rglob("*.json"):
        t = jf.read_text(encoding="utf-8")
        if "__KICRAFT_PROJECT_DIR__" in t:
            jf.write_text(t.replace("__KICRAFT_PROJECT_DIR__", real), encoding="utf-8")
    for p in glob.glob(str(dest / ".experiments" / "subcircuits"
                            / "subcircuit__*" / "parent_pre_freerouting.kicad_pcb")):
        os.remove(p)
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    rc = subprocess.run(
        [sys.executable, "-m", "kicraft.cli.compose_subcircuits",
         "--project", str(dest), "--parent", "USB_PD_TRIGGER",
         "--pcb", str(dest / "USB_PD_TRIGGER.kicad_pcb"),
         "--spacing-mm", "2.0", "--stamp", "--seed", "0"],
        cwd=str(Path(__file__).resolve().parent.parent), env=env,
    ).returncode
    assert rc == 0, f"compose exited {rc}"
    board = sorted(glob.glob(str(dest / ".experiments" / "subcircuits"
                                 / "subcircuit__*" / "parent_pre_freerouting.kicad_pcb")))[-1]
    zones = json.loads(
        (FIXTURE / "USB_PD_TRIGGER_autoplacer.json").read_text(encoding="utf-8")
    )["component_zones"]
    return {g.ref: g for g in connector_edge_gaps(board, zones)}


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (FIXTURE / ".experiments").is_dir(), reason="frozen-leaf fixture missing"
)
def test_edge_connectors_flush_on_fixture(tmp_path):
    pytest.importorskip("pcbnew")
    gaps = _compose_and_measure(tmp_path)
    # The USB-C edge connectors land flush (the metric's positive case).
    assert gaps["J1"].ok, gaps["J1"]
    assert gaps["J2"].ok, gaps["J2"]


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (FIXTURE / ".experiments").is_dir(), reason="frozen-leaf fixture missing"
)
@pytest.mark.xfail(
    reason="known stranding: SW1 (top-zoned) sits several mm inboard -- the "
    "90/270 convention bug + leaf-extremity; fix gated on parent-stamp "
    "robustness (see parent_adapter._rotated / the plan doc)",
    strict=True,
)
def test_top_zoned_switch_not_stranded(tmp_path):
    pytest.importorskip("pcbnew")
    gaps = _compose_and_measure(tmp_path)
    assert gaps["SW1"].ok, gaps["SW1"]
