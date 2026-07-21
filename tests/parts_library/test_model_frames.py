"""The .wrl/.step frame-agreement machinery (validate-part check (10)).

The fab STEP export substitutes the same-named .step for each footprint's
.wrl reference under ONE shared (model ...) transform, so the two files must
sit in the same native frame; a shifted .step embeds that part detached from
the board in every fab package (2026-07-21 sweep: 31/93 vendored bundles).
These tests pin the detector's behavior on real vendored bundles and on
synthetically displaced copies made with the same transform helper the
re-vendor tool (scripts/restep_model_frames.py) uses.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kicraft.parts_library.model_frames import (
    frame_mismatch,
    transform_step_file,
)

LIB = Path(__file__).resolve().parents[2] / "kicraft" / "parts_library"


def _pair(name: str) -> tuple[Path, Path]:
    d = LIB / name / "3d"
    return sorted(d.glob("*.wrl"))[0], sorted(d.glob("*.step"))[0]


def test_matched_bundle_passes():
    wrl, step = _pair("ss14")
    assert frame_mismatch(wrl, step) is None


def test_different_artwork_passes():
    # mes104j2a's wrl draws the winding coil, its step the plain case
    # cylinder: hugely different solids, but pins register at identity and
    # no rigid transform improves the fit -- NOT a frame defect.
    wrl, step = _pair("mes104j2a-7-50r0")
    assert frame_mismatch(wrl, step) is None


@pytest.mark.parametrize(
    "bundle,deg,t",
    [
        ("ss14", 0, (5.0, 0.0, 0.0)),
        ("ss14", 0, (0.0, 0.0, 0.8)),
        # a 180 flip needs an asymmetric part to be observable at all; on a
        # symmetric body (ss14's SMA package) it is genuinely cosmetic-only
        # and the detector deliberately stays quiet.
        ("usb-c-16p", 180, (0.0, 0.0, 0.0)),
    ],
    ids=["shift-x", "shift-z", "rot180"],
)
def test_displaced_step_flagged(tmp_path, bundle, deg, t):
    wrl, step = _pair(bundle)
    moved = tmp_path / step.name
    shutil.copy(step, moved)
    transform_step_file(moved, deg, t)
    reason = frame_mismatch(wrl, moved)
    assert reason is not None and "model frames disagree" in reason


def test_transform_step_file_round_trips(tmp_path):
    wrl, step = _pair("ss14")
    moved = tmp_path / step.name
    shutil.copy(step, moved)
    npts, _ = transform_step_file(moved, 0, (3.0, -2.0, 1.0))
    assert npts > 100
    transform_step_file(moved, 0, (-3.0, 2.0, -1.0))
    assert frame_mismatch(wrl, moved) is None


def test_validate_part_rejects_frame_shifted_step(tmp_path):
    # End-to-end through the CLI: check (10) must fail a bundle whose .step
    # drifted out of the .wrl frame (this is how a raw re-fetch regresses).
    part = tmp_path / "ss14"
    shutil.copytree(LIB / "ss14", part)
    step = sorted((part / "3d").glob("*.step"))[0]
    transform_step_file(step, 0, (5.0, 0.0, 0.0))
    res = subprocess.run(
        [sys.executable, "-m", "kicraft.design.cli_app", "validate-part", str(part)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 2
    assert "model frames disagree" in res.stderr
    assert "restep_model_frames" in res.stderr
