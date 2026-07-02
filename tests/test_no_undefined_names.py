"""Package-wide undefined-name (F821) guard.

Two shipped incidents were plain NameErrors introduced by refactors that
deleted or renamed a binding but kept a use site: the copper_accounting
``_trace_length`` deletion (killed every parent compose, 0/28 fab-ready) and
the 4d359f0 ``render_intermediate`` gate in ``_stamp_trivial_leaf`` (killed
every build with a trivial leaf, KC-V8YWN8). Both are exactly ruff's F821.
This test keeps the whole package at zero so that class of regression fails
in CI instead of in production builds.

Known-false-positive escapes (late-bound closures) are annotated with
``# noqa: F821`` at the use site with a comment explaining the binding.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _ruff() -> str | None:
    venv_ruff = Path(sys.executable).parent / "ruff"
    if venv_ruff.exists():
        return str(venv_ruff)
    return shutil.which("ruff")


@pytest.mark.skipif(_ruff() is None, reason="ruff not installed")
def test_no_undefined_names_in_package():
    proc = subprocess.run(
        [_ruff(), "check", "--select", "F821", "--no-cache", "kicraft/"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "undefined names (future NameErrors) found:\n" + proc.stdout + proc.stderr
    )
