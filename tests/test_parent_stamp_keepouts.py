"""Regression tests for Defect 2: mounting-hole keep-outs must reach the autorouter.

The parent composition builds `parent_local_keep_in_rects` around locked
parent-local components such as mounting holes. Before the D2 fix these rects
only influenced placement but were never emitted as KiCad rule-area zones, so
the autorouter could lay tracks straight through H4/H86.

These tests lock in the two contract points that were broken:
  1. The stamp subprocess script creates rule-area zones (not plain zones),
     with track/via/copper-pour disallowed, on both F.Cu and B.Cu.
  2. State.parent_local_keep_in_rects survive the JSON round-trip into the
     subprocess payload under the `keepouts` key.

The stamp script lives at ``kicraft/cli/_parent_stamp_subprocess.py`` after
being lifted out of an inline string literal -- the test reads its source
directly so a renamed/refactored attribute can't silently dodge the check.
"""

from pathlib import Path

from kicraft.autoplacer.brain.types import Point
from kicraft.cli.compose_subcircuits import _PARENT_STAMP_SCRIPT_PATH


def _stamp_script_source() -> str:
    return Path(_PARENT_STAMP_SCRIPT_PATH).read_text(encoding="utf-8")









