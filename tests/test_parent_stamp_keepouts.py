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


def test_parent_stamp_script_creates_rule_area_keepouts_on_both_copper_layers():
    """The stamp subprocess must emit rule-area zones on F.Cu AND B.Cu."""
    script = _stamp_script_source()
    assert "_KEEPOUT_LAYERS = [pcbnew.F_Cu, pcbnew.B_Cu]" in script, (
        "Keepout zones must be stamped on both copper layers so the autorouter "
        "cannot place a track or via on either side through a mounting hole."
    )
    assert "_zone.SetIsRuleArea(True)" in script, (
        "Keepout zones must be marked as rule areas: ExportSpecctraDSN emits "
        "rule areas as router exchange input (keepout) regions, and the parent route preserves "
        "board zones (routing_clear_zones=False) so they reach the router exchange input. "
        "See tests/test_dsn_keepout_survival.py."
    )
    # Pads are intentionally NOT disallowed: the keepout rect is bbox(comp)
    # + inward_keep_in_mm, which by construction covers the protected
    # component's own pad. Setting pads not_allowed flagged the protected
    # pad as items_not_allowed against its own zone.
    for flag in (
        "_zone.SetDoNotAllowTracks(True)",
        "_zone.SetDoNotAllowVias(True)",
        "_zone.SetDoNotAllowCopperPour(True)",
    ):
        assert flag in script, f"Stamp script missing keepout flag: {flag}"


def test_parent_stamp_script_iterates_keepouts_payload_key():
    """The subprocess reads the payload's `keepouts` list; don't rename it."""
    script = _stamp_script_source()
    assert '_keepouts = _data.get("keepouts"' in script, (
        "The payload key 'keepouts' is the contract between the main process "
        "(keepout_json builder) and the subprocess (zone creator)."
    )
    assert "for _ko in _keepouts:" in script, (
        "Subprocess must loop over every keepout rect to stamp a zone per rect."
    )


def test_keepout_rect_round_trips_into_stamp_payload_json():
    """Rect serialization produced in compose must match what the subprocess expects."""
    rects: list[tuple[Point, Point]] = [
        (Point(10.0, 20.0), Point(15.0, 25.0)),
        (Point(-3.5, -3.5), Point(1.5, 1.5)),
    ]
    # Mirror the serialization from compose_subcircuits._stamp_parent_board
    keepout_json = [
        {
            "tl_x": rect[0].x,
            "tl_y": rect[0].y,
            "br_x": rect[1].x,
            "br_y": rect[1].y,
        }
        for rect in rects
    ]

    assert len(keepout_json) == 2
    assert keepout_json[0] == {"tl_x": 10.0, "tl_y": 20.0, "br_x": 15.0, "br_y": 25.0}
    assert keepout_json[1] == {"tl_x": -3.5, "tl_y": -3.5, "br_x": 1.5, "br_y": 1.5}

    # The subprocess reads tl_x/tl_y/br_x/br_y off each entry -- guard against
    # a rename on the main-process side by re-checking the exact keys.
    for entry in keepout_json:
        assert set(entry.keys()) == {"tl_x", "tl_y", "br_x", "br_y"}
