"""Tests for the leaf pin manifest and snapshot application."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.autoplacer.brain import pins


LEAF_KEY = "leaf-uuid__abcd1234"


def _seed_round_snapshot(
    experiments_dir: Path, leaf_key: str, round_num: int, content: str
) -> Path:
    leaf_dir = experiments_dir / "subcircuits" / leaf_key
    leaf_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"round_{round_num:04d}"
    (leaf_dir / f"{prefix}_leaf_routed.kicad_pcb").write_text(
        f"PCB-r{round_num}-{content}"
    )
    (leaf_dir / f"{prefix}_metadata.json").write_text(
        json.dumps({"round": round_num, "tag": content})
    )
    (leaf_dir / f"{prefix}_solved_layout.json").write_text(
        json.dumps({"round": round_num, "layout": content})
    )
    return leaf_dir


def test_read_pins_returns_empty_manifest_when_missing(tmp_path):
    manifest = pins.read_pins(tmp_path)
    assert manifest["pinned_leaves"] == {}
    assert manifest["schema_version"] == pins.PINS_SCHEMA_VERSION


def test_list_available_rounds_filters_incomplete(tmp_path):
    leaf_dir = _seed_round_snapshot(tmp_path, LEAF_KEY, 3, "good")
    # Round 5 is incomplete -- only the PCB exists, no metadata/solved_layout.
    (leaf_dir / "round_0005_leaf_routed.kicad_pcb").write_text("partial")

    available = pins.list_available_rounds(tmp_path, LEAF_KEY)
    assert available == [3]


def test_pin_leaf_copies_snapshot_and_writes_manifest(tmp_path):
    leaf_dir = _seed_round_snapshot(tmp_path, LEAF_KEY, 7, "winner")

    pins.pin_leaf(tmp_path, LEAF_KEY, 7)

    # Canonical files should now mirror the round-7 snapshot
    assert (leaf_dir / "leaf_routed.kicad_pcb").read_text() == "PCB-r7-winner"
    assert json.loads((leaf_dir / "metadata.json").read_text())["tag"] == "winner"
    assert json.loads((leaf_dir / "solved_layout.json").read_text())["layout"] == "winner"

    # And manifest reflects the pin
    manifest = pins.read_pins(tmp_path)
    assert manifest["pinned_leaves"][LEAF_KEY]["round"] == 7
    assert pins.is_pinned(tmp_path, LEAF_KEY) == 7


def test_pin_leaf_raises_when_snapshot_incomplete(tmp_path):
    leaf_dir = tmp_path / "subcircuits" / LEAF_KEY
    leaf_dir.mkdir(parents=True)
    (leaf_dir / "round_0002_leaf_routed.kicad_pcb").write_text("partial")

    with pytest.raises(FileNotFoundError):
        pins.pin_leaf(tmp_path, LEAF_KEY, 2)

    # And nothing got pinned
    assert pins.is_pinned(tmp_path, LEAF_KEY) is None


def test_unpin_leaf_removes_manifest_entry_but_leaves_files(tmp_path):
    leaf_dir = _seed_round_snapshot(tmp_path, LEAF_KEY, 4, "kept")
    pins.pin_leaf(tmp_path, LEAF_KEY, 4)
    canonical_pcb = leaf_dir / "leaf_routed.kicad_pcb"
    assert canonical_pcb.read_text() == "PCB-r4-kept"

    removed = pins.unpin_leaf(tmp_path, LEAF_KEY)
    assert removed is True
    assert pins.is_pinned(tmp_path, LEAF_KEY) is None
    # On-disk canonical files are deliberately left intact -- documented
    # behavior so unpin doesn't surprise the user with a layout change.
    assert canonical_pcb.read_text() == "PCB-r4-kept"

    # Unpinning again is a no-op
    assert pins.unpin_leaf(tmp_path, LEAF_KEY) is False


def test_ensure_applied_restores_canonical_after_overwrite(tmp_path):
    """If a leaf re-solve overwrites the canonical files, ensure_applied
    must restore the pinned snapshot. This is the composer's hook."""
    leaf_dir = _seed_round_snapshot(tmp_path, LEAF_KEY, 9, "pinned-state")
    pins.pin_leaf(tmp_path, LEAF_KEY, 9)

    # Simulate a later leaf solve overwriting the canonical files
    (leaf_dir / "leaf_routed.kicad_pcb").write_text("OVERWRITTEN")
    (leaf_dir / "metadata.json").write_text("{}")

    statuses = pins.ensure_applied(tmp_path)
    assert statuses[LEAF_KEY] == "applied"
    assert (leaf_dir / "leaf_routed.kicad_pcb").read_text() == "PCB-r9-pinned-state"

    # Second call is a no-op since files already match
    statuses = pins.ensure_applied(tmp_path)
    assert statuses[LEAF_KEY] == "already-current"


def test_ensure_applied_reports_missing_snapshot(tmp_path):
    """If the pinned round's snapshot was deleted, ensure_applied must not
    silently break -- it reports snapshot-missing so the caller can surface it."""
    _seed_round_snapshot(tmp_path, LEAF_KEY, 11, "x")
    pins.pin_leaf(tmp_path, LEAF_KEY, 11)
    # Wipe the snapshot
    leaf_dir = tmp_path / "subcircuits" / LEAF_KEY
    for snap in leaf_dir.glob("round_0011_*"):
        snap.unlink()

    statuses = pins.ensure_applied(tmp_path)
    assert statuses[LEAF_KEY] == "snapshot-missing"


def test_ensure_applied_with_no_pins_is_noop(tmp_path):
    assert pins.ensure_applied(tmp_path) == {}
