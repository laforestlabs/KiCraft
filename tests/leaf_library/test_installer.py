"""Tests for kicraft.leaf_library.installer.

These exercise the install pipeline against synthetic kicad_sch / kicad_pcb
text (not real solver output) so we don't need KiCad on the test host.
The renumber + pin-manager + autoplacer-merge behaviors are covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.autoplacer.brain import pins as pins_module
from kicraft.leaf_library.installer import (
    LIBRARY_IMPORT_SNAPSHOT_ID,
    install_leaf,
)
from kicraft.leaf_library.loader import LeafLibrary
from tests.leaf_library.test_loader import _populate_valid_leaf


SYNTH_SCH = """
(kicad_sch (version 20250114)
  (symbol (lib_id "Device:R") (uuid "u1")
    (property "Reference" "U1" (at 0 0 0))
    (instances (project "X" (path "/sheet-uuid"
      (reference "U1") (unit 1)))))
  (symbol (lib_id "Device:C") (uuid "c1")
    (property "Reference" "C1" (at 0 0 0))
    (instances (project "X" (path "/sheet-uuid"
      (reference "C1") (unit 1)))))
)
""".strip()

SYNTH_PCB = """
(kicad_pcb (version 20241229)
  (footprint "Resistor_SMD:R_0402"
    (property "Reference" "U1" (at 0 0 0) (layer "F.SilkS"))
    (fp_text reference "U1" (at 0 0 0) (layer "F.SilkS")))
  (footprint "Capacitor_SMD:C_0402"
    (property "Reference" "C1" (at 0 0 0) (layer "F.SilkS")))
)
""".strip()


def _populate_installable_leaf(leaf_dir: Path) -> "LeafLibrary":
    """Create a library leaf with the synthetic sch/pcb text."""
    _populate_valid_leaf(leaf_dir)
    (leaf_dir / "schematic.kicad_sch").write_text(SYNTH_SCH, encoding="utf-8")
    (leaf_dir / "leaf_routed.kicad_pcb").write_text(SYNTH_PCB, encoding="utf-8")
    (leaf_dir / "autoplacer_fragment.json").write_text(
        json.dumps({
            "ic_groups": {"U1": ["C1"]},
            "group_labels": {"U1": "TEST"},
            "thermal_refs": ["U1"],
            "signal_flow_order": ["U1", "C1"],
        }),
        encoding="utf-8",
    )
    # Re-stamp the manifest's refs to match the synthetic file.
    from kicraft.leaf_library.manifest import (
        compute_content_hash,
        dump_manifest,
        load_manifest,
    )
    m = load_manifest(leaf_dir)
    m = m.model_copy(update={"refs": ["U1", "C1"]})
    dump_manifest(m, leaf_dir)
    real = compute_content_hash(leaf_dir)
    m = m.model_copy(update={"content_hash": real})
    dump_manifest(m, leaf_dir)


def test_install_renumbers_and_pins(tmp_path):
    lib_dir = tmp_path / "lib"
    proj_dir = tmp_path / "proj"
    lib_dir.mkdir()
    proj_dir.mkdir()

    _populate_installable_leaf(lib_dir / "test-leaf")
    lib = LeafLibrary(lib_dir)
    leaf = lib.find("test-leaf@0.1.0")
    assert leaf is not None

    autoplacer: dict = {}
    result = install_leaf(
        leaf,
        project_dir=proj_dir,
        sheet_name="CHARGER",
        sheet_stem="CHARGER",
        sheet_uuid="sheet-uuid-xyz",
        instance=1,
        project_refs=["U5", "C3"],
        autoplacer_dict=autoplacer,
        check_dependencies=False,
    )

    # Refs got renumbered to next free slots.
    assert result.ref_map == {"U1": "U6", "C1": "C4"}

    # Project schematic exists at <project>/CHARGER.kicad_sch and has the
    # new refs.
    new_sch = (proj_dir / "CHARGER.kicad_sch").read_text(encoding="utf-8")
    assert '(property "Reference" "U6"' in new_sch
    assert '(property "Reference" "C4"' in new_sch
    assert '(reference "U6")' in new_sch
    assert "U1" not in new_sch  # no orphan original ref
    assert "C1" not in new_sch

    # Snapshot triad written under the derived leaf_key.
    leaf_key = result.leaf_key
    artifact_dir = proj_dir / ".experiments" / "subcircuits" / leaf_key
    for canonical in (
        "round_lib0001_leaf_routed.kicad_pcb",
        "round_lib0001_metadata.json",
        "round_lib0001_solved_layout.json",
    ):
        assert (artifact_dir / canonical).exists(), f"missing {canonical}"

    # PCB content was renumbered.
    new_pcb = (artifact_dir / "round_lib0001_leaf_routed.kicad_pcb").read_text(encoding="utf-8")
    assert '"Reference" "U6"' in new_pcb
    assert '"Reference" "C4"' in new_pcb

    # Pin entry exists in pins.json with string snapshot id.
    pins_manifest = pins_module.read_pins(proj_dir / ".experiments")
    assert leaf_key in pins_manifest["pinned_leaves"]
    pin = pins_manifest["pinned_leaves"][leaf_key]
    assert pin["round"] == "lib0001"
    assert pin["source"].startswith("library:test-leaf@")

    # Autoplacer fragment merged with renumbered keys.
    assert autoplacer["ic_groups"] == {"U6": ["C4"]}
    assert autoplacer["group_labels"] == {"U6": "TEST"}
    assert autoplacer["thermal_refs"] == ["U6"]
    assert autoplacer["signal_flow_order"] == ["U6", "C4"]


def test_install_multi_instance_no_collision(tmp_path):
    lib_dir = tmp_path / "lib"
    proj_dir = tmp_path / "proj"
    lib_dir.mkdir()
    proj_dir.mkdir()
    _populate_installable_leaf(lib_dir / "test-leaf")
    lib = LeafLibrary(lib_dir)
    leaf = lib.find("test-leaf@0.1.0")
    assert leaf is not None

    autoplacer: dict = {}
    project_refs: list[str] = []

    r1 = install_leaf(
        leaf,
        project_dir=proj_dir,
        sheet_name="CHARGER",
        sheet_stem="CHARGER",
        sheet_uuid="uuid-1",
        instance=1,
        project_refs=project_refs,
        autoplacer_dict=autoplacer,
        check_dependencies=False,
    )
    project_refs.extend(r1.ref_map.values())

    r2 = install_leaf(
        leaf,
        project_dir=proj_dir,
        sheet_name="CHARGER_2",
        sheet_stem="CHARGER_2",
        sheet_uuid="uuid-2",
        instance=2,
        project_refs=project_refs,
        autoplacer_dict=autoplacer,
        check_dependencies=False,
    )

    # Different leaf_keys (different UUIDs).
    assert r1.leaf_key != r2.leaf_key

    # Ref ranges don't overlap.
    assert set(r1.ref_map.values()).isdisjoint(set(r2.ref_map.values()))

    # Autoplacer merge combined both instances without collision.
    assert set(autoplacer["ic_groups"]) == set(r1.ref_map.values()) & set(
        autoplacer["ic_groups"]
    ) | (set(r2.ref_map.values()) & set(autoplacer["ic_groups"]))


def test_install_content_hash_mismatch_at_install_time(tmp_path):
    lib_dir = tmp_path / "lib"
    proj_dir = tmp_path / "proj"
    lib_dir.mkdir()
    proj_dir.mkdir()
    _populate_installable_leaf(lib_dir / "test-leaf")
    lib = LeafLibrary(lib_dir)
    leaf = lib.find("test-leaf@0.1.0")
    assert leaf is not None

    # Tamper with a file after loader cached the hash.
    (lib_dir / "test-leaf" / "bom.csv").write_text(
        "ref,value\nBOGUS,oops\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="content_hash mismatch at install time"):
        install_leaf(
            leaf,
            project_dir=proj_dir,
            sheet_name="X",
            sheet_stem="X",
            sheet_uuid="u",
            instance=1,
            project_refs=[],
            autoplacer_dict={},
            check_dependencies=False,
        )


def test_ensure_applied_honors_library_pin(tmp_path):
    """ensure_applied after a library install should be a no-op (already current)
    or re-apply identically — i.e. the snapshot id is found by the pin manager.
    """
    lib_dir = tmp_path / "lib"
    proj_dir = tmp_path / "proj"
    lib_dir.mkdir()
    proj_dir.mkdir()
    _populate_installable_leaf(lib_dir / "test-leaf")
    lib = LeafLibrary(lib_dir)
    leaf = lib.find("test-leaf@0.1.0")
    assert leaf is not None

    install_leaf(
        leaf,
        project_dir=proj_dir,
        sheet_name="X",
        sheet_stem="X",
        sheet_uuid="u",
        instance=1,
        project_refs=[],
        autoplacer_dict={},
        check_dependencies=False,
    )

    statuses = pins_module.ensure_applied(proj_dir / ".experiments")
    assert len(statuses) == 1
    # Either "applied" or "already-current" is acceptable; we just want
    # the pin to be valid (not "snapshot-missing").
    assert next(iter(statuses.values())) in ("applied", "already-current")
