"""Tests for kicraft.leaf_library.renumber."""

from __future__ import annotations

import pytest

from kicraft.leaf_library.renumber import apply_ref_map, parse_ref, renumber_leaf


def test_parse_ref_basic():
    assert parse_ref("U7") == ("U", 7)
    assert parse_ref("BT1") == ("BT", 1)
    assert parse_ref("R42") == ("R", 42)


def test_parse_ref_rejects_suffix():
    with pytest.raises(ValueError):
        parse_ref("U1A")
    with pytest.raises(ValueError):
        parse_ref("U")
    with pytest.raises(ValueError):
        parse_ref("u1")


def test_renumber_into_empty_project():
    """No prior refs -> leaf refs land in slot 1 per class."""
    ref_map = renumber_leaf(
        leaf_refs=["U1", "U2", "C1", "C2", "R1"],
        project_refs=[],
    )
    assert ref_map == {"C1": "C1", "C2": "C2", "R1": "R1", "U1": "U1", "U2": "U2"}


def test_renumber_with_existing_refs():
    ref_map = renumber_leaf(
        leaf_refs=["U1", "U2", "C1", "R1"],
        project_refs=["U5", "C3", "R8"],
    )
    assert ref_map["U1"] == "U6"
    assert ref_map["U2"] == "U7"
    assert ref_map["C1"] == "C4"
    assert ref_map["R1"] == "R9"


def test_renumber_multi_instance():
    """Second instance sees first instance's allocations."""
    leaf_refs = ["U1", "C1", "C2"]
    project_refs = ["U7", "C3"]

    inst1 = renumber_leaf(leaf_refs, project_refs)
    assert inst1 == {"U1": "U8", "C1": "C4", "C2": "C5"}

    # The caller now includes inst1's allocations in project_refs.
    project_refs_after = project_refs + list(inst1.values())
    inst2 = renumber_leaf(leaf_refs, project_refs_after)
    assert inst2 == {"U1": "U9", "C1": "C6", "C2": "C7"}

    # No overlap.
    assert set(inst1.values()).isdisjoint(set(inst2.values()))


def test_renumber_determinism():
    """Same inputs -> same outputs across runs."""
    leaf_refs = ["U2", "U1", "C5", "C1", "R3"]
    project_refs = ["R7", "C2", "U10"]
    m1 = renumber_leaf(leaf_refs, project_refs)
    m2 = renumber_leaf(leaf_refs, project_refs)
    assert m1 == m2


def test_renumber_ignores_non_pattern_project_refs():
    """Refs like #PWR01 in the project don't trip the parser."""
    ref_map = renumber_leaf(
        leaf_refs=["U1"],
        project_refs=["U2", "#PWR01", "#FLG02"],
    )
    assert ref_map == {"U1": "U3"}


def test_apply_ref_map_to_dict_keys_and_values():
    src = {
        "ic_groups": {"U1": ["C1", "C2"]},
        "group_labels": {"U1": "CHARGER"},
        "thermal_refs": ["U1", "Q1"],
        "signal_flow_order": ["J1", "U1", "U2"],
        "sheet_name": "CHARGER",  # NOT a ref; passes through
    }
    ref_map = {"U1": "U7", "U2": "U8", "C1": "C12", "C2": "C13", "Q1": "Q3", "J1": "J5"}
    result = apply_ref_map(src, ref_map)
    assert result == {
        "ic_groups": {"U7": ["C12", "C13"]},
        "group_labels": {"U7": "CHARGER"},
        "thermal_refs": ["U7", "Q3"],
        "signal_flow_order": ["J5", "U7", "U8"],
        "sheet_name": "CHARGER",
    }


def test_apply_ref_map_leaves_unmapped_strings_alone():
    src = {"ic_groups": {"U1": ["C1", "C99"]}}
    ref_map = {"U1": "U2", "C1": "C5"}  # C99 not mapped
    result = apply_ref_map(src, ref_map)
    assert result == {"ic_groups": {"U2": ["C5", "C99"]}}
