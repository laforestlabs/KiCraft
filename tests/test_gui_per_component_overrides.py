"""Round-trip tests for the GUI per-component overrides plumbing.

The GUI's per-component panel reads project ``autoplacer.json``, lets
the user toggle leaf-level + ref-level overrides via NiceGUI controls,
and writes back. This file pins down the read/write/diff contract
without spinning up the NiceGUI server -- import the helpers, drive
them through ``AppState``, and assert on the resulting config dict.

Why this matters: the leaf-level switches (currently
``backside_through_hole_leaves``) sit at ``parent_placement.<key>`` in
the JSON. Past edits to the panel have re-written ``parent_placement``
without preserving sibling keys (e.g. ``candidate_search``). These
tests guard the partial-update contract so the GUI never wipes a hand-
edited cap or budget when the user toggles a sheet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.gui.state import AppState
from kicraft.layout_editor.rules import (
    _build_updated_config,
    _diff_dicts,
    _load_overrides_into_state,
    _set_backside_through_hole,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Empty project root with a minimal autoplacer.json."""
    cfg = tmp_path / f"{tmp_path.name}_autoplacer.json"
    cfg.write_text(
        json.dumps(
            {
                "component_zones": {"J1": {"edge": "left"}},
                "thermal_refs": ["U2"],
                "parent_placement": {
                    "candidate_search": {
                        "k": 4,
                        "time_budget_s": 240.0,
                        "max_outline_height_mm": 120.0,
                    },
                    "backside_through_hole_leaves": ["BATT"],
                },
            },
            indent=2,
        )
    )
    return tmp_path


def _state(project_root: Path) -> AppState:
    return AppState(project_root=project_root, strategy={})


def test_load_seeds_backside_overrides_from_cfg(project: Path) -> None:
    state = _state(project)
    _load_overrides_into_state(state)
    assert state.backside_through_hole_overrides == {"BATT"}


def test_load_handles_missing_parent_placement(tmp_path: Path) -> None:
    """A project without parent_placement (or without the backside list)
    should leave the override set empty -- not raise on the dotted lookup."""
    cfg = tmp_path / f"{tmp_path.name}_autoplacer.json"
    cfg.write_text(json.dumps({"component_zones": {}}))

    state = AppState(project_root=tmp_path, strategy={})
    _load_overrides_into_state(state)
    assert state.backside_through_hole_overrides == set()


def test_toggle_writes_sorted_list_back_to_parent_placement(project: Path) -> None:
    state = _state(project)
    _load_overrides_into_state(state)

    _set_backside_through_hole(state, "TERMINAL_BLOCK", True)
    _set_backside_through_hole(state, "AAA_HOLDER", True)

    cfg = _build_updated_config(state)
    assert cfg["parent_placement"]["backside_through_hole_leaves"] == [
        "AAA_HOLDER",
        "BATT",
        "TERMINAL_BLOCK",
    ]


def test_disable_drops_key_but_preserves_candidate_search(project: Path) -> None:
    """When the user disables the last backside leaf, the
    backside_through_hole_leaves key is dropped from parent_placement,
    but other parent_placement entries (candidate_search caps, budget)
    must survive."""
    state = _state(project)
    _load_overrides_into_state(state)

    _set_backside_through_hole(state, "BATT", False)

    cfg = _build_updated_config(state)
    assert "backside_through_hole_leaves" not in cfg["parent_placement"]
    assert cfg["parent_placement"]["candidate_search"] == {
        "k": 4,
        "time_budget_s": 240.0,
        "max_outline_height_mm": 120.0,
    }


def test_disable_strips_empty_parent_placement_block(tmp_path: Path) -> None:
    """If parent_placement only carried backside_through_hole_leaves and
    the user clears the list, the entire empty block is dropped so the
    JSON stays minimal."""
    cfg_path = tmp_path / f"{tmp_path.name}_autoplacer.json"
    cfg_path.write_text(
        json.dumps(
            {"parent_placement": {"backside_through_hole_leaves": ["BATT"]}}
        )
    )

    state = AppState(project_root=tmp_path, strategy={})
    _load_overrides_into_state(state)
    _set_backside_through_hole(state, "BATT", False)

    cfg = _build_updated_config(state)
    assert "parent_placement" not in cfg


def test_diff_reports_added_and_removed_backside_entries(project: Path) -> None:
    state = _state(project)
    _load_overrides_into_state(state)

    _set_backside_through_hole(state, "BATT", False)
    _set_backside_through_hole(state, "TERMINAL_BLOCK", True)

    before = json.loads((project / f"{project.name}_autoplacer.json").read_text())
    after = _build_updated_config(state)
    rows = _diff_dicts(before, after)
    keys = {row[1] for row in rows}
    assert "backside_through_hole_leaves: BATT" in keys
    assert "backside_through_hole_leaves: TERMINAL_BLOCK" in keys


def test_unrelated_parent_placement_keys_pass_through_unchanged(
    tmp_path: Path,
) -> None:
    """A future cfg key under parent_placement we don't manage in the GUI
    must survive a save round-trip. Pin this so a panel author doesn't
    accidentally rewrite parent_placement = {only_their_keys}."""
    cfg_path = tmp_path / f"{tmp_path.name}_autoplacer.json"
    cfg_path.write_text(
        json.dumps(
            {
                "parent_placement": {
                    "candidate_search": {"k": 8},
                    "future_unrelated_key": {"nested": True},
                    "backside_through_hole_leaves": ["BATT"],
                }
            }
        )
    )

    state = AppState(project_root=tmp_path, strategy={})
    _load_overrides_into_state(state)
    _set_backside_through_hole(state, "BATT", False)
    _set_backside_through_hole(state, "TERMINAL_BLOCK", True)

    cfg = _build_updated_config(state)
    pp = cfg["parent_placement"]
    assert pp["candidate_search"] == {"k": 8}
    assert pp["future_unrelated_key"] == {"nested": True}
    assert pp["backside_through_hole_leaves"] == ["TERMINAL_BLOCK"]
