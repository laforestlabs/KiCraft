"""The durable `placement` state section: model, commit, and merge.

- PlacementSection validates anchor-spec SHAPE (catches UI bugs) but
  deliberately not ref existence (parts churn across BOM re-runs).
- `stage-commit placement` is deterministic: it commits without
  touching any LLM stage, invalidates nothing downstream, and surfaces
  stale refs as warnings.
- write_autoplacer_json merges with precedence library fragments <
  BOM (LLM) < placement (user), drops stale placement refs (the §9.6
  named-refs check would otherwise fail synthesis), and emits fixed
  board dimensions.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    ConversationState,
    PlacementSection,
    Sheet,
)
from kicraft.design.synthesis.autoplacer import write_autoplacer_json
from kicraft.server.session import downstream_stages


# ---- model -------------------------------------------------------------------


def test_placement_section_validates_spec_shape():
    PlacementSection(component_zones={"J1": {"edge": "left", "rotation": 90.0}})
    with pytest.raises(ValidationError, match="unknown keys"):
        PlacementSection(component_zones={"J1": {"side": "left"}})
    with pytest.raises(ValidationError, match="at most one anchor"):
        PlacementSection(component_zones={"J1": {"edge": "left", "corner": "top-left"}})
    with pytest.raises(ValidationError, match="not in"):
        PlacementSection(component_zones={"J1": {"edge": "diagonal"}})
    with pytest.raises(ValidationError, match="0..360"):
        PlacementSection(component_zones={"J1": {"rotation": 720.0}})
    with pytest.raises(ValidationError, match=">= 10"):
        PlacementSection(board={"width_mm": 2.0, "height_mm": 50.0})


def test_placement_refs_not_validated_against_parts():
    """Stale refs must survive the model (they degrade at synthesis)."""
    PlacementSection(component_zones={"GHOST9": {"corner": "top-left"}},
                     thermal_refs=["GHOST9"])


def test_placement_is_not_a_design_stage():
    assert downstream_stages("placement") == []


# ---- stage-commit --------------------------------------------------------------


def _bom() -> BOM:
    return BOM(parts=[
        BomPart(ref="U1", value="ESP32", sheet="MCU",
                symbol="Device:R", footprint="Resistor_SMD:R_0402_1005Metric"),
        BomPart(ref="J1", value="USB-C", sheet="MCU",
                symbol="Device:R", footprint="Resistor_SMD:R_0402_1005Metric"),
    ])


def test_stage_commit_placement_round_trips_and_warns_on_stale_refs(tmp_path, capsys):
    import kicraft.design.cli_app as cli_app

    state = ConversationState(project_stem="WIDGET", bom=_bom())
    state_path = tmp_path / "state.json"
    state_path.write_text(state.model_dump_json(), encoding="utf-8")

    slot = {
        "component_zones": {"J1": {"edge": "left"}, "GHOST9": {"corner": "top-left"}},
        "thermal_refs": ["U1"],
        "backside_through_hole_leaves": ["BATT"],
        "board": {"width_mm": 60.0, "height_mm": 60.0, "size_search": False},
    }
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(json.dumps(slot), encoding="utf-8")

    rc = cli_app.main(["stage-commit", "placement", str(state_path),
                       "--slot-file", str(slot_file), "--no-archive"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert any("GHOST9" in w for w in out.get("warnings", [])), (
        "stale placement refs must surface as commit warnings")

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["placement"]["component_zones"]["J1"] == {"edge": "left"}
    assert persisted["placement"]["board"]["size_search"] is False
    # Nothing upstream was touched.
    assert persisted["bom"]["parts"][0]["ref"] == "U1"


def test_stage_commit_placement_rejects_bad_spec(tmp_path, capsys):
    import kicraft.design.cli_app as cli_app

    state_path = tmp_path / "state.json"
    state_path.write_text(
        ConversationState(project_stem="W").model_dump_json(), encoding="utf-8")
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(
        json.dumps({"component_zones": {"J1": {"edge": "diagonal"}}}),
        encoding="utf-8")
    rc = cli_app.main(["stage-commit", "placement", str(state_path),
                       "--slot-file", str(slot_file), "--no-archive"])
    assert rc == 3
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False


# ---- autoplacer merge -----------------------------------------------------------


def _arch() -> Architecture:
    return Architecture(
        sheets=[Sheet(name="MCU", stem="MCU", function="microcontroller")],
        power_nets=["+3V3", "GND"],
        inter_sheet_nets=[],
    )


def test_write_autoplacer_merge_precedence_and_stale_drop(tmp_path, capsys):
    bom = _bom()
    # LLM zone for J1 says bottom; the user's placement says left -> user wins.
    bom.component_zones["J1"] = {"edge": "bottom"}
    placement = PlacementSection(
        component_zones={"J1": {"edge": "left"},
                         "GHOST9": {"corner": "top-left"}},
        thermal_refs=["U1"],
        backside_through_hole_leaves=["BATT"],
        board={"width_mm": 70.0, "height_mm": 50.0, "size_search": False},
    )
    out = write_autoplacer_json(
        tmp_path, "WIDGET", _arch(), bom,
        library_fragments={"component_zones": {"J1": {"edge": "top"}}},
        placement=placement,
    )
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["component_zones"]["J1"] == {"edge": "left"}, "user wins"
    assert "GHOST9" not in cfg["component_zones"], "stale ref dropped"
    assert "U1" in cfg["thermal_refs"]
    assert cfg["parent_placement"]["backside_through_hole_leaves"] == ["BATT"]
    assert cfg["board_width_mm"] == 70.0
    assert cfg["board_height_mm"] == 50.0
    assert cfg["enable_board_size_search"] is False
    assert "GHOST9" in capsys.readouterr().out, "drop is warned"


def test_write_autoplacer_without_placement_keeps_size_search():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        out = write_autoplacer_json(Path(d), "WIDGET", _arch(), _bom())
        cfg = json.loads(out.read_text(encoding="utf-8"))
        assert cfg["enable_board_size_search"] is True
        assert "board_width_mm" not in cfg
