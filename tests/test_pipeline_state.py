"""Tests for the GUI whole-pipeline state reader (kicraft.gui.pipeline_state).

Pure-Python: imports the reader without pulling in NiceGUI.
"""
import json
from pathlib import Path

from kicraft.gui.pipeline_state import find_synth_project, pipeline_progress


def _write_state(root: Path, **slots) -> None:
    d = root / ".kicraft"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(slots), encoding="utf-8")


def _stages(root: Path) -> dict:
    return {s.key: s for s in pipeline_progress(root)}


def test_empty_project_makes_intent_active(tmp_path):
    s = _stages(tmp_path)
    assert s["intent"].state == "active"  # nothing done -> first stage is active
    assert s["bom"].state == "pending"
    assert s["fab"].state == "pending"


def test_upstream_progress_marks_next_active(tmp_path):
    _write_state(tmp_path, intent={"goal": "x"}, functional_spec={"blocks": []})
    s = _stages(tmp_path)
    assert s["intent"].state == "done"
    assert s["functional_spec"].state == "done"
    assert s["architecture"].state == "active"  # first pending
    assert s["wiring"].state == "pending"


def test_bom_parts_vs_wiring_connections(tmp_path):
    _write_state(
        tmp_path,
        intent={"goal": "x"},
        functional_spec={"b": 1},
        architecture={"sheets": []},
        bom={"parts": [{"ref": "U1"}], "connections": []},
    )
    s = _stages(tmp_path)
    assert s["bom"].state == "done"  # has parts
    assert s["wiring"].state == "active"  # parts but no connections yet


def _full_upstream(root: Path) -> None:
    _write_state(
        root,
        intent={"g": 1},
        functional_spec={"b": 1},
        architecture={"s": 1},
        bom={"parts": [{"ref": "U1"}], "connections": [{"net": "GND"}]},
    )


def test_synth_route_fab_in_generated_layout(tmp_path):
    _full_upstream(tmp_path)
    proj = tmp_path / "generated" / "DEMO"
    proj.mkdir(parents=True)
    (proj / "DEMO.kicad_pro").write_text("{}", encoding="utf-8")

    assert find_synth_project(tmp_path) == proj
    s = _stages(tmp_path)
    assert s["synthesize"].state == "done"
    assert s["route"].state == "active"  # synth done, route is next-up

    exp = proj / ".experiments"
    exp.mkdir()
    (exp / "run_status.json").write_text(
        json.dumps({"phase": "running", "progress_percent": 42}), encoding="utf-8"
    )
    s = _stages(tmp_path)
    assert s["route"].state == "active"
    assert "42" in s["route"].detail

    fab = proj / "fab"
    fab.mkdir()
    (fab / "DEMO-F_Cu.gtl").write_text("x", encoding="utf-8")
    s = _stages(tmp_path)
    assert s["fab"].state == "done"
    assert "1 layers" in s["fab"].detail


def test_synth_project_at_root_layout(tmp_path):
    """Synthesized dir == project root (no generated/ nesting)."""
    _full_upstream(tmp_path)
    (tmp_path / "DEMO.kicad_pro").write_text("{}", encoding="utf-8")
    assert find_synth_project(tmp_path) == tmp_path
    assert _stages(tmp_path)["synthesize"].state == "done"
