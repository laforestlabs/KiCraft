"""`_lower_project_netclass_clearance`: bring a project's netclass clearances down
to the clearance the board was routed to.

kicad-cli DRC enforces netclass clearances from the ``.kicad_pro``, not the board.
When the fine-pitch rule lowers FreeRouting's clearance globally (e.g. to 0.153 mm
so traces escape a USB-C pad field) but the project keeps a wider class (a 0.30 mm
Power class), the verify gate flags every power trace -> build exit 7. This guards
the helper that aligns the project rule with the routed clearance."""
from __future__ import annotations

import json

from kicraft.design.cli_app import _lower_project_netclass_clearance


def _write_pro(path, classes):
    path.write_text(json.dumps({"net_settings": {"classes": classes}}), encoding="utf-8")


def _clearances(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return {c["name"]: c["clearance"] for c in d["net_settings"]["classes"]}


def test_lowers_wider_classes_only(tmp_path):
    pro = tmp_path / "p.kicad_pro"
    _write_pro(pro, [
        {"name": "Default", "clearance": 0.2},
        {"name": "Power", "clearance": 0.3},
        {"name": "Tight", "clearance": 0.1},  # already below target -> untouched
    ])
    changed = _lower_project_netclass_clearance(pro, 0.153)
    assert changed is True
    assert _clearances(pro) == {"Default": 0.153, "Power": 0.153, "Tight": 0.1}


def test_noop_when_all_classes_already_tight(tmp_path):
    pro = tmp_path / "p.kicad_pro"
    _write_pro(pro, [{"name": "Default", "clearance": 0.12}])
    assert _lower_project_netclass_clearance(pro, 0.153) is False
    assert _clearances(pro) == {"Default": 0.12}


def test_missing_file_is_non_fatal(tmp_path):
    assert _lower_project_netclass_clearance(tmp_path / "nope.kicad_pro", 0.153) is False
