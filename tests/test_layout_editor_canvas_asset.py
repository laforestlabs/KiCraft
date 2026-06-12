"""Guards for the shared layout-canvas static asset + bootstrap.

The interactive canvas controller was extracted from a Python f-string
template into ``kicraft/layout_editor/static/layout_canvas.js``. A
single mangled brace during that kind of extraction kills the whole
controller silently (the browser logs a SyntaxError, the canvas just
never paints), so these tests pin:

- the asset parses as JavaScript (via ``node --check`` when node is
  available);
- the global entry point and every Python-called API method exist by
  name;
- the bootstrap embeds the canvas config and is itself valid JS.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from kicraft.layout_editor.canvas import (
    build_canvas_html,
    build_canvas_init_script,
    static_js_path,
)
from kicraft.layout_editor.leaves import LeafInfo

_NODE = shutil.which("node")


def _node_check(source: str, tmp_path) -> None:
    js = tmp_path / "check.js"
    js.write_text(source, encoding="utf-8")
    proc = subprocess.run(
        [_NODE, "--check", str(js)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"JS syntax error:\n{proc.stderr}"


def _leaf(instance_path: str = "/battery") -> LeafInfo:
    return LeafInfo(
        instance_path=instance_path,
        sheet_name="BATT",
        width_mm=20.0,
        height_mm=10.0,
        artifact_dir=static_js_path().parent,  # any existing dir
        render_url="/experiments/subcircuits/x/renders/leaf_canvas.png?v=1",
        image_width_mm=20.0,
        image_height_mm=10.0,
        edge_max_x=20.0,
        edge_max_y=10.0,
    )


def _initial() -> dict:
    return {
        "placements": [
            {"instance_path": "/battery", "origin": {"x": 1.0, "y": 2.0}, "rotation": 0.0}
        ],
        "board_outline": {"min": {"x": 0.0, "y": 0.0}, "max": {"x": 80.0, "y": 60.0}},
        "mounting_holes": [],
    }


def test_asset_exists_and_has_no_template_leftovers():
    src = static_js_path().read_text(encoding="utf-8")
    assert "window.kicraftInitLayoutCanvas = function(cfg)" in src
    # Leftovers from the old .format() template would mean the
    # extraction regressed (or someone pasted template text back in).
    assert "{config_json}" not in src
    assert "__KICRAFT_CFG_PLACEHOLDER__" not in src


def test_asset_keeps_python_called_api_surface():
    """Python (offline GUI today, web panel next) calls these by name
    via ui.run_javascript; renaming any of them in the JS without
    updating the hosts breaks silently."""
    src = static_js_path().read_text(encoding="utf-8")
    assert "window.manualLayoutCanvases" in src
    for method in (
        "getState",
        "reset",
        "getOutlineSize",
        "setOutlineSize",
        "setMountingHoles",
        "getMountingHoles",
        "setViewOptions",
    ):
        assert f"{method}: function" in src, f"missing canvas API method {method}"


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_asset_is_valid_javascript(tmp_path):
    _node_check(static_js_path().read_text(encoding="utf-8"), tmp_path)


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_bootstrap_is_valid_javascript(tmp_path):
    script = build_canvas_init_script([_leaf()], _initial(), "test-canvas")
    _node_check(script, tmp_path)


def test_bootstrap_embeds_config_and_calls_entry_point():
    script = build_canvas_init_script(
        [_leaf()], _initial(), "test-canvas", asset_url="/layout-static/layout_canvas.js?v=7"
    )
    assert "window.kicraftInitLayoutCanvas(cfg)" in script
    # The embedded config must round-trip: find the JSON object literal
    # assigned to cfg and parse it.
    start = script.index("const cfg = ") + len("const cfg = ")
    end = script.index(";\n", start)
    cfg = json.loads(script[start:end])
    assert cfg["canvas_id"] == "test-canvas"
    assert cfg["asset_url"] == "/layout-static/layout_canvas.js?v=7"
    assert cfg["leaves"][0]["instance_path"] == "/battery"
    assert cfg["leaves"][0]["edge_max_x"] == 20.0
    assert cfg["initial"]["board_outline"]["max"]["x"] == 80.0


def test_canvas_html_contains_host_and_svg():
    html = build_canvas_html([_leaf()], _initial(), "my-canvas")
    assert 'id="my-canvas-host"' in html
    assert 'id="my-canvas"' in html
    assert "<script" not in html  # NiceGUI ui.html() rejects script tags
