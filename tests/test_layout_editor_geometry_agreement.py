"""Cross-language agreement: canvas JS geometry vs Python OutlineSpec.

The canvas (layout_canvas.js, window.kicraftLayoutGeometry) and the
composer/validator (kicraft.layout_editor.outline) each implement the
shape geometry; if they drift, the editor shows green while the stamp
fails (or vice versa). This test drives BOTH implementations over the
same fixtures and requires near-bit-identical answers: polyline
vertices, containment verdicts, and mounting-hole peg positions.

Skipped when node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from kicraft.autoplacer.brain.types import Point
from kicraft.layout_editor.canvas import static_js_path
from kicraft.layout_editor.outline import OutlineSpec

_NODE = shutil.which("node")

_HARNESS = """
global.window = {};
const fs = require('fs');
eval(fs.readFileSync(process.argv[2], 'utf8'));
const fixtures = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const g = window.kicraftLayoutGeometry;
const out = fixtures.map(fx => ({
  polyline: g.outlinePolyline(fx.spec, fx.min, fx.max).map(p => [p.x, p.y]),
  contains: fx.probes.map(p => g.outlineContainsPoint(
    fx.spec, fx.min, fx.max, p[0], p[1], fx.tol)),
  holes: fx.holes.map(h => {
    const pos = g.mountingHolePosition(fx.spec, fx.min, fx.max, h[0], h[1]);
    return pos ? [pos.x, pos.y] : null;
  }),
}));
process.stdout.write(JSON.stringify(out));
"""

_CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def _fixtures() -> list[dict]:
    probes = [
        [0.5, 0.5], [30.0, 20.0], [59.5, 39.5], [1.0, 39.0],
        [60.04, 20.0], [0.0, 0.0], [50.0, 5.0],
    ]
    holes = [[c, 5.0] for c in _CORNERS] + [["top-left", 0.0]]
    out = []
    for spec in (
        {"shape": "rect", "corner_radius_mm": 0.0, "chamfer_mm": 0.0},
        {"shape": "rounded_rect", "corner_radius_mm": 4.0, "chamfer_mm": 0.0},
        {"shape": "chamfered_rect", "corner_radius_mm": 0.0, "chamfer_mm": 5.0},
        {"shape": "circle", "corner_radius_mm": 0.0, "chamfer_mm": 0.0},
    ):
        h = 60.0 if spec["shape"] == "circle" else 40.0
        out.append(
            {
                "spec": spec,
                "min": {"x": 0.0, "y": 0.0},
                "max": {"x": 60.0, "y": h},
                "tol": 0.01,
                "probes": probes,
                "holes": holes,
            }
        )
    return out


def _python_results(fixtures: list[dict]) -> list[dict]:
    results = []
    for fx in fixtures:
        spec = OutlineSpec(
            shape=fx["spec"]["shape"],
            min_pt=Point(fx["min"]["x"], fx["min"]["y"]),
            max_pt=Point(fx["max"]["x"], fx["max"]["y"]),
            corner_radius_mm=fx["spec"]["corner_radius_mm"],
            chamfer_mm=fx["spec"]["chamfer_mm"],
        )
        results.append(
            {
                "polyline": [[p.x, p.y] for p in spec.polyline()],
                "contains": [
                    spec.contains_point(px, py, tol=fx["tol"])
                    for px, py in fx["probes"]
                ],
                "holes": [
                    [pos.x, pos.y]
                    for pos in (
                        spec.mounting_hole_position(corner, inset)
                        for corner, inset in fx["holes"]
                    )
                ],
            }
        )
    return results


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_js_and_python_geometry_agree(tmp_path):
    fixtures = _fixtures()
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures), encoding="utf-8")

    proc = subprocess.run(
        [_NODE, str(harness), str(static_js_path()), str(fixtures_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"node harness failed:\n{proc.stderr}"
    js = json.loads(proc.stdout)
    py = _python_results(fixtures)

    assert len(js) == len(py)
    for fx, j, p in zip(fixtures, js, py):
        shape = fx["spec"]["shape"]
        assert len(j["polyline"]) == len(p["polyline"]), shape
        for (jx, jy), (px, py_) in zip(j["polyline"], p["polyline"]):
            assert jx == pytest.approx(px, abs=1e-9), shape
            assert jy == pytest.approx(py_, abs=1e-9), shape
        assert j["contains"] == p["contains"], (
            f"{shape}: containment verdicts diverge: js={j['contains']} "
            f"py={p['contains']} probes={fx['probes']}"
        )
        for jh, ph in zip(j["holes"], p["holes"]):
            assert jh is not None and ph is not None, shape
            assert jh[0] == pytest.approx(ph[0], abs=1e-9), shape
            assert jh[1] == pytest.approx(ph[1], abs=1e-9), shape
