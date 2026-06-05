"""Web app: the render-PNG endpoint and per-leaf layout-progress discovery.

These back the Place/Route progress gallery (leaf placement previews shown before
routing) and the Synthesize sheet selector. They are pure functions / a plain
endpoint handler, tested directly like the other ``kicraft.server`` tests rather
than through an HTTP client."""
from __future__ import annotations

import os
from pathlib import Path

from starlette.responses import FileResponse

from kicraft.server import web


def _touch(path: Path, *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _leaf(project: Path, uuid: str, sheet_name: str) -> Path:
    d = project / ".experiments" / "subcircuits" / uuid
    (d / "renders").mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        '{"sheet_name": "%s", "internal_trace_count": 7, "internal_via_count": 1}'
        % sheet_name,
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------- render endpoint

def test_render_endpoint_serves_png(tmp_path):
    token = web._register_project_dir(tmp_path)
    rel = ".experiments/subcircuits/aa__bb/renders/pre_route_front_all.png"
    _touch(tmp_path / rel)
    resp = web.serve_project_render(token, rel)
    assert isinstance(resp, FileResponse)
    assert resp.media_type == "image/png"


def test_render_endpoint_rejects_non_png(tmp_path):
    token = web._register_project_dir(tmp_path)
    (tmp_path / "secret.kicad_sch").write_text("x", encoding="utf-8")
    assert web.serve_project_render(token, "secret.kicad_sch").status_code == 404


def test_render_endpoint_rejects_missing_and_bad_token(tmp_path):
    token = web._register_project_dir(tmp_path)
    assert web.serve_project_render(token, "nope/x.png").status_code == 404
    assert web.serve_project_render("not-a-token", "x.png").status_code == 404


def test_render_endpoint_blocks_traversal(tmp_path):
    # A png that exists OUTSIDE the project dir must not be reachable via `..`.
    outside = _touch(tmp_path.parent / "outside.png")
    token = web._register_project_dir(tmp_path / "project")
    (tmp_path / "project").mkdir()
    resp = web.serve_project_render(token, f"../{outside.name}")
    assert resp.status_code == 404


# ------------------------------------------------------------ _latest_render

def test_latest_render_prefers_newest_round(tmp_path):
    r = tmp_path / "renders"
    _touch(r / "pre_route_front_all.png", mtime=1000)
    _touch(r / "round_0000_pre_route_front_all.png", mtime=1001)
    newest = _touch(r / "round_0001_pre_route_front_all.png", mtime=1002)
    assert web._latest_render(r, "pre_route_front_all") == newest
    assert web._latest_render(r, "routed_front_all") is None


# ------------------------------------------------------- _leaf_layout_progress

def test_leaf_progress_placed_then_routed(tmp_path):
    token = web._register_project_dir(tmp_path)
    a = _leaf(tmp_path, "aaaa__1", "USB INPUT")
    b = _leaf(tmp_path, "bbbb__2", "MCU")
    # leaf A: only placement render -> "Placed"; leaf B: placement + routed -> "Routed".
    _touch(a / "renders" / "pre_route_front_all.png")
    _touch(b / "renders" / "pre_route_front_all.png")
    _touch(b / "renders" / "routed_front_all.png")

    prog = web._leaf_layout_progress(tmp_path, token)
    by_name = {d["sheet_name"]: d for d in prog}
    assert set(by_name) == {"USB INPUT", "MCU"}
    assert by_name["USB INPUT"]["status"] == "Placed"
    assert by_name["MCU"]["status"] == "Routed"
    # placement-only leaf shows its pre-route image; URL is token-scoped + cache-busted.
    assert by_name["USB INPUT"]["url"].startswith(
        f"/project/{token}/render/.experiments/subcircuits/aaaa__1/renders/"
        "pre_route_front_all.png?v="
    )
    assert by_name["MCU"]["traces"] == 7 and by_name["MCU"]["vias"] == 1
    # sorted by sheet name (MCU before USB INPUT).
    assert [d["sheet_name"] for d in prog] == ["MCU", "USB INPUT"]


def test_leaf_progress_empty_when_no_experiments(tmp_path):
    token = web._register_project_dir(tmp_path)
    assert web._leaf_layout_progress(tmp_path, token) == []


# ------------------------------------------------- _schematic_sources (selector feed)

def test_schematic_sources_root_first(tmp_path):
    for n in ("MCU.kicad_sch", "DEMO.kicad_sch", "USB_INPUT.kicad_sch"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    srcs = web._schematic_sources(tmp_path, "DEMO", "tok")
    names = [f for _u, f in srcs]
    assert names[0] == "DEMO.kicad_sch"  # root sheet first (selector's "Overview")
    assert set(names) == {"DEMO.kicad_sch", "MCU.kicad_sch", "USB_INPUT.kicad_sch"}
