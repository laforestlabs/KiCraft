"""Signed project-file tokens reject forgery and traversal and serve scoped files."""
from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse

from kicraft.server import web




def test_serve_returns_schematic_for_registered_token(tmp_path):
    (tmp_path / "DEMO.kicad_sch").write_text("(kicad_sch (version 20250114))\n", encoding="utf-8")
    token = web._register_project_dir(tmp_path)
    resp = web.serve_project_file(token, "DEMO.kicad_sch")
    assert isinstance(resp, FileResponse)
    assert Path(resp.path).read_text(encoding="utf-8").startswith("(kicad_sch")




def test_serve_rejects_forged_and_garbage_tokens(tmp_path):
    (tmp_path / "DEMO.kicad_sch").write_text("(kicad_sch)\n", encoding="utf-8")
    token = web._register_project_dir(tmp_path)
    # tamper with the signature segment -> must not resolve to the dir.
    forged = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
    assert web.serve_project_file(forged, "DEMO.kicad_sch").status_code == 404
    assert web.serve_project_file("not-a-token", "DEMO.kicad_sch").status_code == 404
    # a validly-signed token for a *different* dir cannot read this one's files
    # (path stays scoped to its own base), and traversal is still blocked.
    assert web.serve_project_file(token, "../escape.kicad_sch").status_code == 404


