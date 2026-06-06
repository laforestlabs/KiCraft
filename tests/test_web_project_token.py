"""Project-file serving must survive a server restart.

Regression test for the Synthesize "teal blob": the schematic viewer (KiCanvas)
fetches each sheet from ``/project/{token}/<sheet>.kicad_sch``. When that fetch
404s, KiCanvas throws ``Unable to load ... 404``, aborts before painting, and the
``<kicanvas-embed>`` shows its ``:host{background:aqua}`` fallback (pure cyan).

The token used to be an in-memory dict entry, so a ``kicraft-web`` restart (a
deploy) wiped every token and open tabs 404'd on every schematic fetch while the
server-side sheet *list* (a glob) still rendered -- exactly the reported blob.

These tests pin the fix: tokens are stateless + signed, so a token minted before a
restart still resolves afterwards, and a forged token is rejected. Tested directly
like the other ``kicraft.server`` tests (no HTTP client)."""
from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse

from kicraft.server import web


def _drop_in_memory_token_state() -> None:
    """Simulate a process restart: discard any in-memory token bookkeeping.

    The fix makes tokens stateless, so there may be nothing to clear; this stays
    robust whether or not a legacy map still exists."""
    m = getattr(web, "_PROJECT_TOKENS", None)
    if isinstance(m, dict):
        m.clear()


def test_serve_returns_schematic_for_registered_token(tmp_path):
    (tmp_path / "DEMO.kicad_sch").write_text("(kicad_sch (version 20250114))\n", encoding="utf-8")
    token = web._register_project_dir(tmp_path)
    resp = web.serve_project_file(token, "DEMO.kicad_sch")
    assert isinstance(resp, FileResponse)
    assert Path(resp.path).read_text(encoding="utf-8").startswith("(kicad_sch")


def test_token_survives_restart(tmp_path):
    """A token minted before a restart must still serve the file afterwards."""
    (tmp_path / "DEMO.kicad_sch").write_text("(kicad_sch)\n", encoding="utf-8")
    token = web._register_project_dir(tmp_path)
    _drop_in_memory_token_state()  # the restart
    resp = web.serve_project_file(token, "DEMO.kicad_sch")
    assert isinstance(resp, FileResponse), "schematic 404'd after restart -> teal blob"


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


def test_render_endpoint_survives_restart(tmp_path):
    rel = ".experiments/subcircuits/aa__bb/renders/pre_route_front_all.png"
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n")
    token = web._register_project_dir(tmp_path)
    _drop_in_memory_token_state()
    resp = web.serve_project_render(token, rel)
    assert isinstance(resp, FileResponse)
    assert web.serve_project_render("not-a-token", rel).status_code == 404
