"""Showcase sample-project manifest: asset integrity + the no-LLM-before-signup gate.

The public landing page and the in-app explorer both render the curated samples in
``kicraft.server.samples``. Two things must hold:

  * every advertised sample is actually present on disk (board, preview render, and
    all schematic sheets), so neither surface shows a broken image or a dead viewer;
  * every public call-to-action routes through ``/signup``. The hard product rule is
    that no model runs before a valid email signup, so the showcase must be pure
    static assets and its links must never trigger a design run.
"""
from __future__ import annotations

import re
from urllib.parse import quote

from kicraft.server import samples as S
from kicraft.server import web


def test_samples_present_and_nonempty():
    assert S.SAMPLES, "expected at least one curated sample"
    # 3-5 curated boards (the product brief); structured to grow but never empty.
    assert 3 <= len(S.SAMPLES) <= 5


def test_every_sample_resolves_on_disk():
    """Bundled files exist: root + leaf schematics, board, project, and preview."""
    for s in S.SAMPLES:
        assert s.dir.is_dir(), f"{s.id}: missing bundle dir {s.dir}"
        assert s.board_pcb.is_file(), f"{s.id}: missing {s.board_pcb.name}"
        assert (s.dir / f"{s.stem}.kicad_sch").is_file(), f"{s.id}: missing root sch"
        assert (s.dir / f"{s.stem}.kicad_pro").is_file(), f"{s.id}: missing .kicad_pro"
        assert s.board_png.is_file(), f"{s.id}: missing preview render"
        assert s.board_png.stat().st_size > 1024, f"{s.id}: preview render looks empty"
        assert s.exists()


def test_available_samples_is_all_when_bundled():
    """With the repo checkout intact, every declared sample is available."""
    assert [s.id for s in S.available_samples()] == [s.id for s in S.SAMPLES]


def test_schematic_sources_root_first_and_complete():
    for s in S.SAMPLES:
        files = s.schematic_files()
        assert files, f"{s.id}: no schematic sheets"
        assert files[0].name == f"{s.stem}.kicad_sch", f"{s.id}: root not first"
        assert s.sheets == len(files), f"{s.id}: sheets stat != file count"
        srcs = s.schematic_sources()
        assert len(srcs) == len(files)
        for url, name in srcs:
            assert url == f"/samples/{s.id}/{name}"
            assert name.endswith(".kicad_sch")


def test_board_source_and_preview_urls_under_static_mount():
    for s in S.SAMPLES:
        burl, bname = s.board_source()
        assert bname == f"{s.stem}.kicad_pcb"
        assert burl == f"/samples/{s.id}/{bname}"
        # The preview URL is under the static mount; a ``?v=`` cache-bust may be
        # appended so a regenerated render/GLB defeats the browser cache.
        assert s.board_png_url.split("?")[0] == f"/samples/{s.id}/previews/board.png"
        assert re.fullmatch(r"(\?v=\d+)?", s.board_png_url.split("/previews/board.png")[1])


def test_metadata_is_sane():
    ids = [s.id for s in S.SAMPLES]
    assert len(ids) == len(set(ids)), "sample ids must be unique"
    for s in S.SAMPLES:
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", s.id), f"bad slug: {s.id}"
        assert s.title.strip() and s.blurb.strip() and s.prompt.strip()
        assert s.sheets > 0 and s.parts > 0


def test_exactly_one_featured_sample():
    featured = [s for s in S.SAMPLES if s.featured]
    assert len(featured) == 1, "exactly one sample should lead the hero"
    assert S.featured_sample() is featured[0]


def test_landing_cards_route_through_signup_only():
    """The gate: a public sample card links to /signup with the brief prefilled, and
    never to a route that could start a design (no /?prompt=, no run trigger)."""
    for s in S.SAMPLES:
        card = web._landing_sample_card(s)
        assert f'href="/signup?prompt={quote(s.prompt)}"' in card
        assert s.board_png_url in card
        # No public path may deep-link into the workspace (which can spend tokens).
        assert "/?prompt=" not in card
        assert "href=\"/\"" not in card
