"""Showcase sample-project manifest: asset integrity and public detail routes.

The public landing page and the in-app explorer both render the curated samples in
``kicraft.server.samples``. The landing cards open a public detail page containing
the static 3D model, schematic, and routed board; only the action to build a new
design remains signup-gated.
"""
from __future__ import annotations

import inspect
import re

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


def test_landing_cards_open_public_example_pages():
    """Landing cards must open static public example pages, not start a build."""
    for s in S.SAMPLES:
        card = web._landing_sample_card(s)
        assert f'href="/examples/{s.id}"' in card
        assert s.board_png_url in card
        assert "/signup?prompt=" not in card
        assert "/?prompt=" not in card
        assert 'href="/"' not in card


def test_public_example_detail_page_is_3d_and_not_login_gated():
    """Each public detail page exposes the curated 3D viewer without auth."""
    routes = [getattr(route, "path", "") for route in web.app.routes]
    assert "/examples/{sample_id}" in routes
    src = inspect.getsource(web.sample_detail_page)
    assert "_current_user" not in src
    assert "_render_sample_3d(sample)" in src
    assert "/signup?prompt=" in src


def test_landing_card_keeps_prompt_as_display_only():
    """The card displays the brief but leaves build navigation to the detail CTA."""
    for s in S.SAMPLES:
        card = web._landing_sample_card(s)
        assert s.prompt in card
        assert "/signup?prompt=" not in card
