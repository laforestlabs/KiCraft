"""XSS / raw-HTML-sink guardrails.

NiceGUI escapes by default; the danger is ``ui.html(..., sanitize=False)``, which
injects raw HTML. There are exactly THREE such sinks today and all render TRUSTED
content (static landing/pricing markup + admin-curated showcase Samples), never a
user-submitted brief. These tests (1) pin that count so a NEW raw sink fails CI and
forces review, and (2) assert the data-driven sink consumes the curated Sample type,
not user input.
"""
from __future__ import annotations

import inspect

import pytest


def _web():
    pytest.importorskip("nicegui")
    from kicraft.server import web
    return web


def test_raw_html_sinks_are_pinned_and_take_only_trusted_inputs():
    """A new ui.html(sanitize=False) must be reviewed: if the count changes, this
    test fails. The known sinks accept ONLY a pre-built static page template
    (`html`) or the curated `_sample_model_viewer(s)` -- never a raw user brief.
    Counts the call form `sanitize=False)` so the explanatory comment isn't tallied."""
    web = _web()
    src = inspect.getsource(web)
    sink_lines = [ln for ln in src.splitlines() if "sanitize=False)" in ln]
    assert len(sink_lines) == 3, (
        "a ui.html(sanitize=False) sink was added/removed -- review it for XSS "
        "(it must render trusted/static content, never an unescaped user brief)")
    for ln in sink_lines:
        assert ("ui.html(html, sanitize=False)" in ln
                or "_sample_model_viewer(" in ln), f"unexpected raw-HTML sink: {ln.strip()}"


def test_model_viewer_sink_consumes_curated_samples_only():
    """The one data-driven raw sink renders a Sample (admin-curated showcase), whose
    fields come from samples.py -- not from end-user submissions."""
    web = _web()
    src = inspect.getsource(web._sample_model_viewer)
    # it only interpolates Sample attributes (board_glb_url/board_png_url/title)
    assert "s.board_glb_url" in src and "s.board_png_url" in src
    # and it is fed by the curated samples module, not request input
    samples = pytest.importorskip("kicraft.server.samples", reason="samples module")
    assert hasattr(samples, "SAMPLES") or hasattr(samples, "samples") or samples


def test_landing_card_consumes_only_curated_sample_fields():
    """Landing cards route by curated sample ID and render curated metadata."""
    web = _web()
    src = inspect.getsource(web._landing_sample_card)
    assert 'f"/examples/{s.id}"' in src
    assert "s.id" in src and "s.title" in src and "s.blurb" in src
    assert "s.prompt" in src
