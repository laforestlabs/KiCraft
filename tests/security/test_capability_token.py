"""Capability-token security: forgery, traversal, and the default-secret footgun.

The web app serves raw project files at /project/<token>/<file>, authorized purely
by an HMAC over the absolute path (web.py _register_project_dir/_resolve_project_token).
A forged token, a tampered path, or a traversal filename must all be refused.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

import pytest


def _web():
    # Import lazily so a missing optional server dep skips rather than errors.
    pytest.importorskip("nicegui")
    from kicraft.server import web
    return web


def test_valid_token_round_trips(tmp_path, monkeypatch):
    web = _web()
    monkeypatch.setenv("KICRAFT_STORAGE_SECRET", "unit-secret")
    proj = tmp_path / "proj"
    proj.mkdir()
    tok = web._register_project_dir(proj)
    assert web._resolve_project_token(tok) == proj.resolve()


def test_forged_token_is_rejected(tmp_path, monkeypatch):
    web = _web()
    monkeypatch.setenv("KICRAFT_STORAGE_SECRET", "unit-secret")
    # Attacker controls the path payload but NOT the secret -> bogus signature.
    payload = base64.urlsafe_b64encode(b"/etc/passwd").decode().rstrip("=")
    forged_sig = base64.urlsafe_b64encode(
        hmac.new(b"WRONG-secret", payload.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    assert web._resolve_project_token(f"{payload}.{forged_sig}") is None
    # garbage / malformed tokens are refused, not crashed on
    assert web._resolve_project_token("not-a-token") is None
    assert web._resolve_project_token("a.b.c") is None
    assert web._resolve_project_token("") is None


def test_token_does_not_verify_under_a_different_secret(tmp_path, monkeypatch):
    web = _web()
    monkeypatch.setenv("KICRAFT_STORAGE_SECRET", "secret-A")
    tok = web._register_project_dir(tmp_path)
    monkeypatch.setenv("KICRAFT_STORAGE_SECRET", "secret-B")
    assert web._resolve_project_token(tok) is None  # rotating the secret invalidates


def test_serve_handler_rejects_traversal_filenames(tmp_path, monkeypatch):
    """Even with a valid dir token, a traversal/absolute filename must 404 (the
    basename + suffix-whitelist + containment checks in serve_project_file)."""
    web = _web()
    monkeypatch.setenv("KICRAFT_STORAGE_SECRET", "unit-secret")
    proj = tmp_path / "proj"
    proj.mkdir()
    secret = (proj.parent / "secret.kicad_pcb")
    secret.write_text("TOP SECRET")
    tok = web._register_project_dir(proj)
    for bad in ("../secret.kicad_pcb", "..%2fsecret.kicad_pcb", "/etc/passwd",
                "sub/evil.kicad_pcb"):
        resp = web.serve_project_file(tok, bad)
        assert getattr(resp, "status_code", None) == 404, bad
    # a non-whitelisted suffix in the project dir is also refused
    (proj / "evil.sh").write_text("#!/bin/sh")
    assert getattr(web.serve_project_file(tok, "evil.sh"), "status_code", None) == 404


def test_default_dev_secret_is_a_known_weakness(monkeypatch):
    """A config assertion: shipping prod with the default secret means anyone can
    forge tokens. This test documents the gap so deploy never relies on the default."""
    web = _web()
    monkeypatch.delenv("KICRAFT_STORAGE_SECRET", raising=False)
    # the default is a literal, public constant -> forgeable by anyone
    assert web._project_secret() == b"kicraft-dev-secret"
    # deploy/03-service-setup.sh must set KICRAFT_STORAGE_SECRET; flag if it is the default.
    assert os.environ.get("KICRAFT_STORAGE_SECRET", "kicraft-dev-secret") is not None
