"""Tests for kicraft.server.mailer: message building and the fail-soft SMTP send.

Pure stdlib. The configured-send paths are exercised with a fake SMTP injected via
monkeypatch, so no socket or mail server is needed.
"""
from __future__ import annotations

import io
import json
import urllib.error

from kicraft.server import mailer
from kicraft.server.config import Settings


def _settings(**kw) -> Settings:
    """A Settings with just the bits the mailer reads (api_key is the only required
    field; everything else defaults, e.g. smtp_host='' = unconfigured)."""
    return Settings(api_key="x", **kw)


def _plain(msg) -> str:
    return msg.get_body(preferencelist=("plain",)).get_content()


def _html(msg) -> str:
    return msg.get_body(preferencelist=("html",)).get_content()


def test_build_reset_email_has_link_and_headers():
    msg = mailer.build_reset_email("user@e.st", "no-reply@kicraft.io",
                                   "https://kicraft.io/reset?token=ABC", ttl_minutes=60)
    assert msg["To"] == "user@e.st"
    assert msg["From"] == "no-reply@kicraft.io"
    assert "password" in msg["Subject"].lower()
    # Multipart/alternative: the link and TTL live in both the text and HTML parts,
    # and the HTML carries a real clickable CTA button (an <a href> to the link).
    body = _plain(msg)
    assert "https://kicraft.io/reset?token=ABC" in body
    assert "60 minutes" in body
    html_body = _html(msg)
    assert 'href="https://kicraft.io/reset?token=ABC"' in html_body
    assert "KiCraft" in html_body and "Choose a new password" in html_body


def test_build_verification_email_has_button_and_link():
    msg = mailer.build_verification_email("user@e.st", "no-reply@kicraft.io",
                                          "https://kicraft.io/verify?token=XYZ",
                                          ttl_hours=24)
    assert "email" in msg["Subject"].lower()
    body = _plain(msg)
    assert "https://kicraft.io/verify?token=XYZ" in body
    assert "24 hours" in body
    html_body = _html(msg)
    assert 'href="https://kicraft.io/verify?token=XYZ"' in html_body
    assert "Verify email address" in html_body


def test_from_addr_adds_display_name_to_bare_address():
    # A bare mailbox is wrapped so the inbox shows "KiCraft", not a raw address.
    assert mailer._from_addr(_settings(email_from="no-reply@kicraft.io")) \
        == "KiCraft <no-reply@kicraft.io>"


def test_from_addr_preserves_existing_display_name():
    # An operator-supplied display name is passed through untouched.
    assert mailer._from_addr(_settings(email_from="Team <hi@kicraft.io>")) \
        == "Team <hi@kicraft.io>"


def test_send_returns_false_when_unconfigured():
    s = _settings()  # smtp_host == "" by default
    assert mailer.send_email(s, "user@e.st", "Hi", "body") is False
    assert mailer.send_reset_email(s, "user@e.st", "https://x/reset?token=ABC") is False


def test_send_uses_starttls_and_login(monkeypatch):
    sent: dict = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=0):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            sent["starttls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["to"] = msg["To"]

    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    s = _settings(smtp_host="smtp.example.com", smtp_port=587,
                  smtp_username="bot@kicraft.io", smtp_password="secret",
                  smtp_from="no-reply@kicraft.io", smtp_starttls=True)
    assert mailer.send_reset_email(s, "user@e.st",
                                   "https://kicraft.io/reset?token=ABC") is True
    assert sent["starttls"] is True
    assert sent["login"] == ("bot@kicraft.io", "secret")
    assert sent["to"] == "user@e.st"
    assert (sent["host"], sent["port"]) == ("smtp.example.com", 587)


def test_send_ssl_path_skips_starttls(monkeypatch):
    sent: dict = {}

    class FakeSMTPSSL:
        def __init__(self, host, port, timeout=0, context=None):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["sent"] = True

    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTPSSL)
    s = _settings(smtp_host="smtp.example.com", smtp_port=465, smtp_ssl=True,
                  smtp_username="bot@kicraft.io", smtp_password="secret")
    assert mailer.send_reset_email(s, "user@e.st", "https://x/reset?token=ABC") is True
    assert sent["sent"] is True


def test_send_returns_false_on_smtp_error(monkeypatch):
    class BoomSMTP:
        def __init__(self, *a, **k):
            raise OSError("connection refused")

    monkeypatch.setattr(mailer.smtplib, "SMTP", BoomSMTP)
    s = _settings(smtp_host="smtp.example.com", smtp_port=587)
    assert mailer.send_email(s, "user@e.st", "Hi", "body") is False  # caught, not raised


# ---- Resend backend -------------------------------------------------------

def test_send_via_resend_posts_to_api(monkeypatch):
    captured: dict = {}

    class FakeResp:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr(mailer.urllib.request, "urlopen", fake_urlopen)
    s = _settings(resend_api_key="re_test123", email_from="no-reply@kicraft.io")
    ok = mailer.send_reset_email(s, "user@e.st", "https://kicraft.io/reset?token=ABC")
    assert ok is True
    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["headers"]["authorization"] == "Bearer re_test123"
    # A real UA is required or Cloudflare 403s the default urllib signature (1010).
    assert captured["headers"]["user-agent"] == "KiCraft/1.0 (+https://kicraft.io)"
    assert "urllib" not in captured["headers"]["user-agent"].lower()
    # A bare configured address is wrapped with a friendly display name.
    assert captured["body"]["from"] == "KiCraft <no-reply@kicraft.io>"
    assert captured["body"]["to"] == ["user@e.st"]
    assert "kicraft.io/reset?token=ABC" in captured["body"]["text"]
    # Resend gets the branded HTML alternative, not just plain text.
    assert 'href="https://kicraft.io/reset?token=ABC"' in captured["body"]["html"]


def test_resend_takes_precedence_over_smtp(monkeypatch):
    used: dict = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        used["resend"] = True
        return FakeResp()

    def boom(*a, **k):
        raise AssertionError("SMTP must not be used when Resend is configured")

    monkeypatch.setattr(mailer.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mailer.smtplib, "SMTP", boom)
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", boom)
    s = _settings(resend_api_key="re_x", smtp_host="smtp.example.com",
                  email_from="no-reply@kicraft.io")
    assert mailer.send_email(s, "u@e.st", "Hi", "body") is True
    assert used.get("resend") is True


def test_resend_http_error_returns_false(monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url, 422, "Unprocessable Entity", {},
            io.BytesIO(b'{"message":"domain not verified"}'))

    monkeypatch.setattr(mailer.urllib.request, "urlopen", fake_urlopen)
    s = _settings(resend_api_key="re_x", email_from="no-reply@kicraft.io")
    assert mailer.send_reset_email(s, "u@e.st", "https://x/reset?token=A") is False
