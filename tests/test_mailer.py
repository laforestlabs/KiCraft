"""Tests for kicraft.server.mailer: message building and the fail-soft SMTP send.

Pure stdlib. The configured-send paths are exercised with a fake SMTP injected via
monkeypatch, so no socket or mail server is needed.
"""
from __future__ import annotations

from kicraft.server import mailer
from kicraft.server.config import Settings


def _settings(**kw) -> Settings:
    """A Settings with just the bits the mailer reads (api_key is the only required
    field; everything else defaults, e.g. smtp_host='' = unconfigured)."""
    return Settings(api_key="x", **kw)


def test_build_reset_email_has_link_and_headers():
    msg = mailer.build_reset_email("user@e.st", "no-reply@kicraft.io",
                                   "https://kicraft.io/reset?token=ABC", ttl_minutes=60)
    assert msg["To"] == "user@e.st"
    assert msg["From"] == "no-reply@kicraft.io"
    assert "password" in msg["Subject"].lower()
    body = msg.get_content()
    assert "https://kicraft.io/reset?token=ABC" in body
    assert "60 minutes" in body


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
