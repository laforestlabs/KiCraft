"""Outbound email for the KiCraft web app, used to deliver password-reset links.

No third-party dependency: SMTP goes through stdlib smtplib, and the Resend
backend posts JSON with stdlib urllib. The mailer is tiny and fail-soft: with no
backend configured it logs the message that would have been sent and returns
False, so local dev and the test suite exercise the full reset flow without a mail
server. In production the box configures one backend.

Two backends, selected by config (Resend wins if both are set):
  - Resend HTTP API  -> settings.resend_api_key
  - SMTP             -> settings.smtp_host (+ the other smtp_* fields)

The send functions never raise on mail trouble; they return a bool. Callers
(/forgot) show the same neutral message either way, so a delivery failure does not
leak whether an account exists.
"""
from __future__ import annotations

import json
import logging
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage

from .config import Settings

log = logging.getLogger("kicraft.mailer")

_RESET_SUBJECT = "Reset your KiCraft password"
_VERIFY_SUBJECT = "Confirm your KiCraft email"
_RESEND_ENDPOINT = "https://api.resend.com/emails"
# A descriptive User-Agent is required: the Resend API is fronted by Cloudflare,
# which 403s the stdlib default "Python-urllib/x.y" signature (Cloudflare error
# 1010, "banned browser signature"). Any real UA gets through.
_USER_AGENT = "KiCraft/1.0 (+https://kicraft.io)"


def _from_addr(settings: Settings) -> str:
    """The sender address, provider-agnostic. email_from is the canonical setting;
    fall back to the SMTP sender/login for back-compat."""
    return settings.email_from or settings.smtp_from or settings.smtp_username


def build_reset_email(to_addr: str, from_addr: str, reset_url: str,
                      ttl_minutes: int = 60) -> EmailMessage:
    """Compose the password-reset email. Pure (no I/O), so it is unit-testable."""
    msg = EmailMessage()
    msg["Subject"] = _RESET_SUBJECT
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(
        "We received a request to reset your KiCraft password.\n\n"
        f"Open this link to choose a new one (valid for {ttl_minutes} minutes):\n\n"
        f"    {reset_url}\n\n"
        "Resetting your password also signs out every other device, so if someone "
        "else had access to your account they will be logged out.\n\n"
        "If you did not request this you can ignore this email; your password will "
        "not change.\n"
    )
    return msg


def send_reset_email(settings: Settings, to_addr: str, reset_url: str,
                     ttl_minutes: int = 60) -> bool:
    """Compose and send the reset email. See `send_email` for the return contract."""
    msg = build_reset_email(to_addr, _from_addr(settings), reset_url, ttl_minutes)
    return _send(settings, msg)


def build_verification_email(to_addr: str, from_addr: str, verify_url: str,
                             ttl_hours: int = 24) -> EmailMessage:
    """Compose the signup email-verification email. Pure (no I/O), unit-testable."""
    msg = EmailMessage()
    msg["Subject"] = _VERIFY_SUBJECT
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(
        "Welcome to KiCraft! Confirm your email to start designing boards.\n\n"
        f"Open this link to verify your address (valid for {ttl_hours} hours):\n\n"
        f"    {verify_url}\n\n"
        "You can browse example boards while unverified, but the Design button "
        "stays disabled until you confirm.\n\n"
        "If you did not create a KiCraft account you can ignore this email.\n"
    )
    return msg


def send_verification_email(settings: Settings, to_addr: str, verify_url: str,
                            ttl_hours: int = 24) -> bool:
    """Compose and send the signup verification email. See `send_email` for the
    return contract."""
    msg = build_verification_email(to_addr, _from_addr(settings), verify_url,
                                   ttl_hours)
    return _send(settings, msg)


def send_email(settings: Settings, to_addr: str, subject: str, body: str) -> bool:
    """Send one plain-text email. Returns True on success.

    Returns False (and logs) when no backend is configured or sending fails, so the
    caller never crashes a request over mail trouble."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = _from_addr(settings)
    msg["To"] = to_addr
    msg.set_content(body)
    return _send(settings, msg)


def _send(settings: Settings, msg: EmailMessage) -> bool:
    if settings.resend_api_key:
        return _send_via_resend(settings, msg)
    if settings.smtp_host:
        return _send_via_smtp(settings, msg)
    # Dev / unconfigured path: no backend, so surface the message (which for a
    # reset contains the link) in the log for the operator to relay.
    log.warning("no email backend configured (set KICRAFT_RESEND_API_KEY or "
                "KICRAFT_SMTP_HOST); email to %s not sent. Message follows:\n%s",
                msg["To"], msg.get_content())
    return False


def _send_via_resend(settings: Settings, msg: EmailMessage) -> bool:
    """POST the message to the Resend HTTP API (stdlib urllib, no SDK dependency)."""
    payload = json.dumps({
        "from": msg["From"],
        "to": [msg["To"]],
        "subject": msg["Subject"],
        "text": msg.get_content(),
    }).encode("utf-8")
    req = urllib.request.Request(
        _RESEND_ENDPOINT, data=payload, method="POST",
        headers={"Authorization": f"Bearer {settings.resend_api_key}",
                 "Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except OSError:
            pass
        log.error("Resend API error sending to %s: HTTP %s %s", msg["To"], e.code, detail)
        return False
    except (urllib.error.URLError, OSError) as e:
        log.error("failed to reach Resend sending to %s: %s", msg["To"], e)
        return False


def _send_via_smtp(settings: Settings, msg: EmailMessage) -> bool:
    try:
        if settings.smtp_ssl:
            with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30,
                                  context=ssl.create_default_context()) as smtp:
                _login_send(settings, smtp, msg)
        else:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
                if settings.smtp_starttls:
                    smtp.starttls(context=ssl.create_default_context())
                _login_send(settings, smtp, msg)
        return True
    except (smtplib.SMTPException, OSError) as e:
        log.error("failed to send email to %s: %s", msg["To"], e)
        return False


def _login_send(settings: Settings, smtp: smtplib.SMTP, msg: EmailMessage) -> None:
    if settings.smtp_username:
        smtp.login(settings.smtp_username, settings.smtp_password)
    smtp.send_message(msg)
