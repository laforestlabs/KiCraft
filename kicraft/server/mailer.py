"""Outbound email for the KiCraft web app, used to deliver password-reset links.

Stdlib smtplib only, so there is no third-party dependency to add or audit. The
mailer is intentionally tiny and fail-soft: when SMTP is not configured (no
smtp_host) it logs the message that would have been sent and returns False, so
local development and the test suite exercise the full reset flow without a mail
server. In production the box sets the KICRAFT_SMTP_* vars and real mail goes out.

The send functions never raise on mail trouble; they return a bool. Callers
(/forgot) show the same neutral message either way, so a delivery failure does not
leak whether an account exists.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from .config import Settings

log = logging.getLogger("kicraft.mailer")

_RESET_SUBJECT = "Reset your KiCraft password"


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
    from_addr = settings.smtp_from or settings.smtp_username
    return _send(settings, build_reset_email(to_addr, from_addr, reset_url, ttl_minutes))


def send_email(settings: Settings, to_addr: str, subject: str, body: str) -> bool:
    """Send one plain-text email via SMTP. Returns True on success.

    Returns False (and logs) when SMTP is unconfigured or sending fails, so the
    caller never crashes a request over mail trouble."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from or settings.smtp_username
    msg["To"] = to_addr
    msg.set_content(body)
    return _send(settings, msg)


def _send(settings: Settings, msg: EmailMessage) -> bool:
    if not settings.smtp_host:
        # Dev / unconfigured path: no mail server, so surface the message (which
        # for a reset contains the link) in the log for the operator to relay.
        log.warning("SMTP not configured (KICRAFT_SMTP_HOST unset); email to %s not "
                    "sent. Message follows:\n%s", msg["To"], msg.get_content())
        return False
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
