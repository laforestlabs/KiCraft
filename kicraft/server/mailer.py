"""Outbound email for the KiCraft web app: password resets, signup verification,
and walk-away run notifications.

No third-party dependency: SMTP goes through stdlib smtplib, and the Resend
backend posts JSON with stdlib urllib. The mailer is tiny and fail-soft: with no
backend configured it logs the message that would have been sent and returns
False, so local dev and the test suite exercise the full flow without a mail
server. In production the box configures one backend.

Two backends, selected by config (Resend wins if both are set):
  - Resend HTTP API  -> settings.resend_api_key
  - SMTP             -> settings.smtp_host (+ the other smtp_* fields)

Every message is sent multipart/alternative: a plain-text part (accessibility,
deliverability, and the dev-log fallback) plus a branded HTML part rendered by
`render_html`. The HTML is built for hostile email clients: table layout, inline
styles only, no CSS variables / flexbox / <style> reliance, a bulletproof
table-cell CTA button, and a visible pasteable-link fallback under it (remote
images are blocked by default, so the brand is drawn in CSS, not fetched).

The send functions never raise on mail trouble; they return a bool. Callers
(/forgot) show the same neutral message either way, so a delivery failure does not
leak whether an account exists.
"""
from __future__ import annotations

import html
import json
import logging
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from email.utils import formataddr, parseaddr

from .config import Settings

log = logging.getLogger("kicraft.mailer")

_RESET_SUBJECT = "Reset your KiCraft password"
_VERIFY_SUBJECT = "Confirm your KiCraft email"
_RESEND_ENDPOINT = "https://api.resend.com/emails"
# A descriptive User-Agent is required: the Resend API is fronted by Cloudflare,
# which 403s the stdlib default "Python-urllib/x.y" signature (Cloudflare error
# 1010, "banned browser signature"). Any real UA gets through.
_USER_AGENT = "KiCraft/1.0 (+https://kicraft.io)"

# The sender display name shown in the inbox's "From" column. A bare address reads
# as noise; "KiCraft" reads as the product.
_FROM_NAME = "KiCraft"
_HOME_URL = "https://kicraft.io"

# --- Brand palette (mirrors kicraft.server.theme / the landing page) ----------
# Email clients don't support CSS variables, so these are inlined at render time.
_BG = "#0b0f14"          # page background (dark)
_PANEL = "#12171f"       # card surface
_BORDER = "#232c38"      # hairline borders
_BRAND = "#4ade80"       # signature circuit-green
_ACCENT = "#22d3ee"      # cyan accent (wordmark second tone)
_TEXT = "#e8eef5"        # primary text
_BODY = "#c7d0da"        # body copy (slightly softer than headings)
_MUTED = "#9aa7b5"       # secondary text
_DIM = "#657085"         # footer / fine print
_FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
         "Arial, sans-serif")


def _from_addr(settings: Settings) -> str:
    """The sender header, provider-agnostic and with a friendly display name.

    ``email_from`` is the canonical setting; fall back to the SMTP sender/login
    for back-compat. If the configured value is a bare address (no display name),
    wrap it as ``KiCraft <addr>`` so the inbox shows the product, not a raw
    mailbox. A value that already carries a display name is passed through."""
    raw = settings.email_from or settings.smtp_from or settings.smtp_username or ""
    name, addr = parseaddr(raw)
    if addr and not name:
        return formataddr((_FROM_NAME, addr))
    return raw


# --------------------------------------------------------------------------- #
# Branded HTML rendering
# --------------------------------------------------------------------------- #
def render_html(*, preheader: str, heading: str, paragraphs: list[str],
                cta_label: str | None = None, cta_url: str | None = None,
                note: str | None = None) -> str:
    """Render one transactional email as brand-matched, client-safe HTML.

    Pure (no I/O), so it is unit-testable. Everything caller-supplied is
    HTML-escaped. Layout is table-based with inline styles only; the CTA is a
    bulletproof table-cell button, and the destination URL is repeated as a
    pasteable link below it for clients that strip buttons or block them.
    """
    esc_pre = html.escape(preheader)
    esc_heading = html.escape(heading)

    body_blocks = "".join(
        f'<p style="margin:0 0 16px;font-size:15px;line-height:1.6;color:{_BODY};">'
        f'{html.escape(p)}</p>'
        for p in paragraphs
    )

    cta_block = ""
    if cta_label and cta_url:
        safe_href = html.escape(cta_url, quote=True)
        safe_label = html.escape(cta_label)
        safe_url_text = html.escape(cta_url)
        cta_block = f"""
        <table role="presentation" cellpadding="0" cellspacing="0" border="0"
               align="center" style="margin:8px auto 20px;">
          <tr>
            <td align="center" bgcolor="{_BRAND}" style="border-radius:8px;">
              <a href="{safe_href}" target="_blank"
                 style="display:inline-block;padding:13px 30px;font-family:{_FONT};
                        font-size:15px;font-weight:700;line-height:1;color:{_BG};
                        text-decoration:none;border-radius:8px;">{safe_label}</a>
            </td>
          </tr>
        </table>
        <p style="margin:0 0 4px;font-size:12px;line-height:1.5;color:{_DIM};">
          Button not working? Copy and paste this link into your browser:
        </p>
        <p style="margin:0 0 20px;font-size:12px;line-height:1.5;word-break:break-all;">
          <a href="{safe_href}" target="_blank"
             style="color:{_BRAND};text-decoration:underline;">{safe_url_text}</a>
        </p>"""

    note_block = ""
    if note:
        note_block = (
            f'<p style="margin:0;font-size:13px;line-height:1.6;color:{_MUTED};'
            f'border-top:1px solid {_BORDER};padding-top:16px;">'
            f'{html.escape(note)}</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="dark">
<title>{esc_heading}</title>
</head>
<body style="margin:0;padding:0;background-color:{_BG};">
  <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:{_BG};">
    {esc_pre}
  </div>
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
         bgcolor="{_BG}" style="background-color:{_BG};">
    <tr>
      <td align="center" style="padding:32px 16px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
               border="0" style="max-width:520px;width:100%;">
          <!-- Brand header -->
          <tr>
            <td style="padding:4px 4px 20px;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td width="40" height="40" align="center" valign="middle"
                      bgcolor="{_PANEL}"
                      style="width:40px;height:40px;border:1.5px solid {_BRAND};
                             border-radius:10px;font-family:{_FONT};font-size:22px;
                             font-weight:800;color:{_BRAND};">K</td>
                  <td style="padding-left:12px;font-family:{_FONT};font-size:22px;
                             font-weight:800;letter-spacing:-0.01em;color:{_BRAND};">
                    KiCraft</td>
                </tr>
              </table>
            </td>
          </tr>
          <!-- Card -->
          <tr>
            <td bgcolor="{_PANEL}"
                style="background-color:{_PANEL};border:1px solid {_BORDER};
                       border-radius:14px;padding:32px 32px 28px;">
              <h1 style="margin:0 0 18px;font-family:{_FONT};font-size:21px;
                         font-weight:700;line-height:1.3;color:{_TEXT};">
                {esc_heading}</h1>
              {body_blocks}
              {cta_block}
              {note_block}
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding:22px 8px 4px;font-family:{_FONT};font-size:12px;
                       line-height:1.6;color:{_DIM};">
              <p style="margin:0 0 4px;">
                <a href="{_HOME_URL}" target="_blank"
                   style="color:{_MUTED};text-decoration:none;font-weight:600;">
                   KiCraft</a>
                &nbsp;&middot;&nbsp; Fabricable KiCad PCBs from a sentence.
              </p>
              <p style="margin:0;">
                This is a transactional message from
                <a href="{_HOME_URL}" target="_blank"
                   style="color:{_DIM};text-decoration:underline;">kicraft.io</a>.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def _message(*, to_addr: str, from_addr: str, subject: str, text: str,
             html_body: str) -> EmailMessage:
    """Build a multipart/alternative message (plain text + branded HTML)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    return msg


# --------------------------------------------------------------------------- #
# Password reset
# --------------------------------------------------------------------------- #
def build_reset_email(to_addr: str, from_addr: str, reset_url: str,
                      ttl_minutes: int = 60) -> EmailMessage:
    """Compose the password-reset email. Pure (no I/O), so it is unit-testable."""
    text = (
        "We received a request to reset your KiCraft password.\n\n"
        f"Open this link to choose a new one (valid for {ttl_minutes} minutes):\n\n"
        f"    {reset_url}\n\n"
        "Resetting your password also signs out every other device, so if someone "
        "else had access to your account they will be logged out.\n\n"
        "If you did not request this you can ignore this email; your password will "
        "not change.\n"
    )
    html_body = render_html(
        preheader="Choose a new KiCraft password.",
        heading="Reset your password",
        paragraphs=[
            "We received a request to reset the password for your KiCraft account. "
            "Click the button below to choose a new one.",
        ],
        cta_label="Choose a new password",
        cta_url=reset_url,
        note=(
            f"This link is valid for {ttl_minutes} minutes. Resetting your password "
            "also signs out every other device. If you did not request this, you can "
            "safely ignore this email: your password will not change."
        ),
    )
    return _message(to_addr=to_addr, from_addr=from_addr, subject=_RESET_SUBJECT,
                    text=text, html_body=html_body)


def send_reset_email(settings: Settings, to_addr: str, reset_url: str,
                     ttl_minutes: int = 60) -> bool:
    """Compose and send the reset email. See `send_email` for the return contract."""
    msg = build_reset_email(to_addr, _from_addr(settings), reset_url, ttl_minutes)
    return _send(settings, msg)


# --------------------------------------------------------------------------- #
# Signup verification
# --------------------------------------------------------------------------- #
def build_verification_email(to_addr: str, from_addr: str, verify_url: str,
                             ttl_hours: int = 24) -> EmailMessage:
    """Compose the signup email-verification email. Pure (no I/O), unit-testable."""
    text = (
        "Welcome to KiCraft! Confirm your email to start designing boards.\n\n"
        f"Open this link to verify your address (valid for {ttl_hours} hours):\n\n"
        f"    {verify_url}\n\n"
        "You can browse example boards while unverified, but the Design button "
        "stays disabled until you confirm.\n\n"
        "If you did not create a KiCraft account you can ignore this email.\n"
    )
    html_body = render_html(
        preheader="Confirm your email to start designing boards.",
        heading="Welcome to KiCraft",
        paragraphs=[
            "You're one click away from designing fabricable PCBs from a sentence. "
            "Confirm your email address to unlock the Design button.",
        ],
        cta_label="Verify email address",
        cta_url=verify_url,
        note=(
            f"This link is valid for {ttl_hours} hours. You can browse example boards "
            "before verifying, but the Design button stays disabled until you "
            "confirm. If you did not create a KiCraft account, you can ignore this "
            "email."
        ),
    )
    return _message(to_addr=to_addr, from_addr=from_addr, subject=_VERIFY_SUBJECT,
                    text=text, html_body=html_body)


def send_verification_email(settings: Settings, to_addr: str, verify_url: str,
                            ttl_hours: int = 24) -> bool:
    """Compose and send the signup verification email. See `send_email` for the
    return contract."""
    msg = build_verification_email(to_addr, _from_addr(settings), verify_url,
                                   ttl_hours)
    return _send(settings, msg)


# --------------------------------------------------------------------------- #
# Generic send (used by run notifications in notify.py)
# --------------------------------------------------------------------------- #
def send_email(settings: Settings, to_addr: str, subject: str, body: str,
               html_body: str | None = None) -> bool:
    """Send one email. When `html_body` is given the message is multipart
    (branded HTML + the plain-text `body` fallback); otherwise it is plain text.

    Returns True on success. Returns False (and logs) when no backend is
    configured or sending fails, so the caller never crashes a request over mail
    trouble."""
    if html_body is not None:
        msg = _message(to_addr=to_addr, from_addr=_from_addr(settings),
                       subject=subject, text=body, html_body=html_body)
    else:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = _from_addr(settings)
        msg["To"] = to_addr
        msg.set_content(body)
    return _send(settings, msg)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _parts(msg: EmailMessage) -> tuple[str, str | None]:
    """Extract (plain_text, html_or_None) from a message, whether it is a simple
    text message or multipart/alternative."""
    text_part = msg.get_body(preferencelist=("plain",))
    html_part = msg.get_body(preferencelist=("html",))
    text = text_part.get_content() if text_part is not None else ""
    html_body = html_part.get_content() if html_part is not None else None
    return text, html_body


def _send(settings: Settings, msg: EmailMessage) -> bool:
    if settings.resend_api_key:
        return _send_via_resend(settings, msg)
    if settings.smtp_host:
        return _send_via_smtp(settings, msg)
    # Dev / unconfigured path: no backend, so surface the message (which for a
    # reset contains the link) in the log for the operator to relay.
    text, _ = _parts(msg)
    log.warning("no email backend configured (set KICRAFT_RESEND_API_KEY or "
                "KICRAFT_SMTP_HOST); email to %s not sent. Message follows:\n%s",
                msg["To"], text)
    return False


def _send_via_resend(settings: Settings, msg: EmailMessage) -> bool:
    """POST the message to the Resend HTTP API (stdlib urllib, no SDK dependency)."""
    text, html_body = _parts(msg)
    body: dict = {
        "from": msg["From"],
        "to": [msg["To"]],
        "subject": msg["Subject"],
        "text": text,
    }
    if html_body is not None:
        body["html"] = html_body
    payload = json.dumps(body).encode("utf-8")
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
