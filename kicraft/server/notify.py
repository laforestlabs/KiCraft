"""Walk-away email notifications for design runs.

A run takes many minutes (LLM stages + a possibly-queued build), so users leave.
This module emails them on the three transitions worth coming back for:

  ok             the board finished and the download is ready
  failed         the run ended without a board (so they can retry or rephrase)
  awaiting_input the run PARKED on a clarifying question; without an email a
                 walked-away user blocks their own run indefinitely, which makes
                 this the highest-value notification of the three

Sends are gated on the user's notify_email preference (the checkbox on the
design page) and suppressed when the user was active moments ago (they are
watching the live stream; the page itself shows the transition). Activity is
tracked in-process via mark_active(): _current_user() calls it on every authed
page render and the design page's render timer calls it each tick, so "active"
fades within seconds of the tab closing. The suppression window errs short: a
duplicate email to a watching user is mildly redundant, a missing email to a
walked-away one strands a run.

All sends are best-effort (the mailer never raises); a mail hiccup must never
affect a run's outcome.
"""
from __future__ import annotations

import logging
import threading
import time

from .accounts import AccountStore
from .config import Settings
from .mailer import render_html, send_email

log = logging.getLogger("kicraft.notify")

ACTIVE_WINDOW_S = 120.0

_last_seen: dict[int, float] = {}
_lock = threading.Lock()


def mark_active(user_id: int | None) -> None:
    """Record that this user just interacted with (or is watching) a page."""
    if user_id is None:
        return
    with _lock:
        _last_seen[int(user_id)] = time.monotonic()


def recently_active(user_id: int | None, window_s: float = ACTIVE_WINDOW_S) -> bool:
    if user_id is None:
        return False
    with _lock:
        seen = _last_seen.get(int(user_id))
    return seen is not None and (time.monotonic() - seen) <= window_s


# Per-status copy for the walk-away emails. Each entry drives both the plain-text
# body and the branded HTML (subject/heading/lead/CTA) rendered by _subject_body.
_RUN_EVENTS = {
    "ok": {
        "subject": "Your KiCraft board is ready: {title}",
        "heading": "Your board is ready",
        "preheader": "The routed board and fab package are ready to download.",
        "lead": ("Good news: your design “{title}” finished. The routed "
                 "board and fab package are ready to download."),
        "cta": "Download your board",
    },
    "failed": {
        "subject": "Your KiCraft run did not finish: {title}",
        "heading": "Your run didn't finish",
        "preheader": "Open it to see how far it got and try again.",
        "lead": ("Your design “{title}” ended without a finished board. "
                 "Open it to see how far it got and what failed; you can edit a "
                 "stage and re-run from there, or adjust the brief and try again."),
        "cta": "Open your design",
    },
    "awaiting_input": {
        "subject": "Your KiCraft design has a question: {title}",
        "heading": "Your design has a question",
        "preheader": "It's paused on a clarifying question and waiting for you.",
        "lead": ("Your design “{title}” is paused on a clarifying "
                 "question and will wait for you. Answer it to continue the run "
                 "right where it stopped."),
        "cta": "Answer and continue",
    },
}


def _subject_body(status: str, brief: str, url: str) -> tuple[str, str, str] | None:
    """Return (subject, plain_text, html) for a run transition, or None for a
    status we don't email about."""
    spec = _RUN_EVENTS.get(status)
    if spec is None:
        return None
    title = (brief or "your design").strip()
    if len(title) > 60:
        title = title[:57] + "..."
    lead = spec["lead"].format(title=title)
    subject = spec["subject"].format(title=title)
    text = f"{lead}\n\n    {url}\n"
    html_body = render_html(
        preheader=spec["preheader"],
        heading=spec["heading"],
        paragraphs=[lead],
        cta_label=spec["cta"],
        cta_url=url,
    )
    return subject, text, html_body


def notify_run_event(store: AccountStore, settings: Settings, *,
                     user_id: int | None, project_id: int | None,
                     status: str, brief: str = "",
                     skip_if_active: bool = True) -> bool:
    """Email `user_id` about a run transition. Returns True only when a message
    was actually handed to a backend. `skip_if_active=False` is for the startup
    sweep, where the process just restarted and the activity map is empty-but-
    meaningless anyway."""
    if user_id is None or status not in ("ok", "failed", "awaiting_input"):
        return False
    try:
        user = store.get_user(int(user_id))
        if user is None or not user.notify_email:
            return False
        if skip_if_active and recently_active(user.id):
            return False
        url = f"{settings.public_url}/?project={project_id}" if project_id \
            else settings.public_url
        sb = _subject_body(status, brief, url)
        if sb is None:
            return False
        sent = send_email(settings, user.email, sb[0], sb[1], html_body=sb[2])
        if sent:
            log.info("notified %s: project %s -> %s", user.email, project_id, status)
        return sent
    except Exception:  # a notification must never break a run
        log.exception("notify failed for user %s project %s", user_id, project_id)
        return False
