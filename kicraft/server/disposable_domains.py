"""Blocklist of known disposable/throwaway email domains.

Used by ``AccountStore.create_user`` to reject signups from auto-receiving
services (mailinator, 10minutemail, guerrillamail, ...) that would let a script
burn free-tier compute with no real inbox behind it. Verification alone does
not stop these — the blocklist filters the obvious throwaway domains.

Kept in its own module so the list can be refreshed without churning the
accounts logic. Lower-cased; ``create_user`` normalizes the email domain the
same way before the lookup.
"""
from __future__ import annotations

# A curated set of the most common throwaway domains. Not exhaustive (new ones
# appear constantly), but cheap to maintain and catches the high-volume ones.
# Extend here; no other file needs to change.
DISPOSABLE_DOMAINS: frozenset[str] = frozenset({
    "mailinator.com",
    "10minutemail.com",
    "10minutemail.net",
    "guerrillamail.com",
    "guerrillamail.net",
    "guerrillamailblock.com",
    "tempmail.com",
    "tempmail.net",
    "temp-mail.org",
    "throwawaymail.com",
    "yopmail.com",
    "getnada.com",
    "maildrop.cc",
    "dispostable.com",
    "trashmail.com",
    "trashmail.net",
    "trash-mail.com",
    "fakeinbox.com",
    "sharklasers.com",
    "guerrillamail.info",
    "grr.la",
    "mailnesia.com",
    "mintemail.com",
    "mohmal.com",
    "tempr.email",
    "tmpmail.org",
    "tmpmail.net",
    "emailondeck.com",
    "spambog.com",
    "spambog.ru",
    "mailcatch.com",
    "inboxbear.com",
    "mytemp.email",
    "tempinbox.com",
    "moakt.com",
    "burnermail.io",
    "mail.tm",
})
