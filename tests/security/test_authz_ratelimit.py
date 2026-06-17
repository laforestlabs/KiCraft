"""Login/auth rate-limiting.

RED-UNTIL-FIXED: there is currently NO per-account or per-IP throttle on
authenticate()/login, so credential stuffing is unmetered. This test ENCODES that
gap (it documents today's behavior) and is the regression that will flip to
asserting the new limiter once hardening item F lands. Marked xfail so CI is green
while the gap is tracked, and will fail loudly (xpass) the moment a limiter exists,
prompting this test to be rewritten to assert it.
"""
from __future__ import annotations

import pytest


def test_wrong_password_returns_none(store):
    store.create_user("user@x.io", "correct-horse")
    assert store.authenticate("user@x.io", "wrong") is None
    assert store.authenticate("user@x.io", "correct-horse") is not None


@pytest.mark.xfail(reason="no login rate-limit yet (hardening item F); flips to a "
                          "real assertion once a limiter is added", strict=True)
def test_repeated_failures_are_throttled(store):
    """Hammer authenticate() with many wrong passwords; assert a lockout/throttle
    kicks in. EXPECTED TO FAIL today (no limiter) -> the tracked finding."""
    store.create_user("victim@x.io", "secret")
    attempts = 0
    for _ in range(50):
        store.authenticate("victim@x.io", "guess")
        attempts += 1
    # A limiter would either raise, return a sentinel, or expose a lock counter.
    # None of these exist yet, so this assertion fails (xfail) until F is built.
    assert hasattr(store, "is_rate_limited") and store.is_rate_limited("victim@x.io")
