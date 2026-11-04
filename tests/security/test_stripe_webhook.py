"""Stripe webhook hardening: signature verification + idempotent replay.

The /billing/webhook endpoint mutates billing tier, so it must reject unsigned /
mis-signed payloads and safely replay events until processing completes.
billing.verify_event re-raises a bad signature as ValueError.
"""
from __future__ import annotations

import pytest

from kicraft.server.config import Settings


def test_bad_signature_is_rejected():
    pytest.importorskip("stripe")
    from kicraft.server import billing
    s = Settings(api_key="k", stripe_secret_key="sk_test", stripe_webhook_secret="whsec_test")
    with pytest.raises(ValueError):
        billing.verify_event(s, b'{"type":"customer.subscription.updated"}', "t=1,v1=deadbeef")
    with pytest.raises(ValueError):
        billing.verify_event(s, b'{}', "")  # absent signature


def test_event_dedupe_is_idempotent(store):
    """First record of an event id returns True (new); a replay returns False
    (already processed) so the handler can no-op -- a captured/replayed webhook
    cannot double-apply a subscription change."""
    assert store.record_billing_event("evt_123", "customer.subscription.updated") is True
    assert store.record_billing_event("evt_123", "customer.subscription.updated") is False
    # a different event id is independent
    assert store.record_billing_event("evt_456", "checkout.session.completed") is True


def test_legacy_uncompleted_claim_does_not_suppress_retry(store):
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO billing_events (stripe_event_id, type, created_at, completed) "
            "VALUES ('evt_retry', 'x', '2026-01-01', 0)")
    assert not store.billing_event_processed("evt_retry")
    assert store.record_billing_event("evt_retry", "x")
    assert store.billing_event_processed("evt_retry")
