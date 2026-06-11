"""Tests for the /billing/webhook endpoint: signature verification, replay
dedupe, claim-release on failure, and the happy-path tier sync.

The endpoint is called directly with a hand-built Starlette Request (the house
style: no HTTP client). Payloads are signed with Stripe's documented
``t=<ts>,v1=HMAC-SHA256(secret, "<ts>.<payload>")`` scheme and verified by the
real stripe SDK; only the API calls behind the gateway are faked.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

import pytest

from kicraft.server import web
from kicraft.server.accounts import DEFAULT_TIER, AccountStore

WEBHOOK_SECRET = "whsec_test_secret"


@pytest.fixture
def billing_env(tmp_path, monkeypatch):
    """Env + store wiring so Settings.from_env() is billing-enabled and the
    web module's shared store points at a throwaway DB."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("KICRAFT_STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("KICRAFT_STRIPE_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("KICRAFT_STRIPE_PRICE_PRO", "price_pro")
    monkeypatch.setenv("KICRAFT_STRIPE_PRICE_MAX", "price_max")
    monkeypatch.setenv("KICRAFT_USERS_DB", str(tmp_path / "accounts.db"))
    monkeypatch.setenv("KICRAFT_PROJECTS_DIR", str(tmp_path / "projects"))
    store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    old = web._STORE
    web._STORE = store
    yield store
    web._STORE = old


class FakeGateway:
    def __init__(self, subs=None):
        self.subs = dict(subs or {})
        self.retrieve_count = 0

    def retrieve_subscription(self, subscription_id):
        self.retrieve_count += 1
        return self.subs[subscription_id]


def _sub(customer="cus_1", status="active", price="price_pro"):
    return {"id": "sub_1", "customer": customer, "status": status,
            "metadata": {},
            "items": {"data": [{"price": {"id": price},
                                "current_period_end": int(time.time()) + 86400}]}}


def _sign(payload: bytes, secret: str = WEBHOOK_SECRET) -> str:
    ts = int(time.time())
    mac = hmac.new(secret.encode("utf-8"),
                   f"{ts}.".encode("utf-8") + payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def _event_payload(event_id: str, etype: str, obj: dict) -> bytes:
    return json.dumps({"id": event_id, "object": "event", "type": etype,
                       "data": {"object": obj}}).encode("utf-8")


def _request(body: bytes, sig: str | None):
    headers = [(b"stripe-signature", sig.encode("ascii"))] if sig else []
    scope = {"type": "http", "method": "POST", "path": "/billing/webhook",
             "headers": headers, "query_string": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    from starlette.requests import Request
    return Request(scope, receive)


def _post(body: bytes, sig: str | None):
    return asyncio.run(web.stripe_webhook(_request(body, sig)))


def test_valid_event_syncs_tier(billing_env, monkeypatch):
    store = billing_env
    u = store.create_user("payer@example.com", "hunter2hunter2")
    store.set_stripe_customer(u.id, "cus_1")
    monkeypatch.setattr(web.billing, "gateway", lambda s: FakeGateway(
        subs={"sub_1": _sub()}))
    body = _event_payload("evt_1", "customer.subscription.updated", _sub())
    resp = _post(body, _sign(body))
    assert resp.status_code == 200
    fresh = store.get_user(u.id)
    assert fresh.tier == "pro" and fresh.subscription_status == "active"


def test_bad_signature_rejected(billing_env, monkeypatch):
    gw = FakeGateway(subs={"sub_1": _sub()})
    monkeypatch.setattr(web.billing, "gateway", lambda s: gw)
    body = _event_payload("evt_1", "customer.subscription.updated", _sub())
    assert _post(body, _sign(body, secret="whsec_wrong")).status_code == 400
    assert _post(body, None).status_code == 400
    assert gw.retrieve_count == 0


def test_replayed_event_is_acked_but_not_reprocessed(billing_env, monkeypatch):
    store = billing_env
    u = store.create_user("payer@example.com", "hunter2hunter2")
    store.set_stripe_customer(u.id, "cus_1")
    gw = FakeGateway(subs={"sub_1": _sub()})
    monkeypatch.setattr(web.billing, "gateway", lambda s: gw)
    body = _event_payload("evt_1", "customer.subscription.updated", _sub())
    assert _post(body, _sign(body)).status_code == 200
    assert _post(body, _sign(body)).status_code == 200  # ack so Stripe stops
    assert gw.retrieve_count == 1                       # ... without rework


def test_handler_failure_releases_claim_for_retry(billing_env, monkeypatch):
    store = billing_env
    u = store.create_user("payer@example.com", "hunter2hunter2")
    store.set_stripe_customer(u.id, "cus_1")

    class Exploding(FakeGateway):
        def retrieve_subscription(self, _sid):
            raise RuntimeError("stripe API down")

    monkeypatch.setattr(web.billing, "gateway", lambda s: Exploding())
    body = _event_payload("evt_1", "customer.subscription.updated", _sub())
    assert _post(body, _sign(body)).status_code == 500
    assert store.get_user(u.id).tier == DEFAULT_TIER

    # Stripe retries the same event id; the released claim lets it process.
    monkeypatch.setattr(web.billing, "gateway", lambda s: FakeGateway(
        subs={"sub_1": _sub()}))
    assert _post(body, _sign(body)).status_code == 200
    assert store.get_user(u.id).tier == "pro"


def test_unknown_event_type_acked(billing_env, monkeypatch):
    gw = FakeGateway()
    monkeypatch.setattr(web.billing, "gateway", lambda s: gw)
    body = _event_payload("evt_1", "charge.refunded", {"id": "ch_1"})
    assert _post(body, _sign(body)).status_code == 200
    assert gw.retrieve_count == 0


def test_unconfigured_billing_refuses(billing_env, monkeypatch):
    monkeypatch.delenv("KICRAFT_STRIPE_SECRET_KEY", raising=False)
    body = _event_payload("evt_1", "invoice.paid", {})
    assert _post(body, _sign(body)).status_code == 503
