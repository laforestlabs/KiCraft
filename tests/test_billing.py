"""Tests for kicraft.server.billing and the billing-side of the account store.

Pure stdlib + sqlite: every Stripe API call goes through a FakeGateway, so no
network and no stripe account are needed (the stripe SDK itself is only
imported by the real StripeGateway / verify_event, which these tests avoid).
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from kicraft.server import billing
from kicraft.server.accounts import DEFAULT_TIER, AccountStore
from kicraft.server.config import Settings


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


@pytest.fixture
def settings():
    return Settings(api_key="test", public_url="https://kicraft.test",
                    stripe_secret_key="sk_test_x",
                    stripe_webhook_secret="whsec_test_x",
                    stripe_price_pro="price_pro", stripe_price_max="price_max")


class FakeGateway:
    """Dict-backed stand-in for billing.StripeGateway (its dict-style access
    contract is exactly why the fakes can be plain dicts)."""

    def __init__(self, subs=None, sessions=None):
        self.subs = dict(subs or {})
        self.sessions = dict(sessions or {})
        self.customers_created = []
        self.canceled = []
        self.checkout_kwargs = None
        self.retrieve_count = 0

    def create_customer(self, *, email, metadata):
        cid = f"cus_fake{len(self.customers_created) + 1}"
        self.customers_created.append((cid, email, metadata))
        return {"id": cid}

    def create_checkout_session(self, **kwargs):
        self.checkout_kwargs = kwargs
        return {"url": "https://checkout.stripe.test/cs_1"}

    def retrieve_checkout_session(self, session_id):
        return self.sessions[session_id]

    def create_portal_session(self, *, customer, return_url):
        return {"url": f"https://portal.stripe.test/{customer}"}

    def retrieve_subscription(self, subscription_id):
        self.retrieve_count += 1
        return self.subs[subscription_id]

    def cancel_subscription(self, subscription_id):
        self.canceled.append(subscription_id)
        return {"id": subscription_id, "status": "canceled"}


def _sub(sid="sub_1", customer="cus_1", status="active", price="price_pro",
         period_end=None, metadata=None, basil=True):
    """A minimal subscription object. basil=True puts current_period_end on the
    item (the 2025-03-31 API shape stripe-python >= 12 pins); False puts it on
    the subscription (the legacy shape)."""
    if period_end is None:
        period_end = int(time.time()) + 30 * 86400
    item = {"price": {"id": price}}
    sub = {"id": sid, "customer": customer, "status": status,
           "metadata": metadata or {}, "items": {"data": [item]}}
    if basil:
        item["current_period_end"] = period_end
    else:
        sub["current_period_end"] = period_end
    return sub


# ---- price/tier mapping and object plumbing ---------------------------------

def test_price_tier_mapping(settings):
    assert billing.price_to_tier(settings, "price_pro") == "pro"
    assert billing.price_to_tier(settings, "price_max") == "max"
    assert billing.price_to_tier(settings, "price_other") is None
    assert billing.price_to_tier(settings, None) is None
    assert billing.tier_to_price(settings, "pro") == "price_pro"
    assert billing.tier_to_price(settings, "free") is None


def test_period_end_read_from_both_api_shapes():
    end = int(time.time()) + 1000
    assert billing._sub_period_end(_sub(period_end=end, basil=True)) == end
    assert billing._sub_period_end(_sub(period_end=end, basil=False)) == end
    assert billing._sub_period_end({"items": {"data": []}}) is None


def test_invoice_subscription_id_both_api_shapes():
    assert billing._invoice_subscription_id({"subscription": "sub_9"}) == "sub_9"
    assert billing._invoice_subscription_id(
        {"parent": {"subscription_details": {"subscription": "sub_9"}}}) == "sub_9"
    assert billing._invoice_subscription_id({}) is None


# ---- sync_subscription -------------------------------------------------------

def _paying_user(store, customer="cus_1"):
    u = store.create_user("payer@example.com", "hunter2hunter2")
    store.set_stripe_customer(u.id, customer)
    return u


def test_active_sub_sets_tier_and_expiry_with_grace(store, settings):
    u = _paying_user(store)
    end = int(time.time()) + 30 * 86400
    out = billing.sync_subscription(store, settings, _sub(period_end=end))
    assert out.startswith("synced")
    fresh = store.get_user(u.id)
    assert fresh.tier == "pro"
    assert fresh.subscription_status == "active"
    assert fresh.stripe_subscription_id == "sub_1"
    # expiry = period end + grace, so a renewal webhook always lands in time
    expiry = fresh.tier_expires_at
    assert expiry is not None and expiry[:10] != ""
    import datetime as dt
    parsed = dt.datetime.fromisoformat(expiry)
    expected = dt.datetime.fromtimestamp(end, tz=dt.timezone.utc) \
        + dt.timedelta(days=billing.GRACE_DAYS)
    assert abs((parsed - expected).total_seconds()) < 2


def test_max_price_maps_to_max_tier(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub(price="price_max"))
    assert store.get_user(u.id).tier == "max"


def test_past_due_keeps_tier(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    billing.sync_subscription(store, settings, _sub(status="past_due"))
    fresh = store.get_user(u.id)
    assert fresh.tier == "pro" and fresh.subscription_status == "past_due"


def test_canceled_records_status_but_leaves_access_to_lapse(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    before = store.get_user(u.id).tier_expires_at
    out = billing.sync_subscription(store, settings, _sub(status="canceled"))
    assert out.startswith("lapsing")
    fresh = store.get_user(u.id)
    assert fresh.tier == "pro"                      # paid-through access stays
    assert fresh.tier_expires_at == before          # ... until the old expiry
    assert fresh.subscription_status == "canceled"


def test_lapsed_expiry_downgrades_on_next_read(store, settings):
    """The end-to-end lapse: a sub synced with an already-past period end is
    downgraded to free by the existing _downgrade_if_expired machinery."""
    u = _paying_user(store)
    past = int(time.time()) - (billing.GRACE_DAYS + 1) * 86400
    billing.sync_subscription(store, settings, _sub(period_end=past))
    assert store.get_user(u.id).tier == DEFAULT_TIER


def test_unknown_customer_and_unknown_price(store, settings):
    assert billing.sync_subscription(
        store, settings, _sub(customer="cus_nobody")) == "unknown-customer"
    u = _paying_user(store)
    assert billing.sync_subscription(
        store, settings, _sub(price="price_other")) == "unknown-price"
    assert store.get_user(u.id).tier == DEFAULT_TIER  # nothing applied


def test_metadata_user_id_breaks_customer_link_race(store, settings):
    """checkout.session.completed can lose the race against the first
    subscription event; the user_id stamped into subscription metadata at
    Checkout resolves the user and back-fills the customer link."""
    u = store.create_user("racer@example.com", "hunter2hunter2")
    sub = _sub(customer="cus_new", metadata={"user_id": str(u.id)})
    out = billing.sync_subscription(store, settings, sub)
    assert out.startswith("synced")
    fresh = store.get_user(u.id)
    assert fresh.tier == "pro"
    assert fresh.stripe_customer_id == "cus_new"


# ---- handle_event ------------------------------------------------------------

def test_handle_event_checkout_completed_links_and_syncs(store, settings):
    u = store.create_user("buyer@example.com", "hunter2hunter2")
    gw = FakeGateway(subs={"sub_1": _sub(customer="cus_77")})
    event = {"id": "evt_1", "type": "checkout.session.completed",
             "data": {"object": {"client_reference_id": str(u.id),
                                 "customer": "cus_77", "subscription": "sub_1"}}}
    out = billing.handle_event(store, settings, event, gw)
    assert out.startswith("synced")
    fresh = store.get_user(u.id)
    assert fresh.tier == "pro" and fresh.stripe_customer_id == "cus_77"


def test_handle_event_refetches_rather_than_trusting_payload(store, settings):
    """An out-of-order 'canceled' event must not lapse a sub that Stripe says
    is active: the handler syncs from the re-fetched object."""
    u = _paying_user(store)
    gw = FakeGateway(subs={"sub_1": _sub(status="active")})
    stale = {"id": "evt_2", "type": "customer.subscription.updated",
             "data": {"object": _sub(status="canceled")}}
    billing.handle_event(store, settings, stale, gw)
    fresh = store.get_user(u.id)
    assert fresh.subscription_status == "active" and fresh.tier == "pro"
    assert gw.retrieve_count == 1


def test_handle_event_invoice_paid_basil_shape(store, settings):
    u = _paying_user(store)
    gw = FakeGateway(subs={"sub_1": _sub()})
    event = {"id": "evt_3", "type": "invoice.paid",
             "data": {"object": {"parent": {"subscription_details":
                                            {"subscription": "sub_1"}}}}}
    assert billing.handle_event(store, settings, event, gw).startswith("synced")
    assert store.get_user(u.id).tier == "pro"


def test_handle_event_ignores_unknown_types(store, settings):
    gw = FakeGateway()
    out = billing.handle_event(store, settings,
                               {"id": "evt_4", "type": "charge.refunded",
                                "data": {"object": {}}}, gw)
    assert out.startswith("ignored")
    assert gw.retrieve_count == 0


# ---- checkout / portal -------------------------------------------------------

def test_checkout_creates_customer_and_session(store, settings):
    u = store.create_user("new@example.com", "hunter2hunter2")
    gw = FakeGateway()
    url = billing.checkout_or_portal_url(store, settings, u, "pro", gw)
    assert url == "https://checkout.stripe.test/cs_1"
    kw = gw.checkout_kwargs
    assert kw["mode"] == "subscription"
    assert kw["line_items"] == [{"price": "price_pro", "quantity": 1}]
    assert kw["client_reference_id"] == str(u.id)
    assert kw["subscription_data"]["metadata"]["user_id"] == str(u.id)
    assert kw["success_url"].startswith("https://kicraft.test/billing/success")
    assert kw["cancel_url"] == "https://kicraft.test/pricing"
    # the created customer is persisted, so the webhook can resolve the user
    assert store.get_user(u.id).stripe_customer_id == "cus_fake1"


def test_checkout_with_live_sub_routes_to_portal(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    u = store.get_user(u.id)
    gw = FakeGateway(subs={"sub_1": _sub()})
    url = billing.checkout_or_portal_url(store, settings, u, "max", gw)
    assert url.startswith("https://portal.stripe.test/")
    assert gw.checkout_kwargs is None  # no second subscription


def test_checkout_with_stale_active_status_falls_through(store, settings):
    """Local status can say 'active' after webhooks were lost; the live check
    sees the sub is canceled, resyncs, and proceeds to a fresh Checkout."""
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    u = store.get_user(u.id)
    gw = FakeGateway(subs={"sub_1": _sub(status="canceled")})
    url = billing.checkout_or_portal_url(store, settings, u, "pro", gw)
    assert url == "https://checkout.stripe.test/cs_1"
    assert store.get_user(u.id).subscription_status == "canceled"


def test_checkout_rejects_unpriced_tier(store, settings):
    u = store.create_user("new@example.com", "hunter2hunter2")
    with pytest.raises(ValueError):
        billing.checkout_or_portal_url(store, settings, u, "free", FakeGateway())


def test_ensure_customer_reuses_existing(store, settings):
    u = _paying_user(store, customer="cus_keep")
    gw = FakeGateway()
    assert billing.ensure_customer(store, settings, store.get_user(u.id), gw) \
        == "cus_keep"
    assert gw.customers_created == []


# ---- success page sync -------------------------------------------------------

def test_sync_from_checkout_session_applies_tier(store, settings):
    u = store.create_user("buyer@example.com", "hunter2hunter2")
    gw = FakeGateway(
        subs={"sub_1": _sub(customer="cus_77")},
        sessions={"cs_1": {"client_reference_id": str(u.id),
                           "customer": "cus_77", "subscription": "sub_1"}})
    out = billing.sync_from_checkout_session(store, settings, u, "cs_1", gw)
    assert out.startswith("synced")
    assert store.get_user(u.id).tier == "pro"


def test_sync_from_checkout_session_rejects_other_users_session(store, settings):
    u = store.create_user("buyer@example.com", "hunter2hunter2")
    other = store.create_user("other@example.com", "hunter2hunter2")
    gw = FakeGateway(sessions={"cs_1": {"client_reference_id": str(other.id),
                                        "customer": "cus_x",
                                        "subscription": "sub_1"}})
    assert billing.sync_from_checkout_session(
        store, settings, u, "cs_1", gw) == "session-user-mismatch"
    assert store.get_user(u.id).tier == DEFAULT_TIER


# ---- cancellation on account deletion -----------------------------------------

def test_cancel_subscription_for_user(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    u = store.get_user(u.id)
    gw = FakeGateway()
    assert billing.cancel_subscription_for_user(settings, u, gw) is True
    assert gw.canceled == ["sub_1"]


def test_cancel_subscription_noop_when_unconfigured_or_unsubscribed(store, settings):
    u = store.create_user("nobody@example.com", "hunter2hunter2")
    assert billing.cancel_subscription_for_user(settings, u, FakeGateway()) is False
    bare = Settings(api_key="test")
    assert bare.billing_enabled is False
    sub_user = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    sub_user = store.get_user(sub_user.id)
    assert billing.cancel_subscription_for_user(bare, sub_user, FakeGateway()) is False


def test_cancel_subscription_swallows_api_errors(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    u = store.get_user(u.id)

    class Exploding(FakeGateway):
        def cancel_subscription(self, _sid):
            raise RuntimeError("stripe down")

    assert billing.cancel_subscription_for_user(settings, u, Exploding()) is False


# ---- store: event dedupe + migration -----------------------------------------

def test_billing_event_claim_and_release(store):
    assert store.record_billing_event("evt_1", "invoice.paid") is True
    assert store.record_billing_event("evt_1", "invoice.paid") is False
    store.forget_billing_event("evt_1")  # failed processing releases the claim
    assert store.record_billing_event("evt_1", "invoice.paid") is True


def test_old_db_gains_billing_columns(tmp_path):
    """A users table from before the Stripe columns existed migrates additively
    and its rows read back with the new fields as None."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL,"
            "password_hash TEXT NOT NULL,"
            "tier TEXT NOT NULL DEFAULT 'free',"
            "role TEXT NOT NULL DEFAULT 'user',"
            "created_at TEXT NOT NULL,"
            "last_login_at TEXT,"
            "accepted_terms_version TEXT,"
            "accepted_terms_at TEXT,"
            "allow_training INTEGER NOT NULL DEFAULT 1,"
            "session_epoch INTEGER NOT NULL DEFAULT 0,"
            "tier_expires_at TEXT,"
            "notify_email INTEGER NOT NULL DEFAULT 1)")
        conn.execute(
            "INSERT INTO users (email, password_hash, tier, created_at) "
            "VALUES ('old@example.com', 'x', 'pro', '2026-01-01T00:00:00+00:00')")
    store = AccountStore(db, tmp_path / "projects")
    u = store.get_user_by_email("old@example.com")
    assert u is not None and u.tier == "pro"
    assert u.stripe_customer_id is None
    assert u.stripe_subscription_id is None
    assert u.subscription_status is None
    store.set_stripe_customer(u.id, "cus_1")  # and the new methods work on it
    assert store.get_user_by_stripe_customer("cus_1").id == u.id
