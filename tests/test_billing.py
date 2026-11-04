"""Tests for kicraft.server.billing and the billing-side of the account store.

Pure stdlib + sqlite: every Stripe API call goes through a FakeGateway, so no
network and no stripe account are needed (the stripe SDK itself is only
imported by the real StripeGateway / verify_event, which these tests avoid).
"""
from __future__ import annotations

import sqlite3
import time
import threading

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
        self.customer_keys = {}
        self.customer_lock = threading.Lock()
        self.canceled = []
        self.checkout_kwargs = None
        self.checkout_keys = {}
        self.checkout_lock = threading.Lock()
        self.retrieve_count = 0

    def create_customer(self, *, email, metadata, idempotency_key):
        with self.customer_lock:
            if idempotency_key not in self.customer_keys:
                cid = f"cus_fake{len(self.customers_created) + 1}"
                self.customers_created.append((cid, email, metadata))
                self.customer_keys[idempotency_key] = cid
            return {"id": self.customer_keys[idempotency_key]}

    def create_checkout_session(self, **kwargs):
        with self.checkout_lock:
            self.checkout_kwargs = kwargs
            key = kwargs["idempotency_key"]
            if key not in self.checkout_keys:
                sid = f"cs_{len(self.checkout_keys) + 1}"
                self.checkout_keys[key] = sid
                self.sessions[sid] = {
                    "id": sid, "url": f"https://checkout.stripe.test/{sid}",
                    "status": "open", "customer": kwargs["customer"],
                    "mode": kwargs["mode"], "line_items": kwargs["line_items"],
                    "client_reference_id": kwargs["client_reference_id"],
                }
            return dict(self.sessions[self.checkout_keys[key]])

    def retrieve_checkout_session(self, session_id):
        return self.sessions[session_id]

    def expire_checkout_session(self, session_id):
        self.sessions[session_id]["status"] = "expired"
        return dict(self.sessions[session_id])

    def list_checkout_sessions(self, customer):
        return [s for s in self.sessions.values()
                if s.get("customer") == customer and s.get("status") == "open"]

    def list_checkout_line_items(self, session_id):
        return self.sessions[session_id]["line_items"]

    def list_subscriptions(self, customer):
        return [s for s in self.subs.values() if s.get("customer") == customer]

    def create_portal_session(self, *, customer, return_url):
        return {"url": f"https://portal.stripe.test/{customer}"}

    def retrieve_subscription(self, subscription_id):
        self.retrieve_count += 1
        return self.subs[subscription_id]

    def cancel_subscription(self, subscription_id):
        self.canceled.append(subscription_id)
        if subscription_id in self.subs:
            self.subs[subscription_id]["status"] = "canceled"
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


def test_concurrent_checkout_has_one_owner_and_one_provider_session(store, settings):
    from concurrent.futures import ThreadPoolExecutor

    user = _paying_user(store)
    started, release = threading.Event(), threading.Event()

    class RacingGateway(FakeGateway):
        def create_checkout_session(self, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().create_checkout_session(**kwargs)

    gw = RacingGateway()
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(billing.checkout_or_portal_url, store, settings, user, "pro", gw)
        try:
            assert started.wait(timeout=5)
            with pytest.raises(ValueError, match="being prepared"):
                billing.checkout_or_portal_url(store, settings, user, "pro", gw)
        finally:
            release.set()
        assert first.result().endswith("cs_1")
    assert billing.checkout_or_portal_url(store, settings, user, "pro", gw).endswith("cs_1")
    assert len(gw.checkout_keys) == 1


def test_checkout_never_substitutes_different_pending_plan(store, settings):
    user = _paying_user(store)
    gw = FakeGateway()
    assert billing.checkout_or_portal_url(store, settings, user, "pro", gw).endswith("cs_1")
    with pytest.raises(ValueError, match="another plan"):
        billing.checkout_or_portal_url(store, settings, user, "max", gw)
    gw.expire_checkout_session("cs_1")
    assert billing.checkout_or_portal_url(store, settings, user, "max", gw).endswith("cs_2")
    assert gw.checkout_kwargs["line_items"] == [{"price": "price_max", "quantity": 1}]


def test_unknown_old_checkout_is_not_redispatched(store, settings):
    user = _paying_user(store)
    pending = store.reserve_checkout(user.id, "pro")
    store.mark_checkout_dispatched(user.id, pending["idempotency_key"], pending["owner_token"])
    store.release_checkout(user.id, pending["idempotency_key"], pending["owner_token"])
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE pending_checkouts SET created_at='2000-01-01T00:00:00+00:00'")
    gw = FakeGateway()
    with pytest.raises(ValueError, match="Unresolved checkout"):
        billing.checkout_or_portal_url(store, settings, user, "pro", gw)
    assert gw.checkout_kwargs is None


def test_legacy_open_checkout_is_reused_only_for_matching_plan(store, settings):
    user = _paying_user(store)
    legacy = {"id": "cs_legacy", "customer": "cus_1", "status": "open",
              "mode": "subscription", "client_reference_id": str(user.id),
              "url": "https://checkout.stripe.test/legacy",
              "line_items": [{"price": "price_pro", "quantity": 1}]}
    gw = FakeGateway(sessions={"cs_legacy": legacy})
    with pytest.raises(ValueError, match="another plan"):
        billing.checkout_or_portal_url(store, settings, user, "max", gw)
    assert store.get_pending_checkout(user.id) is None
    assert billing.checkout_or_portal_url(store, settings, user, "pro", gw) == legacy["url"]
    assert gw.checkout_kwargs is None
    assert store.get_pending_checkout(user.id)["session_id"] == "cs_legacy"


def test_legacy_active_subscription_routes_to_portal_without_second_checkout(store, settings):
    user = _paying_user(store)
    gw = FakeGateway(subs={"sub_1": _sub()})
    assert billing.checkout_or_portal_url(store, settings, user, "pro", gw).startswith(
        "https://portal.stripe.test/")
    assert gw.checkout_kwargs is None
    assert store.get_user(user.id).stripe_subscription_id == "sub_1"


def test_unknown_legacy_open_checkout_blocks_account_deletion(store, settings):
    user = _paying_user(store)
    legacy = {"id": "cs_old", "customer": "cus_1", "status": "open",
              "client_reference_id": str(user.id), "mode": "subscription"}
    gw = FakeGateway(sessions={"cs_old": legacy})
    assert not billing.prepare_account_deletion(store, user, settings=settings, gw=gw)
    assert store.get_user(user.id) is not None
    assert gw.sessions["cs_old"]["status"] == "open"


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


def test_concurrent_customer_creation_uses_one_idempotency_key(store, settings):
    from concurrent.futures import ThreadPoolExecutor

    user = store.create_user("race@example.com", "pw")
    barrier = threading.Barrier(2)

    class RacingGateway(FakeGateway):
        def create_customer(self, **kwargs):
            barrier.wait(timeout=5)
            return super().create_customer(**kwargs)

    gw = RacingGateway()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _: billing.ensure_customer(store, settings, user, gw),
                            range(2)))
    assert ids == ["cus_fake1", "cus_fake1"]
    assert len(gw.customers_created) == 1
    assert store.get_user(user.id).stripe_customer_id == "cus_fake1"


def test_unresolved_customer_key_expires_without_second_remote_create(store, settings, monkeypatch):
    import datetime as dt
    from kicraft.server import accounts

    now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(accounts, "_utcnow", lambda: now)
    user = store.create_user("uncertain@example.com", "pw")

    class LostResponse(FakeGateway):
        def create_customer(self, **kwargs):
            super().create_customer(**kwargs)
            raise ConnectionError("response lost")

    gw = LostResponse()
    with pytest.raises(ConnectionError):
        billing.ensure_customer(store, settings, user, gw)
    now += dt.timedelta(hours=25)
    with pytest.raises(ValueError, match="Unresolved customer"):
        billing.ensure_customer(store, settings, user, gw)
    assert len(gw.customers_created) == 1


def test_customer_link_cannot_be_replaced_or_shared(store):
    first = store.create_user("first@example.com", "pw")
    second = store.create_user("second@example.com", "pw")
    store.set_stripe_customer(first.id, "cus_first")
    with pytest.raises(ValueError):
        store.set_stripe_customer(first.id, "cus_other")
    with pytest.raises(ValueError):
        store.set_stripe_customer(second.id, "cus_first")
    assert store.get_user(first.id).stripe_customer_id == "cus_first"
    assert store.get_user(second.id).stripe_customer_id is None


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
    assert billing.cancel_subscription_for_user(store, u, gw=gw) is True
    assert gw.canceled == ["sub_1"]
    # If deleting local data fails, a retry uses the persisted cancellation.
    assert store.get_user(u.id).subscription_status == "canceled"
    assert billing.cancel_subscription_for_user(store, store.get_user(u.id)) is True


def test_cancel_subscription_noop_when_unconfigured_or_unsubscribed(store, settings):
    u = store.create_user("nobody@example.com", "hunter2hunter2")
    assert billing.cancel_subscription_for_user(store, u) is True
    bare = Settings(api_key="test")
    assert bare.billing_enabled is False
    sub_user = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    sub_user = store.get_user(sub_user.id)
    assert billing.cancel_subscription_for_user(store, sub_user, settings=bare) is False


def test_cancel_subscription_swallows_api_errors(store, settings):
    u = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    u = store.get_user(u.id)

    class Exploding(FakeGateway):
        def cancel_subscription(self, _sid):
            raise RuntimeError("stripe down")

    assert billing.cancel_subscription_for_user(store, u, gw=Exploding()) is False
    assert store.get_user(u.id).subscription_status == "active"


def test_cancel_response_loss_rechecks_authoritative_subscription(store, settings):
    user = _paying_user(store)
    billing.sync_subscription(store, settings, _sub())
    user = store.get_user(user.id)

    class LostResponse(FakeGateway):
        def cancel_subscription(self, _sid):
            raise ConnectionError("response lost after cancellation")

    gw = LostResponse(subs={"sub_1": _sub(status="canceled")})
    assert billing.cancel_subscription_for_user(store, user, gw=gw)
    assert store.get_user(user.id).subscription_status == "canceled"


def test_account_deletion_expires_open_checkout_before_removing_link(store, settings):
    user = _paying_user(store)
    gw = FakeGateway()
    billing.checkout_or_portal_url(store, settings, user, "pro", gw)
    assert billing.prepare_account_deletion(store, user, settings=settings, gw=gw)
    assert gw.sessions["cs_1"]["status"] == "expired"
    assert store.get_pending_checkout(user.id) is None
    with pytest.raises(ValueError):
        store.reserve_checkout(user.id, "pro")
    store.delete_user(user.id, require_billing_safe=True)
    assert store.get_user(user.id) is None


def test_account_deletion_cancels_just_completed_checkout(store, settings):
    user = _paying_user(store)
    gw = FakeGateway()
    billing.checkout_or_portal_url(store, settings, user, "pro", gw)
    gw.subs["sub_1"] = _sub()
    gw.sessions["cs_1"].update(status="complete", subscription="sub_1")
    assert billing.prepare_account_deletion(store, user, settings=settings, gw=gw)
    assert gw.canceled == ["sub_1"]
    assert store.get_user(user.id).subscription_status == "canceled"
    store.delete_user(user.id, require_billing_safe=True)
    assert store.get_user(user.id) is None


def test_account_deletion_refuses_unknown_checkout_creation(store, settings):
    user = _paying_user(store)
    store.reserve_checkout(user.id, "pro")
    assert not billing.prepare_account_deletion(store, user, settings=settings, gw=FakeGateway())
    assert store.get_user(user.id) is not None
    assert store.get_pending_checkout(user.id) is not None


def test_dead_deletion_owner_can_retry_without_opening_checkout_race(store, monkeypatch):
    from kicraft.server import accounts

    user = store.create_user("deletion@example.com", "pw")
    store.begin_billing_deletion(user.id)
    with pytest.raises(ValueError):
        store.begin_billing_deletion(user.id)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE users SET billing_deleting_pid=987654321 WHERE id=?", (user.id,))
    monkeypatch.setattr(accounts, "_pid_alive", lambda pid: pid != 987654321)
    store.begin_billing_deletion(user.id)
    with pytest.raises(ValueError):
        store.reserve_checkout(user.id, "pro")
    store.delete_user(user.id, require_billing_safe=True)
    assert store.get_user(user.id) is None


# ---- store: event dedupe + migration -----------------------------------------

def test_billing_event_records_completion(store):
    assert not store.billing_event_processed("evt_1")
    assert store.record_billing_event("evt_1", "invoice.paid") is True
    assert store.billing_event_processed("evt_1")
    assert store.record_billing_event("evt_1", "invoice.paid") is False


def test_legacy_claim_is_reprocessed_after_migration(store):
    with sqlite3.connect(store.path) as conn:
        conn.execute("INSERT INTO billing_events (stripe_event_id, type, created_at) "
                     "VALUES ('evt_legacy', 'invoice.paid', '2026-01-01')")
        conn.execute("ALTER TABLE billing_events DROP COLUMN completed")
    reopened = AccountStore(store.path, store.projects_dir)
    assert not reopened.billing_event_processed("evt_legacy")
    assert reopened.record_billing_event("evt_legacy", "invoice.paid")
    assert reopened.billing_event_processed("evt_legacy")


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
