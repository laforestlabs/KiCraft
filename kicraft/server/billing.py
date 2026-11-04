"""Stripe billing for the paid tiers: hosted Checkout + Customer Portal.

Design (see also accounts.apply_subscription_state):

- Access is always decided by users.tier + users.tier_expires_at. A billing
  sync never grants a tier forever: each healthy cycle sets the expiry to the
  subscription's current_period_end plus GRACE_DAYS, so a renewal extends it
  and a canceled/failed subscription simply lapses to free via the existing
  _downgrade_if_expired machinery, even if every later webhook is lost.
- Webhooks re-fetch the subscription from Stripe rather than trusting event
  payload ordering. Completion is recorded only after sync, so a crash permits
  replay; concurrent duplicates may perform the same idempotent sync.
- One durable pending Checkout per account prevents parallel subscriptions.
  Its Stripe idempotency key is reused only inside a conservative retention
  window; uncertain older requests require billing support rather than a retry.
- Plan switching and cancellation happen in the Stripe Customer Portal (which
  must allow switching between prices). Live subscribers are sent there rather
  than creating a second Checkout.

Every Stripe API call goes through StripeGateway so tests inject a fake and
the rest of the server never imports stripe. Subscription/session objects are
accessed dict-style throughout (the SDK's StripeObject subclasses dict), so
fakes are plain dicts.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from .accounts import AccountStore, User, billing_key_is_fresh
from .config import Settings

log = logging.getLogger("kicraft.billing")

# Days of access past current_period_end. Covers webhook/renewal latency (a
# renewal invoice can settle hours after the period boundary) without letting
# a dead subscription keep its tier for long.
GRACE_DAYS = 3

# Subscription statuses that keep the paid tier. past_due is included because
# Stripe's smart retries usually recover the payment within the grace window;
# if they do not, the subscription moves to canceled/unpaid and the tier
# lapses at the already-set expiry.
KEEP_STATUSES = frozenset({"active", "trialing", "past_due"})


def _plain(obj) -> dict:
    """A StripeObject as plain nested dicts/lists. The SDK's objects stopped
    subclassing dict (newer majors have no .get), but their str() form is
    canonical JSON, so this conversion is version-proof. Plain dicts (the test
    fakes) pass through untouched."""
    if isinstance(obj, dict):
        return obj
    return json.loads(str(obj))


class StripeGateway:
    """The handful of Stripe API calls billing uses, behind one seam.

    Methods mirror the SDK 1:1, with every response normalized to plain dicts
    (see _plain), which is what lets the rest of this module use dict access
    and the tests use dict fakes. The import is deferred so the many
    non-server consumers of this package never need stripe installed."""

    def __init__(self, api_key: str):
        import stripe
        self._stripe = stripe
        self._key = api_key

    def create_customer(self, *, email: str, metadata: dict, idempotency_key: str) -> dict:
        return _plain(self._stripe.Customer.create(
            api_key=self._key, email=email, metadata=metadata,
            idempotency_key=idempotency_key))

    def create_checkout_session(self, **kwargs) -> dict:
        return _plain(
            self._stripe.checkout.Session.create(api_key=self._key, **kwargs))

    def retrieve_checkout_session(self, session_id: str) -> dict:
        return _plain(self._stripe.checkout.Session.retrieve(
            session_id, api_key=self._key))

    def expire_checkout_session(self, session_id: str) -> dict:
        return _plain(self._stripe.checkout.Session.expire(
            session_id, api_key=self._key))

    def list_checkout_sessions(self, customer: str) -> list[dict]:
        page = self._stripe.checkout.Session.list(
            customer=customer, status="open", limit=100, api_key=self._key)
        return [_plain(item) for item in page.auto_paging_iter()]

    def list_checkout_line_items(self, session_id: str) -> list[dict]:
        page = self._stripe.checkout.Session.list_line_items(
            session_id, limit=100, api_key=self._key)
        return [_plain(item) for item in page.auto_paging_iter()]

    def list_subscriptions(self, customer: str) -> list[dict]:
        page = self._stripe.Subscription.list(
            customer=customer, status="all", limit=100, api_key=self._key)
        return [_plain(item) for item in page.auto_paging_iter()]

    def create_portal_session(self, *, customer: str, return_url: str) -> dict:
        return _plain(self._stripe.billing_portal.Session.create(
            api_key=self._key, customer=customer, return_url=return_url))

    def retrieve_subscription(self, subscription_id: str) -> dict:
        return _plain(self._stripe.Subscription.retrieve(
            subscription_id, api_key=self._key))

    def cancel_subscription(self, subscription_id: str) -> dict:
        return _plain(self._stripe.Subscription.cancel(
            subscription_id, api_key=self._key))


def gateway(settings: Settings) -> StripeGateway:
    return StripeGateway(settings.stripe_secret_key)


def verify_event(settings: Settings, payload: bytes, sig_header: str) -> dict:
    """Parse + verify a webhook payload against the endpoint secret.

    Raises ValueError on a bad signature or malformed payload (the SDK's
    SignatureVerificationError is re-raised as ValueError so web.py does not
    need stripe's exception hierarchy)."""
    import stripe
    try:
        return _plain(stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret))
    except stripe.SignatureVerificationError as e:
        raise ValueError(f"bad stripe signature: {e}") from e


def tier_to_price(settings: Settings, tier: str) -> str | None:
    return {"pro": settings.stripe_price_pro,
            "max": settings.stripe_price_max}.get(tier) or None


def price_to_tier(settings: Settings, price_id: str | None) -> str | None:
    if not price_id:
        return None
    if price_id == settings.stripe_price_pro:
        return "pro"
    if price_id == settings.stripe_price_max:
        return "max"
    return None


def _sub_price_id(sub: dict) -> str | None:
    items = (sub.get("items") or {}).get("data") or []
    if not items:
        return None
    price = items[0].get("price") or {}
    return price.get("id")


def _sub_period_end(sub: dict) -> int | None:
    """Unix current_period_end. Newer Stripe API versions (2025-03-31 "basil",
    which stripe-python >= 12 pins) carry it on the subscription item; older
    ones on the subscription itself. Read both."""
    items = (sub.get("items") or {}).get("data") or []
    if items and items[0].get("current_period_end"):
        return int(items[0]["current_period_end"])
    if sub.get("current_period_end"):
        return int(sub["current_period_end"])
    return None


def _invoice_subscription_id(inv: dict) -> str | None:
    """Subscription id off an invoice event, across API shapes: top-level
    `subscription` pre-basil, `parent.subscription_details.subscription`
    from basil on."""
    if inv.get("subscription"):
        return inv["subscription"]
    details = (inv.get("parent") or {}).get("subscription_details") or {}
    return details.get("subscription") or None


def _expiry_from_sub(sub: dict) -> str:
    """ISO-8601 UTC tier expiry for a healthy subscription: period end plus
    grace. A missing period end (should not happen on a live subscription)
    falls back to one conservative monthly cycle from now, so a paying user
    is never locked out by a malformed object."""
    end = _sub_period_end(sub)
    if end is not None:
        instant = dt.datetime.fromtimestamp(end, tz=dt.timezone.utc)
    else:
        instant = dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(days=31)
    return (instant + dt.timedelta(days=GRACE_DAYS)).isoformat()


def ensure_customer(store: AccountStore, settings: Settings, user: User,
                    gw: StripeGateway) -> str:
    """The user's Stripe customer id, creating + persisting one on first use."""
    fresh = store.get_user(user.id)
    if fresh is None:
        raise ValueError(f"no user with id {user.id}")
    if fresh.stripe_customer_id:
        return fresh.stripe_customer_id
    key = store.stripe_customer_create_key(user.id)
    customer = gw.create_customer(email=fresh.email,
                                  metadata={"user_id": str(user.id)},
                                  idempotency_key=key)
    store.set_stripe_customer(user.id, customer["id"])
    return customer["id"]


def sync_subscription(store: AccountStore, settings: Settings,
                      sub: dict) -> str:
    """Sync local tier/expiry from one authoritative subscription object.

    Returns a short outcome string for logging. Healthy statuses set the tier
    from the price and push the expiry to period end + grace; terminal ones
    only record the status and leave tier/expiry to lapse on their own."""
    user = store.get_user_by_stripe_customer(sub.get("customer") or "")
    if user is None:
        # Customer linkage can lag the first webhook (checkout.session.completed
        # races subscription.updated); the metadata we stamp at Checkout breaks
        # the tie.
        meta_uid = (sub.get("metadata") or {}).get("user_id")
        if meta_uid and str(meta_uid).isdigit():
            user = store.get_user(int(meta_uid))
            if user is not None and sub.get("customer"):
                store.set_stripe_customer(user.id, sub["customer"])
    if user is None:
        return "unknown-customer"

    status = sub.get("status") or ""
    if status in KEEP_STATUSES:
        tier = price_to_tier(settings, _sub_price_id(sub))
        if tier is None:
            return "unknown-price"
        store.apply_subscription_state(
            user.id, tier=tier, tier_expires_at=_expiry_from_sub(sub),
            subscription_id=sub.get("id"), status=status)
        return f"synced user={user.id} tier={tier} status={status}"
    # Terminal (canceled/unpaid/...): keep whatever access was already paid
    # for; the existing expiry lapses it.
    store.apply_subscription_state(
        user.id, tier=user.tier, tier_expires_at=user.tier_expires_at,
        subscription_id=sub.get("id"), status=status)
    return f"lapsing user={user.id} status={status}"


def handle_event(store: AccountStore, settings: Settings, event: dict,
                 gw: StripeGateway) -> str:
    """Dispatch one verified, deduped webhook event. Unknown types are
    acknowledged untouched (Stripe sends whatever the endpoint subscribes to;
    being liberal here means a dashboard misconfiguration cannot 4xx-loop)."""
    etype = event.get("type") or ""
    obj = (event.get("data") or {}).get("object") or {}

    if etype == "checkout.session.completed":
        # Link the customer first so the subscription fetch below (and any
        # racing event) can resolve the user.
        ref = obj.get("client_reference_id")
        if ref and str(ref).isdigit() and obj.get("customer"):
            store.set_stripe_customer(int(ref), obj["customer"])
        sub_id = obj.get("subscription")
    elif etype in ("customer.subscription.updated",
                   "customer.subscription.deleted"):
        sub_id = obj.get("id")
    elif etype in ("invoice.paid", "invoice.payment_failed"):
        sub_id = _invoice_subscription_id(obj)
    else:
        return f"ignored type={etype}"

    if not sub_id:
        return f"no-subscription type={etype}"
    # Re-fetch rather than trusting the event body: events can arrive out of
    # order, the retrieved object cannot.
    sub = gw.retrieve_subscription(sub_id)
    return sync_subscription(store, settings, sub)


def _reconcile_checkout_history(store: AccountStore, settings: Settings, user: User,
                                customer: str, price: str, pending: dict,
                                gw: StripeGateway) -> str | None:
    """Adopt only a uniquely proven legacy session, or fail closed."""
    subscriptions = gw.list_subscriptions(customer)
    sessions = gw.list_checkout_sessions(customer)
    if any(s.get("customer") != customer or not s.get("id") or not s.get("status")
           for s in subscriptions):
        raise ValueError("Subscription history is incomplete; contact billing support.")
    live = [s for s in subscriptions if s["status"] not in ("canceled", "incomplete_expired")]
    if len(live) > 1 or (live and sessions):
        raise ValueError("Multiple billing objects require support reconciliation.")
    if live:
        sub = live[0]
        if sub["status"] not in KEEP_STATUSES or price_to_tier(settings, _sub_price_id(sub)) is None:
            raise ValueError("Existing subscription requires billing support.")
        if not sync_subscription(store, settings, sub).startswith("synced"):
            raise ValueError("Existing subscription could not be synchronized.")
        store.clear_pending_checkout(user.id, pending["idempotency_key"])
        return portal_url(store, settings, store.get_user(user.id), gw)
    if len(sessions) > 1:
        raise ValueError("Multiple open checkouts require billing support.")
    if sessions:
        session = sessions[0]
        sid = session.get("id")
        if (not sid or session.get("customer") != customer
                or session.get("status") != "open" or session.get("mode") != "subscription"
                or str(session.get("client_reference_id") or "") != str(user.id)
                or not session.get("url")):
            raise ValueError("Existing checkout cannot be verified.")
        items = gw.list_checkout_line_items(sid)
        item_price = items[0].get("price") if len(items) == 1 else None
        if isinstance(item_price, dict):
            item_price = item_price.get("id")
        if len(items) != 1 or item_price != price or items[0].get("quantity") != 1:
            raise ValueError("A checkout for another plan is already in progress.")
        store.save_pending_checkout(user.id, pending["idempotency_key"], sid, session["url"])
        return session["url"]
    return None


def checkout_or_portal_url(store: AccountStore, settings: Settings,
                           user: User, tier: str, gw: StripeGateway) -> str:
    """URL to send an upgrading user to: Stripe Checkout normally, or the
    Customer Portal when they already hold a live subscription (switching
    plans there avoids creating a second subscription)."""
    price = tier_to_price(settings, tier)
    if price is None:
        raise ValueError(f"tier {tier!r} has no Stripe price configured")

    user = store.get_user(user.id)
    if user is None:
        raise ValueError("account no longer exists")
    if user.billing_deleting:
        raise ValueError("account deletion is in progress")
    if user.stripe_subscription_id and user.subscription_status in KEEP_STATUSES:
        sub = gw.retrieve_subscription(user.stripe_subscription_id)
        if (sub.get("status") or "") in KEEP_STATUSES:
            return portal_url(store, settings, user, gw)
        sync_subscription(store, settings, sub)

    while True:
        previous = store.get_pending_checkout(user.id)
        if previous and previous["tier"] != tier:
            if previous["session_id"]:
                prior = gw.retrieve_checkout_session(previous["session_id"])
                if prior.get("status") == "expired":
                    store.clear_pending_checkout(user.id, previous["idempotency_key"])
                    continue
                if prior.get("status") == "complete":
                    outcome = sync_from_checkout_session(
                        store, settings, user, previous["session_id"], gw)
                    if outcome.startswith("synced"):
                        return portal_url(store, settings, store.get_user(user.id), gw)
                    if outcome.startswith("lapsing"):
                        continue
            raise ValueError("A checkout for another plan is already in progress.")
        pending = store.reserve_checkout(user.id, tier)
        key = pending["idempotency_key"]
        if pending["session_id"]:
            session = gw.retrieve_checkout_session(pending["session_id"])
            status = session.get("status")
            if status == "open":
                return session.get("url") or pending["session_url"]
            if status == "expired":
                store.clear_pending_checkout(user.id, key)
                continue
            if status == "complete":
                outcome = sync_from_checkout_session(
                    store, settings, user, pending["session_id"], gw)
                if outcome.startswith("synced"):
                    return portal_url(store, settings, store.get_user(user.id), gw)
                if outcome.startswith("lapsing"):
                    continue
                raise ValueError("Checkout is complete but subscription sync is pending.")
            raise ValueError("Checkout status is unavailable; contact billing support.")
        owner = pending["owner_token"]
        try:
            customer = ensure_customer(store, settings, user, gw)
            existing = _reconcile_checkout_history(store, settings, user, customer,
                                                   price, pending, gw)
            if existing is not None:
                return existing
            if not billing_key_is_fresh(pending["created_at"]):
                raise ValueError("Unresolved checkout requires billing support.")
            store.mark_checkout_dispatched(user.id, key, owner)
            session = gw.create_checkout_session(
                mode="subscription",
                customer=customer,
                line_items=[{"price": price, "quantity": 1}],
                client_reference_id=str(user.id),
                subscription_data={"metadata": {"user_id": str(user.id)}},
                success_url=(settings.public_url
                             + "/billing/success?session_id={CHECKOUT_SESSION_ID}"),
                cancel_url=settings.public_url + "/pricing",
                idempotency_key=key,
            )
            if not session.get("id") or not session.get("url"):
                raise ValueError("Checkout response is incomplete; retry safely.")
            store.save_pending_checkout(user.id, key, session["id"], session["url"])
            return session["url"]
        finally:
            store.release_checkout(user.id, key, owner)


def portal_url(store: AccountStore, settings: Settings, user: User,
               gw: StripeGateway) -> str:
    """Stripe Customer Portal link (update card, switch plan, cancel,
    download invoices). Only meaningful once a customer exists."""
    user = store.get_user(user.id)
    if user is None or user.billing_deleting:
        raise ValueError("account is unavailable for billing")
    customer = ensure_customer(store, settings, user, gw)
    session = gw.create_portal_session(
        customer=customer, return_url=settings.public_url + "/profile")
    return session["url"]


def sync_from_checkout_session(store: AccountStore, settings: Settings,
                               user: User, session_id: str,
                               gw: StripeGateway) -> str:
    """Optimistic sync for the /billing/success page: confirm the session
    belongs to this user, then sync its subscription immediately instead of
    waiting on the webhook. Returns the sync outcome string."""
    session = gw.retrieve_checkout_session(session_id)
    if str(session.get("client_reference_id") or "") != str(user.id):
        return "session-user-mismatch"
    if session.get("customer"):
        store.set_stripe_customer(user.id, session["customer"])
    sub_id = session.get("subscription")
    if not sub_id:
        return "no-subscription"
    sub = gw.retrieve_subscription(sub_id)
    outcome = sync_subscription(store, settings, sub)
    pending = store.get_pending_checkout(user.id)
    if outcome.startswith(("synced", "lapsing")) and pending and pending["session_id"] == session_id:
        store.clear_pending_checkout(user.id, pending["idempotency_key"])
    return outcome


def cancel_subscription_for_user(store: AccountStore, user: User, *,
                                 settings: Settings | None = None,
                                 gw: StripeGateway | None = None) -> bool:
    """Confirm cancellation before deletion, persisting it for a safe retry."""
    if not user.stripe_subscription_id or user.subscription_status == "canceled":
        return True
    if gw is None and (settings is None or not settings.billing_enabled):
        return False
    try:
        gw = gw or gateway(settings)
        try:
            result = gw.cancel_subscription(user.stripe_subscription_id)
        except Exception:
            # The provider may have canceled before a response was lost, or a
            # prior local write may have failed. Confirm its authoritative state.
            result = gw.retrieve_subscription(user.stripe_subscription_id)
        if result.get("status") != "canceled":
            return False
        store.apply_subscription_state(
            user.id, tier=user.tier, tier_expires_at=user.tier_expires_at,
            subscription_id=user.stripe_subscription_id, status="canceled")
        return True
    except Exception as e:
        log.warning("could not cancel subscription %s for user %s: %s",
                    user.stripe_subscription_id, user.id, e)
        return False


def prepare_account_deletion(store: AccountStore, user: User, *,
                             settings: Settings | None = None,
                             gw: StripeGateway | None = None) -> bool:
    """Block checkout, settle any pending session, then confirm cancellation.

    On success the caller must delete with require_billing_safe=True and clear
    the deletion flag if that deletion fails. Unknown provider state fails closed.
    """
    user_id = user.id
    try:
        store.begin_billing_deletion(user_id)
    except ValueError:
        return False
    ready = False
    try:
        user = store.get_user(user.id)
        if user is None:
            return False
        if store.customer_creation_unresolved(user_id):
            return False
        pending = store.get_pending_checkout(user.id)
        if user.stripe_customer_id:
            if gw is None and (settings is None or not settings.billing_enabled):
                return False
            gw = gw or gateway(settings)
            open_sessions = gw.list_checkout_sessions(user.stripe_customer_id)
            if any(not pending or s.get("id") != pending["session_id"] for s in open_sessions):
                return False
        elif user.stripe_subscription_id:
            return False
        if pending:
            if not pending["session_id"]:
                return False  # creation is in flight or its result is unknown
            if gw is None and (settings is None or not settings.billing_enabled):
                return False
            gw = gw or gateway(settings)
            session = gw.retrieve_checkout_session(pending["session_id"])
            if (str(session.get("client_reference_id") or "") != str(user.id)
                    or session.get("customer") != user.stripe_customer_id
                    or session.get("mode") != "subscription"):
                return False
            if session.get("status") == "open":
                try:
                    session = gw.expire_checkout_session(pending["session_id"])
                except Exception:
                    session = gw.retrieve_checkout_session(pending["session_id"])
            if session.get("status") == "complete":
                sub_id = session.get("subscription")
                if not sub_id:
                    return False
                if (user.stripe_subscription_id and user.subscription_status != "canceled"
                        and user.stripe_subscription_id != sub_id):
                    return False
                sub = gw.retrieve_subscription(sub_id)
                if sub.get("id") != sub_id or session.get("customer") != sub.get("customer"):
                    return False
                store.set_stripe_customer(user.id, session["customer"])
                store.apply_subscription_state(
                    user.id, tier=user.tier, tier_expires_at=user.tier_expires_at,
                    subscription_id=sub_id, status=sub.get("status"))
                user = store.get_user(user.id)
                if user is None or not cancel_subscription_for_user(store, user, gw=gw):
                    return False
            elif session.get("status") != "expired":
                return False
            if not store.clear_pending_checkout(user.id, pending["idempotency_key"]):
                return False
        user = store.get_user(user.id)
        if user is None or not cancel_subscription_for_user(store, user, settings=settings, gw=gw):
            return False
        if user.stripe_customer_id:
            if gw.list_checkout_sessions(user.stripe_customer_id):
                return False
            subscriptions = gw.list_subscriptions(user.stripe_customer_id)
            if any(s.get("customer") != user.stripe_customer_id or not s.get("id")
                   or s.get("status") not in ("canceled", "incomplete_expired")
                   for s in subscriptions):
                return False
        elif user.stripe_subscription_id:
            return False
        ready = True
        return True
    except Exception as e:
        log.warning("could not prepare account %s for deletion: %s", user_id, e)
        return False
    finally:
        if not ready:
            store.end_billing_deletion(user_id)
