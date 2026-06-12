"""Stripe billing for the paid tiers: hosted Checkout + Customer Portal.

Design (see also accounts.apply_subscription_state):

- Access is always decided by users.tier + users.tier_expires_at. A billing
  sync never grants a tier forever: each healthy cycle sets the expiry to the
  subscription's current_period_end plus GRACE_DAYS, so a renewal extends it
  and a canceled/failed subscription simply lapses to free via the existing
  _downgrade_if_expired machinery, even if every later webhook is lost.
- Webhook handling is order-independent: on any subscription-shaped event we
  re-fetch the subscription from Stripe and sync from that authoritative
  state, instead of trusting event payload ordering. Replay dedupe lives in
  accounts.record_billing_event (the webhook endpoint checks it first).
- Plan switching and cancellation happen in the Stripe Customer Portal (it
  must be configured in the dashboard to allow switching between the two
  prices). A user with a live subscription who hits "upgrade" again is sent
  to the portal rather than Checkout, which would create a second
  subscription.

Every Stripe API call goes through StripeGateway so tests inject a fake and
the rest of the server never imports stripe. Subscription/session objects are
accessed dict-style throughout (the SDK's StripeObject subclasses dict), so
fakes are plain dicts.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from .accounts import AccountStore, User
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

    def create_customer(self, *, email: str, metadata: dict) -> dict:
        return _plain(self._stripe.Customer.create(
            api_key=self._key, email=email, metadata=metadata))

    def create_checkout_session(self, **kwargs) -> dict:
        return _plain(
            self._stripe.checkout.Session.create(api_key=self._key, **kwargs))

    def retrieve_checkout_session(self, session_id: str) -> dict:
        return _plain(self._stripe.checkout.Session.retrieve(
            session_id, api_key=self._key))

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
    if user.stripe_customer_id:
        return user.stripe_customer_id
    customer = gw.create_customer(email=user.email,
                                  metadata={"user_id": str(user.id)})
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


def checkout_or_portal_url(store: AccountStore, settings: Settings,
                           user: User, tier: str, gw: StripeGateway) -> str:
    """URL to send an upgrading user to: Stripe Checkout normally, or the
    Customer Portal when they already hold a live subscription (switching
    plans there avoids creating a second subscription)."""
    price = tier_to_price(settings, tier)
    if price is None:
        raise ValueError(f"tier {tier!r} has no Stripe price configured")

    if user.stripe_subscription_id and user.subscription_status in KEEP_STATUSES:
        # Local state can be stale if webhooks were lost; trust but verify
        # before parking the user in a portal with nothing to manage.
        sub = gw.retrieve_subscription(user.stripe_subscription_id)
        if (sub.get("status") or "") in KEEP_STATUSES:
            return portal_url(store, settings, user, gw)
        sync_subscription(store, settings, sub)

    customer = ensure_customer(store, settings, user, gw)
    session = gw.create_checkout_session(
        mode="subscription",
        customer=customer,
        line_items=[{"price": price, "quantity": 1}],
        client_reference_id=str(user.id),
        subscription_data={"metadata": {"user_id": str(user.id)}},
        success_url=(settings.public_url
                     + "/billing/success?session_id={CHECKOUT_SESSION_ID}"),
        cancel_url=settings.public_url + "/pricing",
    )
    return session["url"]


def portal_url(store: AccountStore, settings: Settings, user: User,
               gw: StripeGateway) -> str:
    """Stripe Customer Portal link (update card, switch plan, cancel,
    download invoices). Only meaningful once a customer exists."""
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
    return sync_subscription(store, settings, sub)


def cancel_subscription_for_user(settings: Settings, user: User,
                                 gw: StripeGateway | None = None) -> bool:
    """Best-effort immediate cancel of the user's subscription, for the
    account-deletion paths: a deleted account must never be charged again.
    Swallows API errors (e.g. already canceled); deletion proceeds anyway."""
    if not (settings.billing_enabled and user.stripe_subscription_id):
        return False
    try:
        gw = gw or gateway(settings)
        gw.cancel_subscription(user.stripe_subscription_id)
        return True
    except Exception as e:
        log.warning("could not cancel subscription %s for user %s: %s",
                    user.stripe_subscription_id, user.id, e)
        return False
