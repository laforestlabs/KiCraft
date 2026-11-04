"""Tests for the public /pricing page: the pure CTA/bullet helpers, and a
NiceGUI user-simulation smoke that the page renders for logged-out visitors
(mirroring tests/test_web_support_reports.py's harness)."""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server import web
from kicraft.server.accounts import TIERS, AccountStore

WEB = "kicraft.server.web"


def _user(tier="free"):
    return SimpleNamespace(tier=tier)


# ---- pure helpers ------------------------------------------------------------

def test_cta_logged_out_always_routes_to_signup():
    """Hard rule: nothing chargeable (and no checkout) before an account."""
    for tier in TIERS:
        label, href = web._pricing_cta(None, tier, billing_on=True)
        assert href == "/signup", (tier, label)


def test_cta_current_plan_is_disabled():
    assert web._pricing_cta(_user("free"), "free", True)[1] is None
    assert web._pricing_cta(_user("pro"), "pro", True)[1] is None


def test_cta_upgrade_and_switch_route_to_checkout():
    label, href = web._pricing_cta(_user("free"), "pro", True)
    assert href == "/billing/checkout?tier=pro" and label.startswith("Upgrade")
    label, href = web._pricing_cta(_user("pro"), "max", True)
    assert href == "/billing/checkout?tier=max" and label.startswith("Switch")


def test_cta_paid_tiers_disabled_until_billing_configured():
    label, href = web._pricing_cta(_user("free"), "pro", billing_on=False)
    assert href is None and label == "Coming soon"
    # the free tier needs no billing at all
    assert web._pricing_cta(None, "free", billing_on=False)[1] == "/signup"


def test_cta_free_card_never_a_purchase_for_paid_users():
    label, href = web._pricing_cta(_user("max"), "free", True)
    assert href is None


def test_bullets_quote_the_enforced_quota():
    """The card copy must come from TIERS, so the page can never promise a
    different quota than quota_status enforces."""
    for key, info in TIERS.items():
        period = "week" if info["window_days"] <= 7 else "month"
        first = web._pricing_bullets(key, info)[0]
        assert str(info["limit"]) in first and period in first


