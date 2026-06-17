"""Quota + spend-ceiling enforcement: a user cannot exceed their tier limit, and
no path spends past the ceiling (the cost kill-switch always trips first)."""
from __future__ import annotations

import pytest

from kicraft.server.accounts import TIERS
from kicraft.server.config import Settings
from kicraft.server.spend_guard import BudgetExceeded, KillSwitchEngaged, SpendGuard


def test_free_user_quota_is_bounded(store):
    u = store.create_user("free@x.io", "pw", tier="free")
    limit = TIERS["free"]["limit"]
    assert store.can_design(u) is True
    for _ in range(limit):
        store.create_project(u.id, "a board")  # each row reserves a quota slot
    q = store.quota_status(u)
    assert q["used"] >= limit and q["remaining"] == 0
    assert store.can_design(u) is False  # cannot exceed the tier limit


def test_awaiting_input_and_ok_still_consume_quota(store):
    """A parked or finished design holds a slot; only a 'failed' build frees it --
    so a user cannot dodge the quota by parking many runs on questions."""
    u = store.create_user("free2@x.io", "pw", tier="free")
    pid = store.create_project(u.id, "b")
    store.update_project_status(pid, "awaiting_input")
    assert store.count_active_designs(u.id, TIERS["free"]["window_days"]) == 1
    store.update_project_status(pid, "failed")
    assert store.count_active_designs(u.id, TIERS["free"]["window_days"]) == 0


def test_paid_tiers_have_higher_but_finite_limits(store):
    for tier in ("pro", "max"):
        u = store.create_user(f"{tier}@x.io", "pw", tier=tier)
        assert store.quota_status(u)["limit"] == TIERS[tier]["limit"]
        assert store.quota_status(u)["limit"] > TIERS["free"]["limit"]


def test_spend_ceiling_trips_before_overspend(tmp_path):
    s = Settings(api_key="k", ledger_path=tmp_path / "ledger.db",
                 daily_usd_ceiling=1.0, total_usd_ceiling=10.0)
    guard = SpendGuard(s)
    guard.preflight()  # clean: under ceilings
    guard.record("m", 1000, 1000, 0.6)
    guard.preflight()  # still under the $1 daily ceiling
    guard.record("m", 1000, 1000, 0.6)  # now $1.20 today
    with pytest.raises(BudgetExceeded):
        guard.preflight()  # daily ceiling reached -> refuse the next call


def test_kill_switch_refuses_all_calls(tmp_path):
    s = Settings(api_key="k", ledger_path=tmp_path / "ledger.db", kill_switch=True)
    with pytest.raises(KillSwitchEngaged):
        SpendGuard(s).preflight()


def test_every_completion_path_goes_through_preflight():
    """The cost cap is only safe if no client code path can spend without calling
    preflight(). Assert both public methods stream through the single _stream()
    helper that calls guard.preflight() (client.py:92)."""
    pytest.importorskip("requests")
    import inspect

    from kicraft.server import client
    src = inspect.getsource(client.CappedOpenRouterClient)
    assert "self.guard.preflight()" in src
    # both chat and chat_with_tools delegate to _stream (the single capped path)
    assert "self._stream(" in inspect.getsource(client.CappedOpenRouterClient.chat)
    assert "self._stream(" in inspect.getsource(client.CappedOpenRouterClient.chat_with_tools)
