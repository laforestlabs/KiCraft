"""Login rate limits at the account and trusted-client boundaries."""
from __future__ import annotations

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor

from kicraft.server import accounts


def test_wrong_password_returns_none(store):
    store.create_user("user@x.io", "correct-horse")
    assert store.authenticate("user@x.io", "wrong") is None
    assert store.authenticate("user@x.io", "correct-horse") is not None


def test_account_lock_expires_and_success_resets_failures(store, monkeypatch):
    now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(accounts, "_utcnow", lambda: now)
    store.create_user("victim@x.io", "secret")
    for _ in range(accounts._LOGIN_ACCOUNT_LIMIT):
        assert store.authenticate("victim@x.io", "guess", client_ip="192.0.2.1") is None
    assert store.authenticate("victim@x.io", "secret", client_ip="192.0.2.1") is None
    now += dt.timedelta(seconds=accounts._LOGIN_WINDOW_SECONDS + 1)
    assert store.authenticate("victim@x.io", "secret", client_ip="192.0.2.1") is not None
    assert store.authenticate("victim@x.io", "wrong", client_ip="192.0.2.1") is None
    assert store.authenticate("victim@x.io", "secret", client_ip="192.0.2.1") is not None


def test_concurrent_attempts_cannot_bypass_account_limit(store, monkeypatch):
    store.create_user("victim@x.io", "secret")
    start = threading.Barrier(10)
    original = accounts.verify_password
    calls = []
    lock = threading.Lock()

    def counted_verify(password, stored):
        with lock:
            calls.append(1)
        return original(password, stored)

    monkeypatch.setattr(accounts, "verify_password", counted_verify)

    def fail(_):
        start.wait(timeout=5)
        return store.authenticate("victim@x.io", "wrong")

    with ThreadPoolExecutor(max_workers=10) as pool:
        assert all(result is None for result in pool.map(fail, range(10)))
    assert len(calls) == accounts._LOGIN_ACCOUNT_LIMIT
    assert store.authenticate("victim@x.io", "secret") is None


def test_ip_limit_covers_many_account_names_and_expires(store, monkeypatch):
    now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(accounts, "_utcnow", lambda: now)
    store.create_user("real@x.io", "secret")
    for i in range(accounts._LOGIN_IP_LIMIT):
        assert store.authenticate(f"missing{i}@x.io", "wrong", client_ip="192.0.2.2") is None
    assert store.authenticate("real@x.io", "secret", client_ip="192.0.2.2") is None
    assert store.authenticate("real@x.io", "secret", client_ip="192.0.2.3") is not None
    now += dt.timedelta(seconds=accounts._LOGIN_WINDOW_SECONDS + 1)
    assert store.authenticate("real@x.io", "secret", client_ip="192.0.2.2") is not None
