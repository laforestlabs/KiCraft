"""Tests for the kicraft-accounts CLI: admin-role bootstrap and guards.

Drives main(argv) against a temp DB via the KICRAFT_USERS_DB / KICRAFT_PROJECTS_DIR
env vars the CLI resolves its paths from (monkeypatched per test). Pure stdlib +
sqlite -- no network, no OPENROUTER_API_KEY (none of these subcommands need it).
"""
from __future__ import annotations

import pytest

from kicraft.server import accounts_cli
from kicraft.server.accounts import AccountStore


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KICRAFT_USERS_DB", str(tmp_path / "accounts.db"))
    monkeypatch.setenv("KICRAFT_PROJECTS_DIR", str(tmp_path / "projects"))

    def run(*argv):
        code = accounts_cli.main(list(argv))
        return code, capsys.readouterr()

    return run


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def test_create_admin_flag_grants_role(run_cli, store):
    code, out = run_cli("create", "boss@e.st", "--admin", "--password", "x")
    assert code == 0 and "role admin" in out.out
    assert store.get_user_by_email("boss@e.st").role == "admin"


def test_grant_and_revoke_admin(run_cli, store):
    run_cli("create", "keep@e.st", "--admin", "--password", "x")  # a 2nd admin, so the
    run_cli("create", "a@e.st", "--password", "x")                # revoke isn't "last admin"
    code, out = run_cli("grant-admin", "a@e.st")
    assert code == 0 and "is now an admin" in out.out
    assert store.get_user_by_email("a@e.st").role == "admin"
    code, out = run_cli("revoke-admin", "a@e.st")
    assert code == 0 and "no longer an admin" in out.out
    assert store.get_user_by_email("a@e.st").role == "user"


def test_revoke_last_admin_refused(run_cli, store):
    run_cli("create", "boss@e.st", "--admin", "--password", "x")
    code, out = run_cli("revoke-admin", "boss@e.st")
    assert code == 1 and "last admin" in out.err
    assert store.get_user_by_email("boss@e.st").role == "admin"  # unchanged


def test_set_tier_rejects_admin(run_cli, store):
    run_cli("create", "a@e.st", "--password", "x")
    with pytest.raises(SystemExit):  # argparse invalid-choice -> SystemExit(2)
        run_cli("set-tier", "a@e.st", "admin")


def test_list_shows_role_column(run_cli):
    run_cli("create", "boss@e.st", "--admin", "--password", "x")
    code, out = run_cli("list")
    assert code == 0 and "role" in out.out and "admin" in out.out
