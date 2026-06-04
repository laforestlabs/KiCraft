"""Admin CLI for KiCraft accounts: list users, grant tiers, seed accounts.

Until Stripe (backlog item 3) lands, tier changes are manual:

    kicraft-accounts list
    kicraft-accounts create alice@example.com --tier pro
    kicraft-accounts set-tier alice@example.com max

Resolves the DB / projects paths from the same env vars the web app uses
(KICRAFT_USERS_DB, KICRAFT_PROJECTS_DIR), so it needs no OPENROUTER_API_KEY.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from .accounts import TIERS, AccountStore


def _default_db() -> Path:
    return Path(os.environ.get(
        "KICRAFT_USERS_DB", str(Path.home() / ".kicraft" / "accounts.db")))


def _default_projects_dir() -> Path:
    return Path(os.environ.get(
        "KICRAFT_PROJECTS_DIR", str(Path.home() / ".kicraft" / "projects")))


def _store() -> AccountStore:
    return AccountStore(_default_db(), _default_projects_dir())


def _cmd_list(args: argparse.Namespace) -> int:
    store = _store()
    users = store.list_users()
    if not users:
        print("(no users yet)")
        return 0
    print(f"{'id':>3}  {'email':<32} {'tier':<5} {'projects':>8}  created")
    for u in users:
        n = len(store.list_projects(u.id))
        print(f"{u.id:>3}  {u.email:<32} {u.tier:<5} {n:>8}  {u.created_at[:19]}")
    return 0


def _cmd_set_tier(args: argparse.Namespace) -> int:
    try:
        u = _store().set_tier(args.email, args.tier)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"{u.email} is now on the {u.tier} tier")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    pw = args.password or getpass.getpass("password: ")
    try:
        u = _store().create_user(args.email, pw, tier=args.tier)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"created {u.email} (tier {u.tier}, id {u.id})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="kicraft-accounts",
        description="Manage KiCraft web accounts and tiers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all users").set_defaults(func=_cmd_list)

    sp = sub.add_parser("set-tier", help="change a user's tier")
    sp.add_argument("email")
    sp.add_argument("tier", choices=sorted(TIERS), help="one of: " + ", ".join(TIERS))
    sp.set_defaults(func=_cmd_set_tier)

    cp = sub.add_parser("create", help="create a user (seed an admin or tester)")
    cp.add_argument("email")
    cp.add_argument("--tier", default="free", choices=sorted(TIERS),
                    help="default: free")
    cp.add_argument("--password", default=None,
                    help="set non-interactively; prompts securely if omitted")
    cp.set_defaults(func=_cmd_create)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
