"""Admin CLI for KiCraft accounts: list users, grant tiers/roles, seed accounts.

Until Stripe (backlog item 3) lands, tier changes are manual. Admin access is a
ROLE, orthogonal to the billing tier; grant it here to bootstrap the first admin
before the /admin dashboard is reachable (the dashboard itself requires an admin):

    kicraft-accounts list
    kicraft-accounts create alice@example.com --tier pro
    kicraft-accounts create boss@example.com --admin           # seed a staff admin
    kicraft-accounts set-tier alice@example.com max            # billing tier only
    kicraft-accounts grant-admin alice@example.com             # promote to admin
    kicraft-accounts revoke-admin alice@example.com            # demote to user
    kicraft-accounts reset-password alice@example.com          # print a reset link
    kicraft-accounts reset-password alice@example.com --set    # set a password now

Resolves the DB / projects paths from the same env vars the web app uses
(KICRAFT_USERS_DB, KICRAFT_PROJECTS_DIR), so most commands need no
OPENROUTER_API_KEY (only `reset-password --send`, which emails via SMTP, does).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
from pathlib import Path

from .accounts import _RESET_TTL_SECONDS, TIERS, AccountStore
from .config import load_dotenv


def _default_db() -> Path:
    return Path(os.environ.get(
        "KICRAFT_USERS_DB", str(Path.home() / ".kicraft" / "accounts.db")))


def _default_projects_dir() -> Path:
    return Path(os.environ.get(
        "KICRAFT_PROJECTS_DIR", str(Path.home() / ".kicraft" / "projects")))


def _default_public_url() -> str:
    """Origin used to build reset links, matching the web app's KICRAFT_PUBLIC_URL
    (so a link printed here resolves on the deployed site). Read directly, not via
    Settings, so the print path needs no OPENROUTER_API_KEY."""
    url = os.environ.get("KICRAFT_PUBLIC_URL", "http://localhost:8080").strip().rstrip("/")
    return url or "http://localhost:8080"


def _store() -> AccountStore:
    return AccountStore(_default_db(), _default_projects_dir())


def _cmd_list(args: argparse.Namespace) -> int:
    store = _store()
    users = store.list_users()
    if not users:
        print("(no users yet)")
        return 0
    print(f"{'id':>3}  {'email':<32} {'tier':<5} {'role':<6} {'projects':>8}  created")
    for u in users:
        n = len(store.list_projects(u.id))
        print(f"{u.id:>3}  {u.email:<32} {u.tier:<5} {u.role:<6} {n:>8}  "
              f"{u.created_at[:19]}")
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
    store = _store()
    try:
        u = store.create_user(args.email, pw, tier=args.tier)
        if args.admin:
            u = store.set_role(u.email, "admin")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"created {u.email} (tier {u.tier}, role {u.role}, id {u.id})")
    return 0


def _cmd_grant_admin(args: argparse.Namespace) -> int:
    try:
        u = _store().set_role(args.email, "admin")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"{u.email} is now an admin")
    return 0


def _cmd_revoke_admin(args: argparse.Namespace) -> int:
    store = _store()
    target = store.get_user_by_email(args.email)
    if target is None:
        print(f"no user with email {args.email!r}", file=sys.stderr)
        return 1
    # Last-admin guard mirrors the dashboard: never leave the system with zero
    # admins (that would force a CLI re-bootstrap to regain access).
    if target.role == "admin" and store.count_role("admin") <= 1:
        print("refusing to remove the last admin", file=sys.stderr)
        return 1
    try:
        u = store.set_role(args.email, "user")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"{u.email} is no longer an admin")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Export a user's account + project metadata and stored files (data request)."""
    store = _store()
    u = store.get_user_by_email(args.email)
    if u is None:
        print(f"no user with email {args.email!r}", file=sys.stderr)
        return 1
    data = store.export_user(u.id)
    out = Path(args.out) if args.out else Path(f"kicraft_export_{u.id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "account.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    src = store.projects_dir / str(u.id)
    n_files = 0
    if src.is_dir():
        dst = out / "projects"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        n_files = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"exported {u.email} to {out} "
          f"({len(data['projects'])} project rows, {n_files} files)")
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    """Delete a user, their project rows, and their stored files (deletion right)."""
    store = _store()
    u = store.get_user_by_email(args.email)
    if u is None:
        print(f"no user with email {args.email!r}", file=sys.stderr)
        return 1
    if not args.yes:
        print(f"refusing to delete {u.email} (id {u.id}) without --yes "
              "(this is irreversible)", file=sys.stderr)
        return 1
    purged = store.delete_user(u.id)
    tail = f"; removed {purged}" if purged else ""
    print(f"deleted {u.email} (id {u.id}) and their projects{tail}")
    return 0


def _cmd_reset_password(args: argparse.Namespace) -> int:
    """Recover an account: print a reset link, email it (--send), or set a new
    password now (--set). All paths that change the password evict existing
    sessions, since set_password bumps the user's session epoch."""
    load_dotenv()  # read KICRAFT_PUBLIC_URL (and any paths) from .env in the CWD
    store = _store()
    u = store.get_user_by_email(args.email)
    if u is None:
        print(f"no user with email {args.email!r}", file=sys.stderr)
        return 1

    if args.set:  # set a password directly, the guaranteed path when SMTP is down
        pw = getpass.getpass("new password: ")
        if not pw:
            print("aborted: empty password", file=sys.stderr)
            return 1
        if pw != getpass.getpass("confirm new password: "):
            print("aborted: passwords did not match", file=sys.stderr)
            return 1
        store.set_password(u.id, pw)
        print(f"password for {u.email} updated; all existing sessions were signed out")
        return 0

    token = store.create_reset_token(u.email)
    if token is None:
        print("a reset link was issued moments ago; wait a minute and retry",
              file=sys.stderr)
        return 1
    url = f"{_default_public_url()}/reset?token={token}"
    ttl_min = _RESET_TTL_SECONDS // 60

    if args.send:
        # Emailing needs Settings (SMTP config + an API key); import lazily so the
        # default print path above stays usable without OPENROUTER_API_KEY.
        from . import mailer
        from .config import Settings
        if mailer.send_reset_email(Settings.from_env(), u.email, url, ttl_min):
            print(f"reset link emailed to {u.email} (valid ~{ttl_min} min)")
            return 0
        print("could not send email (SMTP unconfigured or failed); relay this link "
              f"manually:\n  {url}", file=sys.stderr)
        return 1

    print(f"reset link for {u.email} (valid ~{ttl_min} min, single use):")
    print(f"  {url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="kicraft-accounts",
        description="Manage KiCraft web accounts and tiers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all users").set_defaults(func=_cmd_list)

    sp = sub.add_parser("set-tier",
                        help="change a user's billing tier (admin is a role: "
                             "see grant-admin)")
    sp.add_argument("email")
    sp.add_argument("tier", choices=sorted(TIERS), help="one of: " + ", ".join(TIERS))
    sp.set_defaults(func=_cmd_set_tier)

    cp = sub.add_parser("create", help="create a user (seed an admin or tester)")
    cp.add_argument("email")
    cp.add_argument("--tier", default="free", choices=sorted(TIERS),
                    help="default: free")
    cp.add_argument("--admin", action="store_true",
                    help="also grant the admin role (staff access)")
    cp.add_argument("--password", default=None,
                    help="set non-interactively; prompts securely if omitted")
    cp.set_defaults(func=_cmd_create)

    gp = sub.add_parser("grant-admin", help="grant a user the admin role")
    gp.add_argument("email")
    gp.set_defaults(func=_cmd_grant_admin)

    vp = sub.add_parser("revoke-admin", help="revoke a user's admin role")
    vp.add_argument("email")
    vp.set_defaults(func=_cmd_revoke_admin)

    ep = sub.add_parser("export",
                        help="export a user's account + projects (data/access request)")
    ep.add_argument("email")
    ep.add_argument("--out", default=None,
                    help="output dir (default: kicraft_export_<id>)")
    ep.set_defaults(func=_cmd_export)

    dp = sub.add_parser("delete",
                        help="delete a user, their projects, and stored data")
    dp.add_argument("email")
    dp.add_argument("--yes", action="store_true",
                    help="required: confirm irreversible deletion")
    dp.set_defaults(func=_cmd_delete)

    rp = sub.add_parser(
        "reset-password",
        help="recover an account: issue a reset link or set a new password")
    rp.add_argument("email")
    grp = rp.add_mutually_exclusive_group()
    grp.add_argument("--send", action="store_true",
                     help="email the link via SMTP (needs KICRAFT_SMTP_* and "
                          "OPENROUTER_API_KEY); otherwise the link is printed")
    grp.add_argument("--set", action="store_true",
                     help="set a new password now via a prompt (no email)")
    rp.set_defaults(func=_cmd_reset_password)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
