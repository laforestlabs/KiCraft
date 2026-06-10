"""Per-user accounts, projects, and tiered usage quotas for the KiCraft web app.

A small SQLite store mirroring spend_guard.SpendGuard's conventions (connection
per op, WAL, CREATE TABLE IF NOT EXISTS). It holds the user identities, their
saved projects, and the metering needed to enforce the per-tier design quotas.

Real payment is out of scope (backlog item 3); tiers are assigned by the admin
CLI (`kicraft-accounts set-tier`). Passwords are hashed with stdlib scrypt, so
there is no external dependency. The full design event stream is persisted to
disk per project by the web worker, not into this DB; this store keeps only
queryable metadata (the brief, status, cost, and artifact paths).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

# Billing tier definitions. `price_usd` is display-only until Stripe lands
# (backlog item 3); `limit` designs per rolling `window_days` is what
# count_active_designs enforces. "free" = 1/week, "pro" = 5/month ($5),
# "max" = 25/month ($10). Admin access is no longer a tier -- it is a separate
# `role` (see ROLES below), so a user can hold any billing tier and still be
# staff, and staff bypass the quota outright (see quota_status / can_design).
TIERS: dict[str, dict] = {
    "free": {"label": "Free", "price_usd": 0, "limit": 1, "window_days": 7},
    "pro": {"label": "Pro", "price_usd": 5, "limit": 5, "window_days": 30},
    "max": {"label": "Max", "price_usd": 10, "limit": 25, "window_days": 30},
}
DEFAULT_TIER = "free"

# Access roles, orthogonal to the billing tier above. Extensible: 'support' or
# 'superadmin' can be added here later without touching the call sites that gate
# on is_admin(). 'user' is the default everyone signs up with.
ROLES: frozenset[str] = frozenset({"user", "admin"})
DEFAULT_ROLE = "user"
ADMIN_ROLE = "admin"


def is_admin(user) -> bool:
    """Whether a user holds a staff role (currently 'admin') that unlocks the
    admin dashboard and the self-evaluation tools.

    Decoupled from the billing tier: grant it out-of-band with
    `kicraft-accounts grant-admin <email>` or from the /admin/users UI. Duck-typed
    on `.role` so a User or any object carrying a role works."""
    return bool(user is not None and getattr(user, "role", None) == ADMIN_ROLE)

# scrypt work factors (RFC 7914). Bounded so a hash is sub-100ms but not trivial.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1

# Password-reset tokens: short-lived and single-use. The window is deliberately
# tight (an hour) so a leaked link is useless after lunch; the cooldown stops the
# public /forgot page from being turned into an email-spam relay.
_RESET_TTL_SECONDS = 3600
_RESET_COOLDOWN_SECONDS = 60


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def hash_password(password: str) -> str:
    """Return a self-describing scrypt hash: scrypt$N$r$p$salt_hex$hash_hex."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a stored scrypt hash."""
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


@dataclass
class User:
    id: int
    email: str
    tier: str
    created_at: str
    last_login_at: str | None = None
    # Access role, orthogonal to the billing tier ('user' | 'admin'). Gated on by
    # is_admin(); granted out-of-band (CLI) or from the /admin/users dashboard.
    role: str = DEFAULT_ROLE
    # Consent + data-use preference (see docs/legal/). A user whose
    # accepted_terms_version is None or older than the current LEGAL_VERSION is
    # re-prompted to accept before they can continue. allow_training gates the
    # model-training use only (operate/analytics are not opt-out).
    accepted_terms_version: str | None = None
    accepted_terms_at: str | None = None
    allow_training: bool = True
    # Bumped on every password change (set_password). The web session stores the
    # epoch it logged in with; an authed page that sees a newer epoch on the row
    # force-logs-out, so a password reset evicts every other live session (the
    # point of recovering an account someone else got into).
    session_epoch: int = 0


@dataclass
class Project:
    id: int
    user_id: int
    brief: str
    project_stem: str | None
    status: str
    created_at: str
    finished_at: str | None
    cost_usd: float | None
    dir_path: str | None
    zip_path: str | None
    # When the owner last opened this project in the workspace. NULL on a
    # freshly finished run = "result not yet seen"; the workspace auto-opens it.
    viewed_at: str | None = None
    # Public-browser fields (community catalog). Defaults keep existing
    # constructors and tests valid; _row_to_project fills them from the row.
    is_public: bool = True
    cloned_from_id: int | None = None
    view_count: int = 0
    clone_count: int = 0
    like_count: int = 0
    quality: str | None = None  # 'fab_ready' | 'erc_errors' | 'unverified'


def build_fts_document(brief: str | None, state: dict | None) -> dict:
    """Flatten a project's brief + design state into the four search fields the
    FTS index stores (brief / goal / parts / blocks).

    Pure and tolerant: any missing slot becomes an empty string, so a half-built
    or malformed state.json never breaks indexing. A part query ("esp32") lands
    in `parts` (a BOM mpn/value) or `brief`; a function query ("plant watering")
    lands in `goal` or `blocks`. Stays stdlib-only to match this module.
    """
    state = state or {}
    intent = state.get("intent") or {}
    bom = state.get("bom") or {}
    fspec = state.get("functional_spec") or {}
    arch = state.get("architecture") or {}

    parts_bits: list = list(intent.get("named_parts") or [])
    for p in (bom.get("parts") or []):
        parts_bits += [p.get("mpn"), p.get("value"), p.get("sourcing_note")]
    block_bits: list = []
    for b in (fspec.get("blocks") or []):
        block_bits += [b.get("name"), b.get("purpose")]
    for s in (arch.get("sheets") or []):
        block_bits.append(s.get("function"))

    def _join(bits: list) -> str:
        return " ".join(str(x) for x in bits if x)

    return {
        "brief": brief or "",
        "goal": str(intent.get("goal") or ""),
        "parts": _join(parts_bits),
        "blocks": _join(block_bits),
    }


class AccountStore:
    """SQLite-backed users + projects + quota metering.

    Constructed from explicit paths so the admin CLI can use it without a full
    Settings (no OPENROUTER_API_KEY needed); web.py builds it from Settings.
    """

    def __init__(self, db_path: str | os.PathLike, projects_dir: str | os.PathLike):
        self.path = Path(db_path)
        self.projects_dir = Path(projects_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False  # flipped on by _init_db once projects_fts exists
        self._init_db()
        self._maybe_backfill_search()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users ("
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
                "session_epoch INTEGER NOT NULL DEFAULT 0)"
            )
            self._ensure_columns(conn)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS password_resets ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "user_id INTEGER NOT NULL,"
                "token_hash TEXT NOT NULL,"
                "created_at TEXT NOT NULL,"
                "expires_at TEXT NOT NULL,"
                "used_at TEXT,"
                "FOREIGN KEY(user_id) REFERENCES users(id))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_password_resets_token "
                "ON password_resets(token_hash)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS projects ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "user_id INTEGER NOT NULL,"
                "brief TEXT NOT NULL,"
                "project_stem TEXT,"
                "status TEXT NOT NULL DEFAULT 'running',"
                "created_at TEXT NOT NULL,"
                "finished_at TEXT,"
                "cost_usd REAL,"
                "dir_path TEXT,"
                "zip_path TEXT,"
                "viewed_at TEXT,"
                "is_public INTEGER NOT NULL DEFAULT 1,"
                "cloned_from_id INTEGER,"
                "view_count INTEGER NOT NULL DEFAULT 0,"
                "clone_count INTEGER NOT NULL DEFAULT 0,"
                "like_count INTEGER NOT NULL DEFAULT 0,"
                "quality TEXT,"
                "FOREIGN KEY(user_id) REFERENCES users(id))"
            )
            self._ensure_project_columns(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_projects_user_created "
                "ON projects(user_id, created_at)"
            )
            # Per-user dedup of likes; the count is denormalized onto
            # projects.like_count and kept in sync inside toggle_like's txn.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS project_likes ("
                "user_id INTEGER NOT NULL,"
                "project_id INTEGER NOT NULL,"
                "created_at TEXT NOT NULL,"
                "PRIMARY KEY (user_id, project_id),"
                "FOREIGN KEY(user_id) REFERENCES users(id),"
                "FOREIGN KEY(project_id) REFERENCES projects(id))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_project_likes_project "
                "ON project_likes(project_id)"
            )
            # Full-text search over the catalog. Guarded: a SQLite build without
            # FTS5 degrades to a LIKE fallback (see _public_where) instead of
            # crashing init. porter stemming makes "watering" match "water".
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS projects_fts USING fts5("
                    "project_id UNINDEXED, brief, goal, parts, blocks,"
                    "tokenize='porter unicode61')"
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        """Additively migrate an older users table.

        A DB created before a column was introduced has none of it; the
        CREATE TABLE IF NOT EXISTS above leaves it untouched. ALTER the missing
        columns in so an already-deployed box upgrades without losing its rows.
        Existing rows get NULL consent (so they are re-prompted) and allow_training
        defaults to 1. When the `role` column is first introduced, any user still
        on the retired 'admin' billing tier is promoted to the admin role and reset
        to the free tier (see the inline note for why the backfill is idempotent).
        """
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "accepted_terms_version" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN accepted_terms_version TEXT")
        if "accepted_terms_at" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN accepted_terms_at TEXT")
        if "allow_training" not in cols:
            conn.execute(
                "ALTER TABLE users ADD COLUMN allow_training INTEGER NOT NULL DEFAULT 1")
        if "session_epoch" not in cols:
            conn.execute(
                "ALTER TABLE users ADD COLUMN session_epoch INTEGER NOT NULL DEFAULT 0")
        if "role" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            # One-time backfill: the retired 'admin' billing tier becomes the admin
            # ROLE, so a deployed operator account keeps staff access across the
            # upgrade. Promote the role first, then clear the tier. Guarded by the
            # ADD COLUMN above (runs only when the column is introduced) and ordered
            # so that even a re-run matches nothing -- there are no tier='admin' rows
            # left after the second statement, so it is idempotent.
            conn.execute("UPDATE users SET role='admin' WHERE tier='admin'")
            conn.execute("UPDATE users SET tier='free' WHERE tier='admin'")

    @staticmethod
    def _ensure_project_columns(conn: sqlite3.Connection) -> None:
        """Additively migrate an older projects table for the public browser.

        Mirrors _ensure_columns: a DB created before the browser feature lacks
        these columns; the CREATE TABLE IF NOT EXISTS leaves an existing table
        untouched, so ALTER the missing columns in. The is_public default of 1
        enacts the product rule that free users' projects are public; the
        one-time backfill (guarded by the ADD COLUMN, so it runs only when the
        column is first introduced) then flips paid users' EXISTING projects back
        to private, so we never retroactively expose paid work without consent.
        Idempotent: a re-open finds the column present and skips the whole block.
        """
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(projects)")}
        if "is_public" not in cols:
            conn.execute(
                "ALTER TABLE projects ADD COLUMN is_public INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                "UPDATE projects SET is_public=0 WHERE user_id IN "
                "(SELECT id FROM users WHERE tier IN ('pro','max'))")
        if "cloned_from_id" not in cols:
            conn.execute("ALTER TABLE projects ADD COLUMN cloned_from_id INTEGER")
        if "view_count" not in cols:
            conn.execute(
                "ALTER TABLE projects ADD COLUMN view_count INTEGER NOT NULL DEFAULT 0")
        if "clone_count" not in cols:
            conn.execute(
                "ALTER TABLE projects ADD COLUMN clone_count INTEGER NOT NULL DEFAULT 0")
        if "like_count" not in cols:
            conn.execute(
                "ALTER TABLE projects ADD COLUMN like_count INTEGER NOT NULL DEFAULT 0")
        if "quality" not in cols:
            conn.execute("ALTER TABLE projects ADD COLUMN quality TEXT")
        if "viewed_at" not in cols:
            conn.execute("ALTER TABLE projects ADD COLUMN viewed_at TEXT")
            # One-time backfill: anything already finished counts as seen, so the
            # workspace's "auto-open your newest unseen result" default only
            # fires for runs that finish after this column ships.
            conn.execute("UPDATE projects SET viewed_at=finished_at "
                         "WHERE finished_at IS NOT NULL")

    def _maybe_backfill_search(self) -> None:
        """One-time: populate the FTS index the first time it appears on a DB that
        already has public completed projects (e.g. the live box on upgrade). Gated
        on the index being empty, so it runs once and a normal restart is a no-op."""
        if not self._fts_enabled:
            return
        with self._conn() as conn:
            if conn.execute("SELECT COUNT(*) FROM projects_fts").fetchone()[0]:
                return
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM projects WHERE status='ok' AND is_public=1")]
        for pid in ids:
            try:
                self.reindex_search(pid)
            except Exception:
                pass  # best-effort: a single bad state.json never blocks startup

    # ---- users ------------------------------------------------------------

    @staticmethod
    def _norm_email(email: str) -> str:
        return (email or "").strip().lower()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(id=row["id"], email=row["email"], tier=row["tier"],
                    created_at=row["created_at"], last_login_at=row["last_login_at"],
                    role=row["role"],
                    accepted_terms_version=row["accepted_terms_version"],
                    accepted_terms_at=row["accepted_terms_at"],
                    allow_training=bool(row["allow_training"]),
                    session_epoch=int(row["session_epoch"]))

    def create_user(self, email: str, password: str, tier: str = DEFAULT_TIER, *,
                    accepted_terms_version: str | None = None,
                    allow_training: bool = True) -> User:
        em = self._norm_email(email)
        if not em or "@" not in em:
            raise ValueError("a valid email is required")
        if not password:
            raise ValueError("a password is required")
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        now = _utcnow_iso()
        accepted_at = now if accepted_terms_version else None
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO users (email, password_hash, tier, created_at, "
                    "accepted_terms_version, accepted_terms_at, allow_training) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (em, hash_password(password), tier, now,
                     accepted_terms_version, accepted_at, 1 if allow_training else 0))
                uid = cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise ValueError(f"email {em!r} is already registered") from e
        return User(id=int(uid), email=em, tier=tier, created_at=now,
                    role=DEFAULT_ROLE,
                    accepted_terms_version=accepted_terms_version,
                    accepted_terms_at=accepted_at, allow_training=allow_training)

    def get_user(self, user_id: int) -> User | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=?",
                               (self._norm_email(email),)).fetchone()
        return self._row_to_user(row) if row else None

    def authenticate(self, email: str, password: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=?",
                               (self._norm_email(email),)).fetchone()
            if not row or not verify_password(password, row["password_hash"]):
                return None
            conn.execute("UPDATE users SET last_login_at=? WHERE id=?",
                         (_utcnow_iso(), row["id"]))
        return self._row_to_user(row)

    def set_tier(self, email: str, tier: str) -> User:
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; choose from {', '.join(TIERS)}")
        em = self._norm_email(email)
        with self._conn() as conn:
            cur = conn.execute("UPDATE users SET tier=? WHERE email=?", (tier, em))
            if cur.rowcount == 0:
                raise ValueError(f"no user with email {email!r}")
            row = conn.execute("SELECT * FROM users WHERE email=?", (em,)).fetchone()
        return self._row_to_user(row)

    def set_role(self, email_or_id, role: str) -> User:
        """Grant or revoke an access role, addressing the user by email (str) or id
        (int). Unlike set_password this does NOT bump session_epoch: a role change
        takes effect on the target's next page load and does not force them out of
        their existing session."""
        if role not in ROLES:
            raise ValueError(
                f"unknown role {role!r}; choose from {', '.join(sorted(ROLES))}")
        with self._conn() as conn:
            if isinstance(email_or_id, int):
                cur = conn.execute("UPDATE users SET role=? WHERE id=?",
                                   (role, email_or_id))
                sel = ("SELECT * FROM users WHERE id=?", (email_or_id,))
            else:
                em = self._norm_email(email_or_id)
                cur = conn.execute("UPDATE users SET role=? WHERE email=?",
                                   (role, em))
                sel = ("SELECT * FROM users WHERE email=?", (em,))
            if cur.rowcount == 0:
                raise ValueError(f"no user matching {email_or_id!r}")
            row = conn.execute(*sel).fetchone()
        return self._row_to_user(row)

    def count_role(self, role: str) -> int:
        """How many users hold `role`. Backs the last-admin lockout guard so the
        system is never left with zero admins."""
        with self._conn() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM users WHERE role=?", (role,)).fetchone()[0])

    def list_users(self) -> list[User]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [self._row_to_user(r) for r in rows]

    # ---- password reset + account recovery -------------------------------

    @staticmethod
    def _hash_token(token: str) -> str:
        """SHA-256 of a reset token. Tokens are 256-bit random, so a fast hash is
        enough to make the stored value useless if the DB leaks (no rainbow-table
        risk at that entropy); we never store the raw token."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def set_password(self, user_id: int, new_password: str) -> None:
        """Set a new password and evict every existing session.

        The single chokepoint for changing a password: bumping session_epoch in
        the same statement force-logs-out any other live session (an attacker who
        still holds a valid cookie included), which is what makes this recovery and
        not just a password swap. Used by the reset flow and the admin CLI."""
        if not new_password:
            raise ValueError("a password is required")
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash=?, session_epoch=session_epoch+1 "
                "WHERE id=?", (hash_password(new_password), user_id))
            if cur.rowcount == 0:
                raise ValueError(f"no user with id {user_id!r}")

    def create_reset_token(self, email: str) -> str | None:
        """Mint a single-use, time-limited password-reset token for `email`.

        Returns the raw token for the caller to deliver out-of-band (only its hash
        is stored). Returns None when no such user exists, so the caller can stay
        silent and not leak which emails are registered; also returns None inside
        the cooldown window, so the public /forgot page can't be used as an
        email-spam relay. Minting a token invalidates the user's prior unused ones.
        """
        user = self.get_user_by_email(email)
        if user is None:
            return None
        now = _utcnow()
        with self._conn() as conn:
            recent = conn.execute(
                "SELECT created_at FROM password_resets WHERE user_id=? "
                "AND used_at IS NULL ORDER BY id DESC LIMIT 1", (user.id,)).fetchone()
            if recent is not None:
                try:
                    created = dt.datetime.fromisoformat(recent["created_at"])
                    if (now - created).total_seconds() < _RESET_COOLDOWN_SECONDS:
                        return None  # a fresh link is already in flight
                except ValueError:
                    pass  # unparseable timestamp: fall through and mint a new one
            token = secrets.token_urlsafe(32)
            expires = (now + dt.timedelta(seconds=_RESET_TTL_SECONDS)).isoformat()
            conn.execute(
                "UPDATE password_resets SET used_at=? WHERE user_id=? "
                "AND used_at IS NULL", (now.isoformat(), user.id))
            conn.execute(
                "INSERT INTO password_resets "
                "(user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (user.id, self._hash_token(token), now.isoformat(), expires))
        return token

    def verify_reset_token(self, token: str) -> User | None:
        """The user a valid (unused, unexpired) reset token belongs to, else None.

        Read-only: use it to render the reset form before the user has chosen a new
        password. ISO-8601 UTC timestamps compare lexicographically."""
        if not token:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id FROM password_resets WHERE token_hash=? "
                "AND used_at IS NULL AND expires_at >= ?",
                (self._hash_token(token), _utcnow_iso())).fetchone()
        return self.get_user(int(row["user_id"])) if row else None

    def consume_reset_token(self, token: str, new_password: str) -> User | None:
        """Spend a reset token: set the new password and mark the token used, in one
        transaction. Returns the refreshed user (session_epoch already bumped, so
        other sessions are now evicted) or None if the token is invalid, expired, or
        already used. Raises ValueError on an empty password (before touching the
        token, so the user can retry the link)."""
        if not new_password:
            raise ValueError("a password is required")
        if not token:
            return None
        now = _utcnow().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, user_id FROM password_resets WHERE token_hash=? "
                "AND used_at IS NULL AND expires_at >= ?",
                (self._hash_token(token), now)).fetchone()
            if row is None:
                return None
            user_id = int(row["user_id"])
            # Inlined (not set_password) so the password change and the token
            # consumption commit atomically in this one transaction.
            conn.execute(
                "UPDATE users SET password_hash=?, session_epoch=session_epoch+1 "
                "WHERE id=?", (hash_password(new_password), user_id))
            conn.execute("UPDATE password_resets SET used_at=? WHERE id=?",
                         (now, row["id"]))
        return self.get_user(user_id)

    # ---- consent + data controls -----------------------------------------

    def record_consent(self, user_id: int, version: str) -> None:
        """Stamp acceptance of a Terms version (signup, or re-consent on a bump)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET accepted_terms_version=?, accepted_terms_at=? "
                "WHERE id=?", (version, _utcnow_iso(), user_id))

    def set_training_pref(self, user_id: int, allow: bool) -> None:
        """Toggle the model-training opt-out (Terms 5c / Privacy 5)."""
        with self._conn() as conn:
            conn.execute("UPDATE users SET allow_training=? WHERE id=?",
                         (1 if allow else 0, user_id))

    def export_user(self, user_id: int) -> dict | None:
        """A JSON-able copy of the user's account + project metadata (no password
        hash). The on-disk project tree is copied separately by the admin CLI."""
        user = self.get_user(user_id)
        if user is None:
            return None
        return {
            "exported_at": _utcnow_iso(),
            "user": asdict(user),
            "projects": [asdict(p) for p in self.list_projects(user_id)],
            "projects_dir": str(self.projects_dir / str(user_id)),
        }

    def delete_user(self, user_id: int) -> str | None:
        """Delete the user, their project rows, and their on-disk project tree.

        Honors the deletion right the Privacy Policy promises. Returns the
        filesystem path that was purged (for logging), or None if there was none.
        """
        with self._conn() as conn:
            pids = [r["id"] for r in conn.execute(
                "SELECT id FROM projects WHERE user_id=?", (user_id,))]
            # Drop likes this user gave, plus likes + FTS rows on their projects,
            # before the projects themselves so nothing dangles in the catalog.
            conn.execute("DELETE FROM project_likes WHERE user_id=?", (user_id,))
            for pid in pids:
                conn.execute("DELETE FROM project_likes WHERE project_id=?", (pid,))
                if self._fts_enabled:
                    conn.execute("DELETE FROM projects_fts WHERE project_id=?", (pid,))
            conn.execute("DELETE FROM projects WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        tree = self.projects_dir / str(user_id)
        if tree.exists():
            shutil.rmtree(tree, ignore_errors=True)
            return str(tree)
        return None

    # ---- projects ---------------------------------------------------------

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project(id=row["id"], user_id=row["user_id"], brief=row["brief"],
                       project_stem=row["project_stem"], status=row["status"],
                       created_at=row["created_at"], finished_at=row["finished_at"],
                       cost_usd=row["cost_usd"], dir_path=row["dir_path"],
                       zip_path=row["zip_path"], viewed_at=row["viewed_at"],
                       is_public=bool(row["is_public"]),
                       cloned_from_id=row["cloned_from_id"],
                       view_count=row["view_count"], clone_count=row["clone_count"],
                       like_count=row["like_count"], quality=row["quality"])

    def create_project(self, user_id: int, brief: str, *,
                       is_public: bool | None = None) -> int:
        """Reserve a project row at status 'running' (consumes a quota slot).

        Visibility follows the owner's tier when not given explicitly: free users'
        projects are public (the community-browser rule), paid (pro/max) users'
        projects default private (they can publish later via set_visibility). Pass
        is_public to override -- the clone path sets it directly for either tier."""
        if is_public is None:
            u = self.get_user(user_id)
            is_public = (u is None) or (u.tier not in ("pro", "max"))
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO projects (user_id, brief, status, created_at, is_public) "
                "VALUES (?, ?, 'running', ?, ?)",
                (user_id, brief, _utcnow_iso(), 1 if is_public else 0))
            return int(cur.lastrowid)

    def finish_project(self, project_id: int, status: str, stem: str | None = None,
                       cost_usd: float | None = None, dir_path: str | None = None,
                       zip_path: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE projects SET status=?, project_stem=?, cost_usd=?, "
                "dir_path=?, zip_path=?, finished_at=? WHERE id=?",
                (status, stem, cost_usd, dir_path, zip_path, _utcnow_iso(), project_id))

    def update_project_status(self, project_id: int, status: str) -> None:
        """Set just the status (e.g. 'awaiting_input' when a run parks on a
        question, or back to 'running' when it resumes). Leaves artifacts intact.
        Moving back to 'running' means a new result is in the making, so the
        seen-marker resets and the workspace will auto-open the eventual result."""
        with self._conn() as conn:
            if status == "running":
                conn.execute("UPDATE projects SET status=?, viewed_at=NULL WHERE id=?",
                             (status, project_id))
            else:
                conn.execute("UPDATE projects SET status=? WHERE id=?",
                             (status, project_id))

    def mark_viewed(self, project_id: int) -> None:
        """Stamp that the owner has the project open in the workspace; the index
        page's auto-open then stops treating its result as unseen."""
        with self._conn() as conn:
            conn.execute("UPDATE projects SET viewed_at=? WHERE id=?",
                         (_utcnow_iso(), project_id))

    def get_project(self, project_id: int) -> Project | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id=?",
                               (project_id,)).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self, user_id: int) -> list[Project]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM projects WHERE user_id=? ORDER BY id DESC",
                (user_id,)).fetchall()
        return [self._row_to_project(r) for r in rows]

    # ---- public browser: visibility, metrics, likes, clone ----------------

    def set_visibility(self, project_id: int, is_public: bool) -> None:
        """Set a project's catalog visibility. Mechanism only: the web layer
        enforces that only paid (pro/max) users may make a project private (a free
        user's project must stay public). Reindex search after calling this."""
        with self._conn() as conn:
            conn.execute("UPDATE projects SET is_public=? WHERE id=?",
                         (1 if is_public else 0, project_id))

    def set_quality(self, project_id: int, quality: str | None) -> None:
        """Stamp the precomputed quality badge ('fab_ready' | 'erc_errors' |
        'unverified') so the catalog can filter/sort in SQL without re-reading the
        synthesis check per row."""
        with self._conn() as conn:
            conn.execute("UPDATE projects SET quality=? WHERE id=?",
                         (quality, project_id))

    def record_view(self, project_id: int) -> None:
        """Increment the view counter. Per-session dedup is the web layer's job."""
        with self._conn() as conn:
            conn.execute("UPDATE projects SET view_count=view_count+1 WHERE id=?",
                         (project_id,))

    def has_liked(self, user_id: int, project_id: int) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM project_likes WHERE user_id=? AND project_id=?",
                (user_id, project_id)).fetchone() is not None

    def toggle_like(self, user_id: int, project_id: int) -> bool:
        """Like or unlike, returning the resulting state (True = now liked). The
        like row and the denormalized projects.like_count move together in one
        transaction, so the count can never drift from the rows. A failed INSERT
        (already liked) only rolls back that statement, leaving the txn live for
        the unlike path."""
        with self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO project_likes (user_id, project_id, created_at) "
                    "VALUES (?, ?, ?)", (user_id, project_id, _utcnow_iso()))
                conn.execute("UPDATE projects SET like_count=like_count+1 WHERE id=?",
                             (project_id,))
                return True
            except sqlite3.IntegrityError:
                conn.execute(
                    "DELETE FROM project_likes WHERE user_id=? AND project_id=?",
                    (user_id, project_id))
                conn.execute(
                    "UPDATE projects SET like_count=MAX(0, like_count-1) WHERE id=?",
                    (project_id,))
                return False

    def set_cloned_from(self, project_id: int, source_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE projects SET cloned_from_id=? WHERE id=?",
                         (source_id, project_id))

    def increment_clone_count(self, project_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE projects SET clone_count=clone_count+1 WHERE id=?",
                         (project_id,))

    # ---- public browser: search index -------------------------------------

    def _load_project_state(self, project: Project) -> dict | None:
        """Read a project's persisted design state for indexing. The web worker
        writes a top-level state.json copy next to brief.txt; fall back to the
        copy inside the kicraft/ subtree. Returns None if neither is readable."""
        if not project or not project.dir_path:
            return None
        base = Path(project.dir_path)
        for cand in (base / "state.json", base / "kicraft" / "state.json"):
            if cand.is_file():
                try:
                    return json.loads(cand.read_text())
                except (json.JSONDecodeError, OSError):
                    return None
        return None

    def reindex_search(self, project_id: int) -> None:
        """Refresh a project's FTS row. A project that is not BOTH public and 'ok'
        is removed from the index, so private/failed projects are never searchable.
        No-op when FTS5 is unavailable."""
        if not self._fts_enabled:
            return
        p = self.get_project(project_id)
        if p is None or p.status != "ok" or not p.is_public:
            self.remove_from_search(project_id)
            return
        doc = build_fts_document(p.brief, self._load_project_state(p))
        with self._conn() as conn:
            conn.execute("DELETE FROM projects_fts WHERE project_id=?", (project_id,))
            conn.execute(
                "INSERT INTO projects_fts (project_id, brief, goal, parts, blocks) "
                "VALUES (?, ?, ?, ?, ?)",
                (project_id, doc["brief"], doc["goal"], doc["parts"], doc["blocks"]))

    def remove_from_search(self, project_id: int) -> None:
        if not self._fts_enabled:
            return
        with self._conn() as conn:
            conn.execute("DELETE FROM projects_fts WHERE project_id=?", (project_id,))

    def backfill_search(self) -> int:
        """Index every public, completed project. Returns the count indexed. Safe
        to re-run; reindex_search is idempotent per project."""
        if not self._fts_enabled:
            return 0
        with self._conn() as conn:
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM projects WHERE status='ok' AND is_public=1")]
        for pid in ids:
            self.reindex_search(pid)
        return len(ids)

    # ---- public browser: catalog query ------------------------------------

    _PUBLIC_SORTS = {
        "popularity": "(4*p.clone_count + 3*p.like_count + p.view_count) DESC, p.id DESC",
        "new": "p.finished_at DESC, p.id DESC",
        "clones": "p.clone_count DESC, p.id DESC",
    }

    @staticmethod
    def _fts_match_query(query: str | None) -> str | None:
        """Turn a raw search-box string into a safe FTS5 MATCH expression.

        Each whitespace-separated term is reduced to word characters (plus the
        ./-/_ common in MPNs) and emitted as a double-quoted prefix phrase, so no
        FTS operator (", *, :, ^, parentheses, NEAR) can leak through and raise
        OperationalError. A term with no alphanumeric content is dropped (it would
        tokenize to an empty phrase, which IS a syntax error); if nothing usable
        remains, returns None and the caller applies no text filter."""
        terms = []
        for raw in (query or "").split():
            cleaned = re.sub(r"[^0-9A-Za-z._-]+", " ", raw).strip()
            if any(ch.isalnum() for ch in cleaned):
                terms.append(f'"{cleaned}"*')
        return " ".join(terms) if terms else None

    def _public_where(self, query: str | None, badge: str | None):
        """Shared FROM/WHERE for the catalog: publish gate (public + ok) plus an
        optional full-text match and quality badge. Returns (join, where, params).
        Falls back to a brief LIKE scan when FTS5 is unavailable."""
        where = ["p.status='ok'", "p.is_public=1"]
        params: list = []
        join = ""
        if query and self._fts_enabled:
            match = self._fts_match_query(query)
            if match is not None:
                join = "JOIN projects_fts ON projects_fts.project_id = p.id"
                where.append("projects_fts MATCH ?")
                params.append(match)
        elif query:  # FTS unavailable: best-effort substring match on the brief
            where.append("p.brief LIKE ?")
            params.append(f"%{query}%")
        if badge:
            where.append("p.quality=?")
            params.append(badge)
        return join, " AND ".join(where), params

    def list_public_projects(self, *, sort: str = "popularity", query: str | None = None,
                             badge: str | None = None, limit: int = 24,
                             offset: int = 0) -> list[dict]:
        """The catalog query. Returns dicts (every project column + owner_email),
        gated to public completed projects, optionally full-text filtered by
        `query` (a part or a function) and by quality `badge`, ordered by the
        `sort` preset ('popularity' | 'new' | 'clones')."""
        join, where, params = self._public_where(query, badge)
        order = self._PUBLIC_SORTS.get(sort, self._PUBLIC_SORTS["popularity"])
        sql = (f"SELECT p.*, u.email AS owner_email FROM projects p "
               f"JOIN users u ON u.id = p.user_id {join} "
               f"WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?")
        with self._conn() as conn:
            rows = conn.execute(sql, (*params, limit, offset)).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def count_public_projects(self, *, query: str | None = None,
                              badge: str | None = None) -> int:
        join, where, params = self._public_where(query, badge)
        sql = f"SELECT COUNT(*) FROM projects p {join} WHERE {where}"
        with self._conn() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    # ---- quota ------------------------------------------------------------

    def count_active_designs(self, user_id: int, window_days: int) -> int:
        """Designs that consume quota in the trailing window: a started run
        ('running'), a run parked on a clarifying question ('awaiting_input'),
        and a success ('ok') each hold a slot; a 'failed' build frees it.
        ISO-8601 UTC timestamps compare lexicographically."""
        cutoff = (_utcnow() - dt.timedelta(days=window_days)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM projects WHERE user_id=? "
                "AND status IN ('running','ok','awaiting_input') AND created_at >= ?",
                (user_id, cutoff)).fetchone()
        return int(row[0] or 0)

    def quota_status(self, user: User) -> dict:
        tier = TIERS.get(user.tier, TIERS[DEFAULT_TIER])
        used = self.count_active_designs(user.id, tier["window_days"])
        if is_admin(user):
            # Staff bypass the quota entirely so an operator is never blocked; this
            # replaces the retired admin-tier limit=1000 trick. `unlimited` lets the
            # UI render "Unlimited" instead of doing arithmetic on the None limit.
            return {
                "tier": user.tier,
                "label": tier["label"],
                "price_usd": tier["price_usd"],
                "limit": None,
                "window_days": tier["window_days"],
                "used": used,
                "remaining": float("inf"),
                "unlimited": True,
            }
        return {
            "tier": user.tier,
            "label": tier["label"],
            "price_usd": tier["price_usd"],
            "limit": tier["limit"],
            "window_days": tier["window_days"],
            "used": used,
            "remaining": max(0, tier["limit"] - used),
            "unlimited": False,
        }

    def can_design(self, user: User) -> bool:
        return is_admin(user) or self.quota_status(user)["remaining"] > 0

    # ---- admin stats ------------------------------------------------------

    def overview_stats(self, *, window_days: int = 30) -> dict:
        """Headline counts for the admin dashboard in a handful of aggregate
        queries (no per-user fan-out). `*_new` counts rows created in the trailing
        window; cost/latency aggregates ignore NULLs (free or unfinished runs) and
        come back None when there is nothing to average."""
        cutoff = (_utcnow() - dt.timedelta(days=window_days)).isoformat()
        with self._conn() as conn:
            users_total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            users_new = conn.execute(
                "SELECT COUNT(*) FROM users WHERE created_at >= ?", (cutoff,)).fetchone()[0]
            admins = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role=?", (ADMIN_ROLE,)).fetchone()[0]
            proj_total = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            proj_new = conn.execute(
                "SELECT COUNT(*) FROM projects WHERE created_at >= ?", (cutoff,)).fetchone()[0]
            spend = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0), AVG(cost_usd) "
                "FROM projects WHERE cost_usd IS NOT NULL").fetchone()
            latency = conn.execute(
                "SELECT AVG((julianday(finished_at) - julianday(created_at)) * 86400.0) "
                "FROM projects WHERE finished_at IS NOT NULL").fetchone()[0]
        return {
            "users_total": int(users_total),
            "users_new": int(users_new),
            "admins": int(admins),
            "projects_total": int(proj_total),
            "projects_new": int(proj_new),
            "spend_total_usd": float(spend[0] or 0.0),
            "spend_avg_usd": (float(spend[1]) if spend[1] is not None else None),
            "avg_latency_s": (float(latency) if latency is not None else None),
            "window_days": window_days,
        }

    def tier_distribution(self) -> list[tuple[str, int]]:
        """(tier, count) over all users, busiest first."""
        with self._conn() as conn:
            return [(r["tier"], int(r["n"])) for r in conn.execute(
                "SELECT tier, COUNT(*) AS n FROM users GROUP BY tier ORDER BY n DESC")]

    def status_distribution(self) -> list[tuple[str, int]]:
        """(status, count) over all projects (running/ok/awaiting_input/failed)."""
        with self._conn() as conn:
            return [(r["status"], int(r["n"])) for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM projects "
                "GROUP BY status ORDER BY n DESC")]

    def _per_day(self, table: str, expr: str, days: int) -> list[tuple[str, float]]:
        """Shared YYYY-MM-DD time-series helper. `expr` is the per-day aggregate
        (e.g. 'COUNT(*)' or 'COALESCE(SUM(cost_usd),0)'). Buckets on
        substr(created_at,1,10), valid because created_at is ISO-8601 UTC (it slices
        to a calendar day and compares lexicographically). `table`/`expr` are
        code-controlled constants, never user input -- no injection surface."""
        cutoff = (_utcnow() - dt.timedelta(days=days)).date().isoformat()
        sql = (f"SELECT substr(created_at, 1, 10) AS d, {expr} AS v FROM {table} "
               "WHERE substr(created_at, 1, 10) >= ? GROUP BY d ORDER BY d")
        with self._conn() as conn:
            return [(r["d"], r["v"]) for r in conn.execute(sql, (cutoff,))]

    def signups_per_day(self, days: int = 30) -> list[tuple[str, int]]:
        return [(d, int(v)) for d, v in self._per_day("users", "COUNT(*)", days)]

    def projects_per_day(self, days: int = 30) -> list[tuple[str, int]]:
        return [(d, int(v)) for d, v in self._per_day("projects", "COUNT(*)", days)]

    def spend_per_day(self, days: int = 30) -> list[tuple[str, float]]:
        return [(d, float(v or 0.0)) for d, v in
                self._per_day("projects", "COALESCE(SUM(cost_usd), 0)", days)]

    def users_with_project_counts(self) -> list[dict]:
        """One row per user with project_count, total spend, and most-recent project
        time, via a single LEFT JOIN (keeps zero-project users; no N+1). Backs the
        /admin/users table and the overview 'top users' panel. Newest users first."""
        sql = (
            "SELECT u.id, u.email, u.tier, u.role, u.created_at, u.last_login_at, "
            "COUNT(p.id) AS project_count, "
            "COALESCE(SUM(p.cost_usd), 0) AS spend_usd, "
            "MAX(p.created_at) AS last_project_at "
            "FROM users u LEFT JOIN projects p ON p.user_id = u.id "
            "GROUP BY u.id ORDER BY u.created_at DESC")
        with self._conn() as conn:
            return [{k: r[k] for k in r.keys()} for r in conn.execute(sql).fetchall()]
