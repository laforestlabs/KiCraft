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
import os
import secrets
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

# Tier definitions. `price_usd` is display-only until Stripe lands (backlog item
# 3); `limit` designs per rolling `window_days` is what count_active_designs
# enforces. "free" = 1/week, "pro" = 5/month ($5), "max" = 25/month ($10).
# "admin" is an internal access level (not purchasable) that unlocks the self-
# evaluation tools; it is assigned out-of-band via the admin CLI and carries a
# high design limit so an operator is never quota-blocked while testing.
TIERS: dict[str, dict] = {
    "free": {"label": "Free", "price_usd": 0, "limit": 1, "window_days": 7},
    "pro": {"label": "Pro", "price_usd": 5, "limit": 5, "window_days": 30},
    "max": {"label": "Max", "price_usd": 10, "limit": 25, "window_days": 30},
    "admin": {"label": "Admin", "price_usd": 0, "limit": 1000, "window_days": 30},
}
DEFAULT_TIER = "free"
ADMIN_TIER = "admin"


def is_admin(user) -> bool:
    """Whether a user holds the admin level that unlocks the self-evaluation tools.

    Gating is tier-based (no separate column): grant it out-of-band with
    `kicraft-accounts set-tier <email> admin`. Duck-typed on `.tier` so a User or
    any object carrying a tier works."""
    return bool(user is not None and getattr(user, "tier", None) == ADMIN_TIER)

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


class AccountStore:
    """SQLite-backed users + projects + quota metering.

    Constructed from explicit paths so the admin CLI can use it without a full
    Settings (no OPENROUTER_API_KEY needed); web.py builds it from Settings.
    """

    def __init__(self, db_path: str | os.PathLike, projects_dir: str | os.PathLike):
        self.path = Path(db_path)
        self.projects_dir = Path(projects_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                "FOREIGN KEY(user_id) REFERENCES users(id))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_projects_user_created "
                "ON projects(user_id, created_at)"
            )

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        """Additively migrate a pre-consent users table.

        A DB created before consent tracking has none of the consent columns; the
        CREATE TABLE IF NOT EXISTS above leaves it untouched. ALTER the missing
        columns in so an already-deployed box upgrades without losing its rows.
        Existing rows get NULL consent (so they are re-prompted) and allow_training
        defaults to 1.
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

    # ---- users ------------------------------------------------------------

    @staticmethod
    def _norm_email(email: str) -> str:
        return (email or "").strip().lower()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(id=row["id"], email=row["email"], tier=row["tier"],
                    created_at=row["created_at"], last_login_at=row["last_login_at"],
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
                       zip_path=row["zip_path"])

    def create_project(self, user_id: int, brief: str) -> int:
        """Reserve a project row at status 'running' (consumes a quota slot)."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO projects (user_id, brief, status, created_at) "
                "VALUES (?, ?, 'running', ?)", (user_id, brief, _utcnow_iso()))
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
        question, or back to 'running' when it resumes). Leaves artifacts intact."""
        with self._conn() as conn:
            conn.execute("UPDATE projects SET status=? WHERE id=?", (status, project_id))

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
        return {
            "tier": user.tier,
            "label": tier["label"],
            "price_usd": tier["price_usd"],
            "limit": tier["limit"],
            "window_days": tier["window_days"],
            "used": used,
            "remaining": max(0, tier["limit"] - used),
        }

    def can_design(self, user: User) -> bool:
        return self.quota_status(user)["remaining"] > 0
