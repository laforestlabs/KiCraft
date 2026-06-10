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


# Invite codes: human-typeable, so the charset is strict (no whitespace or
# lookalike punctuation to mistype) and matching is case-insensitive (the column
# is COLLATE NOCASE -- 'freemax' redeems 'FREEMAX').
_INVITE_CODE_RE = re.compile(r"[A-Za-z0-9_-]{3,64}")


def grant_expiry(duration_days: int | None) -> str | None:
    """ISO-8601 UTC instant a code-granted tier lapses, or None for forever.

    Lives here (not web.py) so the date math sits next to the expiry comparison
    in _downgrade_if_expired and the two can never drift."""
    if not duration_days:
        return None
    return (_utcnow() + dt.timedelta(days=int(duration_days))).isoformat()


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
    # When an invite-code-granted tier lapses (ISO-8601 UTC), or None for no
    # expiry. Enforced lazily on read (see _downgrade_if_expired); a manual
    # set_tier clears it.
    tier_expires_at: str | None = None


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
                "role TEXT NOT NULL DEFAULT 'user',"
                "created_at TEXT NOT NULL,"
                "last_login_at TEXT,"
                "accepted_terms_version TEXT,"
                "accepted_terms_at TEXT,"
                "allow_training INTEGER NOT NULL DEFAULT 1,"
                "session_epoch INTEGER NOT NULL DEFAULT 0,"
                "tier_expires_at TEXT)"
            )
            self._ensure_columns(conn)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS invite_codes ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "code TEXT NOT NULL COLLATE NOCASE UNIQUE,"
                "tier TEXT NOT NULL DEFAULT 'free',"
                "duration_days INTEGER,"          # NULL = the tier never expires
                "max_uses INTEGER,"               # NULL = unlimited signups
                "use_count INTEGER NOT NULL DEFAULT 0,"
                "enabled INTEGER NOT NULL DEFAULT 1,"
                "created_at TEXT NOT NULL,"
                "disabled_at TEXT,"
                "last_used_at TEXT)"
            )
            # Site-wide operator switches (e.g. open_free_signup). A KV table so a
            # toggle flipped from the dashboard survives restarts without an env
            # edit + redeploy.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS app_settings ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
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
        if "tier_expires_at" not in cols:
            # NULL = no expiry, so every pre-existing tier persists unchanged.
            conn.execute("ALTER TABLE users ADD COLUMN tier_expires_at TEXT")
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
                    session_epoch=int(row["session_epoch"]),
                    tier_expires_at=row["tier_expires_at"])

    def _downgrade_if_expired(self, conn: sqlite3.Connection,
                              row: sqlite3.Row | None) -> sqlite3.Row | None:
        """Lazily lapse a code-granted tier: if the row's tier_expires_at has
        passed, downgrade it to the free tier (persisted) and return the fresh
        row. Every single-user read path funnels through here, so an expired
        grant ends the moment the account is next touched; the bulk readers the
        admin dashboard uses call expire_due_tiers() instead."""
        if row is None or row["tier_expires_at"] is None:
            return row
        if row["tier_expires_at"] > _utcnow_iso():  # ISO UTC compares lexically
            return row
        conn.execute("UPDATE users SET tier=?, tier_expires_at=NULL WHERE id=?",
                     (DEFAULT_TIER, row["id"]))
        return conn.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone()

    def expire_due_tiers(self) -> int:
        """Sweep every lapsed code-granted tier back to free; returns how many
        users were downgraded. Cheap when nothing is due (matches no rows)."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE users SET tier=?, tier_expires_at=NULL "
                "WHERE tier_expires_at IS NOT NULL AND tier_expires_at <= ?",
                (DEFAULT_TIER, _utcnow_iso()))
            return int(cur.rowcount)

    def create_user(self, email: str, password: str, tier: str = DEFAULT_TIER, *,
                    accepted_terms_version: str | None = None,
                    allow_training: bool = True,
                    tier_expires_at: str | None = None) -> User:
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
                    "accepted_terms_version, accepted_terms_at, allow_training, "
                    "tier_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (em, hash_password(password), tier, now,
                     accepted_terms_version, accepted_at, 1 if allow_training else 0,
                     tier_expires_at))
                uid = cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise ValueError(f"email {em!r} is already registered") from e
        return User(id=int(uid), email=em, tier=tier, created_at=now,
                    role=DEFAULT_ROLE,
                    accepted_terms_version=accepted_terms_version,
                    accepted_terms_at=accepted_at, allow_training=allow_training,
                    tier_expires_at=tier_expires_at)

    def get_user(self, user_id: int) -> User | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            row = self._downgrade_if_expired(conn, row)
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=?",
                               (self._norm_email(email),)).fetchone()
            row = self._downgrade_if_expired(conn, row)
        return self._row_to_user(row) if row else None

    def authenticate(self, email: str, password: str) -> User | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email=?",
                               (self._norm_email(email),)).fetchone()
            if not row or not verify_password(password, row["password_hash"]):
                return None
            row = self._downgrade_if_expired(conn, row)
            conn.execute("UPDATE users SET last_login_at=? WHERE id=?",
                         (_utcnow_iso(), row["id"]))
        return self._row_to_user(row)

    def set_tier(self, email: str, tier: str) -> User:
        """Manually assign a billing tier. Clears any invite-code expiry: an
        explicit admin assignment is indefinite, not a timed grant."""
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; choose from {', '.join(TIERS)}")
        em = self._norm_email(email)
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE users SET tier=?, tier_expires_at=NULL WHERE email=?",
                (tier, em))
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
        self.expire_due_tiers()  # so a lapsed grant never shows as still-active
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
        self.expire_due_tiers()
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
        self.expire_due_tiers()
        sql = (
            "SELECT u.id, u.email, u.tier, u.role, u.created_at, u.last_login_at, "
            "u.tier_expires_at, "
            "COUNT(p.id) AS project_count, "
            "COALESCE(SUM(p.cost_usd), 0) AS spend_usd, "
            "MAX(p.created_at) AS last_project_at "
            "FROM users u LEFT JOIN projects p ON p.user_id = u.id "
            "GROUP BY u.id ORDER BY u.created_at DESC")
        with self._conn() as conn:
            return [{k: r[k] for k in r.keys()} for r in conn.execute(sql).fetchall()]

    # ---- invite codes -------------------------------------------------------

    @staticmethod
    def _row_to_code(row: sqlite3.Row) -> dict:
        d = {k: row[k] for k in row.keys()}
        d["enabled"] = bool(d["enabled"])
        return d

    def create_invite_code(self, code: str, tier: str = DEFAULT_TIER, *,
                           duration_days: int | None = None,
                           max_uses: int | None = None) -> dict:
        """Mint an invite code that lets someone sign up at `tier`, keeping it
        for `duration_days` (None = forever) before lapsing back to free.
        `max_uses` caps how many signups may redeem it (None = unlimited)."""
        c = (code or "").strip()
        if not _INVITE_CODE_RE.fullmatch(c):
            raise ValueError("a code must be 3-64 letters, digits, '-' or '_'")
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; choose from {', '.join(TIERS)}")
        if duration_days is not None and int(duration_days) < 1:
            raise ValueError("duration_days must be at least 1 (or None for forever)")
        if max_uses is not None and int(max_uses) < 1:
            raise ValueError("max_uses must be at least 1 (or None for unlimited)")
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO invite_codes (code, tier, duration_days, max_uses, "
                    "created_at) VALUES (?, ?, ?, ?, ?)",
                    (c, tier,
                     int(duration_days) if duration_days is not None else None,
                     int(max_uses) if max_uses is not None else None,
                     _utcnow_iso()))
                row = conn.execute("SELECT * FROM invite_codes WHERE id=?",
                                   (cur.lastrowid,)).fetchone()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"code {c!r} already exists") from e
        return self._row_to_code(row)

    def list_invite_codes(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM invite_codes ORDER BY id DESC").fetchall()
        return [self._row_to_code(r) for r in rows]

    def set_invite_code_enabled(self, code_id: int, enabled: bool) -> None:
        """Disable (or re-enable) a code. Disabling only stops NEW signups; a
        tier already granted runs until its own tier_expires_at."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE invite_codes SET enabled=?, disabled_at=? WHERE id=?",
                (1 if enabled else 0, None if enabled else _utcnow_iso(), code_id))
            if cur.rowcount == 0:
                raise ValueError(f"no invite code with id {code_id!r}")

    def check_invite_code(self, code: str) -> dict | None:
        """The grant a code confers (tier, duration_days, ...), or None when the
        code is unknown, disabled, or used up. Read-only: the signup flow checks
        first, creates the user, then calls record_invite_use -- so a failed
        signup (duplicate email) never burns a use. Two signups racing the last
        use can both pass; at this scale one extra redemption is fine."""
        c = (code or "").strip()
        if not c:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM invite_codes WHERE code=? AND enabled=1 "
                "AND (max_uses IS NULL OR use_count < max_uses)", (c,)).fetchone()
        return self._row_to_code(row) if row else None

    def record_invite_use(self, code_id: int) -> None:
        """Count a successful signup against a code (call after create_user)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE invite_codes SET use_count=use_count+1, last_used_at=? "
                "WHERE id=?", (_utcnow_iso(), code_id))

    # ---- site settings ------------------------------------------------------

    _OPEN_SIGNUP_KEY = "open_free_signup"

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key=?",
                               (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def signup_open(self) -> bool:
        """Whether anyone may register on the free tier WITHOUT an invite code.
        Defaults closed (invite-only beta); flipped from /admin/invites at
        public launch."""
        return self.get_setting(self._OPEN_SIGNUP_KEY, "0") == "1"

    def set_signup_open(self, open_: bool) -> None:
        self.set_setting(self._OPEN_SIGNUP_KEY, "1" if open_ else "0")
