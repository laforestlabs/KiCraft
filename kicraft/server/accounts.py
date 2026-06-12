"""Per-user accounts, projects, and tiered usage quotas for the KiCraft web app.

A small SQLite store mirroring spend_guard.SpendGuard's conventions (connection
per op, WAL, CREATE TABLE IF NOT EXISTS). It holds the user identities, their
saved projects, and the metering needed to enforce the per-tier design quotas.

Paid tiers are granted three ways: a Stripe subscription (kicraft.server.billing
syncs tier + tier_expires_at from webhook state), an invite code, or the admin
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

# Billing tier definitions. `price_usd` is the monthly subscription price; the
# actual charge amount lives on the Stripe Price objects (KICRAFT_STRIPE_PRICE_*),
# so keep the two in sync when changing pricing. `limit` designs per rolling
# `window_days` is what count_active_designs enforces. "free" = 1/week, "pro" = 5/month ($5),
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

# Board IDs: short, human-quotable codes (KC-7G4K2M) stamped on every project so
# a user can read one off the workspace and quote it in a support report. The
# alphabet drops 0/1/I/L/O so a code survives being read aloud or retyped.
_BOARD_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
_BOARD_CODE_LENGTH = 6


def new_board_code() -> str:
    """Draw a fresh board id, e.g. 'KC-7G4K2M' (~900M combinations at length 6).
    Uniqueness is enforced by the projects.board_code unique index; callers that
    insert retry on the (vanishingly rare) collision."""
    body = "".join(secrets.choice(_BOARD_CODE_ALPHABET)
                   for _ in range(_BOARD_CODE_LENGTH))
    return f"KC-{body}"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _claimant_pid(claimed_by: str | None) -> int | None:
    """Parse the pid out of a build job's 'pid:<n>' claimant tag."""
    if claimed_by and claimed_by.startswith("pid:"):
        try:
            return int(claimed_by.split(":")[1])
        except (ValueError, IndexError):
            return None
    return None


def _pid_alive(pid: int | None) -> bool:
    """Conservative liveness: an unparseable claimant reads as alive, so a tag
    format change can never mass-requeue builds that are actually running."""
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True


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
    # Email the user when a design run finishes or parks on a question, so they
    # can walk away from a multi-minute (possibly queued) build. Suppressed when
    # the user is actively watching (see kicraft.server.notify).
    notify_email: bool = True
    # Stripe linkage (kicraft.server.billing). Access is still decided by
    # tier + tier_expires_at above; subscription_id/status are diagnostics shown
    # on /profile and /admin/users, refreshed on every webhook sync.
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    subscription_status: str | None = None


@dataclass
class BuildJob:
    """One queued/running deterministic build (the heavy place+route phase).

    The row is the cross-process protocol between the web app (which enqueues
    and tails) and the build worker (which claims and executes); see
    kicraft.server.build_worker. `claimed_by` is "pid:<pid>" of the claimant so
    a restarted worker can detect and requeue orphans of a dead predecessor."""
    id: int
    project_id: int | None
    user_id: int | None
    workspace: str
    status: str  # queued | running | done | failed
    created_at: str
    rc: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    attempts: int = 0
    claimed_by: str | None = None
    log_path: str | None = None


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
    # Human-quotable unique id (KC-XXXXXX) shown in the workspace so a user can
    # reference this exact board in a support report. See new_board_code().
    board_code: str | None = None


@dataclass
class SupportReport:
    """One troubleshooting report, written either automatically when a run fails
    (kind='error_auto', logged even if nobody is watching) or by the user from
    the Support button (kind='user'). `diagnostics` is the machine-readable
    snapshot (build-log tail, failed checks, run status) that automated review
    consumes; `message` is the user's optional freeform feedback."""
    id: int
    created_at: str
    user_id: int | None
    project_id: int | None
    board_code: str | None
    kind: str    # 'error_auto' | 'user'
    status: str  # triage state: 'new' | 'reviewed'
    message: str | None
    diagnostics: dict


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
                "session_epoch INTEGER NOT NULL DEFAULT 0,"
                "tier_expires_at TEXT,"
                "notify_email INTEGER NOT NULL DEFAULT 1,"
                "stripe_customer_id TEXT,"
                "stripe_subscription_id TEXT,"
                "subscription_status TEXT)"
            )
            self._ensure_columns(conn)
            # Processed Stripe webhook event ids (INSERT OR IGNORE dedupe).
            # Stripe retries delivery until it gets a 2xx, so the same event can
            # arrive more than once; see record_billing_event.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS billing_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "stripe_event_id TEXT NOT NULL UNIQUE,"
                "type TEXT NOT NULL,"
                "created_at TEXT NOT NULL)"
            )
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
                "viewed_at TEXT,"
                "is_public INTEGER NOT NULL DEFAULT 1,"
                "cloned_from_id INTEGER,"
                "view_count INTEGER NOT NULL DEFAULT 0,"
                "clone_count INTEGER NOT NULL DEFAULT 0,"
                "like_count INTEGER NOT NULL DEFAULT 0,"
                "quality TEXT,"
                "board_code TEXT,"
                "FOREIGN KEY(user_id) REFERENCES users(id))"
            )
            self._ensure_project_columns(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_projects_user_created "
                "ON projects(user_id, created_at)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_board_code "
                "ON projects(board_code)"
            )
            # Support reports: one row per problem report, auto-filed on a failed
            # run and user-filed from the Support button. Kept in the DB (not a
            # log file) so automated review can query by status/board_code and an
            # account deletion can purge the user's reports with their data.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS support_reports ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "created_at TEXT NOT NULL,"
                "user_id INTEGER,"
                "project_id INTEGER,"
                "board_code TEXT,"
                "kind TEXT NOT NULL DEFAULT 'user',"
                "status TEXT NOT NULL DEFAULT 'new',"
                "message TEXT,"
                "diagnostics TEXT,"
                "FOREIGN KEY(user_id) REFERENCES users(id),"
                "FOREIGN KEY(project_id) REFERENCES projects(id))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_support_reports_status "
                "ON support_reports(status, id)"
            )
            # FIFO queue of deterministic builds (see BuildJob). Shared by the web
            # app and the standalone build worker, so it lives here with the rest
            # of the cross-process state rather than in web-process memory.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS build_jobs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "project_id INTEGER,"
                "user_id INTEGER,"
                "workspace TEXT NOT NULL,"
                "status TEXT NOT NULL DEFAULT 'queued',"
                "rc INTEGER,"
                "created_at TEXT NOT NULL,"
                "started_at TEXT,"
                "finished_at TEXT,"
                "attempts INTEGER NOT NULL DEFAULT 0,"
                "claimed_by TEXT,"
                "log_path TEXT)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_build_jobs_status "
                "ON build_jobs(status, id)"
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
        if "tier_expires_at" not in cols:
            # NULL = no expiry, so every pre-existing tier persists unchanged.
            conn.execute("ALTER TABLE users ADD COLUMN tier_expires_at TEXT")
        if "notify_email" not in cols:
            conn.execute(
                "ALTER TABLE users ADD COLUMN notify_email INTEGER NOT NULL DEFAULT 1")
        if "stripe_customer_id" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN stripe_customer_id TEXT")
        if "stripe_subscription_id" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT")
        if "subscription_status" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN subscription_status TEXT")
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
        if "board_code" not in cols:
            conn.execute("ALTER TABLE projects ADD COLUMN board_code TEXT")
            # One-time backfill so EVERY project (including pre-existing ones)
            # is quotable in a support report. Drawn collision-free in Python:
            # the unique index is created right after this migration runs.
            ids = [r["id"] for r in conn.execute("SELECT id FROM projects")]
            seen: set = set()
            for pid in ids:
                code = new_board_code()
                while code in seen:
                    code = new_board_code()
                seen.add(code)
                conn.execute("UPDATE projects SET board_code=? WHERE id=?",
                             (code, pid))

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
                    session_epoch=int(row["session_epoch"]),
                    tier_expires_at=row["tier_expires_at"],
                    notify_email=bool(row["notify_email"]),
                    stripe_customer_id=row["stripe_customer_id"],
                    stripe_subscription_id=row["stripe_subscription_id"],
                    subscription_status=row["subscription_status"])

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
        explicit admin assignment is indefinite, not a timed grant. On a user
        with a live Stripe subscription the manual tier holds only until the
        next webhook resync (apply_subscription_state overwrites it)."""
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

    # ---- billing (Stripe) ---------------------------------------------------

    def set_stripe_customer(self, user_id: int, customer_id: str) -> None:
        """Link a user to their Stripe customer (created on first checkout)."""
        with self._conn() as conn:
            conn.execute("UPDATE users SET stripe_customer_id=? WHERE id=?",
                         (customer_id, user_id))

    def get_user_by_stripe_customer(self, customer_id: str) -> User | None:
        """Resolve a webhook's customer id back to the local account."""
        if not customer_id:
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE stripe_customer_id=?",
                               (customer_id,)).fetchone()
            row = self._downgrade_if_expired(conn, row)
        return self._row_to_user(row) if row else None

    def apply_subscription_state(self, user_id: int, *, tier: str,
                                 tier_expires_at: str | None,
                                 subscription_id: str | None,
                                 status: str | None) -> User:
        """Sync a user's access from authoritative Stripe subscription state.

        The single write path for every billing sync (webhook or success page).
        tier + tier_expires_at carry the access decision: a paid cycle sets the
        expiry to the period end plus a grace window, so a renewal extends it
        and a lapsed subscription falls back to free via _downgrade_if_expired
        with no extra code. subscription_id/status are diagnostics only."""
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE users SET tier=?, tier_expires_at=?, "
                "stripe_subscription_id=?, subscription_status=? WHERE id=?",
                (tier, tier_expires_at, subscription_id, status, user_id))
            if cur.rowcount == 0:
                raise ValueError(f"no user with id {user_id}")
            row = conn.execute(
                "SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return self._row_to_user(row)

    def record_billing_event(self, event_id: str, event_type: str) -> bool:
        """Record a Stripe webhook event id; False when it was already seen.

        The dedupe that makes webhook handling idempotent: Stripe retries
        delivery until it receives a 2xx, so the same event can arrive twice."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO billing_events (stripe_event_id, type, "
                "created_at) VALUES (?, ?, ?)",
                (event_id, event_type, _utcnow_iso()))
            return cur.rowcount > 0

    def forget_billing_event(self, event_id: str) -> None:
        """Release a claimed event id after its processing FAILED, so Stripe's
        retry of the same event is processed instead of acked as a duplicate.
        (record first, process second, forget on failure: the claim is what
        makes a concurrent duplicate delivery a no-op.)"""
        with self._conn() as conn:
            conn.execute("DELETE FROM billing_events WHERE stripe_event_id=?",
                         (event_id,))

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
            "support_reports": [asdict(r) for r in
                                self.list_support_reports(user_id=user_id,
                                                          limit=10000)],
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
            conn.execute("DELETE FROM support_reports WHERE user_id=?", (user_id,))
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
                       like_count=row["like_count"], quality=row["quality"],
                       board_code=row["board_code"])

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
            # Retried on IntegrityError: with no enforced FKs the only unique
            # constraint in play is idx_projects_board_code, so a failure means
            # the freshly drawn code collided; draw again.
            for _ in range(5):
                try:
                    cur = conn.execute(
                        "INSERT INTO projects (user_id, brief, status, created_at, "
                        "is_public, board_code) VALUES (?, ?, 'running', ?, ?, ?)",
                        (user_id, brief, _utcnow_iso(), 1 if is_public else 0,
                         new_board_code()))
                    return int(cur.lastrowid)
                except sqlite3.IntegrityError:
                    continue
            raise RuntimeError("could not draw a unique board code")

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

    # ---- support reports ---------------------------------------------------

    @staticmethod
    def _row_to_support_report(row: sqlite3.Row) -> SupportReport:
        try:
            diag = json.loads(row["diagnostics"] or "{}")
        except json.JSONDecodeError:
            diag = {}
        return SupportReport(
            id=row["id"], created_at=row["created_at"], user_id=row["user_id"],
            project_id=row["project_id"], board_code=row["board_code"],
            kind=row["kind"], status=row["status"], message=row["message"],
            diagnostics=diag if isinstance(diag, dict) else {})

    def create_support_report(self, *, user_id: int | None = None,
                              project_id: int | None = None,
                              board_code: str | None = None,
                              kind: str = "user",
                              message: str | None = None,
                              diagnostics: dict | None = None) -> int:
        """File a report and return its id (the user-facing reference when the
        report has no board code, e.g. filed from a blank composer)."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO support_reports (created_at, user_id, project_id, "
                "board_code, kind, status, message, diagnostics) "
                "VALUES (?, ?, ?, ?, ?, 'new', ?, ?)",
                (_utcnow_iso(), user_id, project_id, board_code, kind, message,
                 json.dumps(diagnostics or {}, ensure_ascii=False, default=str)))
            return int(cur.lastrowid)

    def set_support_report_message(self, report_id: int, message: str) -> None:
        """Attach the user's freeform feedback to an auto-filed error report (the
        run worker files the row; the dialog adds the human context later)."""
        with self._conn() as conn:
            conn.execute("UPDATE support_reports SET message=? WHERE id=?",
                         (message, report_id))

    def set_support_report_status(self, report_id: int, status: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE support_reports SET status=? WHERE id=?",
                         (status, report_id))

    def list_support_reports(self, *, status: str | None = None,
                             user_id: int | None = None,
                             limit: int = 200) -> list[SupportReport]:
        """Newest first; filterable by triage status (automated review polls
        status='new') and by user (the data-export path)."""
        where, params = [], []
        if status is not None:
            where.append("status=?")
            params.append(status)
        if user_id is not None:
            where.append("user_id=?")
            params.append(user_id)
        sql = "SELECT * FROM support_reports"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_support_report(r) for r in rows]

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
            "u.tier_expires_at, u.subscription_status, "
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

    # ---- build queue ---------------------------------------------------------
    # Cross-process FIFO of deterministic builds. The web app enqueues one row per
    # build and tails its log; the standalone worker (kicraft.server.build_worker)
    # claims and executes rows, or the web thread self-claims its own row when no
    # worker is alive (single-service deploys keep working unchanged). All claims
    # go through one guarded UPDATE, so a row can never run twice.

    _WORKER_HEARTBEAT_KEY = "build_worker_heartbeat"

    @staticmethod
    def _row_to_build_job(row: sqlite3.Row) -> BuildJob:
        return BuildJob(id=row["id"], project_id=row["project_id"],
                        user_id=row["user_id"], workspace=row["workspace"],
                        status=row["status"], rc=row["rc"],
                        created_at=row["created_at"], started_at=row["started_at"],
                        finished_at=row["finished_at"], attempts=row["attempts"],
                        claimed_by=row["claimed_by"], log_path=row["log_path"])

    def enqueue_build(self, *, workspace: str, project_id: int | None = None,
                      user_id: int | None = None, log_path: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO build_jobs (project_id, user_id, workspace, status, "
                "created_at, log_path) VALUES (?, ?, ?, 'queued', ?, ?)",
                (project_id, user_id, workspace, _utcnow_iso(), log_path))
            return int(cur.lastrowid)

    def get_build_job(self, job_id: int) -> BuildJob | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM build_jobs WHERE id=?",
                               (job_id,)).fetchone()
        return self._row_to_build_job(row) if row else None

    def claim_build(self, job_id: int, claimed_by: str) -> bool:
        """Atomically move one specific queued job to running. The status guard in
        the WHERE makes concurrent claimants (worker vs web fallback) safe: exactly
        one UPDATE matches."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE build_jobs SET status='running', started_at=?, "
                "claimed_by=?, attempts=attempts+1 WHERE id=? AND status='queued'",
                (_utcnow_iso(), claimed_by, job_id))
            return cur.rowcount == 1

    def claim_next_build(self, claimed_by: str) -> BuildJob | None:
        """Claim the oldest queued job (FIFO), or None when the queue is empty."""
        while True:
            with self._conn() as conn:
                row = conn.execute("SELECT id FROM build_jobs WHERE status='queued' "
                                   "ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            if self.claim_build(int(row["id"]), claimed_by):
                return self.get_build_job(int(row["id"]))
            # Lost the race for that row; the next loop sees the new queue head.

    def finish_build(self, job_id: int, *, rc: int | None,
                     status: str = "done") -> None:
        """`done` = the build process ran to an exit code (rc, any value);
        `failed` = it could not run at all (missing workspace, attempts
        exhausted, aborted by a worker shutdown)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE build_jobs SET status=?, rc=?, finished_at=? WHERE id=?",
                (status, rc, _utcnow_iso(), job_id))

    def requeue_build(self, job_id: int) -> None:
        """Put a claimed-but-aborted job back at its queue position (id order)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE build_jobs SET status='queued', claimed_by=NULL, "
                "started_at=NULL WHERE id=? AND status='running'", (job_id,))

    def build_queue_position(self, job_id: int) -> tuple[int, int, int]:
        """(jobs queued ahead of `job_id`, total queued, running) for the UI."""
        with self._conn() as conn:
            ahead = conn.execute(
                "SELECT COUNT(*) FROM build_jobs WHERE status='queued' AND id<?",
                (job_id,)).fetchone()[0]
            depth = conn.execute(
                "SELECT COUNT(*) FROM build_jobs WHERE status='queued'").fetchone()[0]
            running = conn.execute(
                "SELECT COUNT(*) FROM build_jobs WHERE status='running'").fetchone()[0]
        return int(ahead), int(depth), int(running)

    def list_unfinalized_builds(self) -> list[BuildJob]:
        """Jobs whose owning project is still 'running' but which no longer have
        (or may not have) a live driving thread: finished ones to finalize, and
        queued ones to fail when nothing will ever execute them. Consumed by the
        web app's orphan reaper after a restart."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT b.* FROM build_jobs b JOIN projects p ON p.id=b.project_id "
                "WHERE p.status='running' AND b.status IN ('done','failed','queued') "
                "ORDER BY b.id").fetchall()
        return [self._row_to_build_job(r) for r in rows]

    def count_running_builds(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM build_jobs "
                                    "WHERE status='running'").fetchone()[0])

    def avg_build_seconds(self, last_n: int = 5) -> float | None:
        """Rolling mean wall-clock of the last completed builds, for queue ETAs.
        None until one build has completed (callers show no estimate)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT AVG((julianday(finished_at) - julianday(started_at)) * 86400.0) "
                "FROM (SELECT started_at, finished_at FROM build_jobs "
                "      WHERE status='done' AND started_at IS NOT NULL "
                "      ORDER BY id DESC LIMIT ?)", (last_n,)).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def requeue_stale_builds(self, *, max_attempts: int = 2) -> int:
        """Recover 'running' rows whose claimant process is dead (crashed web or
        worker): requeue them, or fail them once they have burned `max_attempts`
        claims. Same-host pid checks only, which matches the single-box deploy.
        Returns how many rows changed."""
        changed = 0
        with self._conn() as conn:
            rows = conn.execute("SELECT id, attempts, claimed_by FROM build_jobs "
                                "WHERE status='running'").fetchall()
        for r in rows:
            if _pid_alive(_claimant_pid(r["claimed_by"])):
                continue
            if int(r["attempts"]) >= max_attempts:
                self.finish_build(int(r["id"]), rc=None, status="failed")
            else:
                self.requeue_build(int(r["id"]))
            changed += 1
        return changed

    def beat_build_worker(self) -> None:
        self.set_setting(self._WORKER_HEARTBEAT_KEY, _utcnow_iso())

    def build_worker_alive(self, max_age_s: float = 15.0) -> bool:
        """Whether a standalone build worker heartbeated recently. Stale or absent
        -> the web falls back to running builds in-process, so a deploy without
        the worker unit behaves exactly as before this feature."""
        raw = self.get_setting(self._WORKER_HEARTBEAT_KEY)
        if not raw:
            return False
        try:
            beat = dt.datetime.fromisoformat(raw)
        except ValueError:
            return False
        return (_utcnow() - beat).total_seconds() <= max_age_s

    # ---- notification preference --------------------------------------------

    def set_notify_email(self, user_id: int, enabled: bool) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE users SET notify_email=? WHERE id=?",
                         (1 if enabled else 0, user_id))

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
