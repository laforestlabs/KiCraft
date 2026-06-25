# Plan: Email verification at signup (+ anti-abuse layers)

## Context

Today anyone can sign up with a **fake or unverified email** and immediately use the
free tier (1 design / 7-day window) to run expensive PCB builds. Each build burns real
LLM + routing compute that costs the owner money, and there is **nothing** stopping a
script from creating throwaway accounts in a loop to multiply that cost. Current state:

- `signup_page()` (`web.py:1857-1940`) collects email + password, calls
  `store.create_user()`, then **auto-logs-in** and drops the user straight into the
  workspace. The only email check is `create_user`'s `if not em or "@" not in em`
  (`accounts.py:758`).
- There is **no email verification, no signup rate limiting, and no disposable-domain
  filtering**.

**Goal:** require a verified email (and add two cheap anti-abuse layers) before a user
can spend compute, without hurting legitimate-user onboarding.

This feature is low-risk because it mirrors the **existing, proven password-reset token
flow** almost line-for-line (`password_resets` table, `_hash_token`, mint-with-cooldown,
single-use consume) and reuses the **existing email infrastructure** (`mailer.py`,
Resend→SMTP→dev-log fallback). We are adding a parallel `email_verifications` flow, not
inventing machinery.

## Decisions (confirmed with owner)

1. **Enforcement = block building only, not login.** Unverified users can sign up, log
   in, and browse; the **Design button is disabled** until they verify. The threat is
   compute cost, which is only incurred when a build *starts*, so the gate lives at the
   existing `can_design()` quota choke point. (Do **not** touch login / `_user_for_page`.)
2. **Ship both extra anti-abuse layers:** per-IP signup throttle **and** a
   disposable-domain blocklist. Verification alone does not stop mailinator/10minutemail
   auto-receiving services — the throttle blunts *automated* signups and the blocklist
   filters the obvious throwaway domains.

Minor defaults chosen (no further confirmation needed): verification token TTL = **24h**
(vs reset's 1h); 60s resend cooldown (same as reset); `/verify` acts on GET with
idempotent-friendly handling for email-scanner prefetch (see step 6).

## Critical files

| File | What changes |
| --- | --- |
| `kicraft/server/accounts.py` | `email_verified` column + grandfather backfill; `email_verifications` token table + mint/consume methods; `can_design()` gate; disposable-domain check in `create_user`; per-IP `signup_attempts` table + counter |
| `kicraft/server/mailer.py` | `send_verification_email()` + pure `build_verification_email()` builder |
| `kicraft/server/web.py` | signup mints+sends token; `/verify` route; unverified banner + disabled Design button + resend handler; IP throttle call at signup |
| `tests/test_accounts.py` | token roundtrip/expiry/cooldown/single-use, grandfather, `can_design` gate, disposable reject, IP throttle |

No `config.py` changes are required — email backend config already exists.

## Implementation

### 1. Schema + dataclass + grandfather backfill — `accounts.py`

- Add `email_verified INTEGER NOT NULL DEFAULT 0,` to the `CREATE TABLE users` body
  (`accounts.py:383-401`). The `DEFAULT 0` is what makes **new** signups land unverified.
- In `_ensure_columns()` (`accounts.py:590-633`), add the migration **and the
  one-time grandfather backfill** (critical — prevents locking out existing users):
  ```python
  if "email_verified" not in cols:
      conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
      conn.execute("UPDATE users SET email_verified=1")  # grandfather all pre-existing accounts
  ```
  Guarded by `not in cols`, so it runs exactly once — same idempotency pattern as the
  existing `viewed_at`/`board_code` backfills (`accounts.py:668-688`).
- Add `email_verified: bool = True` to the `User` dataclass (`accounts.py:213`, default
  True so in-memory Users are treated as verified) and
  `email_verified=bool(row["email_verified"])` in `_row_to_user()` (`accounts.py:714`).
- `create_user()` (`accounts.py:753-782`): no INSERT change needed (column default = 0);
  set `email_verified=False` on the returned in-memory `User` so it matches the row.

### 2. Verification token model — `accounts.py`

Constants near `accounts.py:75`:
```python
_VERIFY_TTL_SECONDS = 86400        # 24h
_VERIFY_COOLDOWN_SECONDS = 60      # anti-spam, same as reset
```
Add `email_verifications` table in `_init_db()`, mirroring `password_resets`
(`accounts.py:433-446`): `id, user_id, token_hash, created_at, expires_at, used_at` +
a `token_hash` index.

New methods after `consume_reset_token()` (`accounts.py:~1024`), reusing `_hash_token()`:
- **`create_verification_token(user_id) -> str | None`** — mirror `create_reset_token`
  (`accounts.py:949`) but keyed by `user_id` (we have it at signup; no email-enumeration
  concern). Enforce 60s cooldown, invalidate prior unused tokens, insert hashed token
  with 24h expiry, return raw token (`None` inside cooldown).
- **`consume_verification_token(token) -> User | None`** — mirror `consume_reset_token`
  (`accounts.py:998`): in one transaction, find unused/unexpired row, set
  `UPDATE users SET email_verified=1`, mark token `used_at`. **Does NOT bump
  `session_epoch`** (unlike reset — we must not evict the user's own just-created session).

### 3. `send_verification_email()` — `mailer.py`

Alongside `send_reset_email` (`mailer.py:64`): add `_VERIFY_SUBJECT`, a pure
`build_verification_email(to, from, verify_url, ttl_hours=24)` mirroring
`build_reset_email` (`mailer.py:45`), and `send_verification_email(settings, to, url,
ttl_hours=24)`. No backend change — `_send()` already does Resend→SMTP→dev-log, so
tests/local dev read the link from the logged message exactly like the reset flow.

### 4. Signup change — `web.py`

In `signup_page().submit()` (`web.py:1891-1928`), **keep auto-login** but after
`create_user` + `record_invite_use` succeed, mint and send:
```python
token = store.create_verification_token(user.id)
if token:
    s = Settings.from_env()
    send_verification_email(s, user.email, f"{s.public_url}/verify?token={token}", 24)
```
Add the import + a `_VERIFY_TTL_HOURS` constant near `web.py:41/140` (next to the reset
TTL constant). Existing `ValueError` handling at `web.py:1918-1920` already surfaces
`create_user` errors (used by the disposable-domain reject in step 7).

### 5. Enforcement gate — `accounts.py` `can_design()`

Fold the check into `can_design()` (`accounts.py:1573`) so **every** costly site honors
it (both `start()` and `_clone_project`, which already call it):
```python
def can_design(self, user: User) -> bool:
    if not is_admin(user) and not user.email_verified:
        return False
    return is_admin(user) or self.quota_status(user)["remaining"] > 0
```
Keep `quota_status()` purely about counts. In the UI, gate the Design button on
`not user.email_verified or q["remaining"] <= 0` and show an "unverified" message rather
than a misleading "0 of 1 left" — touch `refresh_account_ui()` (`web.py:~4447`) and the
`start()` pre-check (`web.py:4490`). Admins bypass (already handled by `is_admin`).

### 6. `/verify` route — `web.py`

New page modeled on `reset_page` (`web.py:1983-2030`), next to it:
```python
@ui.page("/verify")
def verify_page(token: str = ""):
    user = _store().consume_verification_token(token)
    # success -> card "Email confirmed — you can start designing" + button to "/"
    # None     -> if a logged-in user is ALREADY verified, show success anyway
    #             (handles email-scanner prefetch consuming the token first);
    #             else "Link invalid or expired" + a Resend affordance.
```
Consuming on GET is fine (single-use, short-lived). The "already verified → success"
fallback makes scanner-prefetch double-hits harmless. On success, the user's next page
load re-reads the row and re-enables the Design button (no session change needed).

### 7. Disposable-domain blocklist — `accounts.py`

In `create_user()` (`accounts.py:~757`), right after `_norm_email` + the `"@"` check:
```python
domain = em.rsplit("@", 1)[-1]
if domain in _DISPOSABLE_DOMAINS:
    raise ValueError("Please use a non-disposable email address.")
```
`_DISPOSABLE_DOMAINS` = a `frozenset` of known throwaway domains in a small data
module (e.g. `kicraft/server/disposable_domains.py`) so it's updatable without churn.
`signup_page.submit()` already catches `ValueError` and shows it — no web change needed.

### 8. Per-IP signup throttle — `accounts.py` + `web.py`

- New `signup_attempts(id, ip, created_at)` table in `_init_db()`.
- `record_signup_attempt(ip)` and `count_recent_signups_by_ip(ip, window_seconds) -> int`
  methods (DB-backed so it works across the multi-worker deployment, unlike an in-process
  counter).
- In `signup_page().submit()` (`web.py:~1891`), before `create_user`: read the client IP
  from the NiceGUI request context, and if `count_recent_signups_by_ip(ip, 3600) >= 5`,
  reject with "Too many signups from this network — try again later." Record the attempt
  on each submit. (Threshold 5/hr/IP — tune later.)

## Verification / testing

**Unit tests** (`tests/test_accounts.py`, mirror the reset-token tests
`test_accounts.py:339-404`; add a `_set_verify_times()` helper):
- `test_new_user_is_unverified`, `test_verification_token_roundtrip`,
  `test_verification_token_is_single_use`, `test_verification_token_expires`,
  `test_new_verification_token_invalidates_prior`, `test_create_verification_token_cooldown`,
  `test_consume_garbage_and_empty_token`.
- `test_verification_does_not_bump_session_epoch` (explicit contrast with the reset flow).
- `test_can_design_blocks_unverified_user` (has quota, still blocked) and re-enabled after
  consume; `test_admin_bypasses_verification`.
- `test_legacy_db_users_are_grandfathered_verified` — build a pre-`email_verified` users
  table (mirror `test_accounts.py:138`), open `AccountStore`, assert grandfathered True.
  **This is the critical lock-out regression test.**
- `test_disposable_domain_rejected_at_signup`; `test_signup_ip_throttle_blocks_after_n`.

Run: `pytest tests/test_accounts.py -q` (pure Python + SQLite, no network).

**End-to-end (manual / mock-LLM web driver):**
1. Start the web app locally with **no** email backend configured. Sign up with a fresh
   email → confirm you land in the workspace, the unverified banner shows, and the Design
   button is **disabled**.
2. Grep the app log for the dev-log verification message (the mailer logs the body when
   no backend is set) and open the `/verify?token=...` link → card says confirmed, Design
   button **enables** on next load.
3. Try signing up with a disposable domain (e.g. `x@mailinator.com`) → rejected inline.
4. Submit signup 6× from the same session/IP → 6th is throttled.
5. Click **Resend** twice within 60s → second is a silent no-op (cooldown), neutral toast.

**Deploy note:** per CLAUDE.md, restarting **both** `kicraft-web` and the build worker is
only needed for *pipeline* changes — this is web/DB only, so restarting `kicraft-web`
(`deploy/restart-web.sh`) suffices. The schema migration runs automatically on
`AccountStore` init.
