"""Environment-driven settings for the KiCraft server, with cost-safety caps.

Defaults are intentionally conservative and sit well under a typical prepaid
balance, so the application self-stops before the provider balance is even
touched (defense in depth behind the prepaid + virtual-card limits).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Version of the legal documents in docs/legal/. Stamped into each user's consent
# record at signup; bumping it (here and in the documents) forces existing users
# to re-accept on their next visit. See docs/legal/README.md.
LEGAL_VERSION = "2026-06-11"

# Canonical location of the Terms / Privacy markdown, served at /terms and
# /privacy. Resolves to the repo's docs/legal relative to this package; the box
# runs an editable install tracking repo HEAD, so the path is present there.
_DEFAULT_LEGAL_DIR = Path(__file__).resolve().parents[2] / "docs" / "legal"


def default_legal_dir() -> Path:
    """Resolve the legal-docs directory from the env (or the packaged default).

    Standalone of Settings so the public /terms and /privacy pages can read the
    documents without an OPENROUTER_API_KEY.
    """
    return Path(os.environ.get("KICRAFT_LEGAL_DIR", str(_DEFAULT_LEGAL_DIR)))


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no override).

    A tiny stdlib loader so local runs work without python-dotenv. Existing
    environment variables win, so a secret exported in the shell takes priority.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_bool_default(name: str, default: bool) -> bool:
    """Like `_env_bool` but returns `default` when the variable is unset/blank
    (for flags that default ON, where absence must not read as False)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Resolved server configuration. Build with `Settings.from_env()`."""

    api_key: str
    model: str = "deepseek/deepseek-v4-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens_per_call: int = 1024
    daily_usd_ceiling: float = 5.0
    total_usd_ceiling: float = 50.0
    ledger_path: Path = Path.home() / ".kicraft" / "spend_ledger.db"
    users_db_path: Path = Path.home() / ".kicraft" / "accounts.db"
    projects_dir: Path = Path.home() / ".kicraft" / "projects"
    # Run workspaces (one tempdir per design run). A real directory, not /tmp:
    # the standalone build worker is a separate systemd unit and PrivateTmp
    # would hide the web app's workspaces from it.
    work_dir: Path = Path.home() / ".kicraft" / "work"
    legal_dir: Path = _DEFAULT_LEGAL_DIR
    kill_switch: bool = False
    request_timeout_s: int = 120

    # --- Outbound email + public URL (password-reset delivery) ---------------
    # public_url is the externally reachable origin (e.g. https://kicraft.io); it
    # is used to build absolute reset links so they never depend on a request's
    # (spoofable) Host header. email_from is the sender shown on reset mail.
    #
    # Two delivery backends, picked by whichever is configured (Resend wins if
    # both are set):
    #   - Resend HTTP API: set resend_api_key (KICRAFT_RESEND_API_KEY).
    #   - SMTP: set smtp_host (+ the other smtp_* fields).
    # With neither set, the mailer logs the link instead of sending (local dev).
    public_url: str = "http://localhost:8080"
    email_from: str = ""
    resend_api_key: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_starttls: bool = True
    smtp_ssl: bool = False

    # --- Stripe billing (credit-card checkout for the paid tiers) ------------
    # Hosted Stripe Checkout + Customer Portal; card data never touches KiCraft.
    # The two price ids are Stripe recurring Prices (monthly) mapped onto the
    # paid tiers in accounts.TIERS. All four values must be set for billing to
    # be live (see billing_enabled); with any missing, /pricing still renders
    # but without checkout buttons, and the webhook endpoint refuses events.
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro: str = ""
    stripe_price_max: str = ""

    # --- OpenRouter provider routing + caching (cost safety) -----------------
    # The model id is served by ~14 backends at a tight price band, but only some
    # cache our long, re-sent system prefix (the dominant cost). The DeepSeek
    # first-party backend is excluded by this account's data policy: pinning
    # `deepseek` returns 404, so the previous pin silently fell through to
    # OpenRouter's default, a poorly-caching backend (Baidu, ~46% warm hit). We
    # instead pin an ordered set of verified fp8 backends that actually cache the
    # prefix (92-100% warm), benchmarked via `provider-bench` and led by the
    # lowest-latency ones. `allow_fallbacks` keeps the service up if those are
    # down, and `max_price` is a hard per-Mtok ceiling that bounds any fallback to
    # the cheap caching tier. All are USD per million tokens; 0.0 means "omit".
    provider_order: list[str] = field(
        default_factory=lambda: ["novita/fp8", "siliconflow/fp8", "streamlake"])
    provider_allow_fallbacks: bool = True
    max_price_prompt: float = 0.18
    max_price_completion: float = 0.35
    enable_prompt_cache: bool = True

    # Surface the core-components registry (admin-curated default part per
    # functional block, /admin/core-components) in the architecture/bom prompts
    # so the model adopts defaults instead of researching them. KICRAFT_CORE_DEFAULTS=0
    # disables for A/B cost comparisons.
    enable_core_defaults: bool = True

    # --- Class-J self-evaluation judge (admin-only web feature) ---------------
    # The design loop runs a deliberately cheap/weak model; grading a finished run
    # wants a stronger one. None means "reuse the design model" (`model`), which
    # always works with the provider routing above. Point KICRAFT_EVAL_JUDGE_MODEL
    # at a more capable id for better judgments; that is an admin action, since a
    # different model may need its own provider routing.
    eval_judge_model: str | None = None

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your "
                "key (the .env file is gitignored; never commit it)."
            )
        return cls(
            api_key=key,
            model=os.environ.get("KICRAFT_MODEL", cls.model).strip() or cls.model,
            max_tokens_per_call=int(
                os.environ.get("KICRAFT_MAX_TOKENS_PER_CALL", cls.max_tokens_per_call)),
            daily_usd_ceiling=float(
                os.environ.get("KICRAFT_DAILY_USD_CEILING", cls.daily_usd_ceiling)),
            total_usd_ceiling=float(
                os.environ.get("KICRAFT_TOTAL_USD_CEILING", cls.total_usd_ceiling)),
            ledger_path=Path(os.environ.get("KICRAFT_SPEND_LEDGER", str(cls.ledger_path))),
            users_db_path=Path(os.environ.get("KICRAFT_USERS_DB", str(cls.users_db_path))),
            projects_dir=Path(os.environ.get("KICRAFT_PROJECTS_DIR", str(cls.projects_dir))),
            work_dir=Path(os.environ.get("KICRAFT_WORK_DIR", str(cls.work_dir))),
            legal_dir=Path(os.environ.get("KICRAFT_LEGAL_DIR", str(cls.legal_dir))),
            kill_switch=_env_bool("KICRAFT_KILL_SWITCH"),
            public_url=os.environ.get(
                "KICRAFT_PUBLIC_URL", cls.public_url).strip().rstrip("/") or cls.public_url,
            email_from=(os.environ.get("KICRAFT_EMAIL_FROM", "").strip()
                        or os.environ.get("KICRAFT_SMTP_FROM", "").strip()
                        or os.environ.get("KICRAFT_SMTP_USERNAME", "").strip()),
            resend_api_key=os.environ.get("KICRAFT_RESEND_API_KEY", "").strip(),
            smtp_host=os.environ.get("KICRAFT_SMTP_HOST", "").strip(),
            smtp_port=int(os.environ.get("KICRAFT_SMTP_PORT", cls.smtp_port)),
            smtp_username=os.environ.get("KICRAFT_SMTP_USERNAME", "").strip(),
            smtp_password=os.environ.get("KICRAFT_SMTP_PASSWORD", ""),
            smtp_from=(os.environ.get("KICRAFT_SMTP_FROM", "").strip()
                       or os.environ.get("KICRAFT_SMTP_USERNAME", "").strip()),
            smtp_starttls=_env_bool_default("KICRAFT_SMTP_STARTTLS", True),
            smtp_ssl=_env_bool("KICRAFT_SMTP_SSL"),
            stripe_secret_key=os.environ.get("KICRAFT_STRIPE_SECRET_KEY", "").strip(),
            stripe_webhook_secret=os.environ.get(
                "KICRAFT_STRIPE_WEBHOOK_SECRET", "").strip(),
            stripe_price_pro=os.environ.get("KICRAFT_STRIPE_PRICE_PRO", "").strip(),
            stripe_price_max=os.environ.get("KICRAFT_STRIPE_PRICE_MAX", "").strip(),
            provider_order=[p.strip() for p in os.environ.get(
                "KICRAFT_PROVIDER_ORDER",
                "novita/fp8,siliconflow/fp8,streamlake").split(",") if p.strip()],
            provider_allow_fallbacks=_env_bool_default(
                "KICRAFT_PROVIDER_ALLOW_FALLBACKS", True),
            max_price_prompt=float(
                os.environ.get("KICRAFT_MAX_PRICE_PROMPT", cls.max_price_prompt)),
            max_price_completion=float(
                os.environ.get("KICRAFT_MAX_PRICE_COMPLETION", cls.max_price_completion)),
            enable_prompt_cache=_env_bool_default("KICRAFT_ENABLE_PROMPT_CACHE", True),
            enable_core_defaults=_env_bool_default("KICRAFT_CORE_DEFAULTS", True),
            eval_judge_model=(os.environ.get("KICRAFT_EVAL_JUDGE_MODEL", "").strip() or None),
        )

    @property
    def billing_enabled(self) -> bool:
        """Whether Stripe billing is fully configured (key, webhook secret, and
        both tier price ids). Anything less and the paid-tier checkout is
        hidden, so a partially configured box can never half-charge a card."""
        return bool(self.stripe_secret_key and self.stripe_webhook_secret
                    and self.stripe_price_pro and self.stripe_price_max)

    def redacted(self) -> dict:
        """Settings safe to display/log (without the secret key)."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens_per_call": self.max_tokens_per_call,
            "daily_usd_ceiling": self.daily_usd_ceiling,
            "total_usd_ceiling": self.total_usd_ceiling,
            "ledger_path": str(self.ledger_path),
            "kill_switch": self.kill_switch,
            "public_url": self.public_url,
            "email_from": self.email_from,
            "resend_configured": bool(self.resend_api_key),
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_from": self.smtp_from,
            "smtp_starttls": self.smtp_starttls,
            "smtp_ssl": self.smtp_ssl,
            "billing_enabled": self.billing_enabled,
            "stripe_price_pro": self.stripe_price_pro,
            "stripe_price_max": self.stripe_price_max,
            "provider_order": self.provider_order,
            "provider_allow_fallbacks": self.provider_allow_fallbacks,
            "max_price_prompt": self.max_price_prompt,
            "max_price_completion": self.max_price_completion,
            "enable_prompt_cache": self.enable_prompt_cache,
            "enable_core_defaults": self.enable_core_defaults,
            "eval_judge_model": self.eval_judge_model,
        }
