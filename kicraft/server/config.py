"""Environment-driven settings for the KiCraft server, with cost-safety caps.

Defaults are intentionally conservative and sit well under a typical prepaid
balance, so the application self-stops before the provider balance is even
touched (defense in depth behind the prepaid + virtual-card limits).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Version of the legal documents in docs/legal/. Stamped into each user's consent
# record at signup; bumping it (here and in the documents) forces existing users
# to re-accept on their next visit. See docs/legal/README.md.
LEGAL_VERSION = "2026-06-04"

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
    legal_dir: Path = _DEFAULT_LEGAL_DIR
    kill_switch: bool = False
    request_timeout_s: int = 120

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
            legal_dir=Path(os.environ.get("KICRAFT_LEGAL_DIR", str(cls.legal_dir))),
            kill_switch=_env_bool("KICRAFT_KILL_SWITCH"),
        )

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
        }
