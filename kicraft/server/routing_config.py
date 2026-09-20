"""Admin-controlled design-model routing + behavior knobs (durable JSON file).

The file holds ONLY the knobs the ``/admin/routing`` page may move: the active
design profile (a named ``DESIGN_PROFILES`` entry) and four behavior values.

It MUST NEVER hold secrets or the hard cost/safety ceilings. API keys live in
``.env``, and so do ``KICRAFT_KILL_SWITCH`` / the daily and total USD ceilings --
any key outside ``ALLOWED_KEYS`` is ignored on load so a hand-edited file can
never widen those boundaries.

Path: ``KICRAFT_ROUTING_CONFIG`` (env override) or ``~/.kicraft/routing.json``.

Resolution: a present file whose ``active_profile`` names a known profile is
authoritative for the designer route (its behavior keys override the
env-derived ``Settings`` values). Missing file, unreadable/invalid JSON, or an
unknown/empty ``active_profile`` -> the whole file is ignored and the env wins;
one warning is logged per distinct reason.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

log = logging.getLogger("kicraft.routing_config")

ROUTING_CONFIG_ENV = "KICRAFT_ROUTING_CONFIG"
DEFAULT_ROUTING_CONFIG_PATH = Path.home() / ".kicraft" / "routing.json"

# Bounds for the behavior knobs. These are deliberately the same finite
# ceilings the code already enforces elsewhere: max_tokens_per_call may not
# exceed the largest per-stage serialization cap, and the project budget is a
# soft per-run guard (the hard daily/total ceilings stay .env-only).
MAX_TOKENS_PER_CALL_CEILING = 32768
MAX_PROJECT_BUDGET_USD = 1000.0
MAX_REASONING_TOKENS = 65536

# The only keys the admin page may persist. Everything else is dropped.
ALLOWED_KEYS = frozenset(
    {
        "active_profile",
        "pipeline",
        "max_tokens_per_call",
        "project_llm_budget_usd",
        "design_reasoning_tokens",
        "design_temperature",
    }
)

#: The design pipelines an admin may select (see `pipeline.py` for what they mean).
PIPELINES = ("current", "legacy")

_warned: set[str] = set()


def _warn_once(reason: str) -> None:
    if reason in _warned:
        return
    _warned.add(reason)
    log.warning("routing config: %s", reason)


def default_path() -> Path:
    """Resolve the routing-config path (env override, else the default)."""
    raw = os.environ.get(ROUTING_CONFIG_ENV, "").strip()
    return Path(raw) if raw else DEFAULT_ROUTING_CONFIG_PATH


def _known_profiles() -> frozenset[str]:
    # Lazy import: config imports this module, so a module-level import would
    # be circular. By call time config is fully loaded.
    from .config import DESIGN_PROFILES

    return frozenset(DESIGN_PROFILES)


@dataclass(frozen=True)
class RoutingConfig:
    """The allowlisted admin knobs, with ``None`` meaning "leave env alone"."""

    active_profile: str = ""
    # Which design pipeline builds a project: `current` (the typed contract tree) or `legacy`
    # (the August tree at its pinned commit). Persisted like `active_profile`, and read per
    # design run and per build job, so a swap needs no restart.
    pipeline: str = ""
    max_tokens_per_call: int | None = None
    project_llm_budget_usd: float | None = None
    design_reasoning_tokens: int | None = None
    design_temperature: float | None = None

    @property
    def authoritative(self) -> bool:
        """True when this config selects the designer profile."""
        return bool(self.active_profile)

    def as_dict(self) -> dict:
        """The persisted JSON shape (unset values omitted)."""
        data: dict = {}
        if self.active_profile:
            data["active_profile"] = self.active_profile
        if self.pipeline:
            data["pipeline"] = self.pipeline
        for key, value in (
            ("max_tokens_per_call", self.max_tokens_per_call),
            ("project_llm_budget_usd", self.project_llm_budget_usd),
            ("design_reasoning_tokens", self.design_reasoning_tokens),
            ("design_temperature", self.design_temperature),
        ):
            if value is not None:
                data[key] = value
        return data

    def apply(self, settings):
        """Overlay the set behavior knobs onto a resolved ``Settings``."""
        overrides = {
            key: value
            for key, value in (
                ("pipeline", self.pipeline or None),
                ("max_tokens_per_call", self.max_tokens_per_call),
                ("project_llm_budget_usd", self.project_llm_budget_usd),
                ("design_reasoning_tokens", self.design_reasoning_tokens),
                ("design_temperature", self.design_temperature),
            )
            if value is not None
        }
        return replace(settings, **overrides) if overrides else settings


def _int_value(raw, *, minimum: int, maximum: int) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if minimum <= raw <= maximum else None


def _float_value(raw, *, minimum: float, maximum: float) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value if minimum <= value <= maximum else None


def from_dict(data: dict) -> RoutingConfig:
    """Validate a parsed routing file into a ``RoutingConfig``.

    An unknown/empty ``active_profile`` makes the file unusable -> empty config
    (env wins). An individual malformed behavior value is dropped with a
    warning; the profile selection still applies.
    """
    active = str(data.get("active_profile") or "").strip().lower()
    if active not in _known_profiles():
        _warn_once(
            f"active_profile {active!r} is not one of {sorted(_known_profiles())}; "
            "ignoring the file and using the environment"
        )
        return RoutingConfig()

    dropped: list[str] = []

    max_tokens = _int_value(
        data.get("max_tokens_per_call"), minimum=1, maximum=MAX_TOKENS_PER_CALL_CEILING
    )
    if data.get("max_tokens_per_call") is not None and max_tokens is None:
        dropped.append("max_tokens_per_call")
    budget = _float_value(
        data.get("project_llm_budget_usd"), minimum=0.0, maximum=MAX_PROJECT_BUDGET_USD
    )
    if data.get("project_llm_budget_usd") is not None and budget is None:
        dropped.append("project_llm_budget_usd")
    reasoning = _int_value(
        data.get("design_reasoning_tokens"), minimum=0, maximum=MAX_REASONING_TOKENS
    )
    if data.get("design_reasoning_tokens") is not None and reasoning is None:
        dropped.append("design_reasoning_tokens")
    temperature = _float_value(data.get("design_temperature"), minimum=0.0, maximum=1.0)
    if data.get("design_temperature") is not None and temperature is None:
        dropped.append("design_temperature")

    unknown = sorted(set(data) - ALLOWED_KEYS)
    if unknown:
        _warn_once(f"ignoring non-allowlisted key(s): {', '.join(unknown)}")
    if dropped:
        _warn_once(f"ignoring invalid value(s) for: {', '.join(sorted(dropped))}")

    raw_pipeline = str(data.get("pipeline") or "").strip().lower()
    pipeline = raw_pipeline if raw_pipeline in PIPELINES else ""
    if raw_pipeline and not pipeline:
        dropped.append("pipeline")

    return RoutingConfig(
        active_profile=active,
        pipeline=pipeline,
        max_tokens_per_call=max_tokens,
        project_llm_budget_usd=budget,
        design_reasoning_tokens=reasoning,
        design_temperature=temperature,
    )


def load(path: Path | None = None) -> RoutingConfig:
    """Read + validate the routing config; empty config means "env wins"."""
    target = Path(path) if path is not None else default_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return RoutingConfig()
    except OSError as exc:
        _warn_once(f"{target} is unreadable ({exc}); ignoring it and using the environment")
        return RoutingConfig()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _warn_once(f"{target} is not valid JSON ({exc}); ignoring it and using the environment")
        return RoutingConfig()
    if not isinstance(data, dict):
        _warn_once(f"{target} must contain a JSON object; ignoring it and using the environment")
        return RoutingConfig()
    return from_dict(data)


def validate(config: RoutingConfig) -> RoutingConfig:
    """Return the config if every set knob is in bounds, else raise ValueError."""
    if config.pipeline and config.pipeline not in PIPELINES:
        raise ValueError(
            f"pipeline must be one of {list(PIPELINES)}, got {config.pipeline!r}"
        )
    if config.active_profile not in _known_profiles():
        raise ValueError(
            f"active_profile must be one of {sorted(_known_profiles())}, got "
            f"{config.active_profile!r}"
        )
    if (
        config.max_tokens_per_call is not None
        and _int_value(
            config.max_tokens_per_call, minimum=1, maximum=MAX_TOKENS_PER_CALL_CEILING
        )
        is None
    ):
        raise ValueError(
            f"max_tokens_per_call must be an integer in 1..{MAX_TOKENS_PER_CALL_CEILING}"
        )
    if config.project_llm_budget_usd is not None and _float_value(
        config.project_llm_budget_usd, minimum=0.0, maximum=MAX_PROJECT_BUDGET_USD
    ) is None:
        raise ValueError(f"project_llm_budget_usd must be a number in 0..{MAX_PROJECT_BUDGET_USD}")
    if config.design_reasoning_tokens is not None and _int_value(
        config.design_reasoning_tokens, minimum=0, maximum=MAX_REASONING_TOKENS
    ) is None:
        raise ValueError(
            f"design_reasoning_tokens must be an integer in 0..{MAX_REASONING_TOKENS}"
        )
    if config.design_temperature is not None and _float_value(
        config.design_temperature, minimum=0.0, maximum=1.0
    ) is None:
        raise ValueError("design_temperature must be a number in 0..1")
    return config


def save(config: RoutingConfig, path: Path | None = None) -> Path:
    """Atomically write the routing config (temp file + rename)."""
    validate(config)
    target = Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target
