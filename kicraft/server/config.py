"""Environment-driven settings for the KiCraft server, with cost-safety caps.

Defaults are intentionally conservative and sit well under a typical prepaid
balance, so the application self-stops before the provider balance is even
touched (defense in depth behind the prepaid + virtual-card limits).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class CollectionBound:
    """Cardinality policy for one collection in a design-stage response."""

    field: str
    total: int
    per_group: int | None = None
    group_key: str | None = None

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("collection total bound must be positive")
        if (self.per_group is None) != (self.group_key is None):
            raise ValueError("per_group and group_key must be configured together")
        if self.per_group is not None and self.per_group <= 0:
            raise ValueError("collection per-group bound must be positive")


@dataclass(frozen=True)
class ReasoningGuardPolicy:
    """Client-side limits for one reasoning stream."""

    name: str
    hard_max_tokens: int
    repetition_enabled: bool
    repeat_window: int
    repeat_threshold: int
    wall_stall_s: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reasoning policy name must not be empty")
        if self.hard_max_tokens <= 0:
            raise ValueError("reasoning hard ceiling must be positive")
        if self.repeat_window <= 0:
            raise ValueError("reasoning repeat window must be positive")
        if self.repeat_threshold < 2:
            raise ValueError("reasoning repeat threshold must be at least two")
        if self.wall_stall_s <= 0:
            raise ValueError("reasoning wall-stall ceiling must be positive")


BOM_TOTAL_PART_LIMIT = 500
BOM_SHEET_PART_LIMIT = 450


STAGE_COLLECTION_BOUNDS: dict[str, tuple[CollectionBound, ...]] = {
    "bom": (
        CollectionBound(
            field="groups",
            total=BOM_TOTAL_PART_LIMIT,
            per_group=BOM_SHEET_PART_LIMIT,
            group_key="sheet",
        ),
    ),
}


# Fixed output cap (tokens) for the serialization recovery call per design
# stage: one plain, tool-free, reasoning-disabled re-emission of the slot
# after a parse failure. The model re-serializes content it already drafted,
# so the cap is roughly double the normal floor — finite, and NEVER doubled
# dynamically (KC-7FVTPW: truncation used to double the cap up to 32,768 and
# still produced no JSON). The big-slot stages (bom/wiring) get headroom; the
# small ones stay cheap.
STAGE_SERIALIZATION_MAX_TOKENS = {
    "intent": 8192,
    "functional_spec": 8192,
    "architecture": 16384,
    "bom": 32768,
    "wiring": 32768,
}

DESIGN_PROFILES: dict[str, dict[str, object]] = {
    "flash": {
        "model": "deepseek/deepseek-v4-flash-0731",
        "provider_order": ["deepinfra/fp8"],
        "max_price_prompt": 0.11,
        "max_price_completion": 0.24,
    },
    "pro": {
        "model": "deepseek/deepseek-v4-pro-0813",
        "provider_order": ["alibaba"],
        "max_price_prompt": 1.46,
        "max_price_completion": 4.38,
    },
}


def _resolved_design_profile() -> tuple[str, dict[str, object]]:
    """Resolve and validate the operator-selected designer profile."""
    name = os.environ.get("KICRAFT_DESIGN_PROFILE", "flash").strip().lower()
    if name not in DESIGN_PROFILES:
        raise SystemExit(
            f"KICRAFT_DESIGN_PROFILE must be one of {sorted(DESIGN_PROFILES)}, got {name!r}"
        )
    profile = DESIGN_PROFILES[name]
    checks = {
        "KICRAFT_MODEL": ("model", str),
        "KICRAFT_PROVIDER_ORDER": (
            "provider_order",
            lambda raw: [part.strip() for part in raw.split(",") if part.strip()],
        ),
        "KICRAFT_MAX_PRICE_PROMPT": ("max_price_prompt", float),
        "KICRAFT_MAX_PRICE_COMPLETION": ("max_price_completion", float),
    }
    for env_name, (key, parse) in checks.items():
        raw = os.environ.get(env_name)
        if raw is None or not raw.strip():
            continue
        actual = parse(raw.strip())
        if actual != profile[key]:
            raise SystemExit(
                f"{env_name}={actual!r} conflicts with design profile {name!r} "
                f"({profile[key]!r}); select another profile instead of mixing routes"
            )
    return name, profile


@dataclass(frozen=True)
class StageResponsePolicy:
    """Immutable response policy for one design-stage drive.

    One value per stage so the normal output cap, the normal reasoning payload,
    and the serialization recovery cap/retry budget travel together. The
    serialization retry count is fixed at one for design stages: a parse
    failure may trigger at most one plain, tool-free, reasoning-disabled
    completion at the fixed serialization cap — never repeated tool loops and
    never dynamic cap growth.
    """

    normal_max_tokens: int
    normal_reasoning: dict | None
    serialization_max_tokens: int
    serialization_retries: int = 1
    collection_bounds: tuple[CollectionBound, ...] = ()
    reasoning_guard: ReasoningGuardPolicy | None = None


@dataclass
class Settings:
    """Resolved server configuration. Build with `Settings.from_env()`."""

    api_key: str
    model: str = "deepseek/deepseek-v4-flash-0731"
    design_profile: str = "custom"
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
    # Bounded retry on TRANSIENT OpenRouter failures (HTTP 5xx / 429 / connection
    # reset / timeout) before any token is streamed. A one-off 503 dropped a whole
    # brief from a self-eval batch (#24 daq-8ch) and masqueraded as a design
    # failure; a few backed-off retries make a transient blip invisible. The retry
    # only fires before streaming begins, so it can never double-emit tokens. 0
    # disables. KICRAFT_LLM_MAX_RETRIES / KICRAFT_LLM_RETRY_BACKOFF_S.
    llm_max_retries: int = 3
    llm_retry_backoff_s: float = 1.0
    # Sampling temperature for the DESIGN stages (intent..wiring). Was hardcoded
    # 0.2; lowering toward 0 cuts run-to-run variance (the self-eval noise floor),
    # making real regressions legible. KICRAFT_DESIGN_TEMPERATURE.
    design_temperature: float = 0.0
    # Serialization recovery budget per stage drive: after the first
    # truncated_json / invalid_json, exactly one plain, tool-free,
    # reasoning-disabled completion at `serialization_max_tokens[stage]` is
    # allowed; a second malformed reply is terminal. Fixed at one for design
    # stages. KICRAFT_SERIALIZATION_RETRIES.
    serialization_retries: int = 1
    # Recovery must escape the deterministic sequence that produced the first
    # malformed/overflowing serialization while retaining the same finite cap.
    serialization_escape_temperature: float = 0.4
    # Fixed output cap for that serialization call, per stage (see
    # STAGE_SERIALIZATION_MAX_TOKENS). Never doubled dynamically.
    serialization_max_tokens: dict = field(
        default_factory=lambda: dict(STAGE_SERIALIZATION_MAX_TOKENS)
    )
    # Parse-side degeneration guards. Tuples keep the per-drive policy immutable.
    collection_bounds: dict[str, tuple[CollectionBound, ...]] = field(
        default_factory=lambda: dict(STAGE_COLLECTION_BOUNDS)
    )

    # --- Design-stage reasoning budget + in-stream loop breaker ---------------
    # Reasoning budget for the design stages. Intent/functional_spec (small,
    # serialization-critical, and the observed loop site) always run with the
    # reasoning channel DISABLED; the topology/part/netlist stages get a small
    # budget so deliberation is bounded. 0 disables reasoning for ALL design
    # stages. KICRAFT_DESIGN_REASONING_TOKENS.
    design_reasoning_tokens: int = 2048
    # Hard per-call reasoning ceiling enforced IN-STREAM by the client (provider-
    # independent): a reasoning-only stream that exceeds this many tokens with no
    # answer content is aborted. max_tokens does NOT bound DeepSeek's reasoning
    # channel, so this is the real stop on an unbounded reasoning loop.
    # KICRAFT_REASONING_MAX_TOKENS.
    reasoning_max_tokens: int = 4096
    # Repetition fingerprint: the trailing window (chars) that, when seen this
    # many times verbatim in the recent reasoning buffer, marks a stuck loop
    # before it reaches the token ceiling. KICRAFT_REASONING_REPEAT_WINDOW /
    # KICRAFT_REASONING_REPEAT_THRESHOLD.
    reasoning_repeat_window: int = 256
    reasoning_repeat_threshold: int = 3

    # Judge/review streams legitimately reason far beyond the design ceiling.
    # Their independent finite ceilings preserve cost safety without applying
    # the design repetition canary to long-form analysis.
    eval_judge_reasoning_max_tokens: int = 32768
    eval_judge_wall_stall_s: int = 360
    review_reasoning_max_tokens: int = 32768
    review_wall_stall_s: int = 360

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
    # ``Settings.from_env`` resolves the selected dated designer profile into
    # these existing fields before client construction. Direct Settings(...)
    # construction remains available to tests and one-off admin tools as the
    provider_order: list[str] = field(default_factory=lambda: ["deepinfra/fp8"])
    provider_allow_fallbacks: bool = False
    max_price_prompt: float = 0.11
    max_price_completion: float = 0.24
    enable_prompt_cache: bool = True

    # Surface the core-components registry (admin-curated default part per
    # functional block, /admin/core-components) in the architecture/bom prompts
    # so the model adopts defaults instead of researching them. KICRAFT_CORE_DEFAULTS=0
    # disables for A/B cost comparisons.
    enable_core_defaults: bool = True

    # --- Class-J self-evaluation judge (admin-only web feature) ---------------
    # Role identity is explicit and independent from electrical review even
    # while both production roles use the same incumbent model.
    eval_judge_model: str = "minimax/minimax-m3"
    # Answer-token budget for the Class-J eval judge. Reasoning-heavy judges can
    # consume 10-23k reasoning tokens before the structured answer.
    eval_judge_max_tokens: int = 24000
    # --- Layer-3 electrical-review pass (the in-product design "judge") --------
    # Reviews a committed design for topology/value/completeness defects the
    # deterministic §9 gates cannot judge. Runs the DESIGN model by default
    # (deepseek-v4-flash -- cheap; Claude is prohibitively expensive for a
    # product) but with a higher THINKING BUDGET, since the review is a one-shot
    # reasoning task where extra deliberation is worth far more than it costs.
    # review_model=None reuses `model`. review_reasoning_tokens is the OpenRouter
    # reasoning max_tokens budget (0 disables the reasoning channel).
    # Bakeoff winner (2026-06-19): minimax-m3 gives 100% blocker recall + the
    # lowest over-block (14% clean / 20% warn) vs flash's 83% / 43-71%, at
    # ~$0.012 & ~2.5 min per review (flash: ~$0.001 / 35 s). The gate is
    # once-per-build and fail-soft, so the latency is an acceptable trade. None
    # reuses the design model. See docs/electrical_review_model_bakeoff.md.
    review_model: str | None = "minimax/minimax-m3"
    review_reasoning_tokens: int = 8000
    # Reasoning effort for the review (OpenRouter; portable across the slate --
    # minimax/glm prefer effort and some models 400 on the token form). When
    # non-empty it is used INSTEAD of review_reasoning_tokens.
    # KICRAFT_REVIEW_REASONING_EFFORT='' falls back to the token budget.
    review_reasoning_effort: str = "medium"
    # Answer-token budget for the review. Raised from 3000 -> 24000: the 2026
    # reasoning models emit 10-23k reasoning tokens by default and were
    # truncating (finish=length) before writing the JSON answer. Cheap models
    # stop naturally well under this, so it costs only the heavy reasoners.
    # KICRAFT_REVIEW_MAX_TOKENS.
    review_max_tokens: int = 24000
    # Provider routing for role calls is independent from the designer profile.
    # The incumbent reviewer and judge use the same dated service today, but
    # retain separate fields so either can be promoted independently.
    review_provider_order: list[str] = field(default_factory=lambda: ["coreweave/fp4"])
    review_max_price_prompt: float = 0.30
    review_max_price_completion: float = 1.25
    judge_provider_order: list[str] = field(default_factory=lambda: ["coreweave/fp4"])
    judge_max_price_prompt: float = 0.30
    judge_max_price_completion: float = 1.25
    # Lazy corroboration of the review gate: a blocker-eligible blocker hard-blocks
    # only if `review_corroboration` passes agree on it (same category + refdes),
    # else it demotes to a warning. Pass 2+ runs ONLY when pass 1 proposes such a
    # blocker, so clean/warning-only builds still cost one pass. 1 = legacy single
    # pass. review_temperature>0 makes the passes independent enough to corroborate
    # (0.0 makes them near-identical -> a near no-op). KICRAFT_REVIEW_CORROBORATION
    # / KICRAFT_REVIEW_TEMPERATURE.
    review_corroboration: int = 2
    review_temperature: float = 0.5
    # Layer-4 fab gate: run the electrical review during build verify and block
    # a structurally-sound board from being declared fab-ready if the review
    # finds a blocker. ON by default -- catching an electrically-wrong board is
    # worth one cheap LLM call per build. Set KICRAFT_ELECTRICAL_REVIEW=0 to
    # disable (the gate is also fail-soft: any infra/parse error skips it).
    enable_electrical_review: bool = True

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
        profile_name, profile = _resolved_design_profile()
        return cls(
            api_key=key,
            model=str(profile["model"]),
            design_profile=profile_name,
            max_tokens_per_call=int(
                os.environ.get("KICRAFT_MAX_TOKENS_PER_CALL", cls.max_tokens_per_call)
            ),
            daily_usd_ceiling=float(
                os.environ.get("KICRAFT_DAILY_USD_CEILING", cls.daily_usd_ceiling)
            ),
            total_usd_ceiling=float(
                os.environ.get("KICRAFT_TOTAL_USD_CEILING", cls.total_usd_ceiling)
            ),
            ledger_path=Path(os.environ.get("KICRAFT_SPEND_LEDGER", str(cls.ledger_path))),
            users_db_path=Path(os.environ.get("KICRAFT_USERS_DB", str(cls.users_db_path))),
            projects_dir=Path(os.environ.get("KICRAFT_PROJECTS_DIR", str(cls.projects_dir))),
            work_dir=Path(os.environ.get("KICRAFT_WORK_DIR", str(cls.work_dir))),
            legal_dir=Path(os.environ.get("KICRAFT_LEGAL_DIR", str(cls.legal_dir))),
            kill_switch=_env_bool("KICRAFT_KILL_SWITCH"),
            llm_max_retries=int(os.environ.get("KICRAFT_LLM_MAX_RETRIES", cls.llm_max_retries)),
            llm_retry_backoff_s=float(
                os.environ.get("KICRAFT_LLM_RETRY_BACKOFF_S", cls.llm_retry_backoff_s)
            ),
            design_temperature=float(
                os.environ.get("KICRAFT_DESIGN_TEMPERATURE", cls.design_temperature)
            ),
            serialization_retries=int(
                os.environ.get("KICRAFT_SERIALIZATION_RETRIES", cls.serialization_retries)
            ),
            serialization_escape_temperature=float(
                os.environ.get(
                    "KICRAFT_SERIALIZATION_ESCAPE_TEMPERATURE",
                    cls.serialization_escape_temperature,
                )
            ),
            serialization_max_tokens=dict(STAGE_SERIALIZATION_MAX_TOKENS),
            collection_bounds=dict(STAGE_COLLECTION_BOUNDS),
            design_reasoning_tokens=int(
                os.environ.get("KICRAFT_DESIGN_REASONING_TOKENS", cls.design_reasoning_tokens)
            ),
            reasoning_max_tokens=int(
                os.environ.get("KICRAFT_REASONING_MAX_TOKENS", cls.reasoning_max_tokens)
            ),
            reasoning_repeat_window=int(
                os.environ.get("KICRAFT_REASONING_REPEAT_WINDOW", cls.reasoning_repeat_window)
            ),
            reasoning_repeat_threshold=int(
                os.environ.get("KICRAFT_REASONING_REPEAT_THRESHOLD", cls.reasoning_repeat_threshold)
            ),
            eval_judge_reasoning_max_tokens=int(
                os.environ.get(
                    "KICRAFT_EVAL_JUDGE_REASONING_MAX_TOKENS",
                    cls.eval_judge_reasoning_max_tokens,
                )
            ),
            eval_judge_wall_stall_s=int(
                os.environ.get(
                    "KICRAFT_EVAL_JUDGE_WALL_STALL_S",
                    cls.eval_judge_wall_stall_s,
                )
            ),
            review_reasoning_max_tokens=int(
                os.environ.get(
                    "KICRAFT_REVIEW_REASONING_MAX_TOKENS",
                    cls.review_reasoning_max_tokens,
                )
            ),
            review_wall_stall_s=int(
                os.environ.get(
                    "KICRAFT_REVIEW_WALL_STALL_S",
                    cls.review_wall_stall_s,
                )
            ),
            public_url=os.environ.get("KICRAFT_PUBLIC_URL", cls.public_url).strip().rstrip("/")
            or cls.public_url,
            email_from=(
                os.environ.get("KICRAFT_EMAIL_FROM", "").strip()
                or os.environ.get("KICRAFT_SMTP_FROM", "").strip()
                or os.environ.get("KICRAFT_SMTP_USERNAME", "").strip()
            ),
            resend_api_key=os.environ.get("KICRAFT_RESEND_API_KEY", "").strip(),
            smtp_host=os.environ.get("KICRAFT_SMTP_HOST", "").strip(),
            smtp_port=int(os.environ.get("KICRAFT_SMTP_PORT", cls.smtp_port)),
            smtp_username=os.environ.get("KICRAFT_SMTP_USERNAME", "").strip(),
            smtp_password=os.environ.get("KICRAFT_SMTP_PASSWORD", ""),
            smtp_from=(
                os.environ.get("KICRAFT_SMTP_FROM", "").strip()
                or os.environ.get("KICRAFT_SMTP_USERNAME", "").strip()
            ),
            smtp_starttls=_env_bool_default("KICRAFT_SMTP_STARTTLS", True),
            smtp_ssl=_env_bool("KICRAFT_SMTP_SSL"),
            stripe_secret_key=os.environ.get("KICRAFT_STRIPE_SECRET_KEY", "").strip(),
            stripe_webhook_secret=os.environ.get("KICRAFT_STRIPE_WEBHOOK_SECRET", "").strip(),
            stripe_price_pro=os.environ.get("KICRAFT_STRIPE_PRICE_PRO", "").strip(),
            stripe_price_max=os.environ.get("KICRAFT_STRIPE_PRICE_MAX", "").strip(),
            provider_order=list(profile["provider_order"]),
            provider_allow_fallbacks=False,
            max_price_prompt=float(profile["max_price_prompt"]),
            max_price_completion=float(profile["max_price_completion"]),
            enable_prompt_cache=_env_bool_default("KICRAFT_ENABLE_PROMPT_CACHE", True),
            enable_core_defaults=_env_bool_default("KICRAFT_CORE_DEFAULTS", True),
            eval_judge_model=(
                os.environ.get("KICRAFT_EVAL_JUDGE_MODEL", "").strip() or cls.eval_judge_model
            ),
            eval_judge_max_tokens=int(
                os.environ.get("KICRAFT_EVAL_JUDGE_MAX_TOKENS", cls.eval_judge_max_tokens)
            ),
            review_model=(os.environ.get("KICRAFT_REVIEW_MODEL", "").strip() or cls.review_model),
            review_reasoning_tokens=int(
                os.environ.get("KICRAFT_REVIEW_REASONING_TOKENS", cls.review_reasoning_tokens)
            ),
            review_reasoning_effort=os.environ.get(
                "KICRAFT_REVIEW_REASONING_EFFORT", cls.review_reasoning_effort
            ).strip(),
            review_max_tokens=int(
                os.environ.get("KICRAFT_REVIEW_MAX_TOKENS", cls.review_max_tokens)
            ),
            review_provider_order=[
                p.strip()
                for p in os.environ.get("KICRAFT_REVIEW_PROVIDER_ORDER", "coreweave/fp4").split(",")
                if p.strip()
            ],
            review_max_price_prompt=float(
                os.environ.get("KICRAFT_REVIEW_MAX_PRICE_PROMPT", cls.review_max_price_prompt)
            ),
            review_max_price_completion=float(
                os.environ.get(
                    "KICRAFT_REVIEW_MAX_PRICE_COMPLETION", cls.review_max_price_completion
                )
            ),
            judge_provider_order=[
                p.strip()
                for p in os.environ.get("KICRAFT_EVAL_JUDGE_PROVIDER_ORDER", "coreweave/fp4").split(
                    ","
                )
                if p.strip()
            ],
            judge_max_price_prompt=float(
                os.environ.get("KICRAFT_EVAL_JUDGE_MAX_PRICE_PROMPT", cls.judge_max_price_prompt)
            ),
            judge_max_price_completion=float(
                os.environ.get(
                    "KICRAFT_EVAL_JUDGE_MAX_PRICE_COMPLETION",
                    cls.judge_max_price_completion,
                )
            ),
            review_corroboration=int(
                os.environ.get("KICRAFT_REVIEW_CORROBORATION", cls.review_corroboration)
            ),
            review_temperature=float(
                os.environ.get("KICRAFT_REVIEW_TEMPERATURE", cls.review_temperature)
            ),
            enable_electrical_review=_env_bool_default("KICRAFT_ELECTRICAL_REVIEW", True),
        )

    def for_review(self) -> "Settings":
        """Return the independently capped electrical-review route."""
        return replace(
            self,
            provider_order=self.review_provider_order,
            provider_allow_fallbacks=False,
            max_price_prompt=self.review_max_price_prompt,
            max_price_completion=self.review_max_price_completion,
        )

    def for_judge(self) -> "Settings":
        """Return the independently capped Class-J route."""
        return replace(
            self,
            provider_order=self.judge_provider_order,
            provider_allow_fallbacks=False,
            max_price_prompt=self.judge_max_price_prompt,
            max_price_completion=self.judge_max_price_completion,
        )

    def review_reasoning(self) -> dict | None:
        """OpenRouter reasoning control for the review: effort-based when
        review_reasoning_effort is set (portable across the slate; minimax/glm
        prefer effort and some models 400 on the token form), else a token budget."""
        if self.review_reasoning_effort:
            return {"effort": self.review_reasoning_effort}
        if self.review_reasoning_tokens:
            return {"max_tokens": self.review_reasoning_tokens}
        return None

    def design_reasoning_guard(self) -> ReasoningGuardPolicy:
        return ReasoningGuardPolicy(
            name="design",
            hard_max_tokens=self.reasoning_max_tokens,
            repetition_enabled=True,
            repeat_window=self.reasoning_repeat_window,
            repeat_threshold=self.reasoning_repeat_threshold,
            wall_stall_s=self.request_timeout_s,
        )

    def judge_reasoning_guard(self) -> ReasoningGuardPolicy:
        return ReasoningGuardPolicy(
            name="eval_judge",
            hard_max_tokens=self.eval_judge_reasoning_max_tokens,
            repetition_enabled=False,
            repeat_window=self.reasoning_repeat_window,
            repeat_threshold=self.reasoning_repeat_threshold,
            wall_stall_s=self.eval_judge_wall_stall_s,
        )

    def review_reasoning_guard(self) -> ReasoningGuardPolicy:
        return ReasoningGuardPolicy(
            name="electrical_review",
            hard_max_tokens=self.review_reasoning_max_tokens,
            repetition_enabled=False,
            repeat_window=self.reasoning_repeat_window,
            repeat_threshold=self.reasoning_repeat_threshold,
            wall_stall_s=self.review_wall_stall_s,
        )

    def design_reasoning(self, stage: str) -> dict | None:
        """OpenRouter reasoning control for a design stage."""
        if stage in ("intent", "functional_spec", "wiring"):
            return {"enabled": False}
        if self.design_reasoning_tokens <= 0:
            return {"enabled": False}
        return {"max_tokens": self.design_reasoning_tokens}

    def design_stage_policy(self, stage: str, normal_max_tokens: int) -> StageResponsePolicy:
        """Immutable response policy for one design-stage drive.

        ``normal_max_tokens`` is the caller's already-floored normal output cap;
        the serialization cap is the stage's fixed value from
        ``serialization_max_tokens`` (never doubled). The normal reasoning
        payload comes from :meth:`design_reasoning`, the compatibility source.
        """
        return StageResponsePolicy(
            normal_max_tokens=int(normal_max_tokens),
            normal_reasoning=self.design_reasoning(stage),
            serialization_max_tokens=int(
                self.serialization_max_tokens.get(stage)
                or STAGE_SERIALIZATION_MAX_TOKENS.get(stage, 8192)
            ),
            serialization_retries=max(0, int(self.serialization_retries)),
            collection_bounds=tuple(self.collection_bounds.get(stage, ())),
            reasoning_guard=self.design_reasoning_guard(),
        )

    @property
    def billing_enabled(self) -> bool:
        """Whether Stripe billing is fully configured (key, webhook secret, and
        both tier price ids). Anything less and the paid-tier checkout is
        hidden, so a partially configured box can never half-charge a card."""
        return bool(
            self.stripe_secret_key
            and self.stripe_webhook_secret
            and self.stripe_price_pro
            and self.stripe_price_max
        )

    def redacted(self) -> dict:
        """Settings safe to display/log (without the secret key)."""
        return {
            "model": self.model,
            "design_profile": self.design_profile,
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
            "design_temperature": self.design_temperature,
            "review_model": self.review_model,
            "review_provider_order": self.review_provider_order,
            "review_max_price_prompt": self.review_max_price_prompt,
            "review_max_price_completion": self.review_max_price_completion,
            "judge_provider_order": self.judge_provider_order,
            "judge_max_price_prompt": self.judge_max_price_prompt,
            "judge_max_price_completion": self.judge_max_price_completion,
            "llm_max_retries": self.llm_max_retries,
        }
