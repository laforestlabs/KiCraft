"""Single-stage preparation, provider calls, retries, commits, and finalization."""

from __future__ import annotations

import json
import re
import resource
import time
from dataclasses import asdict, dataclass
from typing import Literal

import requests

from kicraft.design import models
from kicraft.design.stage_semantics import diagnose_stage

from .config import STAGE_COLLECTION_BOUNDS, STAGE_SERIALIZATION_MAX_TOKENS, StageResponsePolicy
from .client import classify_provider_exception
from .stage_bom_tools import BOM_TOOLS, build_bom_executor
from .stage_contracts import (
    StageResponseContract,
    StageSchemaError,
    _extract_json,
    _normalize_stage_response,
    build_stage_response_contract,
)
from .stage_prompts import (
    _bom_part_hints,
    _collection_bounds_sentence,
    _format_core_defaults_block,
    build_system,
)
from .stage_state_io import (
    KICRAFT,
    attach_questions,
    commit_stage,
    committed_bom_refs,
    prepare_stage,
    run_design_cli,
    stamp_stage_status,
)

# Provider/transport failure families that survive the client's own bounded
# transport retries and surface here as terminal stage failures. JSON recovery
# never sees them, and BudgetExceeded / KillSwitchEngaged (budget failures
# owned by SpendGuard) are deliberately NOT caught — they propagate to the
# guard/caller path.
_TRANSPORT_FAILURE_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_PROVIDER_FAILURE_EXC = (requests.exceptions.HTTPError,)


def _child_cpu_s() -> float:
    """User+system CPU seconds consumed by this process's child subprocesses
    (the stage-prep/commit calls and BOM tool lookups). RUSAGE_CHILDREN accumulates
    over the whole process, so the driver snapshots a before/after delta per stage.
    On non-POSIX this reports 0 (resource.RUSAGE_CHILDREN is unavailable); the
    ledger column then stays null.

    CAVEAT — reliable only single-flight: RUSAGE_CHILDREN is per-PROCESS, not
    per-thread. The web app runs designs in concurrent _run_design threads in one
    process, so when two designs are in flight the stage windows overlap and each
    one's cpu_s delta absorbs the other's subprocess CPU. wall_s (a monotonic
    delta) stays correct under concurrency; cpu_s does not. Trust cpu_s only for
    serial measurement (one design at a time, e.g. a single self-eval), and read
    the aggregate cpu/wall ratio as a rough latency-vs-CPU signal, not an exact
    per-stage figure. A future fix could tag each stage_runs row as
    cpu-contended when other stages overlapped its window."""
    try:
        u = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, ValueError):
        return 0.0
    return float(u.ru_utime + u.ru_stime)


def _record_stage_ledger(client, *, run_id, stage, **kw) -> None:
    """Best-effort write to the spend ledger's ``stage_runs`` table. Real clients
    carry a ``guard`` (SpendGuard) that owns ``record_stage``; the mock/replay
    client's guard does not, so this is a silent no-op there."""
    guard = getattr(client, "guard", None)
    if guard is None or not hasattr(guard, "record_stage"):
        return
    try:
        guard.record_stage(run_id=run_id, stage=stage, **kw)
    except Exception:  # ledger trouble must never fail a design run
        pass


def _record_stage_attempt(client, *, run_id, stage, **kw) -> None:
    guard = getattr(client, "guard", None)
    if guard is None or not hasattr(guard, "record_stage_attempt"):
        return
    try:
        guard.record_stage_attempt(run_id=run_id, stage=stage, **kw)
    except Exception:
        pass


def _record_attempt_facts(
    client,
    *,
    run_id,
    stage,
    attempt,
    call_mode,
    outcome,
    facts=None,
    error_facts=None,
    diagnostic_codes=(),
) -> None:
    usage = (facts.usage or {}) if facts is not None else {}
    _record_stage_attempt(
        client,
        run_id=run_id,
        stage=stage,
        attempt=attempt,
        call_mode=call_mode,
        model=_client_model(client),
        provider=facts.provider if facts is not None else None,
        finish_reason=facts.finish if facts is not None else None,
        outcome=outcome,
        wall_s=facts.wall_s if facts is not None else None,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cost_usd=facts.cost_usd if facts is not None else 0.0,
        diagnostic_codes=diagnostic_codes,
        **(error_facts or {}),
    )


# Per-stage self-correction budget. Wiring must satisfy whole-board net coverage
# (§9.11) in a single slot; on a complex board the model needs more correction
# passes than the simpler, smaller-slot stages, so they floor higher (BOM must
# also resolve every symbol/footprint to a real library entry within its budget).
_STAGE_MIN_RETRIES = {"wiring": 4, "bom": 4}

# In-stream reasoning-loop breakout budget: when the client aborts a completion
# (finish_reason="reasoning_loop"), retry once with reasoning disabled + higher
# temperature to escape the deterministic cycle. A second loop in a row means the
# model cannot serialize even without reasoning -- fail with an explicit
# "reasoning_loop" label rather than "no JSON in reply".
_MAX_LOOP_RETRIES = 1


def _stage_max_retries(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_RETRIES.get(stage, 0))


# Tool-loop round budget for the BOM stage. The default (12) lets a weak model
# burn a dozen round-trips re-verifying a trivial 9-part BOM; 6 is plenty to
# resolve real parts, and client.chat_with_tools converges earlier when the
# model thrashes (identical-call cache + forced-final). Each stage attempt gets
# its own loop, so this is per-attempt.
_BOM_MAX_ROUNDS = 6


# Per-stage output token budget. Wiring emits the whole-board netlist in one
# slot; BOM for a large array (200 LEDs + 200 decoupling caps = 401 parts)
# emits every part in one JSON object. Both overflow the default cap and
# truncate into invalid JSON ("no JSON in reply"), so they floor higher.
_STAGE_MIN_TOKENS = {"wiring": 8192, "bom": 16384}


def _stage_max_tokens(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_TOKENS.get(stage, 0))


def _retry_feedback(
    out: dict, *, stage: str | None = None, valid_refs: list[str] | None = None
) -> str:
    """Build correction feedback for a complete same-schema replacement.

    The model always returns the stage's ordinary response shape. There is no
    correction-only patch language or alternate response contract.
    """
    msg = f"stage-commit rejected that with errors: {json.dumps(out.get('errors'))}"
    if out.get("offenders"):
        msg += f"  offenders: {json.dumps(out.get('offenders'))}"
        shown = len(out.get("offenders") or [])
        total = int(out.get("offenders_total") or 0)
        if total > shown:
            # Without the total, the model fixed the visible slice, got
            # bounced with a DIFFERENT slice, and burned the retry budget
            # chasing a moving target (2026-07-19 review §5.5).
            msg += (
                f"  NOTE: only {shown} of {total} offenders are shown -- "
                "fix ALL instances of this defect class across the whole "
                "slot, not just the ones listed."
            )
    msg += (
        ". Return the COMPLETE corrected slot JSON, preserving every entry that was "
        "already valid and changing only the rejected items. When an offender lists "
        "'real options: ...', use one exact option verbatim; otherwise use the BOM "
        "lookup tools. Use compact JSON and output only the slot object."
    )
    # Unknown-ref in wiring means the model tried to wire a part the BOM lacks --
    # it cannot add parts, so retrying with an invented ref just re-fails. Point
    # it at the real refs and the reconcile escape hatch so it stops thrashing and
    # escalates the deficit instead of burning the retry budget (WS6).
    if stage == "wiring" and "unknown ref" in json.dumps(out.get("errors") or ""):
        msg += (
            " NOTE: the wiring stage can ONLY connect refs the BOM already contains -- it "
            "CANNOT add parts. Do not invent a ref. If a part you need is genuinely missing "
            "from the BOM, do NOT wire a made-up ref: instead PARK with a single blocking "
            'question whose "reconcile_target" is "bom", naming the missing part and the '
            "IC pins it serves; the pipeline will add it and re-run wiring."
        )
        if valid_refs:
            msg += f" The only refs you may reference are: {valid_refs}."
    # A power/ground name used as a component ref fails the endpoint shape. The
    # final-pin contract puts the rail in ``net`` and the component in ``ref``.
    if stage == "wiring":
        errs = json.dumps(out.get("errors") or "")
        rails = {
            m.group(1)
            for m in re.finditer(r"PinEndpoint\.ref '([^']+)' must match", errs)
            if models.is_power_or_ground_name(m.group(1))
        }
        if rails:
            msg += (
                " NOTE: "
                + ", ".join(sorted(rails))
                + ' is a net name. Use {"ref": "R1", "pin": "2", "net": "+3V3"}; '
                "never put a rail name in ref."
            )
        rejection_text = json.dumps([*(out.get("errors") or []), *(out.get("offenders") or [])])
        if "9.15 no dangling signal nets" in rejection_text:
            msg += (
                " NOTE: if the sole pin is one terminal of a two-terminal series part, "
                "do not rename that terminal or put both part pins on one net. Complete the "
                "path by moving the destination pin that currently shares the OTHER terminal's "
                "net onto the dangling terminal's net. Pattern: source + Rn.1 = SIG_IN; "
                "Rn.2 + destination = SIG_OUT. Both SIG_IN and SIG_OUT must have two pins."
            )
        if "9.17 two-terminal self-short" in rejection_text:
            msg += (
                " NOTE: fix a self-shorted series part as one complete three-item change: "
                "(1) keep one terminal on the source net, (2) put the other terminal on a new "
                "local net, and (3) MOVE the intended destination IC/connector pin from the "
                "source net onto that new local net. Do not merely rename one part terminal; "
                "that creates a 9.15 dangling net. Required pattern: source + Rn.1 = SIG_IN; "
                "Rn.2 + destination = SIG_OUT."
            )
    return msg


def _offender_identity(raw: object) -> str:
    text = re.sub(r"\s+", " ", str(raw)).strip()
    pins = {
        f"{match.group(1).upper()}.{match.group(2).upper()}"
        for match in re.finditer(
            r"\b([A-Za-z]+[0-9]+[A-Za-z0-9_-]*)(?:\.|\s+pin\s+)([A-Za-z0-9~_+-]+)\b",
            text,
        )
    }
    if pins:
        return "|".join(sorted(pins))
    refs = set(re.findall(r"\b[A-Z]+[0-9]+[A-Z0-9_-]*\b", text))
    if refs:
        return "|".join(sorted(refs))
    quoted = {
        item.strip() for item in re.findall(r"['\"]([^'\"]{1,64})['\"]", text) if item.strip()
    }
    return "|".join(sorted(quoted)) if quoted else text


def _commit_rejection_signature(out: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Stable ordered gate IDs plus offender identities."""
    errors = [re.sub(r"\s+", " ", str(item)).strip() for item in (out.get("errors") or [])]
    gate_ids: list[str] = []
    for error in errors:
        for gate_id in re.findall(r"(?:§\s*)?(9\.\d+)", error):
            if gate_id not in gate_ids:
                gate_ids.append(gate_id)
    if not gate_ids:
        gate_ids = errors
    offenders = tuple(sorted(_offender_identity(item) for item in (out.get("offenders") or [])))
    return tuple(gate_ids), offenders


def _normalize_questions(raw_list, stage: str) -> list[dict]:
    """Coerce a model-emitted questions payload into Question-shaped dicts (so the
    state.json open_questions list stays schema-valid). Caps count and lengths."""
    out = []
    for q in raw_list:
        if isinstance(q, dict) and str(q.get("text", "")).strip():
            # reconcile_target marks a deficit the pipeline repairs itself (re-drive
            # the named stage) rather than a question for the user. Whitelisted so
            # the model can't route a park to an arbitrary/looping target.
            target = q.get("reconcile_target")
            out.append(
                {
                    "text": str(q["text"]).strip()[:500],
                    "stage": stage,
                    "blocking": bool(q.get("blocking", True)),
                    "material": bool(q.get("material", True)),
                    "options": [str(o)[:200] for o in (q.get("options") or [])][:6],
                    "answer": None,
                    "reconcile_target": (target if target in ("bom",) else None),
                }
            )
    return out[:5]


def _client_model(client) -> str | None:
    """Best-effort display name of the model a client will call (shown in the UI)."""
    return getattr(getattr(client, "s", None), "model", None)


def _design_temperature(client) -> float:
    """Sampling temperature for the design stages, from settings (default 0.2 when
    a client carries no settings, e.g. the mock). Lowering it toward 0 cuts the
    run-to-run variance that makes self-eval regressions hard to read."""
    return float(getattr(getattr(client, "s", None), "design_temperature", 0.2))


def _design_reasoning(client, stage: str) -> dict | None:
    """OpenRouter reasoning control for a design stage, from the client settings.
    A mock (or a settings object without the policy method) yields None = no
    reasoning control, which is also safe."""
    fn = getattr(getattr(client, "s", None), "design_reasoning", None)
    return fn(stage) if callable(fn) else None


# A reasoning model can burn its whole output budget re-deriving one decision and
# emit NO content (finish_reason="length" with empty text). That is not a truncated
# JSON answer; it is a stuck reasoning loop. Doubling max_tokens only feeds the loop,
# and greedy decoding (design_temperature=0.0) reproduces it identically next attempt.
# Detect the signature and break it instead: keep the budget, raise temperature to
# escape the deterministic cycle, tell the model to commit. (KC-B7MB7P: architecture
# looped for thousands of tokens on the GND-sheet question.)
_REASONING_LOOP_RETRY_MSG = (
    "You spent your entire output budget reconsidering the same decision and "
    "produced no JSON at all. Stop re-deriving it: commit to your first choice, "
    "record any default in 'assumptions' ending '(defaulted)', and output ONLY the "
    "slot JSON now."
)


def _classify_parse_failure(finish, had_content) -> str:
    """Classify a failed JSON parse into a ``failure_kind`` (no recovery decision).

    ``finish="length"`` with NO answer content is provider reasoning/output
    exhaustion, not a truncated JSON answer: it follows the reasoning-recovery
    path (KC-B7MB7P) and is labeled ``reasoning_loop``, never ``invalid_json``.
    ``finish="length"`` with content is a genuinely truncated answer
    (``truncated_json``). Any other malformed/empty normal stop is
    ``invalid_json``.
    """
    if finish == "length" and not had_content:
        return "reasoning_loop"
    if finish == "length":
        return "truncated_json"
    return "invalid_json"


# Human-readable error strings DERIVED from the durable failure_kind (UI
# compatibility); the classification itself is never free-form. commit_rejected
# carries no error string — the `commit` dict names the gate errors.
_FAILURE_KIND_ERROR = {
    "reasoning_loop": "reasoning_loop",
    "collection_limit": "collection_limit",
    "truncated_json": "truncated JSON at the output token limit",
    "invalid_json": "no JSON in reply",
    "invalid_schema": "provider response did not satisfy the required JSON schema",
    "provider_error": "provider error",
    "provider_rate_limited": "provider temporarily rate limited the request",
    "provider_upstream_5xx": "provider service was temporarily unavailable",
    "provider_auth": "provider authentication failed",
    "provider_request_rejected": "provider rejected the request",
    "provider_response_format_rejected": "provider rejected the response format",
    "provider_capability_rejected": "provider does not support a required capability",
    "provider_unknown": "provider request failed",
    "transport_timeout": "provider request timed out",
    "transport_connection": "provider connection failed",
    "transport_stream_interrupted": "provider response stream was interrupted",
}

# Serialization recovery instruction: rebuild the pristine stage task/state and
# demand ONE compact slot object, no tools, no markdown, no prose. The reply is
# bounded by the stage's fixed serialization cap (never doubled dynamically).
_SERIALIZATION_RETRY_MSG = (
    "Your previous reply was not a single complete JSON object (the prior reply "
    "was about {prior_chars} characters and was truncated or malformed), so "
    "nothing was committed. {bounds_sentence}Do NOT call any tools. Re-emit the "
    "complete slot as ONE compact JSON object now: no markdown fences, no prose, "
    "no explanations. Omit null fields and keep every item on a single line so "
    "the whole slot fits the output budget."
)

_SCHEMA_RETRY_MSG = (
    "Your previous reply was valid JSON but failed KiCraft's local slot validation, "
    "so nothing was committed. Validation error: {schema_error}. "
    "{bounds_sentence}Do NOT call any tools. Re-emit the complete corrected slot as "
    "ONE compact JSON object now: no markdown fences, no prose, no explanations. "
    "Preserve valid entries, correct the reported field, and omit null fields."
)

_COLLECTION_LIMIT_RETRY_MSG = (
    "Your previous reply was stopped at observed item {observed_count} of the "
    "top-level `{field}` collection because its configured {limit_scope} limit is "
    "{configured_total}; {emitted_content_chars} content characters were emitted "
    "and nothing was committed. {bounds_sentence}Do NOT call any tools. Start "
    "again from the project state and emit ONE compact slot JSON within those "
    "canonical limits. Do not continue or salvage the stopped draft."
)

_SEMANTIC_REPAIR_MSG = (
    "The candidate is schema-valid but deterministic semantic checks found the "
    "following high-confidence defects: {diagnostics}. Preserve all valid content, "
    "correct only these defects, add no new assumptions, and return one complete "
    "JSON object matching the same schema. No tools, markdown, or prose."
)


def _response_policy(client, stage: str, normal_max_tokens: int) -> StageResponsePolicy:
    """The stage's immutable response policy: normal cap + reasoning, plus the
    fixed serialization cap and retry budget (see Settings.design_stage_policy).
    Clients whose settings expose the policy method get it; mocks and legacy
    settings fall back to safe defaults: the floored normal cap, the existing
    design_reasoning payload, the fixed serialization cap, one serialization
    retry."""
    fn = getattr(getattr(client, "s", None), "design_stage_policy", None)
    if callable(fn):
        pol = fn(stage, int(normal_max_tokens))
        if isinstance(pol, StageResponsePolicy):
            return pol
    return StageResponsePolicy(
        normal_max_tokens=_stage_max_tokens(stage, normal_max_tokens),
        normal_reasoning=_design_reasoning(client, stage),
        serialization_max_tokens=STAGE_SERIALIZATION_MAX_TOKENS.get(stage, 8192),
        serialization_retries=1,
        collection_bounds=STAGE_COLLECTION_BOUNDS.get(stage, ()),
        reasoning_guard=None,
    )


@dataclass(frozen=True)
class PreparedStage:
    stage: str
    prompt_state: dict
    extras: dict
    base_messages: tuple[dict, ...]
    contract: StageResponseContract
    policy: StageResponsePolicy
    tools: list[dict] | None
    executor: object | None


@dataclass(frozen=True)
class AttemptOutcome:
    kind: Literal["candidate", "questions", "recoverable_failure", "terminal_failure"]
    payload: dict


@dataclass(frozen=True)
class ProviderFacts:
    raw: str
    finish: str | None
    rounds: int | None
    tool_calls: int | None
    cost_usd: float
    had_content: bool
    loop_detected: bool
    collection_limit: dict | None
    loop_abort_reason: str | None
    collection_counts: dict
    provider: str | None = None
    usage: dict | None = None
    wall_s: float | None = None


def call_stage_provider(
    client,
    prepared: PreparedStage,
    *,
    messages: list[dict],
    response_format: dict,
    max_tokens: int,
    temperature: float,
    reasoning: dict | None,
    reasoning_guard,
    progress,
    meta_ctx: dict,
) -> ProviderFacts:
    """Make exactly one normal or tool-enabled provider call."""
    call_t0 = time.monotonic()
    if prepared.tools:
        result = client.chat_with_tools(
            messages,
            prepared.tools,
            prepared.executor,
            max_tokens=max_tokens,
            temperature=temperature,
            max_rounds=_BOM_MAX_ROUNDS,
            progress=progress,
            meta_ctx=meta_ctx,
            reasoning=reasoning,
            reasoning_guard=reasoning_guard,
            collection_bounds=prepared.policy.collection_bounds,
            response_format=response_format,
        )
        content = result["text"]
        rounds = result.get("rounds")
        tool_calls = result.get("tool_calls")
    else:
        result = client.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            progress=progress,
            meta_ctx=meta_ctx,
            reasoning=reasoning,
            reasoning_guard=reasoning_guard,
            collection_bounds=prepared.policy.collection_bounds,
            response_format=response_format,
        )
        content = result.get("text") or ""
        rounds = None
        tool_calls = None
    return ProviderFacts(
        raw=content or result.get("reasoning") or "",
        finish=result.get("finish_reason"),
        rounds=rounds,
        tool_calls=tool_calls,
        cost_usd=result["cost_usd"],
        had_content=bool(content),
        loop_detected=bool(result.get("loop_detected")),
        collection_limit=result.get("collection_limit"),
        loop_abort_reason=result.get("loop_abort_reason"),
        collection_counts=result.get("collection_counts") or {},
        provider=result.get("provider"),
        usage=result.get("usage") or {},
        wall_s=round(time.monotonic() - call_t0, 3),
    )


def run_serialization_recovery(
    client,
    prepared: PreparedStage,
    *,
    messages: list[dict],
    response_format: dict,
    temperature: float,
    reasoning_guard,
    progress,
    meta_ctx: dict,
) -> ProviderFacts:
    """Make the one tool-free recovery call with the prepared full-slot contract."""
    call_t0 = time.monotonic()
    result = client.chat(
        messages,
        max_tokens=int(prepared.policy.serialization_max_tokens),
        temperature=temperature,
        progress=progress,
        meta_ctx=meta_ctx,
        reasoning={"enabled": False},
        reasoning_guard=reasoning_guard,
        collection_bounds=prepared.policy.collection_bounds,
        response_format=response_format,
    )
    content = result.get("text") or ""
    return ProviderFacts(
        raw=content or result.get("reasoning") or "",
        finish=result.get("finish_reason"),
        rounds=None,
        tool_calls=None,
        cost_usd=result["cost_usd"],
        had_content=bool(content),
        loop_detected=bool(result.get("loop_detected")),
        collection_limit=result.get("collection_limit"),
        loop_abort_reason=result.get("loop_abort_reason"),
        collection_counts=result.get("collection_counts") or {},
        provider=result.get("provider"),
        usage=result.get("usage") or {},
        wall_s=round(time.monotonic() - call_t0, 3),
    )


def _unknown_sheet_references(prepared: PreparedStage, candidate: dict) -> list[dict[str, str]]:
    if prepared.stage != "bom":
        return []
    architecture = prepared.prompt_state.get("architecture") or {}
    known = {
        str(sheet.get("name"))
        for sheet in architecture.get("sheets") or []
        if isinstance(sheet, dict) and sheet.get("name")
    }
    violations = []
    for part in candidate.get("parts") or []:
        if not isinstance(part, dict):
            continue
        sheet = str(part.get("sheet") or "")
        if sheet not in known:
            violations.append({"ref": str(part.get("ref") or ""), "sheet": sheet})
    return violations


def decode_stage_response(
    prepared: PreparedStage,
    facts: ProviderFacts,
) -> AttemptOutcome:
    """Parse and normalize one response without mutating durable state."""
    try:
        if facts.finish == "collection_limit":
            raise ValueError("stream collection limit")
        parsed = _extract_json(facts.raw)
        if isinstance(parsed.get("questions"), list):
            return AttemptOutcome(
                "questions",
                {
                    "candidate": {
                        "questions": _normalize_questions(parsed["questions"], prepared.stage)
                    },
                    "expanded_component_count": 0,
                },
            )
        candidate, expanded = _normalize_stage_response(
            prepared.stage, parsed, prepared.prompt_state
        )
        kind = "questions" if isinstance(candidate.get("questions"), list) else "candidate"
        return AttemptOutcome(
            kind,
            {
                "candidate": candidate,
                "expanded_component_count": expanded,
            },
        )
    except StageSchemaError as exc:
        return AttemptOutcome(
            "recoverable_failure",
            {"failure_kind": "invalid_schema", "schema_error": str(exc)},
        )
    except (json.JSONDecodeError, ValueError):
        kind = (
            "collection_limit"
            if facts.finish == "collection_limit"
            else (
                "reasoning_loop"
                if facts.loop_detected
                else _classify_parse_failure(facts.finish, facts.had_content)
            )
        )
        return AttemptOutcome(
            "recoverable_failure",
            {"failure_kind": kind, "schema_error": None},
        )


def commit_candidate(prepared: PreparedStage, candidate: dict, state_path, brief, workspace):
    """Commit one candidate exactly once."""
    slot = dict(candidate)
    project_stem = slot.pop("project_stem", None)
    ok, result = commit_stage(
        prepared.stage, dict(slot), state_path, brief, project_stem, workspace
    )
    return ok, result, slot


def next_attempt(
    rejection: dict,
    prior_signature: tuple | None,
    *,
    was_clean_slate: bool,
    clean_slate_spent: bool = False,
    clean_slate_armed_signature: tuple | None = None,
) -> tuple[tuple, bool, bool]:
    """Classify a commit rejection: preserving correction, the one clean-slate
    escape, or terminal rejection (KC-VKUT5H A3 bounded continuation).

    State machine (the caller owns ``clean_slate_spent`` /
    ``clean_slate_armed_signature``; the escape is armed at most once):

    * ordinary response, signature EQUAL to the prior one and the escape is
      not yet spent -> arm exactly one clean-slate call and record the arming
      signature;
    * ordinary response, signature equal and the escape already spent ->
      TERMINAL (an adjacent repeat after the escape is churn, not progress);
    * the clean-slate response itself (``was_clean_slate``): the escape is
      spent either way. A signature equal to the arming signature is no
      progress -> TERMINAL. A DIFFERENT signature is bounded churn, not
      proven improvement: it may consume remaining ordinary preserving
      iterations, but it can never arm a second clean slate;
    * any other case -> ordinary preserving correction feedback.

    Outer-loop and provider-call bounds are the caller's; this function only
    classifies.
    """
    signature = _commit_rejection_signature(rejection)
    if was_clean_slate:
        if clean_slate_armed_signature is not None and signature == clean_slate_armed_signature:
            return signature, False, True  # escape repeated the arming defect
        return signature, False, False
    if signature == prior_signature:
        if clean_slate_spent:
            return signature, False, True
        return signature, True, False
    return signature, False, False


def finalize_stage(
    client,
    *,
    run_id,
    stage: str,
    state_path,
    progress,
    ok: bool,
    t0: float,
    cpu0: float,
    cost_usd: float,
    attempts: int,
    rounds,
    tool_calls,
    emitted_collection_count: int,
    expanded_component_count: int,
    outcome: dict,
) -> dict:
    """Persist status and ledger once, then build the caller-visible result."""
    wall_s = round(time.monotonic() - t0, 3)
    cpu_s = round(_child_cpu_s() - cpu0, 3)
    stamp_stage_status(
        state_path,
        stage,
        ok,
        cost_usd=cost_usd,
        attempts=attempts,
        rounds=rounds,
        tool_calls=tool_calls,
        wall_s=wall_s,
        cpu_s=cpu_s,
        provider_ok=outcome.get("provider_ok"),
        schema_ok=outcome.get("schema_ok"),
        semantic_clean=outcome.get("semantic_clean"),
        repair_required=outcome.get("repair_required", False),
        fab_safe=outcome.get("fab_safe"),
        repair_attempted=outcome.get("repair_attempted", False),
        repair_adopted=outcome.get("repair_adopted", False),
        diagnostics=outcome.get("diagnostics") or [],
        error=outcome.get("error"),
        failure_kind=outcome.get("failure_kind"),
    )
    _record_stage_ledger(
        client,
        run_id=run_id,
        stage=stage,
        ok=ok,
        attempts=attempts,
        rounds=rounds,
        tool_calls=tool_calls,
        wall_s=wall_s,
        cpu_s=cpu_s,
        cost_usd=cost_usd,
        failure_kind=outcome.get("failure_kind"),
        emitted_collection_count=emitted_collection_count,
        expanded_component_count=expanded_component_count,
    )
    if progress:
        for diagnostic in outcome.get("diagnostics") or []:
            progress({"kind": "stage_diagnostic", "stage": stage, **diagnostic})
    if progress:
        progress(
            {
                "kind": "stage_done",
                "stage": stage,
                "ok": ok,
                "cost": cost_usd,
                "attempts": attempts,
                "warning": bool(outcome.get("diagnostics")),
                "semantic_clean": outcome.get("semantic_clean"),
                "fab_safe": outcome.get("fab_safe"),
            }
        )
    return {
        "stage": stage,
        "commit_ok": ok,
        "cost_usd": cost_usd,
        "attempts": attempts,
        "rounds": rounds,
        "tool_calls": tool_calls,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "emitted_collection_count": emitted_collection_count,
        "expanded_component_count": expanded_component_count,
        **outcome,
    }


def drive_stage(
    client,
    stage,
    brief,
    state_path,
    workspace,
    max_tokens=4096,
    max_retries=2,
    progress=None,
    answers=None,
    instruction=None,
    meta_ctx=None,
    core_defaults=None,
    *,
    review_before_commit: bool = False,
) -> dict:
    run_id = (meta_ctx or {}).get("run_id")
    t0 = time.monotonic()
    cpu0 = _child_cpu_s()
    if progress:
        progress({"kind": "stage_start", "stage": stage, "model": _client_model(client)})
    prep = prepare_stage(stage, state_path, workspace)
    if prep.returncode != 0:
        err = (prep.stderr.strip() or prep.stdout.strip())[:600]
        _wall = round(time.monotonic() - t0, 3)
        _cpu = round(_child_cpu_s() - cpu0, 3)
        if not review_before_commit:
            stamp_stage_status(state_path, stage, False, wall_s=_wall, cpu_s=_cpu)
            _record_stage_ledger(
                client,
                run_id=run_id,
                stage=stage,
                ok=False,
                attempts=None,
                rounds=None,
                tool_calls=None,
                wall_s=_wall,
                cpu_s=_cpu,
                cost_usd=0.0,
            )
            if progress:
                progress({"kind": "stage_done", "stage": stage, "ok": False})
        return {
            "stage": stage,
            "commit_ok": False,
            "cost_usd": 0.0,
            "wall_s": _wall,
            "cpu_s": _cpu,
            "error": f"stage-prep failed: {err}",
        }
    prep_json = json.loads(prep.stdout)
    extras = prep_json.get("extras") or {}

    # Core-components registry (admin-curated default parts): rendered fresh from
    # the rows the caller fetched on this run, never persisted into state.json,
    # so admin edits land on every resume/re-drive.
    if stage in ("architecture", "bom") and core_defaults:
        block = _format_core_defaults_block(core_defaults)
        if block:
            extras["core_defaults_block"] = block
    if stage in {"architecture", "bom"}:
        from kicraft.design.recipes import recipe_summaries

        extras["circuit_recipes"] = recipe_summaries()

    # Bookkeeping the model has no use for stays out of its prompt.
    prompt_state = dict(prep_json["state"])
    prompt_state.pop("stage_status", None)
    # Wiring sees only the canonical component digest. Recipe provenance stays
    # so normalization can recreate immutable assignments.
    if stage == "wiring" and isinstance(prompt_state.get("bom"), dict):
        full_bom = prompt_state["bom"]
        from kicraft.design.recipes import locked_pin_assignments

        locked = locked_pin_assignments(full_bom)
        if locked:
            extras["recipe_locked_pins"] = [
                {"ref": ref, "pin": pin, "net": net} for (ref, pin), net in sorted(locked.items())
            ]
        prompt_state["bom"] = {
            "parts": [
                {
                    "ref": p.get("ref"),
                    "sheet": p.get("sheet"),
                    "symbol": p.get("symbol"),
                    "value": p.get("value"),
                    "recipe_id": p.get("recipe_id"),
                    "recipe_instance": p.get("recipe_instance"),
                    "recipe_role": p.get("recipe_role"),
                }
                for p in full_bom.get("parts", [])
            ],
        }
    user = f"PROJECT BRIEF:\n{brief}\n\nCURRENT DESIGN STATE (JSON):\n{json.dumps(prompt_state)}"
    if extras:
        # bom carries the full parts table + core defaults (the adoption rule
        # depends on both being complete), wiring carries symbol_pinouts.
        budget = {"wiring": 40000, "bom": 20000}.get(stage, 24000)
        user += f"\n\nSTAGE EXTRAS (reference data from stage-prep):\n{json.dumps(extras)[:budget]}"
    if answers:
        qa = "\n".join(f"Q: {a.get('text', '')}\nA: {a.get('answer', '')}" for a in answers)
        user += f"\n\nThe user answered your earlier clarifying question(s):\n{qa}"
    if instruction:
        user += (
            f"\n\nThe user requests this change to the {stage}: {instruction}\n"
            "Re-draft the slot to honor it, keeping everything else consistent."
        )
    if stage == "bom":
        user += _bom_part_hints(
            brief, instruction or "", *(str(a.get("answer", "")) for a in (answers or []))
        )
    user += f"\n\nProduce the {stage} slot JSON now."

    try:
        contract = build_stage_response_contract(stage, prompt_state)
    except ValueError as exc:
        wall_s = round(time.monotonic() - t0, 3)
        cpu_s = round(_child_cpu_s() - cpu0, 3)
        error = f"stage contract failed: {exc}"
        if not review_before_commit:
            stamp_stage_status(
                state_path,
                stage,
                False,
                cost_usd=0.0,
                attempts=0,
                wall_s=wall_s,
                cpu_s=cpu_s,
                error=error,
            )
            _record_stage_ledger(
                client,
                run_id=run_id,
                stage=stage,
                ok=False,
                attempts=0,
                rounds=None,
                tool_calls=None,
                wall_s=wall_s,
                cpu_s=cpu_s,
                cost_usd=0.0,
            )
            if progress:
                progress({"kind": "stage_done", "stage": stage, "ok": False})
        return {
            "stage": stage,
            "commit_ok": False,
            "cost_usd": 0.0,
            "attempts": 0,
            "wall_s": wall_s,
            "cpu_s": cpu_s,
            "error": error,
        }
    policy = _response_policy(client, stage, max_tokens)

    messages = [
        {
            "role": "system",
            "content": build_system(contract, policy.collection_bounds),
        },
        {"role": "user", "content": user},
    ]
    tools = BOM_TOOLS if stage == "bom" else None
    executor = build_bom_executor(workspace, run_design_cli, KICRAFT) if stage == "bom" else None
    response_format = contract.response_format

    # Retries rebuild the conversation from this pristine base instead of
    # appending to it. chat_with_tools mutates the list it's handed (it appends
    # every tool-call turn + tool result), so a naive append-feedback-and-loop
    # re-sends the WHOLE accumulated transcript on every later attempt — BOM
    # snowballed to ~830K input tokens for ~28K output (30:1) this way. A retry
    # only needs the task, the model's last slot, and the correction: resolved
    # parts persist in the mpn cache + parts library and the executor memo
    # dedupes any re-issued lookup, so the dropped transcript is free to rebuild.
    base_messages = list(messages)
    prepared = PreparedStage(
        stage=stage,
        prompt_state=prompt_state,
        extras=extras,
        base_messages=tuple(base_messages),
        contract=contract,
        policy=policy,
        tools=tools,
        executor=executor,
    )

    def _debug_context(raw_response: str) -> dict:
        return {
            "prompt_state": prompt_state,
            "extras": extras,
            "base_messages": base_messages,
            "response_schema": contract.schema,
            "response_format": response_format,
            "raw_response": raw_response,
            "response_policy": {
                **asdict(policy),
                "design_temperature": _design_temperature(client),
                "serialization_escape_temperature": float(
                    getattr(
                        getattr(client, "s", None),
                        "serialization_escape_temperature",
                        0.4,
                    )
                ),
                "stage_semantics": getattr(
                    getattr(client, "s", None), "stage_semantics", "observe"
                ),
                "max_retries": max_retries,
            },
        }

    def _lean_retry(assistant_text: str | None, user_msg: str) -> list[dict]:
        msgs = list(base_messages)
        if assistant_text:
            msgs.append({"role": "assistant", "content": assistant_text})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    total_cost = 0.0
    last: dict = {}
    # Policy was resolved before prompt construction so the first attempt and
    # serialization retry consume the same immutable collection bounds.
    normal_cap = int(policy.normal_max_tokens)
    serialization_budget = max(0, int(policy.serialization_retries))
    temperature = _design_temperature(client)
    reasoning = policy.normal_reasoning
    reasoning_guard = policy.reasoning_guard
    escape_temperature = float(
        getattr(getattr(client, "s", None), "serialization_escape_temperature", 0.4)
    )
    # Recovery budgets are independent: reasoning recovery gets ONE
    # reasoning-disabled retry, serialization recovery gets exactly ONE plain
    # tool-free call at the fixed cap, and commit correction gets the normal
    # `max_retries + 1` attempts. `attempts` counts ACTUAL provider calls made
    # (never the configured maximum), and every call carries a finite cap.
    loop_retries = 0
    prior_rejection_signature = None
    clean_slate_next = False
    # KC-VKUT5H A3: the pristine escape may be armed exactly once. ``spent``
    # is set when the escape is armed (so the clean-slate response itself is
    # classified through the post-escape rules), and ``armed`` remembers the
    # signature that triggered it — a clean-slate response repeating that
    # signature is no progress and stays terminal, while a different one may
    # continue with ordinary preserving corrections (never a second escape).
    clean_slate_spent = False
    clean_slate_armed_signature: tuple | None = None
    serialization_calls = 0
    attempts = 0
    rounds = None
    tool_calls_ct = None
    expanded_component_count = 0
    emitted_collection_count = 0
    provider_call_budget = max_retries + 2
    provider_ok = False
    schema_ok = False
    semantic_repair_attempted = False
    semantic_repair_adopted = False
    semantic_mode = getattr(getattr(client, "s", None), "stage_semantics", "observe")
    current_facts = None
    current_call_mode = "normal"
    current_attempt_number = 0

    def emit_candidate_decoded(
        decoded: AttemptOutcome,
        *,
        provider_attempt: int,
        serialization_recovery: bool,
        clean_slate: bool,
    ) -> None:
        if not progress or decoded.kind not in {"candidate", "questions"}:
            return
        candidate = decoded.payload["candidate"]
        progress(
            {
                "kind": "candidate_decoded",
                "stage": stage,
                "attempt": provider_attempt,
                "serialization_recovery": serialization_recovery,
                "clean_slate": clean_slate,
                "expanded_component_count": int(decoded.payload["expanded_component_count"]),
                "unknown_sheet_references": _unknown_sheet_references(prepared, candidate),
            }
        )

    for attempt in range(max_retries + 1):
        if attempts >= provider_call_budget:
            break
        ctx = {**(meta_ctx or {}), "stage": stage, "attempt": attempt}
        tool_calls_ct = None
        raw = ""
        finish = None
        had_content = False
        loop_detected = False
        collection_limit = None
        loop_abort_reason = None
        was_clean_slate = clean_slate_next
        clean_slate_next = False
        call_messages = messages
        call_response_format = response_format
        attempts += 1  # a call IS attempted even when it raises below
        current_attempt_number = attempts
        try:
            facts = call_stage_provider(
                client,
                prepared,
                messages=call_messages,
                response_format=call_response_format,
                max_tokens=normal_cap,
                temperature=temperature,
                reasoning=reasoning,
                reasoning_guard=reasoning_guard,
                progress=progress,
                meta_ctx=ctx,
            )
            raw = facts.raw
            finish = facts.finish
            rounds = facts.rounds
            tool_calls_ct = facts.tool_calls
            had_content = facts.had_content
            loop_detected = facts.loop_detected
            collection_limit = facts.collection_limit
            loop_abort_reason = facts.loop_abort_reason
            emitted_collection_count = max(
                emitted_collection_count,
                max(facts.collection_counts.values(), default=0),
            )
            total_cost += facts.cost_usd
            provider_ok = True
            current_facts = facts
            current_call_mode = "clean_slate" if was_clean_slate else "normal"
        except (*_TRANSPORT_FAILURE_EXC, *_PROVIDER_FAILURE_EXC) as exc:
            failure = classify_provider_exception(exc)
            kind = failure["failure_kind"]
            last = {
                **failure,
                "error": _FAILURE_KIND_ERROR[kind],
                "reply_head": "",
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
                "provider_ok": provider_ok,
                "schema_ok": schema_ok,
            }
            _record_attempt_facts(
                client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome=kind,
                error_facts={k: v for k, v in failure.items() if k != "failure_kind"},
            )
            break

        # Reasoning recovery: the in-stream loop detector aborted, the client
        # reported finish_reason="reasoning_loop", OR an empty length completion
        # (reasoning/output exhaustion with no answer text — that is NOT a
        # truncated JSON answer and must never be labeled invalid_json). Retry
        # once with reasoning disabled + a higher temperature to escape the
        # deterministic cycle, then fail honestly as reasoning_loop.
        if loop_detected or finish == "reasoning_loop" or (finish == "length" and not had_content):
            last = {
                "error": "reasoning_loop",
                "failure_kind": "reasoning_loop",
                "reply_head": (raw or "")[:200],
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
                "loop_abort_reason": loop_abort_reason,
            }
            _record_attempt_facts(
                client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome="reasoning_loop",
                facts=facts,
            )
            if progress:
                progress(
                    {
                        "kind": "retry",
                        "stage": stage,
                        "errors": [
                            "reasoning loop detected"
                            + (f" ({loop_abort_reason})" if loop_abort_reason else "")
                            + " — retrying with reasoning disabled"
                        ],
                    }
                )
            if loop_retries >= _MAX_LOOP_RETRIES:
                break
            loop_retries += 1
            reasoning = {"enabled": False}
            temperature = max(temperature + 0.4, 0.4)
            messages = _lean_retry(None, _REASONING_LOOP_RETRY_MSG)
            continue

        outcome = decode_stage_response(prepared, facts)
        emit_candidate_decoded(
            outcome,
            provider_attempt=attempts,
            serialization_recovery=False,
            clean_slate=was_clean_slate,
        )
        schema_error = outcome.payload.get("failure_kind") == "invalid_schema"
        schema_error_detail = outcome.payload.get("schema_error")
        kind = outcome.payload.get("failure_kind")
        if outcome.kind in {"candidate", "questions"}:
            obj = outcome.payload["candidate"]
            expanded_component_count = outcome.payload["expanded_component_count"]
            schema_ok = True
        if outcome.kind == "questions":
            _record_attempt_facts(
                client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome="question",
                facts=facts,
            )
        if schema_error or kind in {"collection_limit", "truncated_json", "invalid_json"}:
            last = {
                "failure_kind": kind,
                "reply_head": (raw or "")[:200],
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
                "error": _FAILURE_KIND_ERROR.get(kind, kind),
                "schema_error": schema_error_detail,
            }
            _record_attempt_facts(
                client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome=kind,
                facts=facts,
            )
            if progress:
                progress(
                    {
                        "kind": "retry",
                        "stage": stage,
                        "errors": [last["error"]],
                        "failure_kind": kind,
                    }
                )
            if serialization_calls >= serialization_budget or attempts >= provider_call_budget:
                break  # serialization budget or provider-call budget exhausted
            # Serialization recovery: exactly ONE plain, tool-free,
            # reasoning-disabled completion at the fixed serialization cap,
            # rebuilt from the pristine base messages + the serialization
            # instruction (never the BOM tool transcript, never a doubled cap).
            serialization_calls += 1
            attempts += 1  # the serialization completion is a provider call too
            current_attempt_number = attempts
            sctx = {**ctx, "serialization": True}
            smessages = list(call_messages)
            bounds_sentence = _collection_bounds_sentence(policy.collection_bounds)
            if kind == "collection_limit":
                retry_template = _COLLECTION_LIMIT_RETRY_MSG
            elif kind == "invalid_schema":
                retry_template = _SCHEMA_RETRY_MSG
            else:
                retry_template = _SERIALIZATION_RETRY_MSG
            limit_info = collection_limit or {}
            retry_message = retry_template.format(
                prior_chars=len(raw or ""),
                bounds_sentence=(bounds_sentence + " ") if bounds_sentence else "",
                field=limit_info.get("field", "unknown"),
                observed_count=limit_info.get("observed_count", "unknown"),
                configured_total=limit_info.get("configured_total", "unknown"),
                limit_scope=limit_info.get("limit_scope", "total"),
                emitted_content_chars=limit_info.get("emitted_content_chars", len(raw or "")),
                schema_error=(schema_error_detail or "unspecified schema violation")[:1200],
            )
            smessages.append({"role": "user", "content": retry_message})
            resolution_ledger = getattr(executor, "resolution_ledger", {}) if executor else {}
            if resolution_ledger:
                smessages.append(
                    {
                        "role": "user",
                        "content": "BOUNDED RESOLUTION LEDGER (reuse these exact accepted values):\n"
                        + json.dumps(list(resolution_ledger.values())[:16], separators=(",", ":")),
                    }
                )
            if progress:
                progress(
                    {
                        "kind": "serialization_recovery",
                        "stage": stage,
                        "failure_kind": kind,
                        "resolution_ledger_entries": int(len(resolution_ledger)),
                    }
                )
            try:
                sfacts = run_serialization_recovery(
                    client,
                    prepared,
                    messages=smessages,
                    response_format=call_response_format,
                    temperature=max(escape_temperature, 0.0),
                    reasoning_guard=reasoning_guard,
                    progress=progress,
                    meta_ctx=sctx,
                )
                total_cost += sfacts.cost_usd
            except (*_TRANSPORT_FAILURE_EXC, *_PROVIDER_FAILURE_EXC) as exc:
                failure = classify_provider_exception(exc)
                skind = failure["failure_kind"]
                last = {
                    **failure,
                    "error": _FAILURE_KIND_ERROR[skind],
                    "reply_head": "",
                    "rounds": rounds,
                    "tool_calls": tool_calls_ct,
                    "provider_ok": provider_ok,
                    "schema_ok": schema_ok,
                }
                _record_attempt_facts(
                    client,
                    run_id=run_id,
                    stage=stage,
                    attempt=attempts,
                    call_mode="serialization",
                    outcome=skind,
                    error_facts={k: v for k, v in failure.items() if k != "failure_kind"},
                )
                break
            sraw = sfacts.raw
            scollection_limit = sfacts.collection_limit
            serialization_outcome = decode_stage_response(prepared, sfacts)
            emit_candidate_decoded(
                serialization_outcome,
                provider_attempt=attempts,
                serialization_recovery=True,
                clean_slate=was_clean_slate,
            )
            schema_error = serialization_outcome.payload.get("failure_kind") == "invalid_schema"
            schema_error_detail = serialization_outcome.payload.get("schema_error")
            skind = serialization_outcome.payload.get("failure_kind")
            if serialization_outcome.kind in {"candidate", "questions"}:
                obj = serialization_outcome.payload["candidate"]
                expanded_component_count = serialization_outcome.payload["expanded_component_count"]
                schema_ok = True
            if serialization_outcome.kind != "candidate":
                _record_attempt_facts(
                    client,
                    run_id=run_id,
                    stage=stage,
                    attempt=attempts,
                    call_mode="serialization",
                    outcome=(
                        "question"
                        if serialization_outcome.kind == "questions"
                        else skind or "invalid_schema"
                    ),
                    facts=sfacts,
                )
            if schema_error or skind in {
                "collection_limit",
                "reasoning_loop",
                "truncated_json",
                "invalid_json",
            }:
                last = {
                    "failure_kind": skind,
                    "error": _FAILURE_KIND_ERROR.get(skind, skind),
                    "reply_head": (sraw or "")[:200],
                    "rounds": rounds,
                    "tool_calls": tool_calls_ct,
                    "collection_limit": scollection_limit,
                    "schema_error": schema_error_detail,
                }
                break
            # Parseable serialization output: the commit path owns it from here.
            # A commit rejection may still use remaining commit-correction
            # attempts at the normal stage/tool policy.
            raw = sraw
            current_facts = sfacts
            current_call_mode = "serialization"

        # A clarifying-question payload parks the stage (no slot this turn). No slot
        # model has a top-level "questions" key, so the shape is unambiguous. Never
        # re-park right after an answer (caps the back-and-forth at one round/stage).
        qpayload = obj.get("questions") if isinstance(obj, dict) else None
        if isinstance(qpayload, list) and qpayload:
            qs = _normalize_questions(qpayload, stage)
            # A reconcile_target park is the pipeline's ESCALATION (a BOM shortfall
            # wiring can't fix), not a user question. Surface it even after answers
            # were applied, so the shared bom-reconcile re-drive can add the parts
            # -- otherwise the "do not ask more questions" retry below burns the
            # stage's whole budget on a park it can never satisfy (WS6).
            is_reconcile_park = any(q.get("reconcile_target") for q in qs)
            if any(q["blocking"] for q in qs) and (not answers or is_reconcile_park):
                if not review_before_commit:
                    attach_questions(state_path, stage, qs)
                if progress:
                    progress({"kind": "question", "stage": stage, "questions": qs})
                result = {
                    "stage": stage,
                    "commit_ok": False,
                    "needs_input": True,
                    "questions": qs,
                    "cost_usd": total_cost,
                    "attempts": attempts,
                }
                if review_before_commit:
                    result.update(
                        {
                            "rounds": rounds,
                            "tool_calls": tool_calls_ct,
                            "wall_s": round(time.monotonic() - t0, 3),
                            "cpu_s": round(_child_cpu_s() - cpu0, 3),
                            "provider_ok": provider_ok,
                            "schema_ok": schema_ok,
                            "debug_context": _debug_context(raw),
                        }
                    )
                return result
            messages = _lean_retry(
                None,
                "Do not ask more questions. Apply sensible defaults (record each "
                "in assumptions, ending '(defaulted)') and output ONLY the slot "
                "JSON now.",
            )
            continue

        original_obj = obj
        diagnostics = diagnose_stage(stage, brief=brief, upstream_state=prompt_state, candidate=obj)
        severe = [d for d in diagnostics if d.severity in {"repair_required", "fab_gate"}]
        if severe and semantic_mode in {"repair", "enforce"} and not semantic_repair_attempted:
            semantic_repair_attempted = True
            attempts += 1
            repair_message = _SEMANTIC_REPAIR_MSG.format(
                diagnostics=json.dumps(
                    [{"code": d.code, "evidence": d.evidence} for d in severe],
                    separators=(",", ":"),
                )
            )
            repair_messages = _lean_retry(raw, repair_message)
            try:
                repair_facts = run_serialization_recovery(
                    client,
                    prepared,
                    messages=repair_messages,
                    response_format=response_format,
                    temperature=max(escape_temperature, 0.0),
                    reasoning_guard=reasoning_guard,
                    progress=progress,
                    meta_ctx={
                        **(meta_ctx or {}),
                        "stage": stage,
                        "attempt": attempts,
                        "semantic_repair": True,
                    },
                )
                total_cost += repair_facts.cost_usd
                repair_outcome = decode_stage_response(prepared, repair_facts)
                if repair_outcome.kind == "candidate":
                    repaired = repair_outcome.payload["candidate"]
                    repaired_diagnostics = diagnose_stage(
                        stage, brief=brief, upstream_state=prompt_state, candidate=repaired
                    )
                    repaired_severe = [
                        d
                        for d in repaired_diagnostics
                        if d.severity in {"repair_required", "fab_gate"}
                    ]
                    if len(repaired_severe) < len(severe):
                        obj = repaired
                        diagnostics = repaired_diagnostics
                        severe = repaired_severe
                        semantic_repair_adopted = True
                    _record_attempt_facts(
                        client,
                        run_id=run_id,
                        stage=stage,
                        attempt=attempts,
                        call_mode="semantic_repair",
                        outcome="candidate",
                        facts=repair_facts,
                        diagnostic_codes=[d.code for d in repaired_diagnostics],
                    )
                else:
                    repair_kind = repair_outcome.payload.get("failure_kind") or repair_outcome.kind
                    _record_attempt_facts(
                        client,
                        run_id=run_id,
                        stage=stage,
                        attempt=attempts,
                        call_mode="semantic_repair",
                        outcome=repair_kind,
                        facts=repair_facts,
                    )
            except (*_TRANSPORT_FAILURE_EXC, *_PROVIDER_FAILURE_EXC) as exc:
                failure = classify_provider_exception(exc)
                _record_attempt_facts(
                    client,
                    run_id=run_id,
                    stage=stage,
                    attempt=attempts,
                    call_mode="semantic_repair",
                    outcome=failure["failure_kind"],
                    error_facts={k: v for k, v in failure.items() if k != "failure_kind"},
                )

        diagnostic_rows = [d.model_dump(exclude_none=True) for d in diagnostics]
        if review_before_commit:
            _record_attempt_facts(
                client,
                run_id=run_id,
                stage=stage,
                attempt=current_attempt_number,
                call_mode=current_call_mode,
                outcome="candidate_review",
                facts=current_facts,
                diagnostic_codes=[d.code for d in diagnostics],
            )
            fab_safe = not any(d.severity == "fab_gate" for d in diagnostics)
            wall_s = round(time.monotonic() - t0, 3)
            cpu_s = round(_child_cpu_s() - cpu0, 3)
            if progress:
                for diagnostic in diagnostic_rows:
                    progress({"kind": "stage_diagnostic", "stage": stage, **diagnostic})
                progress(
                    {
                        "kind": "candidate_review",
                        "stage": stage,
                        "attempt": current_attempt_number,
                    }
                )
            return {
                "stage": stage,
                "needs_review": True,
                "commit_ok": False,
                "slot": obj,
                "diagnostics": diagnostic_rows,
                "cost_usd": total_cost,
                "attempts": attempts,
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "provider_ok": provider_ok,
                "schema_ok": schema_ok,
                "semantic_clean": not diagnostics,
                "repair_required": bool(severe),
                "fab_safe": fab_safe,
                "debug_context": _debug_context(
                    repair_facts.raw if semantic_repair_adopted else raw
                ),
            }
        ok, out, obj = commit_candidate(prepared, obj, state_path, brief, workspace)
        if not ok and semantic_repair_adopted:
            semantic_repair_adopted = False
            obj = original_obj
            diagnostics = diagnose_stage(
                stage, brief=brief, upstream_state=prompt_state, candidate=obj
            )
            severe = [d for d in diagnostics if d.severity in {"repair_required", "fab_gate"}]
            diagnostic_rows = [d.model_dump(exclude_none=True) for d in diagnostics]
            ok, out, obj = commit_candidate(prepared, obj, state_path, brief, workspace)
        _record_attempt_facts(
            client,
            run_id=run_id,
            stage=stage,
            attempt=current_attempt_number,
            call_mode=current_call_mode,
            outcome="candidate" if ok else "commit_rejected",
            facts=current_facts,
            diagnostic_codes=[d.code for d in diagnostics],
        )
        if ok:
            fab_safe = not any(d.severity == "fab_gate" for d in diagnostics)
            return finalize_stage(
                client,
                run_id=run_id,
                stage=stage,
                state_path=state_path,
                progress=progress,
                ok=True,
                t0=t0,
                cpu0=cpu0,
                cost_usd=total_cost,
                attempts=attempts,
                rounds=rounds,
                tool_calls=tool_calls_ct,
                emitted_collection_count=emitted_collection_count,
                expanded_component_count=expanded_component_count,
                outcome={
                    "commit": out,
                    "slot": obj,
                    "provider_ok": provider_ok,
                    "schema_ok": schema_ok,
                    "semantic_clean": not diagnostics,
                    "repair_required": bool(severe),
                    "fab_safe": fab_safe,
                    "repair_attempted": semantic_repair_attempted,
                    "repair_adopted": semantic_repair_adopted,
                    "diagnostics": diagnostic_rows,
                },
            )
        last = {
            "commit": out,
            "provider_ok": provider_ok,
            "schema_ok": schema_ok,
            "repair_attempted": semantic_repair_attempted,
            "repair_adopted": semantic_repair_adopted,
            "diagnostics": diagnostic_rows,
        }
        if progress:
            progress(
                {
                    "kind": "retry",
                    "stage": stage,
                    "errors": out.get("errors"),
                    "offenders": out.get("offenders"),
                }
            )
        signature, clean_slate_next, terminal = next_attempt(
            out,
            prior_rejection_signature,
            was_clean_slate=was_clean_slate,
            clean_slate_spent=clean_slate_spent,
            clean_slate_armed_signature=clean_slate_armed_signature,
        )
        if terminal:
            break
        prior_rejection_signature = signature
        if clean_slate_next:
            clean_slate_spent = True
            clean_slate_armed_signature = signature
            reasoning = {"enabled": False}
            temperature = max(escape_temperature, 0.0)
            messages = _lean_retry(
                None,
                _retry_feedback(out, stage=stage, valid_refs=None),
            )
            continue
        # Bounded continuation: a post-escape response with a NEW signature
        # (or a first-seen signature) gets the ordinary preserving correction
        # feedback; it cannot re-arm the escape (clean_slate_spent stays True).
        _valid_refs = committed_bom_refs(state_path) if stage == "wiring" else None
        messages = _lean_retry(raw, _retry_feedback(out, stage=stage, valid_refs=_valid_refs))

    # Terminal failure: a stage whose JSON parsed but every commit gate
    # rejected it classifies as commit_rejected (never mislabeled a parse
    # failure); every other terminal path already carries its failure_kind.
    if "failure_kind" not in last and last.get("commit") is not None:
        last["failure_kind"] = "commit_rejected"
    last.setdefault("provider_ok", provider_ok)
    last.setdefault("schema_ok", schema_ok)
    if review_before_commit:
        return {
            "stage": stage,
            "commit_ok": False,
            "cost_usd": total_cost,
            "attempts": attempts,
            "rounds": rounds,
            "tool_calls": tool_calls_ct,
            "wall_s": round(time.monotonic() - t0, 3),
            "cpu_s": round(_child_cpu_s() - cpu0, 3),
            **last,
            "debug_context": _debug_context(raw),
        }
    return finalize_stage(
        client,
        run_id=run_id,
        stage=stage,
        state_path=state_path,
        progress=progress,
        ok=False,
        t0=t0,
        cpu0=cpu0,
        cost_usd=total_cost,
        attempts=attempts,
        rounds=rounds,
        tool_calls=tool_calls_ct,
        emitted_collection_count=emitted_collection_count,
        expanded_component_count=expanded_component_count,
        outcome=last,
    )
