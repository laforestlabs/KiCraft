"""Single-stage preparation, provider calls, retries, commits, and finalization."""

from __future__ import annotations

import json
import re
import resource
import time
from dataclasses import dataclass
from typing import Literal

import requests

from kicraft.design import models

from .config import STAGE_COLLECTION_BOUNDS, STAGE_SERIALIZATION_MAX_TOKENS, StageResponsePolicy
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
    "transport_error": "transport error",
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

_COLLECTION_LIMIT_RETRY_MSG = (
    "Your previous reply was stopped at observed item {observed_count} of the "
    "top-level `{field}` collection because its configured {limit_scope} limit is "
    "{configured_total}; {emitted_content_chars} content characters were emitted "
    "and nothing was committed. {bounds_sentence}Do NOT call any tools. Start "
    "again from the project state and emit ONE compact slot JSON within those "
    "canonical limits. Do not continue or salvage the stopped draft."
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
) -> tuple[tuple, bool, bool]:
    """Classify preserving correction, one clean-slate escape, or terminal rejection."""
    signature = _commit_rejection_signature(rejection)
    if was_clean_slate:
        return signature, False, True
    return signature, signature == prior_signature, False


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
        progress(
            {
                "kind": "stage_done",
                "stage": stage,
                "ok": ok,
                "cost": cost_usd,
                "attempts": attempts,
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

    # Bookkeeping the model has no use for stays out of its prompt.
    prompt_state = dict(prep_json["state"])
    prompt_state.pop("stage_status", None)
    # Wiring sees only the canonical component digest. Existing connection rows
    # use a different durable representation and would contradict the final-pin
    # response contract; a wiring drive deliberately replaces them wholesale.
    if stage == "wiring" and isinstance(prompt_state.get("bom"), dict):
        full_bom = prompt_state["bom"]
        prompt_state["bom"] = {
            "parts": [
                {
                    "ref": p.get("ref"),
                    "sheet": p.get("sheet"),
                    "symbol": p.get("symbol"),
                    "value": p.get("value"),
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
    serialization_calls = 0
    attempts = 0
    rounds = None
    tool_calls_ct = None
    expanded_component_count = 0
    emitted_collection_count = 0
    provider_call_budget = max_retries + 2

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
        except _TRANSPORT_FAILURE_EXC:
            # Transport retries exhausted: terminal, never sent through JSON
            # recovery (BudgetExceeded is NOT caught — it propagates to the
            # guard/caller path).
            last = {
                "failure_kind": "transport_error",
                "error": "transport error",
                "reply_head": (raw or "")[:200],
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
            }
            break
        except _PROVIDER_FAILURE_EXC:
            last = {
                "failure_kind": "provider_error",
                "error": "provider error",
                "reply_head": (raw or "")[:200],
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
            }
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
        if schema_error or kind in {"collection_limit", "truncated_json", "invalid_json"}:
            last = {
                "failure_kind": kind,
                "reply_head": (raw or "")[:200],
                "rounds": rounds,
                "tool_calls": tool_calls_ct,
                "error": _FAILURE_KIND_ERROR.get(kind, kind),
                "schema_error": schema_error_detail,
            }
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
            sctx = {**ctx, "serialization": True}
            smessages = list(call_messages)
            bounds_sentence = _collection_bounds_sentence(policy.collection_bounds)
            retry_template = (
                _COLLECTION_LIMIT_RETRY_MSG
                if kind == "collection_limit"
                else _SERIALIZATION_RETRY_MSG
            )
            limit_info = collection_limit or {}
            retry_message = retry_template.format(
                prior_chars=len(raw or ""),
                bounds_sentence=(bounds_sentence + " ") if bounds_sentence else "",
                field=limit_info.get("field", "unknown"),
                observed_count=limit_info.get("observed_count", "unknown"),
                configured_total=limit_info.get("configured_total", "unknown"),
                limit_scope=limit_info.get("limit_scope", "total"),
                emitted_content_chars=limit_info.get("emitted_content_chars", len(raw or "")),
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
            except _TRANSPORT_FAILURE_EXC:
                last = {
                    "failure_kind": "transport_error",
                    "error": "transport error",
                    "reply_head": (raw or "")[:200],
                    "rounds": rounds,
                    "tool_calls": tool_calls_ct,
                }
                break
            except _PROVIDER_FAILURE_EXC:
                last = {
                    "failure_kind": "provider_error",
                    "error": "provider error",
                    "reply_head": (raw or "")[:200],
                    "rounds": rounds,
                    "tool_calls": tool_calls_ct,
                }
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
                attach_questions(state_path, stage, qs)
                if progress:
                    progress({"kind": "question", "stage": stage, "questions": qs})
                return {
                    "stage": stage,
                    "commit_ok": False,
                    "needs_input": True,
                    "questions": qs,
                    "cost_usd": total_cost,
                    "attempts": attempts,
                }
            messages = _lean_retry(
                None,
                "Do not ask more questions. Apply sensible defaults (record each "
                "in assumptions, ending '(defaulted)') and output ONLY the slot "
                "JSON now.",
            )
            continue

        ok, out, obj = commit_candidate(prepared, obj, state_path, brief, workspace)
        if ok:
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
                outcome={"commit": out, "slot": obj},
            )
        last = {"commit": out}
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
        )
        if terminal:
            break
        prior_rejection_signature = signature
        if clean_slate_next:
            reasoning = {"enabled": False}
            temperature = max(escape_temperature, 0.0)
            messages = _lean_retry(
                None,
                _retry_feedback(out, stage=stage, valid_refs=None),
            )
            continue
        prior_rejection_signature = signature
        # Echo the complete rejected response with structured commit feedback.
        # Correction uses the same stage schema; no alternate patch contract.
        _valid_refs = committed_bom_refs(state_path) if stage == "wiring" else None
        messages = _lean_retry(raw, _retry_feedback(out, stage=stage, valid_refs=_valid_refs))

    # Terminal failure: a stage whose JSON parsed but every commit gate
    # rejected it classifies as commit_rejected (never mislabeled a parse
    # failure); every other terminal path already carries its failure_kind.
    if "failure_kind" not in last and last.get("commit") is not None:
        last["failure_kind"] = "commit_rejected"
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
