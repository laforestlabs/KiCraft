"""Single-stage preparation, provider calls, retries, commits, and finalization."""

from __future__ import annotations

import hashlib
import json
import re
import resource
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Literal

import requests

from kicraft.design import models
from kicraft.design.stage_semantics import (
    complete_intent_classification,
    complete_unsourced_external_rails,
    diagnose_stage,
    normalize_project_stem,
    remove_mislabeled_architecture_defaults,
    remove_mislabeled_functional_defaults,
)
from .config import (
    STAGE_COLLECTION_BOUNDS,
    STAGE_SERIALIZATION_MAX_TOKENS,
    CollectionBound,
    StageResponsePolicy,
)
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
    stage_spec_sha256,
)
from .stage_work_units import (
    StageDraftStore,
    StageWorkUnit,
    WorkUnitValidationError,
    deterministic_bom_candidate,
    deterministic_wiring_candidate,
    merge_bom_units,
    merge_wiring_units,
    plan_stage_work_units,
    route_work_unit_ids,
    stage_draft_fingerprint,
    validate_unit_candidate,
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
NONINTERACTIVE_DEFAULTS_INSTRUCTION = (
    "This is a non-interactive evaluation. Apply sensible electrical defaults, "
    "record each in assumptions ending '(defaulted)', and return a complete draft "
    "without asking questions."
)


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
    unit_id=None,
    unit_attempt=None,
    aggregate_round=None,
    commit_result=None,
    candidate_retained: bool | None = None,
    fallback_reason: str | None = None,
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
        provider_profile=getattr(getattr(client, "s", None), "design_profile", None),
        fallback_reason=fallback_reason,
        reasoning_failure_kind=_reasoning_failure_kind(facts),
        diagnostic_codes=diagnostic_codes,
        unit_id=unit_id,
        unit_attempt=unit_attempt,
        aggregate_round=aggregate_round,
        **_redacted_rejection_facts(
            commit_result,
            candidate_retained=candidate_retained,
        ),
        **(error_facts or {}),
    )


def _work_unit_attempt_event(
    client,
    *,
    stage: str,
    unit,
    provider_attempt: int,
    unit_attempt: int,
    call_mode: str,
    outcome: str,
    facts=None,
    error_facts=None,
    aggregate_round: int | None = None,
    commit_result: dict | None = None,
    candidate_retained: bool | None = None,
    fallback_reason: str | None = None,
) -> dict:
    """Build the redacted user-visible counterpart of one ledger attempt row."""
    usage = (facts.usage or {}) if facts is not None else {}
    rejection = _redacted_rejection_facts(
        commit_result,
        candidate_retained=candidate_retained,
    )
    errors = error_facts or {}
    return {
        "kind": "work_unit_attempt",
        "version": 1,
        "stage": stage,
        "unit_id": unit.unit_id,
        "unit_sheet": unit.sheet,
        "source": "llm",
        "provider_attempt": provider_attempt,
        "unit_attempt": unit_attempt,
        "call_mode": call_mode,
        "outcome": outcome,
        "model": _client_model(client),
        "provider": facts.provider if facts is not None else None,
        "finish_reason": facts.finish if facts is not None else None,
        "wall_s": facts.wall_s if facts is not None else None,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "cost_usd": facts.cost_usd if facts is not None else 0.0,
        "http_status": errors.get("http_status"),
        "error_code": errors.get("error_code"),
        "request_id": errors.get("request_id"),
        "aggregate_round": aggregate_round,
        "fallback_reason": fallback_reason,
        **rejection,
    }


# Per-stage self-correction budget. Wiring must satisfy whole-board net coverage
# (§9.11) in a single slot; on a complex board the model needs more correction
# passes than the simpler, smaller-slot stages, so they floor higher (BOM must
# also resolve every symbol/footprint to a real library entry within its budget).
_STAGE_MIN_RETRIES = {"wiring": 7, "bom": 4}

# In-stream reasoning-loop breakout budget: when the client aborts a completion
# (finish_reason="reasoning_loop"), retry once with reasoning disabled + higher
# temperature to escape the deterministic cycle. A second loop in a row means the
# model cannot serialize even without reasoning -- fail with an explicit
# "reasoning_loop" label rather than "no JSON in reply".
_MAX_LOOP_RETRIES = 1


def _stage_max_retries(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_RETRIES.get(stage, 0))


# One optional batched lookup round followed by one schema-bound final response.
# The stage-prep parts block and verified generic defaults should let most work
# units finish on the first response; a model may spend the second only when a
# genuinely missing part needs resolution.
_BOM_MAX_ROUNDS = 2


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
                "do not rename that terminal or put both part pins on one net. Keep the "
                "two terminals on different nets and make both sides complete: each side "
                "has the resistor terminal plus a non-resistor endpoint. Use architecture "
                "direction and resolved pin function to keep endpoints on the correct side. "
                "Do not assume the populated side is the source or blindly move its endpoint."
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


def _redacted_rejection_facts(
    commit_result: dict | None,
    *,
    candidate_retained: bool | None,
) -> dict:
    """Ledger-safe deterministic rejection attribution."""
    if not isinstance(commit_result, dict):
        return {"candidate_retained": candidate_retained}
    gate_ids, offenders = _commit_rejection_signature(commit_result)
    stable_gates: list[str] = []
    for gate in gate_ids:
        match = re.fullmatch(r"9\.\d+", str(gate))
        if match:
            stable_gates.append(match.group(0))
            continue
        digest = hashlib.sha256(str(gate).encode("utf-8")).hexdigest()[:12]
        stable_gates.append(f"unclassified_{digest}")
    if commit_result.get("failure_kind") == "commit_process_failed":
        stable_gates = ["commit_process_failed"]
    signature_payload = json.dumps(
        [stable_gates, list(offenders)],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return {
        "commit_gate_codes": stable_gates,
        "offender_count": len(commit_result.get("offenders") or []),
        "rejection_signature": hashlib.sha256(signature_payload.encode("utf-8")).hexdigest(),
        "candidate_retained": candidate_retained,
    }


def _reasoning_failure_kind(facts) -> str | None:
    if facts is None:
        return None
    if facts.loop_abort_reason == "wall_stall":
        return "provider_wall_stall"
    if facts.loop_abort_reason == "repetition":
        return "repeated_reasoning"
    if facts.loop_abort_reason == "hard_ceiling":
        return "reasoning_token_exhaustion"
    if facts.finish == "length" and not facts.had_content:
        return "output_token_exhaustion"
    return None


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


def _trace_candidate(stage: str, candidate: dict | None) -> dict | None:
    """Return only the normalized design slot; never provider request/response data."""
    if candidate is None:
        return None
    if stage == "wiring":
        return {
            key: candidate[key]
            for key in ("connections", "no_connect_pins", "questions")
            if key in candidate
        }
    return candidate


def _observe_attempt(
    observer: Callable[[dict], None] | None,
    *,
    client,
    stage: str,
    provider_attempt: int,
    call_mode: str,
    outcome: str,
    facts: ProviderFacts | None = None,
    candidate: dict | None = None,
    commit_result: dict | None = None,
    rejection_signature: tuple | None = None,
    clean_slate_armed: bool = False,
    clean_slate_used: bool = False,
    escalated: bool = False,
    provider_fallback: bool = False,
    unit_id: str | None = None,
    unit_attempt: int | None = None,
    unit_index: int | None = None,
    unit_total: int | None = None,
    aggregate_round: int | None = None,
) -> None:
    """Emit one sanitized, normalized attempt record to an opt-in observer."""
    if observer is None:
        return
    settings = getattr(client, "s", None)
    signature = rejection_signature
    observer(
        {
            "version": 2,
            "stage": stage,
            "provider_attempt": provider_attempt,
            "call_mode": call_mode,
            "outcome": outcome,
            "model": _client_model(client),
            "design_profile": getattr(settings, "design_profile", None),
            "provider": facts.provider if facts is not None else None,
            "provider_order": list(getattr(settings, "provider_order", ()) or ()),
            "candidate": _trace_candidate(stage, candidate),
            "commit_result": commit_result,
            "rejection_signature": (
                [list(signature[0]), list(signature[1])] if signature is not None else None
            ),
            "clean_slate_armed": clean_slate_armed,
            "clean_slate_used": clean_slate_used,
            "escalated": escalated,
            "provider_fallback": provider_fallback,
            "unit_id": unit_id,
            "unit_attempt": unit_attempt,
            "unit_index": unit_index,
            "unit_total": unit_total,
            "aggregate_round": aggregate_round,
        }
    )


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
_MAX_SEMANTIC_REPAIR_ROUNDS = 1


def _semantic_repair_message(stage: str, diagnostics: list[models.StageDiagnostic]) -> str:
    message = _SEMANTIC_REPAIR_MSG.format(
        diagnostics=json.dumps(
            [{"code": d.code, "evidence": d.evidence} for d in diagnostics],
            separators=(",", ":"),
        )
    )
    if stage == "intent":
        message += (
            " For the intent stage, re-read the original brief: constraints must "
            "classify every explicit voltage, interface, inclusion or exclusion, "
            "and mechanical requirement; named_parts must contain every exact "
            "part or family named by the user. Do not leave either list empty "
            "when the brief contains those facts."
        )
    if any(d.code == "functional_spec_external_load_power_assumed" for d in diagnostics):
        message += (
            " Do not guess whether the board powers the external display or LED "
            "load. Return a questions object with one blocking question asking "
            "whether those loads use board-supplied or separate power."
        )
    if any(d.code == "architecture_external_load_current_unspecified" for d in diagnostics):
        message += (
            " Do not guess the external-load current. Return a questions object "
            "with one blocking question asking for the maximum total 5V output "
            "current for the HUB75 panel and LED string."
        )
    if any(
        d.code
        in {
            "architecture_external_load_source_capacity_unspecified",
            "architecture_external_load_source_has_no_headroom",
            "architecture_usb_pd_current_exceeds_standard",
            "architecture_5v_converter_capacity_unspecified",
            "architecture_5v_converter_has_no_headroom",
        }
        for d in diagnostics
    ):
        message += (
            " Size the input contract above the 5V external-load power budget, "
            "including board and conversion overhead. USB-PD is limited to 5A: "
            "for a 5V/5A load, negotiate a higher voltage at no more than 5A and "
            "use a dedicated buck converter rated above 5A to generate the "
            "regulated 5V rail. If the user's 5V wording could refer either to "
            "the PD contract or the load rail, return one blocking question "
            "asking which meaning is required."
        )

    if any(d.code == "architecture_duplicate_voltage_rails_unrelated" for d in diagnostics):
        message += (
            " Use one canonical net for a direct same-voltage connection. Keep "
            "two same-voltage rail names only when a named fuse, switch, filter, "
            "net tie, or converter explicitly connects them."
        )

    if any(
        d.code
        in {
            "architecture_rail_source_unspecified",
            "architecture_mcu_regulator_incomplete",
        }
        for d in diagnostics
    ):
        message += (
            " Add a REGULATOR_3V3 topology naming a 5V-to-3.3V regulator rated "
            "for at least 1A. Add a separate 3V3 REGULATOR sheet for that IC and "
            "connect +5V into it and +3V3 from it in inter_sheet_nets. Keep the "
            "+3V3 rail and do not cite a core default unless one was supplied."
        )
    if any(d.code == "architecture_unsupported_esp32s3_dac" for d in diagnostics):
        message += (
            " ESP32-S3 has no internal DAC. Replace direct analog audio with an "
            "I2S digital amplifier topology or an explicitly filtered PWM path."
        )
    return message


def _semantic_defect_score(diagnostics: list[models.StageDiagnostic]) -> int:
    """Count concrete offenders so a repair that shrinks one diagnostic can progress."""
    return sum(max(1, len(diagnostic.evidence)) for diagnostic in diagnostics)


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
    response_format: dict | None,
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
    response_format: dict | None,
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
    **_legacy_state,
) -> tuple[tuple, bool, bool]:
    """Stop immediately when deterministic gates and offenders do not change."""
    signature = _commit_rejection_signature(rejection)
    return signature, False, signature == prior_signature


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
        work_units=outcome.get("work_units"),
        reused_work_units=outcome.get("reused_work_units"),
        aggregate_repair_rounds=outcome.get("aggregate_repair_rounds"),
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
        work_units=outcome.get("work_units"),
        reused_work_units=outcome.get("reused_work_units"),
        aggregate_repair_rounds=outcome.get("aggregate_repair_rounds"),
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
                "failure_kind": outcome.get("failure_kind"),
                "work_units": outcome.get("work_units"),
                "reused_work_units": outcome.get("reused_work_units"),
                "aggregate_repair_rounds": outcome.get("aggregate_repair_rounds"),
                "retryable": outcome.get("failure_kind") == "provider_rate_limited",
                "retry_action": (
                    "retry_stage"
                    if outcome.get("failure_kind") == "provider_rate_limited"
                    else None
                ),
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


WORK_UNIT_MIN_REPAIR_ROUNDS = 1
WORK_UNIT_MAX_REPAIR_ROUNDS = 1


def _work_unit_summary(units: tuple[StageWorkUnit, ...], candidates: dict[str, dict]) -> list[dict]:
    summaries: list[dict] = []
    for unit in units:
        candidate = candidates.get(unit.unit_id)
        if candidate is None:
            continue
        if unit.stage == "bom":
            content = {
                "groups": [
                    {
                        key: group[key]
                        for key in ("id", "reference_prefix", "quantity", "value", "sheet")
                        if key in group
                    }
                    for group in candidate.get("groups") or []
                ]
            }
        else:
            nets: dict[str, list[str]] = {}
            no_connects: list[str] = []
            for row in candidate.get("pins") or []:
                endpoint = f"{row['ref']}.{row['pin']}"
                if row.get("no_connect") is True:
                    no_connects.append(endpoint)
                else:
                    nets.setdefault(str(row.get("net")), []).append(endpoint)
            content = {"nets": nets, "no_connects": no_connects}
        summaries.append({"unit_id": unit.unit_id, "sheet": unit.sheet, **content})
    return summaries


def _work_unit_provenance(unit: StageWorkUnit) -> dict:
    return {
        "requirement_ids": list(unit.requirement_ids),
        "owned_roles": list(unit.owned_roles),
        "excluded_refs": list(unit.excluded_refs),
        "excluded_pins": [{"ref": ref, "pin": pin} for ref, pin in unit.excluded_pins],
        "planned_resolution_source": unit.planned_resolution_source,
        "recipe_ids": list(unit.recipe_ids),
        "lowerer_ids": list(unit.lowerer_ids),
    }


def _work_unit_instructions(
    unit: StageWorkUnit,
    unit_index: int,
    unit_total: int,
    prompt_state: dict,
) -> str:
    boundary = {
        "unit_id": unit.unit_id,
        "unit_index": unit_index,
        "unit_total": unit_total,
        "target_sheet": unit.sheet,
        "requirement_ids": list(unit.requirement_ids),
        "owned_roles": list(unit.owned_roles),
        "excluded_refs": list(unit.excluded_refs),
        "excluded_pins": [{"ref": ref, "pin": pin} for ref, pin in unit.excluded_pins],
        "planned_resolution_source": unit.planned_resolution_source,
        "recipe_ids": list(unit.recipe_ids),
        "lowerer_ids": list(unit.lowerer_ids),
    }
    if unit.stage == "bom":
        architecture_sheets = (prompt_state.get("architecture") or {}).get("sheets") or []
        target = next(
            (
                sheet
                for sheet in architecture_sheets
                if isinstance(sheet, dict) and sheet.get("name") == unit.sheet
            ),
            {},
        )

        def topology_key(value: object) -> str:
            return re.sub(r"[^a-z0-9]", "", str(value).lower())

        target_tokens = {
            topology_key(unit.sheet),
            topology_key(target.get("stem") or ""),
        }
        topologies = (prompt_state.get("architecture") or {}).get("topologies") or {}
        target_topology = next(
            (value for key, value in topologies.items() if topology_key(key) in target_tokens),
            None,
        )
        boundary["target_topology"] = target_topology
        boundary["target_function"] = target.get("function")
        boundary["excluded_sheets"] = [
            sheet.get("name")
            for sheet in architecture_sheets
            if isinstance(sheet, dict) and sheet.get("name") != unit.sheet
        ]
        boundary["owned_output"] = (
            "Only components physically installed in target_sheet to implement "
            "target_function and target_topology. Emit at least one group unless "
            "a locked circuit recipe already populates this sheet. Never recreate "
            "the whole-board BOM, another sheet's parts, or any component listed "
            "in PRIOR ACCEPTED WORK UNITS."
        )
    else:
        boundary["owned_refs"] = list(unit.refs)
        boundary["owned_pins"] = [{"ref": ref, "pin": pin} for ref, pin in unit.expected_pins]
    return json.dumps(boundary, separators=(",", ":"))


def _work_unit_extras(unit: StageWorkUnit, extras: dict) -> dict:
    if unit.stage == "bom":
        return dict(extras)
    refs = set(unit.refs)
    pinouts = extras.get("symbol_pinouts") or {}
    filtered = {ref: info for ref, info in pinouts.items() if ref in refs}
    if not filtered:
        filtered = dict(pinouts)
    result = {"symbol_pinouts": filtered}
    for key in ("locked_pin_assignments", "locked_no_connect_pins", "recipe_locked_pins"):
        if key in extras:
            result[key] = [
                row
                for row in extras.get(key) or []
                if isinstance(row, dict) and str(row.get("ref")) in refs
            ]
    return result


def _work_unit_user_prompt(
    *,
    stage: str,
    brief: str,
    prompt_state: dict,
    extras: dict,
    unit: StageWorkUnit,
    accepted_summary: list[dict],
    answers,
    instruction,
    feedback: dict | str | None,
) -> str:
    user = (
        f"PROJECT BRIEF:\n{brief}\n\n"
        f"CURRENT DESIGN STATE (JSON):\n{json.dumps(prompt_state, separators=(',', ':'))}\n\n"
        "WORK-UNIT REFERENCE DATA (complete, not truncated):\n"
        f"{json.dumps(_work_unit_extras(unit, extras), separators=(',', ':'))}\n\n"
        "PRIOR ACCEPTED WORK UNITS (read-only):\n"
        f"{json.dumps(accepted_summary, separators=(',', ':'))}"
    )
    if answers:
        qa = "\n".join(
            f"Q: {answer.get('text', '')}\nA: {answer.get('answer', '')}" for answer in answers
        )
        user += f"\n\nThe user answered earlier clarifying questions:\n{qa}"
    if instruction:
        user += f"\n\nUser change request for {stage}: {instruction}"
    if stage == "bom":
        user += _bom_part_hints(
            brief,
            instruction or "",
            *(str(answer.get("answer", "")) for answer in (answers or [])),
        )
    if feedback is not None:
        rendered = feedback if isinstance(feedback, str) else json.dumps(feedback, default=str)
        user += (
            "\n\nAGGREGATE OR LOCAL DEFECTS TO REPAIR:\n"
            f"{rendered}\nReturn a complete replacement for only this work unit."
        )
    return user + f"\n\nProduce the {stage} work-unit JSON now."


def _drive_work_unit_stage(
    client,
    stage: str,
    brief: str,
    state_path,
    workspace,
    *,
    prompt_state: dict,
    extras: dict,
    max_tokens: int,
    max_retries: int,
    progress,
    answers,
    instruction,
    meta_ctx: dict | None,
    review_before_commit: bool,
    attempt_observer: Callable[[dict], None] | None,
    t0: float,
    cpu0: float,
) -> dict:
    """Drive bounded BOM/wiring replacements, then normalize and commit once."""
    run_id = (meta_ctx or {}).get("run_id")
    units = plan_stage_work_units(stage, prompt_state, extras)
    unit_by_id = {unit.unit_id: unit for unit in units}
    repair_rounds = min(
        WORK_UNIT_MAX_REPAIR_ROUNDS,
        max(WORK_UNIT_MIN_REPAIR_ROUNDS, int(max_retries)),
    )
    call_budget = min(64, (repair_rounds + 3) * len(units) + 6)
    policy = _response_policy(client, stage, max_tokens)
    executor = build_bom_executor(workspace, run_design_cli, KICRAFT) if stage == "bom" else None
    candidates: dict[str, dict] = {}
    unit_sources: dict[str, str] = {}
    contracts_by_unit = {
        unit.unit_id: build_stage_response_contract(
            stage,
            prompt_state,
            bom_sheet=unit.sheet if stage == "bom" else None,
            wiring_refs=unit.refs if stage == "wiring" else None,
        )
        for unit in units
    }
    response_contract_names = tuple(
        contracts_by_unit[unit.unit_id].response_format["json_schema"]["name"] for unit in units
    )
    if not response_contract_names:
        response_contract_names = (
            build_stage_response_contract(stage, prompt_state).response_format["json_schema"][
                "name"
            ],
        )
    draft_store = StageDraftStore(state_path, stage)
    draft_fingerprint = stage_draft_fingerprint(
        stage=stage,
        brief=brief,
        prompt_state=prompt_state,
        answers=answers,
        instruction=instruction,
        stage_spec_sha256=stage_spec_sha256(stage),
        response_contract_names=response_contract_names,
        units=units,
        extras=extras,
    )
    loaded_candidates = draft_store.load(draft_fingerprint, units)
    for unit in units:
        loaded = loaded_candidates.get(unit.unit_id)
        if loaded is None:
            continue
        try:
            candidates[unit.unit_id] = validate_unit_candidate(unit, loaded, prompt_state, extras)
            unit_sources[unit.unit_id] = "reuse"
        except (WorkUnitValidationError, TypeError, ValueError):
            candidates.clear()
            unit_sources.clear()
            break
    reused_work_units = len(candidates)
    architecture = prompt_state.get("architecture") or {}
    selections = [
        selection
        for selection in architecture.get("recipe_selections") or []
        if isinstance(selection, dict)
    ]
    recipe_sheets = {
        str(sheet) for selection in selections for sheet in (selection.get("sheets") or {}).values()
    }
    if progress:
        for selection in selections:
            progress(
                {
                    "kind": "recipe_selected",
                    "version": 1,
                    "stage": stage,
                    "source": "recipe",
                    "recipe": selection.get("recipe"),
                    "instance": selection.get("instance"),
                    "sheets": selection.get("sheets") or {},
                    "parameters": selection.get("parameters") or {},
                    "requirement_ids": selection.get("requirement_ids") or [],
                    "port_bindings": selection.get("port_bindings") or {},
                    "pin_allocations": selection.get("pin_allocations") or [],
                    "owned_call_count": 0,
                    "resolution": [
                        record
                        for record in architecture.get("recipe_resolution") or []
                        if isinstance(record, dict)
                        and record.get("requirement_id") in (selection.get("requirement_ids") or [])
                    ],
                }
            )
    for unit in units:
        if unit.unit_id in candidates:
            if progress:
                progress(
                    {
                        "kind": "work_unit_plan",
                        "version": 1,
                        "stage": stage,
                        "unit_id": unit.unit_id,
                        "unit_sheet": unit.sheet,
                        "source": "reused_validated_draft",
                        "refs": list(unit.refs),
                        "expected_pin_count": len(unit.expected_pins),
                        **_work_unit_provenance(unit),
                    }
                )
                progress(
                    {
                        "kind": "work_unit_done",
                        "version": 1,
                        "stage": stage,
                        "unit_id": unit.unit_id,
                        "unit_sheet": unit.sheet,
                        "source": "reused_validated_draft",
                        **_work_unit_provenance(unit),
                    }
                )
            continue
        deterministic = (
            deterministic_bom_candidate(unit, prompt_state)
            if stage == "bom"
            else deterministic_wiring_candidate(unit, prompt_state, extras)
        )
        if deterministic is None:
            if progress:
                progress(
                    {
                        "kind": "work_unit_plan",
                        "version": 1,
                        "stage": stage,
                        "unit_id": unit.unit_id,
                        "unit_sheet": unit.sheet,
                        "source": ("recipe_plus_llm" if unit.sheet in recipe_sheets else "llm"),
                        "refs": list(unit.refs),
                        "expected_pin_count": len(unit.expected_pins),
                        **_work_unit_provenance(unit),
                    }
                )
            continue
        if progress:
            progress(
                {
                    "kind": "work_unit_plan",
                    "version": 1,
                    "stage": stage,
                    "unit_id": unit.unit_id,
                    "unit_sheet": unit.sheet,
                    "source": "deterministic_architecture_lowering",
                    "refs": list(unit.refs),
                    "expected_pin_count": len(unit.expected_pins),
                    **_work_unit_provenance(unit),
                }
            )
        candidates[unit.unit_id] = validate_unit_candidate(
            unit,
            deterministic,
            prompt_state,
            extras,
        )
        unit_sources[unit.unit_id] = "lowerer"
        if progress:
            progress(
                {
                    "kind": "work_unit_done",
                    "version": 1,
                    "stage": stage,
                    "unit_id": unit.unit_id,
                    "unit_sheet": unit.sheet,
                    "source": "deterministic_architecture_lowering",
                    **_work_unit_provenance(unit),
                }
            )
    total_cost = 0.0
    attempts = 0
    rounds_total = 0 if stage == "bom" else None
    tool_calls_total = 0 if stage == "bom" else None
    stage_client = client
    provider_ok = False
    schema_ok = False
    expanded_component_count = 0
    aggregate_repair_rounds = 0
    last: dict = {}
    validated_invocation_cache: dict[str, dict] = {}

    def record(
        active_client,
        *,
        unit: StageWorkUnit,
        unit_attempt: int,
        call_mode: str,
        outcome: str,
        facts: ProviderFacts | None = None,
        error_facts: dict | None = None,
        aggregate_round: int | None = None,
        aggregate_signature: tuple | None = None,
        commit_result: dict | None = None,
        candidate_retained: bool | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        _record_attempt_facts(
            active_client,
            run_id=run_id,
            stage=stage,
            attempt=attempts,
            call_mode=call_mode,
            outcome=outcome,
            facts=facts,
            error_facts=error_facts,
            unit_id=unit.unit_id,
            unit_attempt=unit_attempt,
            aggregate_round=aggregate_round,
            commit_result=commit_result,
            candidate_retained=candidate_retained,
            fallback_reason=fallback_reason,
        )
        if progress:
            progress(
                _work_unit_attempt_event(
                    active_client,
                    stage=stage,
                    unit=unit,
                    provider_attempt=attempts,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome=outcome,
                    facts=facts,
                    error_facts=error_facts,
                    aggregate_round=aggregate_round,
                    commit_result=commit_result,
                    candidate_retained=candidate_retained,
                    fallback_reason=fallback_reason,
                )
            )
        if attempt_observer is not None:
            settings = getattr(active_client, "s", None)
            attempt_observer(
                {
                    "version": 2,
                    "stage": stage,
                    "provider_attempt": attempts,
                    "call_mode": call_mode,
                    "outcome": outcome,
                    "model": _client_model(active_client),
                    "design_profile": getattr(settings, "design_profile", None),
                    "provider": facts.provider if facts is not None else None,
                    "provider_order": list(getattr(settings, "provider_order", ()) or ()),
                    "unit_id": unit.unit_id,
                    "unit_attempt": unit_attempt,
                    "unit_index": units.index(unit) + 1,
                    "unit_total": len(units),
                    "aggregate_round": aggregate_round,
                    "aggregate_signature": (
                        [list(aggregate_signature[0]), list(aggregate_signature[1])]
                        if aggregate_signature is not None
                        else None
                    ),
                }
            )

    def invoke(
        unit: StageWorkUnit,
        *,
        unit_attempt: int,
        feedback: dict | str | None,
        pristine: bool,
        serialization: bool,
        aggregate_round: int | None,
    ) -> tuple[str, dict | None, ProviderFacts | None, str | None]:
        nonlocal attempts, total_cost, rounds_total, tool_calls_total, provider_ok, stage_client
        aggregate_signature = (
            _commit_rejection_signature(feedback["aggregate_commit_rejection"])
            if isinstance(feedback, dict)
            and isinstance(feedback.get("aggregate_commit_rejection"), dict)
            else None
        )
        unit_policy = (
            replace(
                policy,
                collection_bounds=(
                    CollectionBound(
                        field="pins",
                        total=max(1, len(unit.expected_pins)),
                    ),
                ),
            )
            if stage == "wiring"
            else policy
        )
        contract = contracts_by_unit[unit.unit_id]
        instructions = _work_unit_instructions(
            unit,
            units.index(unit) + 1,
            len(units),
            prompt_state,
        )
        user = _work_unit_user_prompt(
            stage=stage,
            brief=brief,
            prompt_state=prompt_state,
            extras=extras,
            unit=unit,
            accepted_summary=_work_unit_summary(units, candidates),
            answers=answers,
            instruction=instruction,
            feedback=feedback,
        )
        messages = [
            {
                "role": "system",
                "content": build_system(
                    contract,
                    unit_policy.collection_bounds,
                    work_unit_instructions=instructions,
                ),
            },
            {"role": "user", "content": user},
        ]
        active_client = stage_client
        prepared = PreparedStage(
            stage=stage,
            prompt_state=prompt_state,
            extras=extras,
            base_messages=tuple(messages),
            contract=contract,
            policy=unit_policy,
            tools=BOM_TOOLS if stage == "bom" else None,
            executor=executor,
        )
        response_format: dict | None = contract.response_format
        fallback_spent = False
        schema_less_spent = False
        call_mode = "serialization" if serialization else "clean_slate" if pristine else "normal"
        cache_payload = {
            "stage": stage,
            "unit_id": unit.unit_id,
            "messages": messages,
            "response_format": response_format,
            "call_mode": call_mode,
            "model": _client_model(active_client),
            "profile": getattr(getattr(active_client, "s", None), "design_profile", None),
        }
        cache_key = hashlib.sha256(
            json.dumps(
                cache_payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if cache_key in validated_invocation_cache:
            return (
                "candidate",
                json.loads(json.dumps(validated_invocation_cache[cache_key])),
                None,
                None,
            )
        while attempts < call_budget:
            attempts += 1
            ctx = {
                **(meta_ctx or {}),
                "stage": stage,
                "attempt": attempts,
                "unit_id": unit.unit_id,
                "unit_sheet": unit.sheet,
                "unit_refs": list(unit.refs),
                "expected_pins": [list(pin) for pin in unit.expected_pins],
                "unit_attempt": unit_attempt,
                "aggregate_round": aggregate_round,
                "serialization": serialization,
                "call_mode": call_mode,
            }
            try:
                if serialization:
                    facts = run_serialization_recovery(
                        active_client,
                        prepared,
                        messages=list(messages),
                        response_format=response_format,
                        temperature=max(
                            float(
                                getattr(
                                    getattr(active_client, "s", None),
                                    "serialization_escape_temperature",
                                    0.4,
                                )
                            ),
                            0.0,
                        ),
                        reasoning_guard=policy.reasoning_guard,
                        progress=progress,
                        meta_ctx=ctx,
                    )
                else:
                    facts = call_stage_provider(
                        active_client,
                        prepared,
                        messages=list(messages),
                        response_format=response_format,
                        max_tokens=int(policy.normal_max_tokens),
                        temperature=(
                            max(
                                float(
                                    getattr(
                                        getattr(active_client, "s", None),
                                        "serialization_escape_temperature",
                                        0.4,
                                    )
                                ),
                                0.0,
                            )
                            if pristine
                            else _design_temperature(active_client)
                        ),
                        reasoning={"enabled": False} if pristine else policy.normal_reasoning,
                        reasoning_guard=policy.reasoning_guard,
                        progress=progress,
                        meta_ctx=ctx,
                    )
            except (*_TRANSPORT_FAILURE_EXC, *_PROVIDER_FAILURE_EXC) as exc:
                failure = classify_provider_exception(exc)
                kind = failure["failure_kind"]
                record(
                    active_client,
                    unit=unit,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome=kind,
                    error_facts={
                        key: value for key, value in failure.items() if key != "failure_kind"
                    },
                    aggregate_round=aggregate_round,
                    aggregate_signature=aggregate_signature,
                )
                if (
                    kind
                    in {
                        "provider_response_format_rejected",
                        "provider_capability_rejected",
                        "provider_request_rejected",
                    }
                    and response_format is not None
                    and not schema_less_spent
                ):
                    response_format = None
                    schema_less_spent = True
                    continue
                transient = kind == "provider_rate_limited"
                fallback_profile = getattr(
                    getattr(active_client, "s", None),
                    "provider_fallback_profile",
                    "",
                )
                switch = getattr(active_client, "with_design_profile", None)
                if transient and not fallback_spent and fallback_profile and callable(switch):
                    active_client = switch(fallback_profile)
                    fallback_spent = True
                    continue
                return "terminal", failure, None, kind
            total_cost += facts.cost_usd
            provider_ok = True
            if stage == "bom":
                rounds_total += int(facts.rounds or 0)
                tool_calls_total += int(facts.tool_calls or 0)
            if facts.finish == "collection_limit":
                record(
                    active_client,
                    unit=unit,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome="collection_limit",
                    facts=facts,
                    aggregate_round=aggregate_round,
                    aggregate_signature=aggregate_signature,
                    commit_result=(
                        feedback.get("aggregate_commit_rejection")
                        if isinstance(feedback, dict)
                        else None
                    ),
                    candidate_retained=False,
                    fallback_reason=("provider_rate_limited" if fallback_spent else None),
                )
                return "recoverable", None, facts, "collection_limit"
            try:
                parsed = _extract_json(facts.raw)
            except (json.JSONDecodeError, ValueError):
                parse_kind = _classify_parse_failure(facts.finish, facts.had_content)
                record(
                    active_client,
                    unit=unit,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome=parse_kind,
                    facts=facts,
                    aggregate_round=aggregate_round,
                    aggregate_signature=aggregate_signature,
                    candidate_retained=False,
                    fallback_reason=("provider_rate_limited" if fallback_spent else None),
                )
                return "recoverable", None, facts, parse_kind
            if isinstance(parsed.get("questions"), list):
                record(
                    active_client,
                    unit=unit,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome="question",
                    facts=facts,
                    aggregate_round=aggregate_round,
                    aggregate_signature=aggregate_signature,
                    candidate_retained=False,
                    fallback_reason=("provider_rate_limited" if fallback_spent else None),
                )
                return "questions", parsed, facts, None
            try:
                validated = validate_unit_candidate(unit, parsed, prompt_state, extras)
            except (WorkUnitValidationError, TypeError, ValueError) as exc:
                record(
                    active_client,
                    unit=unit,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome="invalid_work_unit",
                    facts=facts,
                    aggregate_round=aggregate_round,
                    aggregate_signature=aggregate_signature,
                    candidate_retained=False,
                    fallback_reason=("provider_rate_limited" if fallback_spent else None),
                )
                return "recoverable", parsed, facts, str(exc)
            record(
                active_client,
                unit=unit,
                unit_attempt=unit_attempt,
                call_mode=call_mode,
                outcome="candidate",
                facts=facts,
                aggregate_round=aggregate_round,
                aggregate_signature=aggregate_signature,
                commit_result=(
                    feedback.get("aggregate_commit_rejection")
                    if isinstance(feedback, dict)
                    else None
                ),
                candidate_retained=True,
                fallback_reason=("provider_rate_limited" if fallback_spent else None),
            )
            validated_invocation_cache[cache_key] = json.loads(json.dumps(validated))
            return "candidate", validated, facts, None
        return "terminal", None, None, "provider_call_budget_exhausted"

    def draft(
        unit: StageWorkUnit,
        *,
        feedback: dict | str | None = None,
        force_pristine: bool = False,
        aggregate_round: int | None = None,
    ) -> tuple[str, dict | None]:
        prior_signature: str | None = None
        local_feedback = feedback
        serialization_next = False
        serialization_spent = False
        for unit_attempt in range(1, repair_rounds + 3):
            used_serialization = serialization_next
            kind, payload, facts, error = invoke(
                unit,
                unit_attempt=unit_attempt,
                feedback=local_feedback,
                pristine=False,
                serialization=used_serialization,
                aggregate_round=aggregate_round,
            )
            serialization_next = False
            if kind == "candidate":
                if progress:
                    progress(
                        {
                            "kind": "work_unit_done",
                            "version": 1,
                            "stage": stage,
                            "unit_id": unit.unit_id,
                            "unit_sheet": unit.sheet,
                            "source": "llm",
                            "unit_attempt": unit_attempt,
                            "provider_attempt": attempts,
                            "aggregate_round": aggregate_round,
                        }
                    )
                return kind, payload
            if kind == "questions":
                questions = _normalize_questions(payload.get("questions") or [], stage)
                reconcile = any(question.get("reconcile_target") for question in questions)
                if any(question["blocking"] for question in questions) and (
                    (not answers and not instruction) or reconcile
                ):
                    return kind, {"questions": questions}
                local_feedback = (
                    "Do not ask more questions. Apply sensible defaults, record each "
                    "assumption ending '(defaulted)', and return this unit."
                )
                continue
            if kind == "terminal":
                return kind, {"failure_kind": error or "provider_failure"}
            serialization_errors = {
                "collection_limit",
                "truncated_json",
                "invalid_json",
                "reasoning_loop",
            }
            if used_serialization and error in serialization_errors:
                return "terminal", {"failure_kind": error}
            if not serialization_spent and not used_serialization and error in serialization_errors:
                serialization_spent = True
                serialization_next = True
                bounds_sentence = _collection_bounds_sentence(policy.collection_bounds)
                local_feedback = _SERIALIZATION_RETRY_MSG.format(
                    prior_chars=len(facts.raw if facts is not None else ""),
                    bounds_sentence=(bounds_sentence + " ") if bounds_sentence else "",
                )
                continue
            signature = str(error)
            if signature == prior_signature:
                return "terminal", {
                    "failure_kind": "repeated_unit_defect",
                    "error": signature,
                }
            local_feedback = {
                "defect": signature,
                "rejected_unit": payload,
                "instruction": "Correct every listed defect in this complete unit replacement.",
            }
            prior_signature = signature
        return "terminal", {
            "failure_kind": "unit_repair_exhausted",
            "error": str(local_feedback),
        }

    def stop_for_question(payload: dict) -> dict:
        questions = payload["questions"]
        draft_store.delete()
        if not review_before_commit:
            attach_questions(state_path, stage, questions)
        if progress:
            progress({"kind": "question", "stage": stage, "questions": questions})
        return {
            "stage": stage,
            "commit_ok": False,
            "needs_input": True,
            "questions": questions,
            "cost_usd": total_cost,
            "attempts": attempts,
            "rounds": rounds_total,
            "tool_calls": tool_calls_total,
            "work_units": len(units),
            "reused_work_units": reused_work_units,
            "aggregate_repair_rounds": aggregate_repair_rounds,
        }

    for unit in units:
        if unit.unit_id in candidates:
            continue
        kind, payload = draft(unit)
        if kind == "questions":
            return stop_for_question(payload)
        if kind != "candidate":
            last = payload or {"failure_kind": "unit_generation_failed"}
            break
        candidates[unit.unit_id] = payload
        unit_sources[unit.unit_id] = "llm"
        draft_store.save(draft_fingerprint, units, candidates)
    else:
        last = {}

    def aggregate():
        nonlocal expanded_component_count
        if stage == "bom":
            merged, ref_to_unit, ref_to_lowering = merge_bom_units(units, candidates, prompt_state)
            normalized, expanded_component_count = _normalize_stage_response(
                stage, merged, prompt_state
            )
            for part in normalized.get("parts") or []:
                if part.get("recipe_id"):
                    continue
                unit_id = ref_to_unit.get(str(part.get("ref")))
                if unit_id:
                    part["resolution_source"] = unit_sources.get(unit_id, "llm")
                    part["resolution_id"] = unit_id
                    lowering = ref_to_lowering.get(str(part.get("ref")))
                    if lowering:
                        part.update(lowering)
            return normalized, ref_to_unit, {}, {}
        merged, pin_to_unit, ref_to_unit_ids = merge_wiring_units(units, candidates)
        normalized, _expanded = _normalize_stage_response(stage, merged, prompt_state)
        return normalized, {}, pin_to_unit, ref_to_unit_ids

    if not last:
        try:
            candidate, ref_to_unit, pin_to_unit, ref_to_unit_ids = aggregate()
            schema_ok = True
        except (StageSchemaError, TypeError, ValueError) as exc:
            last = {"failure_kind": "invalid_schema", "error": str(exc)}
    if last:
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
            rounds=rounds_total,
            tool_calls=tool_calls_total,
            emitted_collection_count=0,
            expanded_component_count=expanded_component_count,
            outcome={
                **last,
                "provider_ok": provider_ok,
                "schema_ok": schema_ok,
                "work_units": len(units),
                "reused_work_units": reused_work_units,
                "aggregate_repair_rounds": aggregate_repair_rounds,
            },
        )

    semantic_state = {
        **prompt_state,
        "_stage_answers": list(answers or []),
        "_stage_extras": extras,
    }
    diagnostics = diagnose_stage(
        stage, brief=brief, upstream_state=semantic_state, candidate=candidate
    )
    severe = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.severity in {"repair_required", "fab_gate"}
    ]
    semantic_mode = getattr(getattr(client, "s", None), "stage_semantics", "observe")
    for semantic_round in range(1, _MAX_SEMANTIC_REPAIR_ROUNDS + 1):
        if not severe or semantic_mode not in {"repair", "enforce"}:
            break
        targeted = route_work_unit_ids(
            [diagnostic.model_dump(exclude_none=True) for diagnostic in severe],
            units,
            ref_to_unit=ref_to_unit,
            pin_to_unit=pin_to_unit,
            ref_to_unit_ids=ref_to_unit_ids,
        )
        prior_candidates = dict(candidates)
        for unit_id in targeted:
            unit = unit_by_id[unit_id]
            kind, payload = draft(
                unit,
                feedback={
                    "aggregate_diagnostics": [
                        diagnostic.model_dump(exclude_none=True) for diagnostic in severe
                    ],
                    "aggregate_summary": _work_unit_summary(units, candidates),
                },
                aggregate_round=semantic_round,
            )
            if kind == "questions":
                return stop_for_question(payload)
            if kind != "candidate":
                candidates = prior_candidates
                break
            candidates[unit_id] = payload
            draft_store.save(draft_fingerprint, units, candidates)
        else:
            repaired, new_ref_map, new_pin_map, new_ref_units = aggregate()
            repaired_diagnostics = diagnose_stage(
                stage,
                brief=brief,
                upstream_state=semantic_state,
                candidate=repaired,
            )
            repaired_severe = [
                diagnostic
                for diagnostic in repaired_diagnostics
                if diagnostic.severity in {"repair_required", "fab_gate"}
            ]
            if _semantic_defect_score(repaired_severe) < _semantic_defect_score(severe):
                candidate = repaired
                diagnostics = repaired_diagnostics
                severe = repaired_severe
                ref_to_unit = new_ref_map
                pin_to_unit = new_pin_map
                ref_to_unit_ids = new_ref_units
                continue
            candidates = prior_candidates
        break

    diagnostic_rows = [diagnostic.model_dump(exclude_none=True) for diagnostic in diagnostics]
    if review_before_commit:
        return {
            "stage": stage,
            "needs_review": True,
            "commit_ok": False,
            "slot": candidate,
            "diagnostics": diagnostic_rows,
            "cost_usd": total_cost,
            "attempts": attempts,
            "rounds": rounds_total,
            "tool_calls": tool_calls_total,
            "wall_s": round(time.monotonic() - t0, 3),
            "cpu_s": round(_child_cpu_s() - cpu0, 3),
            "provider_ok": provider_ok,
            "schema_ok": schema_ok,
            "semantic_clean": not diagnostics,
            "repair_required": bool(severe),
            "fab_safe": not any(diagnostic.severity == "fab_gate" for diagnostic in diagnostics),
            "work_units": len(units),
            "reused_work_units": reused_work_units,
            "aggregate_repair_rounds": aggregate_repair_rounds,
            "debug_context": {
                "prompt_state": prompt_state,
                "extras": extras,
                "work_units": [asdict(unit) for unit in units],
                "accepted_units": _work_unit_summary(units, candidates),
            },
        }

    prior_signature = None
    for aggregate_round in range(repair_rounds + 1):
        ok, commit_result = commit_stage(stage, dict(candidate), state_path, brief, None, workspace)
        if ok:
            draft_store.delete()
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
                rounds=rounds_total,
                tool_calls=tool_calls_total,
                emitted_collection_count=0,
                expanded_component_count=expanded_component_count,
                outcome={
                    "commit": commit_result,
                    "slot": candidate,
                    "provider_ok": provider_ok,
                    "schema_ok": schema_ok,
                    "semantic_clean": not diagnostics,
                    "repair_required": bool(severe),
                    "fab_safe": not any(
                        diagnostic.severity == "fab_gate" for diagnostic in diagnostics
                    ),
                    "diagnostics": diagnostic_rows,
                    "work_units": len(units),
                    "reused_work_units": reused_work_units,
                    "aggregate_repair_rounds": aggregate_repair_rounds,
                },
            )
        _record_attempt_facts(
            stage_client,
            run_id=run_id,
            stage=stage,
            attempt=max(1, attempts),
            call_mode="deterministic_commit",
            outcome=(
                "commit_process_failed"
                if commit_result.get("failure_kind") == "commit_process_failed"
                else "commit_rejected"
            ),
            commit_result=commit_result,
            candidate_retained=True,
        )
        if commit_result.get("failure_kind") == "commit_process_failed":
            last = {
                "failure_kind": "commit_process_failed",
                "error": (commit_result.get("errors") or ["stage-commit process failed"])[0],
                "commit": commit_result,
            }
            break
        if progress:
            progress(
                {
                    "kind": "retry",
                    "stage": stage,
                    "errors": commit_result.get("errors"),
                    "offenders": commit_result.get("offenders"),
                    "model": _client_model(stage_client),
                }
            )
        signature = _commit_rejection_signature(commit_result)
        repeated = signature == prior_signature
        if repeated:
            last = {
                "failure_kind": "commit_rejected",
                "commit": commit_result,
            }
            break
        pristine = False
        if aggregate_round >= repair_rounds:
            last = {
                "failure_kind": "commit_rejected",
                "commit": commit_result,
            }
            break
        targeted = route_work_unit_ids(
            commit_result,
            units,
            ref_to_unit=ref_to_unit,
            pin_to_unit=pin_to_unit,
            ref_to_unit_ids=ref_to_unit_ids,
        )
        aggregate_repair_rounds += 1
        prior_candidates = dict(candidates)
        for unit_id in targeted:
            kind, payload = draft(
                unit_by_id[unit_id],
                feedback={
                    "aggregate_commit_rejection": commit_result,
                    "aggregate_summary": _work_unit_summary(units, candidates),
                    "repair_instruction": _retry_feedback(
                        commit_result,
                        stage=stage,
                        valid_refs=(committed_bom_refs(state_path) if stage == "wiring" else None),
                    ),
                },
                force_pristine=pristine,
                aggregate_round=aggregate_repair_rounds,
            )
            if kind == "questions":
                return stop_for_question(payload)
            if kind != "candidate":
                last = payload or {"failure_kind": "unit_generation_failed"}
                break
            candidates[unit_id] = payload
            draft_store.save(draft_fingerprint, units, candidates)
        else:
            try:
                candidate, ref_to_unit, pin_to_unit, ref_to_unit_ids = aggregate()
            except (StageSchemaError, TypeError, ValueError) as exc:
                candidates = prior_candidates
                last = {"failure_kind": "invalid_schema", "error": str(exc)}
                break
            prior_signature = signature
            continue
        candidates = prior_candidates
        break
    else:
        last = {"failure_kind": "commit_rejected"}
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
        rounds=rounds_total,
        tool_calls=tool_calls_total,
        emitted_collection_count=0,
        expanded_component_count=expanded_component_count,
        outcome={
            **last,
            "provider_ok": provider_ok,
            "schema_ok": schema_ok,
            "diagnostics": diagnostic_rows,
            "work_units": len(units),
            "reused_work_units": reused_work_units,
            "aggregate_repair_rounds": aggregate_repair_rounds,
        },
    )


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
    attempt_observer: Callable[[dict], None] | None = None,
) -> dict:
    run_id = (meta_ctx or {}).get("run_id")
    active_client = client
    t0 = time.monotonic()
    cpu0 = _child_cpu_s()
    if progress:
        progress({"kind": "stage_start", "stage": stage, "model": _client_model(client)})

    def operational_failure(failure_kind: str, error: str) -> dict:
        wall_s = round(time.monotonic() - t0, 3)
        cpu_s = round(_child_cpu_s() - cpu0, 3)
        if not review_before_commit:
            stamp_stage_status(
                state_path,
                stage,
                False,
                attempts=0,
                wall_s=wall_s,
                cpu_s=cpu_s,
                error=error,
                failure_kind=failure_kind,
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
                failure_kind=failure_kind,
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
            "failure_kind": failure_kind,
        }

    prep = prepare_stage(stage, state_path, workspace)
    if prep.returncode != 0:
        err = (prep.stderr.strip() or prep.stdout.strip() or "no subprocess output")[:600]
        return operational_failure("stage_prep_failed", f"stage-prep failed: {err}")
    try:
        prep_json = json.loads(prep.stdout)
        if not isinstance(prep_json, dict) or not isinstance(prep_json.get("state"), dict):
            raise ValueError("stage-prep output is missing the state object")
        raw_extras = prep_json.get("extras")
        if raw_extras is not None and not isinstance(raw_extras, dict):
            raise ValueError("stage-prep extras must be an object")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return operational_failure(
            "stage_prep_failed",
            f"stage-prep returned invalid process output: {exc}",
        )
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
    if stage == "architecture":
        from kicraft.design.lowering import lowerer_summaries

        extras["circuit_lowerers"] = lowerer_summaries()

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
            "recipe_ownership": full_bom.get("recipe_ownership") or [],
        }
    if stage in {"bom", "wiring"}:
        try:
            return _drive_work_unit_stage(
                client,
                stage,
                brief,
                state_path,
                workspace,
                prompt_state=prompt_state,
                extras=extras,
                max_tokens=max_tokens,
                max_retries=max_retries,
                progress=progress,
                answers=answers,
                instruction=instruction,
                meta_ctx=meta_ctx,
                review_before_commit=review_before_commit,
                attempt_observer=attempt_observer,
                t0=t0,
                cpu0=cpu0,
            )
        except ValueError as exc:
            return operational_failure(
                "stage_contract_failed",
                f"stage contract failed: {exc}",
            )

    user = f"PROJECT BRIEF:\n{brief}\n\nCURRENT DESIGN STATE (JSON):\n{json.dumps(prompt_state)}"
    if extras:
        # bom carries the full parts table + core defaults (the adoption rule
        # depends on both being complete), wiring carries symbol_pinouts.
        budget = {"wiring": 40000, "bom": 20000}.get(stage, 24000)
        user += f"\n\nSTAGE EXTRAS (reference data from stage-prep):\n{json.dumps(extras)[:budget]}"
    if answers:
        qa = "\n".join(f"Q: {a.get('text', '')}\nA: {a.get('answer', '')}" for a in answers)
        user += f"\n\nThe user answered your earlier clarifying question(s):\n{qa}"
    semantic_state = {
        **prompt_state,
        "_stage_answers": list(answers or []),
        "_stage_extras": extras,
    }
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
        contract = build_stage_response_contract(
            stage,
            prompt_state,
            allow_questions=not (
                stage == "architecture" and instruction == NONINTERACTIVE_DEFAULTS_INSTRUCTION
            ),
        )
    except ValueError as exc:
        return operational_failure(
            "stage_contract_failed",
            f"stage contract failed: {exc}",
        )
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
                "design_temperature": _design_temperature(active_client),
                "serialization_escape_temperature": float(
                    getattr(
                        getattr(active_client, "s", None),
                        "serialization_escape_temperature",
                        0.4,
                    )
                ),
                "stage_semantics": getattr(
                    getattr(active_client, "s", None), "stage_semantics", "observe"
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
    temperature = _design_temperature(active_client)
    reasoning = policy.normal_reasoning
    reasoning_guard = policy.reasoning_guard
    escape_temperature = float(
        getattr(getattr(active_client, "s", None), "serialization_escape_temperature", 0.4)
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
    provider_call_budget = max_retries + 1
    provider_ok = False
    schema_ok = False
    semantic_repair_attempted = False
    semantic_repair_adopted = False
    semantic_repair_rounds = 0
    adopted_repair_facts: ProviderFacts | None = None
    semantic_mode = getattr(getattr(active_client, "s", None), "stage_semantics", "observe")
    current_facts = None
    current_call_mode = "normal"
    current_attempt_number = 0
    escalation_pending = False
    escalation_done = False
    provider_fallback_spent = False
    provider_fallback_active = False

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
        if escalation_pending:
            profile_name = getattr(getattr(active_client, "s", None), "escalation_profile", "")
            switch_profile = getattr(active_client, "with_design_profile", None)
            if (
                profile_name
                and profile_name
                != getattr(getattr(active_client, "s", None), "design_profile", None)
                and callable(switch_profile)
            ):
                prior_model = _client_model(active_client)
                active_client = switch_profile(profile_name)
                escalation_done = True
                if progress:
                    progress(
                        {
                            "kind": "escalation",
                            "stage": "wiring",
                            "from": prior_model,
                            "to": _client_model(active_client),
                            "attempt": attempts + 1,
                            "reason": "repeated_commit_signature",
                        }
                    )
            escalation_pending = False
        ctx = {
            **(meta_ctx or {}),
            "stage": stage,
            "attempt": attempt,
            "call_mode": "clean_slate" if clean_slate_next else "normal",
        }
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
        transport_terminal = False
        while True:
            if attempts >= provider_call_budget:
                transport_terminal = True
                break
            attempts += 1  # a call IS attempted even when it raises below
            current_attempt_number = attempts
            try:
                facts = call_stage_provider(
                    active_client,
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
                break
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
                _observe_attempt(
                    attempt_observer,
                    client=active_client,
                    stage=stage,
                    provider_attempt=attempts,
                    call_mode="clean_slate" if was_clean_slate else "normal",
                    outcome=kind,
                    clean_slate_used=was_clean_slate,
                    escalated=escalation_done,
                    provider_fallback=provider_fallback_active,
                )
                _record_attempt_facts(
                    active_client,
                    run_id=run_id,
                    stage=stage,
                    attempt=attempts,
                    call_mode="clean_slate" if was_clean_slate else "normal",
                    outcome=kind,
                    error_facts={k: v for k, v in failure.items() if k != "failure_kind"},
                    fallback_reason=("provider_rate_limited" if provider_fallback_active else None),
                )
                fallback_profile = getattr(
                    getattr(active_client, "s", None),
                    "provider_fallback_profile",
                    "",
                )
                switch_profile = getattr(active_client, "with_design_profile", None)
                can_fallback = (
                    kind == "provider_rate_limited"
                    and not provider_fallback_spent
                    and attempts < provider_call_budget
                    and fallback_profile
                    and fallback_profile
                    != getattr(getattr(active_client, "s", None), "design_profile", None)
                    and callable(switch_profile)
                )
                if not can_fallback:
                    transport_terminal = True
                    break
                provider_fallback_spent = True
                prior_model = _client_model(active_client)
                prior_profile = getattr(getattr(active_client, "s", None), "design_profile", None)
                prior_providers = list(
                    getattr(getattr(active_client, "s", None), "provider_order", ())
                )
                active_client = switch_profile(fallback_profile)
                provider_fallback_active = True
                if progress:
                    progress(
                        {
                            "kind": "retry",
                            "stage": stage,
                            "errors": [last["error"]],
                            "failure_kind": kind,
                            "model": prior_model,
                        }
                    )
                    progress(
                        {
                            "kind": "provider_fallback",
                            "stage": stage,
                            "from": prior_model,
                            "to": _client_model(active_client),
                            "from_profile": prior_profile,
                            "to_profile": fallback_profile,
                            "from_providers": prior_providers,
                            "to_providers": list(
                                getattr(
                                    getattr(active_client, "s", None),
                                    "provider_order",
                                    (),
                                )
                            ),
                            "attempt": attempts + 1,
                            "reason": "provider_rate_limited",
                        }
                    )
        if transport_terminal:
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
            _observe_attempt(
                attempt_observer,
                client=active_client,
                stage=stage,
                provider_attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome="reasoning_loop",
                facts=facts,
                clean_slate_used=was_clean_slate,
                escalated=escalation_done,
                provider_fallback=provider_fallback_active,
            )
            _record_attempt_facts(
                active_client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome="reasoning_loop",
                facts=facts,
                fallback_reason=("provider_rate_limited" if provider_fallback_active else None),
            )
            reasoning_was_enabled = not (
                isinstance(reasoning, dict) and reasoning.get("enabled") is False
            )
            if (
                not reasoning_was_enabled
                or loop_retries >= _MAX_LOOP_RETRIES
                or attempts >= provider_call_budget
            ):
                break
            if progress:
                progress(
                    {
                        "kind": "retry",
                        "stage": stage,
                        "errors": [
                            "reasoning loop detected"
                            + (f" ({loop_abort_reason})" if loop_abort_reason else "")
                            + " — retrying once with reasoning disabled"
                        ],
                        "model": _client_model(active_client),
                    }
                )
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
                active_client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome="question",
                facts=facts,
            )
        if outcome.kind == "recoverable_failure":
            _observe_attempt(
                attempt_observer,
                client=active_client,
                stage=stage,
                provider_attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome=kind or "invalid_schema",
                facts=facts,
                clean_slate_used=was_clean_slate,
                escalated=escalation_done,
                provider_fallback=provider_fallback_active,
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
                active_client,
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
                        "model": _client_model(active_client),
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
            sctx = {**ctx, "serialization": True, "call_mode": "serialization"}
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
                    active_client,
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
                _observe_attempt(
                    attempt_observer,
                    client=active_client,
                    stage=stage,
                    provider_attempt=attempts,
                    call_mode="serialization",
                    outcome=skind,
                    clean_slate_used=was_clean_slate,
                    escalated=escalation_done,
                    provider_fallback=provider_fallback_active,
                )
                _record_attempt_facts(
                    active_client,
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
                    active_client,
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
                _observe_attempt(
                    attempt_observer,
                    client=active_client,
                    stage=stage,
                    provider_attempt=attempts,
                    call_mode="serialization",
                    outcome=(
                        "question"
                        if serialization_outcome.kind == "questions"
                        else skind or "invalid_schema"
                    ),
                    facts=sfacts,
                    candidate=(
                        serialization_outcome.payload.get("candidate")
                        if serialization_outcome.kind == "questions"
                        else None
                    ),
                    clean_slate_used=was_clean_slate,
                    escalated=escalation_done,
                    provider_fallback=provider_fallback_active,
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
            if current_call_mode != "serialization":
                _observe_attempt(
                    attempt_observer,
                    client=active_client,
                    stage=stage,
                    provider_attempt=current_attempt_number,
                    call_mode=current_call_mode,
                    outcome="question",
                    facts=current_facts,
                    candidate=obj,
                    clean_slate_used=was_clean_slate,
                    escalated=escalation_done,
                    provider_fallback=provider_fallback_active,
                )
            # A reconcile_target park is the pipeline's ESCALATION (a BOM shortfall
            # wiring can't fix), not a user question. Surface it even after answers
            # were applied, so the shared bom-reconcile re-drive can add the parts
            # -- otherwise the "do not ask more questions" retry below burns the
            # stage's whole budget on a park it can never satisfy (WS6).
            is_reconcile_park = any(q.get("reconcile_target") for q in qs)
            if any(q["blocking"] for q in qs) and (
                (not answers and not instruction) or is_reconcile_park
            ):
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

        if stage == "intent":
            obj = complete_intent_classification(brief, obj)
            obj["project_stem"] = normalize_project_stem(obj.get("project_stem", ""))
        elif stage == "functional_spec":
            obj = remove_mislabeled_functional_defaults(brief, semantic_state, obj)
        elif stage == "architecture":
            obj = remove_mislabeled_architecture_defaults(semantic_state, obj)
        original_obj = obj
        diagnostics = diagnose_stage(
            stage, brief=brief, upstream_state=semantic_state, candidate=obj
        )
        provider_diagnostic_codes = [diagnostic.code for diagnostic in diagnostics]
        if progress:
            for diagnostic in diagnostics:
                progress(
                    {
                        "kind": "stage_diagnostic",
                        "stage": stage,
                        "attempt": current_attempt_number,
                        **diagnostic.model_dump(exclude_none=True),
                    }
                )
        if stage == "architecture":
            completed_obj = complete_unsourced_external_rails(obj, diagnostics)
            if completed_obj is not obj:
                obj = completed_obj
                diagnostics = diagnose_stage(
                    stage,
                    brief=brief,
                    upstream_state=semantic_state,
                    candidate=obj,
                )
        severe = [d for d in diagnostics if d.severity in {"repair_required", "fab_gate"}]
        repair_source_raw = raw
        while (
            severe
            and semantic_mode in {"repair", "enforce"}
            and semantic_repair_rounds < _MAX_SEMANTIC_REPAIR_ROUNDS
            and attempts < provider_call_budget
        ):
            semantic_repair_attempted = True
            semantic_repair_rounds += 1
            attempts += 1
            repair_message = _semantic_repair_message(stage, severe)
            repair_messages = _lean_retry(repair_source_raw, repair_message)
            try:
                repair_facts = run_serialization_recovery(
                    active_client,
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
                    if stage == "functional_spec":
                        repaired = remove_mislabeled_functional_defaults(
                            brief, semantic_state, repaired
                        )
                    elif stage == "architecture":
                        repaired = remove_mislabeled_architecture_defaults(semantic_state, repaired)
                        initial_repaired_diagnostics = diagnose_stage(
                            stage,
                            brief=brief,
                            upstream_state=semantic_state,
                            candidate=repaired,
                        )
                        repaired = complete_unsourced_external_rails(
                            repaired,
                            initial_repaired_diagnostics,
                        )
                    repaired_diagnostics = diagnose_stage(
                        stage, brief=brief, upstream_state=semantic_state, candidate=repaired
                    )
                    repaired_severe = [
                        d
                        for d in repaired_diagnostics
                        if d.severity in {"repair_required", "fab_gate"}
                    ]
                    _record_attempt_facts(
                        active_client,
                        run_id=run_id,
                        stage=stage,
                        attempt=attempts,
                        call_mode="semantic_repair",
                        outcome="candidate",
                        facts=repair_facts,
                        diagnostic_codes=[d.code for d in repaired_diagnostics],
                    )
                    if _semantic_defect_score(repaired_severe) >= _semantic_defect_score(severe):
                        break
                    obj = repaired
                    diagnostics = repaired_diagnostics
                    severe = repaired_severe
                    semantic_repair_adopted = True
                    adopted_repair_facts = repair_facts
                    repair_source_raw = repair_facts.raw
                    continue

                if repair_outcome.kind == "questions":
                    qs = repair_outcome.payload["candidate"]["questions"]
                    if any(question["blocking"] for question in qs):
                        if not review_before_commit:
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
                            "rounds": rounds,
                            "tool_calls": tool_calls_ct,
                            "wall_s": round(time.monotonic() - t0, 3),
                            "cpu_s": round(_child_cpu_s() - cpu0, 3),
                            "provider_ok": provider_ok,
                            "schema_ok": schema_ok,
                            "debug_context": _debug_context(repair_facts.raw),
                        }

                repair_kind = repair_outcome.payload.get("failure_kind") or repair_outcome.kind
                _record_attempt_facts(
                    active_client,
                    run_id=run_id,
                    stage=stage,
                    attempt=attempts,
                    call_mode="semantic_repair",
                    outcome=repair_kind,
                    facts=repair_facts,
                )
                break
            except (*_TRANSPORT_FAILURE_EXC, *_PROVIDER_FAILURE_EXC) as exc:
                failure = classify_provider_exception(exc)
                _record_attempt_facts(
                    active_client,
                    run_id=run_id,
                    stage=stage,
                    attempt=attempts,
                    call_mode="semantic_repair",
                    outcome=failure["failure_kind"],
                    error_facts={k: v for k, v in failure.items() if k != "failure_kind"},
                )
                break

        diagnostic_rows = [d.model_dump(exclude_none=True) for d in diagnostics]
        if review_before_commit:
            _record_attempt_facts(
                active_client,
                run_id=run_id,
                stage=stage,
                attempt=current_attempt_number,
                call_mode=current_call_mode,
                outcome="candidate_review",
                facts=current_facts,
                diagnostic_codes=provider_diagnostic_codes,
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
                    adopted_repair_facts.raw if adopted_repair_facts is not None else raw
                ),
            }
        ok, out, obj = commit_candidate(prepared, obj, state_path, brief, workspace)
        if (
            not ok
            and out.get("failure_kind") != "commit_process_failed"
            and semantic_repair_adopted
        ):
            semantic_repair_adopted = False
            obj = original_obj
            diagnostics = diagnose_stage(
                stage, brief=brief, upstream_state=semantic_state, candidate=obj
            )
            severe = [d for d in diagnostics if d.severity in {"repair_required", "fab_gate"}]
            diagnostic_rows = [d.model_dump(exclude_none=True) for d in diagnostics]
            ok, out, obj = commit_candidate(prepared, obj, state_path, brief, workspace)
        if not ok and out.get("failure_kind") == "commit_process_failed":
            _record_attempt_facts(
                active_client,
                run_id=run_id,
                stage=stage,
                attempt=current_attempt_number,
                call_mode=current_call_mode,
                outcome="commit_process_failed",
                facts=current_facts,
                commit_result=out,
                candidate_retained=True,
                fallback_reason=("provider_rate_limited" if provider_fallback_active else None),
            )
            last = {
                "failure_kind": "commit_process_failed",
                "error": (out.get("errors") or ["stage-commit process failed"])[0],
                "commit": out,
                "provider_ok": provider_ok,
                "schema_ok": schema_ok,
                "diagnostics": diagnostic_rows,
            }
            break
        _record_attempt_facts(
            active_client,
            run_id=run_id,
            stage=stage,
            attempt=current_attempt_number,
            call_mode=current_call_mode,
            outcome="candidate" if ok else "commit_rejected",
            facts=current_facts,
            diagnostic_codes=provider_diagnostic_codes,
            commit_result=out if not ok else None,
            candidate_retained=True,
            fallback_reason=("provider_rate_limited" if provider_fallback_active else None),
        )
        if ok:
            _observe_attempt(
                attempt_observer,
                client=active_client,
                stage=stage,
                provider_attempt=current_attempt_number,
                call_mode=current_call_mode,
                outcome="committed",
                facts=current_facts,
                candidate=obj,
                commit_result=out,
                clean_slate_used=was_clean_slate,
                escalated=escalation_done,
                provider_fallback=provider_fallback_active,
            )
            fab_safe = not any(d.severity == "fab_gate" for d in diagnostics)
            return finalize_stage(
                active_client,
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
                    "model": _client_model(active_client),
                }
            )
        signature, clean_slate_next, terminal = next_attempt(
            out,
            prior_rejection_signature,
            was_clean_slate=was_clean_slate,
            clean_slate_spent=clean_slate_spent,
            clean_slate_armed_signature=clean_slate_armed_signature,
        )
        _observe_attempt(
            attempt_observer,
            client=active_client,
            stage=stage,
            provider_attempt=current_attempt_number,
            call_mode=current_call_mode,
            outcome="commit_rejected",
            facts=current_facts,
            candidate=obj,
            commit_result=out,
            rejection_signature=signature,
            clean_slate_armed=clean_slate_next,
            clean_slate_used=was_clean_slate,
            escalated=escalation_done,
            provider_fallback=provider_fallback_active,
        )
        if terminal:
            break
        prior_rejection_signature = signature
        if clean_slate_next:
            clean_slate_spent = True
            clean_slate_armed_signature = signature
            escalation_pending = stage == "wiring" and attempts + 1 >= 3 and not escalation_done
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
        active_client,
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
