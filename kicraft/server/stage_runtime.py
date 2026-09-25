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
    DETECTOR_VERSION,
    EXTERNAL_LOAD_CURRENT_CODE,
    complete_intent_classification,
    complete_unstated_power_input,
    complete_unsourced_external_rails,
    complete_usb_socket_rail,
    diagnose_stage,
    external_load_budget_stated,
    normalize_project_stem,
    remove_board_feature_blocks,
    remove_mislabeled_architecture_defaults,
    remove_mislabeled_functional_defaults,
)
from .config import (
    CONTRACT_LADDER_MODES,
    STAGE_COLLECTION_BOUNDS,
    STAGE_SERIALIZATION_MAX_TOKENS,
    CollectionBound,
    StageResponsePolicy,
)
from .client import classify_provider_exception
from .question_policy import normalize_question_options
from .stage_bom_tools import BOM_TOOLS, build_bom_executor
from .stage_contracts import (
    StageResponseContract,
    StageSchemaError,
    _extract_json,
    _normalize_stage_response,
    apply_collection_bounds,
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
    unit_defect_diagnostic,
    validate_unit_candidate,
)
from .reconciliation import resolve_demanded_classes, resolve_quantity_subjects
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
    schema_error: str | None = None,
) -> None:
    usage = (facts.usage or {}) if facts is not None else {}
    errors = error_facts or {}
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
        schema_error=schema_error,
        failure_detail=_redacted_schema_error(errors.get("message")),
        response_chars=len(facts.raw) if facts is not None else None,
        response_format_mode=(
            facts.response_format_mode if facts is not None else errors.get("response_format_mode")
        ),
        diagnostic_codes=diagnostic_codes,
        unit_id=unit_id,
        unit_attempt=unit_attempt,
        aggregate_round=aggregate_round,
        **_redacted_rejection_facts(
            commit_result,
            candidate_retained=candidate_retained,
        ),
        http_status=errors.get("http_status"),
        error_code=errors.get("error_code"),
        request_id=errors.get("request_id"),
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
    schema_error: str | None = None,
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
        "schema_error": schema_error,
        "response_chars": len(facts.raw) if facts is not None else None,
        "response_format_mode": (
            facts.response_format_mode if facts is not None else errors.get("response_format_mode")
        ),
        "failure_detail": _redacted_schema_error(errors.get("message")),
        **rejection,
    }


# Per-stage self-correction budget. Wiring must satisfy whole-board net coverage
# (§9.11) in a single slot; on a complex board the model needs more correction
# passes than the simpler, smaller-slot stages, so they floor higher (BOM must
# also resolve every symbol/footprint to a real library entry within its budget).
_STAGE_MIN_RETRIES = {"architecture": 3, "wiring": 7, "bom": 4}

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
# Architecture is comparably large (sheets, inter-sheet nets, and requirement
# rows for a multi-IC board) and truncated at the 4096 default on RP2040-class
# boards, so it floors at the same cap as BOM.
_STAGE_MIN_TOKENS = {"architecture": 16384, "wiring": 8192, "bom": 16384}


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
    if stage == "architecture":
        rejection_text = json.dumps([*(out.get("errors") or []), *(out.get("offenders") or [])])
        if "fs-connection mapping" in rejection_text:
            msg += (
                " ARCHITECTURE FIX: add every missing cross-sheet signal to "
                "`inter_sheet_nets`. Each entry needs a concise net name and at least "
                "two endpoints using the exact canonical `sheets[].name` values. Bind "
                "that exact net name as a `requirements[].ports` value on each owning "
                "sheet; direction words are not net bindings. Do not remove required "
                "signals or endpoints to bypass the binding check."
            )
        if "block-sheet mapping" in rejection_text:
            msg += (
                " ARCHITECTURE FIX: preserve the physical sheet hierarchy and emit "
                "nonempty implementation requirements whose `functional_blocks` use "
                "the exact committed Functional Spec block names. Every block needs "
                "an explicit requirement owner, including blocks sharing one sheet."
            )
    if stage == "bom":
        rejection_text = json.dumps([*(out.get("errors") or []), *(out.get("offenders") or [])])
        if "spec-named part accountability" in rejection_text:
            msg += (
                " BOM FIX: for every named-part offender, either use that exact part or add "
                "one `substitutions` entry whose `wanted` is the exact offender token and "
                "whose `got` names the shipped replacement. Do not leave any offender "
                "implicit in assumptions."
            )
        if (
            "do not resolve to a real .kicad_mod" in rejection_text
            or "do not resolve to a pin inventory" in rejection_text
        ):
            msg += (
                " BOM FIX: replace every unresolved symbol/footprint pair with an exact pair "
                "from the available-parts table or BOM lookup tools. Never invent a library "
                "prefix or package name."
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


_COMMIT_GATE_PATTERNS = (
    ("footprint(s) do not resolve", "bom_footprint_unresolved"),
    ("symbol(s) do not resolve", "bom_symbol_unresolved"),
    ("symbol pin numbers do not match", "9.27"),
    ("bom part(s) not orderable", "9.26"),
    ("multi-element arrays", "9.28"),
    ("net coverage", "9.11"),
    ("dangling signal", "9.15"),
    ("inter-sheet", "architecture_inter_sheet_contract"),
    ("slot validation failed", "stage_schema"),
    ("library validation failed", "architecture_library"),
)


def _stable_commit_gate_codes(gate_ids: tuple[str, ...]) -> list[str]:
    stable: list[str] = []
    for gate in gate_ids:
        text = str(gate)
        match = re.fullmatch(r"9\.\d+", text)
        if match:
            code = match.group(0)
        else:
            lowered = text.lower()
            code = next(
                (candidate for fragment, candidate in _COMMIT_GATE_PATTERNS if fragment in lowered),
                None,
            )
            if code is None:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
                code = f"unclassified_{digest}"
        if code not in stable:
            stable.append(code)
    return stable


# --------------------------------------------------------------------------- #
# Correction-ladder arms (docs/plans/architecture-contract-correction-ladder.md)
# --------------------------------------------------------------------------- #
def _contract_ladder_modes(client) -> frozenset[str]:
    """The correction-ladder arms this drive runs (default: ``stock``)."""
    raw = str(getattr(getattr(client, "s", None), "contract_ladder", "stock") or "stock")
    modes = frozenset(part.strip().lower() for part in raw.split(",") if part.strip())
    return (modes & CONTRACT_LADDER_MODES) or frozenset({"stock"})


def _declared_identities(payload: dict) -> set[str]:
    """The declared content a revision must preserve (O5).

    ``inter_sheet_nets[*].name`` plus every non-empty ``requirements[*].ports``
    value, each tagged by kind so a dropped net declaration is distinguishable
    from a dropped port binding of the same name.
    """
    identities: set[str] = set()
    for net in payload.get("inter_sheet_nets") or []:
        if isinstance(net, dict) and net.get("name"):
            identities.add(f"net:{net['name']}")
    for requirement in payload.get("requirements") or []:
        if not isinstance(requirement, dict):
            continue
        for value in (requirement.get("ports") or {}).values():
            if value:
                identities.add(f"port:{value}")
    return identities


def _identity_name(identity: str) -> str:
    return identity.split(":", 1)[1]


def _diagnostic_member_rows(row: dict | None) -> list[dict]:
    """One diagnostic's own row plus every member row it nests, outer first.

    An aggregate (``multiple_intent_contracts``) carries its members as typed `findings`; the
    artifacts written before that field existed carry the same rows in `evidence` as bare dicts.
    Both shapes are read here, so the signature, the counting and the correction feedback keep
    seeing the leaf codes whichever writer produced the row.
    """
    if not isinstance(row, dict):
        return []
    members = [row]
    for source in (row.get("findings") or [], row.get("evidence") or []):
        members.extend(
            item for item in source if isinstance(item, dict) and item.get("code")
        )
    return members


def _diagnostic_codes(row: dict | None) -> list[str]:
    """Every blocking code one rejection named, directly or in its member rows."""
    return [str(member["code"]) for member in _diagnostic_member_rows(row)]


def stage_telemetry(
    *,
    ok: bool,
    attempts: int,
    defect_codes: tuple[str, ...],
    outcome: dict,
    contract_rejections: int | None = None,
    semantic_repair_rounds: int | None = None,
) -> dict:
    """The acceptance counters published on ``stage_done``.

    See `docs/plans/architecture-constructive-slot-2026-09-14.md` §6: the primary
    endpoint is the share of runs accepted with ZERO corrections, so `drafts` is
    the number of provider calls that drafted a slot and `first_draft_accepted`
    is true only when the first one already committed. `defect_codes` counts the
    blocking classes that draft was rejected for, and the two architecture
    counters keep an unsupported part (`unknown_part_refused`) and a declared
    interface (`declared_interfaces`) distinguishable from instability.

    `first_draft_accepted` counts a semantic repair round as a correction, and
    `docs/plans/architecture-slot-next-steps-2026-09-14.md` §3 splits that: the
    caller that knows the two correction kinds passes `contract_rejections`
    (reader refusals) and `semantic_repair_rounds` (repair calls the driver
    spent), and the endpoint then also reports `first_draft_contract_clean` —
    the first draft reached the commit gates without a reader refusal, whether
    or not a semantic round followed. A caller that does not know the kinds
    (the work-unit stages) omits both and publishes no such counter, so its
    `stage_done` keeps its shape instead of reporting a false zero.
    """
    codes = sorted({str(code) for code in defect_codes if code})
    slot = outcome.get("slot") if isinstance(outcome, dict) else None
    declared = slot.get("declared_interfaces") if isinstance(slot, dict) else None
    telemetry = {
        "drafts": attempts,
        "first_draft_accepted": bool(ok and attempts == 1),
        "defect_codes": codes,
        "declared_interfaces": len(declared or []),
        "unknown_part_refused": codes.count("unknown_part_refused"),
    }
    if contract_rejections is not None:
        telemetry["contract_rejections"] = contract_rejections
        telemetry["first_draft_contract_clean"] = bool(
            ok and contract_rejections == 0 and attempts >= 1
        )
    if semantic_repair_rounds is not None:
        telemetry["semantic_repair_rounds"] = semantic_repair_rounds
    return telemetry


def _raw_declared_identities(raw: str) -> set[str] | None:
    """Declared identities of ONE raw reply, or None when it is not one object."""
    if not raw:
        return None
    try:
        payload = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return _declared_identities(payload) if isinstance(payload, dict) else None


def _schema_rejection_signature(
    failure_kind: str | None,
    diagnostic: dict | None,
) -> tuple | None:
    """Stable identity of one schema/contract rejection (the O3 comparison).

    The schema path carries no signature today: ``contract_rejected``'s error
    string is a constant whatever the defect, so the identity is built from the
    blocking diagnostic codes (the outer code plus any the evidence carries) and
    the names the diagnostic quotes — the same shape as
    ``_commit_rejection_signature`` (ordered codes + offender identity), never a
    second stall detector. Returns None when the rejection carries no diagnostic:
    a parse failure has no identity to compare, so it stays terminal exactly as
    before.
    """
    row = diagnostic if isinstance(diagnostic, dict) else {}
    if not row.get("code"):
        return None
    members = _diagnostic_member_rows(row)
    codes: set[str] = {str(member["code"]) for member in members}
    texts: list[str] = [str(row.get("message") or "")]
    for member in members:
        if member is not row:
            if member.get("message"):
                texts.append(str(member["message"]))
            texts.extend(str(extra) for extra in member.get("evidence") or [])
    for item in row.get("evidence") or []:
        if not isinstance(item, dict):
            texts.append(str(item))
    names = set(re.findall(r"'([^']{1,64})'", " ".join(texts)))
    return (str(failure_kind or ""), tuple(sorted(codes)), tuple(sorted(names)))


def _rejection_text(*parts: object) -> str:
    """Everything a rejection told the model, for the O5 suppression check."""
    return " ".join(
        json.dumps(part, separators=(",", ":")) if isinstance(part, (dict, list)) else str(part)
        for part in parts
        if part is not None
    )


def _redacted_schema_error(detail: object) -> str | None:
    if not detail:
        return None
    lines = []
    for raw_line in str(detail).splitlines():
        line = re.sub(r"input_value=.*", "input_value=<redacted>", raw_line).strip()
        if line:
            lines.append(line)
        if len(lines) >= 8:
            break
    return "\n".join(lines)[:1200] or None


def _redacted_rejection_facts(
    commit_result: dict | None,
    *,
    candidate_retained: bool | None,
) -> dict:
    """Ledger-safe deterministic rejection attribution."""
    if not isinstance(commit_result, dict):
        return {"candidate_retained": candidate_retained}
    gate_ids, offenders = _commit_rejection_signature(commit_result)
    stable_gates = _stable_commit_gate_codes(gate_ids)
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


def _commit_rejection_diagnostics(commit_result: object) -> list[dict]:
    """A rejected deterministic commit's own reasons, as diagnostics rows.

    A commit gate names its offenders on the transient ``retry`` event and in the ledger
    signature, but the stage status recorded only the *semantic* diagnostics. A BOM whose
    commit was refused for six unproven realization contracts plus a fabrication gate
    therefore reported the fabrication row alone, and the six reasons a reader needs had to
    be recovered from the event stream — the same shape as a contract refusal's empty
    ``diagnostics``. What the stage refused the candidate for is durable evidence: record
    it, with the gate codes, and let the status name every reason.
    """
    if not isinstance(commit_result, dict):
        return []
    if commit_result.get("failure_kind") == "commit_process_failed":
        return []
    gate_codes, _offenders = _commit_rejection_signature(commit_result)
    errors = [re.sub(r"\s+", " ", str(item)).strip() for item in (commit_result.get("errors") or [])]
    offenders = [str(item)[:600] for item in (commit_result.get("offenders") or [])]
    if not errors and not offenders:
        return []
    return [
        {
            "code": "commit_gate_rejected",
            "severity": "fab_gate",
            "message": errors[0]
            if errors
            else "the deterministic commit gate refused the candidate",
            "evidence": [*errors[1:8], *offenders[:24]],
            "gate_codes": list(gate_codes),
            # Required by StageDiagnostic: a row without it makes the saved state
            # unloadable, so `replay` cannot reopen the run it was written for.
            "detector_version": DETECTOR_VERSION,
        }
    ]


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


_AUTO_DEFAULT_QUESTION_STAGES = frozenset({"intent", "functional_spec", "architecture", "bom"})

# The external 5 V load current is a physical fact only the brief or the user can
# supply. The first option is deliberately an action, not an invented rating.
EXTERNAL_LOAD_CURRENT_QUESTION = {
    "text": (
        "How much current must the board supply to the external 5 V loads "
        "(the display and the LED string, for example)?"
    ),
    "blocking": True,
    "options": [
        "Determine the load current from the load datasheets first",
        "Up to 1 A",
        "Up to 2 A",
        "Another current requirement (enter amperes)",
    ],
}


def _normalize_questions(raw_list, stage: str) -> list[dict]:
    """Return bounded, actionable questions safe to expose to a user.

    Ordinary questions have already passed the stage response contract. Preserve
    their option order—the first option is the provider's recommendation—while
    retaining the state bounds and only allowing the internal BOM reconcile route.
    """
    out = []
    for q in raw_list:
        if not isinstance(q, dict):
            continue
        text = str(q.get("text", "")).strip()
        if not text:
            continue
        target = q.get("reconcile_target")
        reconcile_target = target if target in ("bom",) else None
        blocking = bool(q.get("blocking", False))
        if reconcile_target is None:
            try:
                options = normalize_question_options(q.get("options"))
            except ValueError:
                options = []
                blocking = False
        else:
            options = []
            for raw_option in q.get("options") or []:
                option = str(raw_option).strip()[:200]
                if option and option not in options:
                    options.append(option)
                if len(options) == 4:
                    break
        out.append(
            {
                "text": text[:500],
                "stage": stage,
                "blocking": blocking,
                "material": bool(q.get("material", True)),
                "options": options,
                "answer": None,
                "reconcile_target": reconcile_target,
            }
        )
    return out[:3]


def _auto_default_questions_enabled(
    stage: str,
    *,
    review_before_commit: bool,
    auto_default_questions: bool | None,
) -> bool:
    """Whether the driver may resolve an ordinary question without parking."""
    if auto_default_questions is None:
        auto_default_questions = not review_before_commit
        return bool(auto_default_questions) and stage in _AUTO_DEFAULT_QUESTION_STAGES
    return auto_default_questions


def _questions_need_input(
    questions: list[dict],
    stage: str,
    *,
    auto_default: bool,
    answers,
    instruction,
    auto_default_questions: bool | None = None,
) -> bool:
    """Whether blocking questions pause this stage under the chosen policy."""
    if not any(question["blocking"] for question in questions):
        return False
    # BOM reconciliation is pipeline control, never an ordinary user decision.
    if any(question.get("reconcile_target") for question in questions):
        return True
    if auto_default_questions is True:
        return False
    if auto_default_questions is False:
        return True
    # Preserve legacy callers' one-round noninteractive behavior.
    return not auto_default and not answers and not instruction


def _client_model(client) -> str | None:
    """Best-effort display name of the model a client will call (shown in the UI)."""
    return getattr(getattr(client, "s", None), "model", None)


def _requirement_identity(requirement: dict) -> str:
    """Requirement id plus the identity a group must carry to satisfy it.

    The bare id ("relay_driver") does not tell a repair turn what part to emit;
    role / family / exact_part do.
    """
    from kicraft.design.part_identity import accepted_part_identities, reviewed_parts_for_feature

    parts = [str(requirement.get("id"))]
    for field in ("role", "family", "exact_part"):
        value = requirement.get(field)
        if value:
            parts.append(f"{field}={value}")
    identity = str(requirement.get("exact_part") or requirement.get("family") or "")
    accepted = accepted_part_identities(identity)
    if accepted:
        parts.append(f"accepted_concrete_parts={list(accepted)!r}")
    physical_candidates = reviewed_parts_for_feature(identity)
    if physical_candidates:
        parts.append(
            "reviewed_physical_candidates="
            + json.dumps(
                [
                    {"mpn": part.identity, "symbol": part.symbol, "footprint": part.footprint}
                    for part in physical_candidates
                ],
                separators=(",", ":"),
            )
        )
    for field in ("obligations", "declared_interface", "ports"):
        if requirement.get(field):
            parts.append(f"{field}={json.dumps(requirement[field], separators=(',', ':'))}")
    return " ".join(parts)


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
    "contract_rejected": (
        "the reply was schema-valid but a deterministic design contract refused it "
        "(see the diagnostic)"
    ),
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

# Both kinds arrive through StageSchemaError and take the same bounded
# correction path. They differ in CAUSE: ``invalid_schema`` is unusable provider
# output, while ``contract_rejected`` is a schema-clean candidate that a
# semantic/recipe contract (an attached `diagnostic`) refused. Callers that gate
# on "this was a schema-path failure" must use this set, never the single kind —
# the label is for investigators, not for routing.
_SCHEMA_REJECTION_KINDS = frozenset({"invalid_schema", "contract_rejected"})


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
    "and nothing was committed. {bounds_sentence}{collection_hint}Do NOT call any "
    "tools. Start again from the project state and emit ONE compact slot JSON "
    "within those canonical limits. Do not continue or salvage the stopped draft."
)

_DUPLICATE_IDENTITY_RETRY_MSG = (
    "Your previous reply was stopped because `{field}` repeats identity values "
    "{duplicate_values} for unique keys {unique_keys}; "
    "{emitted_content_chars} content characters were emitted and nothing was "
    "committed. This is a duplicate-identity error, not a collection-size error. "
    "{bounds_sentence}{collection_hint}Emit exactly one entry per logical identity. "
    "Do NOT call tools. Start again from the binding project state and emit ONE "
    "complete compact slot JSON. Do not continue or salvage the stopped draft."
)

_PROPERTY_LIMIT_RETRY_MSG = (
    "Your previous reply was stopped at object `{field}` because property "
    "`{property}` is forbidden by the response schema; {emitted_content_chars} "
    "content characters were emitted and nothing was committed. "
    "{bounds_sentence}Use only the declared fields at that object. Do NOT call "
    "tools. Start again from the binding project state and emit ONE complete "
    "compact slot JSON. Do not continue or salvage the stopped draft."
)

_SYNTAX_LIMIT_RETRY_MSG = (
    "Your previous reply was stopped because its JSON syntax became invalid at "
    "character offset {character_offset} (zero-based), line {line}, column {column}, "
    "at `{field}`: {syntax_error}. {emitted_content_chars} content characters were "
    "emitted and nothing was committed. {bounds_sentence}Do NOT call tools. "
    "Start again from the binding project state and emit ONE complete compact "
    "JSON object with unique object keys and correct delimiters and escapes. "
    "Do not continue or salvage the stopped draft."
)


def _audit_notes_sentence(audit_notes) -> str:
    """One sentence naming what the Jev audit of the same draft reported.

    The notes are left out of the rejection's identity (``_schema_rejection_signature`` reads the
    diagnostic, not this message), so an audit note that changes wording can never be mistaken
    for progress on the defect itself.
    """
    rows = [note for note in (audit_notes or []) if isinstance(note, dict) and note.get("code")]
    if not rows:
        return ""
    described = "; ".join(
        f"{row['code']}: {str(row.get('message') or '').strip()}"
        + (f" [{str((row.get('evidence') or [''])[0])[:120]}]" if row.get("evidence") else "")
        for row in rows[:4]
    )
    return (
        "\nA Jev audit of this same draft, taken from the draft alone before the compiler "
        f"derived it, additionally reports: {described}. Fix these in the same correction."
    )


def _stage_recovery_message(
    kind: str | None,
    raw: str,
    bounds_sentence: str,
    *,
    schema_error: str | None = None,
    diagnostic: dict | None = None,
    collection_limit: dict | None = None,
    audit_notes=None,
) -> str:
    limit = collection_limit or {}
    if kind == "collection_limit":
        if limit.get("limit_scope") == "syntax":
            template = _SYNTAX_LIMIT_RETRY_MSG
        elif limit.get("limit_scope") == "property":
            template = _PROPERTY_LIMIT_RETRY_MSG
        elif limit.get("limit_scope") == "duplicate":
            template = _DUPLICATE_IDENTITY_RETRY_MSG
        else:
            template = _COLLECTION_LIMIT_RETRY_MSG
    elif kind in _SCHEMA_REJECTION_KINDS:
        template = _SCHEMA_RETRY_MSG
    else:
        template = _SERIALIZATION_RETRY_MSG
    collection_hint = ""
    if limit.get("field") == "inter_sheet_nets" and limit.get("limit_scope") == "duplicate":
        collection_hint = (
            "A net shared by several sheets is ONE net record containing all of "
            "those sheet endpoints, not a separate record for each pair of sheets. "
            "Combine its participants without dropping endpoints or renaming the net. "
        )
    elif (
        limit.get("field") == "connections"
        and limit.get("limit_scope") == "duplicate"
        and limit.get("unique_keys") == ["from_block", "to_block", "signal_type"]
    ):
        collection_hint = (
            "These values identify ONE functional graph edge, not one physical signal. "
            "Combine all same-type signals between this directed block pair in that "
            "edge's description. Keep other signal types, other consumers and real "
            "reverse flows as separate edges. Individual pin/net connections belong "
            "in architecture and wiring, not repeated functional graph edges. "
        )
    message = template.format(
        prior_chars=len(raw),
        bounds_sentence=(bounds_sentence + " ") if bounds_sentence else "",
        field=limit.get("field", "unknown"),
        property=limit.get("property", "unknown"),
        observed_count=limit.get("observed_count", "unknown"),
        configured_total=limit.get("configured_total", "unknown"),
        limit_scope=limit.get("limit_scope", "total"),
        emitted_content_chars=limit.get("emitted_content_chars", len(raw)),
        collection_hint=collection_hint,
        duplicate_values=(
            json.dumps(limit.get("duplicate_values") or [])
            if template is _DUPLICATE_IDENTITY_RETRY_MSG
            else ""
        ),
        unique_keys=(
            json.dumps(limit.get("unique_keys") or [])
            if template is _DUPLICATE_IDENTITY_RETRY_MSG
            else ""
        ),
        schema_error=(schema_error or "unspecified schema violation")[:1200],
        syntax_error=str(limit.get("syntax_error", "invalid JSON syntax"))[:1200],
        character_offset=limit.get("character_offset", "unknown"),
        line=limit.get("line", "unknown"),
        column=limit.get("column", "unknown"),
    )
    if diagnostic:
        message += "\nConcrete diagnostic:\n" + json.dumps(diagnostic, separators=(",", ":"))
    return message + _audit_notes_sentence(audit_notes)


_SEMANTIC_REPAIR_MSG = (
    "The candidate is schema-valid but deterministic semantic checks found the "
    "following high-confidence defects: {diagnostics}. Preserve all valid content, "
    "correct only these defects, add no new assumptions, and return one complete "
    "JSON object matching the same schema. No tools, markdown, or prose."
)
_MAX_SEMANTIC_REPAIR_ROUNDS = 1

#: Preserving corrections a *design-contract* refusal earns by default (no ladder flag needed).
#: A schema-clean draft that a design contract refused is a design defect, and one draft can carry
#: several independent ones: the seed-37 walkthrough's four architecture drafts each fixed the
#: reported defect and surfaced another (USB rail, unwired USB pins, unnumbered terminal contacts,
#: a prose string where a part identity belongs). Stock behaviour answered them with a single
#: from-scratch rewrite and then died; with this bound the loop keeps the draft while the defect
#: set keeps changing, for two rounds. An unchanged defect set still terminates immediately.
_MAX_DESIGN_CONTRACT_REPAIR_ROUNDS = 2


def _audit_parsed_draft(pre_audit, parsed) -> list[dict]:
    """Audit a parsed draft if a hook is configured; findings as plain rows, never shared state."""
    if pre_audit is None or not isinstance(parsed, dict):
        return []
    try:
        findings = pre_audit(parsed)
    except Exception:
        return []
    return [
        finding.model_dump(exclude_none=True)
        for finding in findings or []
        if hasattr(finding, "model_dump")
    ]


def _pre_audit_hook(client):
    """The pre-derivation audit hook for the stages that have one (architecture)."""

    def audit(parsed: dict):
        settings = getattr(client, "s", None)
        if not getattr(settings, "enable_draft_audit", False):
            return []
        return _audit_architecture_draft(client, parsed)

    return audit


def _audit_architecture_draft(client, candidate: dict) -> list[models.StageDiagnostic]:
    """Jev's typed findings about one architecture draft, billed and never blocking.

    The audit is one decision call over the draft (input-only pricing, a fraction of a cent at
    this size). It is optional by setting, absent by default on a client without settings, and
    silent on every failure: a stage must not depend on an auditor being reachable.
    """
    settings = getattr(client, "s", None)
    if not getattr(settings, "enable_draft_audit", False):
        return []
    from . import draft_audit

    def bill(usage) -> None:
        guard = getattr(client, "guard", None)
        record = getattr(guard, "record", None)
        if not callable(record):
            return
        try:
            record(
                str(getattr(settings, "draft_audit_model", draft_audit.DEFAULT_MODEL)),
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                float(usage.get("cost") or 0.0),
                meta={"stage": "architecture", "phase": "draft_audit"},
            )
        except Exception:  # accounting must never break a stage
            pass

    try:
        return draft_audit.audit_architecture(
            candidate,
            model=str(getattr(settings, "draft_audit_model", draft_audit.DEFAULT_MODEL)),
            confidence=float(getattr(settings, "draft_audit_confidence", 0.7)),
            recorder=bill,
        )
    except Exception:
        return []


def _name_reconciliation_decider(client, *, stage: str):
    """The typed-decision decider that reconciles naming differences, or None.

    One decider per drive: each batch of naming questions is a single typed decision
    (input-only pricing, a fraction of a cent at this size), billed through the same
    guard as every other provider call. Off by setting, and silent on every failure:
    a candidate must never depend on a decider being reachable, because every
    question it asks has a deterministic fallback (the stage's own refusal).
    """
    settings = getattr(client, "s", None)
    if not getattr(settings, "enable_name_reconciliation", False):
        return None
    from . import draft_audit
    from .reconciliation import jev_decider

    model = str(getattr(settings, "draft_audit_model", draft_audit.DEFAULT_MODEL))

    def bill(usage) -> None:
        guard = getattr(client, "guard", None)
        record = getattr(guard, "record", None)
        if not callable(record):
            return
        try:
            record(
                model,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                float(usage.get("cost") or 0.0),
                meta={"stage": stage, "phase": "name_reconciliation"},
            )
        except Exception:  # accounting must never break a stage
            pass

    return jev_decider(model=model, recorder=bill)


def _semantic_repair_message(stage: str, diagnostics: list[models.StageDiagnostic]) -> str:
    message = _SEMANTIC_REPAIR_MSG.format(
        diagnostics=json.dumps(
            [{"code": d.code, "message": d.message, "evidence": d.evidence} for d in diagnostics],
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
    if any(d.code == "intent_obligation_class_unrealizable" for d in diagnostics):
        message += (
            " Fix each flagged physical obligation by its evidence: when the evidence "
            "names a reviewed class, use that spelling; when it says 'not a part class', "
            "move the fact out of `obligations` into `constraints`, or record it as a "
            "`fabrication`/`negative` row. NEVER rename a demand to an unrelated reviewed "
            "class. A part category the reviewed library does not cover yet (a GPS module, "
            "a new sensor class) is legitimate — keep the user's own class name and let the "
            "parts stage resolve the real part."
        )
    if any(d.code == "intent_quantity_subject_unbound" for d in diagnostics):
        message += (
            " Fix each flagged `quantity` row: it counts a part class that this slot does not "
            "record, so add that class's obligation row (`kind` `physical`, `component_class` "
            "spelled as the evidence shows) or delete the count if it described a property of "
            "one part. A count of a property (pins on a header, contacts in a connector, "
            "channels of one driver) is a `quantitative` row, never a `quantity` row."
        )
    if any(d.code == "intent_prototyping_area_omitted" for d in diagnostics):
        message += (
            " The brief asks for a prototyping area (a pad field the user solders into). "
            "Record it as an obligation row with kind `fabrication` and feature "
            "`prototyping-area` — no size, no unit, and no component class: it is a "
            "property of the board itself, so it owns no requirement. Keep the brief's "
            "own wording in constraints too."
        )
    if any(d.code == "functional_spec_external_load_power_assumed" for d in diagnostics):
        message += (
            " Do not guess whether the board powers the external display or LED "
            "load. Return one blocking question with exactly these options: "
            '["Board supplies the external loads", '
            '"External loads use a separate power supply"].'
        )
    if any(d.code == "architecture_external_load_current_unspecified" for d in diagnostics):
        message += (
            " Do not guess the external-load current. Return one blocking question "
            "asking for the maximum total 5V output current, with concrete choices "
            'such as ["1 A", "2 A", "3 A", "5 A"].'
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
            "the PD contract or the load rail, return one blocking question with "
            'exactly these options: ["5 V is the external load rail", '
            '"5 V is only the USB-PD input contract"].'
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


def _normalize_candidate_for_diagnostics(
    stage: str, candidate: dict, brief: str, semantic_state: dict, *, decider=None
) -> dict:
    """Apply the stage's deterministic pre-diagnosis normalization to one candidate.

    The first candidate and a semantic-repair candidate MUST pass through exactly
    these helpers before ``diagnose_stage``. While only the first candidate was
    normalized, a repair was scored against diagnostics the original never faced:
    an intent repair kept its raw ``named_parts`` while
    ``complete_intent_classification`` had already added the exact brief token to
    the original, so the adoption guard saw equal defect scores and discarded a
    repair that really did remove the flagged defect (a physical obligation whose
    class the reviewed library spells differently). Architecture still runs
    ``complete_unsourced_external_rails`` separately, after its first diagnose.
    """
    if stage == "intent":
        candidate = complete_intent_classification(brief, candidate)
        candidate = complete_unstated_power_input(brief, candidate)
        candidate["project_stem"] = normalize_project_stem(candidate.get("project_stem", ""))
        if decider is not None:
            # An intent class the reviewed library cannot read and a count whose subject names
            # no class are naming questions with a closed answer set (the reviewed classes; the
            # classes this candidate already demands), so one typed decision resolves them before
            # diagnosis: the refusal round stays for the residue the decider declines.
            candidate, _ = resolve_demanded_classes(candidate, decider=decider)
            candidate, _ = resolve_quantity_subjects(candidate, decider=decider)
    elif stage == "functional_spec":
        # A board-feature block cannot be wired (it owns no net), so it is removed
        # deterministically here: the pad field survives as the intent's fabrication
        # row, and the stage still reports the block through its diagnostic.
        candidate = remove_board_feature_blocks(
            remove_mislabeled_functional_defaults(brief, semantic_state, candidate)
        )
    elif stage == "architecture":
        candidate = complete_usb_socket_rail(candidate)
        candidate = remove_mislabeled_architecture_defaults(semantic_state, candidate)
    return candidate


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
    response_format_mode: str | None = None


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
        response_format_mode=("json_schema" if response_format is not None else "none"),
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
        response_format_mode=("json_schema" if response_format is not None else "none"),
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
    *,
    pre_audit=None,
) -> AttemptOutcome:
    """Parse and normalize one response without mutating durable state.

    ``pre_audit`` sees the *parsed* draft before ``_normalize_stage_response`` derives it — which
    is where the compiler's contracts run, so a draft they refuse still gets audited and the
    audit's findings travel with the refusal instead of being lost with it.
    """
    audit_findings: list[dict] = []
    try:
        if facts.finish == "collection_limit":
            raise ValueError("stream collection limit")
        parsed = _extract_json(facts.raw)
        asked = parsed.get("questions")
        if isinstance(asked, list) and asked:
            return AttemptOutcome(
                "questions",
                {
                    "candidate": {"questions": _normalize_questions(asked, prepared.stage)},
                    "expanded_component_count": 0,
                },
            )
        audit_findings = _audit_parsed_draft(pre_audit, parsed)
        candidate, expanded = _normalize_stage_response(
            prepared.stage,
            parsed,
            prepared.prompt_state,
        )
        kind = "questions" if isinstance(candidate.get("questions"), list) else "candidate"
        payload = {
            "candidate": candidate,
            "expanded_component_count": expanded,
        }
        if audit_findings:
            payload["audit_findings"] = audit_findings
        return AttemptOutcome(kind, payload)
    except StageSchemaError as exc:
        diagnostic = getattr(exc, "diagnostic", None)
        # A StageSchemaError that carries a diagnostic is a semantic/recipe
        # contract refusing a schema-clean candidate, not malformed provider
        # output. Label it separately so the failure_kind, the ledger and the
        # investigation all name the cause instead of "this was JSON".
        return AttemptOutcome(
            "recoverable_failure",
            {
                "failure_kind": "contract_rejected" if diagnostic else "invalid_schema",
                "schema_error": str(exc),
                "diagnostic": diagnostic,
                "audit_findings": audit_findings,
            },
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
    defect_codes: tuple[str, ...] = (),
    contract_rejections: int | None = None,
    semantic_repair_rounds: int | None = None,
) -> dict:
    """Persist status and ledger once, then build the caller-visible result."""
    wall_s = round(time.monotonic() - t0, 3)
    cpu_s = round(_child_cpu_s() - cpu0, 3)
    # A refusal raised by a deterministic contract is not a draft defect, so it
    # carries a bare `diagnostic` rather than a `StageDiagnostic` row. Surface it
    # as this stage's one diagnostics row: the status, the event stream and every
    # later investigation read `diagnostics`, and an empty list there is why a
    # refusal's reason had to be recovered by hand (a replayed contract refusal
    # printed "see the diagnostic" and nothing else).
    diagnostics = list(outcome.get("diagnostics") or [])
    if not diagnostics:
        refusal = outcome.get("diagnostic")
        if isinstance(refusal, dict) and refusal:
            diagnostics = [refusal]
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
        diagnostics=diagnostics,
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
                "schema_error": outcome.get("schema_error"),
                "diagnostic": outcome.get("diagnostic"),
                "work_units": outcome.get("work_units"),
                "reused_work_units": outcome.get("reused_work_units"),
                "aggregate_repair_rounds": outcome.get("aggregate_repair_rounds"),
                "retryable": outcome.get("failure_kind") == "provider_rate_limited",
                "retry_action": (
                    "retry_stage"
                    if outcome.get("failure_kind") == "provider_rate_limited"
                    else None
                ),
                **stage_telemetry(
                    ok=ok,
                    attempts=attempts,
                    defect_codes=defect_codes,
                    outcome=outcome,
                    contract_rejections=contract_rejections,
                    semantic_repair_rounds=semantic_repair_rounds,
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
WORK_UNIT_MAX_REPAIR_ROUNDS = 3


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
        "excluded_refs": list(unit.excluded_refs),
        "excluded_pins": [{"ref": ref, "pin": pin} for ref, pin in unit.excluded_pins],
        "planned_resolution_source": unit.planned_resolution_source,
        "recipe_ids": list(unit.recipe_ids),
        "lowerer_ids": list(unit.lowerer_ids),
    }
    if unit.requirement_ids:
        boundary["scope"] = "owned_requirements"
        boundary["requirement_ids"] = list(unit.requirement_ids)
        boundary["owned_roles"] = list(unit.owned_roles)
        boundary["owned_requirements"] = [
            requirement
            for requirement in (prompt_state.get("architecture") or {}).get("requirements") or []
            if isinstance(requirement, dict) and str(requirement.get("id")) in unit.requirement_ids
        ]
    else:
        # Empty requirement_ids means the architecture did not decompose this
        # sheet, not that the model owns nothing.
        boundary["scope"] = "complete_sheet"
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
        declared = set((prompt_state.get("architecture") or {}).get("declared_interfaces") or [])
        owned_declared = [
            requirement_id for requirement_id in unit.requirement_ids if requirement_id in declared
        ]
        boundary["owned_output"] = (
            "Only components physically installed in target_sheet to implement "
            "the owned_requirements, or target_function and target_topology when "
            "scope is complete_sheet. Stage-wide programming and decoupling rules "
            "apply only to ICs owned by this unit, never to a sibling or recipe IC. "
            "Emit at least one group unless a locked circuit recipe already populates "
            "this sheet. Never recreate the whole-board BOM, another requirement's "
            "parts, read_only_exclusions, or PRIOR ACCEPTED WORK UNITS."
        )
        if owned_declared:
            # `docs/plans/architecture-constructive-slot-2026-09-14.md` §4.2: this part has
            # no curated recipe, so its pin functions are a model claim, not verified data.
            boundary["declared_interfaces"] = owned_declared
            boundary["owned_output"] += (
                " DECLARED INTERFACES: " + ", ".join(owned_declared) + " have no curated "
                "recipe — their pin functions are a claim. Source a real orderable part, and "
                "check each claimed function against it rather than assuming the assignment."
            )
    else:
        boundary["owned_refs"] = list(unit.refs)
        boundary["owned_pins"] = [{"ref": ref, "pin": pin} for ref, pin in unit.expected_pins]
    return json.dumps(boundary, separators=(",", ":"))


def _work_unit_extras(unit: StageWorkUnit, extras: dict) -> dict:
    if unit.stage == "bom":
        return {key: value for key, value in extras.items() if not str(key).startswith("_")}
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


def _work_unit_prompt_state(unit: StageWorkUnit, prompt_state: dict) -> dict:
    if unit.stage not in {"bom", "wiring"}:
        return prompt_state
    refs = set(unit.refs)
    architecture = dict(prompt_state.get("architecture") or {})
    relevant_nets = [
        net
        for net in architecture.get("inter_sheet_nets") or []
        if isinstance(net, dict)
        and any(
            isinstance(endpoint, dict) and endpoint.get("sheet") == unit.sheet
            for endpoint in net.get("endpoints") or []
        )
    ]
    relevant_sheets = {
        str(endpoint.get("sheet"))
        for net in relevant_nets
        for endpoint in net.get("endpoints") or []
        if isinstance(endpoint, dict) and endpoint.get("sheet")
    } | {unit.sheet}
    if unit.stage == "bom":
        # The sheet enum already constrains every emitted group to this sheet.
        # Showing sibling functions here causes whole-board parts to be moved
        # into that sheet rather than limiting the model to its actual owner.
        relevant_sheets = {unit.sheet}
    architecture["sheets"] = [
        sheet
        for sheet in architecture.get("sheets") or []
        if isinstance(sheet, dict) and str(sheet.get("name")) in relevant_sheets
    ]
    architecture["inter_sheet_nets"] = relevant_nets
    architecture["requirements"] = [
        requirement
        for requirement in architecture.get("requirements") or []
        if isinstance(requirement, dict)
        and requirement.get("sheet") == unit.sheet
        and (
            unit.stage != "bom"
            or not unit.requirement_ids
            or str(requirement.get("id")) in unit.requirement_ids
        )
    ]
    architecture["recipe_selections"] = [
        selection
        for selection in architecture.get("recipe_selections") or []
        if isinstance(selection, dict)
        and unit.sheet in {str(sheet) for sheet in (selection.get("sheets") or {}).values()}
    ]
    architecture["recipe_resolution"] = [
        record
        for record in architecture.get("recipe_resolution") or []
        if isinstance(record, dict)
        and str(record.get("requirement_id")) in set(unit.requirement_ids)
    ]
    architecture["unresolved_requirement_ids"] = [
        requirement_id
        for requirement_id in architecture.get("unresolved_requirement_ids") or []
        if str(requirement_id) in set(unit.requirement_ids)
    ]
    if unit.stage == "bom":
        target_tokens = {
            re.sub(r"[^a-z0-9]", "", str(value).lower())
            for sheet in architecture["sheets"]
            for value in (sheet.get("name"), sheet.get("stem"))
            if value
        }
        architecture["topologies"] = {
            key: value
            for key, value in (architecture.get("topologies") or {}).items()
            if re.sub(r"[^a-z0-9]", "", str(key).lower()) in target_tokens
        }
        upstream = prompt_state.get("architecture") or {}
        owned_ids = {str(row.get("id")) for row in architecture["requirements"]}
        exclusions = {
            "requirements": [
                {
                    key: row[key]
                    for key in ("id", "sheet", "role", "family", "exact_part")
                    if key in row
                }
                for row in upstream.get("requirements") or []
                if isinstance(row, dict) and str(row.get("id")) not in owned_ids
            ],
            "recipes": [
                {
                    key: row[key]
                    for key in ("recipe", "instance", "sheets", "requirement_ids")
                    if key in row
                }
                for row in upstream.get("recipe_selections") or []
                if isinstance(row, dict)
            ],
        }
        # Intent remains complete: explicit part choices, assembly/mechanical
        # constraints and expertise still govern sourcing. Full upstream state
        # is independently retained for validation, normalization and commit.
        return {
            "intent": prompt_state.get("intent"),
            "architecture": architecture,
            "read_only_exclusions": exclusions,
        }
    bom = dict(prompt_state.get("bom") or {})
    bom["parts"] = [
        part
        for part in bom.get("parts") or []
        if isinstance(part, dict) and str(part.get("ref")) in refs
    ]
    bom["connections"] = [
        connection
        for connection in bom.get("connections") or []
        if isinstance(connection, dict)
        and (connection.get("endpoints") or [])
        and all(
            isinstance(endpoint, dict) and str(endpoint.get("ref")) in refs
            for endpoint in connection.get("endpoints") or []
        )
    ]
    bom["no_connect_pins"] = [
        endpoint
        for endpoint in bom.get("no_connect_pins") or []
        if isinstance(endpoint, dict) and str(endpoint.get("ref")) in refs
    ]
    return {"architecture": architecture, "bom": bom}


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
    visible_state = _work_unit_prompt_state(unit, prompt_state)
    visible_accepted_summary = [] if stage == "wiring" else accepted_summary
    user = (
        f"PROJECT BRIEF:\n{brief}\n\n"
        f"CURRENT DESIGN STATE (JSON):\n{json.dumps(visible_state, separators=(',', ':'))}\n\n"
        "WORK-UNIT REFERENCE DATA (complete, not truncated):\n"
        f"{json.dumps(_work_unit_extras(unit, extras), separators=(',', ':'))}\n\n"
        "PRIOR ACCEPTED WORK UNITS (read-only):\n"
        f"{json.dumps(visible_accepted_summary, separators=(',', ':'))}"
    )
    if stage == "bom":
        user += (
            "\n\nThe project brief, global architecture facts and sourcing reference "
            "data are constraints, not a request to emit the whole board. Implement "
            "only the WORK UNIT's owned requirements (or its complete target sheet "
            "when no requirement decomposition exists). All read_only_exclusions "
            "and prior accepted groups are already owned elsewhere."
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
    decider=None,
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
    auto_default: bool,
    auto_default_questions: bool | None = None,
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
            # Explicit interactive runs may park on ordinary BOM questions. Wiring
            # always retains this affordance so it can request internal BOM
            # reconciliation even while ordinary questions auto-default.
            allow_questions=stage == "wiring" or auto_default_questions is False,
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
        deterministic = (
            deterministic_bom_candidate(unit, prompt_state)
            if stage == "bom"
            else deterministic_wiring_candidate(unit, prompt_state, extras)
        )
        if deterministic is not None:
            continue
        try:
            candidates[unit.unit_id] = validate_unit_candidate(
                unit, loaded, prompt_state, extras, decider=decider
            )
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
    # Deterministic lowerings that fail their own unit's validation: the unit is
    # re-driven by the model with the refusal as its first feedback, instead of
    # aborting the stage at attempts=0 before any provider call.
    refused_deterministic: dict[str, str] = {}
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
        deterministic["_trusted_deterministic_candidate"] = True
        try:
            candidates[unit.unit_id] = validate_unit_candidate(
                unit,
                deterministic,
                prompt_state,
                extras,
                decider=decider,
            )
        except (WorkUnitValidationError, TypeError, ValueError) as exc:
            # The lowering could not satisfy its own reviewed obligations (a real
            # refusal — see _validate_bom_unit's defect text). The unit is handed to
            # the model with that refusal as its first feedback, so the stage spends
            # its bounded unit attempts instead of failing the whole stage at
            # attempts=0. The work_unit_plan event above stays; work_unit_done is
            # emitted only by the path that actually produced the unit.
            refused_deterministic[unit.unit_id] = str(exc)
            continue
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
    model_unit_count = sum(unit.unit_id not in candidates for unit in units)
    call_budget = min(64, (repair_rounds + 3) * model_unit_count + 6) if model_unit_count else 0
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
        schema_error: str | None = None,
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
            schema_error=schema_error,
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
                    schema_error=schema_error,
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
                    "error_facts": dict(error_facts or {}) or None,
                }
            )

    def response_policy_for_unit(unit: StageWorkUnit) -> StageResponsePolicy:
        if stage == "wiring":
            return replace(
                policy,
                collection_bounds=(
                    CollectionBound(
                        field="pins",
                        total=max(1, len(unit.expected_pins)),
                    ),
                ),
            )
        if stage == "bom":
            unit_count = max(1, len(units))
            unit_bounds = []
            for bound in policy.collection_bounds:
                total = max(1, bound.total // unit_count)
                unit_bounds.append(
                    replace(
                        bound,
                        total=total,
                        per_group=(
                            min(total, bound.per_group) if bound.per_group is not None else None
                        ),
                    )
                )
            # Work-unit BOM responses are bounded fragments, not the full slot.
            # Cap output at 2K: the pro route's 24K-input call ceiling then fits
            # twice inside the default $0.10 project budget, so one invalid
            # fragment can still be repaired instead of exhausting the run.
            return replace(
                policy,
                normal_max_tokens=min(policy.normal_max_tokens, 2048),
                serialization_max_tokens=min(policy.serialization_max_tokens, 2048),
                collection_bounds=tuple(unit_bounds),
            )
        return policy

    # The provider schema and streaming collector must advertise the same cap.
    # In particular a sheet-local schema must not invite 64 groups when the
    # collector stops that unit at 21. Contracts are private to this invocation.
    for unit in units:
        apply_collection_bounds(
            contracts_by_unit[unit.unit_id].schema,
            response_policy_for_unit(unit).collection_bounds,
        )

    def invoke(
        unit: StageWorkUnit,
        *,
        unit_attempt: int,
        feedback: dict | str | None,
        pristine: bool,
        serialization: bool,
        aggregate_round: int | None,
    ) -> tuple[str, dict | None, ProviderFacts | None, str | Exception | None]:
        nonlocal attempts, total_cost, rounds_total, tool_calls_total, provider_ok, stage_client
        aggregate_signature = (
            _commit_rejection_signature(feedback["aggregate_commit_rejection"])
            if isinstance(feedback, dict)
            and isinstance(feedback.get("aggregate_commit_rejection"), dict)
            else None
        )
        unit_policy = response_policy_for_unit(unit)
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
        if stage == "bom" or unit_attempt > 1:
            profile_name = getattr(getattr(stage_client, "s", None), "escalation_profile", "")
            switch_profile = getattr(stage_client, "with_design_profile", None)
            if profile_name and callable(switch_profile):
                active_client = switch_profile(profile_name)
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
                        reasoning_guard=unit_policy.reasoning_guard,
                        progress=progress,
                        meta_ctx=ctx,
                    )
                else:
                    facts = call_stage_provider(
                        active_client,
                        prepared,
                        messages=list(messages),
                        response_format=response_format,
                        max_tokens=int(
                            unit_policy.serialization_max_tokens
                            if pristine
                            else unit_policy.normal_max_tokens
                        ),
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
                        reasoning={"enabled": False} if pristine else unit_policy.normal_reasoning,
                        reasoning_guard=unit_policy.reasoning_guard,
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
                        **{key: value for key, value in failure.items() if key != "failure_kind"},
                        "response_format_mode": (
                            "json_schema" if response_format is not None else "none"
                        ),
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
                if transient and not fallback_spent:
                    active_profile = getattr(
                        getattr(active_client, "s", None), "design_profile", None
                    )
                    if fallback_profile and fallback_profile != active_profile and callable(switch):
                        active_client = switch(fallback_profile)
                        fallback_spent = True
                        continue
                    if active_client is not stage_client:
                        active_client = stage_client
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
            if isinstance(parsed.get("questions"), list) and parsed["questions"]:
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
            if "questions" in parsed:
                # The strict envelope always carries `questions`; an empty list is
                # not part of the unit payload.
                parsed = {key: value for key, value in parsed.items() if key != "questions"}
            try:
                validated = validate_unit_candidate(
                    unit,
                    parsed,
                    prompt_state,
                    extras,
                    decider=decider,
                    # A unit whose own lowering was already refused gets its
                    # answer judged on its own merits: adopting the refused
                    # lowering could only re-raise that lowering's defect and
                    # hide the model's, so the repair feedback would repeat
                    # itself instead of naming what to fix.
                    allow_deterministic_fallback=unit.unit_id not in refused_deterministic,
                )
            except (WorkUnitValidationError, TypeError, ValueError) as exc:
                record(
                    active_client,
                    unit=unit,
                    unit_attempt=unit_attempt,
                    call_mode=call_mode,
                    outcome="invalid_work_unit",
                    facts=facts,
                    error_facts={"message": str(exc)},
                    schema_error=_redacted_schema_error(exc),
                    aggregate_round=aggregate_round,
                    aggregate_signature=aggregate_signature,
                    candidate_retained=False,
                    fallback_reason=("provider_rate_limited" if fallback_spent else None),
                )
                return "recoverable", parsed, facts, exc
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

    def failure_signature(error: str | Exception | None, facts: ProviderFacts | None) -> str:
        if isinstance(error, WorkUnitValidationError):
            defects = {}
            for name, rows in error.defects.items():
                if name in {"expected-pin-set", "rejected-assignment-set"}:
                    continue
                identities = []
                for row in rows:
                    try:
                        detail = json.loads(row)
                    except (TypeError, ValueError):
                        detail = None
                    if isinstance(detail, dict) and detail.get("rejected_identifier"):
                        # Group names, diagnostic prose and suggested alternatives
                        # can change while the same invalid library ID recurs.
                        identity = [detail.get("field"), detail["rejected_identifier"]]
                    else:
                        identity = re.sub(r"\s+", " ", str(row)).strip()
                    identities.append(json.dumps(identity, sort_keys=True))
                if identities:
                    defects[name] = sorted(set(identities))
            return json.dumps(defects, sort_keys=True)
        if error == "collection_limit" and facts is not None:
            limit = facts.collection_limit or {}
            return json.dumps(
                [
                    "collection_limit",
                    {
                        key: limit[key]
                        for key in (
                            "field",
                            "limit_scope",
                            "configured_total",
                            "group_value",
                            "property",
                            "syntax_error",
                        )
                        if key in limit
                    },
                ],
                sort_keys=True,
            )
        return re.sub(r"\s+", " ", str(error)).strip()

    seen_unit_failures: dict[str, dict[str, int]] = {}

    def draft(
        unit: StageWorkUnit,
        *,
        feedback: dict | str | None = None,
        force_pristine: bool = False,
        aggregate_round: int | None = None,
    ) -> tuple[str, dict | None]:
        seen_signatures = seen_unit_failures.setdefault(unit.unit_id, {})
        local_feedback = feedback
        failure_detail: dict = {}
        serialization_next = False
        serialization_spent = False
        pristine_next = force_pristine
        unit_attempt_stop = repair_rounds + (5 if stage == "bom" else 3)
        for unit_attempt in range(1, unit_attempt_stop):
            used_serialization = serialization_next
            used_pristine = pristine_next
            kind, payload, facts, error = invoke(
                unit,
                unit_attempt=unit_attempt,
                feedback=local_feedback,
                pristine=used_pristine,
                serialization=used_serialization,
                aggregate_round=aggregate_round,
            )
            serialization_next = False
            pristine_next = False
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
                if _questions_need_input(
                    questions,
                    stage,
                    auto_default=auto_default,
                    answers=answers,
                    instruction=instruction,
                    auto_default_questions=auto_default_questions,
                ):
                    return kind, {"questions": questions}
                if stage == "bom":
                    local_feedback = (
                        "Do not ask more questions. The committed architecture is binding: "
                        "do not change its MCU presence, control architecture, topology, "
                        "requirements, or sheet partition. This work unit owns only "
                        f"requirement_ids={list(unit.requirement_ids)!r} and "
                        f"roles={list(unit.owned_roles)!r}; do not emit components owned by "
                        "another requirement. Select one concrete suitable part, apply sensible "
                        "defaults, record each assumption ending '(defaulted)', and return only "
                        "this complete unit."
                    )
                else:
                    local_feedback = (
                        "Do not ask more questions. Apply sensible defaults, record each "
                        "assumption ending '(defaulted)', and return this unit."
                    )
                continue
            if kind == "terminal":
                return kind, {"failure_kind": error or "provider_failure"}
            signature = failure_signature(error, facts)
            error_text = str(error)
            failure_detail = {
                "error": error_text,
                "unit_id": unit.unit_id,
                "schema_error": (
                    _redacted_schema_error(error) if isinstance(error, Exception) else None
                ),
                "diagnostic": (
                    unit_defect_diagnostic(error, error_text)
                    if isinstance(error, WorkUnitValidationError)
                    else getattr(error, "diagnostic", None)
                ),
            }
            serialization_errors = {
                "collection_limit",
                "truncated_json",
                "invalid_json",
                "reasoning_loop",
            }
            if used_pristine and error in serialization_errors:
                return "terminal", {
                    "failure_kind": error,
                    "error": "clean-slate serialization retry failed",
                }
            # One identical defect may repeat: the model often fixes it on the
            # next nudge, and the loop is already bounded by the unit attempt stop
            # and the call budget. A third identical signature is terminal, and a
            # protected-identity conflict still stops after one repair attempt so
            # the model can never override deterministic ownership.
            ownership_conflict = stage == "bom" and any(
                marker in error_text
                for marker in ("recipe-duplicate", "model_authored_protected_identity")
            )
            seen_signatures[signature] = seen_signatures.get(signature, 0) + 1
            repeats = seen_signatures[signature]
            if repeats > 2 or (ownership_conflict and repeats > 1):
                return "terminal", {
                    "failure_kind": (
                        "unit_ownership_conflict" if ownership_conflict else "unit_repair_exhausted"
                    ),
                    **failure_detail,
                }
            if used_serialization and error in serialization_errors:
                bounds_sentence = _collection_bounds_sentence(
                    response_policy_for_unit(unit).collection_bounds
                )
                pristine_next = True
                local_feedback = _stage_recovery_message(
                    str(error),
                    facts.raw if facts is not None else "",
                    bounds_sentence,
                    collection_limit=facts.collection_limit if facts is not None else None,
                )
                continue
            if not serialization_spent and not used_serialization and error in serialization_errors:
                serialization_spent = True
                serialization_next = True
                bounds_sentence = _collection_bounds_sentence(
                    response_policy_for_unit(unit).collection_bounds
                )
                local_feedback = _stage_recovery_message(
                    str(error),
                    facts.raw if facts is not None else "",
                    bounds_sentence,
                    collection_limit=facts.collection_limit if facts is not None else None,
                )
                continue
            if stage == "bom":
                if unit.requirement_ids:
                    owned_scope = (
                        f"This unit owns requirement_ids={list(unit.requirement_ids)!r} and "
                        f"roles={list(unit.owned_roles)!r}. Emit the complete real component "
                        "groups needed to implement every owned requirement unless verified "
                        "recipe parts already satisfy one. Do not emit requirements outside "
                        "this list or invent a part capability."
                    )
                else:
                    owned_scope = (
                        f"This unit owns the complete target sheet {unit.sheet!r}. Emit the "
                        "complete real component groups needed to implement its committed "
                        "target_function and target_topology. Do not return an empty groups "
                        "list unless a verified recipe already populates this sheet, and do "
                        "not emit another sheet's components."
                    )
                repair_instruction = (
                    "The committed architecture is binding. Correct every listed defect "
                    "without changing MCU presence, control architecture, topology, "
                    f"requirements, or sheet partition. {owned_scope}"
                )
                if "empty-sheet" in error_text:
                    owned_target = "owned requirements" if unit.requirement_ids else "target sheet"
                    repair_instruction += (
                        " The empty groups list is the defect: emit the smallest complete set "
                        f"of groups needed for this {owned_target}."
                    )
                if any(
                    marker in error_text
                    for marker in (
                        "unresolved-footprint",
                        "unresolved-symbol",
                        "symbol-footprint-pad-mismatch",
                    )
                ):
                    repair_instruction += (
                        " For each unresolved-footprint / unresolved-symbol / "
                        "symbol-footprint-pad-mismatch defect, replace the rejected identifier "
                        "with one of that defect's `candidates` entries VERBATIM (call "
                        "search_symbols / search_footprints when none of the candidates fit); "
                        "never invent a library id, and keep the symbol and footprint drawn for "
                        "the same real part."
                    )
                if any(
                    marker in error_text
                    for marker in (
                        "missing-requirement-implementation",
                        "physical-obligation-unfulfilled",
                        "declared-interface-unrealized",
                    )
                ):
                    architecture = prompt_state.get("architecture") or {}
                    owned_ids = set(unit.requirement_ids)
                    identities = [
                        _requirement_identity(row)
                        for row in architecture.get("requirements") or []
                        if isinstance(row, dict)
                        and (not owned_ids or str(row.get("id")) in owned_ids)
                    ]
                    if identities:
                        repair_instruction += (
                            " The missing-requirement-implementation defect means no group "
                            "carries that requirement's identity; emit the real part for: "
                            + "; ".join(identities)
                            + "."
                        )
                if any(
                    marker in error_text
                    for marker in ("model_authored_protected_identity", "recipe-duplicate")
                ):
                    repair_instruction += (
                        " The groups named in model_authored_protected_identity / "
                        "recipe-duplicate are already supplied by the deterministic pipeline "
                        "or a sibling unit: DROP those groups entirely and emit only the "
                        "groups that implement this unit's own requirement_ids. Do not "
                        "re-emit or rename a dropped group."
                    )
            else:
                repair_instruction = (
                    "Correct every listed defect in this complete unit replacement."
                )
            local_feedback = {
                "defect": error_text,
                "rejected_unit": payload,
                "instruction": repair_instruction,
            }
        return "terminal", {
            "failure_kind": "unit_repair_exhausted",
            "error": str(local_feedback),
            **failure_detail,
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
        kind, payload = draft(unit, feedback=refused_deterministic.get(unit.unit_id))
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
            (
                merged,
                ref_to_unit,
                ref_to_lowering,
                trusted_lowering_group_ids,
            ) = merge_bom_units(
                units,
                candidates,
                prompt_state,
                trusted_unit_ids=frozenset(
                    unit_id for unit_id, source in unit_sources.items() if source == "lowerer"
                ),
            )
            normalization_state = {
                **prompt_state,
                "_trusted_lowering_group_ids": trusted_lowering_group_ids,
            }
            normalized, expanded_component_count = _normalize_stage_response(
                stage, merged, normalization_state
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
            diagnostic = getattr(exc, "diagnostic", None)
            last = {
                "failure_kind": "contract_rejected" if diagnostic else "invalid_schema",
                "error": str(exc),
                "schema_error": _redacted_schema_error(exc),
                "diagnostic": diagnostic,
            }
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
        targeted = tuple(unit_id for unit_id in targeted if unit_sources.get(unit_id) != "lowerer")
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

    seen_commit_failures: set[tuple] = set()
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
        gate_codes = list(
            _redacted_rejection_facts(
                commit_result,
                candidate_retained=True,
            ).get("commit_gate_codes")
            or []
        )
        targeted = route_work_unit_ids(
            commit_result,
            units,
            ref_to_unit=ref_to_unit,
            pin_to_unit=pin_to_unit,
            ref_to_unit_ids=ref_to_unit_ids,
        )
        undeclared_singletons = [
            offender
            for offender in commit_result.get("offenders") or []
            if "neither a power net nor a declared inter-sheet net" in str(offender)
        ]
        immutable_singleton_units = {
            unit_id
            for offender in undeclared_singletons
            for identity in _offender_identity(offender).split("|")
            if (unit_id := pin_to_unit.get(tuple(identity.split(".", 1))))
            and unit_sources.get(unit_id) == "lowerer"
        }
        if stage == "wiring" and "9.15" in gate_codes and immutable_singleton_units:
            last = {
                "failure_kind": "architecture_reconciliation_required",
                "reconcile_target": "architecture",
                "commit": commit_result,
                "work_unit_ids": sorted(immutable_singleton_units),
                "error": (
                    "deterministic wiring binds a singleton signal without a declared "
                    "architecture endpoint; sibling model units cannot repair its contract"
                ),
            }
            break
        targeted = tuple(unit_id for unit_id in targeted if unit_sources.get(unit_id) != "lowerer")
        if progress:
            progress(
                {
                    "kind": "retry",
                    "stage": stage,
                    "failure_kind": "commit_rejected",
                    "errors": commit_result.get("errors"),
                    "offenders": commit_result.get("offenders"),
                    "commit_gate_codes": gate_codes,
                    "work_unit_ids": list(targeted),
                    "accepted_siblings_retained": max(0, len(candidates) - len(targeted)),
                    "model": _client_model(stage_client),
                }
            )
        signature = _commit_rejection_signature(commit_result)
        candidate_fingerprint = hashlib.sha256(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rejection_key = (signature, candidate_fingerprint)
        repeated = rejection_key in seen_commit_failures
        if repeated:
            last = {
                "failure_kind": "commit_rejected",
                "commit": commit_result,
            }
            break
        seen_commit_failures.add(rejection_key)
        pristine = False
        if aggregate_round >= repair_rounds:
            last = {
                "failure_kind": "commit_rejected",
                "commit": commit_result,
            }
            break
        if aggregate_round == 0:
            escalation_profile = getattr(getattr(stage_client, "s", None), "escalation_profile", "")
            switch_profile = getattr(stage_client, "with_design_profile", None)
            if escalation_profile and callable(switch_profile):
                stage_client = switch_profile(escalation_profile)
        targeted = route_work_unit_ids(
            commit_result,
            units,
            ref_to_unit=ref_to_unit,
            pin_to_unit=pin_to_unit,
            ref_to_unit_ids=ref_to_unit_ids,
        )
        targeted = tuple(unit_id for unit_id in targeted if unit_sources.get(unit_id) != "lowerer")
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
                diagnostic = getattr(exc, "diagnostic", None)
                last = {
                    "failure_kind": "contract_rejected" if diagnostic else "invalid_schema",
                    "error": str(exc),
                    "schema_error": _redacted_schema_error(exc),
                    "diagnostic": diagnostic,
                }
                break
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
            "diagnostics": diagnostic_rows + _commit_rejection_diagnostics(last.get("commit")),
            "work_units": len(units),
            "reused_work_units": reused_work_units,
            "aggregate_repair_rounds": aggregate_repair_rounds,
        },
    )


def _standard_form_factor_block(intent: dict) -> str | None:
    """The pin/net contract of the design's standard mechanical template.

    A brief that resolved to a standard form factor (an Arduino Uno shield, say) owns that
    standard's *fixed* connectors, and the architecture stage must emit exactly those
    connectors with the template's own pin map. The template lives in
    ``kicraft.form_factors``, so without this block the stage is asked to reproduce role
    names and a 32-pin map it was never shown: the proto-shield brief answered with ONE
    composite requirement whose stacking role was the template key, and three correction
    rounds could not recover the four roles or their maps. Same shape as the recipe and
    lowerer summaries: deterministic reference data, rendered from the committed intent.

    Returns None when no validated standard is in play, so every other design's prompt is
    unchanged.
    """
    from kicraft.form_factors import get_template

    form_factor = intent.get("form_factor") or {}
    standard = form_factor.get("standard") if isinstance(form_factor, dict) else None
    template = get_template(standard)
    if template is None or not template.validated:
        return None
    lines = [
        f"STANDARD FORM FACTOR: {template.key} ({template.display_name}), "
        f"{template.board_width_mm} x {template.board_height_mm} mm, "
        f"{len(template.mounting_holes)} fixed mounting holes.",
        "This board's host interface is the standard's own FIXED CONNECTORS. Emit exactly "
        "one requirement per role below — never a composite, renamed, or arbitrary header — "
        "each with:",
        '  role: "connector" | family: "pin-header" | exact_part: null | '
        'parameters: {"rows": 1, "gender": "female"}',
        "  standard_stacking_role: the role name below",
        "  functional_blocks: the committed functional block that owns the host interface",
        "  ties: leave empty to accept this template's pin map (state it only if a pin must "
        "tie to a net this map does not already name)",
    ]
    for connector in template.fixed_connectors:
        pins = ", ".join(
            f"pin{index}={net}" for index, net in enumerate(connector.net_by_pin, start=1)
        )
        lines.append(
            f"  {connector.role} ({len(connector.net_by_pin)} pins): {pins}"
        )
    lines.append(
        "The template owns these pin positions and nets; do not re-plan them, and do not add "
        "headers for them after wiring."
    )
    return "\n".join(lines)


def _reviewed_option_detail(part, ratings: list[str]) -> list[str]:
    """What a stage needs to pick between reviewed carriers of one class, count first.

    A class can be carried by several reviewed parts that differ in exactly one property, and the
    model is choosing blind without it. The terminal class is carried by a 2-, a 3- and a
    4-position block that differ in nothing else, and the architecture stage that named the
    4-position part for a 2-contact requirement was refused five times out of six on the frozen
    live inputs *after* the refusal text had been taught to name the mismatch -- the choice has to
    be informed where it is made, which is what the ratings in this block are for too.
    """
    detail: list[str] = []
    if part.contacts:
        count = len(part.contacts)
        detail.append(f"{count} contact" + ("" if count == 1 else "s"))
    detail.extend(ratings)
    return detail


def _reviewed_class_options_block(intent: dict) -> str | None:
    """The reviewed parts that can satisfy each physical class this design demands.

    A demanded class is realized only by a reviewed part (``E_PHYSICAL_REALIZATION``), and
    the stage that picks the part never sees which parts those are: one proto-shield run
    answered the demanded ``voltage-regulator`` with the familiar but unreviewed
    ``AMS1117-3.3``, which resolves as a bundle and therefore passes every earlier gate,
    then refuses at the BOM with "found 0 ... exact MPN/symbol/footprint evidence" and no
    correction round left. Listing the reviewed options with their ratings lets the stage
    choose a realizable part where it is still choosing. Classes with no reviewed coverage
    are omitted: their honest route is the exact part the user named.
    """
    from kicraft.design.part_identity import (
        canonical_physical_features,
        reviewed_parts_for_feature,
    )

    demanded = sorted(
        {
            str(row.get("component_class") or "").strip().casefold()
            for row in (intent.get("obligations") or [])
            if isinstance(row, dict) and row.get("kind") == "physical"
        }
        - {""}
    )
    lines: list[str] = []
    for component_class in demanded:
        options: dict[str, str] = {}
        for feature in sorted(canonical_physical_features(component_class)):
            for part in reviewed_parts_for_feature(feature):
                limits = dict(part.operating_limits or {})
                ratings = [
                    f"{key.removesuffix('_v').removesuffix('_a').replace('_', ' ')}="
                    f"{limits[key]}"
                    for key in ("output_voltage_v", "output_current_a", "input_voltage_max_v")
                    if limits.get(key) is not None
                ]
                options.setdefault(
                    part.identity,
                    f"{part.identity}"
                    + (
                        f" ({part.family}: {', '.join(detail)})"
                        if (detail := _reviewed_option_detail(part, ratings))
                        else f" ({part.family})"
                    ),
                )
        if options:
            lines.append(
                f"  {component_class}: " + "; ".join(sorted(options.values())[:6])
            )
    if not lines:
        return None
    return "\n".join(
        [
            "REVIEWED PARTS FOR THE DEMANDED CLASSES: a demanded physical class is realized "
            "only by one of these reviewed identities (with its own symbol/footprint pair), "
            "so prefer them, and name `exact_part` only from this list. Where a part's contact "
            "count is given, it must match the contacts the requirement declares:",
            *lines,
        ]
    )


def _architecture_recipe_summaries(intent: dict) -> list[dict]:
    """Do not offer unrelated MCU alternatives to an explicitly named design.

    Identity comes from the reviewed ``part_identity`` relations as well as the
    name shape: a brief naming an order code of a registered family (the
    ESP32-S3-WROOM-1 ``-N16R8`` against the registered ``-N8R8``) must be offered
    that family's recipe, or the stage is asked to bind a circuit it was never
    shown and blocks on a variant it cannot discover. When the named parts
    identify no MCU recipe, every MCU recipe stays on offer rather than starving
    the stage.
    """
    from kicraft.design.part_identity import matches_part_identity
    from kicraft.design.recipes import get_recipe, recipe_summaries

    summaries = recipe_summaries()
    raw_named = [str(value) for value in intent.get("named_parts") or []]
    named = {
        re.sub(r"[^a-z0-9]", "", str(value).lower()) for value in intent.get("named_parts") or []
    }
    named = {value for value in named if len(value) >= 5}
    if not named:
        return summaries
    selected = set()
    for summary in summaries:
        if "mcu" not in summary["required_sheet_roles"]:
            continue
        definition = get_recipe(summary["recipe"])
        selectors = {
            re.sub(r"[^a-z0-9]", "", str(value).lower())
            for value in (definition.exact_part, definition.family, *definition.identity_aliases)
            if value
        }
        if any(
            selector.startswith(part) or part.startswith(selector)
            for selector in selectors
            if len(selector) >= 5
            for part in named
        ) or any(
            # Reviewed order-code membership preserves punctuation, so it gets
            # the raw name, not the separator-stripped token above.
            definition.exact_part and matches_part_identity(part, definition.exact_part)
            for part in raw_named
        ):
            selected.add(summary["recipe"])
    if not selected:
        return summaries
    return [
        summary
        for summary in summaries
        if "mcu" not in summary["required_sheet_roles"] or summary["recipe"] in selected
    ]


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
    auto_default_questions: bool | None = None,
    attempt_observer: Callable[[dict], None] | None = None,
) -> dict:
    run_id = (meta_ctx or {}).get("run_id")
    active_client = client
    t0 = time.monotonic()
    cpu0 = _child_cpu_s()
    # Explicit policy applies uniformly across all five stages. Legacy callers
    # retain their historical early-stage-only automatic behavior.
    auto_default = _auto_default_questions_enabled(
        stage,
        review_before_commit=review_before_commit,
        auto_default_questions=auto_default_questions,
    )
    # One typed-decision decider per drive: it reconciles naming differences the deterministic
    # path cannot settle (a class spelling, an unbound count subject, a part class no group
    # claims, a declared contact no symbol publishes). None when the setting is off.
    decider = _name_reconciliation_decider(client, stage=stage)
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
                # Carry the terminal reason. This event stream is what the stage tabs
                # render and what per-brief failure attribution reads, so a bare
                # {"ok": False} leaves a real failure with no cause anywhere the operator
                # or the harness looks (state.json still has it, the stream did not).
                progress(
                    {
                        "kind": "stage_done",
                        "stage": stage,
                        "ok": False,
                        "cost": 0.0,
                        "attempts": 0,
                        "warning": False,
                        "failure_kind": failure_kind,
                        "error": error,
                        "retryable": failure_kind == "provider_rate_limited",
                        "retry_action": (
                            "retry_stage"
                            if failure_kind == "provider_rate_limited"
                            else None
                        ),
                    }
                )
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
    if stage == "architecture":
        extras["circuit_recipes"] = _architecture_recipe_summaries(
            prep_json["state"].get("intent") or {}
        )
        from kicraft.design.lowering import lowerer_summaries

        extras["circuit_lowerers"] = lowerer_summaries()
        standard_block = _standard_form_factor_block(prep_json["state"].get("intent") or {})
        if standard_block:
            extras["standard_form_factor"] = standard_block
        class_options = _reviewed_class_options_block(prep_json["state"].get("intent") or {})
        if class_options:
            extras["reviewed_class_options"] = class_options

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
                    "resolution_source": p.get("resolution_source"),
                    "resolution_id": p.get("resolution_id"),
                    "lowering_requirement_id": p.get("lowering_requirement_id"),
                    "lowering_role": p.get("lowering_role"),
                    "lowering_index": p.get("lowering_index"),
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
                decider=decider,
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
                auto_default=auto_default,
                auto_default_questions=auto_default_questions,
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
        # Reference contracts must remain complete JSON: slicing can hide a
        # selected device's GPIO restrictions or the lowerer port contracts.
        user += "\n\nSTAGE EXTRAS (reference data from stage-prep):\n" + json.dumps(
            extras, separators=(",", ":")
        )
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

    # Explicit automatic runs use the no-questions contract for early stages;
    # interactive runs and legacy noninteractive instruction behavior stay intact.
    questions_allowed = not (
        (stage == "architecture" and instruction == NONINTERACTIVE_DEFAULTS_INSTRUCTION)
        or (
            auto_default_questions is True
            and stage in {"intent", "functional_spec", "architecture"}
        )
    )
    try:
        contract = build_stage_response_contract(
            stage,
            prompt_state,
            allow_questions=questions_allowed,
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
    ladder_modes = _contract_ladder_modes(active_client)
    pre_audit = _pre_audit_hook(active_client)
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
    if "no_serialization" in ladder_modes:
        # O1: no dedicated serialization rung. Contract rejections spend every
        # remaining call on ordinary preserving corrections; because the
        # clean-slate escape is armed only from inside the serialization
        # sub-path, it is unreachable in this arm.
        serialization_budget = 0
    # O5 compares each revision's declared identities against the previous one;
    # O7 accumulates every blocking diagnostic this drive has been shown.
    ladder_previous_identities: set[str] | None = None
    ladder_diagnostics: dict[str, str] = {}

    def ladder_suffix(
        raw_text: str,
        rejection_text: str,
        *,
        diagnostic: dict | None = None,
        commit_out: dict | None = None,
    ) -> str:
        """Extra text for the next correction: O5 dropped content + O7 history."""
        nonlocal ladder_previous_identities
        parts: list[str] = []
        if "dropped_gate" in ladder_modes:
            current = _raw_declared_identities(raw_text)
            if current is not None:
                if ladder_previous_identities is not None:
                    lost = sorted(
                        _identity_name(item)
                        for item in ladder_previous_identities - current
                        # A diagnostic that already names the item owns its fix.
                        if _identity_name(item) not in rejection_text
                    )
                    if lost:
                        parts.append(
                            "\nREGRESSION FIX: the previous revision declared "
                            + ", ".join(repr(name) for name in lost)
                            + " and this revision dropped it. Restore every one "
                            "unless a diagnostic above asks for its removal."
                        )
                ladder_previous_identities = current
        if "full_feedback" in ladder_modes:
            # Every member row, whichever shape it was written in: the aggregate carries them in
            # `findings` now and in `evidence` in older artifacts, and this arm's whole point is
            # re-sending every blocking defect the stage has reported.
            for entry in _diagnostic_member_rows(diagnostic if isinstance(diagnostic, dict) else None):
                ladder_diagnostics[str(entry["code"])] = str(entry.get("message") or "")[:280]
            for error in (commit_out or {}).get("errors") or []:
                label = " ".join(str(error).split())[:60]
                ladder_diagnostics.setdefault(f"commit:{label}", str(error)[:280])
            if ladder_diagnostics:
                parts.append(
                    "\nALL BLOCKING DEFECTS THIS STAGE HAS REPORTED (fix every one; "
                    "do not trade one for another):\n"
                    + "\n".join(
                        f"- {code}: {message}" for code, message in ladder_diagnostics.items()
                    )
                )
        return "".join(parts)

    def ladder_identities(raw_text: str, candidate: dict | None = None) -> list[str]:
        """The rung's declared identities — what the O5 arm is measured on.

        Recorded on the correction event so an operator can diff consecutive
        rungs (which declared net or port binding a revision lost) without the
        raw reply ever leaving the process.
        """
        if isinstance(candidate, dict):
            return sorted(_declared_identities(candidate))
        return sorted(_raw_declared_identities(raw_text) or [])

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
    design_repair_rounds = 0
    clean_slate_armed_signature: tuple | None = None
    serialization_calls = 0
    attempts = 0
    # Blocking defect classes this drive's drafts were rejected for (plan §6).
    defect_codes: list[str] = []
    # Reader refusals, counted once per rejected ATTEMPT (not per code), so
    # `first_draft_contract_clean` can tell a reader refusal apart from a
    # semantic repair round (next-steps plan §3).
    contract_rejections = 0
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

    for attempt in range(max_retries + 1 + _MAX_DESIGN_CONTRACT_REPAIR_ROUNDS):
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
                    max_tokens=(
                        int(policy.serialization_max_tokens) if was_clean_slate else normal_cap
                    ),
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
                        "call_mode": current_call_mode,
                        "model": _client_model(active_client),
                    }
                )
            loop_retries += 1
            reasoning = {"enabled": False}
            temperature = max(temperature + 0.4, 0.4)
            messages = _lean_retry(None, _REASONING_LOOP_RETRY_MSG)
            continue

        outcome = decode_stage_response(prepared, facts, pre_audit=pre_audit)
        emit_candidate_decoded(
            outcome,
            provider_attempt=attempts,
            serialization_recovery=False,
            clean_slate=was_clean_slate,
        )
        schema_error = outcome.payload.get("failure_kind") in _SCHEMA_REJECTION_KINDS
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
                "diagnostic": outcome.payload.get("diagnostic"),
                "audit_findings": outcome.payload.get("audit_findings") or [],
            }
            rejection_codes = _diagnostic_codes(last.get("diagnostic"))
            defect_codes.extend(rejection_codes)
            if rejection_codes:
                # A `retry` carrying a diagnostic is the reader refusing the
                # draft; a parse failure carries none and is not a contract
                # rejection.
                contract_rejections += 1
            _record_attempt_facts(
                active_client,
                run_id=run_id,
                stage=stage,
                attempt=attempts,
                call_mode="clean_slate" if was_clean_slate else "normal",
                outcome=kind,
                facts=facts,
                schema_error=_redacted_schema_error(schema_error_detail),
            )
            if progress:
                progress(
                    {
                        "kind": "retry",
                        "stage": stage,
                        "errors": [last["error"]],
                        "failure_kind": kind,
                        # Which rung of the correction ladder this attempt took:
                        # normal | clean_slate | serialization. A reader needs it
                        # because a rejected clean-slate escape is terminal by
                        # policy, whatever the nominal budget still allows.
                        "call_mode": current_call_mode,
                        "schema_error": _redacted_schema_error(schema_error_detail),
                        "diagnostic": outcome.payload.get("diagnostic"),
                        "declared_identities": ladder_identities(raw),
                        "model": _client_model(active_client),
                    }
                )
            if was_clean_slate:
                # O3: like the commit path, a rejected clean-slate is terminal
                # only when the rejection identity is unchanged. A different
                # defect set is progress and earns ordinary preserving
                # corrections (never a second escape: clean_slate_spent stays).
                # A rejection with no identity to compare (a parse failure)
                # keeps the stock terminal behaviour.
                #
                # A design-contract refusal earns the same treatment by default, without the
                # ladder flag: it is a design defect with an itemized fix, and a single rewrite
                # fixes one defect while breaking another (four live architecture drafts). Two
                # such rounds, then the stock terminal behaviour returns.
                rejection_identity = _schema_rejection_signature(kind, last.get("diagnostic"))
                earned_by_ladder = "signature" in ladder_modes
                earned_by_design = (
                    not earned_by_ladder
                    and kind == "contract_rejected"
                    and rejection_identity is not None
                    and design_repair_rounds < _MAX_DESIGN_CONTRACT_REPAIR_ROUNDS
                )
                if (
                    not (earned_by_ladder or earned_by_design)
                    or clean_slate_armed_signature is None
                    or rejection_identity is None
                    or rejection_identity == clean_slate_armed_signature
                ):
                    break
                if earned_by_design:
                    # Each granted round sanctions its own call: the budget carries them, so no
                    # other lane's ceiling moves (an unchanged defect set still stops at stock).
                    design_repair_rounds += 1
                    clean_slate_armed_signature = rejection_identity
                    provider_call_budget += 1
            elif (
                clean_slate_spent
                and kind == "contract_rejected"
                and "signature" not in ladder_modes
            ):
                # The escape is spent, so a later design-contract refusal is judged on its own
                # identity: a changed defect set earns the second round (and its call), while an
                # unchanged one stays terminal exactly as before.
                rejection_identity = _schema_rejection_signature(kind, last.get("diagnostic"))
                if (
                    rejection_identity is None
                    or rejection_identity == clean_slate_armed_signature
                    or design_repair_rounds >= _MAX_DESIGN_CONTRACT_REPAIR_ROUNDS
                ):
                    break
                design_repair_rounds += 1
                clean_slate_armed_signature = rejection_identity
                provider_call_budget += 1
            if attempts >= provider_call_budget:
                break
            if serialization_calls >= serialization_budget:
                messages = _lean_retry(
                    raw if schema_error else None,
                    _stage_recovery_message(
                        kind,
                        raw or "",
                        _collection_bounds_sentence(policy.collection_bounds),
                        schema_error=schema_error_detail,
                        diagnostic=last.get("diagnostic"),
                        collection_limit=collection_limit,
                        audit_notes=last.get("audit_findings"),
                    )
                    + ladder_suffix(
                        raw,
                        _rejection_text(last.get("schema_error"), last.get("diagnostic")),
                        diagnostic=last.get("diagnostic"),
                    ),
                )
                reasoning = {"enabled": False}
                temperature = max(escape_temperature, 0.0)
                continue
            # Use at most one dedicated plain, tool-free serialization call.
            # Any remaining bounded attempts may still correct its failure.
            serialization_calls += 1
            attempts += 1  # the serialization completion is a provider call too
            current_attempt_number = attempts
            sctx = {**ctx, "serialization": True, "call_mode": "serialization"}
            smessages = list(call_messages)
            bounds_sentence = _collection_bounds_sentence(policy.collection_bounds)
            retry_message = _stage_recovery_message(
                kind,
                raw or "",
                bounds_sentence,
                schema_error=schema_error_detail,
                diagnostic=last.get("diagnostic"),
                collection_limit=collection_limit,
                audit_notes=last.get("audit_findings"),
            ) + ladder_suffix(
                raw,
                _rejection_text(last.get("schema_error"), last.get("diagnostic")),
                diagnostic=last.get("diagnostic"),
            )
            if schema_error and raw:
                smessages.append({"role": "assistant", "content": raw})
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
            serialization_outcome = decode_stage_response(prepared, sfacts, pre_audit=pre_audit)
            emit_candidate_decoded(
                serialization_outcome,
                provider_attempt=attempts,
                serialization_recovery=True,
                clean_slate=was_clean_slate,
            )
            schema_error = (
                serialization_outcome.payload.get("failure_kind") in _SCHEMA_REJECTION_KINDS
            )
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
                    schema_error=_redacted_schema_error(schema_error_detail),
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
                    "diagnostic": serialization_outcome.payload.get("diagnostic"),
                }
                if attempts >= provider_call_budget or clean_slate_spent:
                    break
                clean_slate_spent = True
                clean_slate_next = True
                clean_slate_armed_signature = _schema_rejection_signature(
                    skind, last.get("diagnostic")
                )
                # O4: the escape stays a single bounded call, but a preserving
                # one — the candidate that was already closest to committing
                # travels with it instead of a from-scratch slot.
                preserve = "preserving" in ladder_modes
                messages = _lean_retry(
                    (sraw or None) if preserve else None,
                    _stage_recovery_message(
                        skind,
                        sraw if preserve else "",
                        _collection_bounds_sentence(policy.collection_bounds),
                        schema_error=schema_error_detail,
                        diagnostic=last.get("diagnostic"),
                        collection_limit=scollection_limit,
                        audit_notes=last.get("audit_findings"),
                    )
                    + (
                        " Preserve every already-valid net, port binding and "
                        "endpoint; change only the reported defect and emit one "
                        "complete compact JSON object."
                        if preserve
                        else " Start from the binding state; emit one fresh compact JSON object."
                    )
                    + ladder_suffix(
                        sraw,
                        _rejection_text(last.get("schema_error"), last.get("diagnostic")),
                        diagnostic=last.get("diagnostic"),
                    ),
                )
                reasoning = {"enabled": False}
                temperature = max(escape_temperature, 0.0)
                continue
            # Parseable serialization output: the commit path owns it from here.
            # A commit rejection may still use remaining commit-correction
            # attempts at the normal stage/tool policy.
            raw = sraw
            current_facts = sfacts
            current_call_mode = "serialization"

        # A clarifying-question payload has no slot this turn. The persisted
        # policy decides whether an ordinary question parks or receives the
        # bounded sensible-default retry; reconciliation always parks.
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
            if _questions_need_input(
                qs,
                stage,
                auto_default=auto_default,
                answers=answers,
                instruction=instruction,
                auto_default_questions=auto_default_questions,
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

        obj = _normalize_candidate_for_diagnostics(
            stage, obj, brief, semantic_state, decider=decider
        )
        original_obj = obj
        diagnostics = diagnose_stage(
            stage, brief=brief, upstream_state=semantic_state, candidate=obj
        )
        provider_diagnostic_codes = [diagnostic.code for diagnostic in diagnostics]
        defect_codes.extend(provider_diagnostic_codes)
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
            # The rulebook, asked of Jev before the compiler refuses the draft: a finding lands in
            # the same correction round as the deterministic ones instead of costing the stage a
            # contract refusal. Fail-soft by construction — an unavailable auditor contributes
            # nothing and never blocks the draft.
            audit_findings = _audit_architecture_draft(active_client, obj)
            if audit_findings:
                diagnostics = [*diagnostics, *audit_findings]
                provider_diagnostic_codes = [
                    *provider_diagnostic_codes,
                    *(finding.code for finding in audit_findings),
                ]
                defect_codes.extend(finding.code for finding in audit_findings)
                if progress:
                    for finding in audit_findings:
                        progress(
                            {
                                "kind": "stage_diagnostic",
                                "stage": stage,
                                "attempt": current_attempt_number,
                                **finding.model_dump(exclude_none=True),
                            }
                        )
        severe = [d for d in diagnostics if d.severity in {"repair_required", "fab_gate"}]
        # The external-load current is a physical fact. Interactive policy parks
        # immediately; automatic policy must fail honestly rather than inventing a
        # current or spending another provider call trying to manufacture one.
        if (
            severe
            and stage == "architecture"
            and semantic_mode in {"repair", "enforce"}
            and not external_load_budget_stated(brief)
            and any(d.code == EXTERNAL_LOAD_CURRENT_CODE for d in severe)
        ):
            questions = _normalize_questions([EXTERNAL_LOAD_CURRENT_QUESTION], stage)
            if questions_allowed and (
                _questions_need_input(
                    questions,
                    stage,
                    auto_default=auto_default,
                    answers=answers,
                    instruction=instruction,
                    auto_default_questions=auto_default_questions,
                )
                # The special missing-current path historically parked legacy
                # callers before semantic repair, even in production.
                or (auto_default_questions is None and not answers)
            ):
                if not review_before_commit:
                    attach_questions(state_path, stage, questions)
                if progress:
                    progress({"kind": "question", "stage": stage, "questions": questions})
                parked = {
                    "stage": stage,
                    "commit_ok": False,
                    "needs_input": True,
                    "questions": questions,
                    "cost_usd": total_cost,
                    "attempts": attempts,
                }
                if review_before_commit:
                    parked.update(
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
                return parked
            if auto_default_questions is True:
                diagnostic_rows = [d.model_dump(exclude_none=True) for d in diagnostics]
                failure_outcome = {
                    "failure_kind": EXTERNAL_LOAD_CURRENT_CODE,
                    "error": next(
                        d.message for d in severe if d.code == EXTERNAL_LOAD_CURRENT_CODE
                    ),
                    "provider_ok": provider_ok,
                    "schema_ok": schema_ok,
                    "semantic_clean": False,
                    "repair_required": True,
                    "fab_safe": not any(d.severity == "fab_gate" for d in diagnostics),
                    "diagnostics": diagnostic_rows,
                }
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
                        **failure_outcome,
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
                    defect_codes=tuple(defect_codes),
                    contract_rejections=contract_rejections,
                    semantic_repair_rounds=semantic_repair_rounds,
                    outcome=failure_outcome,
                )
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
                repair_outcome = decode_stage_response(
                    prepared, repair_facts, pre_audit=pre_audit
                )
                if repair_outcome.kind == "candidate":
                    repaired = _normalize_candidate_for_diagnostics(
                        stage,
                        repair_outcome.payload["candidate"],
                        brief,
                        semantic_state,
                        decider=decider,
                    )
                    if stage == "architecture":
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
                    if stage == "architecture":
                        # The audit judges the draft that would be adopted, not only the first
                        # one: a correction is exactly where a rule can be broken again.
                        repaired_diagnostics = [
                            *repaired_diagnostics,
                            *_audit_architecture_draft(active_client, repaired),
                        ]
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
                    # The adopted candidate is THIS call's, not the first attempt's. The
                    # review row and the candidate_review event below read the current_*
                    # context, and leaving it stale attributed the reviewed candidate to the
                    # superseded call (debug walkthrough 2026-09-24, seed 37: attempts=2 but
                    # candidate_review attempt=1 and a review row carrying call 1's facts).
                    current_attempt_number = attempts
                    current_facts = repair_facts
                    current_call_mode = "semantic_repair"
                    continue

                if repair_outcome.kind == "questions":
                    qs = _normalize_questions(
                        repair_outcome.payload["candidate"]["questions"], stage
                    )
                    if _questions_need_input(
                        qs,
                        stage,
                        auto_default=auto_default,
                        answers=answers,
                        instruction=instruction,
                        auto_default_questions=auto_default_questions,
                    ):
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
                "repair_attempted": semantic_repair_attempted,
                "repair_adopted": semantic_repair_adopted,
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
                defect_codes=tuple(defect_codes),
                contract_rejections=contract_rejections,
                semantic_repair_rounds=semantic_repair_rounds,
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
            "diagnostics": diagnostic_rows + _commit_rejection_diagnostics(out),
        }
        if progress:
            progress(
                {
                    "kind": "retry",
                    "stage": stage,
                    "errors": out.get("errors"),
                    "offenders": out.get("offenders"),
                    "call_mode": current_call_mode,
                    "declared_identities": ladder_identities(raw, obj),
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
                _retry_feedback(out, stage=stage, valid_refs=None)
                + ladder_suffix(
                    raw,
                    _rejection_text(out.get("errors"), out.get("offenders")),
                    commit_out=out,
                ),
            )
            continue
        # Bounded continuation: a post-escape response with a NEW signature
        # (or a first-seen signature) gets the ordinary preserving correction
        # feedback; it cannot re-arm the escape (clean_slate_spent stays True).
        _valid_refs = committed_bom_refs(state_path) if stage == "wiring" else None
        messages = _lean_retry(
            raw,
            _retry_feedback(out, stage=stage, valid_refs=_valid_refs)
            + ladder_suffix(
                raw,
                _rejection_text(out.get("errors"), out.get("offenders")),
                commit_out=out,
            ),
        )

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
        defect_codes=tuple(defect_codes),
        contract_rejections=contract_rejections,
        semantic_repair_rounds=semantic_repair_rounds,
        outcome=last,
    )
