"""Shared presentation contract for pipeline activity (no NiceGUI, no I/O).

Three surfaces read the same event stream and the same durable facts and must
agree on what they say: the My-projects list, the live/reopened workspace
summary, and the per-stage tabs. This module owns the vocabulary they share.

* WHICH events are durable (journaled while a run is in flight, so a crash
  leaves an ordered, attributable record) and which are presentation-only
  token deltas that only ever land in the final transcript,
* how one attempt's activity is reduced from that stream
  (:func:`reduce_activity`),
* which stage an implicitly-staged event belongs to
  (:func:`attribute_stage`),
* how a project is NAMED from a committed stem or its brief
  (:func:`project_title`),
* what technical text may be persisted or shown to a user
  (:func:`sanitize_activity_event`),
* how a flat event history splits back into attempts
  (:func:`group_attempts`).

Nothing here changes execution semantics. The database statuses
(``running`` / ``awaiting_input`` / ``ok`` / ``failed`` / ``interrupted``),
quota accounting, retry policy, and the artifact-derived stage outcomes in
:func:`kicraft.server.session.derive_stage_statuses` stay exactly where they
are; this module is display-only.
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Event taxonomy
# --------------------------------------------------------------------------- #

#: Execution-provenance kinds: which recipe / work unit produced a candidate.
#: Defined once here; web.py and stagetabs.py both import it (they used to
#: carry byte-identical duplicates that could drift apart).
PROVENANCE_EVENT_KINDS = frozenset(
    {"recipe_selected", "work_unit_plan", "work_unit_attempt", "work_unit_done"}
)

#: The explicit structural subset journaled to ``provenance.jsonl`` WHILE a run
#: is in flight (appended + flushed, fsynced for structural/terminal records).
#: Token deltas (``reasoning_delta`` / ``answer_delta``) are deliberately absent:
#: they are high-volume, low-information, and the final transcript snapshot
#: already carries them.
DURABLE_ACTIVITY_KINDS = frozenset(
    {
        "run_started",
        "run_finished",
        "stage_start",
        "stage_done",
        "question",
        "retry",
        "stage_diagnostic",
        "tool",
        "tool_result",
        "provider_fallback",
        "escalation",
        "serialization_recovery",
        "candidate_decoded",
        *PROVENANCE_EVENT_KINDS,
        "build_start",
        "queue",
        "build_log",
        "build_done",
        "run_error",
    }
)

#: Kinds that may appear without an explicit ``stage`` and inherit the most
#: recently announced one. An event that still cannot be attributed stays
#: unattributed rather than being blamed on Intent.
IMPLICIT_STAGE_KINDS = frozenset(
    {
        "tool",
        "tool_result",
        "retry",
        "stage_diagnostic",
        "reasoning_delta",
        "answer_delta",
        "serialization_recovery",
        "provider_fallback",
        "escalation",
        "candidate_decoded",
        "build_log",
        "queue",
    }
)

#: One attempt's presentation phase. Distinct from the durable DB status: a
#: queued build has no DB status of its own, and ``finalizing`` is the window
#: between a job finishing and its project row catching up.
PHASE_STATUSES = (
    "starting",
    "running",
    "queued",
    "awaiting_input",
    "failed",
    "interrupted",
    "complete",
)

#: Presentation-only statuses consumed by ``web._project_presentation``.
PRESENTATION_STATUSES = (
    "starting",
    "running",
    "retrying",
    "queued",
    "awaiting_input",
    "finalizing",
    "complete",
    "complete_with_warnings",
    "failed",
    "interrupted",
    "unavailable",
)

#: Actions a surface may offer. The handlers themselves stay in web.py.
ACTIONS = (
    "answer",
    "continue",
    "rebuild",
    "download",
    "open",
    "new_from_brief",
    "support",
)

_RUN_MODES = ("design", "continue", "build", "manual_route")

# Human activity per stage. These are NOT stage names (those come from
# ``stagetabs.PHASES``) -- they describe what the stage is doing right now.
_STAGE_ACTIVITY = {
    "intent": "Reading your brief",
    "functional_spec": "Planning the functional blocks",
    "architecture": "Planning the sheet architecture",
    "bom": "Choosing components",
    "wiring": "Wiring the connections",
    "synthesize": "Synthesizing the schematic",
    "place_route": "Placing and routing the board",
    "electrical_review": "Reviewing the design",
    "fab": "Exporting the fab package",
}

# Fallback labels for stage keys this module is asked about before stagetabs
# has been imported (and for genuinely unknown keys).
_STAGE_LABEL_FALLBACK = {
    "intent": "Intent",
    "functional_spec": "Functional",
    "architecture": "Architecture",
    "bom": "BOM",
    "wiring": "Wiring",
    "synthesize": "Synthesize",
    "place_route": "Place/Route",
    "electrical_review": "Elec Review",
    "fab": "Fab",
}


def now_iso() -> str:
    """UTC ISO-8601 timestamp for an event envelope."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Stage labels (single source: stagetabs.PHASES)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _phase_labels() -> dict[str, str]:
    """`stagetabs.PHASES` labels, imported lazily.

    stagetabs imports this module for its event-kind constants, so a
    module-level import here would be circular. The import is deferred to first
    use (inside a UI/render context, where stagetabs is already loaded) and
    cached; the human label table is never duplicated.
    """
    try:
        from .stagetabs import PHASES
    except Exception:  # pragma: no cover - defensive: never break rendering
        return dict(_STAGE_LABEL_FALLBACK)
    return {key: label for key, label, _icon, _accent in PHASES}


def stage_label(stage: str | None) -> str:
    """The human tab label for a stage key ('' for an unattributed stage)."""
    if not stage:
        return ""
    return _phase_labels().get(str(stage)) or _STAGE_LABEL_FALLBACK.get(
        str(stage), str(stage)
    )


def stage_order() -> tuple[str, ...]:
    """Stage keys in pipeline order (the tab row's order, one source)."""
    try:
        from .stagetabs import PHASES
    except Exception:  # pragma: no cover - defensive
        return tuple(_STAGE_LABEL_FALLBACK)
    return tuple(key for key, _label, _icon, _accent in PHASES)


def stage_activity(stage: str | None) -> str:
    """What a stage is doing, as a user-facing sentence."""
    if not stage:
        return "Working"
    return _STAGE_ACTIVITY.get(str(stage), f"Working on {stage_label(stage)}")


def build_substage(text: str) -> str | None:
    """Which build sub-phase a ``[build] N/5`` log line belongs to.

    Delegates to the live tabs' own classifier (``stagetabs._build_substage``) so
    a reopened timeline and the live stream agree on the split, instead of
    carrying a second copy of the marker parser.
    """
    try:
        from .stagetabs import _build_substage
    except Exception:  # pragma: no cover - defensive
        return None
    return _build_substage(text)


def attribute_stage(event: dict, prior_stage: str | None) -> str | None:
    """Resolve and stamp the stage an event belongs to; return it.

    An explicit ``stage`` wins. For the implicit kinds (tool calls, deltas,
    retries without a stage, build-log lines) the most recently announced stage
    is inherited -- build logs first through the ``[build] N/5`` classifier, so
    ``Build stopped during Place/Route`` can name the right phase. An event that
    still cannot be attributed is left unattributed.
    """
    stage = event.get("stage")
    if not stage and event.get("kind") == "build_log":
        stage = build_substage(str(event.get("text") or ""))
    if not stage and event.get("kind") in IMPLICIT_STAGE_KINDS:
        stage = prior_stage
    if stage:
        event["stage"] = stage
    return stage or None


# --------------------------------------------------------------------------- #
# Titles
# --------------------------------------------------------------------------- #

_TITLE_MAX = 80
_ELLIPSIS = "…"


def project_title(
    brief: str | None,
    stem: str | None,
    *,
    board_code: str | None = None,
    project_id: int | None = None,
) -> str:
    """The one display name for a project, on every surface.

    A nonblank committed stem wins (its spelling is preserved: it IS the name of
    the KiCad project). Otherwise the brief, whitespace-normalized and truncated
    at a word boundary to at most 80 characters including the ellipsis.
    Otherwise the board code, the project id, or a placeholder -- never an empty
    label. This is display-only: the brief excerpt is never used as a filesystem
    or artifact stem.
    """
    if stem is not None and str(stem).strip():
        return str(stem).strip()
    text = " ".join(str(brief or "").split())
    if text:
        return _truncate_at_word(text, _TITLE_MAX)
    if board_code is not None and str(board_code).strip():
        return f"Board {str(board_code).strip()}"
    if project_id is not None:
        return f"Project {project_id}"
    return "Untitled project"


def _truncate_at_word(text: str, limit: int) -> str:
    """`text` cut to at most `limit` characters ending on a word boundary."""
    if len(text) <= limit:
        return text
    budget = limit - len(_ELLIPSIS)
    head = text[:budget]
    cut = head.rfind(" ")
    if cut > 0:
        head = head[:cut]
    head = head.rstrip()
    if not head:  # one giant token: hard cut rather than an empty label
        head = text[:budget].rstrip()
    return head + _ELLIPSIS


# --------------------------------------------------------------------------- #
# Activity reduction
# --------------------------------------------------------------------------- #

_MAX_TOOLS = 200
_MAX_ISSUES = 50
_ISSUE_SEVERITIES = ("error", "warning", "info")


def blank_activity() -> dict:
    """A run's activity before any event has been seen."""
    return {
        "run_id": None,
        "mode": None,
        "stage": None,
        "phase_status": "starting",
        "started_at": None,
        "last_event_at": None,
        "last_activity": None,
        "failure": None,
        "issues": [],
        "tools": [],
        "planned_units": set(),
        "completed_units": set(),
        "queue": None,
        "build_attempt": None,
    }


def reduce_activity(previous: dict | None, event: dict) -> dict:
    """Fold one event into a run's activity.

    Mutates and returns `previous` when given -- the page calls this for every
    streamed event, including token deltas, so it must not copy a dict per
    token. Pass ``None`` to start a fresh activity.

    Returns an activity with ``run_id``, ``mode``, ``stage``, ``phase_status``,
    ``started_at``, ``last_event_at``, ``last_activity``, ``failure``,
    ``issues``, ``tools``, ``planned_units`` / ``completed_units`` id sets, and
    the last observed ``queue`` / ``build_attempt`` facts. Unknown event kinds
    update only the freshness timestamp: an unrecognized event can never change
    the run's outcome.
    """
    act = blank_activity() if previous is None else previous
    kind = event.get("kind")
    ts = event.get("ts")
    if ts:
        act["last_event_at"] = ts
    else:
        act["last_event_at"] = now_iso()

    if not isinstance(kind, str):
        return act

    if kind == "run_started":
        act.update(
            run_id=event.get("run_id"),
            mode=event.get("mode") or act.get("mode"),
            stage=None,
            phase_status="running",
            started_at=event.get("ts") or now_iso(),
            failure=None,
            last_activity="Starting the design run",
        )
        return act

    if event.get("run_id") and not act.get("run_id"):
        act["run_id"] = event.get("run_id")
    if event.get("stage"):
        act["stage"] = event.get("stage")

    if kind == "run_finished":
        status = str(event.get("status") or "")
        act["phase_status"] = {
            "ok": "complete",
            "failed": "failed",
            "awaiting_input": "awaiting_input",
            "interrupted": "interrupted",
        }.get(status, act["phase_status"])
        if event.get("stage"):
            act["stage"] = event.get("stage")
        failure_kind = event.get("failure_kind")
        if (status and status != "ok") or failure_kind:
            # The cause a structured event already recorded ("BOM failed")
            # outranks whatever the activity line holds at that moment: the last
            # line can be a driver's raw stdout dump, which is not a cause. Only
            # text that may be shown to a person is carried into the fact.
            prior = act.get("failure") or {}
            message = (human_failure_text(prior.get("message"))
                       or human_failure_text(act.get("last_activity")))
            act["failure"] = _failure_fact(
                kind=failure_kind or prior.get("kind"),
                stage=event.get("stage") or act.get("stage") or prior.get("stage"),
                retryable=(prior.get("retryable") if event.get("retryable") is None
                           else event.get("retryable")),
                retry_action=(prior.get("retry_action")
                              if event.get("retry_action") is None
                              else event.get("retry_action")),
                message=message,
            )
        act["last_activity"] = {
            "ok": "Run finished",
            "failed": "Run stopped",
            "awaiting_input": "Waiting for your answer",
            "interrupted": "Run was interrupted",
        }.get(status, act.get("last_activity"))
        return act

    if kind == "stage_start":
        act["phase_status"] = "running"
        act["last_activity"] = stage_activity(act.get("stage"))
        return act

    if kind == "stage_done":
        ok = bool(event.get("ok"))
        stage = act.get("stage")
        if ok:
            act["last_activity"] = f"{stage_label(stage) or 'Stage'} finished"
            if event.get("warning"):
                _add_issue(
                    act,
                    {
                        "severity": "warning",
                        "stage": stage,
                        "code": "committed_with_findings",
                        "message": (
                            f"{stage_label(stage) or 'Stage'} committed with findings"
                        ),
                        "evidence": [],
                    },
                )
        else:
            act["phase_status"] = "failed"
            act["failure"] = _failure_fact(
                kind=event.get("failure_kind") or "stage_failed",
                stage=stage,
                retryable=event.get("retryable"),
                retry_action=event.get("retry_action"),
                message=f"{stage_label(stage) or 'Stage'} failed",
            )
            act["last_activity"] = act["failure"]["message"]
            _add_issue(
                act,
                {
                    "severity": "error",
                    "stage": stage,
                    "code": str(event.get("failure_kind") or "stage_failed"),
                    "message": act["failure"]["message"],
                    "evidence": _evidence_list(event.get("errors")),
                },
            )
        return act

    if kind == "question":
        act["phase_status"] = "awaiting_input"
        act["last_activity"] = "Waiting for your answer"
        return act

    if kind == "retry":
        errors = _evidence_list(event.get("errors"))
        failure_kind = str(event.get("failure_kind") or "retry")
        act["last_activity"] = (
            f"Retrying {stage_label(act.get('stage')) or 'the stage'}"
            + (f" after {_humanize_kind(failure_kind)}" if failure_kind else "")
        )
        if act["phase_status"] not in ("awaiting_input", "queued"):
            act["phase_status"] = "running"
        _add_issue(
            act,
            {
                "severity": "warning",
                "stage": act.get("stage"),
                "code": failure_kind,
                "message": act["last_activity"],
                "evidence": errors,
            },
        )
        return act

    if kind == "stage_diagnostic":
        code = str(event.get("code") or "semantic_finding")
        _add_issue(
            act,
            {
                "severity": str(event.get("severity") or "warning"),
                "stage": act.get("stage"),
                "code": code,
                "message": str(event.get("message") or code),
                "evidence": _evidence_list(event.get("evidence")),
            },
        )
        act["last_activity"] = f"{stage_label(act.get('stage')) or 'Stage'}: {code}"
        return act

    if kind == "tool":
        _tool_call(act, event)
        act["last_activity"] = f"Using {event.get('name') or 'a tool'}"
        return act

    if kind == "tool_result":
        name = _tool_result(act, event)
        act["last_activity"] = (
            f"{name} returned" if name else (act.get("last_activity") or "Tool returned")
        )
        return act

    if kind == "run_error":
        failure_kind = str(event.get("failure_kind") or "run_error")
        message = str(event.get("message") or _humanize_kind(failure_kind))
        act["phase_status"] = "failed"
        act["failure"] = _failure_fact(
            kind=failure_kind,
            stage=act.get("stage"),
            retryable=event.get("retryable"),
            retry_action=event.get("retry_action"),
            message=message,
            exception_type=event.get("exception_type"),
        )
        act["last_activity"] = message
        _add_issue(
            act,
            {
                "severity": "error",
                "stage": act.get("stage"),
                "code": failure_kind,
                "message": message,
                "evidence": _evidence_list(event.get("evidence")),
            },
        )
        return act

    if kind == "build_start":
        act["phase_status"] = "running"
        act["build_attempt"] = {
            "job_id": event.get("job_id"),
            "log_offset": event.get("log_offset"),
        }
        act["last_activity"] = "Starting the board build"
        return act

    if kind == "queue":
        position = event.get("position")
        depth = event.get("depth")
        eta_s = event.get("eta_s")
        act["phase_status"] = "queued"
        act["queue"] = {"position": position, "depth": depth, "eta_s": eta_s}
        act["last_activity"] = _queue_sentence(position, eta_s)
        return act

    if kind == "build_log":
        text = str(event.get("text") or "").rstrip()
        if text:
            act["last_activity"] = text.splitlines()[-1][:200]
        if act["phase_status"] == "queued":
            act["phase_status"] = "running"
        return act

    if kind == "build_done":
        if not event.get("ok"):
            message = "Board build failed"
            stage = act.get("stage")
            if stage:
                message = f"Build stopped during {stage_label(stage)}"
            act["phase_status"] = "failed"
            act["failure"] = _failure_fact(
                kind=str(event.get("failure_kind") or "build_failed"),
                stage=stage,
                retryable=event.get("retryable"),
                retry_action=event.get("retry_action"),
                message=message,
            )
            act["last_activity"] = message
            _add_issue(
                act,
                {
                    "severity": "error",
                    "stage": stage,
                    "code": str(event.get("failure_kind") or "build_failed"),
                    "message": message,
                    "evidence": [],
                },
            )
        else:
            act["last_activity"] = "Board build finished"
        return act

    if kind == "provider_fallback":
        act["last_activity"] = (
            f"Switching model to {event.get('to') or 'a fallback'}"
        )
        return act

    if kind == "escalation":
        act["last_activity"] = f"Escalating to {event.get('to') or 'a stronger model'}"
        return act

    if kind == "serialization_recovery":
        act["last_activity"] = "Recovering the model's response format"
        return act

    if kind == "candidate_decoded":
        act["last_activity"] = (
            f"Decoded a {stage_label(act.get('stage')) or 'stage'} candidate"
        )
        return act

    if kind == "recipe_selected":
        act["last_activity"] = f"Selected recipe {event.get('recipe') or 'unknown'}"
        return act

    if kind == "work_unit_plan":
        unit = event.get("unit_id")
        if unit:
            act["planned_units"].add(str(unit))
        act["last_activity"] = (
            f"Planning unit {unit}" if unit else "Planning work units"
        )
        return act

    if kind == "work_unit_attempt":
        act["last_activity"] = f"Generating unit {event.get('unit_id') or ''}".strip()
        return act

    if kind == "work_unit_done":
        unit = event.get("unit_id")
        if unit:
            act["completed_units"].add(str(unit))
        act["last_activity"] = f"Completed unit {unit}" if unit else "Unit completed"
        return act

    return act


_FAILURE_SENTENCE_LIMIT = 400
_FAILURE_TEXT_KEYS = ("errors", "offenders", "message", "error")
# Mapping-opening character + mapping-punctuation: a dump, not a log line. A
# bracketed sentence ("[3/5] verify: clean") is ordinary activity text.
_DUMP_MARKERS = ("': ", '": ', "', '", '", "')


def _mapping_text(value) -> str | None:
    """The user-readable sentences a machine mapping carries, or None.

    Pipeline stdout can print a result mapping (``{'ok': False, 'errors': [...]}``,
    with nested ``offenders``). Those inner sentences ARE the cause; the mapping
    is only its envelope. The first field that says something wins, and a mapping
    that says nothing yields None."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in _FAILURE_TEXT_KEYS:
            if key in value:
                text = _mapping_text(value[key])
                if text:
                    return text
        return None
    if isinstance(value, (list, tuple)):
        sentences = [s for s in (_mapping_text(item) for item in value) if s]
        return "; ".join(sentences) or None
    return None


def _parse_mapping(text: str):
    """Parse a Python/JSON mapping repr, or None when it is not one."""
    if not text.startswith("{"):
        return None
    for parse in (ast.literal_eval, json.loads):
        try:
            parsed = parse(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_arrow(text: str) -> str:
    """Drop one leading pipeline marker ('->', '=>', '-', '=') and trim."""
    for prefix in ("->", "=>", "-", "="):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


# A driver's own status prefixes, and a Traceback: never the cause itself.
_NOISE_PREFIXES = ("[legacy]", "[ok  ]", "[FAIL]", "[err", "[warn", "Traceback")
# A printed value with no prose in it ("-> None"): not a sentence.
_BARE_LITERALS = ("None", "True", "False", "null", "{}", "[]")


def _is_dump_or_noise(body: str) -> bool:
    """Whether a line is machine output that must never be quoted as a cause."""
    if not body:
        return True
    if body in _BARE_LITERALS:
        return True
    if body.startswith(_NOISE_PREFIXES):
        return True
    if body.startswith(("{", "[")) and any(m in body for m in _DUMP_MARKERS):
        return True
    return False


def human_failure_text(value) -> str | None:
    """A user-quotable failure sentence from a run's raw text, or None.

    This is the ONE place that decides whether a run's last words may be shown
    to a person. A design driver's stdout can leave a machine repr on the
    activity line (``-> {'ok': False, 'errors': ['9.11 net coverage: ...']}``) --
    quoting that repr is what made a live failure read as unreadable Python. The
    sentences inside it are lifted out instead, and a dump with nothing to say is
    refused rather than displayed. Ordinary prose passes through unchanged, so a
    structured statement (``Wiring failed``) is untouched."""
    if value is None:
        return None
    lines = [" ".join(line.split()) for line in str(value).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    # A mapping on any line says the most -- even inside a multi-line log block.
    for line in lines:
        parsed = _parse_mapping(_strip_arrow(line))
        if parsed is not None:
            sentences = _mapping_text(parsed)
            if sentences:
                return sentences[:_FAILURE_SENTENCE_LIMIT]
    for line in lines:
        body = _strip_arrow(line)
        if not _is_dump_or_noise(body):
            return body[:_FAILURE_SENTENCE_LIMIT]
    return None


def _failure_fact(*, kind, stage, retryable=None, retry_action=None, message=None,
                  exception_type=None) -> dict:
    fact = {
        "kind": str(kind) if kind else None,
        "stage": stage,
        "retryable": bool(retryable),
        "retry_action": retry_action,
        "message": message,
        "exception_type": exception_type,
    }
    return fact


def _add_issue(act: dict, issue: dict) -> None:
    """Append an issue, deduplicated by stage+code+evidence (newest wins)."""
    issue = {
        "severity": issue.get("severity") if issue.get("severity") in _ISSUE_SEVERITIES
        else "warning",
        "stage": issue.get("stage"),
        "code": str(issue.get("code") or ""),
        "message": str(issue.get("message") or ""),
        "evidence": [str(item) for item in (issue.get("evidence") or [])],
    }
    key = (issue["stage"], issue["code"], tuple(issue["evidence"]))
    issues = act["issues"]
    for i, existing in enumerate(issues):
        if (existing.get("stage"), existing.get("code"),
                tuple(existing.get("evidence") or [])) == key:
            issues[i] = issue
            return
    issues.append(issue)
    if len(issues) > _MAX_ISSUES:
        del issues[0:len(issues) - _MAX_ISSUES]


def _evidence_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("message") or item.get("error") or json.dumps(
                    item, default=str)
            else:
                text = item
            out.append(str(text))
        return out
    return [str(value)]


def _queue_sentence(position, eta_s) -> str:
    try:
        ahead = int(position or 0)
    except (TypeError, ValueError):
        ahead = 0
    text = (
        "Queued for board build: next up"
        if ahead <= 0
        else f"Queued for board build: {ahead} ahead"
    )
    if isinstance(eta_s, (int, float)) and eta_s > 0:
        text += f" · est. ~{max(1, round(eta_s / 60))} min"
    return text


def _humanize_kind(kind: str) -> str:
    return str(kind or "").replace("_", " ").strip()


def _tool_call(act: dict, event: dict) -> None:
    name = str(event.get("name") or "tool")
    call_id = str(event.get("call_id") or "")
    args = event.get("args")
    entry = {
        "call_id": call_id,
        "name": name,
        "args": args if isinstance(args, dict) else {},
        "status": "running",
        "started_at": event.get("ts"),
        "duration_ms": None,
        "cached": False,
        "ok": None,
        "output": None,
        "output_chars": None,
        "output_truncated": False,
    }
    tools = act["tools"]
    tools.append(entry)
    if len(tools) > _MAX_TOOLS:
        del tools[0:len(tools) - _MAX_TOOLS]


def _tool_result(act: dict, event: dict) -> str | None:
    """Pair a result with its call; returns the resolved tool name (or None).

    Pairing key is the local ``call_id`` minted by the client. A legacy result
    with no id pairs to the oldest unmatched call of the same name in this run.
    """
    name = str(event.get("name") or "")
    call_id = str(event.get("call_id") or "")
    ok = event.get("ok")
    tools = act["tools"]
    match = None
    if call_id:
        for entry in tools:
            if entry.get("call_id") == call_id and entry.get("status") == "running":
                match = entry
                break
        if match is None:
            for entry in reversed(tools):
                if entry.get("call_id") == call_id:
                    match = entry
                    break
    if match is None and not call_id:
        for entry in tools:
            if entry.get("status") == "running" and (not name or entry.get("name") == name):
                match = entry
                break
    if match is None:
        # Orphan result (its call predates the durable window): surface it as its
        # own named card instead of dropping the evidence.
        entry = {
            "call_id": call_id,
            "name": name or "tool",
            "args": {},
            "status": "returned",
            "started_at": None,
            "duration_ms": event.get("duration_ms"),
            "cached": bool(event.get("cached")),
            "ok": ok if isinstance(ok, bool) else None,
            "output": event.get("output"),
            "output_chars": event.get("output_chars"),
            "output_truncated": bool(event.get("output_truncated")),
            "orphan": True,
        }
        tools.append(entry)
        if len(tools) > _MAX_TOOLS:
            del tools[0:len(tools) - _MAX_TOOLS]
        return str(event.get("name") or "") or None
    match["duration_ms"] = event.get("duration_ms")
    match["cached"] = bool(event.get("cached"))
    match["ok"] = ok if isinstance(ok, bool) else None
    match["output"] = event.get("output")
    match["output_chars"] = event.get("output_chars")
    match["output_truncated"] = bool(event.get("output_truncated"))
    match["status"] = "failed" if ok is False else "returned"
    return match.get("name")


# --------------------------------------------------------------------------- #
# Attempt grouping
# --------------------------------------------------------------------------- #

def _event_sort_key(event: dict, index: int):
    seq = event.get("seq")
    if isinstance(seq, int):
        return (0, seq, index)
    derived = seq_from_event_id(event.get("event_id"))
    if derived is not None:
        return (0, derived, index)
    return (1, 0, index)


def seq_from_event_id(event_id) -> int | None:
    if not isinstance(event_id, str) or ":" not in event_id:
        return None
    tail = event_id.rsplit(":", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return None


def _first_ts(events: list[dict]) -> str:
    for event in events:
        ts = event.get("ts")
        if isinstance(ts, str) and ts:
            return ts
    return ""


def group_attempts(events) -> list[dict]:
    """Split a flat, ordered event history into attempts.

    Returns ``[{"run_id", "legacy", "events", "started_at"}]``, oldest attempt
    first, metadata-free legacy rows as one labeled group ahead of them. Each
    attempt's events are ordered by numeric ``seq`` (timestamp fallback for the
    older provenance rows that predate ``seq``). The current attempt is the last
    group; everything before it is history and must not be able to overwrite the
    current outcome.
    """
    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        run_id = event.get("run_id")
        key = str(run_id) if run_id else ""
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(event)

    groups: list[dict] = []
    for key in order:
        rows = buckets[key]
        if not key:
            groups.append(
                {
                    "run_id": None,
                    "legacy": True,
                    "events": list(rows),
                    "started_at": "",
                }
            )
            continue
        ordered = sorted(
            enumerate(rows), key=lambda pair: _event_sort_key(pair[1], pair[0])
        )
        events_out = [event for _i, event in ordered]
        groups.append(
            {
                "run_id": key,
                "legacy": False,
                "events": events_out,
                "started_at": _first_ts(events_out),
            }
        )

    # Legacy history first (it predates the versioned attempts), then attempts
    # oldest-first. A versioned group that carries no timestamp keeps its
    # discovery order rather than jumping the queue.
    legacy = [g for g in groups if g["legacy"]]
    versioned = [g for g in groups if not g["legacy"]]
    versioned.sort(key=lambda g: (g["started_at"] == "", g["started_at"]))
    return legacy + versioned


def current_attempt(groups: list[dict]) -> dict | None:
    """The attempt the UI shows by default (the latest one)."""
    return groups[-1] if groups else None


# --------------------------------------------------------------------------- #
# Sanitization (persisted + user-visible technical text)
# --------------------------------------------------------------------------- #

MAX_TECHNICAL_TEXT = 16384

_CREDENTIAL_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "x_api_key",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "cookie",
        "set_cookie",
    }
)
_DROP_KEYS = frozenset(
    {"request_headers", "response_headers", "headers", "body", "response_body",
     "request_body"}
)

_REDACTED = "[redacted]"
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}")
_URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/@\s]+@")
_URL_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api_?key|access_?token|refresh_?token|token|password|secret|"
    r"signature|sig|auth|code)=)[^&#\s]*"
)
# A tool's output is usually TEXT, and text can contain a JSON/INI-style
# credential pair. Redact the value of any creditor-named key wherever it appears
# (the recursive pass below only sees real dict keys).
_TEXT_CREDENTIAL_RE = re.compile(
    r"(?i)(\"?(?:api_?key|apikey|x_?api_?key|access_?token|refresh_?token|"
    r"id_?token|password|passwd|secret|client_?secret|authorization|cookie|"
    r"set_?cookie)\"?\s*[:=]\s*)(\"?)([^\"'&,;\s}]+)"
)


def _normalized_key(key) -> str:
    return str(key).strip().lower().replace("-", "_")


def _scrub_text(value: str, workspace: str | None) -> str:
    if not value:
        return value
    value = _BEARER_RE.sub(f"Bearer {_REDACTED}", value)
    value = _TEXT_CREDENTIAL_RE.sub(rf"\1\2{_REDACTED}", value)
    if "://" in value:
        value = _URL_USERINFO_RE.sub(r"\1", value)
        value = _URL_QUERY_SECRET_RE.sub(rf"\1{_REDACTED}", value)
    if workspace and workspace in value:
        value = value.replace(workspace, "<project>")
    if len(value) > MAX_TECHNICAL_TEXT:
        dropped = len(value) - MAX_TECHNICAL_TEXT
        value = value[:MAX_TECHNICAL_TEXT] + f"\n…[truncated {dropped} characters]"
    return value


def _sanitize_value(value, workspace: str | None, key_hint: str | None):
    if isinstance(value, str):
        return _scrub_text(value, workspace)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            norm = _normalized_key(key)
            if norm in _CREDENTIAL_KEYS or norm in _DROP_KEYS:
                out[str(key)] = _REDACTED
                continue
            out[str(key)] = _sanitize_value(item, workspace, norm)
        return out
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, workspace, key_hint) for item in value]
    return value


def sanitize_activity_event(event: dict, *, workspace: str | None = None) -> dict:
    """A copy of `event` safe to persist and to show a user.

    Recursively redacts values under credential keys, scrubs Bearer credentials
    and the credential/query components of URLs, replaces the workspace prefix
    with ``<project>``, and bounds technical text to
    :data:`MAX_TECHNICAL_TEXT` characters with an explicit truncation marker.

    ``run_error`` carries no response body or request headers by construction --
    the failure is described by its classification plus the exception type -- so
    any such payload that reached this point is dropped rather than persisted.

    This never changes what a model sees or what a tool returns: the model
    message policy and the executor's string API are untouched. Render the
    result as text or escaped JSON, never as raw HTML.
    """
    if not isinstance(event, dict):
        return {}
    out = dict(event)
    for key in list(out.keys()):
        norm = _normalized_key(key)
        if norm in _CREDENTIAL_KEYS or norm in _DROP_KEYS:
            out[key] = _REDACTED
            continue
        out[key] = _sanitize_value(out[key], workspace, norm)
    return out
