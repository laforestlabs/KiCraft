"""Run one legacy design session behind the current tree's process boundary.

This file is executed by the legacy interpreter with the legacy checkout first on
``PYTHONPATH``.  It deliberately imports only legacy-owned APIs: the current tree
passes a JSON request over stdin and consumes the marked JSON response on stdout.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import os
import sqlite3
import sys
import types
from pathlib import Path

from kicraft.server import session
from kicraft.server import stage_driver
from kicraft.server.spend_guard import BudgetExceeded, KillSwitchEngaged
from kicraft.server.stage_driver import make_budget_client

# This runner is executed as a script by the pinned interpreter.  Its directory is
# consequently on sys.path, while the current package must not be imported there.
from question_policy import QUESTION_OPTIONS_INSTRUCTION, normalize_question_options


RESULT_PREFIX = "__KICRAFT_LEGACY_SESSION_RESULT__="
EVENT_PREFIX = "__KICRAFT_LEGACY_EVENT__="

_STRICT_BUDGET_ENV = "KICRAFT_EVAL_STRICT_BUDGET"


def _strict_budget_enabled() -> bool:
    return os.environ.get(_STRICT_BUDGET_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_today_start() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")


class _StrictNativeBudgetGuard:
    """Reserve each fully bounded native POST before it is dispatched.

    The pinned client calls its guard without an estimate, so its otherwise
    correct post-call ledger accounting cannot prevent one completion from
    crossing a cap.  This current-owned adapter is deliberately opt-in for
    evaluation: production retains the pinned client's behavior unchanged.

    ``strict_budget_exposure`` is not spend.  It is a durable upper-bound
    reservation written before an outbound request, retained as ``uncertain``
    when the native client loses a stream before trustworthy accounting, and
    included in later admission checks alongside actual ``spend`` rows.
    """

    def __init__(self, guard, settings, *, run_id: str, run_budget_usd: float):
        self._guard = guard
        self._settings = settings
        self._run_id = run_id
        self._run_budget_usd = self._positive_limit(run_budget_usd)
        self._project_budget_usd = self._positive_limit(
            os.environ.get("KICRAFT_PROJECT_LLM_BUDGET_USD", "")
        )
        self._ledger_path = Path(guard.path)
        self._lock_path = self._ledger_path.with_suffix(self._ledger_path.suffix + ".strict.lock")
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_exposure_table()
        self._lock_file = None
        self._body = None
        self._call_ceiling = None
        self._reservation_id = None
        self._network_attempted = False

    @staticmethod
    def _positive_limit(value) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) and numeric > 0.0 else None

    @staticmethod
    def _bounded_int(value) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric > 0 and str(numeric) == str(value).strip() else None

    def _init_exposure_table(self) -> None:
        with sqlite3.connect(self._ledger_path, timeout=30) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS strict_budget_exposure ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "created_at TEXT NOT NULL,"
                "run_id TEXT NOT NULL,"
                "ceiling_usd REAL NOT NULL,"
                "state TEXT NOT NULL,"
                "settled_at TEXT)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS strict_budget_exposure_active "
                "ON strict_budget_exposure(state, run_id, created_at)"
            )

    def begin_call(self, body: dict) -> None:
        self._body = body

    def _now(self) -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _finish_lock(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _set_state(self, state: str) -> None:
        if self._reservation_id is None:
            return
        with sqlite3.connect(self._ledger_path, timeout=30) as conn:
            conn.execute(
                "UPDATE strict_budget_exposure SET state=?, settled_at=? WHERE id=?",
                (
                    state,
                    self._now() if state in {"settled", "cancelled"} else None,
                    self._reservation_id,
                ),
            )

    def abort_call(self) -> None:
        """Cancel pre-dispatch state; retain every POST without a verified receipt."""
        try:
            if self._reservation_id is not None:
                if self._network_attempted:
                    self._set_state("uncertain")
                else:
                    self._set_state("cancelled")
        except Exception:
            # If cleanup itself fails, leave the durable reservation active. That
            # is conservative and prevents an accounting outage from widening spend.
            pass
        finally:
            self._body = None
            self._call_ceiling = None
            self._reservation_id = None
            self._network_attempted = False
            self._finish_lock()

    def _reject(self, message: str) -> None:
        raise BudgetExceeded(message)

    def _call_ceiling_usd(self) -> float:
        body = self._body
        if not isinstance(body, dict):
            self._reject("strict budget has no bounded completion context; refusing dispatch.")
        model = body.get("model") or getattr(self._settings, "model", "")
        if not isinstance(model, str) or not model.strip():
            self._reject("strict budget cannot identify the completion model; refusing dispatch.")
        output_tokens = self._bounded_int(body.get("max_tokens"))
        if output_tokens is None:
            self._reject(
                "strict budget requires an explicit finite positive max_tokens; refusing dispatch."
            )
        reasoning = body.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                self._reject(
                    "strict budget cannot bound non-object reasoning policy; refusing dispatch."
                )
            if reasoning.get("enabled") is not False:
                reasoning_tokens = self._bounded_int(reasoning.get("max_tokens"))
                if reasoning_tokens is None:
                    self._reject(
                        "strict budget requires a finite reasoning max_tokens; refusing dispatch."
                    )
                output_tokens += reasoning_tokens
        prompt_price = self._positive_limit(getattr(self._settings, "max_price_prompt", None))
        completion_price = self._positive_limit(
            getattr(self._settings, "max_price_completion", None)
        )
        if prompt_price is None or completion_price is None:
            self._reject(
                "strict budget requires positive provider price ceilings; refusing dispatch."
            )
        # Serialize every public request field (including tool schemas), then use
        # its UTF-8 byte count as a tokenizer-independent upper bound.  This is
        # intentionally stricter than the current client's chars/4 estimate.
        request = {key: value for key, value in body.items() if not key.startswith("_")}
        # The serialized byte count above bounds prompt cost for any provider
        # field, so unknown fields are admissible.  What it cannot bound is a
        # field that multiplies billed completions: those are refused explicitly
        # rather than by enumerating every legitimate provider field, because a
        # missing name in such an allow-list silently breaks a real stage (the
        # tool-loop's `parallel_tool_calls` did exactly that).
        for field in ("n", "best_of"):
            multiple = request.get(field)
            if multiple is not None and multiple != 1:
                self._reject(
                    f"strict budget cannot bound {field}={multiple!r} completions; refusing dispatch."
                )
        messages = request.get("messages")
        if not isinstance(messages, list):
            self._reject("strict budget requires a text message list; refusing dispatch.")
        for message in messages:
            if not isinstance(message, dict):
                self._reject("strict budget cannot bound a non-object message; refusing dispatch.")
            content = message.get("content")
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, list) or any(
                not isinstance(part, dict)
                or part.get("type") != "text"
                or not isinstance(part.get("text"), str)
                for part in content
            ):
                self._reject("strict budget cannot bound multimodal input; refusing dispatch.")
        request.setdefault("model", model)
        request.setdefault("max_tokens", output_tokens)
        try:
            prompt_tokens = max(
                1,
                len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
            )
        except (TypeError, ValueError):
            self._reject("strict budget cannot serialize completion request; refusing dispatch.")
        ceiling = (prompt_tokens * prompt_price + output_tokens * completion_price) / 1_000_000
        if not math.isfinite(ceiling) or ceiling <= 0.0:
            self._reject("strict budget cannot compute a finite call ceiling; refusing dispatch.")
        return ceiling

    @staticmethod
    def _spend(conn: sqlite3.Connection, where: str = "", params: tuple = ()) -> float:
        row = conn.execute(
            f"SELECT COALESCE(SUM(cost_usd), 0) FROM spend {where}", params
        ).fetchone()
        return float(row[0] or 0.0)

    @staticmethod
    def _exposure(conn: sqlite3.Connection, where: str = "", params: tuple = ()) -> float:
        row = conn.execute(
            "SELECT COALESCE(SUM(ceiling_usd), 0) FROM strict_budget_exposure "
            "WHERE state IN ('reserved','uncertain')" + (f" AND {where}" if where else ""),
            params,
        ).fetchone()
        return float(row[0] or 0.0)

    def _reconcile_recorded_reservations(self, conn: sqlite3.Connection) -> None:
        """Recover a process death after native record() but before settlement."""
        conn.execute(
            "UPDATE strict_budget_exposure SET state='settled', settled_at=? "
            "WHERE state IN ('reserved','uncertain') AND EXISTS ("
            "SELECT 1 FROM spend WHERE json_valid(meta) AND "
            "json_extract(meta, '$.strict_budget_usage_verified')=1 AND "
            "CAST(json_extract(meta, '$.strict_budget_reservation_id') AS TEXT)="
            "CAST(strict_budget_exposure.id AS TEXT))",
            (self._now(),),
        )

    def _reserve_attempt(self) -> None:
        reserve = self._call_ceiling
        if reserve is None:
            self._reject("strict budget saw native network dispatch without admission; refusing.")
        with sqlite3.connect(self._ledger_path, timeout=30) as conn:
            self._reconcile_recorded_reservations(conn)
            total = self._spend(conn) + self._exposure(conn)
            total_limit = self._positive_limit(getattr(self._settings, "total_usd_ceiling", None))
            daily_limit = self._positive_limit(getattr(self._settings, "daily_usd_ceiling", None))
            if total_limit is None or daily_limit is None:
                self._reject(
                    "strict budget requires finite positive global ceilings; refusing dispatch."
                )
            if total + reserve > total_limit:
                self._reject(
                    f"total remaining budget cannot cover call ceiling ${reserve:.4f} "
                    f"(actual plus reserved exposure ${total:.4f} of ${total_limit:.2f}); refusing."
                )
            today = _utc_today_start()
            day = self._spend(conn, "WHERE ts >= ?", (today,)) + self._exposure(
                conn, "created_at >= ?", (today,)
            )
            if day + reserve > daily_limit:
                self._reject(
                    f"daily remaining budget cannot cover call ceiling ${reserve:.4f} "
                    f"(actual plus reserved exposure ${day:.4f} of ${daily_limit:.2f}); refusing."
                )
            run_spend = self._spend(
                conn,
                "WHERE json_valid(meta) AND json_extract(meta, '$.run_id') = ?",
                (self._run_id,),
            ) + self._exposure(conn, "run_id = ?", (self._run_id,))
            for label, limit in (
                ("run", self._run_budget_usd),
                ("project", self._project_budget_usd),
            ):
                if limit is not None and run_spend + reserve > limit:
                    self._reject(
                        f"{label} run {self._run_id} remaining budget cannot cover call "
                        f"ceiling ${reserve:.4f} (actual plus reserved exposure "
                        f"${run_spend:.4f} of ${limit:.2f}); refusing."
                    )
            cursor = conn.execute(
                "INSERT INTO strict_budget_exposure "
                "(created_at,run_id,ceiling_usd,state,settled_at) VALUES (?,?,?,?,NULL)",
                (self._now(), self._run_id, reserve, "reserved"),
            )
            self._reservation_id = int(cursor.lastrowid)
            meta_ctx = self._body.setdefault("_meta_ctx", {})
            meta_ctx["run_id"] = self._run_id
            meta_ctx["strict_budget_reservation_id"] = str(self._reservation_id)

    def preflight(self) -> None:
        """Validate one bounded completion and lock its native record lifecycle."""
        if self._lock_file is not None:
            self._reject("strict budget received nested completion dispatch; refusing dispatch.")
        self._lock_file = self._lock_path.open("a+")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if bool(getattr(self._settings, "kill_switch", False)):
                raise KillSwitchEngaged("KICRAFT_KILL_SWITCH is engaged; refusing all model calls.")
            self._call_ceiling = self._call_ceiling_usd()
        except Exception:
            self.abort_call()
            raise

    def note_network_attempt(self) -> None:
        """Reserve exactly one ceiling immediately before each native POST."""
        if self._call_ceiling is None:
            self._reject("strict budget saw native network dispatch without admission; refusing.")
        if self._reservation_id is not None:
            # A prior POST had no verified usage receipt.  It remains uncertain
            # even if the native retry eventually returns a clean completion.
            self._set_state("uncertain")
            self._reservation_id = None
            self._network_attempted = False
        self._reserve_attempt()
        self._network_attempted = True

    @staticmethod
    def _record_meta(meta, reservation_id: int | None):
        if not isinstance(meta, dict):
            return meta
        return {
            **meta,
            "strict_budget_reservation_id": str(reservation_id) if reservation_id else None,
        }

    def record(self, *args, **kwargs):
        """Preserve native recording; an estimated cost is not a usage receipt."""
        if "meta" in kwargs:
            kwargs = dict(kwargs)
            kwargs["meta"] = self._record_meta(kwargs["meta"], self._reservation_id)
        elif len(args) >= 5:
            args = list(args)
            args[4] = self._record_meta(args[4], self._reservation_id)
        return self._guard.record(*args, **kwargs)

    def settle_receipt(self, message: dict) -> None:
        usage = message.get("usage") if isinstance(message, dict) else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if self._reservation_id is None or self._positive_limit(cost) is None:
            return
        with sqlite3.connect(self._ledger_path, timeout=30) as conn:
            updated = conn.execute(
                "UPDATE spend SET meta=json_set(meta, '$.strict_budget_usage_verified', 1) "
                "WHERE json_valid(meta) AND json_extract(meta, '$.run_id')=? "
                "AND CAST(json_extract(meta, '$.strict_budget_reservation_id') AS TEXT)=?",
                (self._run_id, str(self._reservation_id)),
            ).rowcount
        if updated:
            self._set_state("settled")
            self._reservation_id = None

    def __getattr__(self, name):
        return getattr(self._guard, name)


class _PostAdmittingRequests:
    """Per-call proxy; the pinned module's shared ``requests`` object is untouched."""

    def __init__(self, requests_module, admit):
        self._requests_module = requests_module
        self._admit = admit

    def post(self, *args, **kwargs):
        self._admit()
        return self._requests_module.post(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._requests_module, name)


def _post_admitted_open_stream(native_open_stream, admit):
    """Bind a private copy of native ``_open_stream`` with POST admission injected."""
    native_function = getattr(native_open_stream, "__func__", None)
    native_self = getattr(native_open_stream, "__self__", None)
    native_globals = getattr(native_function, "__globals__", None)
    requests_module = native_globals.get("requests") if isinstance(native_globals, dict) else None
    if (
        native_function is None
        or native_self is None
        or not callable(getattr(requests_module, "post", None))
    ):
        raise BudgetExceeded(
            "strict budget requires the native requests.post stream boundary; refusing dispatch."
        )
    private_globals = dict(native_globals)
    private_globals["requests"] = _PostAdmittingRequests(requests_module, admit)
    private_function = types.FunctionType(
        native_function.__code__,
        private_globals,
        native_function.__name__,
        native_function.__defaults__,
        native_function.__closure__,
    )
    private_function.__kwdefaults__ = native_function.__kwdefaults__
    return types.MethodType(private_function, native_self)


def _enable_strict_native_budget(client, *, run_id, budget_usd: float):
    """Instrument native POSTs without changing its retry or provider policy."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise BudgetExceeded("strict budget requires an exact run_id; refusing native dispatch.")
    strict_guard = _StrictNativeBudgetGuard(
        client.guard, client.s, run_id=run_id, run_budget_usd=budget_usd
    )
    native_stream = client._stream
    native_open_stream = getattr(client, "_open_stream", None)
    if not callable(native_open_stream):
        raise BudgetExceeded(
            "strict budget requires the native _open_stream boundary; refusing dispatch."
        )
    admitted_open_stream = _post_admitted_open_stream(
        native_open_stream, strict_guard.note_network_attempt
    )

    def guarded_open_stream(payload):
        return admitted_open_stream(payload)

    def guarded_stream(body, on_delta=None):
        prepared = dict(body)
        meta_ctx = body.get("_meta_ctx")
        prepared["_meta_ctx"] = {
            **(meta_ctx if isinstance(meta_ctx, dict) else {}),
            "run_id": run_id,
        }
        strict_guard.begin_call(prepared)
        try:
            message, cost = native_stream(prepared, on_delta=on_delta)
            strict_guard.settle_receipt(message)
            return message, cost
        finally:
            strict_guard.abort_call()

    client.guard = strict_guard
    client._open_stream = guarded_open_stream
    client._stream = guarded_stream
    return client


def emit_event(event):
    print(EVENT_PREFIX + json.dumps(event, separators=(",", ":"), default=str), flush=True)


_AUTO_POLICY_INSTRUCTION = (
    "Clarification policy: use recommended sensible defaults for ordinary user "
    "ambiguities. Do not leave ordinary user questions blocking; record defaults "
    "in assumptions and continue to a slot. Internal BOM reconciliation questions "
    "remain blocking."
)
_INTERACTIVE_POLICY_INSTRUCTION = (
    "Clarification policy: ordinary user ambiguities must remain blocking until "
    "the user answers. Do not silently default them, including after earlier "
    "answers or an ordinary re-drive instruction. Internal BOM reconciliation "
    "questions remain pipeline control."
)


class _UserQuestionPause(Exception):
    """Runner-local control flow for an explicit interactive user question."""

    def __init__(self, stage: str, questions: list[dict]):
        super().__init__(stage)
        self.stage = stage
        self.questions = questions


class _InvalidClarification(Exception):
    """A model returned an ordinary interactive question without real choices."""

    def __init__(self, stage: str, diagnostic: str):
        super().__init__(diagnostic)
        self.stage = stage
        self.diagnostic = diagnostic


class _QuestionPolicyClient:
    """Decorate the one native budget client with the web question contract.

    The native driver owns retries, state writes, tool conversations and its spend
    guard.  This wrapper only changes a parseable question payload immediately
    after the native client returns it, so it neither creates a client nor changes
    the driver's bounded retry structure.
    """

    def __init__(self, client, auto_default_questions: bool):
        self._client = client
        self.auto_default_questions = auto_default_questions
        self._stage_stats: dict[str, dict] = {}
        self._completed_rows: list[dict] = []

    @property
    def s(self):
        return self._client.s

    @property
    def guard(self):
        return self._client.guard

    def __getattr__(self, name):
        return getattr(self._client, name)

    @staticmethod
    def _stage_from(kwargs) -> str | None:
        meta_ctx = kwargs.get("meta_ctx")
        if isinstance(meta_ctx, dict):
            stage = meta_ctx.get("stage")
            if isinstance(stage, str) and stage:
                return stage
        return None

    def _append_policy_once(self, messages) -> None:
        instruction = (
            QUESTION_OPTIONS_INSTRUCTION
            + "\n\n"
            + (
                _AUTO_POLICY_INSTRUCTION
                if self.auto_default_questions
                else _INTERACTIVE_POLICY_INSTRUCTION
            )
        )
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content")
                content = content if isinstance(content, str) else ""
                if instruction not in content:
                    message["content"] = (
                        content.rstrip() + "\n\n" + instruction if content else instruction
                    )
                return
        messages.insert(0, {"role": "system", "content": instruction})

    def _record_response(self, stage: str | None, response) -> None:
        if not stage or not isinstance(response, dict):
            return
        stats = self._stage_stats.setdefault(
            stage, {"cost_usd": 0.0, "attempts": 0, "rounds": None, "tool_calls": None}
        )
        stats["attempts"] += 1
        try:
            stats["cost_usd"] += float(response.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            pass
        if response.get("rounds") is not None:
            stats["rounds"] = response.get("rounds")
        if response.get("tool_calls") is not None:
            stats["tool_calls"] = response.get("tool_calls")

    def _rewrite_candidate(self, response: dict, key: str, payload: dict) -> dict:
        # The driver accepts the same JSON with or without fences.  Re-serializing
        # only the candidate preserves all provider metadata/cost fields.
        response[key] = json.dumps(payload, separators=(",", ":"))
        return response

    def _apply_question_policy(self, response, stage: str | None, *, reasoning_fallback=True):
        if not stage or not isinstance(response, dict):
            return response
        key = "reasoning" if reasoning_fallback and not response.get("text") else "text"
        candidate = response.get(key) or ""
        if not isinstance(candidate, str):
            return response
        try:
            payload = stage_driver._extract_json(candidate)
        except (json.JSONDecodeError, ValueError):
            # The native driver's serialization/parse recovery owns non-JSON.
            return response
        if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
            return response
        raw_questions = payload["questions"]
        if not raw_questions:
            return response

        # A BOM reconcile request is internal control.  In automatic mode the
        # ordinary questions beside it still become nonblocking; the reconcile
        # item itself remains blocking and therefore retains precedence in the
        # native driver.
        has_reconcile = any(
            isinstance(question, dict) and question.get("reconcile_target") == "bom"
            for question in raw_questions
        )
        ordinary = [
            question
            for question in raw_questions
            if (
                isinstance(question, dict)
                and str(question.get("text", "")).strip()
                and question.get("reconcile_target") != "bom"
            )
        ]
        if not ordinary:
            return response

        if self.auto_default_questions:
            for question in ordinary:
                question["blocking"] = False
            return self._rewrite_candidate(response, key, payload)

        # Internal reconciliation wins over an ordinary question in the same
        # provider payload and follows the legacy reconciliation loop unchanged.
        if has_reconcile:
            return response

        # Validate before native normalization, whose legacy schema otherwise
        # silently accepts freeform-only options.  The native normalizer supplies
        # the durable Question shape and actual invocation stage.
        for question in ordinary:
            try:
                question["options"] = normalize_question_options(question.get("options"))
            except ValueError as exc:
                raise _InvalidClarification(stage, str(exc)) from exc
        normalized = stage_driver._normalize_questions(raw_questions, stage)[:3]
        blocking = [
            question
            for question in normalized
            if question.get("blocking") and question.get("reconcile_target") != "bom"
        ]
        if blocking:
            raise _UserQuestionPause(stage, normalized)
        return self._rewrite_candidate(response, key, payload)

    def chat(self, messages, **kwargs):
        self._append_policy_once(messages)
        stage = self._stage_from(kwargs)
        response = self._client.chat(messages, **kwargs)
        self._record_response(stage, response)
        return self._apply_question_policy(response, stage)

    def chat_with_tools(self, messages, tools, executor, **kwargs):
        self._append_policy_once(messages)
        stage = self._stage_from(kwargs)
        response = self._client.chat_with_tools(messages, tools, executor, **kwargs)
        self._record_response(stage, response)
        return self._apply_question_policy(response, stage, reasoning_fallback=False)

    def observe_progress(self, event) -> None:
        if not isinstance(event, dict):
            return
        if event.get("kind") == "stage_start":
            self._stage_stats.pop(event.get("stage"), None)
            return
        if event.get("kind") != "stage_done":
            return
        stage = event.get("stage")
        if not isinstance(stage, str) or not stage:
            return
        stats = self._stage_stats.get(stage, {})
        row = {
            "stage": stage,
            "commit_ok": bool(event.get("ok")),
            "cost_usd": event.get("cost", stats.get("cost_usd", 0.0)),
            "attempts": event.get("attempts", stats.get("attempts")),
        }
        if event.get("error") is not None:
            row["error"] = event["error"]
        if event.get("failure_kind") is not None:
            row["failure_kind"] = event["failure_kind"]
        self._completed_rows.append(row)

    def stage_stats(self, stage: str) -> dict:
        return dict(self._stage_stats.get(stage, {}))

    def completed_rows(self) -> list[dict]:
        return [dict(row) for row in self._completed_rows]


def _state_path(workspace: Path) -> Path:
    # Let the pinned session own its storage layout instead of duplicating it.
    return session._state_path(workspace)


def _pause_result(
    workspace: Path, client: _QuestionPolicyClient, pause: _UserQuestionPause
) -> dict:
    stage_driver._attach_questions(_state_path(workspace), pause.stage, pause.questions)
    emit_event({"kind": "question", "stage": pause.stage, "questions": pause.questions})
    stats = client.stage_stats(pause.stage)
    parked = {
        "stage": pause.stage,
        "commit_ok": False,
        "needs_input": True,
        "questions": pause.questions,
        "cost_usd": stats.get("cost_usd", 0.0),
        "attempts": stats.get("attempts", 0),
    }
    for key in ("rounds", "tool_calls"):
        if stats.get(key) is not None:
            parked[key] = stats[key]
    return {
        "status": "awaiting_input",
        "results": client.completed_rows() + [parked],
        "guard": client.guard.status(),
        "state_path": str(_state_path(workspace)),
        "questions": pause.questions,
        "last_stage": pause.stage,
    }


def _invalid_result(
    workspace: Path, client: _QuestionPolicyClient, invalid: _InvalidClarification
) -> dict:
    stats = client.stage_stats(invalid.stage)
    stage_driver._stamp_stage_status(
        _state_path(workspace),
        invalid.stage,
        False,
        cost_usd=stats.get("cost_usd", 0.0),
        attempts=stats.get("attempts", 0),
        rounds=stats.get("rounds"),
        tool_calls=stats.get("tool_calls"),
        error=invalid.diagnostic,
        failure_kind="invalid_clarification",
    )
    event = {
        "kind": "stage_done",
        "stage": invalid.stage,
        "ok": False,
        "cost": stats.get("cost_usd", 0.0),
        "attempts": stats.get("attempts", 0),
        "error": invalid.diagnostic,
        "failure_kind": "invalid_clarification",
    }
    client.observe_progress(event)
    emit_event(event)
    return {
        "status": "failed",
        "results": client.completed_rows(),
        "guard": client.guard.status(),
        "state_path": str(_state_path(workspace)),
        "questions": None,
        "last_stage": invalid.stage,
        "error": invalid.diagnostic,
        "failure_kind": "invalid_clarification",
    }


def _guard_refusal_result(
    workspace: Path, client, stage: str | None, failure_kind: str, error: str
) -> dict:
    """Report an evaluation-guard refusal as a stage failure, not a broken protocol.

    ``stage_driver`` deliberately lets ``BudgetExceeded`` escape to the caller so
    a cap cannot be silently swallowed.  Returning a real result packet here is
    that caller path: without it the process dies before printing its result and
    the harness records a meaningless protocol error instead of the true
    first-blocking category.
    """
    observed = stage
    stats = (
        client.stage_stats(observed)
        if isinstance(client, _QuestionPolicyClient) and observed
        else {}
    )
    if observed:
        stage_driver._stamp_stage_status(
            _state_path(workspace),
            observed,
            False,
            cost_usd=stats.get("cost_usd"),
            attempts=stats.get("attempts"),
            rounds=stats.get("rounds"),
            tool_calls=stats.get("tool_calls"),
            error=error,
            failure_kind=failure_kind,
        )
    event = {
        "kind": "stage_done",
        "stage": observed,
        "ok": False,
        "cost": stats.get("cost_usd", 0.0),
        "attempts": stats.get("attempts", 0),
        "error": error,
        "failure_kind": failure_kind,
    }
    progress = getattr(client, "observe_progress", None)
    if callable(progress):
        progress(event)
    emit_event(event)

    rows = [dict(row) for row in (_completed_rows(client))]
    failed = {
        "stage": observed,
        "commit_ok": False,
        "cost_usd": stats.get("cost_usd", 0.0),
        "attempts": stats.get("attempts"),
        "error": error,
        "failure_kind": failure_kind,
    }
    for index in range(len(rows) - 1, -1, -1):
        if rows[index].get("stage") == observed:
            rows[index] = {**rows[index], **{k: v for k, v in failed.items() if k != "stage"}}
            rows[index].pop("needs_input", None)
            rows[index].pop("questions", None)
            break
    else:
        rows.append(failed)
    return {
        "status": "failed",
        "results": rows,
        "guard": client.guard.status(),
        "state_path": str(_state_path(workspace)),
        "questions": None,
        "last_stage": observed,
        "error": error,
        "failure_kind": failure_kind,
    }


def _completed_rows(client) -> list[dict]:
    completed = getattr(client, "completed_rows", None)
    return list(completed()) if callable(completed) else []


def _reconcile_exhausted_result(workspace: Path, result: dict, client, progress) -> dict:
    deficits = session.bom_reconcile_deficits(result)
    stage = result.get("last_stage") or next(
        (question.get("stage") for question in deficits if isinstance(question, dict)), "wiring"
    )
    error = (
        "\n".join(
            str(question.get("text", "")).strip()
            for question in deficits
            if isinstance(question, dict)
        ).strip()
        or "BOM reconciliation exhausted before the deficit was resolved."
    )
    stats = client.stage_stats(stage) if isinstance(client, _QuestionPolicyClient) else {}
    stage_driver._attach_questions(_state_path(workspace), stage, [])
    stage_driver._stamp_stage_status(
        _state_path(workspace),
        stage,
        False,
        cost_usd=stats.get("cost_usd"),
        attempts=stats.get("attempts"),
        rounds=stats.get("rounds"),
        tool_calls=stats.get("tool_calls"),
        error=error,
        failure_kind="bom_reconcile_exhausted",
    )
    event = {
        "kind": "stage_done",
        "stage": stage,
        "ok": False,
        "cost": stats.get("cost_usd", 0.0),
        "attempts": stats.get("attempts"),
        "error": error,
        "failure_kind": "bom_reconcile_exhausted",
    }
    progress(event)

    rows = [dict(row) for row in (result.get("results") or [])]
    failed = {
        "stage": stage,
        "commit_ok": False,
        "cost_usd": stats.get("cost_usd", 0.0),
        "attempts": stats.get("attempts"),
        "error": error,
        "failure_kind": "bom_reconcile_exhausted",
    }
    for index in range(len(rows) - 1, -1, -1):
        if rows[index].get("stage") == stage:
            rows[index] = {**rows[index], **failed}
            rows[index].pop("needs_input", None)
            rows[index].pop("questions", None)
            break
    else:
        rows.append(failed)
    return {
        **result,
        "status": "failed",
        "results": rows,
        "questions": None,
        "last_stage": stage,
        "error": error,
        "failure_kind": "bom_reconcile_exhausted",
    }


def main() -> int:
    request = json.load(sys.stdin)
    workspace = Path(request["workspace"])
    budget_usd = float(request["budget_usd"])
    run_id = request.get("run_id")
    raw_policy = request.get("auto_default_questions")
    explicit_policy = raw_policy if isinstance(raw_policy, bool) else None
    native_client = make_budget_client(budget_usd)
    if _strict_budget_enabled():
        native_client = _enable_strict_native_budget(
            native_client, run_id=run_id, budget_usd=budget_usd
        )
    client = (
        _QuestionPolicyClient(native_client, explicit_policy)
        if explicit_policy is not None
        else native_client
    )

    active_stage = {"name": None}

    def progress(event):
        if isinstance(event, dict) and event.get("kind") == "stage_start":
            active_stage["name"] = event.get("stage")
        if isinstance(client, _QuestionPolicyClient):
            client.observe_progress(event)
        emit_event(event)

    def refusal(exc: Exception, failure_kind: str) -> dict:
        return _guard_refusal_result(
            workspace, client, active_stage["name"], failure_kind, str(exc)
        )

    try:
        result = session.run_session(
            workspace,
            str(request["brief"]),
            list(request["stages"]),
            answers=request.get("answers"),
            instruction=request.get("instruction"),
            client=client,
            run_id=run_id,
            progress=progress,
        )
    except _UserQuestionPause as pause:
        result = _pause_result(workspace, client, pause)
    except _InvalidClarification as invalid:
        result = _invalid_result(workspace, client, invalid)
    except BudgetExceeded as exceeded:
        # The guard's refusal is the design's terminal cause; report it as such.
        result = refusal(exceeded, "budget_exceeded")
    except KillSwitchEngaged as engaged:
        result = refusal(engaged, "kill_switch")

    # A reconciliation is internal legacy-stage recovery, not a user answer.
    # Keep the one client for every re-drive so its per-run guard snapshots once
    # and its shared persistent ledger sees the whole chain under the same run id.
    reconcile_passes = 0
    while result.get("status") == "awaiting_input" and session.bom_reconcile_deficits(result):
        previous_passes = reconcile_passes
        try:
            result, reconcile_passes = session.maybe_bom_reconcile(
                workspace,
                str(request["brief"]),
                result,
                client=client,
                run_id=run_id,
                reconcile_passes=reconcile_passes,
                progress=progress,
            )
        except _UserQuestionPause as pause:
            result = _pause_result(workspace, client, pause)
            break
        except _InvalidClarification as invalid:
            result = _invalid_result(workspace, client, invalid)
            break
        except BudgetExceeded as exceeded:
            result = refusal(exceeded, "budget_exceeded")
            break
        except KillSwitchEngaged as engaged:
            result = refusal(engaged, "kill_switch")
            break
        if reconcile_passes == previous_passes:
            break

    if result.get("status") == "awaiting_input" and session.bom_reconcile_deficits(result):
        result = _reconcile_exhausted_result(workspace, result, client, progress)

    print(
        RESULT_PREFIX
        + json.dumps(
            {"result": result, "reconcile_passes": reconcile_passes},
            separators=(",", ":"),
            default=str,
        )
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
