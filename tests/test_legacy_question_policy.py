"""Pinned-native regressions for runner-owned clarification policy.

These tests deliberately enter the current-owned runner as a separate process.  A
``sitecustomize`` overlay replaces only the native budget-client factory, leaving
the native session, stage driver, and stage-prep/stage-commit CLIs intact.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = _ROOT / "kicraft" / "server" / "legacy_session_runner.py"
_LEGACY_ROOT = Path("/home/kicraft/KiCraft-legacy")
_LEGACY_PYTHON = _LEGACY_ROOT / ".venv" / "bin" / "python"
_EVENT_PREFIX = "__KICRAFT_LEGACY_EVENT__="
_RESULT_PREFIX = "__KICRAFT_LEGACY_SESSION_RESULT__="

_INTENT_SLOT = {
    "goal": "A USB-powered indicator board",
    "constraints": [],
    "named_parts": [],
    "inferred_expertise": "intermediate",
    "assumptions": ["USB power is used (defaulted)"],
    "project_stem": "USB_INDICATOR",
}


@pytest.fixture
def native_python() -> Path:
    if not _LEGACY_PYTHON.is_file():
        pytest.skip(
            "pinned native integration requires /home/kicraft/KiCraft-legacy/.venv/bin/python"
        )
    return _LEGACY_PYTHON


def _fake_client_overlay(tmp_path: Path, replies: list[dict], trace: Path) -> Path:
    """Install an import-time native factory replacement for one child process."""
    overlay = tmp_path / "native-overlay"
    overlay.mkdir(parents=True)
    config = overlay / "canned.json"
    config.write_text(json.dumps({"replies": replies, "trace": str(trace)}), encoding="utf-8")
    (overlay / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            from kicraft.server import stage_driver

            _config = json.loads(Path(os.environ["KICRAFT_TEST_CANNED_CLIENT"]).read_text())

            class _Guard:
                def __init__(self):
                    self.calls = 0

                def status(self):
                    return {"spent_total_usd": round(self.calls * 0.125, 6)}

            class _Client:
                def __init__(self):
                    self.s = type("Settings", (), {"design_temperature": 0.0})()
                    self.guard = _Guard()
                    self._replies = list(_config["replies"])
                    self._calls = []

                def _reply(self, method, messages, kwargs):
                    self.guard.calls += 1
                    self._calls.append({
                        "method": method,
                        "meta_ctx": kwargs.get("meta_ctx"),
                        "reasoning": kwargs.get("reasoning"),
                        "system": [m.get("content") for m in messages if m.get("role") == "system"],
                        "user": [m.get("content") for m in messages if m.get("role") == "user"],
                    })
                    Path(_config["trace"]).write_text(json.dumps({"calls": self._calls}))
                    return {
                        "text": json.dumps(self._replies.pop(0)),
                        "cost_usd": 0.125,
                        "finish_reason": "stop",
                        "provider": "canned-native",
                        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                        "response_marker": "preserve-me",
                    }

                def chat(self, messages, **kwargs):
                    return self._reply("chat", messages, kwargs)

                def chat_with_tools(self, messages, tools, executor, **kwargs):
                    # Match the native method's observable mutation contract.
                    messages.append({"role": "assistant", "content": "canned tool completion"})
                    response = self._reply("chat_with_tools", messages, kwargs)
                    response.update({"rounds": 1, "tool_calls": 0})
                    return response

            stage_driver.make_budget_client = lambda budget_usd: _Client()
            """
        ),
        encoding="utf-8",
    )
    return overlay


def _run_runner(
    native_python: Path,
    tmp_path: Path,
    *,
    name: str,
    replies: list[dict],
    request: dict,
) -> tuple[subprocess.CompletedProcess[str], list[dict], dict, dict]:
    trace = tmp_path / f"{name}-trace.json"
    overlay = _fake_client_overlay(tmp_path / name, replies, trace)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "KICRAFT_ROUTING_CONFIG": str(tmp_path / "routing.json"),
        "PYTHONPATH": os.pathsep.join((str(overlay), str(_LEGACY_ROOT))),
        "KICRAFT_TEST_CANNED_CLIENT": str(overlay / "canned.json"),
        # No provider client is ever constructed, but keep the native process isolated
        # from an operator's local configuration if another import reads these variables.
        "KICRAFT_LCSC_RETAIL": "0",
    }
    completed = subprocess.run(
        [str(native_python), "-u", str(_RUNNER)],
        input=json.dumps(request),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        timeout=45,
        check=False,
    )
    events: list[dict] = []
    packet = None
    for line in completed.stdout.splitlines():
        if line.startswith(_EVENT_PREFIX):
            events.append(json.loads(line[len(_EVENT_PREFIX) :]))
        elif line.startswith(_RESULT_PREFIX):
            packet = json.loads(line[len(_RESULT_PREFIX) :])
    assert packet is not None, completed.stderr + "\n" + completed.stdout
    return completed, events, packet, json.loads(trace.read_text(encoding="utf-8"))


def _request(workspace: Path, auto_default_questions: bool, **extra: object) -> dict:
    return {
        "workspace": str(workspace),
        "brief": "A USB-powered indicator board",
        "stages": ["intent"],
        "budget_usd": 1.0,
        "run_id": "native-policy-test",
        "auto_default_questions": auto_default_questions,
        **extra,
    }


def test_native_auto_default_retries_question_without_outward_park(native_python, tmp_path):
    """Checked policy uses the native in-stage default retry, rather than a fake answer."""
    question = {
        "questions": [
            {
                "text": "Which power source should the board use?",
                "options": ["USB-C 5 V", "LiPo 1S"],
                "blocking": True,
            }
        ]
    }
    completed, events, packet, trace = _run_runner(
        native_python,
        tmp_path,
        name="automatic",
        replies=[question, _INTENT_SLOT],
        request=_request(tmp_path / "automatic-workspace", True),
    )

    result = packet["result"]
    assert completed.returncode == 0, completed.stderr
    assert result["status"] == "ok"
    assert result["questions"] is None
    assert [event["kind"] for event in events] == ["stage_start", "stage_done"]
    # The shape of native rows is intentionally not pinned; the observable accounting is.
    row = result["results"][0]
    assert row["stage"] == "intent" and row["commit_ok"] is True
    assert row["attempts"] == 2 and row["cost_usd"] == pytest.approx(0.25)
    assert result["guard"]["spent_total_usd"] == pytest.approx(0.25)
    assert len(trace["calls"]) == 2
    assert trace["calls"][0]["meta_ctx"]["stage"] == "intent"


def test_native_interactive_questions_pause_after_answer_until_every_ambiguity_resolves(
    native_python, tmp_path
):
    """An explicit False remains interactive even after a prior answer reached native prompts."""
    workspace = tmp_path / "interactive-workspace"
    first_question = {
        "questions": [
            {
                "text": "Which input connector should be fitted?",
                "options": [" USB-C receptacle ", "Terminal block", "USB-C receptacle", ""],
                "blocking": True,
            }
        ]
    }
    first, first_events, first_packet, _ = _run_runner(
        native_python,
        tmp_path,
        name="first-question",
        replies=[first_question],
        request=_request(workspace, False),
    )
    first_result = first_packet["result"]
    assert first.returncode != 0
    assert first_result["status"] == "awaiting_input"
    assert [event["kind"] for event in first_events] == ["stage_start", "question"]
    assert first_result["questions"][0]["options"] == ["USB-C receptacle", "Terminal block"]
    assert first_result["results"][0]["cost_usd"] == pytest.approx(0.125)
    assert first_result["guard"]["spent_total_usd"] == pytest.approx(0.125)
    persisted = json.loads((workspace / ".kicraft" / "state.json").read_text(encoding="utf-8"))
    assert persisted["open_questions"][0]["options"] == ["USB-C receptacle", "Terminal block"]

    second_question = {
        "questions": [
            {
                "text": "Should the indicator be green or amber?",
                "options": ["Green indicator", "Amber indicator"],
                "blocking": True,
            }
        ]
    }
    second, second_events, second_packet, trace = _run_runner(
        native_python,
        tmp_path,
        name="second-question",
        replies=[second_question],
        request=_request(
            workspace,
            False,
            answers=[
                {
                    "text": "Which input connector should be fitted?",
                    "answer": "USB-C receptacle",
                }
            ],
        ),
    )
    second_result = second_packet["result"]
    assert second.returncode != 0
    assert second_result["status"] == "awaiting_input"
    assert [event["kind"] for event in second_events] == ["stage_start", "question"]
    assert second_result["questions"][0]["text"] == "Should the indicator be green or amber?"
    assert "A: USB-C receptacle" in trace["calls"][0]["user"][0]

    third, third_events, third_packet, _ = _run_runner(
        native_python,
        tmp_path,
        name="resolved",
        replies=[_INTENT_SLOT],
        request=_request(
            workspace,
            False,
            answers=[
                {
                    "text": "Should the indicator be green or amber?",
                    "answer": "Green indicator",
                }
            ],
        ),
    )
    assert third.returncode == 0, third.stderr
    assert third_packet["result"]["status"] == "ok"
    assert [event["kind"] for event in third_events] == ["stage_start", "stage_done"]


def test_native_interactive_invalid_options_fail_instead_of_parking_freeform(
    native_python, tmp_path
):
    malformed = {
        "questions": [
            {
                "text": "Which input voltage is required?",
                "options": [" 5 V ", "5 V"],
                "blocking": True,
            }
        ]
    }
    completed, events, packet, _ = _run_runner(
        native_python,
        tmp_path,
        name="invalid-options",
        replies=[malformed],
        request=_request(tmp_path / "invalid-workspace", False),
    )

    result = packet["result"]
    assert completed.returncode != 0
    assert result["status"] == "failed"
    assert result["questions"] is None
    assert result["failure_kind"] == "invalid_clarification"
    assert "at least two distinct options" in result["error"]
    assert [event["kind"] for event in events] == ["stage_start", "stage_done"]
    assert events[-1]["ok"] is False
    state = json.loads(
        (tmp_path / "invalid-workspace" / ".kicraft" / "state.json").read_text(encoding="utf-8")
    )
    assert state["stage_status"]["intent"]["failure_kind"] == "invalid_clarification"


def test_native_policy_decorator_preserves_tool_metadata_and_internal_reconcile(
    native_python, tmp_path
):
    """The narrow native-runtime probe covers tool mutation and reconcile precedence."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import json
            import sys

            sys.path.insert(0, {str(_RUNNER.parent)!r})
            spec = importlib.util.spec_from_file_location("runner_under_test", {str(_RUNNER)!r})
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)

            class Guard:
                def status(self):
                    return {{"spent_total_usd": 0.0}}

            class Client:
                s = object()
                guard = Guard()
                def chat_with_tools(self, messages, tools, executor, **kwargs):
                    messages.append({{"role": "assistant", "content": "tool answer"}})
                    return {{"text": '{{}}', "cost_usd": 0.125, "provider": "canned", "usage": {{"n": 1}}, "rounds": 1, "tool_calls": 0, "response_marker": "kept"}}
                def chat(self, messages, **kwargs):
                    return {{"text": json.dumps({{"questions": [{{"text": "Add C1", "options": ["Add 1 uF", "Add 100 nF"], "blocking": True, "reconcile_target": "bom"}}]}}), "cost_usd": 0.125, "finish_reason": "stop", "provider": "canned"}}

            tool_messages = [{{"role": "system", "content": "base"}}]
            automatic = runner._QuestionPolicyClient(Client(), True)
            first = automatic.chat_with_tools(tool_messages, [], lambda *_: None, meta_ctx={{"stage": "bom"}})
            second = automatic.chat_with_tools(tool_messages, [], lambda *_: None, meta_ctx={{"stage": "bom"}})
            interactive = runner._QuestionPolicyClient(Client(), False)
            reconcile = interactive.chat([{{"role": "system", "content": "base"}}], meta_ctx={{"stage": "wiring"}})
            print(json.dumps({{"first": first, "second": second, "messages": tool_messages, "reconcile": reconcile}}))
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(native_python), str(probe)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(_LEGACY_ROOT),
        },
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["first"]["provider"] == "canned"
    assert output["first"]["usage"] == {"n": 1}
    assert output["second"]["response_marker"] == "kept"
    assert [m["role"] for m in output["messages"]].count("assistant") == 2
    assert json.loads(output["reconcile"]["text"])["questions"][0]["blocking"] is True


def test_native_pause_retains_completed_stages_and_total_spend(native_python, tmp_path):
    question = {
        "questions": [
            {
                "text": "Which indicator interface?",
                "options": ["Obtain the interface specification", "Use a supplied specification"],
                "blocking": True,
                "stage": "wiring",
            }
        ]
    }
    completed, events, packet, _ = _run_runner(
        native_python,
        tmp_path,
        name="downstream",
        replies=[_INTENT_SLOT, question],
        request=_request(
            tmp_path / "downstream-workspace", False, stages=["intent", "functional_spec"]
        ),
    )
    result = packet["result"]
    assert result["status"] == "awaiting_input"
    assert result["last_stage"] == "functional_spec"
    assert result["questions"][0]["stage"] == "functional_spec"
    assert [(r["stage"], r["commit_ok"]) for r in result["results"]] == [
        ("intent", True),
        ("functional_spec", False),
    ]
    assert [r["cost_usd"] for r in result["results"]] == [0.125, 0.125]
    assert result["guard"]["spent_total_usd"] == 0.25
    assert [(e["kind"], e["stage"]) for e in events] == [
        ("stage_start", "intent"),
        ("stage_done", "intent"),
        ("stage_start", "functional_spec"),
        ("question", "functional_spec"),
    ]


def test_strict_native_budget_admits_each_native_post_before_dispatch(native_python, tmp_path):
    """Cover header retries, discarded partials, receipts, exact-run caps, and races."""
    probe = tmp_path / "strict_budget_probe.py"
    ledger = tmp_path / "strict-ledger.db"
    probe.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import json
            import multiprocessing
            import os
            import sqlite3
            import time
            from datetime import datetime, timezone
            from pathlib import Path

            sys_path = {str(_RUNNER.parent)!r}
            import sys
            sys.path.insert(0, sys_path)
            from budget_exposure import strict_budget_exposure_status
            spec = importlib.util.spec_from_file_location("runner_under_test", {str(_RUNNER)!r})
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)

            NOW = datetime.now(timezone.utc).isoformat()

            LEDGER = Path({str(ledger)!r})
            with sqlite3.connect(LEDGER) as conn:
                conn.execute(
                    "CREATE TABLE spend (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, model TEXT, "
                    "input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL NOT NULL, meta TEXT)"
                )

            class Settings:
                model = "strict-model"
                max_tokens_per_call = 100
                max_price_prompt = 1.0
                max_price_completion = 1.0
                daily_usd_ceiling = 2.0
                total_usd_ceiling = 2.0
                kill_switch = False

            class Guard:
                path = LEDGER
                def __init__(self, settings):
                    self.s = settings
                def record(self, model, intok, outtok, cost, meta=""):
                    with sqlite3.connect(self.path) as conn:
                        conn.execute(
                            "INSERT INTO spend (ts,model,input_tokens,output_tokens,cost_usd,meta) "
                            "VALUES (?,?,?,?,?,?)",
                            (NOW, model, intok, outtok, cost, json.dumps(meta, sort_keys=True)),
                        )
                def status(self):
                    return {{}}
            class _Requests:
                class Timeout(Exception):
                    pass

                def __init__(self):
                    self.timeout_once = False

                def post(self, url, *, json, timeout):
                    if self.timeout_once:
                        self.timeout_once = False
                        raise self.Timeout("timeout before response headers")
                    return object()

            requests = _Requests()

            class Client:
                def __init__(self, *, pause=False, retry=False, abort=False, receipt=True):
                    self.s = Settings()
                    self.guard = Guard(self.s)
                    self.dispatches = []
                    self.pause = pause
                    self.retry = retry
                    self.abort = abort
                    self.receipt = receipt
                def _open_stream(self, body):
                    for attempt in range(2 if self.retry else 1):
                        try:
                            return requests.post("https://provider.invalid/chat", json=body, timeout=1)
                        except requests.Timeout:
                            if attempt:
                                raise
                def _stream(self, body, on_delta=None):
                    self.guard.preflight()
                    self._open_stream(body)
                    self.dispatches.append(body)
                    if self.abort:
                        raise RuntimeError("stream lost after dispatch")
                    if self.pause:
                        time.sleep(0.15)
                    self.guard.record(
                        "strict-model", 1, 1, 0.001,
                        meta={{"run_id": body["_meta_ctx"]["run_id"], "round": body["_meta_ctx"].get("round")}},
                    )
                    usage = {{"cost": 0.001}} if self.receipt else None
                    return {{"content": "ok", "usage": usage}}, 0.001
                def chat_with_tools(self):
                    for round_number in range(3):
                        self._stream({{
                            "messages": [{{"role": "user", "content": "tool round"}}],
                            "tools": [{{"type": "function", "function": {{"name": "lookup"}}}}],
                            "max_tokens": 100,
                            "_meta_ctx": {{"round": round_number}},
                        }})

            def spend(cost, run_id):
                with sqlite3.connect(LEDGER) as conn:
                    conn.execute(
                        "INSERT INTO spend (ts,model,input_tokens,output_tokens,cost_usd,meta) "
                        "VALUES (?,'seed',0,0,?,?)",
                        (NOW, cost, json.dumps({{"run_id": run_id}})),
                    )

            # A single call's full prompt+completion reservation must be refused
            # before the native stream's dispatch list is touched.
            spend(1.9999, "other")
            refused = Client()
            runner._enable_strict_native_budget(refused, run_id="ceiling", budget_usd=2.0)
            try:
                refused._stream({{"messages": [{{"role": "user", "content": "x"}}], "max_tokens": 100}})
            except runner.BudgetExceeded:
                call_refused = not refused.dispatches
            else:
                call_refused = False

            unbounded = Client()
            runner._enable_strict_native_budget(unbounded, run_id="unbounded", budget_usd=2.0)
            try:
                unbounded._stream({{"messages": [{{"role": "user", "content": "x"}}]}})
            except runner.BudgetExceeded:
                unbounded_refused = not unbounded.dispatches
            else:
                unbounded_refused = False

            # The pinned tool-loop body carries provider tool-routing fields. An
            # allow-list of request fields once rejected `parallel_tool_calls`,
            # and the uncaught refusal killed every run at its first tool stage.
            with sqlite3.connect(LEDGER) as conn:
                conn.execute("DELETE FROM spend")
                conn.execute("DELETE FROM strict_budget_exposure")
            os.environ["KICRAFT_PROJECT_LLM_BUDGET_USD"] = "0.01"
            tool_body = Client()
            runner._enable_strict_native_budget(tool_body, run_id="tool-body", budget_usd=0.01)
            tool_body._stream({{
                "messages": [{{"role": "user", "content": "x"}}],
                "max_tokens": 100,
                "tools": [{{"type": "function", "function": {{"name": "lookup"}}}}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }})
            tool_body_dispatched = len(tool_body.dispatches) == 1

            # An explicit completion multiplier is still refused: the serialized
            # request size cannot bound n completions.
            multi = Client()
            runner._enable_strict_native_budget(multi, run_id="multi", budget_usd=0.01)
            try:
                multi._stream({{"messages": [{{"role": "user", "content": "x"}}],
                               "max_tokens": 100, "n": 3}})
            except runner.BudgetExceeded:
                multi_refused = not multi.dispatches
            else:
                multi_refused = False

            # Other runs' ledger rows do not consume this exact run's project
            # budget. The injected run id, not the caller's metadata, is recorded.
            with sqlite3.connect(LEDGER) as conn:
                conn.execute("DELETE FROM spend")
            spend(0.6, "other")
            spend(0.0005, "isolated")
            os.environ["KICRAFT_PROJECT_LLM_BUDGET_USD"] = "0.003"
            isolated = Client()
            runner._enable_strict_native_budget(isolated, run_id="isolated", budget_usd=0.003)
            isolated._stream({{
                "messages": [{{"role": "user", "content": "x"}}],
                "max_tokens": 100,
                "_meta_ctx": {{"run_id": "forged"}},
            }})
            with sqlite3.connect(LEDGER) as conn:
                isolated_meta = json.loads(conn.execute(
                    "SELECT meta FROM spend WHERE json_extract(meta, '$.run_id')='isolated' "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()[0])

            # A timeout before headers makes the pinned _open_stream POST again.
            # The prior POST stays uncertain; the verified final receipt settles
            # only its own reservation.
            with sqlite3.connect(LEDGER) as conn:
                conn.execute("DELETE FROM spend")
                conn.execute("DELETE FROM strict_budget_exposure")
            os.environ["KICRAFT_PROJECT_LLM_BUDGET_USD"] = "0.01"
            requests.timeout_once = True
            retry = Client(retry=True)
            runner._enable_strict_native_budget(retry, run_id="retry", budget_usd=0.01)
            retry._stream({{"messages": [{{"role": "user", "content": "x"}}], "max_tokens": 100}})
            retry_exposure = strict_budget_exposure_status(LEDGER, run_id="retry")
            retry_next_limit = (
                retry_exposure["ledger_spend_usd"]
                + 1.5 * retry_exposure["active_exposure_usd"]
            )
            Settings.daily_usd_ceiling = retry_next_limit
            Settings.total_usd_ceiling = retry_next_limit
            retry_next = Client()
            runner._enable_strict_native_budget(retry_next, run_id="retry-next", budget_usd=0.01)
            try:
                retry_next._stream(
                    {{"messages": [{{"role": "user", "content": "x"}}], "max_tokens": 100}}
                )
            except runner.BudgetExceeded:
                retry_next_refused = not retry_next.dispatches
            else:
                retry_next_refused = False
            Settings.daily_usd_ceiling = 2.0
            Settings.total_usd_ceiling = 2.0
            failed = Client(abort=True)
            runner._enable_strict_native_budget(failed, run_id="lost", budget_usd=0.01)
            try:
                failed._stream({{"messages": [{{"role": "user", "content": "x"}}], "max_tokens": 100}})
            except RuntimeError:
                pass
            estimated = Client(receipt=False)
            runner._enable_strict_native_budget(estimated, run_id="estimated", budget_usd=0.01)
            estimated._stream({{"messages": [{{"role": "user", "content": "x"}}], "max_tokens": 100}})
            estimated_exposure = strict_budget_exposure_status(LEDGER, run_id="estimated")
            lost_exposure = strict_budget_exposure_status(LEDGER, run_id="lost")

            # The tool-loop calls the patched stream once per round. Two calls fit,
            # while the third must be refused before it can dispatch.
            with sqlite3.connect(LEDGER) as conn:
                conn.execute("DELETE FROM spend")
                conn.execute("DELETE FROM strict_budget_exposure")
            os.environ["KICRAFT_PROJECT_LLM_BUDGET_USD"] = "0.0021"
            tool = Client()
            runner._enable_strict_native_budget(tool, run_id="tool-loop", budget_usd=0.0021)
            try:
                tool.chat_with_tools()
            except runner.BudgetExceeded:
                tool_rounds = [row["_meta_ctx"]["round"] for row in tool.dispatches]
            else:
                tool_rounds = []

            # fcntl locking spans admission through actual record: one of two
            # competing process calls can reserve the last global headroom, never both.
            with sqlite3.connect(LEDGER) as conn:
                conn.execute("DELETE FROM spend")
                conn.execute("DELETE FROM strict_budget_exposure")
            Settings.daily_usd_ceiling = 0.0011
            Settings.total_usd_ceiling = 0.0011
            os.environ.pop("KICRAFT_PROJECT_LLM_BUDGET_USD", None)
            def worker(run_id, queue):
                client = Client(pause=True)
                runner._enable_strict_native_budget(client, run_id=run_id, budget_usd=1.0)
                try:
                    client._stream({{"messages": [{{"role": "user", "content": "x"}}], "max_tokens": 100}})
                except runner.BudgetExceeded:
                    queue.put("refused")
                else:
                    queue.put("dispatched")
            context = multiprocessing.get_context("fork")
            queue = context.Queue()
            first = context.Process(target=worker, args=("race-a", queue))
            second = context.Process(target=worker, args=("race-b", queue))
            first.start()
            second.start()
            first.join(10)
            second.join(10)
            race = sorted([queue.get(timeout=2), queue.get(timeout=2)])
            print(json.dumps({{
                "unbounded_refused": unbounded_refused,
                "tool_body_dispatched": tool_body_dispatched,
                "multi_refused": multi_refused,
                "retry_next_refused": retry_next_refused,
                "call_refused": call_refused,
                "isolated_meta": isolated_meta,
                "retry_exposure": retry_exposure,
                "lost_exposure": lost_exposure,
                "estimated_exposure": estimated_exposure,
                "tool_rounds": tool_rounds,
                "race": race,
                "exitcodes": [first.exitcode, second.exitcode],
            }}))
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(native_python), str(probe)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(_LEGACY_ROOT),
        },
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    proof = json.loads(completed.stdout)
    assert proof["retry_exposure"]["ledger_spend_usd"] == pytest.approx(0.001)
    assert proof["retry_exposure"]["reserved_exposure_usd"] == pytest.approx(0.0)
    assert proof["retry_exposure"]["uncertain_exposure_usd"] > 0.0
    assert proof["lost_exposure"]["ledger_spend_usd"] == pytest.approx(0.0)
    assert proof["lost_exposure"]["uncertain_exposure_usd"] > 0.0
    assert proof["estimated_exposure"]["ledger_spend_usd"] == pytest.approx(0.001)
    assert proof["estimated_exposure"]["uncertain_exposure_usd"] > 0.0
    assert proof["unbounded_refused"] is True
    assert proof["tool_body_dispatched"] is True
    assert proof["multi_refused"] is True
    assert proof["retry_next_refused"] is True
    assert proof["call_refused"] is True
    assert proof["isolated_meta"]["run_id"] == "isolated"
    assert proof["tool_rounds"] == [0, 1]
    assert proof["race"] == ["dispatched", "refused"]
    assert proof["exitcodes"] == [0, 0]
