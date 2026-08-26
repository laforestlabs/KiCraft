"""Tests for the mock-LLM provider mode (kicraft/loadtest/mockllm.py + make_client)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from kicraft.loadtest import mockllm
from kicraft.server.config import Settings

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "kicraft"
    / "loadtest"
    / "fixtures"
    / "transcript_usb_pd_trigger.json"
)


# --- transcript reconstruction ------------------------------------------------
def test_transcript_from_state_reconstructs_every_committed_stage():
    state = {
        "project_stem": "DEMO",
        "intent": {"summary": "x", "assumptions": []},
        "functional_spec": {"blocks": []},
        "architecture": {"sheets": []},
        "bom": {
            "parts": [
                {
                    "ref": "R1",
                    "value": "1k",
                    "symbol": "Device:R",
                    "footprint": "Resistor_SMD:R_0603_1608Metric",
                    "sheet": "MAIN",
                }
            ],
            "connections": [
                {
                    "net_name": "N",
                    "sheet": "MAIN",
                    "endpoints": [{"ref": "R1", "pin": "1"}],
                }
            ],
            "no_connect_pins": [{"ref": "R1", "pin": "2"}],
        },
    }
    t = mockllm.transcript_from_state(state)
    assert t["stem"] == "DEMO"
    assert set(t["stages"]) == {"intent", "functional_spec", "architecture", "bom", "wiring"}
    # intent slot carries project_stem (the driver pops it before commit)
    assert json.loads(t["stages"]["intent"])["project_stem"] == "DEMO"
    bom_slot = json.loads(t["stages"]["bom"])
    assert list(bom_slot) == ["groups", "assumptions", "substitutions"]
    assert bom_slot["groups"][0]["reference_prefix"] == "R"
    assert bom_slot["groups"][0]["quantity"] == 1
    wiring = json.loads(t["stages"]["wiring"])
    assert wiring["pins"] == [
        {"ref": "R1", "pin": "1", "net": "N"},
        {"ref": "R1", "pin": "2", "no_connect": True},
    ]


def test_transcript_from_state_partial_state_yields_partial_transcript():
    t = mockllm.transcript_from_state({"project_stem": "P", "intent": {"a": 1}})
    assert set(t["stages"]) == {"intent"}


def test_committed_fixture_has_the_full_design_chain():
    t = mockllm.load_transcript(_FIXTURE)
    assert set(t["stages"]) == {"intent", "functional_spec", "architecture", "bom", "wiring"}


# --- MockClient surface -------------------------------------------------------
def test_mockclient_returns_stage_text_at_zero_cost():
    t = {"stem": "S", "stages": {"intent": '{"ok": 1}', "bom": '{"parts": []}'}}
    c = mockllm.MockClient(transcript=t)
    res = c.chat([{"role": "user", "content": "hi"}], meta_ctx={"stage": "intent"})
    assert res["text"] == '{"ok": 1}'
    assert res["cost_usd"] == 0.0
    assert res["finish_reason"] == "stop"
    # tool path (BOM) keys off the same meta_ctx and is also free
    calls = []
    res2 = c.chat_with_tools(
        [], [], lambda n, a: calls.append(n) or "out", meta_ctx={"stage": "bom"}
    )
    assert res2["text"] == '{"parts": []}'
    assert res2["cost_usd"] == 0.0
    assert calls == []  # tools not invoked by default


def test_mockclient_runs_bom_tools_when_enabled():
    t = {"stem": "S", "stages": {"bom": "{}"}}
    c = mockllm.MockClient(transcript=t, run_bom_tools=True)
    calls = []
    c.chat_with_tools([], [], lambda n, a: calls.append(n) or "out", meta_ctx={"stage": "bom"})
    assert calls == ["list_parts"]


def test_mockclient_guard_never_spends():
    c = mockllm.MockClient(transcript={"stages": {}})
    st = c.guard.status()
    assert st["spent_total_usd"] == 0.0 and st["mock"] is True
    c.guard.record("m", 1, 1, 9.99)  # no-op, must not raise
    assert c.guard.status()["spent_total_usd"] == 0.0


def test_mockclient_unknown_stage_returns_empty_object():
    c = mockllm.MockClient(transcript={"stages": {"intent": "{}"}})
    assert c.chat([], meta_ctx={"stage": "wiring"})["text"] == "{}"


def test_mockclient_without_transcript_raises_clearly(monkeypatch):
    monkeypatch.delenv("KICRAFT_MOCK_TRANSCRIPT", raising=False)
    c = mockllm.MockClient()
    with pytest.raises(RuntimeError, match="no transcript"):
        c.chat([], meta_ctx={"stage": "intent"})


def test_recording_client_captures_committed_text():
    class _Inner:
        s, guard = object(), object()

        def chat(self, messages, **kw):
            return {"text": '{"slot": 1}', "cost_usd": 0.01}

        def chat_with_tools(self, messages, tools, executor, **kw):
            return {"text": '{"bom": 1}', "cost_usd": 0.02}

    rec = mockllm.RecordingClient(_Inner())
    rec.chat([], meta_ctx={"stage": "intent"})
    rec.chat_with_tools([], [], None, meta_ctx={"stage": "bom"})
    assert rec.transcript["stages"] == {"intent": '{"slot": 1}', "bom": '{"bom": 1}'}


# --- make_client factory: prod no-op unless explicitly switched ---------------
def test_make_client_defaults_to_real_client(monkeypatch, tmp_path):
    pytest.importorskip("requests")  # client.py imports requests at module load
    monkeypatch.delenv("KICRAFT_LLM_MODE", raising=False)
    from kicraft.server.client import CappedOpenRouterClient, make_client

    s = Settings(api_key="test-key", ledger_path=tmp_path / "ledger.db")
    assert isinstance(make_client(s), CappedOpenRouterClient)


@pytest.mark.parametrize("mode", ["mock", "replay"])
def test_make_client_returns_mockclient_in_mock_modes(monkeypatch, mode):
    pytest.importorskip("requests")  # client.py imports requests at module load
    monkeypatch.setenv("KICRAFT_LLM_MODE", mode)
    from kicraft.server.client import make_client

    # No settings and no OPENROUTER_API_KEY needed for the mock path.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert isinstance(make_client(), mockllm.MockClient)


# --- full pipeline integration (needs KiCad; skips where absent) --------------
def _kicad_available() -> bool:
    if shutil.which("kicad-cli") is None:
        return False
    try:
        import pcbnew  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kicad_available(), reason="kicad-cli / pcbnew not available")
def test_replay_drives_full_chain_at_zero_cost(monkeypatch):
    """KICRAFT_LLM_MODE=replay commits every design stage from the frozen fixture,
    with no API key and no spend -- the proof the mock exercises the real
    stage-prep / stage-commit subprocess machinery for free."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("KICRAFT_LLM_MODE", "replay")
    monkeypatch.setenv("KICRAFT_MOCK_TRANSCRIPT", str(_FIXTURE))
    from kicraft.server.stage_driver import DESIGN_STAGES, drive_chain

    ws = Path(tempfile.mkdtemp(prefix="mocktest_"))
    try:
        results, guard, _ = drive_chain(list(DESIGN_STAGES), "a usb-c pd trigger", ws)
        assert [r["stage"] for r in results] == list(DESIGN_STAGES)
        assert all(r["commit_ok"] for r in results), [
            (r["stage"], r.get("error") or r.get("commit")) for r in results
        ]
        assert guard["spent_total_usd"] == 0.0
    finally:
        shutil.rmtree(ws, ignore_errors=True)
