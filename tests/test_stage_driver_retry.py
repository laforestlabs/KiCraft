"""Tests for the stage driver's self-correction feedback and per-stage retries.

Pure functions, no OpenRouter / network: exercises the retry-message construction
and the per-stage retry budget that help the wiring stage converge.
"""
from __future__ import annotations

import json

from kicraft.server.stage_driver import (
    BOM_TOOLS,
    _attach_questions,
    _normalize_questions,
    _retry_feedback,
    _stage_max_retries,
    _stage_max_tokens,
    build_system,
)


def test_retry_feedback_includes_errors_and_offenders():
    out = {"ok": False, "errors": ["9.11 net coverage: uncovered pin(s)"],
           "offenders": ["U2.4 ('EN', unspecified) not in connections or no_connect_pins"]}
    msg = _retry_feedback(out)
    assert "9.11 net coverage" in msg          # the rule that failed
    assert "U2.4" in msg                        # the exact pin the model must fix
    assert "preserv" in msg.lower()             # patch, do not redraft
    assert "ONLY the slot JSON" in msg


def test_retry_feedback_without_offenders_omits_that_line():
    msg = _retry_feedback({"ok": False, "errors": ["some other error"]})
    assert "some other error" in msg
    assert "offenders" not in msg               # no offenders line when none present


def test_wiring_gets_more_retries_than_the_simple_stages():
    assert _stage_max_retries("wiring", 2) >= 4   # wiring floors higher than default
    assert _stage_max_retries("intent", 2) == 2   # simple stages keep the default
    assert _stage_max_retries("functional_spec", 2) == 2


def test_caller_default_wins_when_higher_than_the_floor():
    assert _stage_max_retries("wiring", 6) == 6
    assert _stage_max_retries("intent", 6) == 6


def test_wiring_gets_a_larger_token_budget():
    assert _stage_max_tokens("wiring", 4096) >= 8192   # wiring floors higher
    assert _stage_max_tokens("intent", 4096) == 4096   # simple stages keep default
    assert _stage_max_tokens("wiring", 16000) == 16000  # a higher caller default wins


def test_bom_gets_more_retries_for_symbol_resolution():
    assert _stage_max_retries("bom", 2) >= 4           # bom floors higher now
    assert _stage_max_retries("architecture", 2) == 2


def test_bom_has_a_symbol_search_tool():
    names = {t["function"]["name"] for t in BOM_TOOLS}
    assert "search_symbols" in names                   # discover, do not guess
    assert {"list_parts", "lookup_symbol", "lookup_lcsc_id", "add_part_from_lcsc"} <= names


def test_bom_has_a_footprint_search_tool():
    names = {t["function"]["name"] for t in BOM_TOOLS}
    assert "search_footprints" in names                # footprint discovery, do not guess
    assert "lookup_footprint" in names                 # verify a footprint exists + pad count


# ---- clarifying questions -------------------------------------------------

def test_normalize_questions_shapes_and_drops_junk():
    qs = _normalize_questions(
        [{"text": "Battery chemistry?", "options": ["LiPo", "18650"], "blocking": True},
         {"text": "   ", "blocking": True},   # dropped: blank text
         {"nope": 1}],                        # dropped: not a question
        "intent")
    assert len(qs) == 1
    q = qs[0]
    assert q["stage"] == "intent" and q["blocking"] is True
    assert q["options"] == ["LiPo", "18650"] and q["answer"] is None


def test_attach_questions_writes_open_questions(tmp_path):
    sp = tmp_path / ".kicraft" / "state.json"  # not yet created (a first-stage question)
    qs = _normalize_questions([{"text": "Q1", "blocking": True}], "intent")
    _attach_questions(sp, "intent", qs)
    sj = json.loads(sp.read_text())
    assert sj["open_questions"][0]["text"] == "Q1"
    assert sj["open_questions"][0]["stage"] == "intent"


def test_attach_questions_replaces_only_that_stage(tmp_path):
    sp = tmp_path / ".kicraft" / "state.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps({"open_questions": [
        {"text": "old-intent", "stage": "intent"}, {"text": "keep-arch", "stage": "architecture"}]}))
    _attach_questions(sp, "intent", _normalize_questions([{"text": "new-intent"}], "intent"))
    texts = {q["text"] for q in json.loads(sp.read_text())["open_questions"]}
    assert texts == {"new-intent", "keep-arch"}  # intent replaced, architecture kept


def test_build_system_offers_clarifying_questions():
    sysmsg = build_system("intent")
    assert '"questions"' in sysmsg   # the model is told it may ask
    assert "blocking" in sysmsg


def test_bom_part_hints_extracts_pasted_lcsc_ids():
    from kicraft.server.stage_driver import _bom_part_hints
    brief = ("ToF breakout with the sensor at "
             "https://www.lcsc.com/product-detail/C7386355.html and an LDO C6186")
    hint = _bom_part_hints(brief, "also use c2924337 please")
    assert "C7386355" in hint and "C6186" in hint and "C2924337" in hint
    assert "add_part_from_lcsc" in hint


def test_bom_part_hints_ignores_refdes_and_embedded_runs():
    from kicraft.server.stage_driver import _bom_part_hints
    # C1/C104 are refdes/values, C8051F320 is an MPN — none are LCSC ids.
    assert _bom_part_hints("decouple C1 with 100nF, C104 pattern, MCU C8051F320") == ""
    assert _bom_part_hints("", None) == ""


def test_bom_prompt_mentions_search_budget():
    msg = build_system("bom")
    assert "SEARCH BUDGET" in msg
    assert "STOP searching" in msg
