"""Tests for the stage driver's self-correction feedback and per-stage retries.

Pure functions, no OpenRouter / network: exercises the retry-message construction
and the per-stage retry budget that help the wiring stage converge.
"""
from __future__ import annotations

import json

from kicraft.server.stage_driver import (
    BOM_TOOLS,
    _attach_questions,
    _design_reasoning,
    _json_failure_recovery,
    _normalize_questions,
    _retry_feedback,
    _stage_max_retries,
    _stage_max_tokens,
    build_system,
)
from kicraft.server.session import run_session


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


def test_reasoning_loop_recovery_keeps_budget_and_bumps_temp():
    # finish=length with NO content is a stuck reasoning loop, not a truncated
    # JSON answer: keep the budget and raise temperature to break the greedy cycle.
    msg, new_max, new_temp = _json_failure_recovery(
        "length", had_content=False, cur_max_tokens=4096, temperature=0.0)
    assert "reconsidering" in msg
    assert new_max == 4096           # NOT doubled
    assert new_temp == 0.4


def test_truncated_json_recovery_still_doubles_budget():
    # finish=length WITH content is a real truncated answer: still double the cap.
    msg, new_max, new_temp = _json_failure_recovery(
        "length", had_content=True, cur_max_tokens=4096, temperature=0.0)
    assert new_max == 8192
    assert "cut off" in msg
    assert new_temp == 0.0


def test_no_json_recovery_keeps_budget_and_temp():
    msg, new_max, new_temp = _json_failure_recovery(
        "stop", had_content=False, cur_max_tokens=4096, temperature=0.0)
    assert "not a single valid JSON object" in msg
    assert new_max == 4096
    assert new_temp == 0.0


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


def test_normalize_questions_carries_and_whitelists_reconcile_target():
    # A "bom" target is preserved (the pipeline can self-repair it); an unknown
    # target is dropped to None so a park can't route to an arbitrary/looping
    # stage, and an untagged question stays a plain user question.
    qs = _normalize_questions(
        [{"text": "Add 3 more 100nF for U1 DEC pins", "blocking": True,
          "reconcile_target": "bom"},
         {"text": "Active-high or active-low button?", "blocking": True},
         {"text": "route to nowhere", "blocking": True, "reconcile_target": "wiring"}],
        "wiring")
    assert [q["reconcile_target"] for q in qs] == ["bom", None, None]
    # the normalized dicts still validate as Question (schema-safe for state.json)
    from kicraft.design.models import Question
    for q in qs:
        Question.model_validate(q)


def test_wiring_prompt_tells_model_to_self_repair_a_bom_shortfall():
    sysmsg = build_system("wiring")
    # wiring must be told to tag a BOM parts shortfall for automatic repair
    # instead of asking the user.
    assert "reconcile_target" in sysmsg
    assert '"bom"' in sysmsg


def test_bom_prompt_demands_decoupling_completeness():
    # The requirement rides the bom.md spec block (single source; the
    # _stage_extra restatement was deduped 2026-07-19 review §7.2).
    sysmsg = build_system("bom")
    assert "Decoupling completeness" in sysmsg
    assert "per dedicated supply/decoupling pin" in sysmsg


def test_architecture_spec_declares_power_rails_are_not_sheets():
    # The architecture spec must resolve the power-block contradiction: a
    # functional-spec block whose category is `power` is a net, never a sheet.
    sysmsg = build_system("architecture")
    assert "power/ground NETS" in sysmsg
    assert "never emit a Sheet" in sysmsg


def test_bom_reconcile_instruction_lists_the_missing_parts():
    from kicraft.server.web import _bom_reconcile_instruction
    instr = _bom_reconcile_instruction(
        [{"text": "Add three 100nF caps for U1 DEC3-DEC5", "reconcile_target": "bom"},
         {"text": "", "reconcile_target": "bom"}])   # blank text is skipped
    assert "Add three 100nF caps for U1 DEC3-DEC5" in instr
    assert "Do NOT ask the user" in instr
    assert instr.count("\n- ") == 1   # only the one non-blank deficit line


def test_retry_feedback_unknown_ref_in_wiring_points_at_reconcile(tmp_path):
    # WS6: an unknown-ref rejection in wiring must tell the model it cannot add
    # parts and to park with reconcile_target=bom, and list the real refs -- so it
    # stops inventing refs and burning its retry budget.
    from kicraft.server.stage_driver import _retry_feedback
    out = {"errors": ["NetConnection 'PWR' references unknown ref 'Q99'"]}
    msg = _retry_feedback(out, stage="wiring", valid_refs=["C1", "U1"])
    assert "CANNOT add parts" in msg
    assert 'reconcile_target' in msg and '"bom"' in msg
    assert "C1" in msg and "U1" in msg


def test_retry_feedback_no_reconcile_note_for_non_wiring_stage():
    from kicraft.server.stage_driver import _retry_feedback
    out = {"errors": ["some symbol not found"]}
    msg = _retry_feedback(out, stage="bom")
    assert "reconcile_target" not in msg


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


# ---- in-stream reasoning-loop breakout (KC-VWW5X7) -------------------------

class _LoopGuard:
    def status(self):
        return {"spent_total_usd": 0.0}


class _LoopClient:
    """First N chat replies are a reasoning-loop abort; the next is a valid
    intent slot. Records the `reasoning` policy each call received."""
    def __init__(self, loop_replies, ok_json):
        self.loop_replies = loop_replies
        self.ok_json = ok_json
        self.reasoning_seen = []
        self.guard = _LoopGuard()
        self._n = 0

    def chat(self, messages, max_tokens=4096, temperature=0.2, progress=None,
             meta_ctx=None, reasoning=None):
        self.reasoning_seen.append(reasoning)
        self._n += 1
        if self._n <= self.loop_replies:
            return {"text": "", "reasoning": "x" * 600, "finish_reason": "reasoning_loop",
                    "loop_detected": True, "cost_usd": 0.0}
        return {"text": self.ok_json, "reasoning": "", "finish_reason": "stop",
                "loop_detected": False, "cost_usd": 0.0}


_OK_INTENT = json.dumps({
    "goal": "a USB-powered LED", "constraints": [], "named_parts": [],
    "inferred_expertise": "intermediate", "assumptions": [],
    "project_stem": "USB_LED",
})


def test_loop_detected_retries_reasoning_disabled_then_commits(tmp_path):
    client = _LoopClient(loop_replies=1, ok_json=_OK_INTENT)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"
    assert len(client.reasoning_seen) == 2
    assert client.reasoning_seen[1] == {"enabled": False}  # loop retry drops reasoning


def test_second_loop_fails_with_reasoning_loop_label(tmp_path):
    client = _LoopClient(loop_replies=99, ok_json=_OK_INTENT)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert last.get("error") == "reasoning_loop"
    assert len(client.reasoning_seen) == 2  # one initial + one anti-loop retry, then give up


def test_design_reasoning_policy_selection():
    class _S:
        def design_reasoning(self, stage):
            if stage in ("intent", "functional_spec"):
                return {"enabled": False}
            return {"max_tokens": 2048}

    class _C:
        s = _S()

    assert _design_reasoning(_C(), "intent") == {"enabled": False}
    assert _design_reasoning(_C(), "functional_spec") == {"enabled": False}
    assert _design_reasoning(_C(), "architecture") == {"max_tokens": 2048}
    assert _design_reasoning(object(), "intent") is None  # mock: no .s policy
