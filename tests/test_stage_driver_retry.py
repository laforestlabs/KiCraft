"""Tests for the stage driver's self-correction feedback and per-stage retries.

Pure functions, no OpenRouter / network: exercises the retry-message construction
and the per-stage retry budget that help the wiring stage converge.
"""
from __future__ import annotations

from kicraft.server.stage_driver import _retry_feedback, _stage_max_retries


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
    assert _stage_max_retries("bom", 2) == 2


def test_caller_default_wins_when_higher_than_the_floor():
    assert _stage_max_retries("wiring", 6) == 6
    assert _stage_max_retries("intent", 6) == 6
