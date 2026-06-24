"""_hoist_positionals: let a subcommand's positional appear in any position.

Python 3.12 argparse refuses to bind an optional (nargs="?") positional placed
after an option. The CLI normalizes token order before parsing so users can
write `kicraft stage-commit intent --slot-file x state.json`. These tests pin
the reorder algorithm and, crucially, its conservatism: anything it cannot
classify with certainty must be returned unchanged (so it can only fix the
broken ordering, never break a working one).
"""
from __future__ import annotations

import argparse

import pytest

from kicraft.design.cli_app import _hoist_positionals


def _parser() -> argparse.ArgumentParser:
    """A stand-in mirroring the real subcommands' option/positional arities."""
    ap = argparse.ArgumentParser(prog="kicraft")
    sub = ap.add_subparsers()

    c = sub.add_parser("stage-commit")          # 1 required + 1 optional positional
    c.add_argument("stage")
    c.add_argument("--slot-file", required=True)   # single-value option
    c.add_argument("--project-stem")               # single-value option
    c.add_argument("state", nargs="?", default=".kicraft/state.json")
    c.add_argument("--no-archive", action="store_true")  # flag (0 values)

    r = sub.add_parser("replay")                # TWO optional positionals
    r.add_argument("state", nargs="?")
    r.add_argument("out_dir", nargs="?")
    r.add_argument("--project")
    r.add_argument("--no-route", action="store_true")
    return ap


def hoist(*argv: str) -> list[str]:
    return _hoist_positionals(_parser(), list(argv))


def test_state_after_options_is_hoisted_ahead():
    assert hoist("stage-commit", "intent", "--slot-file", "s.json",
                 "--no-archive", "state.json") == [
        "stage-commit", "intent", "state.json", "--slot-file", "s.json", "--no-archive"]


def test_single_value_option_keeps_its_value():
    # the value of --slot-file must NOT be mistaken for the state positional
    out = hoist("stage-commit", "intent", "--slot-file", "s.json", "st.json")
    assert out == ["stage-commit", "intent", "st.json", "--slot-file", "s.json"]


def test_already_positionals_first_is_unchanged():
    argv = ["stage-commit", "intent", "state.json", "--slot-file", "s.json", "--no-archive"]
    assert hoist(*argv) == argv


def test_two_optional_positionals_preserve_relative_order():
    assert hoist("replay", "--no-route", "state.json", "out/") == [
        "replay", "state.json", "out/", "--no-route"]
    # ...and when only options are used (positionals absent), order is stable
    assert hoist("replay", "--project", "foo") == ["replay", "--project", "foo"]


def test_opt_equals_value_is_self_contained():
    out = hoist("stage-commit", "intent", "--slot-file=s.json", "state.json")
    assert out == ["stage-commit", "intent", "state.json", "--slot-file=s.json"]


def test_bails_on_unknown_or_abbreviated_option():
    # an abbreviation argparse would still accept, but we cannot classify its
    # arity from the option-string map -> return unchanged (argparse then handles
    # it exactly as before).
    argv = ["stage-commit", "intent", "--slot", "s.json", "state.json"]
    assert hoist(*argv) == argv


def test_bails_on_double_dash_marker():
    argv = ["stage-commit", "intent", "--slot-file", "s.json", "--", "state.json"]
    assert hoist(*argv) == argv


def test_non_subcommand_passthrough():
    assert hoist("-h") == ["-h"]
    assert hoist() == []
    assert hoist("totally-unknown", "--x", "y") == ["totally-unknown", "--x", "y"]
