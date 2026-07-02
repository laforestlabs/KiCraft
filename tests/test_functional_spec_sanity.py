"""Unit tests for the R6 functional_spec sanity gate in
``kicraft.design.cli_app._cmd_stage_commit``.

The gate fires when ``stage == "functional_spec"`` and the committed
``FunctionalSpec`` is present, and enforces three deterministic invariants:

* no self-loop connections (``from_block == to_block``),
* no fully isolated block when there is more than one block (every block
  must appear in at least one connection's ``from_block`` or ``to_block``),
* the block count must be between 1 and 12 inclusive.

On failure the command prints ``{"ok": false, "errors": [...]}`` and returns
``3``; on success it prints ``{"ok": true, ...}`` and returns ``0``.

The tests drive the real CLI (``cli_app.main``) with a temp ``state.json``
that already carries an ``intent`` slot (required so the commit does not fail
on a missing upstream intent) and a ``--slot-file`` holding the proposed
``functional_spec``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.design.cli_app import main
from kicraft.design.models import (
    BlockConnection,
    ConversationState,
    FunctionalBlock,
    FunctionalSpec,
    IntentSlot,
)


def _write_state(tmp_path: Path) -> Path:
    """Write a state.json with the intent slot populated."""
    state = ConversationState(intent=IntentSlot(goal="test design"))
    state_path = tmp_path / "state.json"
    state_path.write_text(state.model_dump_json(indent=2) + "\n")
    return state_path


def _write_slot(tmp_path: Path, spec: FunctionalSpec) -> Path:
    slot_path = tmp_path / "functional_spec_slot.json"
    slot_path.write_text(json.dumps(spec.model_dump()))
    return slot_path


def _commit(
    tmp_path: Path, capsys: pytest.CaptureFixture, spec: FunctionalSpec
) -> tuple[int, dict]:
    state_path = _write_state(tmp_path)
    slot_path = _write_slot(tmp_path, spec)
    rc = main(
        [
            "stage-commit",
            "functional_spec",
            str(state_path),
            "--slot-file",
            str(slot_path),
            "--no-archive",
        ]
    )
    out = capsys.readouterr().out.strip()
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        payload = {"_raw_stdout": out}
    return rc, payload


# --- rejection cases --------------------------------------------------------


def test_rejects_self_loop(tmp_path, capsys):
    spec = FunctionalSpec(
        blocks=[
            FunctionalBlock(name="POWER", category="power", purpose="regulate", count=1),
            FunctionalBlock(name="MCU", category="process", purpose="compute", count=1),
        ],
        connections=[
            BlockConnection(from_block="POWER", to_block="MCU", signal_type="power"),
            BlockConnection(from_block="MCU", to_block="MCU", signal_type="digital"),
        ],
    )
    rc, payload = _commit(tmp_path, capsys, spec)
    assert rc == 3
    assert payload["ok"] is False
    assert any("self-loop" in e.lower() for e in payload["errors"]), payload


def test_rejects_isolated_block(tmp_path, capsys):
    # ORPHAN never appears in any connection's from_block or to_block, so it is
    # fully isolated while len(blocks) > 1 — the isolated check fires.
    spec = FunctionalSpec(
        blocks=[
            FunctionalBlock(name="POWER", category="power", purpose="regulate", count=1),
            FunctionalBlock(name="SENSOR", category="sense", purpose="measure", count=1),
            FunctionalBlock(name="ORPHAN", category="process", purpose="unused", count=1),
        ],
        connections=[
            BlockConnection(from_block="POWER", to_block="SENSOR", signal_type="power"),
        ],
    )
    rc, payload = _commit(tmp_path, capsys, spec)
    assert rc == 3
    assert payload["ok"] is False
    assert any("isolated" in e.lower() for e in payload["errors"]), payload


def test_rejects_too_many_blocks(tmp_path, capsys):
    spec = FunctionalSpec(
        blocks=[
            FunctionalBlock(name=f"B{i:02d}", category="process", purpose="p", count=1)
            for i in range(13)
        ],
        # Chain them so only the count check fires.
        connections=[
            BlockConnection(
                from_block=f"B{i:02d}", to_block=f"B{i + 1:02d}", signal_type="digital"
            )
            for i in range(12)
        ],
    )
    rc, payload = _commit(tmp_path, capsys, spec)
    assert rc == 3
    assert payload["ok"] is False
    assert any("max 12" in e for e in payload["errors"]), payload


# --- acceptance case --------------------------------------------------------


def test_accepts_valid_single_block(tmp_path, capsys):
    # A single block with no connections is allowed: the isolated check only
    # applies when len(blocks) > 1.
    spec = FunctionalSpec(
        blocks=[
            FunctionalBlock(name="CORE", category="process", purpose="test", count=1),
        ],
        connections=[],
    )
    rc, payload = _commit(tmp_path, capsys, spec)
    assert rc == 0
    assert payload["ok"] is True
