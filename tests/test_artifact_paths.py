"""Unit tests for the central artifact resolver + freshness gate
(``kicraft/cli/artifact_paths.py``).

These pin the contract that fixes the ``replay --no-route`` stale-board trap:
intent-based resolution (placed never returns routed), deterministic newest-dir
selection, and a freshness gate that trusts a positive run_id match but falls
back to mtime so a freshly *placed* board (whose metadata.json is NOT rewritten)
is never wrongly rejected. See docs/ARTIFACTS.md."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from kicraft.cli import artifact_paths as ap


@pytest.fixture(autouse=True)
def _isolate_run_env():
    """Save/restore the run-identity env vars so ensure_run_context() in one test
    can't leak a run_id into the next."""
    saved = {k: os.environ.get(k) for k in (ap.ENV_RUN_ID, ap.ENV_RUN_STARTED_AT)}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _touch(p: Path, text: str = "(kicad_pcb)\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _set_mtime(p: Path, ts: float) -> None:
    os.utime(p, (ts, ts))


def _pdir(root: Path, slug: str = "subcircuit__abc") -> Path:
    return root / ".experiments" / "subcircuits" / slug


# --- intent-based resolution --------------------------------------------------

def test_placed_never_returns_routed(tmp_path):
    """The core fix: with BOTH boards present, kind='placed' returns the placement
    and kind='routed' returns the routed board. placed never falls back to routed."""
    d = _pdir(tmp_path)
    placed = _touch(d / ap.PARENT_PLACED)
    routed = _touch(d / ap.PARENT_ROUTED)
    assert ap.resolve_parent_board(tmp_path, kind="placed") == placed
    assert ap.resolve_parent_board(tmp_path, kind="routed") == routed


def test_routed_is_none_when_only_placed_exists(tmp_path):
    d = _pdir(tmp_path)
    placed = _touch(d / ap.PARENT_PLACED)
    assert ap.resolve_parent_board(tmp_path, kind="routed") is None
    assert ap.resolve_parent_board(tmp_path, kind="placed") == placed


def test_latest_parent_dir_picks_newest_by_mtime(tmp_path):
    """When stale parent dirs accumulate, the newest (by board/metadata mtime) is
    chosen -- deterministic, unlike iterdir first-match or alphabetical sort."""
    old = _pdir(tmp_path, "subcircuit__old")
    new = _pdir(tmp_path, "subcircuit__new")
    _touch(old / ap.PARENT_PLACED)
    new_board = _touch(new / ap.PARENT_PLACED)
    _set_mtime(old / ap.PARENT_PLACED, 1_000.0)
    _set_mtime(new_board, 2_000.0)
    assert ap.latest_parent_artifact_dir(tmp_path) == new
    assert ap.resolve_parent_board(tmp_path, kind="placed") == new_board


def test_best_leaf_tier_order(tmp_path):
    """Routed leaf beats placed beats the rejected stamp; None when nothing."""
    root = tmp_path / ".experiments" / "subcircuits"
    assert ap.resolve_best_leaf_board(tmp_path) is None
    _touch(root / "leafX" / ap.LEAF_ILLEGAL)
    _touch(root / "leafX" / "round_0000_leaf_pre_freerouting.kicad_pcb")
    routed = _touch(root / "leafX" / "round_0000_leaf_routed.kicad_pcb")
    assert ap.resolve_best_leaf_board(tmp_path) == routed


# --- freshness gate -----------------------------------------------------------

def test_positive_runid_match_overrides_old_mtime(tmp_path):
    """A run_id match is authoritative -- immune to clock skew / ancient mtime."""
    d = _pdir(tmp_path)
    board = _touch(d / ap.PARENT_ROUTED)
    _touch(d / ap.METADATA_JSON, json.dumps({"run_id": "R1"}))
    _set_mtime(board, 1_000.0)  # ancient
    assert ap.produced_by_this_run(board, run_id="R1", run_started_at=time.time())


def test_runid_mismatch_falls_back_to_mtime(tmp_path):
    """The --no-route stamp-only case: metadata.json run_id is STALE but the board
    was freshly re-saved -> mtime says fresh -> accepted (never reject on
    mismatch). And a genuinely old board is rejected."""
    d = _pdir(tmp_path)
    board = _touch(d / ap.PARENT_PLACED)
    _touch(d / ap.METADATA_JSON, json.dumps({"run_id": "OLD"}))
    started = time.time() - 100.0
    _set_mtime(board, time.time())  # fresh write this run
    assert ap.produced_by_this_run(board, run_id="NEW", run_started_at=started)
    _set_mtime(board, started - 50.0)  # from a previous run
    assert not ap.produced_by_this_run(board, run_id="NEW", run_started_at=started)


def test_no_metadata_uses_mtime(tmp_path):
    d = _pdir(tmp_path)
    board = _touch(d / ap.PARENT_PLACED)
    started = time.time() - 10.0
    _set_mtime(board, time.time())
    assert ap.produced_by_this_run(board, run_id="X", run_started_at=started)
    _set_mtime(board, started - 100.0)
    assert not ap.produced_by_this_run(board, run_id="X", run_started_at=started)


def test_no_run_context_is_permissive(tmp_path):
    """Ad-hoc tooling with no run context gets legacy permissive behavior."""
    board = _touch(_pdir(tmp_path) / ap.PARENT_PLACED)
    assert ap.produced_by_this_run(board, run_id=None, run_started_at=None)


# --- provenance + run context -------------------------------------------------

def test_provenance_path_single_suffix():
    got = ap.provenance_path(Path("/x/USB_PD_TRIGGER.kicad_pcb"))
    assert got.name == "USB_PD_TRIGGER.provenance.json"


def test_promote_provenance_roundtrip(tmp_path):
    pcb = _touch(tmp_path / "PROJ.kicad_pcb")
    src = _touch(_pdir(tmp_path) / ap.PARENT_ROUTED)
    _touch(src.parent / ap.METADATA_JSON, json.dumps({"run_id": "RSRC"}))
    out = ap.write_promote_provenance(
        pcb, run_id="RUN", run_started_at=123.0,
        source_board=src, source_kind="routed", fresh=True,
    )
    assert out == ap.provenance_path(pcb)
    data = ap.read_provenance(pcb)
    assert data["run_id"] == "RUN"
    assert data["source_kind"] == "routed"
    assert data["source_run_id"] == "RSRC"  # read from the source's metadata
    assert data["fresh"] is True
    assert data["md5"]  # fingerprint computed


def test_ensure_run_context_idempotent():
    rid1, t1 = ap.ensure_run_context()
    rid2, t2 = ap.ensure_run_context()
    assert rid1 == rid2 and t1 == t2
    assert ap.current_run_id() == rid1
    assert ap.current_run_started_at() == t1


def test_ensure_run_context_honors_injected_id():
    os.environ[ap.ENV_RUN_ID] = "web-driver-123"
    rid, _ = ap.ensure_run_context()
    assert rid == "web-driver-123"
