"""Integration tests for the run-scoped freshness gate AT THE PROMOTE SITES.

``tests/test_artifact_paths.py`` pins the gate *function*
(``produced_by_this_run``) in isolation; these pin its *wiring* into the three
places the build tail actually promotes a board, which is where the original
``replay --no-route`` stale-board trap bit:

1. routed fab promote (``_promote_verify_fab``): a routed parent from a PREVIOUS
   run must be ignored -- fall through to the rc6 inspection preview rather than
   ship a stale board as fab-ready.
2. ``--no-route`` placed promote (``_layout_route_fab``): a placed parent that
   isn't from this run is a HARD rc6 error -- never silently promote a previous
   run's placement.
3. rc6 best-partial preview (``_promote_verify_fab``): this path exists to show
   "whatever this run got" on failure, so a non-fresh partial is a WARNING (still
   shown, ``fresh=False`` in provenance), not a hard error.

These call the real production functions, stubbing only ``_run_layout`` (the
layout subprocess) so the gate logic runs unmocked. See docs/ARTIFACTS.md."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kicraft.cli import artifact_paths as ap
from kicraft.design import cli_app


@pytest.fixture(autouse=True)
def _isolate_run_env():
    """Save/restore the run-identity env vars so one test's run_id can't leak."""
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


def _write_meta(sub_dir: Path, run_id: str) -> None:
    (sub_dir / ap.METADATA_JSON).write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )


def _set_run(run_id: str, started_at: float) -> None:
    """Pin THIS run's identity so ensure_run_context() inside the promote code
    returns a known (run_id, run_started_at) we can judge boards against."""
    os.environ[ap.ENV_RUN_ID] = run_id
    os.environ[ap.ENV_RUN_STARTED_AT] = repr(started_at)


def _sub(project: Path, slug: str = "subcircuit__abc") -> Path:
    return project / ".experiments" / "subcircuits" / slug


def _noroute_args() -> SimpleNamespace:
    return SimpleNamespace(
        quality="draft", seed=0, route=False, no_fab=True,
        done_label="DONE", no_archive=True,
    )


# --- site 2: --no-route placed promote ----------------------------------------

def test_noroute_promote_rejects_stale_placed_board(tmp_path, monkeypatch):
    """A placed parent from a PREVIOUS run (old mtime + mismatched metadata
    run_id) is refused with a hard rc6 -- the core --no-route stale-board fix.
    Nothing is promoted to <stem>.kicad_pcb."""
    t0 = time.time()
    _set_run("NOW", t0)
    stem = "BOARD"
    pcb = tmp_path / f"{stem}.kicad_pcb"
    sub = _sub(tmp_path)
    placed = _touch(sub / ap.PARENT_PLACED)
    _write_meta(sub, run_id="OLD")
    _set_mtime(placed, t0 - 100)  # predates this run -> stale

    # _run_layout "succeeds" but (as in the bug) leaves the stale board in place.
    monkeypatch.setattr(cli_app, "_run_layout", lambda *a, **k: 0)

    rc = cli_app._layout_route_fab(
        _noroute_args(), None, tmp_path / "state.json", None, None,
        stem, tmp_path, tmp_path / "root.kicad_sch", pcb,
    )

    assert rc == 6
    assert not pcb.exists(), "a stale placed board must NOT be promoted"
    assert ap.read_provenance(pcb) is None


def test_noroute_promote_accepts_fresh_placed_board(tmp_path, monkeypatch):
    """A placed parent written by THIS run (fresh mtime) is promoted, with
    provenance recording source_kind=placed, fresh=True, and this run's id."""
    t0 = time.time()
    _set_run("NOW", t0)
    stem = "BOARD"
    pcb = tmp_path / f"{stem}.kicad_pcb"
    sub = _sub(tmp_path)
    placed = _touch(sub / ap.PARENT_PLACED)
    _set_mtime(placed, t0 + 10)  # written this run -> fresh

    monkeypatch.setattr(cli_app, "_run_layout", lambda *a, **k: 0)

    rc = cli_app._layout_route_fab(
        _noroute_args(), None, tmp_path / "state.json", None, None,
        stem, tmp_path, tmp_path / "root.kicad_sch", pcb,
    )

    assert rc == 0
    assert pcb.exists()
    prov = ap.read_provenance(pcb)
    assert prov is not None
    assert prov["source_kind"] == "placed"
    assert prov["fresh"] is True
    assert prov["run_id"] == "NOW"
    assert prov["source_board"].endswith(ap.PARENT_PLACED)


# --- site 1: routed fab promote -----------------------------------------------

def test_fab_promote_ignores_stale_routed_and_falls_through(tmp_path):
    """A routed parent from a previous run must NOT be promoted as fab-ready: the
    gate falls through to the rc6 partial preview, so the promoted board is the
    (fresh) placed parent -- source_kind=partial, never the stale routed board."""
    t0 = time.time()
    _set_run("NOW", t0)
    stem = "BOARD"
    pcb = tmp_path / f"{stem}.kicad_pcb"
    sub = _sub(tmp_path)
    routed = _touch(sub / ap.PARENT_ROUTED)
    placed = _touch(sub / ap.PARENT_PLACED)
    _write_meta(sub, run_id="OLD")
    _set_mtime(routed, t0 - 100)  # stale routed from a previous run
    _set_mtime(placed, t0 + 10)   # but this run did place a parent

    rc = cli_app._promote_verify_fab(
        None, tmp_path / "state.json", [], stem, tmp_path, pcb,
    )

    assert rc == 6, "a stale routed parent must not pass the fab gate"
    prov = ap.read_provenance(pcb)
    assert prov is not None
    assert prov["source_kind"] == "partial", "must promote the partial, not routed"
    assert prov["source_board"].endswith(ap.PARENT_PLACED)
    assert prov["fresh"] is True  # the partial itself IS from this run


# --- site 3: rc6 best-partial preview -----------------------------------------

def test_rc6_preview_shows_stale_partial_flagged_not_fresh(tmp_path):
    """The rc6 preview exists to show what this run reached, so even a non-fresh
    partial is still surfaced (warning, not hard error): the board is promoted for
    inspection but provenance marks fresh=False."""
    t0 = time.time()
    _set_run("NOW", t0)
    stem = "BOARD"
    pcb = tmp_path / f"{stem}.kicad_pcb"
    sub = _sub(tmp_path)
    placed = _touch(sub / ap.PARENT_PLACED)  # no routed parent at all
    _write_meta(sub, run_id="OLD")
    _set_mtime(placed, t0 - 100)  # stale partial

    rc = cli_app._promote_verify_fab(
        None, tmp_path / "state.json", [], stem, tmp_path, pcb,
    )

    assert rc == 6
    assert pcb.exists(), "rc6 preview must still surface a board for inspection"
    prov = ap.read_provenance(pcb)
    assert prov is not None
    assert prov["source_kind"] == "partial"
    assert prov["fresh"] is False, "a non-fresh partial must be flagged, not hidden"


# --- KC-9G4YPT GAP 2: pre-promote seed snapshot + replay restore ---------------

def test_rc6_promote_snapshots_seed_and_replay_restore_recovers_it(tmp_path):
    """The rc6 partial promote deliberately clobbers <stem>.kicad_pcb (the
    preview must show what the build reached) -- but it must snapshot the
    full-component seed first, and the replay-side restore must bring the seed
    back and drop the now-inaccurate provenance. Without this, every rc6 run
    is unreplayable (leaf extraction: "no matching components")."""
    t0 = time.time()
    _set_run("NOW", t0)
    stem = "BOARD"
    pcb = _touch(tmp_path / f"{stem}.kicad_pcb", "(kicad_pcb FULL-SEED)\n")
    sub = _sub(tmp_path)
    placed = _touch(sub / ap.PARENT_PLACED, "(kicad_pcb PARTIAL)\n")
    _set_mtime(placed, t0 + 10)  # fresh partial, no routed parent -> rc6

    rc = cli_app._promote_verify_fab(
        None, tmp_path / "state.json", [], stem, tmp_path, pcb,
    )

    assert rc == 6
    assert pcb.read_text() == "(kicad_pcb PARTIAL)\n"  # preview promote intact
    snap = ap.pre_promote_seed_path(tmp_path)
    assert snap.is_file() and snap.read_text() == "(kicad_pcb FULL-SEED)\n"

    cli_app._restore_pre_promote_seed(tmp_path, pcb)

    assert pcb.read_text() == "(kicad_pcb FULL-SEED)\n"
    assert ap.read_provenance(pcb) is None, "stale partial provenance must go"


def test_replay_restore_errors_when_snapshot_missing(tmp_path):
    """Pre-fix rc6 runs have partial provenance but no snapshot: the restore
    must fail with the honest remedy, not let replay die later with the
    misleading 'no matching components' from inside leaf extraction."""
    t0 = time.time()
    _set_run("NOW", t0)
    pcb = _touch(tmp_path / "BOARD.kicad_pcb", "(kicad_pcb PARTIAL)\n")
    src = _touch(_sub(tmp_path) / ap.PARENT_PLACED, "(kicad_pcb PARTIAL)\n")
    ap.write_promote_provenance(
        pcb, run_id="NOW", run_started_at=t0,
        source_board=src, source_kind="partial", fresh=True,
    )

    with pytest.raises(cli_app._ReplayInputError, match="predates seed snapshotting"):
        cli_app._restore_pre_promote_seed(tmp_path, pcb)
    assert pcb.read_text() == "(kicad_pcb PARTIAL)\n"  # untouched on error


def test_replay_restore_noop_for_routed_and_unprovenanced_boards(tmp_path):
    """rc0/rc7 (routed promote) and plain boards with no provenance replay
    as-is -- the restore must not touch them even when a snapshot exists."""
    t0 = time.time()
    _set_run("NOW", t0)
    _touch(ap.pre_promote_seed_path(tmp_path), "(kicad_pcb OLD-SEED)\n")

    routed_pcb = _touch(tmp_path / "BOARD.kicad_pcb", "(kicad_pcb ROUTED)\n")
    src = _touch(_sub(tmp_path) / ap.PARENT_ROUTED, "(kicad_pcb ROUTED)\n")
    ap.write_promote_provenance(
        routed_pcb, run_id="NOW", run_started_at=t0,
        source_board=src, source_kind="routed", fresh=True,
    )
    cli_app._restore_pre_promote_seed(tmp_path, routed_pcb)
    assert routed_pcb.read_text() == "(kicad_pcb ROUTED)\n"
    assert ap.read_provenance(routed_pcb) is not None

    ap.provenance_path(routed_pcb).unlink()  # now: no provenance at all
    cli_app._restore_pre_promote_seed(tmp_path, routed_pcb)
    assert routed_pcb.read_text() == "(kicad_pcb ROUTED)\n"


def test_rc6_preview_marks_fresh_partial_fresh(tmp_path):
    """A partial board from this run is shown with fresh=True (the normal rc6
    case: placed+composed but the parent never routed)."""
    t0 = time.time()
    _set_run("NOW", t0)
    stem = "BOARD"
    pcb = tmp_path / f"{stem}.kicad_pcb"
    sub = _sub(tmp_path)
    placed = _touch(sub / ap.PARENT_PLACED)
    _set_mtime(placed, t0 + 10)  # fresh

    rc = cli_app._promote_verify_fab(
        None, tmp_path / "state.json", [], stem, tmp_path, pcb,
    )

    assert rc == 6
    assert pcb.exists()
    prov = ap.read_provenance(pcb)
    assert prov is not None
    assert prov["source_kind"] == "partial"
    assert prov["fresh"] is True
