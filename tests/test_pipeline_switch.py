"""Guards for the pipeline switch (design-yield-recovery plan option 3, §4.3.1 guard 3).

A swap between the two design pipelines must be visible everywhere it matters, and the
dispatch must be readable from the durable config without a restart.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from kicraft.server import pipeline, routing_config
from kicraft.server.accounts import AccountStore


def test_routing_config_round_trips_the_pipeline(tmp_path):
    path = tmp_path / "routing.json"
    routing_config.save(
        routing_config.RoutingConfig(active_profile="luna", pipeline="legacy"), path
    )
    loaded = routing_config.load(path)
    assert loaded.pipeline == "legacy"
    assert json.loads(path.read_text())["pipeline"] == "legacy"


def test_unknown_pipeline_is_rejected():
    with pytest.raises(ValueError):
        routing_config.validate(
            routing_config.RoutingConfig(active_profile="luna", pipeline="august")
        )
    # A hand-edited file with a bad value drops the key rather than selecting it.
    loaded = routing_config.from_dict({"active_profile": "luna", "pipeline": "august"})
    assert loaded.pipeline == ""


def test_apply_overlays_the_pipeline_on_settings():
    from kicraft.server.config import Settings

    settings = routing_config.RoutingConfig(active_profile="luna", pipeline="legacy").apply(
        Settings(api_key="k")
    )
    assert settings.pipeline == "legacy"


def test_workspace_marker_is_what_the_build_reads(tmp_path):
    """A native design keeps its backend even if the configured default changes.

    The marker is written when the design is dispatched, so a workspace's pipeline is a
    property of the workspace, not of the config at build time.
    """
    ws = tmp_path / "ws"
    assert pipeline.read_marker(ws) is None
    pipeline.write_marker(ws, pipeline.PIPELINE_LEGACY)
    assert pipeline.read_marker(ws) == "legacy"
    marker = json.loads((ws / ".kicraft" / "pipeline.json").read_text())
    assert marker["legacy_commit"] == pipeline.LEGACY_COMMIT


def test_project_row_records_the_pipeline(tmp_path):
    """The project row is where a user's board says which pipeline built it."""
    store = AccountStore(db_path=tmp_path / "accounts.db", projects_dir=tmp_path / "projects")
    pid = store.create_project(1, "a board")
    store.finish_project(pid, "ok", stem="BOARD", pipeline="legacy")
    row = store.get_project(pid)
    assert row is not None and row.pipeline == "legacy"


def test_existing_database_gains_the_pipeline_column(tmp_path):
    """The production box's accounts.db predates this column: it must migrate in place."""
    import sqlite3

    db = tmp_path / "accounts.db"
    conn = sqlite3.connect(db)
    # The production schema minus `pipeline` (the column this migration adds).
    conn.execute(
        "CREATE TABLE projects (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,"
        " brief TEXT NOT NULL, project_stem TEXT, status TEXT NOT NULL DEFAULT 'running',"
        " created_at TEXT NOT NULL, finished_at TEXT, cost_usd REAL, dir_path TEXT,"
        " zip_path TEXT, viewed_at TEXT, is_public INTEGER NOT NULL DEFAULT 1,"
        " cloned_from_id INTEGER, view_count INTEGER NOT NULL DEFAULT 0,"
        " clone_count INTEGER NOT NULL DEFAULT 0, like_count INTEGER NOT NULL DEFAULT 0,"
        " quality TEXT, board_code TEXT)"
    )
    conn.execute(
        "INSERT INTO projects (id, user_id, brief, created_at) VALUES (7, 1, 'old', '2026-09-01')"
    )
    conn.commit()
    conn.close()
    store = AccountStore(db_path=db, projects_dir=tmp_path / "projects")
    row = store.get_project(7)
    assert row is not None
    # NULL reads as "built before the switch existed", i.e. the current pipeline.
    assert row.pipeline is None
    store.finish_project(7, "ok", pipeline="legacy")
    assert store.get_project(7).pipeline == "legacy"


def test_current_reconcile_never_starts_a_second_legacy_budget(tmp_path):
    """An exhausted legacy repair park remains a user-visible park, not a new run."""
    from kicraft.server import session as session_mod

    pipeline.write_marker(tmp_path, pipeline.PIPELINE_LEGACY)
    res = {
        "status": "awaiting_input",
        "last_stage": "wiring",
        "questions": [
            {
                "text": "Add a 1uF capacitor for U1.",
                "blocking": True,
                "reconcile_target": "bom",
            }
        ],
    }
    repaired, passes = session_mod.maybe_bom_reconcile(tmp_path, "a brief", res, reconcile_passes=0)
    assert repaired is res
    assert passes == 0


def test_selected_falls_back_to_current_when_the_legacy_tree_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        routing_config, "load", lambda *a, **k: routing_config.RoutingConfig(pipeline="legacy")
    )
    monkeypatch.setattr(pipeline, "legacy_available", lambda root=None: False)
    assert pipeline.selected() == "current"


def test_pipeline_is_named_in_the_report_and_the_scorecard(tmp_path):
    """§4.3.1 guard 3: provenance and the scorecard group by pipeline."""
    from kicraft.eval.self_eval import compile_report

    records = [
        {"index": 1, "run_id": "a", "slug": "a", "pipeline": "current", "build_rc": 0},
        {"index": 2, "run_id": "b", "slug": "b", "pipeline": "legacy", "build_rc": 0},
    ]
    report = compile_report(records, tmp_path, {"out_dir": str(tmp_path), "repeats": 1})
    assert report["pipeline_counts"] == {"current": 1, "legacy": 1}


def test_promote_provenance_carries_the_pipeline(tmp_path):
    from kicraft.cli.artifact_paths import (
        provenance_path,
        read_provenance,
        write_promote_provenance,
    )

    pcb = tmp_path / "BOARD.kicad_pcb"
    pcb.write_text("(kicad_pcb)")
    write_promote_provenance(
        pcb,
        run_id="run1",
        run_started_at=1.0,
        source_board=pcb,
        source_kind="routed",
        fresh=True,
        advisories=["unreviewed_exact_part"],
        pipeline=pipeline.provenance_fields("legacy"),
    )
    payload = read_provenance(pcb)
    assert payload is not None
    assert payload["pipeline"] == "legacy"
    assert payload["pipeline_legacy_commit"] == pipeline.LEGACY_COMMIT
    assert payload["advisories"] == ["unreviewed_exact_part"]
    assert provenance_path(pcb).is_file()


def test_parked_legacy_run_surfaces_its_native_question(monkeypatch, tmp_path):
    """A genuine legacy ambiguity is surfaced from its protocol result."""
    from kicraft.server import session as session_mod

    native = {
        "status": "awaiting_input",
        "results": [{"stage": "wiring", "commit_ok": False, "needs_input": True}],
        "guard": {"spent_total_usd": 0.02},
        "questions": [
            {
                "stage": "wiring",
                "text": "Add a 1x05 SWD programming header, then re-run wiring.",
                "blocking": True,
                "options": [],
                "answer": None,
            }
        ],
        "last_stage": "wiring",
    }

    def fake_run(*args, **kwargs):
        return (
            1,
            pipeline._LEGACY_SESSION_RESULT_PREFIX
            + json.dumps({"result": native, "reconcile_passes": 0}),
            "",
        )

    monkeypatch.setattr(session_mod.pipeline_dispatch, "run_legacy_design", fake_run)
    out = session_mod._run_legacy_session(tmp_path, "a brief", ["intent", "bom", "wiring"])
    assert out["status"] == "awaiting_input"
    assert out["last_stage"] == "wiring"
    assert "SWD programming header" in out["questions"][0]["text"]


@pytest.mark.parametrize(
    "stdout",
    ["legacy runner crashed", pipeline._LEGACY_SESSION_RESULT_PREFIX + "{not-json"],
)
def test_missing_or_malformed_legacy_result_fails_closed(monkeypatch, tmp_path, stdout):
    """Stale state must never become a result when the subprocess protocol breaks."""
    from kicraft.server import session as session_mod

    def fake_run(*args, **kwargs):
        return 0, stdout, ""

    monkeypatch.setattr(session_mod.pipeline_dispatch, "run_legacy_design", fake_run)
    out = session_mod._run_legacy_session(tmp_path, "a brief", ["wiring"])
    assert out["status"] == "failed"
    assert out["failure_kind"] == "legacy_protocol_error"
    assert out["questions"] is None


def test_nonzero_legacy_exit_cannot_report_success(monkeypatch, tmp_path):
    """A contradictory successful packet is a process-boundary failure."""
    from kicraft.server import session as session_mod

    def fake_run(*args, **kwargs):
        return (
            1,
            pipeline._LEGACY_SESSION_RESULT_PREFIX
            + json.dumps({"result": {"status": "ok"}, "reconcile_passes": 0}),
            "",
        )

    monkeypatch.setattr(session_mod.pipeline_dispatch, "run_legacy_design", fake_run)
    out = session_mod._run_legacy_session(tmp_path, "a brief", ["wiring"])
    assert out["status"] == "failed"
    assert out["failure_kind"] == "legacy_protocol_error"


def test_native_events_stream_before_exit(monkeypatch, tmp_path):
    release = tmp_path / "release"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "request = json.load(sys.stdin)\n"
        "print('diagnostic', flush=True)\n"
        "sys.stderr.write('x' * 200000); sys.stderr.flush()\n"
        "print('__KICRAFT_LEGACY_EVENT__={\"kind\":\"stage_start\",\"stage\":\"spec\"}', flush=True)\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
        "print('__KICRAFT_LEGACY_EVENT__={\"kind\":\"stage_done\",\"stage\":\"spec\"}', flush=True)\n"
        "print('__KICRAFT_LEGACY_SESSION_RESULT__={\"result\":{\"status\":\"ok\"}}')\n"
    )
    monkeypatch.setattr(pipeline, "legacy_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "legacy_python", lambda root: sys.executable)
    monkeypatch.setattr(pipeline, "_legacy_session_runner", lambda: runner)
    events = []

    def progress(event):
        events.append(event["kind"])
        if event["kind"] == "stage_start":
            assert not release.exists()
            release.touch()

    rc, stdout, stderr = pipeline.run_legacy_design(
        tmp_path, "brief", ["spec"], budget_usd=.1, progress=progress, timeout_s=5,
    )
    assert events == ["stage_start", "stage_done"]
    assert rc == 0
    assert stdout.startswith("diagnostic\n")
    assert "__KICRAFT_LEGACY_EVENT__" not in stdout
    assert pipeline.legacy_session_result(stdout)["result"]["status"] == "ok"
    assert stderr == "x" * 200000


@pytest.mark.parametrize("mode", ["malformed", "silent", "callback"])
def test_native_bridge_reaps_failed_children(monkeypatch, tmp_path, mode):
    pid_file = tmp_path / "pid"
    runner = tmp_path / "runner.py"
    event = '[]' if mode == "malformed" else '{"kind":"stage_start"}'
    runner.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "json.load(sys.stdin)\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        + (f"print('__KICRAFT_LEGACY_EVENT__=' + {event!r}, flush=True)\n" if mode != "silent" else "")
        + "time.sleep(30)\n"
    )
    monkeypatch.setattr(pipeline, "legacy_root", lambda: tmp_path)
    monkeypatch.setattr(pipeline, "legacy_python", lambda root: sys.executable)
    monkeypatch.setattr(pipeline, "_legacy_session_runner", lambda: runner)

    def progress(event):
        raise RuntimeError("callback failed")

    expected = {"malformed": ValueError, "silent": subprocess.TimeoutExpired, "callback": RuntimeError}[mode]
    with pytest.raises(expected):
        pipeline.run_legacy_design(
            tmp_path, "brief", ["spec"], budget_usd=.1, timeout_s=.3, progress=progress,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
