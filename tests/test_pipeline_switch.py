"""Guards for the pipeline switch (design-yield-recovery plan option 3, §4.3.1 guard 3).

A swap between the two design pipelines must be visible everywhere it matters, and the
dispatch must be readable from the durable config without a restart.
"""
from __future__ import annotations

import json

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
    """A legacy design builds with the legacy interpreter even if the switch moved back.

    The marker is written when the design is dispatched, so a workspace's pipeline is a
    property of the workspace, not of the config at build time.
    """
    ws = tmp_path / "ws"
    assert pipeline.read_marker(ws) is None
    pipeline.write_marker(ws, pipeline.PIPELINE_LEGACY)
    assert pipeline.read_marker(ws) == "legacy"
    marker = json.loads((ws / ".kicraft" / "pipeline.json").read_text())
    assert marker["legacy_commit"] == pipeline.LEGACY_COMMIT


def test_build_command_substitutes_only_the_interpreter(tmp_path):
    base = ["/usr/bin/python3", "-m", "kicraft.design.cli_app", "build", ".kicraft/state.json"]
    assert pipeline.build_command(base, "current") == base
    swapped = pipeline.build_command(base, "legacy")
    assert swapped[0] == pipeline.legacy_build_python()
    assert swapped[1:] == base[1:]  # the argv contract is the same in both trees


def test_provenance_fields_name_the_pinned_commit():
    current = pipeline.provenance_fields("current")
    assert current == {
        "pipeline": "current",
        "pipeline_legacy_commit": None,
        "pipeline_label": "current pipeline",
    }
    legacy = pipeline.provenance_fields("legacy")
    assert legacy["pipeline"] == "legacy"
    assert legacy["pipeline_legacy_commit"] == pipeline.LEGACY_COMMIT


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


def test_legacy_dispatch_uses_the_legacy_tree_and_its_environment(monkeypatch, tmp_path):
    """The design subprocess runs the legacy driver with the measured provider overrides.

    The legacy tree's defaults cap the price below the current model and route to providers
    that do not serve it, so every call failed in under a second at zero cost until these were
    set per process (handoff §6.3). Without them the switch looks like it works and spends
    nothing.
    """
    seen: dict = {}

    class _Completed:
        returncode = 1
        stdout = "out"
        stderr = "err"

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        seen["cwd"] = kwargs["cwd"]
        return _Completed()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    rc, out, err = pipeline.run_legacy_design(
        tmp_path, "a brief", ("intent", "functional_spec"), budget_usd=0.25
    )
    assert (rc, out, err) == (1, "out", "err")
    assert seen["command"][1:4] == ["-m", "kicraft.server.stage_driver", "run"]
    assert "--no-build" in seen["command"]
    for key, value in pipeline.LEGACY_ENV.items():
        assert seen["env"][key] == value
    # The legacy package must win over this checkout's editable install.
    assert seen["env"]["PYTHONPATH"] == str(pipeline.legacy_root())
    assert seen["cwd"] == str(pipeline.legacy_root())


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
    from kicraft.cli.artifact_paths import provenance_path, read_provenance, write_promote_provenance

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


def test_current_tree_never_owns_a_legacy_workspace_tail(tmp_path):
    """B1: the current tree may not write a legacy workspace's state (measured 2026-09-21).

    A legacy design committed, then the current tree's post-wiring lifecycle re-serialized
    state.json through the current models (adding `assembly`, `recipe_id`, `resolution_*`,
    `lowering_*`), and the legacy build then rejected its own file with 171 schema errors. The
    decision the web worker makes before running its tail is therefore pipeline-aware.
    """
    ws = tmp_path / "ws"
    pipeline.write_marker(ws, pipeline.PIPELINE_CURRENT)
    assert pipeline.current_tree_owns_tail(ws) is True
    pipeline.write_marker(ws, pipeline.PIPELINE_LEGACY)
    assert pipeline.current_tree_owns_tail(ws) is False


def test_parked_legacy_run_surfaces_its_question(monkeypatch, tmp_path):
    """B2: a legacy park is a question for the user, not a failure.

    Projects 876 and 877 (2026-09-21) committed every design stage and then parked in wiring
    on a programming-header question; the dispatch reported `failed`, so the question never
    reached the user. The park is translated from the workspace's own open_questions.
    """
    import json

    from kicraft.server import session as session_mod

    ws = tmp_path / "ws"
    (ws / ".kicraft").mkdir(parents=True)

    answered: dict = {"value": None}

    def fake_run(ws_, brief, stages, *, budget_usd, **kw):
        # Like the real park: the stages before wiring committed; wiring itself did not.
        state = {
            "stage_status": {s: {"ok": True} for s in stages[:-1]},
            "open_questions": [
                {
                    "stage": "wiring",
                    "text": "Add a 1x05 SWD programming header, then re-run wiring.",
                    "blocking": True,
                    "options": [],
                    "answer": answered["value"],
                }
            ],
        }
        (ws_ / ".kicraft" / "state.json").write_text(json.dumps(state))
        return 1, "parked: awaiting input", ""

    monkeypatch.setattr(session_mod.pipeline_dispatch, "run_legacy_design", fake_run)
    out = session_mod._run_legacy_session(ws, "a brief", ["intent", "bom", "wiring"])
    assert out["status"] == "awaiting_input"
    assert out["last_stage"] == "wiring"
    assert "SWD programming header" in out["questions"][0]["text"]

    # The same workspace with the question answered keeps the ordinary failure mapping: the
    # driver still exited nonzero (nothing left to commit), and no question is pending.
    answered["value"] = "done"
    out = session_mod._run_legacy_session(ws, "a brief", ["intent", "bom", "wiring"])
    assert out["status"] == "failed"
    assert out["questions"] is None
