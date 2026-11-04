import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import backtest_review_taxonomy, mint_loadtest_token
from tools import ladder_experiment


def _harvest_module():
    path = Path(__file__).parent / "skill-eval" / "bin" / "harvest_run.py"
    spec = importlib.util.spec_from_file_location("harvest_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mint_requires_configured_secret(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KICRAFT_STORAGE_SECRET", raising=False)
    with pytest.raises(SystemExit, match="KICRAFT_STORAGE_SECRET"):
        mint_loadtest_token.mint(tmp_path)


def test_mint_lists_only_servable_encoded_paths(monkeypatch, tmp_path, capsys):
    from kicraft.server.render_serving import _resolve_project_token

    monkeypatch.setenv("KICRAFT_STORAGE_SECRET", "test-secret")
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    (project / "board #1.kicad_pcb").write_text("board")
    (project / "board.net").write_text("net")
    outside = tmp_path / "outside.kicad_sch"
    outside.write_text("outside")
    (project / "external.kicad_sch").symlink_to(outside)
    assert mint_loadtest_token.main([str(project)]) == 0
    lines = capsys.readouterr().out.splitlines()
    token = lines[0].split(": ", 1)[1]
    assert _resolve_project_token(token) == project.resolve()
    assert len([line for line in lines if "/project/" in line]) == 1
    assert lines[-1].endswith("/board%20%231.kicad_pcb")


def test_backtest_missing_required_labels_fails(monkeypatch, tmp_path, capsys):
    (tmp_path / "labels.json").write_text(json.dumps({"designs": [], "synthetics": []}))
    (tmp_path / "results.jsonl").write_text("")
    monkeypatch.setattr(backtest_review_taxonomy.sys, "argv", ["backtest", str(tmp_path)])
    assert backtest_review_taxonomy.main() == 1
    output = capsys.readouterr().out
    assert "missing confirmed-blocker label" in output
    assert "PASS:" not in output


def test_harvest_rejects_missing_transcript_before_creating_record(tmp_path):
    harvest = _harvest_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs = tmp_path / "runs"
    with pytest.raises(SystemExit, match="transcript not found"):
        harvest.main(["--workspace", str(workspace), "--runs-root", str(runs),
                      "--run-id", "run", "--transcript", str(tmp_path / "missing")])
    assert not (runs / "run").exists()


def test_ladder_identity_diagnostics_track_filtered_retry():
    events = [
        {"kind": "retry"},
        {"kind": "retry", "declared_identities": ["block:a", "block:b"]},
        {"kind": "retry", "declared_identities": [], "schema_error": "a"},
    ]
    assert ladder_experiment._run_metrics({}, events, [])['dropped_content'] == [
        {"rung": 2, "items": ["b"]}
    ]


def test_ladder_unique_artifacts_and_aborted_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("KICRAFT_CONTRACT_LADDER", "stock")
    monkeypatch.setattr(ladder_experiment, "Settings", SimpleNamespace(
        from_env=lambda: SimpleNamespace(redacted=lambda: {})))
    monkeypatch.setattr(ladder_experiment, "_git", lambda *args: "")
    monkeypatch.setattr(ladder_experiment.time, "strftime", lambda *args: "same")
    import kicraft.server.stage_pipeline as pipeline
    monkeypatch.setattr(pipeline, "make_budget_client", lambda budget: object())
    def fail(*args, progress, attempt_observer, **kwargs):
        progress({"kind": "retry"})
        attempt_observer({"provider_attempt": 1})
        raise RuntimeError("failure")
    monkeypatch.setattr(ladder_experiment, "drive_replay", fail)
    args = SimpleNamespace(arm="stock", label=None, state="state", board="board",
                           out=str(tmp_path), runs=2, budget=1, max_retries=1,
                           ceiling=10, unattended=False)
    assert ladder_experiment.run_arm(args) == 0
    rows = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len({row["artifacts"]["events"] for row in rows}) == 2
    assert len({row["artifacts"]["trace"] for row in rows}) == 2
    for row in rows:
        assert row["aborted"]


def test_ladder_replay_does_not_count_reader_errors_as_acceptance(monkeypatch, tmp_path, capsys):
    import kicraft.design.architecture_intent as architecture_intent
    stream = tmp_path / "arm-board-stamp.events.jsonl"
    stream.write_text('{}\n')
    state_dir = tmp_path / "states" / "board" / ".kicraft"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text('{}')
    monkeypatch.setattr(ladder_experiment, "_draft_payloads", lambda path: [{}])
    monkeypatch.setattr(ladder_experiment, "_project_to_intent", lambda draft: ({}, []))
    monkeypatch.setattr(architecture_intent, "derive_architecture", lambda intent:
                        SimpleNamespace(model_dump=lambda **kwargs: {}))
    errors = iter([([], "first error"), ([], "derived error")])
    monkeypatch.setattr(ladder_experiment, "_draft_codes", lambda *args: next(errors))
    assert ladder_experiment.replay_corpus(tmp_path, limit=None, state_root=tmp_path / "states") == 0
    output = capsys.readouterr().out
    assert "as written 0/1; after projection+derivation 0/1" in output
    row = json.loads((tmp_path / "replay.jsonl").read_text())
    assert row["unclassified"] == "first error"
    assert row["derived_unclassified"] == "derived error"
