import json
from pathlib import Path

from kicraft.server.build_worker import JOB_KIND_COMMANDS, job_command
from kicraft.server.storage import _discover_generated_dir, _persisted_generated_dir


def test_managed_build_has_stable_isolated_attempt_output(tmp_path):
    old = tmp_path / "generated" / "BOARD"
    old.mkdir(parents=True)
    sentinel = old / "user.kicad_sch"
    sentinel.write_text("do not overwrite")
    command = JOB_KIND_COMMANDS["build"]
    first = job_command("build", command, tmp_path, project_id=1, attempt_generation=4)
    resumed = job_command("build", command, tmp_path, project_id=1, attempt_generation=4)
    later = job_command("build", command, tmp_path, project_id=1, attempt_generation=5)
    assert first == resumed
    assert first[5] == "generated/attempt-4"
    assert later[5] == "generated/attempt-5"
    assert sentinel.read_text() == "do not overwrite"


def test_current_artifact_controls_discovery_and_manual_route(tmp_path):
    old = tmp_path / "generated" / "BOARD"
    old.mkdir(parents=True)
    (old / "BOARD.kicad_sch").write_text("old")
    current = tmp_path / "generated" / "attempt-4" / "BOARD"
    current.mkdir(parents=True)
    (current / "BOARD.kicad_sch").write_text("current")
    metadata = tmp_path / ".kicraft"
    metadata.mkdir()
    state = {"project_stem": "BOARD", "artifacts": {"project_dir": str(current)}}
    (metadata / "state.json").write_text(json.dumps(state))
    assert _discover_generated_dir(tmp_path) == current
    assert _persisted_generated_dir(tmp_path, "BOARD") == current
    command = job_command("manual_route", JOB_KIND_COMMANDS["manual_route"], tmp_path,
                          project_id=1, attempt_generation=5)
    assert Path(command[5]) == current
    state["artifacts"]["project_dir"] = str(tmp_path / "generated" / "attempt-9" / "BOARD")
    (metadata / "state.json").write_text(json.dumps(state))
    assert _discover_generated_dir(tmp_path) is None
    assert (old / "BOARD.kicad_sch").read_text() == "old"
