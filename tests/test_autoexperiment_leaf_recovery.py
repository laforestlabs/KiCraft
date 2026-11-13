"""A leaf that routes boards but serializes no result must be DETECTED rather
than silently dropped from the parent compose (which strands its components
off-board).

A leaf solve can route its rounds (producing ``round_*_leaf_routed`` boards) but
die before writing ``metadata.json`` / ``solved_layout.json`` -- e.g. when every
round fails the acceptance gate. ``_all_leaf_artifacts`` enumerates leaves by
those JSON files, so such a "board-only" leaf is invisible to auto-pin and the
parent silently drops it. ``_board_only_leaf_dirs`` surfaces them so the run can
fail loudly instead of shipping a board missing a whole leaf.
"""

from __future__ import annotations

from pathlib import Path
import json

from kicraft.cli import autoexperiment

from kicraft.cli.autoexperiment import _board_only_leaf_dirs


def _leaf(root: Path, name: str, *, serialized: bool, routed_board: bool) -> Path:
    d = root / ".experiments" / "subcircuits" / name
    d.mkdir(parents=True)
    if serialized:
        (d / "metadata.json").write_text("{}")
        (d / "solved_layout.json").write_text("{}")
    if routed_board:
        (d / "round_0000_leaf_routed.kicad_pcb").write_text("(kicad_pcb)")
    return d


def test_board_only_leaf_is_detected(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    _leaf(proj, "ok__a", serialized=True, routed_board=True)  # normal leaf
    _leaf(proj, "dropped__b", serialized=False, routed_board=True)  # board-only
    _leaf(proj, "empty__c", serialized=False, routed_board=False)  # nothing to pin

    assert [d.name for d in _board_only_leaf_dirs(proj)] == ["dropped__b"]


def test_no_board_only_leaves_when_all_serialized(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    _leaf(proj, "a__1", serialized=True, routed_board=True)
    _leaf(proj, "b__2", serialized=True, routed_board=True)
    assert _board_only_leaf_dirs(proj) == []


def test_missing_subcircuits_dir_is_empty(tmp_path: Path) -> None:
    assert _board_only_leaf_dirs(tmp_path / "nonexistent") == []


def _accepted_unit(project: Path, name: str, accepted: bool = True) -> dict:
    directory = project / ".experiments" / "subcircuits" / name
    directory.mkdir(parents=True)
    layout = {
        "instance_path": f"/{name}",
        "validation": {"leaf_acceptance_result": {"accepted": accepted}},
    }
    files = {
        "leaf_routed.kicad_pcb": "board",
        "metadata.json": json.dumps({"instance_path": f"/{name}"}),
        "solved_layout.json": json.dumps(layout),
    }
    for filename, contents in files.items():
        (directory / filename).write_text(contents)
        (directory / f"round_0003_{filename}").write_text(contents)
    return {"solved": {"instance_path": f"/{name}", "rounds": [{"round_index": 3}]}}


def test_convergence_reuses_only_current_complete_accepted_units(tmp_path: Path) -> None:
    good = _accepted_unit(tmp_path, "good")
    rejected = _accepted_unit(tmp_path, "rejected", False)
    old = _accepted_unit(tmp_path, "old")
    pins = tmp_path / ".experiments" / "pins.json"
    pins.write_text(json.dumps({"pinned_leaves": {
        name: {"round": 3} for name in ("good", "rejected", "old")
    }}))
    units: dict = {}
    # The old accepted snapshot was not produced by this subprocess.
    autoexperiment._remember_solved_units(tmp_path, {"results": [good, rejected]}, units)
    assert set(units) == {"/good"}
    assert autoexperiment._reusable_leaf_paths(tmp_path, units) == {"/good"}
    stale_units: dict = {}
    autoexperiment._remember_solved_units(
        tmp_path, {"results": [good]}, stale_units,
        prior_snapshots={tmp_path / ".experiments/subcircuits/good/round_0003_leaf_routed.kicad_pcb"},
    )
    assert stale_units == {}
    assert autoexperiment._convergence_selectors(
        {"/good", "/rejected", "/old"}, {"/good"}, []
    ) == (["/old", "/rejected"], False)
    assert autoexperiment._convergence_selectors({"/good"}, {"/good"}, []) == ([], True)
    assert autoexperiment._convergence_selectors(
        {"/good", "/old"}, {"/good"}, ["/good"]
    ) == (["/old"], False)
    # A changed canonical board or missing snapshot invalidates the unit.
    directory = tmp_path / ".experiments" / "subcircuits" / "good"
    (directory / "leaf_routed.kicad_pcb").write_text("changed")
    assert autoexperiment._reusable_leaf_paths(tmp_path, units) == set()
    (directory / "leaf_routed.kicad_pcb").write_text("board")
    (directory / "round_0003_metadata.json").unlink()
    assert autoexperiment._reusable_leaf_paths(tmp_path, units) == set()
    assert old["solved"]["instance_path"] == "/old"


def test_convergence_input_fingerprint_tracks_source_and_config(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    (root / "kicraft").mkdir(parents=True)
    source = root / "kicraft" / "solver.py"
    source.write_text("a")
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(autoexperiment, "PROJECT_DIR", root)
    first = autoexperiment._convergence_inputs(project, str(config))
    source.write_text("b")
    second = autoexperiment._convergence_inputs(project, str(config))
    assert first != second
    config.write_text('{"clearance": 1}')
    assert second != autoexperiment._convergence_inputs(project, str(config))


def test_leaves_only_retries_unresolved_and_keeps_qualified_snapshot(tmp_path, monkeypatch):
    """Exercise the production two-phase leaf CLI, not just cache helpers."""
    from types import SimpleNamespace
    from kicraft.autoplacer import fab_profile
    from kicraft.autoplacer.brain import leaf_acceptance

    pcb = tmp_path / "project.kicad_pcb"
    schematic = tmp_path / "project.kicad_sch"
    pcb.write_text("input board")
    schematic.write_text("input schematic")
    monkeypatch.setattr(autoexperiment, "_planned_leaf_paths", lambda *_: {"/a", "/b"})
    monkeypatch.setattr(autoexperiment, "_convergence_inputs", lambda *_: "same-source")
    monkeypatch.setattr(fab_profile, "stamp_fab_floors_for_project", lambda *_: None)
    monkeypatch.setattr(
        leaf_acceptance, "evaluate_leaf_acceptance",
        lambda validation, *args, **kwargs: SimpleNamespace(
            accepted=validation.get("accepted") is True
        ),
    )
    commands = []
    saved_a = []

    def solve(cmd, **kwargs):
        commands.append(cmd)
        selectors = [cmd[i + 1] for i, item in enumerate(cmd) if item == "--only"]
        names = [item[1:] for item in selectors] if selectors else ["a", "b"]
        if len(commands) == 2:
            saved_a.append((tmp_path / ".experiments/subcircuits/a/leaf_routed.kicad_pcb").read_text())
        results = []
        for name in names:
            directory = tmp_path / ".experiments" / "subcircuits" / name
            directory.mkdir(parents=True, exist_ok=True)
            index = len(commands) - 1
            accepted = name == "a" or index > 0
            validation = {
                "accepted": accepted,
                "leaf_acceptance_result": {"accepted": accepted},
            }
            layout = {
                "instance_path": f"/{name}", "sheet_name": name,
                "validation": validation, "config_hash": "{}",
            }
            metadata = {"instance_path": f"/{name}", "sheet_name": name}
            record = {
                "round_index": index, "score": 10.0, "routed": True,
                "routing": {"validation": validation},
            }
            previous = directory / "debug.json"
            history = json.loads(previous.read_text())["extra"]["all_rounds"] if previous.exists() else []
            previous.write_text(json.dumps({"extra": {"all_rounds": history + [record]}}))
            for filename, content in {
                "leaf_routed.kicad_pcb": f"board-{name}-{index}",
                "metadata.json": json.dumps(metadata),
                "solved_layout.json": json.dumps(layout),
            }.items():
                (directory / filename).write_text(content)
                (directory / f"round_{index:04d}_{filename}").write_text(content)
            results.append({"solved": {"instance_path": f"/{name}", "rounds": [record]}})
        return 0, json.dumps({"results": results}), ""

    monkeypatch.setattr(autoexperiment, "_run_command", solve)
    rc = autoexperiment.main([
        str(pcb), "--schematic", str(schematic), "--leaves-only",
        "--rounds", "3", "--leaf-rounds", "1", "--workers", "1", "--seed", "0",
    ])
    assert rc == 0
    assert len(commands) == 2
    assert [commands[1][i + 1] for i, item in enumerate(commands[1]) if item == "--only"] == ["/b"]
    assert saved_a == ["board-a-0"]
    leaf = tmp_path / ".experiments/subcircuits/a"
    assert (leaf / "leaf_routed.kicad_pcb").read_text() == "board-a-0"
    assert (leaf / "round_0000_leaf_routed.kicad_pcb").read_text() == "board-a-0"
    pins = json.loads((tmp_path / ".experiments/pins.json").read_text())["pinned_leaves"]
    assert pins["a"]["round"] == 0
    assert pins["b"]["round"] == 1
