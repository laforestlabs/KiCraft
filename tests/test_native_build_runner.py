"""Regression coverage for native-state manufacturing snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import kicraft.design.cli_app as cli_app
from kicraft.design.models import ArtifactPaths, BOM, BomPart, ConversationState, PcbError, Question
from kicraft.server import native_build_runner


def _canonical_state() -> dict:
    state = ConversationState(
        project_stem="NATIVE_WIDGET",
        bom=BOM(
            parts=[
                BomPart(
                    ref="R1",
                    value="10k",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0402_1005Metric",
                    sheet="MAIN",
                )
            ]
        ),
        history=[{"role": "user", "content": "do not rewrite this history"}],
        open_questions=[Question(text="Which connector?", stage="bom", answer="JST")],
    ).model_dump(mode="json")
    # These fields are accepted by current manufacturing but not native design.
    # The snapshot serializes their current defaults; canonical native JSON must
    # retain its exact older BomPart shape.
    for key in (
        "recipe_id",
        "recipe_instance",
        "recipe_role",
        "resolution_source",
        "resolution_id",
        "lowering_requirement_id",
        "lowering_role",
        "lowering_index",
        "assembly",
    ):
        state["bom"]["parts"][0].pop(key)
    state["budget"] = {"spent_usd": 0.37, "limit_usd": 2.0}
    return state


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _current_artifacts(*, failed: bool = False) -> ArtifactPaths:
    errors = []
    if failed:
        errors = [
            PcbError(
                stage="place_route",
                code="keepout",
                title="Antenna keepout violation",
                explanation="keepout violation",
                details=["antenna keepout violated"],
                next_action="Move copper away from the antenna keepout.",
            )
        ]
    return ArtifactPaths(
        project_dir=Path("/tmp/generated/NATIVE_WIDGET"),
        project_stem="NATIVE_WIDGET",
        root_sch=Path("/tmp/generated/NATIVE_WIDGET/NATIVE_WIDGET.kicad_sch"),
        leaf_schs=[],
        kicad_pro=Path("/tmp/generated/NATIVE_WIDGET/NATIVE_WIDGET.kicad_pro"),
        autoplacer_json=Path("/tmp/generated/NATIVE_WIDGET/NATIVE_WIDGET_autoplacer.json"),
        routed_pcb=Path("/tmp/generated/NATIVE_WIDGET/NATIVE_WIDGET.kicad_pcb"),
        status="failed" if failed else "ok",
        pcb_errors=errors,
    )


def _persist_current_manufacturing_state(snapshot_path: Path, artifacts: ArtifactPaths) -> None:
    state = ConversationState.model_validate(json.loads(snapshot_path.read_text(encoding="utf-8")))
    state.artifacts = artifacts
    snapshot_path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


def test_current_manufacturing_serialization_stays_in_snapshot(tmp_path, monkeypatch):
    """Only the shared artifacts payload may cross back from current models."""
    state_path = tmp_path / "state.json"
    original = _canonical_state()
    _write(state_path, original)

    def manufacture(argv):
        snapshot_path = Path(argv[1])
        # This is the actual current-model load/dump boundary: it fills current
        # BomPart defaults and drops the native-only budget field before persisting
        # build artifacts.
        _persist_current_manufacturing_state(snapshot_path, _current_artifacts())
        return 0

    monkeypatch.setattr(cli_app, "main", manufacture)

    assert (
        native_build_runner.main(["replay", str(state_path), "generated", "--quality", "good"]) == 0
    )

    published = json.loads(state_path.read_text(encoding="utf-8"))
    assert {key: value for key, value in published.items() if key != "artifacts"} == {
        key: value for key, value in original.items() if key != "artifacts"
    }
    assert published["artifacts"]["routed_pcb"].endswith("NATIVE_WIDGET.kicad_pcb")


def test_failed_manufacturing_publishes_artifact_diagnostics_and_rc(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    original = _canonical_state()
    _write(state_path, original)

    def failed_manufacture(argv):
        snapshot_path = Path(argv[1])
        _persist_current_manufacturing_state(snapshot_path, _current_artifacts(failed=True))
        return 6

    monkeypatch.setattr(cli_app, "main", failed_manufacture)

    assert native_build_runner.main(["replay", str(state_path), "generated", "--no-fab"]) == 6

    published = json.loads(state_path.read_text(encoding="utf-8"))
    assert published["artifacts"]["status"] == "failed"
    assert published["artifacts"]["pcb_errors"][0]["code"] == "keepout"
    assert published["artifacts"]["pcb_errors"][0]["details"] == ["antenna keepout violated"]
    assert published["history"] == original["history"]
    assert published["budget"] == original["budget"]


def test_concurrent_native_edit_is_never_overwritten(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    original = _canonical_state()
    _write(state_path, original)

    def manufacturing_while_user_edits(argv):
        snapshot_path = Path(argv[1])
        _persist_current_manufacturing_state(snapshot_path, _current_artifacts(failed=True))

        user_edit = _canonical_state()
        user_edit["history"].append({"role": "user", "content": "concurrent authored edit"})
        user_edit["open_questions"][0]["answer"] = "Molex"
        _write(state_path, user_edit)
        return 0

    monkeypatch.setattr(cli_app, "main", manufacturing_while_user_edits)

    assert native_build_runner.main(["manual-route", str(state_path), "generated"]) == 1
    assert (
        json.loads(state_path.read_text(encoding="utf-8"))["open_questions"][0]["answer"] == "Molex"
    )
