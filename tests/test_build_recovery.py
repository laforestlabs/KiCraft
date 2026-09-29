"""Phase D: shared build-to-design recovery policy (web / headless / eval parity).

These are behavioral guards for the transitions, budgets, consent-independent
immutable choices, sibling preservation, and oscillation termination the policy
must implement. They never assert on error wording or source text: each case
drives ``run_build_recovery`` with a scripted deterministic build and a scripted
design re-drive, then checks the durable decision (action, owner, budget,
invalidation, retained state).
"""

from __future__ import annotations

import json
from pathlib import Path

from kicraft.server.session import (
    BUILD_RECOVERY_MAX_ATTEMPTS,
    classify_build_failure,
    read_build_recovery,
    run_build_recovery,
)


def _put(ws: Path, name: str, payload: dict) -> None:
    path = ws / ".kicraft" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _state(ws: Path, **overrides) -> dict:
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    doc = {
        "project_stem": "T",
        "intent": {"goal": "a 5 V buck converter", "named_parts": [], "constraints": []},
        "functional_spec": {"blocks": [{"name": "PWR", "category": "power", "purpose": "convert"}]},
        "architecture": {
            "sheets": [{"name": "PWR", "stem": "PWR"}],
            "power_nets": ["GND"],
            "inter_sheet_nets": [],
            "requirements": [{"id": "power_buck", "sheet": "PWR"}],
            "recipe_selections": [{"recipe": "buck@1", "instance": "b1"}],
            "protected_identities": [],
        },
        "bom": {
            "parts": [
                {
                    "ref": "U1",
                    "value": "TPS54331",
                    "symbol": "Regulator:TPS54331",
                    "footprint": "Package_SO:SOIC-8",
                    "sheet": "PWR",
                    "resolution_source": "llm",
                },
                {
                    "ref": "C1",
                    "value": "100nF",
                    "symbol": "Device:C",
                    "footprint": "Capacitor_SMD:C_0603",
                    "sheet": "PWR",
                    "resolution_source": "llm",
                },
            ],
            "connections": [{"net_name": "VBUS", "sheet": "PWR"}],
            "no_connect_pins": [{"ref": "U1", "pin": "7"}],
        },
        "stage_status": {},
    }
    doc.update(overrides)
    (ws / ".kicraft" / "state.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _failing(ws: Path, gates: list[tuple[str, list[str]]]) -> None:
    _put(
        ws,
        "synthesis_check.json",
        {
            "status": "failed",
            "failed_checks": [name for name, _ in gates],
            "checks": [
                {"name": name, "ok": False, "message": f"{name} failed", "offenders": offenders}
                for name, offenders in gates
            ],
        },
    )


def _passing(ws: Path) -> None:
    _put(ws, "synthesis_check.json", {"status": "ok", "failed_checks": [], "checks": []})


class _Script:
    """A scripted build worker plus a scripted design re-drive."""

    def __init__(self, ws: Path, rcs: list[int]):
        self.ws = ws
        self.rcs = list(rcs)
        self.builds = 0
        self.redrives: list[tuple[list[str], str]] = []

    def build(self) -> int:
        self.builds += 1
        return self.rcs.pop(0) if self.rcs else 0

    def redrive(self, stages, instruction: str) -> dict:
        self.redrives.append((list(stages), instruction))
        return {"status": "ok"}

    def mutate_bom(self, ref: str, value: str) -> None:
        doc = json.loads((self.ws / ".kicraft" / "state.json").read_text(encoding="utf-8"))
        for part in doc["bom"]["parts"]:
            if part["ref"] == ref:
                part["value"] = value
        (self.ws / ".kicraft" / "state.json").write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )


# ---- classification / ownership -------------------------------------------


def test_wiring_gate_is_wiring_owned_and_repairs_only_wiring(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.11 net coverage", ["U1 pin 3 missing from any net"])])
    failure = classify_build_failure(tmp_path, rc=5)
    assert failure.kind == "design_defect"
    assert failure.action == "repair_wiring"
    assert failure.owner_stage == "wiring"


def test_bom_availability_gate_backtracks_to_a_reviewed_alternative(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.26 orderable parts", ["C1 not orderable at any supplier"])])
    failure = classify_build_failure(tmp_path, rc=5)
    assert failure.kind == "design_defect"
    assert failure.action == "try_reviewed_alternative"
    assert failure.owner_stage == "bom"


def test_architecture_gate_backtracks_upstream(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.24 declared sheet ownership", ["requirement power_buck unowned"])])
    failure = classify_build_failure(tmp_path, rc=5)
    assert failure.action == "backtrack_architecture"
    assert failure.owner_stage == "architecture"


def test_reviewed_recipe_defect_is_a_protected_compiler_defect(tmp_path):
    doc = _state(tmp_path)
    doc["bom"]["parts"][0]["resolution_source"] = "recipe"
    (tmp_path / ".kicraft" / "state.json").write_text(json.dumps(doc), encoding="utf-8")
    _failing(tmp_path, [("9.37 reviewed rectifier endpoints", ["U1 PH has no reviewed return path"])])
    failure = classify_build_failure(tmp_path, rc=5)
    assert failure.kind == "compiler_defect"
    assert failure.action == "none"


def test_user_named_part_has_no_constructible_implementation(tmp_path):
    doc = _state(tmp_path)
    doc["intent"]["named_parts"] = ["TPS54331"]
    (tmp_path / ".kicraft" / "state.json").write_text(json.dumps(doc), encoding="utf-8")
    _failing(tmp_path, [("9.26 orderable parts", ["U1 TPS54331 unavailable"])])
    failure = classify_build_failure(tmp_path, rc=5)
    assert failure.kind == "capability_gap"
    assert failure.action == "none"
    assert failure.user_required is True


def test_infrastructure_failure_is_not_a_circuit_change(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["tooling unavailable"])])
    doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
    doc["artifacts"] = {"pcb_errors": [{"stage": "verify", "code": "drc_timeout"}]}
    (tmp_path / ".kicraft" / "state.json").write_text(json.dumps(doc))
    # No synthesis failure at all: routing/tooling failed with no attributable
    # design evidence.
    _put(tmp_path, "synthesis_check.json", {"status": "ok", "failed_checks": [], "checks": []})
    failure = classify_build_failure(tmp_path, rc=6)
    assert failure.kind in ("infrastructure", "unattributable")
    assert failure.action == "none"


# ---- the loop -------------------------------------------------------------


def test_local_repair_preserves_siblings_and_recovers(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])

    def redrive(stages, instruction):
        script.redrives.append((list(stages), instruction))
        # a wiring-only repair must not touch the committed BOM parts
        assert stages == ["wiring"]
        doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
        assert [p["ref"] for p in doc["bom"]["parts"]] == ["U1", "C1"]
        _passing(tmp_path)
        return {"status": "ok"}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "recovered"
    assert result["rc"] == 0
    assert script.builds == 2


def test_repair_wiring_clears_only_wiring_data(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])
    seen = {}

    def redrive(stages, instruction):
        doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
        seen["connection_count"] = len(doc["bom"]["connections"])
        seen["no_connect"] = doc["bom"]["no_connect_pins"]
        seen["parts"] = [p["ref"] for p in doc["bom"]["parts"]]
        _passing(tmp_path)
        return {"status": "ok"}

    run_build_recovery(tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive)
    assert seen["connection_count"] == 0  # stale wiring invalidated
    assert seen["no_connect"] == []
    assert seen["parts"] == ["U1", "C1"]  # valid siblings retained


def test_upstream_backtrack_invalidates_dependents_only(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.24 declared sheet ownership", ["requirement power_buck unowned"])])
    script = _Script(tmp_path, rcs=[5, 0])
    seen = {}

    def redrive(stages, instruction):
        seen["stages"] = list(stages)
        doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
        seen["bom"] = doc.get("bom")
        seen["functional_spec"] = doc.get("functional_spec")
        seen["architecture"] = doc.get("architecture")
        _passing(tmp_path)
        return {"status": "ok"}

    run_build_recovery(tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive)
    assert seen["stages"] == ["architecture", "bom", "wiring"]
    assert seen["bom"] is None  # dependent work invalidated
    assert seen["architecture"] is None  # owning choice re-drafted
    assert seen["functional_spec"] is not None  # upstream input preserved


def test_repeated_failure_stops_without_a_second_request(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 5, 0])

    calls = {"n": 0}

    def redrive(stages, instruction):
        calls["n"] += 1
        script.redrives.append((list(stages), instruction))
        # The repair changes the wiring, but the SAME failure fingerprint returns.
        return {"status": "ok"}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "recovery_repeated"
    assert calls["n"] == 1  # never re-issues an identical request
    assert read_build_recovery(tmp_path).attempts == 1


def test_bom_owned_constraint_backtracks_the_bom_and_rebuilds(tmp_path):
    """An upstream (BOM) defect re-drives bom+wiring, not a downstream redraft."""
    _state(tmp_path)
    _failing(tmp_path, [("9.25 capacitor polarity consistency", ["C1 polarized on a non-polar net"])])
    script = _Script(tmp_path, rcs=[5, 0])
    seen = {}

    def redrive(stages, instruction):
        seen["stages"] = list(stages)
        doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
        seen["parts"] = [p["ref"] for p in doc["bom"]["parts"]]
        seen["connections"] = doc["bom"]["connections"]
        # the BOM correction changes the offending part choice
        for part in doc["bom"]["parts"]:
            if part["ref"] == "C1":
                part["value"] = "1uF"
        (tmp_path / ".kicraft" / "state.json").write_text(json.dumps(doc), encoding="utf-8")
        _passing(tmp_path)
        return {"status": "ok"}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "recovered"
    assert seen["stages"] == ["bom", "wiring"]
    assert seen["parts"] == ["U1", "C1"]  # the parts stay visible to the correction
    assert seen["connections"] == []  # derived wiring invalidated


def test_without_the_recovery_the_same_evidence_stays_a_failed_build(tmp_path):
    """Negative control: the same build evidence with no revision allowance
    fails, and the justified recovery is what turns it into an export."""
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", max_attempts=0
    )
    assert result["status"] == "exhausted"
    assert result["rc"] == 5  # no export without the recovery
    assert script.builds == 1

    # ... and with the allowance it recovers and rebuilds successfully.
    _state(tmp_path)
    _passing(tmp_path)
    script2 = _Script(tmp_path, rcs=[5, 0])

    def redrive(stages, instruction):
        _passing(tmp_path)
        return {"status": "ok"}

    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    result2 = run_build_recovery(
        tmp_path, "a 5 V buck", script2.build, run_id="r2", redrive=redrive
    )
    assert result2["status"] == "recovered"
    assert result2["rc"] == 0


def test_unchanged_choice_stops_before_rebuilding(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.26 orderable parts", ["C1 not orderable"])])
    script = _Script(tmp_path, rcs=[5, 0])  # would recover if it rebuilt

    def redrive(stages, instruction):
        return {"status": "ok"}  # commits nothing: the owning choice is unchanged

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "recovery_stalled"
    assert script.builds == 1  # no rebuild of an unchanged candidate


def test_oscillating_choice_terminates(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.26 orderable parts", ["C1 not orderable for net N0"])])
    script = _Script(tmp_path, rcs=[5] * 5)
    docs = [
        ("C1", "1uF"),  # A -> B
        ("C1", "100nF"),  # B -> A (oscillation)
    ]
    idx = {"n": 0}

    def redrive(stages, instruction):
        ref, value = docs[min(idx["n"], len(docs) - 1)]
        idx["n"] += 1
        script.mutate_bom(ref, value)
        # a genuinely different failure each round, so the repeated-failure
        # guard is not what terminates the loop
        _failing(tmp_path, [("9.26 orderable parts", [f"C1 not orderable for net N{idx['n']}"])])
        return {"status": "ok"}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "recovery_oscillation"
    assert idx["n"] == 2  # A -> B -> A refused before a third repair
    assert script.builds == 3


def test_budget_cannot_reset_through_re_entry(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    first = _Script(tmp_path, rcs=[5] * 6)
    seq = {"n": 0}

    def redrive(stages, instruction):
        seq["n"] += 1
        # a genuinely different defect each round, so fingerprints advance
        _failing(tmp_path, [("9.12 ERC", [f"Pin U1.{seq['n'] + 10} not connected"])])
        return {"status": "ok"}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", first.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "recovery_exhausted"
    assert result["attempts"] == BUILD_RECOVERY_MAX_ATTEMPTS

    # A re-entered run (new run_id, same durable state) must not get a fresh budget.
    second = _Script(tmp_path, rcs=[5, 0])
    result2 = run_build_recovery(
        tmp_path, "a 5 V buck", second.build, run_id="r2", redrive=redrive
    )
    assert result2["status"] == "exhausted"
    assert result2["failure_kind"] == "recovery_exhausted"
    assert result2["attempts"] == BUILD_RECOVERY_MAX_ATTEMPTS
    assert second.builds == 1  # it built once, then refused to revise
    assert read_build_recovery(tmp_path).max_attempts == BUILD_RECOVERY_MAX_ATTEMPTS


def test_immutable_capability_gap_never_re_drives(tmp_path):
    doc = _state(tmp_path)
    doc["intent"]["named_parts"] = ["TPS54331"]
    (tmp_path / ".kicraft" / "state.json").write_text(json.dumps(doc))
    _failing(tmp_path, [("9.26 orderable parts", ["U1 TPS54331 unavailable"])])
    script = _Script(tmp_path, rcs=[5, 0])

    def redrive(stages, instruction):  # must never be reached
        raise AssertionError("an immutable limitation must not trigger a repair")

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "capability_gap"
    assert result["attempts"] == 0
    assert script.builds == 1


def test_compiler_defect_never_triggers_model_repair(tmp_path):
    doc = _state(tmp_path)
    doc["bom"]["parts"][0]["resolution_source"] = "recipe"
    (tmp_path / ".kicraft" / "state.json").write_text(json.dumps(doc))
    _failing(tmp_path, [("9.37 reviewed rectifier endpoints", ["U1 PH missing return"])])
    script = _Script(tmp_path, rcs=[5, 0])

    def redrive(stages, instruction):
        raise AssertionError("a protected compiler defect is not model-repairable")

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "compiler_defect"
    assert script.builds == 1


def test_exhausted_budget_from_the_guard_stops_honestly(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])

    class _Guard:
        def status(self):
            return {
                "kill_switch": False,
                "daily_remaining_usd": 0.0,
                "daily_ceiling_usd": 0.60,
                "total_remaining_usd": 1.0,
                "total_ceiling_usd": 250.0,
            }

    class _Client:
        guard = _Guard()

    def redrive(stages, instruction):
        raise AssertionError("no revision may be paid for once the ceiling is reached")

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", client=_Client(), redrive=redrive
    )
    assert result["status"] == "exhausted"
    assert result["failure_kind"] == "budget_refused"
    assert script.builds == 1


def test_awaiting_input_surfaces_the_question_and_stops(tmp_path):
    _state(tmp_path)
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])
    question = {"text": "Which connector?", "stage": "wiring", "blocking": True}

    def redrive(stages, instruction):
        return {"status": "awaiting_input", "questions": [question]}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "awaiting_input"
    assert result["needs_input"] is True
    assert result["questions"] == [question]
    assert script.builds == 1


def test_clean_build_records_no_recovery_history(tmp_path):
    _state(tmp_path)
    _passing(tmp_path)
    script = _Script(tmp_path, rcs=[0])
    result = run_build_recovery(tmp_path, "a 5 V buck", script.build, run_id="r1")
    assert result["status"] == "ok"
    assert result["attempts"] == 0
    persisted = read_build_recovery(tmp_path)
    assert persisted.attempts == 0
    assert persisted.events == []
    assert persisted.ok is True


def test_failed_redrive_restores_the_last_accepted_state(tmp_path):
    _state(tmp_path, project_stem="KEEP")
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])

    def redrive(stages, instruction):
        # simulate a broken revision: the wiring slot is wiped and the pass
        # fails before committing a validated candidate
        (tmp_path / ".kicraft" / "state.json").write_text(
            json.dumps({"project_stem": "KEEP", "bom": None}), encoding="utf-8"
        )
        return {"status": "failed", "failure_kind": "contract_rejected"}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "blocked"
    assert script.builds == 1  # nothing rebuilt against an unvalidated revision
    doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
    assert [p["ref"] for p in doc["bom"]["parts"]] == ["U1", "C1"]  # accepted state restored
    assert doc["bom"]["connections"]  # the previously accepted wiring survives


def test_parked_redrive_restores_the_last_accepted_state(tmp_path):
    _state(tmp_path, project_stem="KEEP")
    _failing(tmp_path, [("9.12 ERC", ["Pin U1.3 not connected"])])
    script = _Script(tmp_path, rcs=[5, 0])
    question = {"text": "Which regulator?", "stage": "wiring", "blocking": True}

    def redrive(stages, instruction):
        (tmp_path / ".kicraft" / "state.json").write_text(
            json.dumps({"project_stem": "KEEP", "bom": None}), encoding="utf-8"
        )
        return {"status": "awaiting_input", "questions": [question]}

    result = run_build_recovery(
        tmp_path, "a 5 V buck", script.build, run_id="r1", redrive=redrive
    )
    assert result["status"] == "awaiting_input"
    assert result["questions"] == [question]
    doc = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
    assert [p["ref"] for p in doc["bom"]["parts"]] == ["U1", "C1"]  # accepted state restored


def test_recovery_history_is_bounded(tmp_path):
    """The persisted history must not grow without limit."""
    from kicraft.server.session import write_build_recovery

    _state(tmp_path)
    for i in range(40):
        write_build_recovery(
            tmp_path,
            ok=False,
            attempts=i,
            events=[
                {
                    "action": "repair_wiring",
                    "reason": "r",
                    "outcome": "applied",
                    "failure_fingerprint": f"{i:064x}",
                }
            ],
        )
    persisted = read_build_recovery(tmp_path)
    assert len(persisted.events) <= 12


def test_headless_pipeline_uses_the_shared_recovery(tmp_path, monkeypatch):
    """Headless generation must reach the same policy the web worker uses.

    A failing deterministic build is recovered by re-driving the owning stage
    through the pipeline's own stage driver, then rebuilding -- no provider is
    involved (the LLM mode is mock and the stage driver is scripted).
    """
    from types import SimpleNamespace

    from kicraft.server import stage_pipeline as sp

    monkeypatch.setenv("KICRAFT_LLM_MODE", "mock")
    _state(tmp_path, project_stem="B")
    state_path = tmp_path / ".kicraft" / "state.json"
    driver_calls: list[dict] = []

    def fake_drive_chain(stages, brief, workspace, **kw):
        driver_calls.append({"stages": list(stages), **kw})
        if not state_path.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"project_stem": "B", "stage_status": {}}))
        return (
            [{"stage": stage, "commit_ok": True} for stage in stages],
            {"status": "ok"},
            str(state_path),
        )

    builds = {"n": 0}

    def fake_cli(cmd, cwd=None):
        builds["n"] += 1
        if builds["n"] == 1:
            _failing(Path(cwd), [("9.12 ERC", ["Pin R1.1 not connected"])])
            return SimpleNamespace(returncode=5)
        _passing(Path(cwd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sp, "drive_chain", fake_drive_chain)
    monkeypatch.setattr(sp, "run_design_cli", fake_cli)

    result = sp.run_pipeline("a USB LED", tmp_path, stages=("intent",), build=True)

    assert result["build_rc"] == 0
    assert result["build_recovery"]["status"] == "recovered"
    assert builds["n"] == 2
    assert driver_calls[-1]["stages"] == ["wiring"]
    assert driver_calls[-1]["instruction"].startswith("The deterministic build failed")
