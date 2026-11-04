"""Tests for the `stage-prep` and `stage-commit` CLI commands.

These two commands are the entire tool surface KiCraft uses to drive
its LLM-driven stages — keeping the per-stage tool-call count small
(and therefore the user's permission-prompt count small). The tests
exercise the happy path for each stage, the slot-shape validation
errors the LLM is expected to recover from, and the special wiring
merge-into-bom behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.design.cli_app import (
    _footprint_candidates,
    _symbol_candidates,
    _unresolved_symbols,
    main,
)
from kicraft.design.models import BOM, BomPart
from kicraft.design.synthesis.parts_lookup import DEFAULT_KICAD_FOOTPRINT_DIR
from kicraft.design.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR
from kicraft.design.synthesis.symbol_pinout import lookup_pins


def _run(capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, dict]:
    # Args are listed options-first with `state` LAST on purpose: the CLI's
    # _hoist_positionals must accept the state positional in any position
    # (Python 3.12 argparse otherwise refuses a nargs="?" positional after an
    # option). This exercises that fix directly.
    rc = main(list(argv))
    out = capsys.readouterr().out
    try:
        return rc, json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return rc, {"_raw_stdout": out}


def test_position_warnings_count_numbered_physical_pads_not_unique_pin_names(monkeypatch):
    from types import SimpleNamespace

    from kicraft.design import cli_app

    root = Path(__file__).resolve().parents[1]
    c3 = "esp32-c3-mini-1-h4x:WIFIM-SMD_ESP32-C3-MINI-1"
    switch = "kmr232ngulclfs:KMR232NGULCLFS"
    assert len(cli_app._footprint_numbered_pads(c3, root)) == 61
    assert len(cli_app._footprint_pad_numbers(c3, root)) == 53
    assert len(cli_app._footprint_numbered_pads(switch, root)) == 4
    assert cli_app._footprint_pad_numbers(switch, root) == {"1", "2"}

    monkeypatch.setattr(cli_app.jlcparts, "available", lambda: True)
    monkeypatch.setattr(cli_app, "_load_library_parts", lambda _root: ([], []))
    monkeypatch.setattr(
        cli_app.jlcparts, "lookup",
        lambda cid: {"joints": {"C41349510": 61, "C221681": 4, "C12345": 5}[cid]},
    )
    parts = [
        SimpleNamespace(ref="U1", footprint=c3, symbol="", sourcing_note="LCSC C41349510"),
        SimpleNamespace(ref="SW1", footprint=switch, symbol="", sourcing_note="LCSC C221681"),
        SimpleNamespace(ref="J1", footprint=switch, symbol="", sourcing_note="LCSC C12345"),
    ]
    warnings = cli_app.bom_position_mismatches(parts, root)
    assert [row["ref"] for row in warnings] == ["J1"]
    assert "has 4 pad(s)" in warnings[0]["message"]


def test_catalog_inductor_requires_reviewed_identity_and_footprint_pair():
    from types import SimpleNamespace

    from kicraft.design.cli_app import _lcsc_identity_conflict

    part = SimpleNamespace(symbol="Device:L", footprint="Inductor_SMD:L_6.3x6.3_H3")
    wrong = {
        "model": "CKCSA4020-4.7uH/M", "package": "SMD,4x4mm",
        "description": "4.7uH power inductor",
    }
    assert "no reviewed identity/footprint pair" in _lcsc_identity_conflict(part, wrong)
    part.footprint = "Inductor_SMD:L_12x12mm_H8mm"
    reviewed = {"model": "SMDRI127-220MT", "package": "SMD", "description": "22uH inductor"}
    assert _lcsc_identity_conflict(part, reviewed) is None
    part.footprint = "Inductor_SMD:L_6.3x6.3_H3"
    assert _lcsc_identity_conflict(part, reviewed) is not None


@pytest.mark.parametrize("started", [False, True])
def test_synthesis_failure_invalidates_freshness_only_after_emission(tmp_path, monkeypatch, started):
    from kicraft.design import cli_app
    from kicraft.design.models import ArtifactPaths, ConversationState

    state_path = tmp_path / "state.json"
    project = tmp_path / "project"
    receipt = project / ".kicraft" / "synthesis-generated.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"attempt_id": "old", "status": "complete"}))
    state = ConversationState(
        project_stem="TEST",
        artifacts=ArtifactPaths(
            project_dir=project, project_stem="TEST", root_sch=project / "TEST.kicad_sch",
            leaf_schs=[], kicad_pro=project / "TEST.kicad_pro",
            autoplacer_json=project / "TEST_autoplacer.json", fab_zip=project / "fab.zip",
        ),
    )

    def fail(*_args, **_kwargs):
        if started:
            receipt.write_text(json.dumps({"attempt_id": "new", "status": "failed"}))
        raise RuntimeError("emitter failed")

    monkeypatch.setattr(cli_app, "run_synth", fail)
    with pytest.raises(RuntimeError, match="emitter failed"):
        cli_app._run_synth_with_freshness(state, state_path, project)
    if started:
        assert state.artifacts.status == "failed"
        assert state.artifacts.fab_zip is None
        summary = json.loads((tmp_path / "synthesis_check.json").read_text())
        assert summary["status"] == "failed"
        assert summary["checks"][0]["report_path"] is None
    else:
        assert state.artifacts.status == "ok"
        assert state.artifacts.fab_zip == project / "fab.zip"
        assert not (tmp_path / "synthesis_check.json").exists()


def test_ambiguous_legacy_architecture_can_be_repaired_without_mutating_saved_state(tmp_path):
    from kicraft.design.cli_app import LegacyRecipeStateError, _load_state

    state_path = tmp_path / "state.json"
    original = json.dumps({
        "architecture": {"recipe_selections": [{"recipe": "rp2040-minimal@2"}]},
        "bom": {"parts": []},
    })
    state_path.write_text(original)
    with pytest.raises(LegacyRecipeStateError, match="rerun from architecture"):
        _load_state(state_path, repair_stage="bom")
    repaired_view = _load_state(state_path, repair_stage="architecture")
    assert repaired_view.architecture is None
    assert repaired_view.bom is None
    assert state_path.read_text() == original


@pytest.mark.parametrize("recipe", [
    "rp2040-minimal@1", "rp2040-minimal@2", "drv8833-dual-motor@1",
    "tps54331-adjustable@1", "tps54331-12v-1a@1",
])
@pytest.mark.parametrize("location", ["manifest", "part"])
def test_wiring_rejects_old_recipe_graph_without_rewriting_state(tmp_path, capsys, recipe, location):
    state = tmp_path / "state.json"
    bom = {"parts": [], "recipe_ownership": []}
    if location == "manifest":
        bom["recipe_ownership"].append({"recipe": recipe})
    else:
        bom["parts"].append({"recipe_id": recipe})
    original = json.dumps({"bom": bom})
    state.write_text(original)
    slot = tmp_path / "slot.json"
    slot.write_text("{}")
    rc, result = _run(
        capsys, "stage-commit", "wiring", "--slot-file", str(slot),
        "--no-archive", str(state),
    )
    assert rc == 3
    assert result["reconcile_target"] == "bom"
    assert "regenerate" in result["errors"][0]
    assert state.read_text() == original


def test_synthesis_summary_persists_current_report_basename(tmp_path):
    from types import SimpleNamespace

    from kicraft.design.cli_app import _write_synthesis_check

    result = SimpleNamespace(
        name="9.12 ERC", ok=True, message="ok", offenders=[],
        report_path="project_unique_erc.rpt",
    )
    _write_synthesis_check(tmp_path / "state.json", "PROJECT", [result], ok=True)
    summary = json.loads((tmp_path / "synthesis_check.json").read_text())
    assert summary["checks"][0]["report_path"] == "project_unique_erc.rpt"
    result.report_path = None
    _write_synthesis_check(tmp_path / "state.json", "PROJECT", [result], ok=False)
    summary = json.loads((tmp_path / "synthesis_check.json").read_text())
    assert summary["checks"][0]["report_path"] is None


@pytest.mark.parametrize("slot", [42, None, [], "text"])
def test_stage_commit_rejects_non_object_slot_without_changing_state(tmp_path, capsys, slot):
    state = tmp_path / "state.json"
    state.write_text("{}\n")
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(json.dumps(slot))
    rc, result = _run(
        capsys, "stage-commit", "bom", "--slot-file", str(slot_file),
        "--no-archive", str(state),
    )
    assert rc == 3
    assert result["ok"] is False
    assert "expected a JSON object" in result["errors"][0]
    assert state.read_text() == "{}\n"


def test_stage_commit_rejects_non_array_questions_without_changing_state(tmp_path, capsys):
    state = tmp_path / "state.json"
    state.write_text("{}\n")
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(json.dumps(_valid_intent()))
    questions = tmp_path / "questions.json"
    questions.write_text("42")
    rc, result = _run(
        capsys, "stage-commit", "intent", "--slot-file", str(slot_file),
        "--questions-file", str(questions), "--no-archive", str(state),
    )
    assert rc == 2
    assert result["ok"] is False
    assert "expected a JSON array" in result["errors"][0]
    assert state.read_text() == "{}\n"


@pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(), reason="KiCad symbol libraries not installed"
)
def test_db9_rejection_offers_loadable_dsub_contacts_not_telephone_connectors():
    candidates = _symbol_candidates("Connector:DB9_Female")
    assert candidates[0] == "Connector:DE9_Socket"
    assert all(candidate.startswith("Connector:DE9_") for candidate in candidates)
    for candidate in candidates:
        assert len(lookup_pins(candidate)["pins"]) >= 9
    bom = BOM(
        parts=[
            BomPart(
                ref="J1",
                value="CAN",
                symbol="Connector:DB9_Female",
                footprint="Connector_Dsub:DSUB-9_Socket_Vertical_P2.77x2.84mm",
                sheet="CAN",
            )
        ]
    )
    offenders = _unresolved_symbols(bom)
    assert len(offenders) == 1
    assert candidates[0] in offenders[0]
    assert "4P2C" not in offenders[0]


@pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(), reason="KiCad symbol libraries not installed"
)
@pytest.mark.parametrize(
    "identity, layout, pin_count",
    [
        ("Connector_Generic:Conn_02x08", "02x08", 16),
        ("Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical", "01x04", 4),
    ],
)
def test_symbol_candidates_preserve_header_layout(identity, layout, pin_count):
    candidates = _symbol_candidates(identity)
    assert layout in candidates[0]
    assert len(lookup_pins(candidates[0])["pins"]) == pin_count
    for candidate in candidates:
        assert layout in candidate
        assert lookup_pins(candidate)["pins"]


@pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(), reason="KiCad symbol libraries not installed"
)
def test_package_in_symbol_field_does_not_offer_unrelated_ics():
    assert _symbol_candidates("Package_SO:SOIC-28W_7.5x17.9mm_P1.27mm") == []


@pytest.mark.skipif(
    not DEFAULT_KICAD_FOOTPRINT_DIR.is_dir(), reason="KiCad footprint libraries not installed"
)
@pytest.mark.parametrize(
    "identity, prefix, dimension",
    [
        (
            "Connector_Dsub:DSUB-9_Female_Horizontal_P2.77x2.84mm",
            "Connector_Dsub:DSUB-9_Socket_Horizontal_",
            "P2.77x2.84mm",
        ),
        (
            "Button_Switch_SMD:SW_SPST_PTS645",
            "Button_Switch_SMD:SW_SPST_PTS645",
            "",
        ),
        (
            "Inductor_THT:L_Radial_D50.0mm_P5.00mm",
            "Inductor_THT:L_Radial_",
            "P5.00mm",
        ),
    ],
)
def test_footprint_candidates_rank_structure_before_alphabetic_limit(identity, prefix, dimension):
    candidates = _footprint_candidates(identity, limit=3)
    assert 1 <= len(candidates) <= 3
    for candidate in candidates:
        assert candidate.startswith(prefix)
        assert dimension in candidate
        library, _, name = candidate.partition(":")
        assert (DEFAULT_KICAD_FOOTPRINT_DIR / f"{library}.pretty" / f"{name}.kicad_mod").is_file()


def _write_slot(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def _valid_intent() -> dict:
    return {
        "goal": "Battery-powered ESP32 weather logger.",
        "constraints": ["18650 cell", "USB-C charging"],
        "named_parts": ["ESP32-S3-WROOM-1"],
        "inferred_expertise": "intermediate",
        "assumptions": ["JLCPCB fab (defaulted)"],
    }


def _valid_functional_spec() -> dict:
    return {
        "blocks": [
            {"name": "MCU", "category": "process", "purpose": "ESP32-S3 module."},
            {"name": "LDO", "category": "power", "purpose": "3V3 LDO."},
        ],
        "connections": [
            {
                "from_block": "LDO",
                "to_block": "MCU",
                "signal_type": "power",
                "description": "+3V3 rail",
            }
        ],
        "assumptions": [],
    }


def _valid_architecture() -> dict:
    return {
        "topologies": {"MCU": "ESP32-S3 module", "LDO": "AP2112K LDO"},
        "rail_voltages": {"+3V3": 3.3, "VBUS": 5.0},
        "comms_protocols": ["USB 2.0 FS"],
        "mcu_present": True,
        "sheets": [
            {
                "name": "MCU",
                "stem": "MCU",
                "function": "ESP32-S3 module with decoupling.",
                "from_library": None,
                "library_instance": None,
            },
            {
                "name": "LDO",
                "stem": "LDO",
                "function": "AP2112K 3V3 LDO with caps.",
                "from_library": None,
                "library_instance": None,
            },
        ],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "mcu-module",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
                "functional_blocks": ["MCU"],
            },
            {
                "id": "regulator",
                "sheet": "LDO",
                "role": "regulator",
                "family": "ldo",
                "ports": {"vin": "VBUS", "vout": "+3V3", "gnd": "GND"},
                "functional_blocks": ["LDO"],
            },
        ],
        "power_nets": ["VBUS", "+3V3", "GND"],
        "inter_sheet_nets": [
            {
                "name": "+3V3",
                "endpoints": [
                    {"sheet": "LDO", "direction": "bidirectional"},
                    {"sheet": "MCU", "direction": "bidirectional"},
                ],
            }
        ],
        "assumptions": [],
    }


def _valid_bom() -> dict:
    return {
        "parts": [
            {
                "ref": "U1",
                # Classic ESP32-WROOM-32 has no native USB, so its flashing
                # path is the CH340C bridge below -- the fixture stays a
                # 2-pin stand-in symbol on purpose (the reviewed native-USB
                # families require a real data socket with DN/DP pins).
                "value": "ESP32-WROOM-32",
                "symbol": "Device:R",  # cheap stand-in; pin lookup tests skip if KiCad missing
                "footprint": "Resistor_SMD:R_0402_1005Metric",
                "sheet": "MCU",
            },
            {
                "ref": "C1",
                "value": "1uF",
                "symbol": "Device:C",
                "footprint": "Capacitor_SMD:C_0402_1005Metric",
                "sheet": "LDO",
            },
            {
                # §9.29 programming access: an MCU BOM must carry a programming
                # path part, and the classic ESP32's is the bridge below; this
                # USB socket is the connector-class access part (2-pin
                # stand-in symbol keeps the wiring-coverage assertions exact).
                "ref": "J3",
                "value": "USB-C receptacle",
                "symbol": "Device:R",  # cheap stand-in, same as U1 above
                "footprint": "Resistor_SMD:R_0402_1005Metric",
                "sheet": "MCU",
            },
            {
                # §9.29 family strap rule: an ESP32 needs BOOT+RESET buttons
                # OR a USB-UART bridge (DTR/RTS auto-reset) -- USB alone is
                # not a workable download-mode story.
                "ref": "U2",
                "value": "CH340C USB-UART bridge",
                "symbol": "Device:R",  # cheap stand-in, same as U1 above
                "footprint": "Resistor_SMD:R_0402_1005Metric",
                "sheet": "MCU",
            },
            {
                # The 5 V -> 3.3 V rail needs real converter hardware: §9.39
                # refuses a substitution ledger entry standing in for it, so the
                # fixture ships the reviewed AP2112K regulator pair.
                "ref": "U3",
                "value": "AP2112K-3.3TRG1",
                "mpn": "AP2112K-3.3TRG1",
                "symbol": "ap2112k-3v3:AP2112K-3.3TRG1",
                "footprint": "ap2112k-3v3:SOT-25-5_L2.9-W1.6-P0.95-LS2.8-BL",
                "sheet": "LDO",
            },
        ],
        "ic_groups": {},
        "group_labels": {},
        "thermal_refs": [],
        "signal_flow_order": [],
        "component_zones": {},
        "assumptions": [],
        # §9.33: the architecture names AP2112K and the BOM now ships that exact
        # reviewed part, so nothing is substituted.
        "substitutions": [],
    }


# ---------- stage-prep ----------


def test_stage_prep_intent_on_missing_state(tmp_path, capsys):
    rc, payload = _run(
        capsys,
        "stage-prep",
        "intent",
        str(tmp_path / "nope" / "state.json"),
    )
    assert rc == 0
    assert payload["stage"] == "intent"
    assert payload["state"]["intent"] is None
    assert payload["extras"] == {}


def test_stage_prep_architecture_includes_leaves_block(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    rc, payload = _run(capsys, "stage-prep", "architecture", str(state_path))
    assert rc == 0
    assert "leaves_block" in payload["extras"]  # value may be None if library empty


def test_stage_prep_wiring_without_bom_errors(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    rc, _ = _run(capsys, "stage-prep", "wiring", str(state_path))
    assert rc == 4


@pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbols not installed",
)
def test_stage_prep_wiring_batches_pinouts(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    intent_slot = _write_slot(tmp_path, "intent", _valid_intent())
    _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(intent_slot),
        "--project-stem",
        "ESP32_TEST",
        "--no-archive",
        str(state_path),
    )
    fs_slot = _write_slot(tmp_path, "fs", _valid_functional_spec())
    _run(
        capsys,
        "stage-commit",
        "functional_spec",
        "--slot-file",
        str(fs_slot),
        "--no-archive",
        str(state_path),
    )
    arch_slot = _write_slot(tmp_path, "arch", _valid_architecture())
    _run(
        capsys,
        "stage-commit",
        "architecture",
        "--slot-file",
        str(arch_slot),
        "--no-archive",
        str(state_path),
    )
    bom_slot = _write_slot(tmp_path, "bom", _valid_bom())
    _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )

    rc, payload = _run(capsys, "stage-prep", "wiring", str(state_path))
    assert rc == 0
    pinouts = payload["extras"]["symbol_pinouts"]
    # Symbol lookups remain batched, but ownership expands to exact BOM refs.
    assert list(pinouts) == ["U1", "C1", "J3", "U2", "U3"]
    assert {info["symbol"] for info in pinouts.values()} == {
        "Device:R",
        "Device:C",
        "ap2112k-3v3:AP2112K-3.3TRG1",
    }
    for ref, info in pinouts.items():
        assert "pins" in info, f"{ref}: expected pin list, got {info!r}"
    assert payload["extras"]["locked_pin_assignments"] == []
    assert payload["extras"]["locked_no_connect_pins"] == []


@pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbols not installed",
)
def test_stage_prep_wiring_accepts_pinless_mechanical_symbol(tmp_path, capsys):
    """BOM commit accepts Mechanical:MountingHole (zero pins), but stage-prep
    wiring used to re-derive pinouts independently and die with rc=4 before
    the wiring model ever ran -- no retry, no park (self-eval run_20
    encoder-oled-panel, both 2026-07 batches). The zero-pin Mechanical
    exemption must hold on BOTH paths."""
    state_path = tmp_path / "state.json"
    bom = _valid_bom()
    bom["parts"].append(
        {
            "ref": "H1",
            "value": "M2.5 mounting hole",
            "symbol": "Mechanical:MountingHole",
            "footprint": "MountingHole:MountingHole_2.7mm_M2.5",
            "sheet": "MCU",
        }
    )
    for stage, data in [
        ("intent", _valid_intent()),
        ("functional_spec", _valid_functional_spec()),
        ("architecture", _valid_architecture()),
        ("bom", bom),
    ]:
        slot = _write_slot(tmp_path, stage, data)
        argv = [
            "stage-commit",
            stage,
            "--slot-file",
            str(slot),
            "--no-archive",
            str(state_path),
        ]
        if stage == "intent":
            argv += ["--project-stem", "ESP32_TEST"]
        rc, payload = _run(capsys, *argv)
        assert rc == 0, (stage, payload)

    rc, payload = _run(capsys, "stage-prep", "wiring", str(state_path))
    assert rc == 0, payload
    pinouts = payload["extras"]["symbol_pinouts"]
    # The pin-less reference is skipped, while every wireable BOM ref has its
    # own exact pin inventory even when several refs share one symbol lookup.
    assert "H1" not in pinouts
    assert list(pinouts) == ["U1", "C1", "J3", "U2", "U3"]


# ---------- stage-commit ----------


def test_stage_commit_intent_happy_path(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    slot = _write_slot(tmp_path, "intent", _valid_intent())
    rc, payload = _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(slot),
        "--project-stem",
        "ESP32_WEATHER_TEST",
        "--no-archive",
        str(state_path),
    )
    assert rc == 0, payload
    assert payload["ok"] is True
    assert payload["project_stem"] == "ESP32_WEATHER_TEST"
    assert "intent" in payload["slots_filled"]

    written = json.loads(state_path.read_text())
    assert written["intent"]["goal"].startswith("Battery-powered")
    assert written["project_stem"] == "ESP32_WEATHER_TEST"


def test_stage_commit_atomically_invalidates_stale_downstream_state(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "project_stem": "WEATHER",
                "intent": _valid_intent(),
                "functional_spec": _valid_functional_spec(),
                "architecture": _valid_architecture(),
                "bom": _valid_bom(),
                "open_questions": [
                    {
                        "text": "Current-stage note",
                        "stage": "functional_spec",
                        "blocking": False,
                        "material": True,
                    },
                    {
                        "text": "Stale architecture question",
                        "stage": "architecture",
                        "blocking": True,
                        "material": True,
                    },
                ],
                "stage_status": {
                    stage: {"ok": True}
                    for stage in (
                        "intent",
                        "functional_spec",
                        "architecture",
                        "bom",
                        "wiring",
                    )
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    slot = _write_slot(tmp_path, "functional_spec-redraft", _valid_functional_spec())

    rc, payload = _run(
        capsys,
        "stage-commit",
        "functional_spec",
        "--slot-file",
        str(slot),
        "--invalidate-downstream",
        "--no-archive",
        str(state_path),
    )

    assert rc == 0, payload
    assert payload["invalidated_stages"] == ["architecture", "bom", "wiring"]
    written = json.loads(state_path.read_text(encoding="utf-8"))
    assert written["intent"] is not None
    assert written["functional_spec"] is not None
    assert written["architecture"] is None
    assert written["bom"] is None
    assert written["open_questions"] == []
    assert set(written["stage_status"]) == {"intent", "functional_spec"}


def test_rejected_invalidating_commit_preserves_existing_state_bytes(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"project_stem": "KEEP", "intent": _valid_intent()}, indent=2) + "\n",
        encoding="utf-8",
    )
    before = state_path.read_bytes()
    slot = _write_slot(
        tmp_path,
        "bad-intent-redraft",
        _valid_intent() | {"inferred_expertise": "wizard"},
    )

    rc, payload = _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(slot),
        "--invalidate-downstream",
        "--no-archive",
        str(state_path),
    )

    assert rc == 3
    assert payload["ok"] is False
    assert state_path.read_bytes() == before


def test_stage_commit_rejects_malformed_intent(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    slot = _write_slot(
        tmp_path,
        "intent",
        {"goal": "x", "inferred_expertise": "wizard"},  # invalid literal
    )
    rc, payload = _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3
    assert payload["ok"] is False
    assert any("inferred_expertise" in e or "wizard" in e for e in payload["errors"])
    assert not state_path.exists()  # nothing written on validation failure


def test_stage_commit_appends_history(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    slot = _write_slot(tmp_path, "intent", _valid_intent())
    _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(slot),
        "--project-stem",
        "STEM",
        "--history-message",
        "Captured intent: weather logger.",
        "--no-archive",
        str(state_path),
    )
    written = json.loads(state_path.read_text())
    assert written["history"][-1]["role"] == "assistant"
    assert written["history"][-1]["content"] == "Captured intent: weather logger."


def test_stage_commit_attaches_questions(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    slot = _write_slot(tmp_path, "intent", _valid_intent())
    qs = _write_slot(
        tmp_path,
        "qs",
        [
            {
                "text": "How is this powered?",
                "stage": "intent",
                "blocking": True,
                "material": True,
            }
        ],
    )
    # _write_slot writes a dict; questions need a list, so write directly
    qs.write_text(
        json.dumps(
            [
                {
                    "text": "How is this powered?",
                    "stage": "intent",
                    "blocking": True,
                    "material": True,
                }
            ]
        )
    )
    rc, _ = _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(slot),
        "--questions-file",
        str(qs),
        "--project-stem",
        "STEM",
        "--no-archive",
        str(state_path),
    )
    assert rc == 0
    written = json.loads(state_path.read_text())
    assert len(written["open_questions"]) == 1
    assert written["open_questions"][0]["text"] == "How is this powered?"
    assert written["open_questions"][0]["stage"] == "intent"


def test_stage_commit_wiring_preserves_bom_other_fields(tmp_path, capsys, monkeypatch):
    state_path = tmp_path / "state.json"

    # Walk up through intent / functional_spec / architecture / bom
    for stage, data in [
        ("intent", _valid_intent()),
        ("functional_spec", _valid_functional_spec()),
        ("architecture", _valid_architecture()),
        ("bom", _valid_bom()),
    ]:
        slot = _write_slot(tmp_path, stage, data)
        argv = [
            "stage-commit",
            stage,
            "--slot-file",
            str(slot),
            "--no-archive",
            str(state_path),
        ]
        if stage == "intent":
            argv += ["--project-stem", "ESP32_TEST"]
        rc, payload = _run(capsys, *argv)
        assert rc == 0, (stage, payload)

    after_bom = json.loads(state_path.read_text())
    assert after_bom["wiring_complete"] is False
    after_bom["stage_status"]["wiring"] = {"ok": False}
    state_path.write_text(json.dumps(after_bom))

    # Now run wiring with every pin accounted for (the validator enforces
    # full net coverage; 2-pin parts need both pins assigned).
    wiring_slot = _write_slot(
        tmp_path,
        "wiring",
        {
            "connections": [
                {
                    "net_name": "+3V3",
                    "endpoints": [{"ref": "U1", "pin": "1"}],
                    "sheet": "MCU",
                },
                {
                    "net_name": "GND",
                    "endpoints": [{"ref": "U1", "pin": "2"}],
                    "sheet": "MCU",
                },
                {
                    "net_name": "+3V3",
                    "endpoints": [{"ref": "C1", "pin": "1"}],
                    "sheet": "LDO",
                },
                {
                    "net_name": "GND",
                    "endpoints": [{"ref": "C1", "pin": "2"}],
                    "sheet": "LDO",
                },
                {
                    "net_name": "VBUS",
                    "endpoints": [
                        {"ref": "U3", "pin": "1"},
                        {"ref": "U3", "pin": "3"},
                    ],
                    "sheet": "LDO",
                },
                {
                    "net_name": "GND",
                    "endpoints": [{"ref": "U3", "pin": "2"}],
                    "sheet": "LDO",
                },
                {
                    "net_name": "+3V3",
                    "endpoints": [{"ref": "U3", "pin": "5"}],
                    "sheet": "LDO",
                },
            ],
            # J3 (the §9.29 USB programming-access part) is a 2-pin stand-in;
            # net coverage (§9.11) requires every pin accounted for.
            "no_connect_pins": [
                {"ref": "J3", "pin": "1"},
                {"ref": "J3", "pin": "2"},
                {"ref": "U2", "pin": "1"},
                {"ref": "U2", "pin": "2"},
                {"ref": "U3", "pin": "4"},
            ],
        },
    )
    rc, payload = _run(
        capsys,
        "stage-commit",
        "wiring",
        "--slot-file",
        str(wiring_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 0, payload

    written = json.loads(state_path.read_text())
    assert written["wiring_complete"] is True
    assert "wiring" not in written["stage_status"]
    # Wiring fields populated...
    assert len(written["bom"]["connections"]) == 7
    # ...and the rest of the BOM is intact
    assert len(written["bom"]["parts"]) == 5
    assert {p["ref"] for p in written["bom"]["parts"]} == {"U1", "C1", "J3", "U2", "U3"}

    # A commit that finds an under-rated capacitor re-pins it and reports the swap, instead of
    # dying on its own bookkeeping. Loop 3 shipped `bom_normalizations.extend(repinned)` above the
    # list's binding, so the wiring commit returned invalid process output the moment a repair was
    # actually found -- live cohort 2026-09-30, project 1113: "UnboundLocalError: cannot access
    # local variable 'bom_normalizations'". Brief 1 passed the same loop because its re-pin found
    # nothing and never reached the line, which is exactly why the empty case cannot be the test.
    import kicraft.design.cli_app as cli_app
    from kicraft.parts_library import jlcparts

    for part in written["bom"]["parts"]:
        if part["ref"] == "C1":
            part["sourcing_note"] = "LCSC C19702"  # spans +3V3, rated far below it
            part["mpn"] = None
    # The stress is only provable across two typed rails; the fixture architecture declares +3V3
    # and VBUS but no return, so C1's terminal pair would be skipped as "not proved by a DC rail".
    written["architecture"]["rail_voltages"]["GND"] = 0.0
    state_path.write_text(json.dumps(written))
    import hashlib

    parts_digest = hashlib.sha256(
        json.dumps(written["bom"]["parts"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (state_path.parent / "sourcing_validation.json").write_text(json.dumps({
        "parts_digest": parts_digest,
        "parts": [{"ref": "C1", "exact_lcsc": "C19702", "picky_selection": True}],
    }))
    monkeypatch.setattr(
        jlcparts, "parameters",
        lambda cid: {"attributes": {"Voltage Rating": {"C19702": "1V", "C96446": "50V"}.get(cid, "")}},
    )
    monkeypatch.setattr(
        jlcparts, "search",
        lambda term, limit=10: [
            {
                "lcsc": "C96446", "model": "CL05A105KA5NQNC", "package": "0402",
                "description": "1uF ±10% 50V X5R 0402", "stock": 400000, "type": "Basic",
                "brand": "Samsung", "price": 0.01, "joints": 2,
            },
        ],
    )
    monkeypatch.setattr(cli_app.lcsc_retail, "enabled", lambda: False)

    rc, payload = _run(
        capsys,
        "stage-commit",
        "wiring",
        "--slot-file",
        str(wiring_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 0, payload
    assert any("re-pinned C19702 -> C96446" in note for note in payload["bom_normalizations"])

    # A BOM reapproval invalidates completion even without --invalidate-downstream.
    bom_data = _valid_bom()
    bom_data["connections"] = []
    bom_data["no_connect_pins"] = []
    bom_slot = _write_slot(tmp_path, "bom-again", bom_data)
    rc, payload = _run(
        capsys, "stage-commit", "bom", "--slot-file", str(bom_slot),
        "--no-archive", str(state_path),
    )
    assert rc == 0, payload
    assert json.loads(state_path.read_text())["wiring_complete"] is False


def test_wiring_completion_marker_roundtrips_and_legacy_is_nullable():
    from kicraft.design.models import ConversationState

    legacy = ConversationState.model_validate({})
    assert legacy.wiring_complete is None
    for marker in (None, False, True):
        restored = ConversationState.model_validate_json(
            ConversationState(wiring_complete=marker).model_dump_json()
        )
        assert restored.wiring_complete is marker


def test_stage_commit_bom_rejects_per_sheet_overflow(tmp_path, capsys, monkeypatch):
    import kicraft.design.cli_app as cli_app

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "project_stem": "BOUNDS_TEST",
                "intent": _valid_intent(),
                "functional_spec": _valid_functional_spec(),
                "architecture": _valid_architecture(),
            }
        )
    )
    parts = [
        {
            "ref": f"R{index}",
            "value": "10k",
            "symbol": "Device:R",
            "footprint": "Resistor_SMD:R_0402_1005Metric",
            "sheet": "MCU",
        }
        for index in range(1, 452)
    ]
    slot = _write_slot(tmp_path, "oversize_bom", {"parts": parts})

    def fail_if_called(*_):
        raise AssertionError("oversize BOM must be rejected before identity resolution")

    monkeypatch.setattr(cli_app, "_unresolved_footprints", fail_if_called)
    monkeypatch.setattr(cli_app, "_unresolved_symbols", fail_if_called)
    monkeypatch.setattr(cli_app, "_symbol_footprint_pin_mismatches", fail_if_called)
    monkeypatch.setattr(cli_app, "_unresolved_lcsc", fail_if_called)
    monkeypatch.setattr(cli_app, "_resolve_bom_mpn_sourcing", fail_if_called)

    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(slot),
        "--no-archive",
        str(state_path),
    )

    assert rc == 3
    assert any("9.35 BOM emission bounds" in error for error in payload["errors"])
    assert "MCU (451 items, > 450)" in payload["offenders"]


def test_stage_commit_bom_accepts_pinless_mechanical_symbol(tmp_path, capsys):
    """A RESOLVED zero-pin Mechanical symbol (MountingHole) is a deliberate
    pin-less part, not a hallucination -- rejecting it burned BOM retries on
    every board carrying a mounting hole (self-eval 2026-07-19 run_20)."""
    state_path = tmp_path / "state.json"
    bom = _valid_bom()
    bom["parts"].append(
        {
            "ref": "H1",
            "value": "M2.5 mounting hole",
            "symbol": "Mechanical:MountingHole",
            "footprint": "MountingHole:MountingHole_2.7mm_M2.5",
            "sheet": "MCU",
        }
    )
    for stage, data in [
        ("intent", _valid_intent()),
        ("functional_spec", _valid_functional_spec()),
        ("architecture", _valid_architecture()),
        ("bom", bom),
    ]:
        slot = _write_slot(tmp_path, stage, data)
        argv = [
            "stage-commit",
            stage,
            "--slot-file",
            str(slot),
            "--no-archive",
            str(state_path),
        ]
        if stage == "intent":
            argv += ["--project-stem", "ESP32_TEST"]
        rc, payload = _run(capsys, *argv)
        assert rc == 0, (stage, payload)
    written = json.loads(state_path.read_text())
    assert {p["ref"] for p in written["bom"]["parts"]} >= {"U1", "C1", "J3", "H1"}


def test_stage_commit_wiring_rejects_inert_duplicate_connector(tmp_path, capsys):
    """§9.31 must fire AT WIRING COMMIT (with the model's retry budget live),
    not only in the build-time collect_validations: self-eval 2026-07-19
    run_28 committed a 4-jack breakout with 3 jacks all-NC and died as an
    unrecoverable rc=5 at build 1/5 instead of a retry with offenders."""
    state_path = tmp_path / "state.json"
    bom = _valid_bom()
    for ref in ("J4", "J5"):
        bom["parts"].append(
            {
                "ref": ref,
                "value": "3.5mm TRS jack",
                "symbol": "Device:R",  # 2-pin stand-in, same as U1 above
                "footprint": "Resistor_SMD:R_0402_1005Metric",
                "sheet": "MCU",
            }
        )
    for stage, data in [
        ("intent", _valid_intent()),
        ("functional_spec", _valid_functional_spec()),
        ("architecture", _valid_architecture()),
        ("bom", bom),
    ]:
        slot = _write_slot(tmp_path, stage, data)
        argv = [
            "stage-commit",
            stage,
            "--slot-file",
            str(slot),
            "--no-archive",
            str(state_path),
        ]
        if stage == "intent":
            argv += ["--project-stem", "ESP32_TEST"]
        rc, payload = _run(capsys, *argv)
        assert rc == 0, (stage, payload)

    # J4 fully wired; its byte-identical sibling J5 is declared all-NC -- a
    # silently dead channel. Every other pin is accounted for, so §9.31 is
    # the only check that can fire.
    wiring_slot = _write_slot(
        tmp_path,
        "wiring",
        {
            "connections": [
                {"net_name": "+3V3", "endpoints": [{"ref": "U1", "pin": "1"}], "sheet": "MCU"},
                {"net_name": "GND", "endpoints": [{"ref": "U1", "pin": "2"}], "sheet": "MCU"},
                {"net_name": "+3V3", "endpoints": [{"ref": "C1", "pin": "1"}], "sheet": "LDO"},
                {"net_name": "GND", "endpoints": [{"ref": "C1", "pin": "2"}], "sheet": "LDO"},
                {"net_name": "+3V3", "endpoints": [{"ref": "J4", "pin": "1"}], "sheet": "MCU"},
                {"net_name": "GND", "endpoints": [{"ref": "J4", "pin": "2"}], "sheet": "MCU"},
            ],
            "no_connect_pins": [
                {"ref": "J3", "pin": "1"},
                {"ref": "J3", "pin": "2"},
                {"ref": "U2", "pin": "1"},
                {"ref": "U2", "pin": "2"},
                {"ref": "J5", "pin": "1"},
                {"ref": "J5", "pin": "2"},
            ],
        },
    )
    rc, payload = _run(
        capsys,
        "stage-commit",
        "wiring",
        "--slot-file",
        str(wiring_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3, payload
    assert payload["ok"] is False
    assert any("9.31" in e for e in payload["errors"]), payload["errors"]
    assert any("J5" in o for o in payload.get("offenders", [])), payload


def test_stage_commit_wiring_without_bom_errors(tmp_path, capsys):
    state_path = tmp_path / "state.json"
    slot = _write_slot(tmp_path, "wiring", {"connections": [], "no_connect_pins": []})
    rc, payload = _run(
        capsys,
        "stage-commit",
        "wiring",
        "--slot-file",
        str(slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3
    assert payload["ok"] is False
    assert any("bom" in e.lower() for e in payload["errors"])


def test_stage_commit_archive_writes_into_archive_root(tmp_path, capsys):
    state_path = tmp_path / ".kicraft" / "state.json"
    archive_root = tmp_path / "sessions"
    slot = _write_slot(tmp_path, "intent", _valid_intent())
    rc, payload = _run(
        capsys,
        "stage-commit",
        "intent",
        "--slot-file",
        str(slot),
        "--project-stem",
        "ARCH_TEST",
        "--archive-root",
        str(archive_root),
        str(state_path),
    )
    assert rc == 0, payload
    # An archive dir under the supplied root should exist
    subdirs = list(archive_root.iterdir())
    assert subdirs, "expected an archive subdirectory to be created"
    assert (subdirs[0] / "state.json").exists()
    assert (subdirs[0] / "manifest.json").exists()


# ---------- stage-commit bom: footprint resolution (Tier 2.1) ----------


_footprints_installed = pytest.mark.skipif(
    not DEFAULT_KICAD_FOOTPRINT_DIR.is_dir(),
    reason="KiCad footprint libraries not installed at the default path",
)


def _commit_chain_through_arch(tmp_path, capsys) -> Path:
    """Commit intent -> functional_spec -> architecture; return the state path."""
    state_path = tmp_path / "state.json"
    for slot_name, stage, data in (
        ("intent", "intent", _valid_intent()),
        ("fs", "functional_spec", _valid_functional_spec()),
        ("arch", "architecture", _valid_architecture()),
    ):
        slot = _write_slot(tmp_path, slot_name, data)
        rc, payload = _run(
            capsys,
            "stage-commit",
            stage,
            "--slot-file",
            str(slot),
            "--project-stem",
            "TEST",
            "--no-archive",
            str(state_path),
        )
        assert rc == 0, payload
    return state_path


def test_stage_commit_bom_rejects_unknown_architecture_sheet(tmp_path, capsys):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom = _valid_bom()
    bom["parts"].append(
        {
            "ref": "H1",
            "value": "M2.5 mounting hole",
            "symbol": "Mechanical:MountingHole",
            "footprint": "MountingHole:MountingHole_2.7mm_M2.5",
            "sheet": "MOUNTING HOLES",
        }
    )
    bom_slot = _write_slot(tmp_path, "bom", bom)

    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )

    assert rc == 3
    assert payload["ok"] is False
    assert any("9.13 BOM sheet references" in error for error in payload["errors"])
    assert "H1 -> 'MOUNTING HOLES'" in payload["offenders"]
    assert json.loads(state_path.read_text())["bom"] is None


def test_validate_rejects_persisted_unknown_architecture_sheet(tmp_path, capsys):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom = _valid_bom()
    bom["parts"].append(
        {
            "ref": "H1",
            "value": "M2.5 mounting hole",
            "symbol": "Mechanical:MountingHole",
            "footprint": "MountingHole:MountingHole_2.7mm_M2.5",
            "sheet": "MCU",
        }
    )
    bom_slot = _write_slot(tmp_path, "bom", bom)
    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 0, payload

    data = json.loads(state_path.read_text())
    next(p for p in data["bom"]["parts"] if p["ref"] == "H1")["sheet"] = "MOUNTING HOLES"
    state_path.write_text(json.dumps(data))

    rc = main(["validate", str(state_path)])
    captured = capsys.readouterr()

    assert rc == 3
    assert "9.13 BOM sheet references" in captured.err
    assert "H1 -> 'MOUNTING HOLES'" in captured.err


@_footprints_installed
def test_stage_commit_bom_rejects_unresolvable_footprint(tmp_path, capsys):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bad = _valid_bom()
    # Plausible truncation of the real SW_SPST_PTS645Sx43SMTR92.kicad_mod —
    # the library exists but this exact footprint name does not.
    bad["parts"][0]["footprint"] = "Button_Switch_SMD:SW_SPST_PTS645"
    bom_slot = _write_slot(tmp_path, "bom", bad)
    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3
    assert payload["ok"] is False
    assert any("SW_SPST_PTS645" in off for off in payload["offenders"]), payload


@_footprints_installed
def test_stage_commit_bom_accepts_resolvable_footprints(tmp_path, capsys):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom_slot = _write_slot(tmp_path, "bom", _valid_bom())
    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 0, payload
    assert payload["ok"] is True


@_footprints_installed
@pytest.mark.parametrize("owner", ["recipe", "forged", "unowned"])
def test_bom_partial_deterministic_wiring_defers_full_coverage(tmp_path, capsys, owner):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom = _valid_bom()
    partial_wiring = {
        "connections": [
            {
                "net_name": "LOCAL",
                "endpoints": [{"ref": "U1", "pin": "1"}],
                "sheet": "MCU",
            }
        ],
        "no_connect_pins": [],
    }
    bom.update(partial_wiring)
    bom["recipe_ownership"] = [
        {
            "recipe": "test-partial@1",
            "instance": "test_partial",
            "refs": ["U1"],
            "pins": [
                {
                    "ref": "U1",
                    "pin": "1",
                    "net": "LOCAL",
                    "owner": "recipe",
                    "owner_id": "test-partial@1",
                }
            ],
        }
    ]
    if owner == "unowned":
        bom["recipe_ownership"] = []
    if owner == "forged":
        bom["recipe_ownership"] = []
        next(part for part in bom["parts"] if part["ref"] == "U1").update({
            "resolution_source": "lowerer", "resolution_id": "test-lowerer@1",
            "lowering_requirement_id": "test", "lowering_role": "device",
        })
        bom["lowerer_ownership"] = [{
            "ref": "U1", "pin": "1", "net": "LOCAL",
            "owner": "lowerer", "owner_id": "test-lowerer@1",
            "requirement_id": "test",
        }]
    bom_slot = _write_slot(tmp_path, "bom", bom)

    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )

    if owner == "unowned":
        assert rc == 3
        assert payload["offenders"] == ["U1.1"]
        return
    if owner == "forged":
        assert rc == 3
        assert "BOM lowerer connection provenance is not canonical" in payload["errors"]
        return
    assert rc == 0, payload
    wiring_slot = _write_slot(tmp_path, "wiring", partial_wiring)
    rc, payload = _run(
        capsys,
        "stage-commit",
        "wiring",
        "--slot-file",
        str(wiring_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3
    assert any("9.11 net coverage" in error for error in payload["errors"])


@_footprints_installed
def test_bom_commit_proves_actual_lowerer_against_architecture(tmp_path, capsys):
    from kicraft.server.stage_work_units import (
        StageWorkUnit, aggregate_stage_work_units, deterministic_bom_candidate,
        validate_unit_candidate,
    )

    state_path = _commit_chain_through_arch(tmp_path, capsys)
    state = json.loads(state_path.read_text())
    requirement = {
        "id": "terminal", "sheet": "MCU", "role": "connector",
        "family": "screw-terminal", "exact_part": "WJ126V-5.0-02P-14-00A",
        "parameters": {"rows": 1}, "ports": {"positive": "+3V3", "negative": "GND"},
        "functional_blocks": ["TERMINAL"],
    }
    state["architecture"]["requirements"].append(requirement)
    state["functional_spec"]["blocks"].append({
        "name": "TERMINAL", "category": "interface",
        "purpose": "Expose the 3.3 V rail and ground on a screw terminal.",
    })
    state_path.write_text(json.dumps(state))
    unit = StageWorkUnit("bom-r001", "bom", "MCU", requirement_ids=("terminal",))
    candidate = deterministic_bom_candidate(unit, state)
    candidate = validate_unit_candidate(unit, candidate, state, {})
    lowered, _, _, _, _ = aggregate_stage_work_units(
        "bom", (unit,), {unit.unit_id: candidate}, state,
        unit_sources={unit.unit_id: "lowerer"},
    )
    bom = _valid_bom()
    bom["parts"].extend(lowered["parts"])
    bom["connections"] = lowered["connections"]
    bom["no_connect_pins"] = lowered["no_connect_pins"]
    bom["lowerer_ownership"] = lowered["lowerer_ownership"]

    for defect in ("unknown_requirement", "unknown_lowerer", "changed_net"):
        forged = json.loads(json.dumps(bom))
        lowerer_parts = [part for part in forged["parts"] if part.get("resolution_source") == "lowerer"]
        rows = forged["lowerer_ownership"]
        if defect == "unknown_requirement":
            for part in lowerer_parts:
                part["lowering_requirement_id"] = "unknown"
            for row in rows:
                row["requirement_id"] = "unknown"
        elif defect == "unknown_lowerer":
            for part in lowerer_parts:
                part["resolution_id"] = "unknown@1"
            for row in rows:
                row["owner_id"] = "unknown@1"
        else:
            connection = forged["connections"][0]
            endpoint = connection["endpoints"][0]
            connection["net_name"] = "FORGED_NET"
            for row in rows:
                if (row["ref"], row["pin"]) == (endpoint["ref"], endpoint["pin"]):
                    row["net"] = "FORGED_NET"
        slot = _write_slot(tmp_path, "forged_bom", forged)
        rc, payload = _run(
            capsys, "stage-commit", "bom", "--slot-file", str(slot),
            "--no-archive", str(state_path),
        )
        assert rc == 3, (defect, payload)

    slot = _write_slot(tmp_path, "actual_lowerer_bom", bom)
    rc, payload = _run(
        capsys, "stage-commit", "bom", "--slot-file", str(slot),
        "--no-archive", str(state_path),
    )
    assert rc == 0, payload


@_footprints_installed
def test_stage_commit_bom_rejects_unresolvable_symbol(tmp_path, capsys):
    # A hallucinated symbol name is now caught at BOM commit -- where the model
    # still has the lookup tools -- instead of cascading to wiring stage-prep.
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom = _valid_bom()
    bom["parts"][0]["symbol"] = "NoSuchLib:DefinitelyMissing"
    bom_slot = _write_slot(tmp_path, "bom", bom)
    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3
    assert payload["ok"] is False
    assert any("NoSuchLib" in off for off in payload["offenders"]), payload


@_footprints_installed
def test_stage_prep_wiring_fails_loudly_on_unresolved_symbol(tmp_path, capsys):
    # Defense in depth: even if a bad symbol reaches committed state by some other
    # path (it can no longer come through bom stage-commit), wiring stage-prep must
    # still fail loudly rather than emit a partial pinout dict the sub-agent would
    # work around by reading symbol files.
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom_slot = _write_slot(tmp_path, "bom", _valid_bom())
    rc, _ = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 0
    # Inject a bogus symbol directly into committed state, bypassing the
    # bom-commit symbol check that would otherwise reject it.
    data = json.loads(state_path.read_text())
    data["bom"]["parts"][0]["symbol"] = "NoSuchLib:DefinitelyMissing"
    state_path.write_text(json.dumps(data))
    rc, payload = _run(capsys, "stage-prep", "wiring", str(state_path))
    assert rc == 4
    assert payload["ok"] is False
    assert any("NoSuchLib" in off for off in payload["offenders"]), payload


@_footprints_installed
def test_stage_commit_bom_aggregates_identity_failures(tmp_path, capsys):
    # One rejection must name ALL failing gate classes at once (bad footprint
    # AND bad symbol on different parts), not reveal them one retry at a time --
    # self-eval 2026-07-07 runs 18/19 exhausted their 5 commit attempts
    # discovering one gate class per round.
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bad = _valid_bom()
    bad["parts"][0]["footprint"] = "Button_Switch_SMD:SW_SPST_PTS645"
    bad["parts"][1]["symbol"] = "NoSuchLib:DefinitelyMissing"
    bom_slot = _write_slot(tmp_path, "bom", bad)
    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 3
    errors = " ".join(payload["errors"])
    assert "footprint(s) do not resolve" in errors, payload
    assert "symbol(s) do not resolve" in errors, payload
    offenders = " ".join(payload["offenders"])
    assert "SW_SPST_PTS645" in offenders and "NoSuchLib" in offenders, payload


@_footprints_installed
def test_stage_commit_bom_normalizes_opposite_edge_connectors(tmp_path, capsys):
    # The run_21 proto-shield shape: connectors on ONE sheet honestly demanding
    # opposite board edges. The BOM model cannot edit architecture.sheets, so
    # rejecting via 9.24 was an unwinnable retry loop; the commit now runs the
    # same auto-split synthesis uses (KC-WFFXZ3 normalizer shape) and accepts.
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom = _valid_bom()
    for ref, edge in (("J1", "top"), ("J2", "bottom")):
        bom["parts"].append(
            {
                "ref": ref,
                "value": "header",
                "symbol": "Device:R",
                "footprint": "Resistor_SMD:R_0402_1005Metric",
                "sheet": "MCU",
            }
        )
        bom["component_zones"][ref] = {"edge": edge}
    bom_slot = _write_slot(tmp_path, "bom", bom)
    rc, payload = _run(
        capsys,
        "stage-commit",
        "bom",
        "--slot-file",
        str(bom_slot),
        "--no-archive",
        str(state_path),
    )
    assert rc == 0, payload
    assert payload.get("bom_normalizations"), payload
    state = json.loads(state_path.read_text())
    sheets = {p["sheet"] for p in state["bom"]["parts"] if p["ref"] in ("J1", "J2")}
    assert len(sheets) == 2, sheets  # split across two single-edge sheets
