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

from kicraft.design.cli_app import main
from kicraft.design.synthesis.parts_lookup import DEFAULT_KICAD_FOOTPRINT_DIR
from kicraft.design.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR


# stage-commit / stage-prep take `state` as a `nargs="?"` positional. On
# Python 3.12 argparse will not bind such a positional when it is placed AFTER
# an option (`... --slot-file X --no-archive state.json` -> "unrecognized
# arguments: state.json"). Production never hits this because it passes
# positionals first (see stage_driver._commit). Mirror that here so the tests
# exercise the real invocation order regardless of how a call site lists args.
_VALUE_OPTS = frozenset({"--slot-file", "--questions-file", "--history-message",
                         "--project-stem", "--archive-root"})


def _positionals_first(argv: list[str]) -> list[str]:
    if len(argv) < 2 or argv[0] not in ("stage-commit", "stage-prep"):
        return argv
    head, rest = list(argv[:2]), list(argv[2:])  # subcommand + stage stay put
    positionals, options, i = [], [], 0
    while i < len(rest):
        tok = rest[i]
        if isinstance(tok, str) and tok.startswith("-"):
            options.append(tok)
            if tok in _VALUE_OPTS and i + 1 < len(rest):
                options.append(rest[i + 1])
                i += 2
                continue
        else:
            positionals.append(tok)
        i += 1
    return head + positionals + options


def _run(capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, dict]:
    rc = main(_positionals_first(list(argv)))
    out = capsys.readouterr().out
    try:
        return rc, json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return rc, {"_raw_stdout": out}


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
                "value": "ESP32-S3-WROOM-1",
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
        ],
        "ic_groups": {},
        "group_labels": {},
        "thermal_refs": [],
        "signal_flow_order": [],
        "component_zones": {},
        "assumptions": [],
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
    # two distinct symbols in the BOM -> two batched lookups, no per-part calls
    assert set(pinouts.keys()) == {"Device:R", "Device:C"}
    for sym, info in pinouts.items():
        assert "pins" in info, f"{sym}: expected pin list, got {info!r}"


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


def test_stage_commit_wiring_preserves_bom_other_fields(tmp_path, capsys):
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
            ],
            "no_connect_pins": [],
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
    # Wiring fields populated...
    assert len(written["bom"]["connections"]) == 4
    # ...and the rest of the BOM is intact
    assert len(written["bom"]["parts"]) == 2
    assert {p["ref"] for p in written["bom"]["parts"]} == {"U1", "C1"}


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
            capsys, "stage-commit", stage, "--slot-file", str(slot),
            "--project-stem", "TEST", "--no-archive", str(state_path),
        )
        assert rc == 0, payload
    return state_path


@_footprints_installed
def test_stage_commit_bom_rejects_unresolvable_footprint(tmp_path, capsys):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bad = _valid_bom()
    # Plausible truncation of the real SW_SPST_PTS645Sx43SMTR92.kicad_mod —
    # the library exists but this exact footprint name does not.
    bad["parts"][0]["footprint"] = "Button_Switch_SMD:SW_SPST_PTS645"
    bom_slot = _write_slot(tmp_path, "bom", bad)
    rc, payload = _run(
        capsys, "stage-commit", "bom", "--slot-file", str(bom_slot),
        "--no-archive", str(state_path),
    )
    assert rc == 3
    assert payload["ok"] is False
    assert any("SW_SPST_PTS645" in off for off in payload["offenders"]), payload


@_footprints_installed
def test_stage_commit_bom_accepts_resolvable_footprints(tmp_path, capsys):
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom_slot = _write_slot(tmp_path, "bom", _valid_bom())
    rc, payload = _run(
        capsys, "stage-commit", "bom", "--slot-file", str(bom_slot),
        "--no-archive", str(state_path),
    )
    assert rc == 0, payload
    assert payload["ok"] is True


@_footprints_installed
def test_stage_commit_bom_rejects_unresolvable_symbol(tmp_path, capsys):
    # A hallucinated symbol name is now caught at BOM commit -- where the model
    # still has the lookup tools -- instead of cascading to wiring stage-prep.
    state_path = _commit_chain_through_arch(tmp_path, capsys)
    bom = _valid_bom()
    bom["parts"][0]["symbol"] = "NoSuchLib:DefinitelyMissing"
    bom_slot = _write_slot(tmp_path, "bom", bom)
    rc, payload = _run(
        capsys, "stage-commit", "bom", "--slot-file", str(bom_slot),
        "--no-archive", str(state_path),
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
        capsys, "stage-commit", "bom", "--slot-file", str(bom_slot),
        "--no-archive", str(state_path),
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
