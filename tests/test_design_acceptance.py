import json
import zipfile
from pathlib import Path

import pytest

from kicraft.design.part_identity import declares_package, matches_part_identity
from kicraft.eval.acceptance_contracts import contract_for, reference_obligation_ids
from kicraft.eval import artifact_evidence, electrical_artifact_evidence
from kicraft.eval.design_acceptance import (
    _stable_hash,
    build_reference_fixture,
    evaluate_obligation,
    verify_campaign,
    verify_reference_fixture,
)
from kicraft.server.stage_contracts import DESIGN_STAGES
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS


def _reference_result(facts, path="kicraft/design/part_identity.py"):
    return {
        "status": "pass",
        "facts": facts,
        "evidence": [{"kind": "manufacturer-source", "source_url": "https://example.invalid/ds.pdf", "section": "test", "path": path}],
    }


def _lowpass_reference_obligations():
    return {
        "rc-lowpass-bnc.bnc-count": _reference_result({"part_classes": {"bnc_connector": 2}}),
        "rc-lowpass-bnc.trim-pot": _reference_result({"part_classes": {"trim_potentiometer": 1}}),
        "rc-lowpass-bnc.adjustable-response": _reference_result({"cutoff_hz": 15915.0, "declared_ranges": {"cutoff_hz": [15000.0, 17000.0]}}),
        "rc-lowpass-bnc.no-mcu": _reference_result({"part_classes": {"microcontroller": 0}}),
        "rc-lowpass-bnc.sourceable-parts": _reference_result({
            "part_inventory": [{
                "mpn": "KH-BNC50-3511", "manufacturer": "Kinghelm", "source_url": "https://www.lcsc.com/product-detail/C2837587.html",
                "package": "right-angle through-hole BNC jack", "rated_limits": {"assembly_stock": 1},
            }],
        }),
    }


def test_reference_row_defers_board_obligations_exactly():
    row = build_reference_fixture("rc-lowpass-bnc", _lowpass_reference_obligations())
    _, deferred = reference_obligation_ids("rc-lowpass-bnc", "benchmark-original-v1")
    assert tuple(row["acceptance"]["deferred_obligations"]) == deferred
    assert row["acceptance"]["brief_hash"] == _stable_hash(contract_for("rc-lowpass-bnc")["original_brief"])
    assert verify_reference_fixture(row["acceptance"]) == []

    # A dropped deferred obligation hides a gate; an extra one invents a board check.
    row["acceptance"]["deferred_obligations"] = list(deferred[:-1])
    assert "defers exactly" in " ".join(verify_reference_fixture(row["acceptance"]))

    # A board-only gate recorded as a reference obligation is not proof.
    row["acceptance"]["deferred_obligations"] = list(deferred)
    row["acceptance"]["obligations"]["rc-lowpass-bnc.erc"] = _reference_result({}, path="reports/erc.txt")
    assert "reference fixture obligations differ" in " ".join(verify_reference_fixture(row["acceptance"]))


def test_reference_row_rejects_unreviewed_brief_and_wide_package():
    row = build_reference_fixture("stm32-min", {
        "stm32-min.stm32-package": _reference_result({"part_identities": [{"identity": "stm32f103c8t6", "package": "LQFP-48 7x7mm 0.50mm"}]}),
        "stm32-min.usb-crystal-swd": _reference_result({"net_paths": {"usb_c": True, "crystal_8mhz": True, "swd": True}}),
        "stm32-min.boot-reset": _reference_result({"net_paths": {"boot_button": True, "reset_button": True}}),
        "stm32-min.sourceable-parts": _reference_result({
            "part_inventory": [{
                "mpn": "STM32F103C8T6", "manufacturer": "STMicroelectronics", "source_url": "https://www.st.com/resource/en/datasheet/cd00161566.pdf",
                "package": "LQFP-48", "rated_limits": {"assembly_stock": 1},
            }],
        }),
    })
    assert verify_reference_fixture(row["acceptance"]) == []
    row["acceptance"]["obligations"]["stm32-min.stm32-package"] = _reference_result(
        {"part_identities": [{"identity": "stm32f103c8t6", "package": "LQFP-144 20x20mm"}]}
    )
    assert "does not prove required package" in " ".join(verify_reference_fixture(row["acceptance"]))

    row["acceptance"]["brief_hash"] = "sha256:" + "0" * 64
    assert "brief identity differs" in " ".join(verify_reference_fixture(row["acceptance"]))


def test_package_declaration_is_token_exact():
    assert declares_package("LQFP-48", "LQFP-48 7x7mm 0.50mm")
    assert declares_package("QFN-56", "stm32f103c8t6:QFN-56_L7.0-W7.0-P0.40-BL")
    assert not declares_package("QFN-56", "QFN-48_L7.0-W7.0-P0.50-BL")
    assert not declares_package("LQFP-48", "LQFP-48X")
    assert not declares_package("", "LQFP-48")
    assert matches_part_identity("STM32F103", "stm32f103c8t6")


def test_absent_class_is_counted_as_zero_but_active_parts_are_counted():
    assert evaluate_obligation(
        contract_for("rc-lowpass-bnc")["obligations"][3],
        {"status": "pass", "evidence": [{"kind": "tool-report", "path": "facts.json"}], "facts": {"part_classes": {"microcontroller": 0}}},
        artifact_root=None,
    ) is None
    assert "requires exactly 0 microcontroller parts" in evaluate_obligation(
        contract_for("rc-lowpass-bnc")["obligations"][3],
        {"status": "pass", "evidence": [{"kind": "tool-report", "path": "facts.json"}], "facts": {"part_classes": {}}},
        artifact_root=None,
    )
    assert artifact_evidence._classify_part({"mpn": "TPS54331DDAR", "symbol": "tps54331:TPS54331DDAR", "footprint": "tps54331:SOIC-8_L4.9-W3.9-P1.27-LS6.0-BL-EP"}) >= {"reviewed_5v_3a_buck", "active_device"}


def test_recorded_footprint_agrees_with_a_nickname_less_board():
    """Saved boards can drop the library nickname; that must not unclassify parts."""
    reviewed = "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P"
    assert artifact_evidence._reviewed_footprint_agrees(reviewed, "CONN-TH_WJ126V-5.0-2P")
    assert artifact_evidence._reviewed_footprint_agrees(reviewed, reviewed)
    assert not artifact_evidence._reviewed_footprint_agrees(reviewed, "other-lib:CONN-TH_WJ126V-5.0-2P")
    assert not artifact_evidence._reviewed_footprint_agrees(reviewed, "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-3P")
    assert not artifact_evidence._reviewed_footprint_agrees(reviewed, "")
    assert not artifact_evidence._reviewed_footprint_agrees("", "CONN-TH_WJ126V-5.0-2P")


def test_reconciliation_classifies_a_nickname_less_board_and_counts_zero_classes():
    """The delivered part class comes from reviewed identity, not the fact shape."""
    facts = {
        "bom_present": True,
        "board_loaded": True,
        "all_bom_parts": [],
        "board_footprints": {"J1": "CONN-TH_WJ126V-5.0-2P"},
        "board_pad_observations": [
            {"footprint_pad": "J1.1"}, {"footprint_pad": "J1.2"},
        ],
        "bom_parts": [{
            "ref": "J1", "mpn": "WJ126V-5.0-02P-14-00A",
            "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
            "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        }],
    }
    artifact_evidence._reconcile_inventory(facts)
    assert facts["part_classes"]["screw_terminal"] == 1
    assert facts["part_classes"]["active_device"] == 0
    assert facts["unclassified_bom_refs"] == []
    assert facts["part_identities"] == [{
        "identity": "wj126v-5.0-02p-14-00a",
        "package": "KANGNEX 1x02, 5.00 mm-pitch, through-hole screw terminal",
        "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
    }]


def test_unbounded_numeric_check_requires_the_value_inside_its_declared_range():
    obligation = contract_for("speaker-crossover")["obligations"][2]
    assert obligation["check"] == {"kind": "numeric_range", "fact": "crossover_hz"}
    inside = {"status": "pass", "evidence": [{"kind": "tool-report", "path": "facts.json"}], "facts": {"crossover_hz": 2500.0, "declared_ranges": {"crossover_hz": [2000.0, 3000.0]}}}
    assert evaluate_obligation(obligation, inside, artifact_root=None) is None

    # The frozen #4 defect shipped a 0.51 uH inductor, i.e. a ~2.5 MHz crossover.
    shunted = dict(inside, facts={"crossover_hz": 2500000.0, "declared_ranges": {"crossover_hz": [2000.0, 3000.0]}})
    assert "outside its declared range" in evaluate_obligation(obligation, shunted, artifact_root=None)

    undeclared = dict(inside, facts={"crossover_hz": 2500.0})
    assert "no declared range" in evaluate_obligation(obligation, undeclared, artifact_root=None)


def test_applicable_gate_is_derived_from_hardware_and_contract():
    """A gate this design does not owe cannot block, and an owed gate cannot pass unproven."""
    rc = contract_for("rc-lowpass-bnc")
    mcu = contract_for("stm32-min")
    assert artifact_evidence._applicable_gates(rc, {"part_classes": {"microcontroller": 0}}) == {
        "programming": False, "geometry": False,
    }
    assert artifact_evidence._applicable_gates(mcu, {"part_classes": {"microcontroller": 1}})["programming"] is True
    # An MCU on a brief with no geometry requirement owes programming only.
    assert artifact_evidence._applicable_gates(rc, {"part_classes": {"microcontroller": 1}}) == {
        "programming": True, "geometry": False,
    }
    # A shaped-outline brief owes geometry, with or without an MCU.
    assert artifact_evidence._applicable_gates(contract_for("round-led-ring"), {"part_classes": {}})["geometry"] is True

    obligation = next(
        item for item in contract_for("stm32-min")["obligations"]
        if item["id"] == "stm32-min.programming-when-applicable"
    )
    assert obligation["check"] == {"kind": "applicable_gate", "gate": "programming"}
    assert artifact_evidence._result_status(obligation["check"], {"applicable_gates": {"programming": False}}) == "pass"
    assert artifact_evidence._result_status(obligation["check"], {"applicable_gates": {"programming": True}}) == "unverified"
    assert artifact_evidence._result_status(
        obligation["check"], {"applicable_gates": {"programming": True}, "gates": {"programming": "fail"}}
    ) == "fail"
    assert artifact_evidence._result_status(
        obligation["check"], {"applicable_gates": {"programming": True}, "gates": {"programming": "pass"}}
    ) == "pass"
    assert artifact_evidence._result_status(obligation["check"], {}) == "unverified"


def test_set_members_fails_only_on_observed_membership():
    """A declared selector set is proven from the published fact, not assumed."""
    obligation = next(
        item for item in contract_for("usb-pd-trigger")["obligations"]
        if item["id"] == "usb-pd-trigger.selector-modes"
    )
    check = obligation["check"]
    assert artifact_evidence._result_status(check, {}) == "unverified"
    assert artifact_evidence._result_status(check, {"pd_selector_voltages": [9, 12, 20]}) == "pass"
    assert artifact_evidence._result_status(check, {"pd_selector_voltages": [9, 12]}) == "fail"


def test_fabrication_verdict_requires_a_structured_drc_and_erc_verdict(tmp_path):
    """A board and an archive are not a fabrication verdict."""
    run = tmp_path / "run"
    (run / ".kicraft").mkdir(parents=True)
    (run / "generated" / "P").mkdir(parents=True)
    (run / ".kicraft" / "state.json").write_text('{"bom": {"parts": []}}')
    zip_path = run / "generated" / "P" / "P_fab.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("P-job.gbrjob", "{}")
        archive.writestr("P-PTH.drl", "M48")
        archive.writestr("P-F_Cu.gtl", "G04*")
    evidence = artifact_evidence.generate_artifact_evidence(run, "rc-lowpass-bnc", build_rc=0, design_committed=True)
    assert evidence["fabrication"]["status"] == "unverified"
    assert evidence["fabrication"]["artifact_paths"] == ["generated/P/P_fab.zip"]

    # ERC alone is still not a manufacturing verdict.
    (run / ".kicraft" / "synthesis_check.json").write_text(json.dumps({
        "status": "ok", "checks": [{"name": "9.12 ERC", "ok": True}],
    }))
    evidence = artifact_evidence.generate_artifact_evidence(run, "rc-lowpass-bnc", build_rc=0, design_committed=True)
    assert evidence["common_gates"] == {"erc": "pass"}
    assert evidence["fabrication"]["status"] == "unverified"

    # A routed-board gate with shorts, or an unacceptable board, is a DRC fail.
    gate = run / ".kicraft" / "build_gate.json"
    gate.write_text(json.dumps({"fab_acceptable": True, "shorts": 1, "unconnected": 0, "courtyard": 0, "keepout": 0}))
    evidence = artifact_evidence.generate_artifact_evidence(run, "rc-lowpass-bnc", build_rc=0, design_committed=True)
    assert evidence["common_gates"]["drc"] == "fail"
    assert evidence["fabrication"]["status"] == "unverified"

    gate.write_text(json.dumps({"fab_acceptable": True, "shorts": 0, "unconnected": 0, "courtyard": 0, "keepout": 0}))
    evidence = artifact_evidence.generate_artifact_evidence(run, "rc-lowpass-bnc", build_rc=0, design_committed=True)
    assert evidence["common_gates"] == {"erc": "pass", "drc": "pass"}
    assert evidence["fabrication"]["status"] == "pass"
    # Fabrication alone is not fulfillment: an unproven obligation still blocks.
    assert evidence["obligations"]["rc-lowpass-bnc.bnc-count"]["status"] != "pass"
    assert evidence["software_fulfillment"]["status"] == "unverified"
    assert "rc-lowpass-bnc.bnc-count:" in evidence["software_fulfillment"]["reason"]

    # A non-zero build is never fabricated, however clean the reports look.
    evidence = artifact_evidence.generate_artifact_evidence(run, "rc-lowpass-bnc", build_rc=6, design_committed=True)
    assert evidence["fabrication"]["status"] == "unverified"
    assert evidence["software_fulfillment"]["status"] == "unverified"


# Every other brief must have a complete, validating reviewed reference row.
# A blocked brief is listed with the enforced reason it cannot be completed;
# removing an entry here is how an unblocked row becomes a hard requirement.
_EXPECTED_BLOCKED_REFERENCES = {
    # No eligible air-core inductor exists in the 633k-row catalog and the
    # vendored Dayton LW18-50 has no LCSC code; the user's strict sourcing
    # policy blocks the brief rather than allowing a substitute.
    "speaker-crossover": "sourceable-parts",
    # CH224K (C970725) is stocked for JLCPCB assembly but dry at the lcsc.com
    # retail storefront, so no committed board exists under the dual-inventory
    # sourcing gate; the reviewed design is real, the supply is not.
    "usb-pd-trigger": "sourceable-parts",
}


def test_reviewed_reference_corpus_validates_except_the_recorded_blockers():
    """The in-repo reference rows are the reviewed corpus, not scratch data."""
    directory = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "reference_inputs"
    rows = {
        row["acceptance"]["slug"]: row["acceptance"]
        for path in sorted(directory.glob("*.json"))
        for row in json.loads(path.read_text())["references"]
    }
    assert len(rows) == 34, "the reference corpus must cover every original brief"
    unexpected = {}
    for slug, payload in rows.items():
        errors = verify_reference_fixture(payload)
        if errors and slug not in _EXPECTED_BLOCKED_REFERENCES:
            unexpected[slug] = errors
    assert not unexpected, f"reviewed reference rows regressed: {sorted(unexpected)}"
    for slug, reason in _EXPECTED_BLOCKED_REFERENCES.items():
        errors = verify_reference_fixture(rows[slug])
        assert any(reason in error for error in errors), (
            f"{slug} is listed as blocked on {reason!r} but reports {errors}"
        )


def _campaign(tmp_path):
    entry = BENCHMARK_PROMPTS[0]
    slug = entry["slug"]
    rundir = tmp_path / "run"
    (rundir / ".kicraft").mkdir(parents=True)
    corpus = [{"index": 1, "slug": slug, "brief_hash": _stable_hash(entry["brief"])}]
    manifest = {
        "immutable": {
            "corpus": corpus,
            "corpus_hash": _stable_hash(corpus),
            "repeats": 1,
            "source_fingerprint": "sha256:fixture",
            "caps": {"project_usd": 0.15},
            "llm_mode": "live",
        }
    }
    summary = {
        "finished_at": "2026-09-11T12:00:00Z",
        "wall_s": 1,
        "design_only": True,
        "judge": False,
        "resumed": False,
        "source_unchanged": True,
        "n": 1,
        "design_committed": 1,
        "runs": [
            {
                "slug": slug,
                "index": 1,
                "prompt": entry["brief"],
                "rundir": str(rundir),
                "design_committed": True,
                "design_cost_usd": 0.05,
            }
        ],
        "fresh_output_directory": True,
    }
    (tmp_path / "campaign_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (rundir / ".kicraft/state.json").write_text(
        json.dumps(
            {
                "stage_status": {stage: {"ok": True} for stage in DESIGN_STAGES},
            }
        )
    )
    (rundir / "events.jsonl").write_text(
        "\n".join(
            json.dumps({"kind": "stage_done", "stage": stage, "ok": True})
            for stage in DESIGN_STAGES
        )
    )
    return slug, summary, rundir


def test_acceptance_rejects_missing_stage_evidence_despite_green_summary(tmp_path):
    slug, _, rundir = _campaign(tmp_path)
    assert verify_campaign(tmp_path, [slug]) == []
    events = rundir / "events.jsonl"
    events.write_text("\n".join(events.read_text().splitlines()[:-1]))
    assert any("events do not prove" in error for error in verify_campaign(tmp_path, [slug]))


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "substituted_brief", "resumed", "source_changed", "over_budget", "unfinished"],
)
def test_acceptance_rejects_false_green_campaign(tmp_path, defect):
    slug, summary, _ = _campaign(tmp_path)
    if defect == "duplicate":
        summary["runs"].append(summary["runs"][0].copy())
    elif defect == "substituted_brief":
        summary["runs"][0]["prompt"] = "A different design"
    elif defect == "resumed":
        summary["resumed"] = True
    elif defect == "source_changed":
        summary["source_unchanged"] = False
    elif defect == "over_budget":
        summary["runs"][0]["design_cost_usd"] = 0.151
    else:
        del summary["finished_at"]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert verify_campaign(tmp_path, [slug])


def test_acceptance_allows_recovered_stage_but_rejects_later_terminal_failure(tmp_path):
    slug, _, rundir = _campaign(tmp_path)
    events = rundir / "events.jsonl"
    rows = events.read_text().splitlines()
    failed = json.dumps({"kind": "stage_done", "stage": "functional_spec", "ok": False})
    rows.insert(1, failed)
    events.write_text("\n".join(rows))
    assert verify_campaign(tmp_path, [slug]) == []

    events.write_text("\n".join([*rows, failed]))
    assert any("events do not prove" in error for error in verify_campaign(tmp_path, [slug]))


def test_artifact_backed_obligation_rejects_mutated_required_connector_count(tmp_path):
    artifact = tmp_path / "facts.json"
    artifact.write_text("{}")
    obligation = contract_for("rc-lowpass-bnc")["obligations"][0]
    result = {
        "status": "pass",
        "evidence": [{"kind": "tool-report", "path": artifact.name}],
        "facts": {"part_classes": {"bnc_connector": 2}},
    }
    assert evaluate_obligation(obligation, result, artifact_root=tmp_path) is None

    result["facts"]["part_classes"]["bnc_connector"] = 1
    assert "requires at least 2 bnc_connector parts" in evaluate_obligation(
        obligation, result, artifact_root=tmp_path
    )


class _EvidencePad:
    def __init__(self, number, net):
        self._number = number
        self._net = net

    def GetNumber(self):
        return self._number

    def GetNetname(self):
        return self._net


class _EvidenceFootprint:
    def __init__(self, ref, footprint, pads):
        self._ref = ref
        self._footprint = footprint
        self._pads = [_EvidencePad(*pad) for pad in pads]

    def GetReferenceAsString(self):
        return self._ref

    def GetFPID(self):
        return self._footprint

    def Pads(self):
        return self._pads


class _EvidenceBoard:
    def __init__(self, footprints):
        self._footprints = footprints

    def GetFootprints(self):
        return self._footprints


def _led_current_artifact(
    *,
    output_cathode="LED_CATHODE",
    diode_pads=(("1", "VBUS"), ("2", "SW")),
    resistor_value="100m",
    resistor_only=False,
    ep_net="GND",
):
    terminal_symbol = "screw-terminal-5mm-2p:WJ126V-5.0-2P"
    terminal_footprint = "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P"
    rows = [
        ("U1", "al8860:MSOP-8_L3.0-W3.0-P0.65-LS4.9-BL-EP1.8", (("1", "LED_ANODE"), ("2", "GND"), ("3", "GND"), ("5", "SW"), ("6", "SW"), ("8", "VBUS"), ("9", ep_net))),
        ("R3", "R_2512", (("1", "VBUS"), ("2", "LED_ANODE"))),
        ("C1", "C_1206", (("1", "VBUS"), ("2", "GND"))),
        ("D1", "D_SMA", diode_pads),
        ("L1", "L_6x6", (("1", "LED_CATHODE"), ("2", "SW"))),
        ("J1", terminal_footprint, (("1", "VBUS"), ("2", "GND"))),
        ("J2", terminal_footprint, (("1", "LED_ANODE"), ("2", output_cathode))),
    ]
    parts = [
        {"ref": "U1", "mpn": "AL8860MP-13", "symbol": "al8860:AL8860MP-13", "footprint": rows[0][1]},
        {"ref": "R3", "symbol": "Device:R", "footprint": rows[1][1], "value": resistor_value},
        {"ref": "C1", "symbol": "Device:C", "footprint": rows[2][1], "value": "10uF"},
        {"ref": "D1", "symbol": "Device:D_Schottky", "footprint": rows[3][1]},
        {"ref": "L1", "symbol": "Device:L", "footprint": rows[4][1]},
        *[
            {"ref": ref, "mpn": "WJ126V-5.0-02P-14-00A", "symbol": terminal_symbol, "footprint": terminal_footprint}
            for ref in ("J1", "J2")
        ],
    ]
    if resistor_only:
        absent = {"C1", "D1", "L1"}
        rows = [row for row in rows if row[0] not in absent]
        parts = [part for part in parts if part["ref"] not in absent]
    names = {
        ("U1", "1"): "SET", ("U1", "2"): "GND", ("U1", "3"): "GND",
        ("U1", "5"): "SW", ("U1", "6"): "SW", ("U1", "8"): "VIN", ("U1", "9"): "EP",
        ("D1", "1"): "K", ("D1", "2"): "A",
    }
    return {"bom": {"parts": parts}}, _EvidenceBoard([_EvidenceFootprint(*row) for row in rows]), names


def test_led_current_artifact_requires_complete_external_high_side_loop(tmp_path, monkeypatch):
    """Only the observed AL8860 pad loop can publish current/topology facts."""
    state, board, names = _led_current_artifact()
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, {})
    assert facts["led_current_a"] == pytest.approx(1.0)
    assert facts["net_paths"]["current_feedback"] is True
    assert facts["net_paths"]["switcher_support"] is True

    for kwargs in (
        {"resistor_only": True},
        {"ep_net": "EP_FLOAT"},
        {"output_cathode": "BROKEN_RETURN"},
        {"diode_pads": (("1", "SW"), ("2", "VBUS"))},
    ):
        state, board, names = _led_current_artifact(**kwargs)
        monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
        facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, {})
        assert "led_current_a" not in facts
        assert "current_feedback" not in facts["net_paths"]
        assert "switcher_support" not in facts["net_paths"]

    state, board, names = _led_current_artifact(resistor_value="50m")
    monkeypatch.setattr(electrical_artifact_evidence, "_pin_names", lambda *_: names)
    facts = electrical_artifact_evidence.extract_electrical_facts(tmp_path, state, board, {})
    assert "led_current_a" not in facts
    assert "current_feedback" not in facts["net_paths"]
    assert "switcher_support" not in facts["net_paths"]

    artifact = tmp_path / "overcurrent-facts.json"
    artifact.write_text("{}")
    current_path = next(
        obligation
        for obligation in contract_for("led-cc-driver")["obligations"]
        if obligation["id"] == "led-cc-driver.current-path"
    )
    assert evaluate_obligation(
        current_path,
        {
            "status": "pass",
            "evidence": [{"kind": "tool-report", "path": artifact.name}],
            "facts": {"led_current_a": 2.0},
        },
        artifact_root=tmp_path,
    ) == "led_current_a is above required maximum"
