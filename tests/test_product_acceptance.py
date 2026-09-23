"""Focused behavioral boundaries for external-brief product acceptance."""

from __future__ import annotations

import hashlib
import json
import zipfile

from kicraft.eval import artifact_evidence
from kicraft.eval import electrical_artifact_evidence
from kicraft.eval import product_acceptance as product


def _policy(**changes):
    policy = {
        "max_cost_usd": 1.0,
        "max_duration_s": 60.0,
        "max_park_rounds": 2,
        "build_timeout_s": 30.0,
    }
    policy.update(changes)
    return policy


def _record(**changes):
    record = {
        "slug": "external-demo",
        "design_committed": True,
        "design_status": "ok",
        "original_brief_hash": "sha256:" + "a" * 64,
        "execution_brief_hash": "sha256:" + "a" * 64,
        "manual_intervention": False,
        "clarification_assisted": False,
        "ledger_cost_usd": 0.10,
        "duration_s": 12.0,
        "park_rounds": 1,
        "build_rc": 0,
    }
    record.update(changes)
    return record


def _artifact_facts():
    return {
        "board_loaded": True,
        "bom_present": True,
        "delivered_board_path": "generated/board/board.kicad_pcb",
        "part_classes": {"bnc_connector": 2, "microcontroller": 0},
        "part_inventory": [
            {
                "mpn": "RC0603FR-0710KL",
                "manufacturer": "Yageo",
                "source_url": "https://example.test/part",
                "package": "0603",
                "rated_limits": {"voltage_v": 50},
            }
        ],
        "pin_mapping": [{"symbol_pin": "R1.1", "footprint_pad": "R1.1"}],
        "verified_export_paths": ["generated/board/board_fab_20260922.zip"],
        "artifacts": ["generated/board/board_fab_20260922.zip"],
        "build_gate": {"fab_acceptable": True},
        "gates": {
            "erc": "pass",
            "complete_required_connections": "pass",
            "geometry": "pass",
        },
    }


def _install_extracted_artifacts(monkeypatch, tmp_path, facts=None):
    paths = [
        ".kicraft/state.json",
        ".kicraft/build_gate.json",
        ".kicraft/synthesis_check.json",
        "generated/board/board.kicad_pcb",
    ]
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
    archive_path = tmp_path / "generated/board/board_fab_20260922.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "board-job.gbrjob",
            json.dumps(
                {
                    "GeneralSpecs": {"ProjectId": {"Name": "board"}, "LayerNumber": 2},
                    "FilesAttributes": [
                        {"Path": "board-F_Cu.gtl", "FileFunction": "Copper,L1,Top"},
                        {"Path": "board-B_Cu.gbl", "FileFunction": "Copper,L2,Bot"},
                        {"Path": "board-Edge_Cuts.gm1", "FileFunction": "Profile"},
                    ],
                }
            ),
        )
        archive.writestr("board-PTH.drl", "M48\nM30")
        for filename in ("board-F_Cu.gtl", "board-B_Cu.gbl", "board-Edge_Cuts.gm1"):
            archive.writestr(filename, "%FSLAX46Y46*%\nM02*")
    with zipfile.ZipFile(archive_path) as archive:
        receipt = {
            "schema_version": 1,
            "board": "board.kicad_pcb",
            "board_sha256": hashlib.sha256(
                (archive_path.parent / "board.kicad_pcb").read_bytes()
            ).hexdigest(),
            "archive": archive_path.name,
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "files": {
                name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
            },
        }
    archive_path.with_suffix(".receipt.json").write_text(json.dumps(receipt))
    artifacts = [
        {"kind": "artifact", "path": relative_path}
        for relative_path in [*paths, "generated/board/board_fab_20260922.zip"]
        if relative_path != ".kicraft/synthesis_check.json"
    ]
    monkeypatch.setattr(
        product,
        "extract_artifact_facts",
        lambda rundir, contract: (_artifact_facts() if facts is None else facts, list(artifacts)),
    )

    def certify_drc(rundir, extracted, evidence, timeout_s):
        report = rundir / "eval/product_drc_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"violations": [], "unconnected_items": []}), encoding="utf-8")
        evidence.append({"kind": "tool-report", "path": "eval/product_drc_report.json"})
        extracted["drc_certification"] = {"report_path": "eval/product_drc_report.json"}
        extracted.setdefault("gates", {})["drc"] = "pass"
        return []

    monkeypatch.setattr(product, "_certify_drc", certify_drc)


def test_product_success_needs_real_artifact_backed_obligation_and_audits_it(monkeypatch, tmp_path):
    _install_extracted_artifacts(monkeypatch, tmp_path)
    obligation = {
        "id": "external-demo.bnc-count",
        "check": {"kind": "part_class_count", "part_class": "bnc_connector", "minimum": 2},
    }

    result = product.evaluate_product(tmp_path, _record(), [obligation], _policy())

    assert result["product_success"] is True
    assert result["product_failure_kind"] is None
    assert result["product_errors"] == []
    assert result["product_evidence_path"] == "eval/product_acceptance.json"
    saved = {**_record(), **result}
    assert (
        product.verify_product_audit(
            tmp_path, record=saved, obligations=[obligation], policy=_policy()
        )
        == []
    )
    extra_board = tmp_path / "generated/other/other.kicad_pcb"
    extra_board.parent.mkdir()
    extra_board.write_text("(kicad_pcb)")
    assert any(
        error.startswith("artifact_inventory_changed:")
        for error in product.verify_product_audit(
            tmp_path, record=saved, obligations=[obligation], policy=_policy()
        )
    )
    extra_board.unlink()
    (tmp_path / "generated/board/board_fab_20260922.zip").write_text("changed", encoding="utf-8")
    assert any(
        error.startswith("artifact_changed:")
        for error in product.verify_product_audit(
            tmp_path, record=saved, obligations=[obligation], policy=_policy()
        )
    )


def test_reused_success_rejects_replaced_failed_audit_and_changed_obligations(
    monkeypatch, tmp_path
):
    _install_extracted_artifacts(monkeypatch, tmp_path)
    obligations = [
        {
            "id": "external-demo.bnc-count",
            "check": {"kind": "part_class_count", "part_class": "bnc_connector", "minimum": 2},
        }
    ]
    saved = {**_record(), **product.evaluate_product(tmp_path, _record(), obligations, _policy())}
    assert saved["product_success"] is True
    assert product.verify_product_audit(tmp_path, record=saved, obligations=[], policy=_policy())
    failed = product.evaluate_product(tmp_path, _record(build_rc=7), obligations, _policy())
    assert failed["product_success"] is False
    assert any(
        error.startswith("audit_changed:")
        for error in product.verify_product_audit(
            tmp_path, record=saved, obligations=obligations, policy=_policy()
        )
    )
    saved["product_evidence_sha256"] = failed["product_evidence_sha256"]
    assert product.verify_product_audit(
        tmp_path, record=saved, obligations=obligations, policy=_policy()
    )


def test_capability_inventory_separates_defects_from_missing_proof():
    """Ranking must not confuse a violated requirement with absent evidence."""
    entries = [
        {"slug": "a", "eligible": True, "families": ["sensor"]},
        {"slug": "b", "eligible": True, "families": ["sensor"]},
        {"slug": "c", "eligible": True, "families": ["power"]},
    ]
    obligations = {
        "a": [
            {
                "id": "a.analog",
                "check": {"kind": "channel_count", "channel": "analog_input", "minimum": 4},
            },
            {
                "id": "a.power",
                "check": {"kind": "net_paths", "paths": ["power_led", "fused_input"]},
            },
        ],
        "b": [
            {
                "id": "b.analog",
                "check": {"kind": "channel_count", "channel": "analog_input", "minimum": 4},
            }
        ],
        "c": [
            {
                "id": "c.rail",
                "check": {
                    "kind": "numeric_range",
                    "fact": "input_voltage_v",
                    "minimum": 12,
                    "maximum": 12,
                },
            }
        ],
    }
    records = [
        {
            "slug": "a",
            "product_success": False,
            "product_failure_kind": "design_defect",
            "product_obligations": {"a.analog": {"status": "fail"}, "a.power": {"status": "pass"}},
        },
        {
            "slug": "b",
            "product_success": False,
            "product_failure_kind": "unverified_evidence",
            "product_obligations": {"b.analog": {"status": "unverified"}},
        },
        {
            "slug": "c",
            "product_success": True,
            "product_obligations": {"c.rail": {"status": "pass"}},
        },
    ]

    gaps = product.summarize_product(records, entries, obligations)["capability_gaps"]

    assert gaps["channel_count:analog_input"] == {
        "defect_runs": 1,
        "unverified_runs": 1,
        "passed_runs": 0,
    }
    # A multi-path obligation is inventoried per path, not as one opaque key.
    assert gaps["net_paths:power_led"]["passed_runs"] == 1
    assert gaps["net_paths:fused_input"]["passed_runs"] == 1
    assert gaps["numeric_range:input_voltage_v"]["defect_runs"] == 0
    assert next(iter(gaps)) == "channel_count:analog_input"


def test_product_rc0_without_verified_export_or_obligation_evidence_fails_closed(
    monkeypatch, tmp_path
):
    facts = _artifact_facts()
    facts["verified_export_paths"] = []
    _install_extracted_artifacts(monkeypatch, tmp_path, facts)
    obligation = {
        "id": "external-demo.board-outline",
        "check": {"kind": "outline", "shape": "circle"},
    }

    result = product.evaluate_product(tmp_path, _record(), [obligation], _policy())

    assert result["product_success"] is False
    assert result["product_failure_kind"] == "unverified_evidence"
    assert any("external-demo.board-outline" in error for error in result["product_errors"])


def test_product_policy_and_human_assistance_cannot_be_hidden_by_green_artifacts(
    monkeypatch, tmp_path
):
    _install_extracted_artifacts(monkeypatch, tmp_path)

    result = product.evaluate_product(
        tmp_path,
        _record(ledger_cost_usd=1.01, duration_s=61.0, clarification_assisted=True),
        [],
        _policy(),
    )

    assert result["product_success"] is False
    assert result["product_failure_kind"] == "clarification_assisted"
    kinds = {error.split(":", 1)[0] for error in result["product_errors"]}
    assert {"clarification_assisted", "cost_limit_exceeded", "duration_limit_exceeded"} <= kinds


def test_product_requires_an_external_obligation_even_when_common_evidence_is_green(
    monkeypatch, tmp_path
):
    _install_extracted_artifacts(monkeypatch, tmp_path)

    result = product.evaluate_product(tmp_path, _record(), [], _policy())

    assert result["product_success"] is False
    assert result["product_failure_kind"] == "unverified_evidence"


def test_product_missing_electrical_or_mechanical_baseline_proof_is_unverified(
    monkeypatch, tmp_path
):
    facts = _artifact_facts()
    facts["gates"].pop("complete_required_connections")
    facts["gates"].pop("geometry")
    _install_extracted_artifacts(monkeypatch, tmp_path, facts)

    result = product.evaluate_product(
        tmp_path,
        _record(),
        [
            {
                "id": "external-demo.parts",
                "check": {"kind": "part_class_count", "part_class": "bnc_connector", "minimum": 2},
            }
        ],
        _policy(),
    )

    assert result["product_success"] is False
    assert any(
        "product.complete-required-connections" in error for error in result["product_errors"]
    )
    assert any("product.closed-board-outline" in error for error in result["product_errors"])


def test_delivered_board_drc_report_with_an_error_is_a_design_defect(monkeypatch, tmp_path):
    board = tmp_path / "generated/board/board.kicad_pcb"
    board.parent.mkdir(parents=True)
    board.write_text("(kicad_pcb)", encoding="utf-8")
    facts = {"delivered_board_path": "generated/board/board.kicad_pcb"}
    artifacts = [{"kind": "artifact", "path": "generated/board/board.kicad_pcb"}]

    def fake_drc(command, **kwargs):
        report = tmp_path / command[command.index("-o") + 1]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({"violations": [], "unconnected_items": [{"type": "unconnected_items"}]}),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(product.subprocess, "run", fake_drc)

    errors = product._certify_drc(tmp_path, facts, artifacts, timeout_s=1)

    assert errors[0][0] == "design_defect"
    assert facts["drc_certification"]["status"] == "fail"
    assert (tmp_path / "eval/product_drc_report.json").is_file()


def test_artifact_extractor_selects_canonical_delivered_board_not_best_candidate(tmp_path):
    generated = tmp_path / "generated/RC_FILTER_BREAKOUT"
    generated.mkdir(parents=True)
    canonical = generated / "RC_FILTER_BREAKOUT.kicad_pcb"
    canonical.write_text("(kicad_pcb)", encoding="utf-8")
    (generated / "RC_FILTER_BREAKOUT_best.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")
    selected = artifact_evidence._delivered_board_path(tmp_path, {}, tmp_path / "generated")

    assert selected == canonical
    assert (
        artifact_evidence._delivered_board_path(
            tmp_path,
            {"artifacts": {"routed_pcb": str(generated / "RC_FILTER_BREAKOUT_best.kicad_pcb")}},
            tmp_path / "generated",
        )
        is None
    )


def test_electrical_extractor_uses_kicad_footprint_identity_not_swig_repr():
    class Fpid:
        def GetLibItemName(self):
            return "R_0603_1608Metric"

        def GetLibNickname(self):
            return "Resistor_SMD"

        def __str__(self):
            return "<Swig Object of type 'FPID'>"

    class Pad:
        def GetNumber(self):
            return "1"

        def GetNetname(self):
            return "NET"

    class Footprint:
        def GetReferenceAsString(self):
            return "R1"

        def GetFPID(self):
            return Fpid()

        def Pads(self):
            return [Pad()]

    class Board:
        def GetFootprints(self):
            return [Footprint()]

    _pads, _nets, footprints, _delivered_pads = electrical_artifact_evidence._board_graph(Board())

    assert footprints == {"R1": "Resistor_SMD:R_0603_1608Metric"}


def test_terminal_failure_kind_precedes_design_not_committed(monkeypatch, tmp_path):
    _install_extracted_artifacts(monkeypatch, tmp_path)

    result = product.evaluate_product(
        tmp_path,
        _record(design_committed=False, failure_kind="budget_exhausted"),
        [
            {
                "id": "external-demo.parts",
                "check": {"kind": "part_class_count", "part_class": "bnc_connector", "minimum": 2},
            }
        ],
        _policy(),
    )

    assert result["product_failure_kind"] == "budget_exhausted"


def test_product_summary_retains_missing_pending_and_duplicate_eligible_briefs():
    entries = [
        {"slug": "a", "eligible": True, "families": ["analog"]},
        {"slug": "b", "eligible": True, "families": ["analog", "power"]},
        {"slug": "c", "eligible": True, "families": ["power"]},
    ]
    summary = product.summarize_product(
        [
            {"slug": "a", "product_success": True},
            {"slug": "c", "product_success": False, "product_failure_kind": "design_defect"},
            {"slug": "c", "product_success": False, "product_failure_kind": "design_defect"},
        ],
        entries,
    )

    assert summary["eligible_n"] == 3
    assert summary["product_success_n"] == 1
    assert summary["missing_slugs"] == ["b"]
    assert summary["duplicate_slugs"] == ["c"]
    assert summary["family_stats"]["analog"]["product_yield"] == 0.5
    assert summary["failure_pareto"] == {"duplicate_run": 1, "missing_run": 1}
