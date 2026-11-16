"""Engineering-review copper must never cross the fabrication release boundary."""

import hashlib
import json
import os
import subprocess
import time

import pytest

from kicraft.autoplacer.brain import escape_cleanup
from kicraft.autoplacer.hardware import silk_legend
from kicraft.cli import artifact_paths
from kicraft.design import cli_app
from kicraft.design.models import Architecture, ArtifactPaths, BOM, ConversationState
from kicraft.design.synthesis import fab_export


@pytest.mark.parametrize("invalid_bounded_policy", [True, "true"])
def test_review_promotion_blocks_fabrication_even_with_green_verification(
    tmp_path, monkeypatch, invalid_bounded_policy,
):
    class ExportReached(Exception):
        pass

    calls = []

    def export_fab(*args, **kwargs):
        calls.append(args)
        raise ExportReached

    monkeypatch.setattr(fab_export, "export_fab", export_fab)
    monkeypatch.setattr(escape_cleanup, "cleanup_signal_escapes", lambda *a: {"accepted": True})
    monkeypatch.setattr(silk_legend, "apply_silk_legend", lambda *a: {"placed": [], "dropped": []})
    monkeypatch.setattr(cli_app, "_verify_routed_board", lambda *a, **k: {
        "ok": True, "fab_acceptable": True, "shorts": 0, "unconnected": 0,
        "tracks": {"footprint_refs": []}, "reasons": [],
        "usb_routing": {"geometry_ok": True},
    })
    monkeypatch.setattr(cli_app, "_castellation_geometry_violations", lambda *a: [])
    monkeypatch.setattr(cli_app, "_check_form_factor_conformance", lambda *a: None)
    monkeypatch.setattr(cli_app, "_check_outline_shape_conformance", lambda *a: None)
    monkeypatch.setattr(cli_app, "_connector_misoriented", lambda *a: ([], []))
    monkeypatch.setattr(cli_app, "_antenna_edge_contract_violations", lambda *a: [])
    monkeypatch.setattr(cli_app, "_persist_pcb_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    now = time.time()
    monkeypatch.setenv(artifact_paths.ENV_RUN_ID, "review-release-test")
    monkeypatch.setenv(artifact_paths.ENV_RUN_STARTED_AT, repr(now))

    def promote(mode, *, bounded=False, name=None):
        project = tmp_path / (name or mode)
        routed = project / ".experiments" / "subcircuits" / "subcircuit__test" / artifact_paths.PARENT_ROUTED
        routed.parent.mkdir(parents=True)
        routed.write_text("(kicad_pcb)\n")
        os.utime(routed, (now + 10, now + 10))
        (routed.parent / artifact_paths.METADATA_JSON).write_text(
            json.dumps({"run_id": "review-release-test"})
        )
        (project / "BOARD_autoplacer.json").write_text(json.dumps({
            "usb_geometry_mode": mode, "usb_review_allow_bounded_launches": bounded,
        }))
        pcb = project / "BOARD.kicad_pcb"
        state = ConversationState(
            architecture=Architecture(sheets=[], power_nets=[], inter_sheet_nets=[]),
            bom=BOM(parts=[]),
        )
        artifacts = ArtifactPaths(
            project_dir=project, project_stem="BOARD", root_sch=project / "root.kicad_sch",
            leaf_schs=[], kicad_pro=project / "BOARD.kicad_pro",
            autoplacer_json=project / "BOARD_autoplacer.json",
        )
        summary = {}
        result = cli_app._promote_verify_fab(
            state, project / "state.json", artifacts, "BOARD", project, pcb,
            do_fab=True, verify_summary_out=summary,
        )
        return result, pcb, summary

    # Prove the same otherwise-green path actually reaches the export boundary.
    with pytest.raises(ExportReached):
        promote("fabrication")
    assert len(calls) == 1

    rc, pcb, summary = promote("engineering_review")
    assert rc == 7
    assert len(calls) == 1
    verification = json.loads(pcb.with_suffix(".verification.receipt.json").read_text())
    review = json.loads(pcb.with_suffix(".engineering-review.receipt.json").read_text())
    assert verification["ok"] is False
    assert verification["fab_acceptable"] is False
    assert any("fabrication release blocked" in reason for reason in summary["reasons"])
    assert review["fabrication_approved"] is False
    assert review["provenance"] == "calculated_unconfirmed"
    assert review["board_sha256"] == hashlib.sha256(pcb.read_bytes()).hexdigest()
    assert review["verification_receipt"] == pcb.with_suffix(".verification.receipt.json").name

    rc, pcb, summary = promote(
        "fabrication", bounded=invalid_bounded_policy, name="inconsistent-policy",
    )
    assert rc == 7
    assert len(calls) == 1
    verification = json.loads(pcb.with_suffix(".verification.receipt.json").read_text())
    assert verification["fab_acceptable"] is False
    assert any("bounded USB" in reason for reason in summary["reasons"])
