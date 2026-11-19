"""Engineering-review copper must never cross the fabrication release boundary."""

import hashlib
import json
import os
import subprocess
import time
import zipfile

import pytest

from kicraft.autoplacer.brain import escape_cleanup
from kicraft.autoplacer.hardware import silk_legend
from kicraft.cli import artifact_paths
from kicraft.build_outcome import REVIEW_REQUIRED_RC
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
    physical_failure = False

    def export_fab(*args, **kwargs):
        calls.append(args)
        raise ExportReached

    monkeypatch.setattr(fab_export, "export_fab", export_fab)
    monkeypatch.setattr(escape_cleanup, "cleanup_signal_escapes", lambda *a: {"accepted": True})
    monkeypatch.setattr(escape_cleanup, "finish_boundary_escape_landings", lambda *a: {"accepted": True})
    monkeypatch.setattr(silk_legend, "apply_silk_legend", lambda *a: {"placed": [], "dropped": []})
    def verify(*args, **kwargs):
        return {
            "ok": not physical_failure, "physical_ok": not physical_failure,
            "fab_acceptable": not physical_failure and
                kwargs["physical_config"].get("usb_geometry_mode") != "engineering_review",
            "shorts": int(physical_failure), "unconnected": 0,
            "tracks": {"footprint_refs": []}, "reasons": ["short"] if physical_failure else [],
            "usb_routing": {"geometry_ok": not physical_failure},
        }
    monkeypatch.setattr(cli_app, "_verify_routed_board", verify)
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
        (project / "BOARD.kicad_sch").write_text("(kicad_sch)\n")
        (project / "BOARD.kicad_pro").write_text("{}\n")
        (project / "old.gbr").write_text("stale manufacturing data")
        (project / "old_fab.zip").write_text("stale archive")
        state = ConversationState(
            architecture=Architecture(sheets=[], power_nets=[], inter_sheet_nets=[]),
            bom=BOM(parts=[]),
        )
        artifacts = ArtifactPaths(
            project_dir=project, project_stem="BOARD", root_sch=project / "BOARD.kicad_sch",
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
    assert rc == REVIEW_REQUIRED_RC
    assert len(calls) == 1
    verification = json.loads(pcb.with_suffix(".verification.receipt.json").read_text())
    review = json.loads(pcb.with_suffix(".engineering-review.receipt.json").read_text())
    assert verification["ok"] is True
    assert verification["physical_ok"] is True
    assert verification["fab_acceptable"] is False
    assert verification["review_required"] is True
    assert review["physical_ok"] is True
    assert review["fabrication_approved"] is False
    assert review["provenance"] == "calculated_unconfirmed"
    assert review["board_sha256"] == hashlib.sha256(pcb.read_bytes()).hexdigest()
    assert review["verification_receipt"] == pcb.with_suffix(".verification.receipt.json").name
    saved = json.loads((pcb.parent / "state.json").read_text())["artifacts"]
    assert saved["status"] == "review_required"
    assert saved["fab_zip"] is None
    with zipfile.ZipFile(saved["review_zip"]) as archive:
        assert archive.read("BOARD.kicad_pcb") == pcb.read_bytes()
        assert "BOARD.kicad_sch" in archive.namelist()
        assert "DRAFT-NOT-FOR-FABRICATION.txt" in archive.namelist()
        assert "old.gbr" not in archive.namelist()
        assert "old_fab.zip" not in archive.namelist()

    physical_failure = True
    rc, failed_pcb, _ = promote("engineering_review", name="physical-failure")
    assert rc == 7
    failed_review = json.loads(failed_pcb.with_suffix(".engineering-review.receipt.json").read_text())
    assert failed_review["physical_ok"] is False
    assert not (failed_pcb.parent / "BOARD_review.zip").exists()
    physical_failure = False

    with monkeypatch.context() as late_failure:
        late_failure.setattr(cli_app, "_verify_qualified_delivery", lambda *a: {"accepted": False})
        rc, failed_pcb, _ = promote("engineering_review", name="custody-failure")
        assert rc == 7
        failed_review = json.loads(failed_pcb.with_suffix(".engineering-review.receipt.json").read_text())
        assert failed_review["physical_ok"] is False
        assert not (failed_pcb.parent / "BOARD_review.zip").exists()

    rc, pcb, summary = promote(
        "fabrication", bounded=invalid_bounded_policy, name="inconsistent-policy",
    )
    assert rc == 7
    assert len(calls) == 1
    verification = json.loads(pcb.with_suffix(".verification.receipt.json").read_text())
    assert verification["fab_acceptable"] is False
    assert any("bounded USB" in reason for reason in summary["reasons"])


@pytest.mark.parametrize("geometry_ok", [False, True])
def test_review_usb_audit_separates_missing_approval_from_bad_copper(
    tmp_path, monkeypatch, geometry_ok,
):
    from kicraft.autoplacer import routing_board

    pcb = tmp_path / "BOARD.kicad_pcb"
    pcb.write_text("(kicad_pcb)\n")
    monkeypatch.setattr(routing_board, "validate_routed_board", lambda *a, **k: {
        "accepted": True, "drc": {}, "track_summary": {},
    })
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda *a: [])
    monkeypatch.setattr(cli_app, "_required_same_edge_violations", lambda *a: [])
    monkeypatch.setattr(cli_app, "_verify_dangling_copper", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(cli_app, "_verify_usb_routes", lambda *a: {
        "ok": False, "fabrication_approved": False, "geometry_ok": geometry_ok,
        "errors": ["fabrication confirmation missing"],
        "geometry_errors": [] if geometry_ok else ["USB skew exceeds limit"],
    })
    gate = cli_app._verify_routed_board(
        pcb, physical_config={"usb_geometry_mode": "engineering_review"},
    )
    assert gate["physical_ok"] is geometry_ok
    assert gate["fab_acceptable"] is False
    assert ("USB skew exceeds limit" in gate["reasons"]) is (not geometry_ok)
    assert "fabrication confirmation missing" not in gate["reasons"]


@pytest.mark.parametrize("fail_write", [False, True])
def test_review_archive_replacement_preserves_active_download(tmp_path, monkeypatch, fail_write):
    from types import SimpleNamespace

    pcb = tmp_path / "BOARD.kicad_pcb"
    pcb.write_bytes(b"old routed source")
    artifacts = SimpleNamespace(
        root_sch=tmp_path / "BOARD.kicad_sch", leaf_schs=[],
        kicad_pro=tmp_path / "BOARD.kicad_pro",
        autoplacer_json=tmp_path / "BOARD_autoplacer.json",
    )
    for name, content in (
        ("3dmodels/PART.step", b"model"), ("3dmodels/cache.db", b"not source"),
        ("custom.pretty/PART.kicad_mod", b"footprint"), ("custom.pretty/scratch.log", b"not source"),
    ):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
    archive = cli_app._archive_review_sources(tmp_path, "BOARD", artifacts)
    with zipfile.ZipFile(archive) as initial:
        assert initial.read("3dmodels/PART.step") == b"model"
        assert initial.read("custom.pretty/PART.kicad_mod") == b"footprint"
        assert "3dmodels/cache.db" not in initial.namelist()
        assert "custom.pretty/scratch.log" not in initial.namelist()
    original = archive.read_bytes()
    pcb.write_bytes(b"new routed source")
    with archive.open("rb") as active_download:
        if fail_write:
            def fail(*args, **kwargs):
                raise OSError("write failed")
            monkeypatch.setattr(zipfile.ZipFile, "write", fail)
            with pytest.raises(OSError, match="write failed"):
                cli_app._archive_review_sources(tmp_path, "BOARD", artifacts)
            assert archive.read_bytes() == original
        else:
            cli_app._archive_review_sources(tmp_path, "BOARD", artifacts)
            with zipfile.ZipFile(archive) as current:
                assert current.read("BOARD.kicad_pcb") == b"new routed source"
        assert active_download.read() == original
    assert not list(tmp_path.glob(".BOARD_review.zip.*"))
