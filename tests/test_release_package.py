"""Release archives contain one hash-bound promoted PCB and fabrication set."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from kicraft.server.web import _zip_generated


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(tmp_path):
    project = tmp_path / "generated" / "BOARD"
    project.mkdir(parents=True)
    board = project / "BOARD.kicad_pcb"
    board.write_text("promoted copper")
    (project / "BOARD.provenance.json").write_text(json.dumps({"promoted_pcb": str(board)}))
    (project / "BOARD.kicad_pro").write_text("project")
    (project / "BOARD.kicad_sch").write_text("schematic")
    (project / "BOARD_best.kicad_pcb").write_text("intermediate")
    (project / "router.log").write_text("internal")
    fab = project / "fab"
    fab.mkdir()
    (fab / "fabrication_instructions.txt").write_text("2 oz, via-in-pad process")
    (fab / "operating_interfaces.txt").write_text("power separately")
    archive = project / "BOARD_fab_20260101.zip"
    archive.write_text("fab archive")
    receipt = {"board": board.name, "board_sha256": _sha(board),
               "archive": archive.name, "archive_sha256": _sha(archive),
               "files": {p.name: _sha(p) for p in fab.iterdir()}}
    (project / "BOARD_fab_20260101.receipt.json").write_text(json.dumps(receipt))
    for suffix in (".silk.receipt.json", ".escape-cleanup.receipt.json",
                   ".dangling.receipt.json", ".verification.receipt.json"):
        board.with_suffix(suffix).write_text(json.dumps({
            "board_sha256": _sha(board), "ok": True, "fab_acceptable": True,
        }))
    return project, board


def test_release_excludes_intermediate_and_binds_receipts(tmp_path):
    project, board = _release(tmp_path)
    usb = board.with_suffix(".usb-routing.receipt.json")
    usb.write_text(json.dumps({"board_sha256": _sha(board), "ok": True}))
    result = _zip_generated(tmp_path, project)
    with zipfile.ZipFile(result) as archive:
        names = archive.namelist()
        assert "BOARD/BOARD.kicad_pcb" in names
        assert "BOARD/BOARD_best.kicad_pcb" not in names
        assert "BOARD/router.log" not in names
        assert "BOARD/fab/operating_interfaces.txt" in names
        assert "BOARD/BOARD.usb-routing.receipt.json" in names
        manifest = json.loads(archive.read("BOARD/release_manifest.json"))
        assert manifest["board_sha256"] == _sha(board)
    receipt = json.loads((tmp_path / "kicraft_project.receipt.json").read_text())
    assert receipt["board_sha256"] == _sha(board)
    assert receipt["archive_sha256"] == _sha(tmp_path / "kicraft_project.zip")


def test_release_rejects_stale_fab_board(tmp_path):
    project, board = _release(tmp_path)
    board.write_text("changed copper")
    with pytest.raises(ValueError, match="does not match promoted PCB"):
        _zip_generated(tmp_path, project)


def test_release_rejects_stale_usb_receipt(tmp_path):
    project, board = _release(tmp_path)
    board.with_suffix(".usb-routing.receipt.json").write_text(
        json.dumps({"board_sha256": "obsolete"}))
    with pytest.raises(ValueError, match="USB physical receipt"):
        _zip_generated(tmp_path, project)


def test_release_rejects_unplaced_required_silk(tmp_path):
    project, board = _release(tmp_path)
    board.with_suffix(".silk.receipt.json").write_text(json.dumps({
        "board_sha256": _sha(board), "ok": False,
        "errors": ["J1 required power label could not be placed"],
    }))
    with pytest.raises(ValueError, match="Silkscreen receipt records an incomplete delivery"):
        _zip_generated(tmp_path, project)


def test_release_requires_operating_sheet_in_fabrication_receipt(tmp_path):
    project, _ = _release(tmp_path)
    receipt_path = project / "BOARD_fab_20260101.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["files"]["operating_interfaces.txt"]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="hash-bound operating/interface sheet"):
        _zip_generated(tmp_path, project)


@pytest.mark.parametrize("suffix", [
    ".silk.receipt.json", ".escape-cleanup.receipt.json",
    ".dangling.receipt.json", ".verification.receipt.json",
])
def test_release_rejects_missing_physical_delivery_proof(tmp_path, suffix):
    project, board = _release(tmp_path)
    board.with_suffix(suffix).unlink()
    with pytest.raises(ValueError, match="receipt is required for release"):
        _zip_generated(tmp_path, project)


@pytest.mark.parametrize("record", [{"ok": True, "fab_acceptable": False}, {}])
def test_release_rejects_failed_or_unknown_fabrication_verdict(tmp_path, record):
    project, board = _release(tmp_path)
    board.with_suffix(".verification.receipt.json").write_text(json.dumps({
        "board_sha256": _sha(board), **record,
    }))
    with pytest.raises(ValueError, match="Verification receipt records an incomplete delivery"):
        _zip_generated(tmp_path, project)


def _history_release(project, board):
    from kicraft.layout_history import atomic_json, start_capture
    from kicraft.layout_history_export import finalize_capture
    root = start_capture(project, run_id="release")
    root.scene([], phase="delivered", geometry_source="native_board")
    root.end("complete")
    atomic_json(Path(root.descriptor.directory) / "final-board.json", {
        "source_scene": {"scope_id": root.scope_id}, "final_board_path": board.name,
        "final_board_sha256": _sha(board),
    })
    return finalize_capture(project, root.descriptor.capture_id, build_rc=0)


def test_release_binds_only_current_finalized_history(tmp_path):
    project, board = _release(tmp_path)
    manifest_path = _history_release(project, board)
    history = json.loads(manifest_path.read_text())
    result = _zip_generated(tmp_path, project)
    with zipfile.ZipFile(result) as archive:
        release = json.loads(archive.read("BOARD/release_manifest.json"))
        html = "BOARD/" + history["html_path"]
        assert html in archive.namelist()
        assert release["files"][history["html_path"]] == hashlib.sha256(archive.read(html)).hexdigest()
        assert not any(name.endswith(".jsonl.gz") for name in archive.namelist())


def test_release_omits_hash_mismatched_history_without_failing_fab(tmp_path):
    project, board = _release(tmp_path)
    manifest_path = _history_release(project, board)
    (manifest_path.parent / "timelapse.html").write_text("changed")
    result = _zip_generated(tmp_path, project)
    with zipfile.ZipFile(result) as archive:
        assert not any("/layout-history/" in name for name in archive.namelist())
