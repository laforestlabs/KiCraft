"""Release archives contain one hash-bound promoted PCB and fabrication set."""
import hashlib
import json
import zipfile

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
