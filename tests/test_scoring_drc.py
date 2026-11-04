from pathlib import Path
import shutil

import pytest

pytest.importorskip("pcbnew")

from kicraft.scoring.drc_check import DRCCheck


@pytest.mark.skipif(shutil.which("kicad-cli") is None, reason="kicad-cli is unavailable")
def test_drc_scores_real_board_but_not_missing_board(tmp_path):
    board = Path(__file__).parent / "fixtures/replay_workspace/PARENT_LOCAL_CONN/PARENT_LOCAL_CONN.kicad_pcb"
    check = DRCCheck()
    valid = check._run_kicad_cli(str(board))
    assert valid.summary != "DRC failed"
    assert "total_violations" in valid.metrics

    missing = check._run_kicad_cli(str(tmp_path / "missing.kicad_pcb"))
    assert missing.score == 0
    assert any(issue.severity == "error" for issue in missing.issues)


def test_incomplete_drc_reports_do_not_score_as_clean():
    check = DRCCheck()
    for report in ("", "** Found 0 DRC violations **\n", "** Found 1 DRC violations **\n** Found 0 unconnected pads **\n"):
        result = check._parse_report(report)
        assert result.score == 0
        assert any(issue.severity == "error" for issue in result.issues)
