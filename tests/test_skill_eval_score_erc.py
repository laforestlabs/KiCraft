"""The skill-eval consumer must use the ERC report for this synthesis check."""
import importlib.util
import json
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "skill_eval_score_run", Path(__file__).parent / "skill-eval/bin/score_run.py"
)
_SCORE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCORE)


def _report(path, errors):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sheets": [{"violations": [
        {"severity": "error"} for _ in range(errors)
    ]}]}))


def _synth(root, check):
    (root / "synthesis_check.json").write_text(json.dumps({"checks": [check]}))


def _erc(root):
    return _SCORE.collect_metrics(root, None, 0)["erc"]


def test_score_uses_selected_report_not_stale_shallower_report(tmp_path):
    _report(tmp_path / "old_erc.rpt", 3)
    _report(tmp_path / "nested" / "new_erc.rpt", 1)
    _synth(tmp_path, {"name": "9.12 ERC", "ok": False, "report_path": "new_erc.rpt"})
    assert _erc(tmp_path) == {"present": True, "errors": 1, "warnings": 0}


@pytest.mark.parametrize("pointer", [None, "missing_erc.rpt", "../old_erc.rpt"])
def test_score_does_not_fall_back_when_pointer_unusable(tmp_path, pointer):
    _report(tmp_path / "old_erc.rpt", 3)
    _synth(tmp_path, {"name": "9.12 ERC", "ok": False, "report_path": pointer})
    assert _erc(tmp_path)["present"] is False


def test_score_does_not_choose_ambiguous_selected_basename(tmp_path):
    _report(tmp_path / "a" / "new_erc.rpt", 1)
    _report(tmp_path / "b" / "new_erc.rpt", 2)
    _synth(tmp_path, {"name": "9.12 ERC", "ok": False, "report_path": "new_erc.rpt"})
    assert _erc(tmp_path)["present"] is False


def test_score_keeps_legacy_report_discovery(tmp_path):
    _report(tmp_path / "old_erc.rpt", 2)
    _synth(tmp_path, {"name": "9.12 ERC", "ok": False})
    assert _erc(tmp_path) == {"present": True, "errors": 2, "warnings": 0}
