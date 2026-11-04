import json

import pytest

from kicraft.cli.triage import collect_erc


@pytest.mark.parametrize("pointer", ["BOARD.fresh_erc.rpt", None, "missing_erc.rpt"])
def test_triage_uses_only_current_erc_report(tmp_path, pointer):
    generated = tmp_path / "generated" / "BOARD"
    generated.mkdir(parents=True)
    old = generated / "BOARD_erc.rpt"
    old.write_text(json.dumps({"sheets": [{"violations": []}]}))
    fresh = generated / "BOARD.fresh_erc.rpt"
    fresh.write_text(json.dumps({"sheets": [{"violations": [
        {"severity": "error", "description": "current failure"}
    ]}]}))
    metadata = tmp_path / ".kicraft"
    metadata.mkdir()
    (metadata / "synthesis_check.json").write_text(json.dumps({
        "checks": [{"name": "9.12 ERC", "report_path": pointer}]
    }))

    result = collect_erc(tmp_path)
    if pointer == fresh.name:
        assert result["path"] == str(fresh)
        assert [e["description"] for e in result["errors"]] == ["current failure"]
    else:
        assert result == {"present": False}
