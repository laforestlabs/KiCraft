"""Web app: the read-only silkscreen card on the Fab inspector tab (Phase 4a).

``web._silk_sections(sj, arts)`` turns the authored ``state.silk_plan`` plus the
build-tail placement truth (``artifacts.silk_placed`` / ``silk_dropped``) into
the inspector section dicts. Pure function, tested directly like the other
``kicraft.server`` tests."""
from __future__ import annotations

from kicraft.server import web


def _by_title(secs):
    return {s.get("title"): s for s in secs}


def test_no_plan_yields_nothing():
    assert web._silk_sections({}, {}) == []
    assert web._silk_sections({"silk_plan": None}, {"status": "fab_ready"}) == []


def test_full_plan_builds_summary_labels_and_drops():
    sj = {"silk_plan": {
        "version": 1, "title": "USB-C PD Trigger", "board_code": "KC-TEST", "rev": "1.0",
        "labels": [
            {"id": "usb-in", "kind": "io", "text": "IN 9/12/20V",
             "anchor": {"ref": "J1", "prefer": "above"}, "priority": 1},
            {"id": "dip", "kind": "table", "text": "1:9V\n2:12V\n3:20V",
             "anchor": {"ref": "SW1"}, "priority": 2},
        ],
        "dropped_at_lint": ["note-x: uncorroborated claim"],
    }}
    arts = {"status": "fab_ready",
            "silk_placed": ["legend:0", "legend:1", "usb-in"],
            "silk_dropped": ["dip: no clear space on silk"]}

    secs = web._silk_sections(sj, arts)
    by = _by_title(secs)

    # kv summary: legend lines split from labels; placed X / total
    kv = dict(by["Silkscreen"]["rows"])
    assert kv["title"] == "USB-C PD Trigger"
    assert kv["board code"] == "KC-TEST"
    assert kv["legend lines"] == 2
    assert kv["labels placed"] == "1 / 2"
    assert kv["dropped (no space)"] == 1

    # labels table: DIP newline flattened, placed column reflects silk_placed
    tbl = by["Board labels"]
    assert tbl["columns"] == ["type", "text", "near", "priority", "placed"]
    rows = {r[0]: r for r in tbl["rows"]}
    assert rows["io"][1] == "IN 9/12/20V"
    assert rows["io"][2] == "J1"
    assert rows["io"][4] == "yes"           # in silk_placed
    assert rows["table"][1] == "1:9V / 2:12V / 3:20V"
    assert rows["table"][4] == "no"         # dropped, not placed
    assert tbl["note"] and "verify" in tbl["note"].lower()  # DIP verify caution

    # drop lists surfaced honestly
    assert by["Dropped — no clear silk space"]["items"] == ["dip: no clear space on silk"]
    assert by["Rejected by content lint"]["items"] == ["note-x: uncorroborated claim"]


def test_plan_without_table_has_no_verify_note():
    sj = {"silk_plan": {"title": "Sensor", "board_code": "KC-Y", "rev": "1.0",
                        "labels": [{"id": "p", "kind": "note", "text": "3V3 only"}]}}
    arts = {"status": "fab_ready", "silk_placed": ["legend:0", "p"], "silk_dropped": []}
    tbl = _by_title(web._silk_sections(sj, arts))["Board labels"]
    assert tbl.get("note") is None
