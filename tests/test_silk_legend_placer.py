"""Integration tests for the silk-legend placer subprocess.

Runs the real pcbnew subprocess on the routed USB_PD_TRIGGER fixture board
and checks the placement contract: everything lands inside the outline with
edge margin, nothing overlaps courtyards/pads, and what cannot be placed is
dropped with a reason (never silently squeezed).
"""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pcbnew")

from kicraft.autoplacer.routing_board import run_pcbnew_script_file
from kicraft.autoplacer.hardware.silk_geometry import bbox_inside_poly, boxes_overlap

FIXTURE = (Path(__file__).parent / "fixtures" / "replay_workspace"
           / "USB_PD_TRIGGER" / "USB_PD_TRIGGER.kicad_pcb")
SCRIPT = (Path(__file__).parents[1] / "kicraft" / "autoplacer" / "hardware"
          / "_silk_legend_subprocess.py")


def _run(tmp_path: Path, labels: list[dict], legend_lines=None, *, source=FIXTURE) -> tuple[dict, Path]:
    board = tmp_path / "board.kicad_pcb"
    shutil.copy(source, board)
    result_path = tmp_path / "result.json"
    if legend_lines is None:
        legend_lines = [
            {"text": "USB-C PD Trigger", "height_mm": 1.2},
            {"text": "KiCraft KC-TEST rev 1.0 2026-01-01", "height_mm": 0.8},
        ]
    payload = {
        "pcb_path": str(board),
        "output_path": str(board),
        "result_path": str(result_path),
        "clearance_mm": 0.25,
        "edge_margin_mm": 0.5,
        "legend": {"lines": legend_lines, "gap_mm": 0.3},
        "labels": labels,
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    run_pcbnew_script_file(str(SCRIPT), str(payload_path))
    return json.loads(result_path.read_text()), board


def test_required_label_placement_failure_is_named(tmp_path):
    result, _ = _run(tmp_path, [
        {"id": "required:J999:usb", "text": "USB DATA", "ref": "J999",
         "required": True, "priority": 1},
    ], legend_lines=[])
    assert result["required_missing"] == ["required:J999:usb"]
    assert any(item["id"] == "required:J999:usb" for item in result["dropped"])


def _board_texts_and_geometry(board_path: Path):
    import pcbnew

    from kicraft.autoplacer.hardware._silk_legend_subprocess import (
        _collect_obstacles,
        _mm_box,
        _outline_poly,
    )

    board = pcbnew.LoadBoard(str(board_path))
    texts = [
        d for d in board.GetDrawings()
        if isinstance(d, pcbnew.PCB_TEXT)
        and d.GetLayer() in (pcbnew.F_SilkS, pcbnew.B_SilkS)
    ]
    return board, texts, _outline_poly(board), _mm_box


def test_legend_and_anchored_label_placed_inside_outline(tmp_path):
    result, board_path = _run(tmp_path, [
        {"id": "out-rating", "text": "OUT 9/12/20V", "ref": "J2",
         "prefer": "below", "priority": 1, "heights_mm": [1.0, 0.9, 0.8]},
    ])
    placed_ids = {p["id"] for p in result["placed"]}
    assert {"legend:0", "legend:1", "out-rating"} <= placed_ids
    assert result["dropped"] == []

    board, texts, poly, mm_box = _board_texts_and_geometry(board_path)
    new_texts = [t for t in texts if "KiCraft" in t.GetText()
                 or "OUT 9/12/20V" in t.GetText()
                 or "PD Trigger" in t.GetText()]
    assert len(new_texts) == 3
    for t in new_texts:
        assert bbox_inside_poly(mm_box(t.GetBoundingBox()), poly, 0.4)


def test_placed_text_clears_courtyards_and_pads(tmp_path):
    result, board_path = _run(tmp_path, [
        {"id": "dip-table", "kind": "table",
         "text": "VOUT  1 2 3\n 9V   ON - -\n12V   - ON -\n20V   - - ON",
         "ref": "SW1", "prefer": "right", "priority": 1,
         "heights_mm": [1.0, 0.9, 0.8]},
    ])
    assert "dip-table" in {p["id"] for p in result["placed"]}

    import pcbnew

    from kicraft.autoplacer.hardware._silk_legend_subprocess import (
        _collect_obstacles,
    )

    board, texts, poly, mm_box = _board_texts_and_geometry(board_path)
    table = next(t for t in texts if "VOUT" in t.GetText())
    table_box = mm_box(table.GetBoundingBox())
    fresh = pcbnew.LoadBoard(str(FIXTURE))
    for ob in _collect_obstacles(fresh)["F"]:
        assert not boxes_overlap(table_box, ob, 0.1)


def test_unplaceable_and_unanchored_labels_drop_with_reason(tmp_path):
    result, _ = _run(tmp_path, [
        {"id": "ghost", "text": "GHOST", "ref": "Z99", "priority": 2,
         "heights_mm": [0.8]},
        # priority-2 monster block: no anchored spot and no fallback sweep
        {"id": "monster", "text": "\n".join(["X" * 30] * 5), "ref": "U1",
         "priority": 2, "heights_mm": [8.0]},
    ])
    dropped = {d["id"]: d["reason"] for d in result["dropped"]}
    assert "Z99" in dropped["ghost"]
    assert "monster" in dropped


def test_result_reports_positions_for_placed_labels(tmp_path):
    result, _ = _run(tmp_path, [])
    for p in result["placed"]:
        assert {"id", "x_mm", "y_mm", "height_mm", "layer"} <= set(p)
        assert p["layer"] in ("F.SilkS", "B.SilkS")


def test_pinout_labels_placed_per_pin(tmp_path):
    import pcbnew

    from kicraft.autoplacer.hardware._silk_legend_subprocess import (
        _collect_obstacles,
        _mm_box,
    )

    source = _label_board(
        tmp_path, [(0, 0), (30, 0), (30, 30), (0, 30)],
        [("1", 10, 10), ("2", 12.54, 10), ("3", 15.08, 10), ("4", 17.62, 10)],
        (8.7, 8.7, 18.92, 11.3),
    )
    src = pcbnew.LoadBoard(str(source))
    fp = src.FindFootprintByReference("J1")
    ref = fp.GetReference()
    pads = [p for p in fp.Pads() if p.GetNumber().strip()][:3]
    texts = {p.GetNumber(): f"P{i + 1}" for i, p in enumerate(pads)}

    label = {
        "id": "pins", "kind": "pinout", "ref": ref,
        "pins": [{"pin": p.GetNumber(), "text": texts[p.GetNumber()]}
                 for p in pads],
        "priority": 1, "heights_mm": [0.8],
    }
    result, board_path = _run(tmp_path, [label], legend_lines=[], source=source)

    per_pin_ids = [p["id"] for p in result["placed"]
                   if p["id"].startswith("pins:")]
    assert set(per_pin_ids) == {f"pins:{number}" for number in texts}

    board, board_texts, poly, mm_box = _board_texts_and_geometry(board_path)
    by_text = {t.GetText(): t for t in board_texts}
    pad_centers = {
        p.GetNumber(): ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for p in fp.Pads() for b in [_mm_box(p.GetBoundingBox())]
    }

    placed_boxes = {}
    for num, txt in texts.items():
        t = by_text.get(txt)
        if t is None:
            continue  # a pin may drop individually on a dense board
        box = mm_box(t.GetBoundingBox())
        placed_boxes[num] = box
        assert bbox_inside_poly(box, poly, 0.4)

    assert set(placed_boxes) == set(texts)

    # Every per-pin label clears courtyards/pads (the placer's own obstacle
    # set, margin 0.1 is tighter than the placer's 0.25 clearance).
    fresh = pcbnew.LoadBoard(str(source))
    for box in placed_boxes.values():
        for ob in _collect_obstacles(fresh)["F"]:
            assert not boxes_overlap(box, ob, 0.1)


    # Adjacency: each label sits nearest its OWN pad's center (the robust
    # form of "within a few mm of its own pad" — the placer backs off to a
    # larger gap when a tight gap would collide with a neighbour, so a fixed
    # 2.0 mm radius is not guaranteed on a dense board).
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    for num, box in placed_boxes.items():
        tc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        own = pad_centers[num]
        assert all(_dist(tc, own) < _dist(tc, pad_centers[n])
                   for n in pad_centers if n != num)


def _label_board(tmp_path, outline, contacts, courtyard):
    import pcbnew

    source = tmp_path / "seed.kicad_pcb"
    board = pcbnew.NewBoard(str(source))
    point = lambda x, y: pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
    for start, end in zip(outline, outline[1:] + outline[:1]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(point(*start))
        edge.SetEnd(point(*end))
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetWidth(pcbnew.FromMM(0.05))
        board.Add(edge)
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    fp.Reference().SetVisible(False)
    fp.Value().SetVisible(False)
    board.Add(fp)
    court = pcbnew.PCB_SHAPE(fp)
    court.SetShape(pcbnew.SHAPE_T_RECT)
    court.SetStart(point(*courtyard[:2]))
    court.SetEnd(point(*courtyard[2:]))
    court.SetLayer(pcbnew.F_CrtYd)
    court.SetWidth(pcbnew.FromMM(0.05))
    fp.Add(court)
    for number, x, y in contacts:
        pad = pcbnew.PAD(fp)
        pad.SetNumber(number)
        pad.SetPosition(point(x, y))
        pad.SetSize(point(1, 1))
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        fp.Add(pad)
    board.Save(str(source))
    return source


@pytest.mark.parametrize("kind,text", [("pinout", "SIG"), ("io", "IN OUT")])
def test_required_label_uses_clear_local_slot_in_concave_outline(tmp_path, kind, text):
    source = _label_board(
        tmp_path,
        [(9, 9), (11.4, 9), (11.4, 10.25), (16.2, 10.25),
         (16.2, 15.5), (12.3, 15.5), (12.3, 11.5), (9, 11.5)],
        [("1", 10, 10)], (9, 9, 11, 11),
    )
    label = {"id": "required:J1", "kind": kind, "ref": "J1",
             "required": True, "priority": 1, "heights_mm": [0.8]}
    if kind == "pinout":
        label["pins"] = [{"pin": "1", "text": text}]
    else:
        label["text"] = text
    result, path = _run(tmp_path, [label], legend_lines=[], source=source)
    assert result["required_missing"] == []
    board, texts, poly, mm_box = _board_texts_and_geometry(path)
    assert len(texts) == 1
    assert " ".join(texts[0].GetText().split()) == text
    box = mm_box(texts[0].GetBoundingBox())
    assert box[0] > 12.3  # The only usable space is the local wing.
    assert bbox_inside_poly(box, poly, 0.4)
    from kicraft.autoplacer.hardware._silk_legend_subprocess import _collect_obstacles
    import pcbnew
    original = pcbnew.LoadBoard(str(source))
    assert not any(boxes_overlap(box, ob, 0.2) for ob in _collect_obstacles(original)["F"])


def test_required_pin_label_cannot_identify_unlabelled_nc_contact(tmp_path):
    source = _label_board(
        tmp_path, [(9, 8.3), (18, 8.3), (18, 12), (9, 12)],
        [("1", 10, 10), ("2", 12, 10)], (9.4, 9.4, 12.6, 10.6),
    )
    result, _ = _run(tmp_path, [{
        "id": "required:J1", "kind": "pinout", "ref": "J1",
        "required": True, "priority": 1, "heights_mm": [0.8],
        "pins": [{"pin": "1", "text": "SIG"}],
    }], legend_lines=[], source=source)
    assert result["required_missing"] == ["required:J1:1"]
    assert not result["placed"]


def test_contact_annotation_keeps_its_slot_before_flexible_title(tmp_path):
    source = _label_board(
        tmp_path,
        [(9, 9), (11.4, 9), (11.4, 10.25), (15.8, 10.25),
         (15.8, 11.5), (22, 11.5), (22, 18), (17, 18), (17, 12.5),
         (15.8, 12.5), (15.8, 13), (12.3, 13), (12.3, 11.5), (9, 11.5)],
        [("1", 10, 10)], (9, 9, 11, 11),
    )
    result, path = _run(tmp_path, [
        {"id": "required:title", "kind": "io", "ref": "J1", "text": "IN",
         "required": True, "priority": 1, "heights_mm": [0.8]},
        {"id": "required:contact", "kind": "pinout", "ref": "J1",
         "pins": [{"pin": "1", "text": "SIG"}],
         "required": True, "priority": 1, "heights_mm": [0.8]},
    ], legend_lines=[], source=source)
    assert result["required_missing"] == []
    _, texts, poly, mm_box = _board_texts_and_geometry(path)
    boxes = {text.GetText(): mm_box(text.GetBoundingBox()) for text in texts}
    assert boxes["SIG"][2] < 16
    assert boxes["IN"][0] > 17
    assert all(bbox_inside_poly(box, poly, 0.4) for box in boxes.values())


def test_dense_header_keeps_interior_contacts_unambiguous(tmp_path):
    import pcbnew

    source = _label_board(
        tmp_path, [(4.5, 4.2), (17, 4.2), (17, 13.2), (4.5, 13.2)],
        [("1", 9, 10), ("2", 10, 10), ("3", 11, 10), ("4", 12, 10)],
        (8, 8.8, 13, 12.5),
    )
    board = pcbnew.LoadBoard(str(source))
    blocker = pcbnew.FOOTPRINT(board)
    blocker.SetReference("U1")
    blocker.Reference().SetVisible(False)
    blocker.Value().SetVisible(False)
    board.Add(blocker)
    body = pcbnew.PCB_SHAPE(blocker)
    body.SetShape(pcbnew.SHAPE_T_RECT)
    body.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(5.5), pcbnew.FromMM(8)))
    body.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(8), pcbnew.FromMM(11.6)))
    body.SetLayer(pcbnew.F_CrtYd)
    body.SetWidth(pcbnew.FromMM(0.05))
    blocker.Add(body)
    board.Save(str(source))
    names = {"1": "GND", "2": "3V3", "3": "SDA", "4": "SCL"}
    result, path = _run(tmp_path, [{
        "id": "required:bus", "kind": "pinout", "ref": "J1",
        "pins": [{"pin": pin, "text": name} for pin, name in names.items()],
        "required": True, "priority": 1, "heights_mm": [0.8],
    }], legend_lines=[], source=source)
    assert result["required_missing"] == []
    _, texts, _, mm_box = _board_texts_and_geometry(path)
    boxes = {text.GetText(): mm_box(text.GetBoundingBox()) for text in texts}
    assert set(boxes) == set(names.values())
    for pin, name in names.items():
        box = boxes[name]
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        own = (center[0] - (8 + int(pin))) ** 2 + (center[1] - 10) ** 2
        assert all(own < (center[0] - (8 + int(other))) ** 2 + (center[1] - 10) ** 2
                   for other in names if other != pin)


def test_required_contact_reflows_reference_without_moving_electronics(tmp_path):
    import pcbnew

    source = _label_board(
        tmp_path,
        [(9, 9), (11.4, 9), (11.4, 10.25), (15.8, 10.25),
         (15.8, 13), (12.3, 13), (12.3, 11.5), (9, 11.5)],
        [("1", 10, 10)], (9, 9, 11, 11),
    )
    board = pcbnew.LoadBoard(str(source))
    reference = board.FindFootprintByReference("J1").Reference()
    reference.SetVisible(True)
    reference.SetLayer(pcbnew.F_SilkS)
    reference.SetTextSize(pcbnew.VECTOR2I(pcbnew.FromMM(0.8), pcbnew.FromMM(0.8)))
    reference.SetTextThickness(pcbnew.FromMM(0.12))
    reference.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(14), pcbnew.FromMM(11.6)))
    board.Save(str(source))
    result, path = _run(tmp_path, [{
        "id": "required:contact", "kind": "pinout", "ref": "J1",
        "pins": [{"pin": "1", "text": "SIG"}],
        "required": True, "priority": 1, "heights_mm": [0.8],
    }], legend_lines=[], source=source)
    assert result["required_missing"] == []
    actual, texts, poly, mm_box = _board_texts_and_geometry(path)
    assert [text.GetText() for text in texts] == ["SIG"]
    assert bbox_inside_poly(mm_box(texts[0].GetBoundingBox()), poly, 0.4)
    fp = actual.FindFootprintByReference("J1")
    assert fp.Reference().IsVisible() and fp.Reference().GetLayer() == pcbnew.F_Fab
    pad = next(pad for pad in fp.Pads() if pad.GetNumber() == "1")
    assert (pad.GetPosition().x, pad.GetPosition().y) == (pcbnew.FromMM(10), pcbnew.FromMM(10))


def test_full_contact_names_fit_rotated_lanes_without_moving_pads(tmp_path):
    import pcbnew

    contacts = [(str(i + 1), 7.5, 6 + i * 2.54) for i in range(5)]
    source = _label_board(
        tmp_path, [(0, 0), (10, 0), (10, 22), (0, 22)],
        contacts, (5.7, 4.2, 9.3, 18),
    )
    names = ["BUTTON_1", "BUTTON_2", "BUTTON_3", "GND", "3.3V"]
    result, path = _run(tmp_path, [{
        "id": "required:signals", "kind": "pinout", "ref": "J1",
        "pins": [{"pin": str(i + 1), "text": name} for i, name in enumerate(names)],
        "required": True, "priority": 1, "heights_mm": [0.8],
    }], legend_lines=[], source=source)
    assert result["required_missing"] == []
    actual, texts, poly, mm_box = _board_texts_and_geometry(path)
    assert {text.GetText() for text in texts} == set(names)
    assert any(text.GetTextAngle().AsDegrees() == 90 for text in texts)
    original = pcbnew.LoadBoard(str(source))
    from kicraft.autoplacer.hardware._silk_legend_subprocess import _collect_obstacles

    obstacles = _collect_obstacles(original)["F"]
    boxes = {text.GetText(): mm_box(text.GetBoundingBox()) for text in texts}
    for i, name in enumerate(names):
        box = boxes[name]
        assert bbox_inside_poly(box, poly, 0.5)
        assert not any(boxes_overlap(box, ob, 0.25) for ob in obstacles)
        assert not any(boxes_overlap(box, ob, 0.25)
                       for other, ob in boxes.items() if other != name)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        own = (center[0] - contacts[i][1]) ** 2 + (center[1] - contacts[i][2]) ** 2
        assert all(own < (center[0] - x) ** 2 + (center[1] - y) ** 2
                   for j, (_, x, y) in enumerate(contacts) if j != i)
    pads = actual.FindFootprintByReference("J1").Pads()
    assert {pad.GetNumber(): (pad.GetPosition().x, pad.GetPosition().y) for pad in pads} == {
        number: (pcbnew.FromMM(x), pcbnew.FromMM(y)) for number, x, y in contacts
    }
