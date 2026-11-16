"""Consumer-visible reconstruction, prefix recovery and isolation guarantees."""
import gzip
import json
from pathlib import Path

import pytest

from kicraft.layout_history import HistoryDescriptor, history_context, run_capture, start_capture
from kicraft.layout_history_export import (
    _embedded_data, apply_record, check_publication, finalize_capture, recover_scope, RecoveredScope,
)


def _track(key):
    return {"id": key, "type": "track", "start": [0, 0], "end": [2, 0],
            "width": 0.2, "layer": "In1.Cu", "net": "A"}


def _scope(tmp_path):
    return HistoryDescriptor(str(tmp_path), "a" * 32).writer("placement", "test")


def test_coincident_objects_survive_single_removal(tmp_path):
    writer = _scope(tmp_path)
    writer.scene([_track("one"), _track("two")], phase="input", geometry_source="router")
    writer.delta(remove=["one"], phase="rip")
    writer.delta(put=[_track("one")], phase="restore")
    writer.delta(remove=["two"], phase="cleanup")
    writer.end("complete")
    recovered = recover_scope(writer.path)
    assert not recovered.gaps
    assert recovered.objects == {"one": _track("one")}


def test_repair_then_rollback_is_inspectable(tmp_path):
    writer = _scope(tmp_path)
    writer.scene([_track("original")], phase="input", geometry_source="native_board")
    writer.delta(put=[_track("repair")], phase="stamped_repair")
    writer.decision("reject", phase="repair", reason="DRC measured worse", score=9)
    writer.delta(remove=["repair"], phase="byte_restore")
    writer.end("rejected")
    state = RecoveredScope()
    snapshots = {}
    for record in map(json.loads, gzip.decompress(writer.path.read_bytes()).splitlines()):
        apply_record(state, record)
        snapshots[record.get("phase")] = set(state.objects)
    assert snapshots["stamped_repair"] == {"original", "repair"}
    assert snapshots["byte_restore"] == {"original"}


def test_corrupt_interior_stops_before_mutation(tmp_path):
    writer = _scope(tmp_path)
    writer.scene([_track("one")], phase="input", geometry_source="router")
    writer.emit("delta", phase="bad", remove=["one", "unknown"], put=[])
    writer.delta(put=[_track("later")], phase="later")
    writer.end("complete")
    recovered = recover_scope(writer.path)
    assert recovered.last_seq == 1
    assert set(recovered.objects) == {"one"}
    assert any("removal" in gap for gap in recovered.gaps)


def test_truncated_member_retains_only_complete_prefix(tmp_path):
    writer = _scope(tmp_path)
    writer.scene([_track("one")], phase="input", geometry_source="router")
    writer.delta(put=[_track("tail")], phase="tail")
    writer.end("complete")
    writer.path.write_bytes(writer.path.read_bytes()[:-5])
    recovered = recover_scope(writer.path)
    assert set(recovered.objects) == {"one"}
    assert not recovered.ended
    assert any("truncated" in gap for gap in recovered.gaps)


def test_selected_winner_is_not_last_attempt_and_round_ids_are_distinct(tmp_path):
    root = start_capture(tmp_path, run_id="r", generation=3)
    first = root.child("leaf", "round", {"round": 1, "canvas": 2})
    second = root.child("leaf", "round", {"round": 1, "canvas": 2})
    assert first.scope_id != second.scope_id
    for scope, key in ((first, "winner"), (second, "loser")):
        scope.scene([_track(key)], phase="input", geometry_source="router")
        scope.end("complete")
    root.decision("select", phase="winner", source_scene={"scope_id": first.scope_id})
    root.end("complete")
    manifest = finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=6)
    assert manifest
    data = _embedded_data(manifest.parent / "timelapse.html")
    run = next(item for item in data["scopes"] if item["header"]["scope_id"] == root.scope_id)
    records = [json.loads(line) for member in run["members"]
               for line in gzip.decompress(__import__("base64").b64decode(member["data"])).splitlines()]
    assert next(record for record in records if record.get("action") == "select")["source_scene"]["scope_id"] == first.scope_id
    assert json.loads(manifest.read_text())["build_rc"] == 6
    assert check_publication(tmp_path) == []
    original = manifest.read_bytes()
    assert finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=6) == manifest
    assert manifest.read_bytes() == original


def test_old_capture_cannot_replace_new_current_pointer(tmp_path):
    old = start_capture(tmp_path, run_id="same", generation=1)
    new = start_capture(tmp_path, run_id="same", generation=2)
    assert old.descriptor.capture_id != new.descriptor.capture_id
    old.end("complete")
    finalize_capture(tmp_path, old.descriptor.capture_id, build_rc=0)
    pointer = json.loads((tmp_path / "layout-history.json").read_text())
    assert pointer["capture_id"] == new.descriptor.capture_id
    assert pointer["status"] == "recording"


def test_missing_source_is_a_visible_gap(tmp_path):
    root = start_capture(tmp_path, run_id="r")
    root.decision("reuse", phase="reuse", source_scene={"scope_id": "f" * 32})
    root.end("complete")
    manifest = finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=0)
    data = json.loads(manifest.read_text())
    assert data["status"] == "partial"
    assert any("missing source scene" in gap for gap in data["gaps"])


def test_writer_io_error_does_not_replace_build_result(tmp_path, monkeypatch):
    original = Path.open
    def fail_scope(path, *args, **kwargs):
        if path.name.startswith("scope-") and "a" in (args[0] if args else kwargs.get("mode", "r")):
            raise OSError("disk full")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_scope)
    with pytest.warns(RuntimeWarning, match="layout-history"):
        rc, _ = run_capture(tmp_path, tmp_path / "board.kicad_pcb", lambda: 7)
    assert rc == 7


def test_explicit_context_is_isolated_and_child_env_does_not_mutate(monkeypatch, tmp_path):
    monkeypatch.delenv("KICRAFT_LAYOUT_HISTORY_DIR", raising=False)
    descriptor = HistoryDescriptor(str(tmp_path), "a" * 32, "b" * 32)
    env = {"KEEP": "yes"}
    with history_context(descriptor):
        assert HistoryDescriptor.from_env() == descriptor
        child_env = descriptor.child_env(env)
    assert env == {"KEEP": "yes"}
    assert child_env["KICRAFT_LAYOUT_HISTORY_PARENT"] == "b" * 32
    assert HistoryDescriptor.from_env() is None


def test_html_labels_are_data_not_executable(tmp_path):
    root = start_capture(tmp_path, run_id="r")
    child = root.child("leaf", '</script><script>globalThis.pwned=true</script>', {"net": "<img src=x onerror=alert(1)>"})
    child.scene([], phase="empty", geometry_source="native_board")
    child.end("complete")
    root.end("complete")
    manifest = finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=0)
    html = (manifest.parent / "timelapse.html").read_text()
    assert '</script><script>globalThis.pwned=true' not in html
    assert _embedded_data(manifest.parent / "timelapse.html")["scopes"][1]["header"]["label"].startswith("</script>")


def test_publication_resumes_after_current_pointer_write_failure(tmp_path, monkeypatch):
    from kicraft import layout_history_export as export
    root = start_capture(tmp_path, run_id="r")
    root.scene([], phase="empty", geometry_source="native_board")
    root.end("complete")
    original = export.atomic_json
    def fail_pointer(path, value):
        if path == tmp_path / "layout-history.json":
            raise OSError("pointer unavailable")
        original(path, value)
    monkeypatch.setattr(export, "atomic_json", fail_pointer)
    with pytest.warns(RuntimeWarning, match="publication failed"):
        assert finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=6) is None
    directory = Path(root.descriptor.directory)
    assert (directory / "manifest.json").is_file()
    assert root.path.is_file()
    monkeypatch.setattr(export, "atomic_json", original)
    assert finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=6)
    assert json.loads((tmp_path / "layout-history.json").read_bytes())["status"] == "complete"
    assert not root.path.exists()


def test_opt_out_replaces_stale_current_history(tmp_path, monkeypatch):
    old = start_capture(tmp_path, run_id="old")
    old.end("complete")
    finalize_capture(tmp_path, old.descriptor.capture_id, build_rc=0)
    monkeypatch.setenv("KICRAFT_LAYOUT_HISTORY", "0")
    assert start_capture(tmp_path, run_id="new") is None
    pointer = json.loads((tmp_path / "layout-history.json").read_bytes())
    assert pointer["status"] == "disabled" and pointer["run_id"] == "new"


def test_corrupt_scope_does_not_hide_independent_scope(tmp_path):
    root = start_capture(tmp_path, run_id="r")
    child = root.child("leaf", "bad")
    child.scene([_track("one")], phase="input", geometry_source="router")
    child.flush()
    with child.path.open("ab") as stream:
        stream.write(gzip.compress(b'not-json\\n'))
    root.scene([], phase="independent", geometry_source="native_board")
    root.end("complete")
    manifest = finalize_capture(tmp_path, root.descriptor.capture_id, build_rc=6)
    data = _embedded_data(manifest.parent / "timelapse.html")
    assert len(data["scopes"]) == 2 and data["status"] == "partial"
    assert any(item["last_seq"] == 1 and item["header"]["scope_id"] == child.scope_id for item in data["scopes"])


def test_native_multilayer_via_duplicate_pad_numbers_and_curved_outline(tmp_path):
    import subprocess
    from kicraft.autoplacer.hardware.adapter import _pcbnew_subprocess_env
    root = start_capture(tmp_path, run_id="native")
    script = '''
import json, math, pcbnew
from kicraft.autoplacer.hardware.history_geometry import extract_native_scene
from kicraft.layout_history import history_from_env
b=pcbnew.BOARD(); b.SetCopperLayerCount(4)
fp=pcbnew.FOOTPRINT(b); fp.SetReference("J<1>"); b.Add(fp)
for i in range(2):
 p=pcbnew.PAD(fp); p.SetNumber("1"); p.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
 p.SetShape(pcbnew.PAD_SHAPE_OVAL); p.SetSize(pcbnew.VECTOR2I(1200000,800000))
 p.SetDrillSize(pcbnew.VECTOR2I(300000,300000))
 p.SetLayerSet(pcbnew.LSET.AllCuMask()); p.SetPosition(pcbnew.VECTOR2I(i*2000000,0)); fp.Add(p)
fp.Flip(pcbnew.VECTOR2I(0,0),False)
v=pcbnew.PCB_VIA(b); v.SetPosition(pcbnew.VECTOR2I(0,0)); v.SetWidth(600000)
v.SetDrill(300000); v.SetViaType(pcbnew.VIATYPE_THROUGH); v.SetLayerPair(pcbnew.F_Cu,pcbnew.B_Cu); b.Add(v)
a=pcbnew.PCB_ARC(b); a.SetStart(pcbnew.VECTOR2I(1000000,0))
a.SetMid(pcbnew.VECTOR2I(2000000,1000000)); a.SetEnd(pcbnew.VECTOR2I(3000000,0))
a.SetWidth(200000); a.SetLayer(pcbnew.In1_Cu); b.Add(a)
edge=pcbnew.PCB_SHAPE(b); edge.SetShape(pcbnew.SHAPE_T_CIRCLE)
edge.SetStart(pcbnew.VECTOR2I(0,0)); edge.SetEnd(pcbnew.VECTOR2I(10000000,0))
edge.SetLayer(pcbnew.Edge_Cuts); b.Add(edge)
w=history_from_env("finishing","native regression")
objects=extract_native_scene(b,w,phase="native")
w.scene(objects,phase="native",geometry_source="native_board");w.end("complete")
print(json.dumps({"scope_id":w.scope_id}))
'''
    result = subprocess.run(
        ["/usr/bin/python3", "-c", script], env=root.child_env(_pcbnew_subprocess_env()),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    scope_id = json.loads(result.stdout.strip().splitlines()[-1])["scope_id"]
    recovered = recover_scope(Path(root.descriptor.directory) / f"scope-{scope_id}.jsonl.gz")
    assert not recovered.gaps
    footprint = next(obj for obj in recovered.objects.values() if obj["type"] == "footprint")
    assert footprint["pose"]["side"] == "back"
    pads = recovered.templates[footprint["template_id"]]["pads"]
    assert [pad["number"] for pad in pads] == ["1", "1"]
    assert len({pad["id"] for pad in pads}) == 2
    assert all("In1.Cu" in pad["copper_shapes"] for pad in pads)
    arc = next(obj for obj in recovered.objects.values() if obj["type"] == "arc")
    assert arc["layer"] == "In1.Cu" and "mid" in arc
    via = next(obj for obj in recovered.objects.values() if obj["type"] == "via")
    assert via["layers"] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    outline = next(obj for obj in recovered.objects.values() if obj["type"] == "outline")
    points = recovered.templates[outline["template_id"]]["contours"][0]["points"]
    import math
    for first, second in zip(points, points[1:] + points[:1]):
        radius = math.hypot((first[0]+second[0])/2, (first[1]+second[1])/2)
        assert abs(radius - 10) <= 0.01
