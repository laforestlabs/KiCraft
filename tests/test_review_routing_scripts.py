import json
import sys
from types import SimpleNamespace

from scripts import soft_tidiness_ab






def test_latest_round_is_selected_numerically(monkeypatch, tmp_path):
    sub = tmp_path / ".experiments" / "subcircuits" / "leaf"
    sub.mkdir(parents=True)
    for n in (9, 10):
        (sub / f"round_{n}_solved_layout.json").write_text(json.dumps({"sheet_name": "leaf", "round": n}))
    monkeypatch.setattr(soft_tidiness_ab, "_components", lambda layout: {"round": layout["round"]})
    assert soft_tidiness_ab._leaf_layouts(str(tmp_path))["leaf"] == {"round": 10}




def test_soft_solve_does_not_report_failed_or_missing_drc_as_clean(monkeypatch):
    def solve_with(report, rc=0):
        stdout = soft_tidiness_ab.START + json.dumps({"report": report}) + soft_tidiness_ab.END
        monkeypatch.setattr(soft_tidiness_ab.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=rc, stdout=stdout))
        return soft_tidiness_ab._solve("/tmp/project", "board", 0)

    assert solve_with("") == -1
    assert solve_with("Drc report for leaf_routed") == -1
    assert solve_with("Drc report for leaf_routed\\nFound 0 unconnected pads", rc=1) == -1
    assert solve_with("Drc report for leaf_routed\\nFound 2 unconnected pads") == 2


def test_pixel_fraction_keeps_threshold_precision(tmp_path):
    from PIL import Image
    from tools.render_diff import pixel_diff

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (10000, 2), "black").save(a)
    image = Image.new("RGB", (10000, 2), "black")
    image.putpixel((0, 0), (255, 255, 255))
    image.save(b)
    stats = pixel_diff(a, b, tmp_path / "diff.png", threshold=0)
    assert stats["fraction_differing"] == 0.00005


def test_edge_warning_uses_inward_clearance_without_rounding(monkeypatch, tmp_path, capsys):
    import importlib

    class Box:
        def __init__(self, gap, _size=None):
            self.gap = gap

        def GetOrigin(self):
            return self.gap

        def GetSize(self):
            return None

        def Merge(self, other):
            pass

    class Footprint:
        def __init__(self, ref, gap):
            self.ref, self.box = ref, Box(gap)

        def GetReference(self):
            return self.ref

    footprints = [Footprint("SAFE", -1.0), Footprint("NEAR", -0.19999)]
    pcbnew = SimpleNamespace(BOX2I=Box, LoadBoard=lambda path: SimpleNamespace(GetFootprints=lambda: footprints))
    monkeypatch.setitem(sys.modules, "pcbnew", pcbnew)
    edge = importlib.import_module("scripts.measure_edge_legalization")
    monkeypatch.setattr(edge, "pcbnew", pcbnew)
    monkeypatch.setattr(edge, "CORPUS", str(tmp_path))
    stem = tmp_path / "run_1" / "generated" / "design"
    stem.mkdir(parents=True)
    monkeypatch.setattr(edge, "_find_board_and_zones", lambda path: ("board", {
        "SAFE": {"edge": "left"}, "NEAR": {"edge": "left"},
    }))
    monkeypatch.setattr(edge, "_edge_centerline", lambda board: None)
    monkeypatch.setattr(edge, "_pad_bbox", lambda fp: fp.box)
    monkeypatch.setattr(edge, "_court_bbox", lambda fp: fp.box)
    monkeypatch.setattr(edge, "_gap", lambda side, boundary, box: box.gap)
    edge.main()
    rows = capsys.readouterr().out.splitlines()
    assert "<--" not in next(row for row in rows if "SAFE" in row)
    assert "<--" in next(row for row in rows if "NEAR" in row)
