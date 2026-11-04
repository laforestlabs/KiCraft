import subprocess
import sys

import pytest


def test_pad_outside_triangle_but_inside_bounding_rectangle():
    script = '''
import json
try:
    import pcbnew as k
except ImportError:
    raise SystemExit(77)
from kicraft.scoring.placement_check import PlacementCheck
v = lambda x, y: k.VECTOR2I(k.FromMM(x), k.FromMM(y))
results = []
for points in [[(0, 0), (10, 0), (10, 10), (0, 10)], [(0, 0), (10, 0), (0, 10)]]:
    board = k.BOARD()
    for a, b in zip(points, points[1:] + points[:1]):
        edge = k.PCB_SHAPE()
        edge.SetShape(k.SHAPE_T_SEGMENT)
        edge.SetLayer(k.Edge_Cuts)
        edge.SetStart(v(*a))
        edge.SetEnd(v(*b))
        board.Add(edge)
    footprint = k.FOOTPRINT(board)
    footprint.SetReference("R1")
    footprint.SetPosition(v(8, 8))
    pad = k.PAD(footprint)
    pad.SetNumber("1")
    pad.SetSize(v(1, 1))
    pad.SetPosition(v(8, 8))
    footprint.Add(pad)
    board.Add(footprint)
    results.append(PlacementCheck().run(board, {}).metrics["pads_off_board"])
print(json.dumps(results))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("pcbnew is unavailable")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[0, 1]"
