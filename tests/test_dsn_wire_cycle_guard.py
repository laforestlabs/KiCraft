"""_break_locked_wire_cycles: the FreeRouting 1.9 net-normalization hang guard
(2026-07-27 fix-plan P3.1, docs/plans/self-eval-2026-07-27-fix-plan.md).

Self-eval 2026-07-27 run_29 (round-led-ring): the LED-ring leaf legitimately
routed its 5V bus as a CLOSED RING; the parent locked that copper into the
DSN and FreeRouting 1.9.0 hung forever on every attempt ("The normalization
of net '5V' failed." then silence until the rc=-1 watchdog kill), killing the
board at rc6. Bisection proved one loop-closing wire was the poison; opening
the loop with a microscopic DSN-only gap routes the same board in ~18s.

``tests/data/fr_hang_5v_loop.dsn`` is the actual poisoned parent DSN from
that run (verbatim), so the guard is pinned against the real geometry.
"""
from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

from kicraft.autoplacer.freerouting_runner import (
    _DSN_WIRE_ENTRY_RE,
    _break_locked_wire_cycles,
)

FIXTURE = Path(__file__).parent / "data" / "fr_hang_5v_loop.dsn"


def _wire_graph_has_cycle(text: str, net: str) -> bool:
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in _DSN_WIRE_ENTRY_RE.finditer(text):
        if m.group("net").strip('"') != net:
            continue
        vals = [float(t) for t in m.group("coords").split()]
        xy = list(zip(vals[::2], vals[1::2]))
        if all(p == xy[0] for p in xy):
            continue
        a = (round(xy[0][0], 1), round(xy[0][1], 1))
        b = (round(xy[-1][0], 1), round(xy[-1][1], 1))
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


SYNTH = """(pcb synth
  (wiring
    (wire (path F.Cu 500  0 0  1000 0)(net LOOP)(type fix))
    (wire (path F.Cu 500  1000 0  1000 1000)(net LOOP)(type fix))
    (wire (path F.Cu 500  1000 1000  0 0)(net LOOP)(type fix))
    (wire (path F.Cu 500  5000 5000  5000 5000)(net DEGEN)(type fix))
    (wire (path F.Cu 500  9000 0  9000 500)(net OK)(type fix))
  )
)
"""


def test_synthetic_loop_is_opened_and_degenerate_dropped(tmp_path):
    p = tmp_path / "b.dsn"
    p.write_text(SYNTH)
    n = _break_locked_wire_cycles(str(p))
    assert n == 2  # one loop-closer gapped + one zero-length wire dropped
    out = p.read_text()
    assert not _wire_graph_has_cycle(out, "LOOP")
    assert "DEGEN" not in out
    # The healthy wire is untouched, and the LOOP net still has 3 wires
    # (opened, not deleted -- the copper must stay an obstacle).
    assert "(wire (path F.Cu 500  9000 0  9000 500)(net OK)(type fix))" in out
    assert len(re.findall(r"\(net LOOP\)", out)) == 3


def test_gap_is_microscopic_and_never_disconnects(tmp_path):
    p = tmp_path / "b.dsn"
    p.write_text(SYNTH)
    _break_locked_wire_cycles(str(p))
    out = p.read_text()
    # Total LOOP wire length shrinks by at most width/10-ish, not by a segment.
    def total_len(text):
        L = 0.0
        for m in _DSN_WIRE_ENTRY_RE.finditer(text):
            if m.group("net") != "LOOP":
                continue
            vals = [float(t) for t in m.group("coords").split()]
            xy = list(zip(vals[::2], vals[1::2]))
            L += sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(xy, xy[1:]))
        return L
    before, after = total_len(SYNTH), total_len(out)
    # width/10 gap, plus a whisker for the 0.1-unit coordinate quantization.
    assert 0 < before - after <= 500 / 10 + 1.0


def test_real_run29_fixture_sanitizes_and_is_idempotent(tmp_path):
    p = tmp_path / "board.dsn"
    shutil.copy(FIXTURE, p)
    assert _wire_graph_has_cycle(p.read_text(), "5V")
    n = _break_locked_wire_cycles(str(p))
    # The run_29 board: one genuine 5V loop-closer + one zero-length 5V wire.
    assert n == 2
    assert not _wire_graph_has_cycle(p.read_text(), "5V")
    # Idempotent: a second pass finds nothing left to open.
    assert _break_locked_wire_cycles(str(p)) == 0


def test_acyclic_dsn_untouched(tmp_path):
    text = """(pcb ok
  (wiring
    (wire (path F.Cu 500  0 0  1000 0)(net A)(type fix))
    (wire (path B.Cu 500  0 0  1000 0)(net B)(type fix))
  )
)
"""
    p = tmp_path / "b.dsn"
    p.write_text(text)
    assert _break_locked_wire_cycles(str(p)) == 0
    assert p.read_text() == text
