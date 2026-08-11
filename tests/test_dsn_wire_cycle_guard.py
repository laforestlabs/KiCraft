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
SLIVER_FIXTURE = Path(__file__).parent / "data" / "fr_sliver_loop_vout2.dsn"
MIDPOINT_FIXTURE = Path(__file__).parent / "data" / "fr_sliver_loop_vbus_midpoint.dsn"


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


def test_sliver_loop_with_float_drift_is_opened(tmp_path):
    """KC-Z879KB pattern (tests/data/fr_sliver_loop_vout2.dsn, from the real
    run): pass 1's own power routing leaves a hair-thin loop on VOUT_2. Wire
    A (151819,-97032.7)->(152479,-97692.3) starts at the MIDPOINT of wire B's
    last segment (151159,-96372.9)->(152479,-97692.5) -- a T-junction -- and
    ends 0.2 um short of B's endpoint (float drift on the pass-1 -> board ->
    pass-2 round trip), with a 0.2 um stub between them. The endpoint-only
    detector saw neither the branch nor the drift and FreeRouting 1.9 froze
    on the loop ("The normalization of net 'VOUT_2' failed.")."""
    p = tmp_path / "board.dsn"
    shutil.copy(SLIVER_FIXTURE, p)
    # The poison is the 0.2 um stub (FreeRouting collapses its endpoints
    # into a self-loop and normalization fails); it alone is dropped. The
    # remaining A/B pair is a 2-cycle FreeRouting dedupes (verified against
    # the 1.9.0 jar), so nothing else may be snipped.
    assert _break_locked_wire_cycles(str(p)) == 1
    out = p.read_text()
    assert "(wire (path F.Cu 500  152479 -97692.3  152479 -97692.5)" not in out
    assert not _wire_graph_has_cycle(out, "VOUT_2")
    # Idempotent: the opened loop stays open on a second pass.
    assert _break_locked_wire_cycles(str(p)) == 0


def test_branch_on_segment_midpoint_sliver_is_opened(tmp_path):
    """Round-2 VBUS class (KC-Z879KB replay): pass 1's own power routing left
    a 1 um stub (160019,-130543)->(160018,-130543) branching from the EXACT
    midpoint of wire A (160464,-130098)->(159574,-130988), with wire B
    (160018,-130543)->(159574,-130988) closing the triangle. FreeRouting 1.9
    splits A at the exact branch and normalization fails on the 3-edge
    sliver loop; the branch is invisible to an endpoint-only detector."""
    p = tmp_path / "board.dsn"
    shutil.copy(MIDPOINT_FIXTURE, p)
    assert _break_locked_wire_cycles(str(p)) == 1
    out = p.read_text()
    assert not _wire_graph_has_cycle(out, "VBUS")
    assert _break_locked_wire_cycles(str(p)) == 0


def test_coincident_pair_2cycle_is_not_snipped(tmp_path):
    """The plan's literal sliver shape: wire A is a polyline J->N->K and wire
    B is N->K' where K' is K + 0.2 um. The T-junction branch at A's INTERIOR
    vertex N and the sub-micron drift at K make this a 2-cycle (two parallel
    edges N->K) -- which FreeRouting 1.9 DEDUPES (verified against the jar:
    an exact or sub-µm coincident wire pair routes cleanly; only a wire whose
    endpoints collapse to zero length poisons normalization). The detector
    must see the pair but leave it alone: snipping would only re-create
    near-coincident endpoints."""
    text = """(pcb t
  (unit um)
  (wiring
    (wire (path F.Cu 500  0 0  1000 0  2000 0)(net SLIVER)(type fix))
    (wire (path F.Cu 500  1000 0  2000 0.2)(net SLIVER)(type fix))
  )
)
"""
    p = tmp_path / "b.dsn"
    p.write_text(text)
    assert _break_locked_wire_cycles(str(p)) == 0
    assert p.read_text() == text
