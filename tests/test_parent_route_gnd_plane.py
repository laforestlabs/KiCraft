"""The parent route must never route GND as a trace web.

Regression guard for KC-VKRFR7 (5x9 1515-LED array, 3 mm pitch -> rc6 "board
not routable as placed") and the KC-SMQ3HX 200-LED hang.

GND on a parent has far too many pads to route as point-to-point traces on a
dense array (it never converges -- KC-VKRFR7). `_route_parent_board` therefore
strips the leaf GND web and pours a B.Cu GND plane, then routes with the plane
PRESENT (clear_zones=False) so FreeRouting connects every GND pad to the plane
instead of weaving a trace web. Only if that route fails (a large filled plane
hangs FreeRouting 1.9.0 -- KC-SMQ3HX) does it fall back to GND-skip: GND removed
from the DSN (freerouting_skip_nets), signals routed alone, GND rebuilt after.

This guards both halves: the primary route keeps the plane (clear_zones=False),
and the fallback skips GND. It fails if someone reverts to the `ded6e24` state
(clear_zones=True with GND still a routable net), which forced FreeRouting to
route GND as traces and caused KC-VKRFR7.
"""
from __future__ import annotations

from types import SimpleNamespace

import kicraft.autoplacer.brain.gnd_pour as gnd_pour
import kicraft.autoplacer.freerouting_runner as fr
import kicraft.cli._compose_route as cr


def test_parent_route_handles_gnd_out_of_band(monkeypatch, tmp_path):
    events: list = []          # ordered log of strip / pour / route(config)

    monkeypatch.setattr(fr, "strip_net_copper",
                        lambda path, net: events.append(("strip", net)))
    monkeypatch.setattr(gnd_pour, "pour_gnd_planes", lambda *a, **k: None)
    monkeypatch.setattr(gnd_pour, "add_gnd_pour_and_thermal_vias",
                        lambda *a, **k: events.append(("pour_plane",)))

    def _fake_route(*, kicad_pcb_path, output_path, jar_path, config):
        events.append(("route", dict(config)))
        # Fail both the plane attempt and the skip fallback so we capture the
        # config of each; _route_parent_board wraps routing in try/except and
        # returns a discardable failed dict after the fallback also raises.
        raise RuntimeError("stop after capturing route config")

    monkeypatch.setattr(fr, "route_with_freerouting", _fake_route)

    stamped = tmp_path / "parent_pre_freerouting.kicad_pcb"
    stamped.write_text("(kicad_pcb)\n", encoding="utf-8")
    state = SimpleNamespace(
        composition=SimpleNamespace(inferred_interconnect_nets={"NET_A": 1, "NET_B": 1}),
        component_count=49,
    )
    cfg = {
        "freerouting_jar": "unused-stub.jar",
        "gnd_zone_net": "GND",
        "shield_tie_enabled": False,
    }

    cr._route_parent_board(stamped, state, tmp_path, cfg)

    route_idxs = [i for i, e in enumerate(events) if e[0] == "route"]
    routes = [events[i][1] for i in route_idxs]
    assert routes, "FreeRouting was never invoked"
    # GND copper stripped and a GND plane poured before the first route attempt.
    assert ("strip", "GND") in events and ("pour_plane",) in events
    assert events.index(("pour_plane",)) < route_idxs[0]

    # Primary attempt keeps the plane in the DSN -- this is the anti-regression
    # assertion: ded6e24 set clear_zones=True with GND routable, forcing the
    # GND-trace blow-up (KC-VKRFR7).
    assert routes[0].get("freerouting_clear_zones") is False, (
        "primary parent route cleared zones -- that deletes the GND plane and "
        "forces FreeRouting to route GND as a trace web across the array "
        "(KC-VKRFR7 rc6). Route with the plane present (clear_zones=False)."
    )
    assert "GND" not in (routes[0].get("freerouting_skip_nets") or [])

    # Fallback (second attempt) skips GND entirely so a large board still routes.
    assert len(routes) >= 2, "expected a GND-skip fallback after the plane route failed"
    assert "GND" in (routes[1].get("freerouting_skip_nets") or []), (
        "GND-skip fallback must remove GND from the DSN so a board whose filled "
        "plane hangs FreeRouting (KC-SMQ3HX) still routes its signals"
    )


# --- _strip_nets_from_dsn: the DSN edit that excludes a net from the router ---

_DSN = """(pcb board.dsn
  (structure
    (layer F.Cu (type signal))
    (layer B.Cu (type signal))
  )
  (placement
    (component LED:led (place LED1 1000 1000 front 0))
  )
  (network
    (net +5V
      (pins LED1-2 J1-1)
    )
    (net GND
      (pins LED1-4 LED2-4 J1-2)
    )
    (net DATA
      (pins LED1-1 J1-3)
    )
    (class kicad_default +5V DATA GND
      (circuit
        (use_via "Via[0-1]_600:300_um")
      )
      (rule
        (width 200)
        (clearance 200)
      )
    )
  )
  (wiring
    (wire (path B.Cu 250 1000 1000 1100 1000) (net GND) (type protect))
    (wire (path F.Cu 200 1000 1000 1050 1000) (net DATA) (type protect))
    (via "Via[0-1]_600:300_um" 1100 1000 GND)
  )
)
"""


def test_strip_nets_from_dsn_removes_net_everywhere(tmp_path):
    p = tmp_path / "b.dsn"
    p.write_text(_DSN, encoding="utf-8")
    fr._strip_nets_from_dsn(str(p), ["GND"])
    out = p.read_text()

    # GND gone from network, class, and wiring; structurally still balanced.
    assert out.count("(") == out.count(")")
    assert "(net GND" not in out
    assert "(net DATA" in out and "(net +5V" in out          # other nets kept
    cls = out[out.find("(class"):out.find("(circuit")]
    assert "GND" not in cls and "+5V" in cls and "DATA" in cls
    assert "(net GND)" not in out                            # GND wire dropped
    assert "(net DATA)" in out                               # DATA wire kept
    assert "1100 1000 GND" not in out                        # GND via dropped


def test_strip_nets_from_dsn_noop_without_skip(tmp_path):
    p = tmp_path / "b.dsn"
    p.write_text(_DSN, encoding="utf-8")
    fr._strip_nets_from_dsn(str(p), None)
    assert p.read_text() == _DSN                              # untouched


# --- _restrict_dsn_routing_to_nets: the power-first phase-1 DSN edit ---


def test_restrict_dsn_routing_empties_other_nets_pins_only(tmp_path):
    p = tmp_path / "b.dsn"
    p.write_text(_DSN, encoding="utf-8")
    fr._restrict_dsn_routing_to_nets(str(p), ["+5V"])
    out = p.read_text()

    assert out.count("(") == out.count(")")
    # The routed net keeps its pins; every other net stays DECLARED (its
    # wiring must remain resolvable) but has nothing left to connect.
    assert "(pins LED1-2 J1-1)" in out
    assert "(net GND" in out and "(net DATA" in out
    assert "LED1-4" not in out and "J1-3" not in out
    # ALL wiring kept -- this is the difference from _strip_nets_from_dsn:
    # phase 1 routes power on a board carrying every leaf's locked signal
    # copper, and stripping that wiring would let the router short through it.
    assert "(net GND)" in out and "(net DATA)" in out
    assert "1100 1000 GND" in out
    # Class membership (rules) intact.
    cls = out[out.find("(class"):out.find("(circuit")]
    assert "+5V" in cls and "GND" in cls and "DATA" in cls


def test_restrict_dsn_routing_noop_without_list(tmp_path):
    p = tmp_path / "b.dsn"
    p.write_text(_DSN, encoding="utf-8")
    fr._restrict_dsn_routing_to_nets(str(p), None)
    assert p.read_text() == _DSN                              # untouched
