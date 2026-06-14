"""Route a stamped parent board via FreeRouting + post-route validation.

Split out of ``compose_subcircuits.py`` (Lever 2.5); re-exported there.
"""
from __future__ import annotations

import json
import sys



def _route_parent_board(
    stamped_pcb: Path,
    state: ParentCompositionState,
    project_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Route parent interconnects via FreeRouting, then import and validate.

    1. Resolve output path for the routed board
    2. Run FreeRouting (preserving stamped child copper in the DSN export)
    3. Import routed copper from the result
    4. Validate the routed board
    5. Return a result dict
    """
    from kicraft.autoplacer.freerouting_runner import (
        import_routed_copper,
        route_with_freerouting,
        validate_routed_board,
    )

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    routed_pcb = stamped_pcb.parent / "parent_routed.kicad_pcb"

    jar_path = cfg.get("freerouting_jar", "")
    if not jar_path:
        raise RuntimeError(
            "No FreeRouting JAR path configured; pass --jar or set "
            "freerouting_jar in project config"
        )

    # Build a routing config that preserves child copper already stamped
    # onto the board.  FreeRouting's DSN export will see those traces as
    # wires so it only routes the remaining unconnected (interconnect) nets.
    route_cfg = dict(cfg)
    route_cfg["freerouting_preserve_existing_copper"] = True
    route_cfg["freerouting_clear_existing_copper"] = False
    route_cfg["freerouting_clear_zones"] = False

    # Ground handling: route signals first, pour ground last (standard practice,
    # and what the leaves already do). The stamped parent carries each leaf's GND
    # as a web of F.Cu traces; that saturates the signal layer so FreeRouting
    # cannot complete a cross-block signal interconnect (a lone MCU pin to a
    # sensor net stays unrouted while the dense GND web blocks every path). So
    # strip the GND copper and pour a B.Cu GND plane up front: GND pads then
    # connect via the plane, FreeRouting skips GND, and F.Cu is clear for the
    # signal interconnects. The plane is refilled after routing to close around
    # the new traces and tie in every GND pad.
    from kicraft.autoplacer.brain.gnd_pour import (
        add_gnd_pour_and_thermal_vias,
        pour_gnd_planes,
    )
    from kicraft.autoplacer.freerouting_runner import strip_net_copper

    gnd_net = cfg.get("gnd_zone_net", "GND")
    # The pre-route GND strip/pour and the route itself all shell out to pcbnew;
    # any of them can fail (e.g. a pcbnew SIGSEGV mid-strip). Guard the whole
    # block so a failure returns a discardable result and the search tries the
    # next round, instead of an uncaught exception killing the compose subprocess
    # and taking the entire build down with it.
    try:
        if gnd_net:
            # Strip the leaf-composed GND web and pour a B.Cu GND plane (+ IC
            # thermal vias) before routing, so FreeRouting ties GND to the plane
            # with short drops instead of re-creating the dense cross-block GND
            # web that saturates F.Cu and blocks signal interconnects.
            strip_net_copper(str(stamped_pcb), gnd_net)
            add_gnd_pour_and_thermal_vias(str(stamped_pcb), cfg)
            # The strip just deleted the leaves' locked shield ties along with
            # the GND web -- and the plane cannot replace them: a connector's
            # through-hole shield legs sit where the B.Cu fill loses its
            # thermal spokes to the slot holes, so without the ties they come
            # back as the 8/8 'unconnected GND at J1' rc7 signature. Re-stamp
            # them here, pre-route; the stamper's pad/track guards drop any
            # tie that would cross the composed leaf copper.
            if cfg.get("shield_tie_enabled", True):
                try:
                    from kicraft.autoplacer.freerouting_runner import (
                        _run_pcbnew_script,
                    )

                    _tie_cfg = json.dumps({
                        k: cfg[k]
                        for k in (
                            "shield_tie_enabled",
                            "shield_tie_exclude_refs",
                            "shield_tie_max_mm",
                            "freerouting_min_clearance_mm",
                            "freerouting_fine_pitch_track_mm",
                            "gnd_zone_net",
                        )
                        if k in cfg
                    })
                    _run_pcbnew_script(
                        "import pcbnew, json\n"
                        "from kicraft.autoplacer.brain.breakout_stubs import (\n"
                        "    add_breakout_stubs, shield_tie_specs)\n"
                        f"cfg = json.loads({_tie_cfg!r})\n"
                        f"board = pcbnew.LoadBoard({str(stamped_pcb)!r})\n"
                        "specs = shield_tie_specs(board, cfg)\n"
                        "del board\n"
                        f"s = add_breakout_stubs({str(stamped_pcb)!r}, specs, cfg=cfg)\n"
                        "print('parent shield ties:', s['stubs'], 'stamped,',\n"
                        "      len(s['skipped']), 'skipped')\n"
                    )
                except Exception as exc:
                    print(f"warning: parent shield re-tie failed: {exc}",
                          file=sys.stderr)
        freerouting_stats = route_with_freerouting(
            kicad_pcb_path=str(stamped_pcb),
            output_path=str(routed_pcb),
            jar_path=jar_path,
            config=route_cfg,
        )
    except Exception as exc:
        return {
            "failed": True,
            "error": str(exc),
            "routed_board_path": str(routed_pcb),
            "_trace_segments": [],
            "_via_objects": [],
            "validation": {},
            "freerouting_stats": {},
        }

    # Pour GND on BOTH layers, closing around the freshly-routed interconnects.
    # The F.Cu pour is what ties in every F.Cu GND pad via thermal relief on its
    # own layer (a B.Cu-only plane can't reach an F.Cu SMD pad without a via),
    # tied down to the B.Cu plane through the thermal vias placed pre-route.
    if gnd_net:
        pour_gnd_planes(str(routed_pcb), cfg, layers=("B.Cu", "F.Cu"))

    # Pour the primary power rail (e.g. VBUS) as an F.Cu plane, post-route only:
    # it lives on the signal layer, so pouring it before routing would saturate
    # F.Cu. At higher priority than GND it wins its region; GND fills the rest.
    # This ties paired connector power pads the autorouter can't thread together.
    if cfg.get("power_plane_enabled", True):
        from kicraft.autoplacer.brain.gnd_pour import pour_power_planes

        pour_power_planes(
            str(routed_pcb), cfg, layers=(cfg.get("power_plane_layer", "F.Cu"),)
        )

    # GND strand repair: a THT connector GND pin can end up on a tiny plane
    # fragment with no path to the main pour (FreeRouting never routes GND,
    # a 2-pad connector has no shield-tie mate, and no via drops through).
    # Find every GND cluster isolated from the main plane and tie it back
    # with a guarded same-net track, then refill. pcbnew work -> subprocess.
    if gnd_net and cfg.get("gnd_strand_repair_enabled", True):
        try:
            from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script

            _rep_cfg = json.dumps({
                k: cfg[k]
                for k in (
                    "gnd_zone_net",
                    "gnd_strand_repair_enabled",
                    "gnd_strand_repair_max_mm",
                    "freerouting_min_clearance_mm",
                    "freerouting_fine_pitch_track_mm",
                )
                if k in cfg
            })
            _run_pcbnew_script(
                "import json\n"
                "from kicraft.autoplacer.brain.gnd_pour import repair_stranded_gnd\n"
                f"cfg = json.loads({_rep_cfg!r})\n"
                f"s = repair_stranded_gnd({str(routed_pcb)!r}, cfg)\n"
                "print('gnd strand repair:', s['stranded'], 'stranded,',\n"
                "      s['tied'], 'tied,', len(s['skipped']), 'skipped')\n"
            )
        except Exception as exc:
            print(f"warning: gnd strand repair failed: {exc}", file=sys.stderr)

    # Power strand repair: the power-rail pour fragments exactly like the GND
    # plane (KC-Z57JEZ: +3V3 split into two F.Cu islands around a fine-pitch
    # LGA, stranding the part's supply pads) -- tie each stranded power
    # cluster back the same way. pcbnew work -> subprocess.
    if cfg.get("power_plane_enabled", True) and cfg.get(
            "power_strand_repair_enabled", True):
        try:
            from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script

            _pwr_cfg = json.dumps({
                k: cfg[k]
                for k in (
                    "power_strand_repair_enabled",
                    "power_plane_enabled",
                    "power_plane_nets",
                    "power_plane_max_nets",
                    "gnd_zone_net",
                    "gnd_strand_repair_max_mm",
                    "freerouting_min_clearance_mm",
                    "freerouting_fine_pitch_track_mm",
                )
                if k in cfg
            })
            _run_pcbnew_script(
                "import json\n"
                "from kicraft.autoplacer.brain.gnd_pour import repair_stranded_power\n"
                f"cfg = json.loads({_pwr_cfg!r})\n"
                f"s = repair_stranded_power({str(routed_pcb)!r}, None, cfg)\n"
                "print('power strand repair:', s['nets'], '--', s['stranded'],\n"
                "      'stranded,', s['tied'], 'tied,', len(s['skipped']), 'skipped')\n"
            )
        except Exception as exc:
            print(f"warning: power strand repair failed: {exc}", file=sys.stderr)

    # Import all copper from the routed board (child + new parent traces)
    copper = import_routed_copper(str(routed_pcb))

    # Root parent has no interface anchors -- skip anchor validation.
    # Anchor completeness is a leaf-level gate, not a parent-level gate.
    validation = validate_routed_board(
        str(routed_pcb),
        cfg=cfg,
        expected_anchor_names=[],
        actual_anchor_names=[],
        required_anchor_names=[],
    )

    # A parent must close every net. Unlike a leaf -- whose interface ports are
    # legitimately open, so validate_routed_board waives unconnected -- unrouted
    # nets on the parent mean an unusable board (the final build verify requires
    # 0 unconnected). Reject here so the search keeps trying other rounds for a
    # fully-routed parent instead of promoting one the verify gate would fail.
    unconnected = int((validation.get("drc") or {}).get("unconnected", 0) or 0)
    if unconnected > 0:
        validation["accepted"] = False
        reasons = validation.setdefault("rejection_reasons", [])
        if "unconnected_nets" not in reasons:
            reasons.append("unconnected_nets")

    return {
        "failed": False,
        "routed_board_path": str(routed_pcb),
        "_trace_segments": copper.get("traces", []),
        "_via_objects": copper.get("vias", []),
        "validation": validation,
        "freerouting_stats": freerouting_stats,
    }
