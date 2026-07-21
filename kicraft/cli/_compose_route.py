"""Route a stamped parent board via FreeRouting + post-route validation.

Split out of ``compose_subcircuits.py`` (Lever 2.5); re-exported there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kicraft.cli._compose_state import ParentCompositionState


def _scale_parent_route_budget(
    n_interconnect: int, n_components: int, cfg: dict
) -> tuple[int, int]:
    """Scale the PARENT FreeRouting budget to board complexity.

    Two independent scalers, both ONLY raise (never lower a hand-tuned config):

    1. Component-count timeout (the dominant cost on large parents): FreeRouting
       loads and processes the ENTIRE board -- every component, every fixed wire
       from the leaf routing -- even when only a handful of interconnect nets
       remain unrouted. A 200-LED parent needs ~170 s (72 s routing + 94 s
       optimization) though it has only 3 unrouted nets; the fixed 60 s default
       kills it mid-route. The per-component floor mirrors the leaf budget
       (leaf_freerouting_s_per_component) at a lower rate because the parent
       starts from pre-routed leaf copper, not scratch.

    2. Dense-interconnect passes/time (C2): a parent with many inter-leaf nets
       needs more passes to converge every cross-leaf net.

    Returns (max_passes, timeout_s)."""
    base_passes = int(cfg.get("freerouting_max_passes", 20))
    base_timeout = int(cfg.get("freerouting_timeout_s", 60))

    # Component-count timeout floor: FreeRouting's wall-time is dominated by
    # board size (it processes all fixed wires + optimizes all traces), not by
    # the count of unrouted nets. Scale the timeout so a dense parent (200-LED
    # matrix, 50-component MCU board) gets enough time to finish.
    per_comp = float(cfg.get("parent_freerouting_s_per_component", 1.0))
    cap = int(cfg.get("parent_freerouting_timeout_cap_s", 600))
    comp_timeout = min(cap, int(n_components * per_comp)) if n_components > 0 else 0
    timeout = max(base_timeout, comp_timeout)

    # Dense-interconnect passes/time (C2).
    threshold = int(cfg.get("parent_dense_interconnect_threshold", 10))
    if n_interconnect >= threshold:
        timeout = max(timeout, int(cfg.get("parent_dense_timeout_s", 180)))
        base_passes = max(base_passes, int(cfg.get("parent_dense_max_passes", 40)))

    return base_passes, timeout


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
    # Clear any stray copper zone before DSN export: a filled zone makes
    # FreeRouting 1.9.0 hang (it never completes a pass on a board with a large
    # filled plane -- the KC-SMQ3HX 200-LED hang). GND is stripped + skipped
    # below and re-poured after routing, so there is normally no zone left to
    # clear; this is defensive against a leaf zone surviving the strip.
    route_cfg["freerouting_clear_zones"] = True

    # Scale the parent routing budget to board complexity. FreeRouting
    # processes the ENTIRE board (every component, every fixed leaf wire)
    # even when only a few interconnect nets remain unrouted, so wall-time
    # scales with component count, not interconnect count. A 200-LED parent
    # needs ~170 s though it has only 3 unrouted nets; the fixed 60 s default
    # kills it mid-route.
    n_interconnect = len(getattr(composition, "inferred_interconnect_nets", {}) or {})
    n_components = getattr(state, "component_count", 0) or len(
        getattr(composition, "board_state", None) and composition.board_state.components or {}
    )
    mp, to = _scale_parent_route_budget(n_interconnect, n_components, route_cfg)
    if mp != route_cfg.get("freerouting_max_passes") or to != route_cfg.get("freerouting_timeout_s"):
        print(f"  parent route: {n_components} components, {n_interconnect} inter-leaf nets -> "
              f"{mp} passes / {to}s budget")
    route_cfg["freerouting_max_passes"] = mp
    route_cfg["freerouting_timeout_s"] = to

    # Ground handling is adaptive, because FreeRouting 1.9.0 has two opposite
    # failure modes on a parent and no single GND strategy beats both:
    #   * A filled GND plane in the DSN HANGS it on a large board (zero output
    #     for the whole timeout -- the 200-LED KC-SMQ3HX hang).
    #   * Skipping GND entirely leaves dense-array pads stranded -- a tiny 1515
    #     pad can't host a via and FreeRouting, not caring about GND, boxes it in
    #     with signal traces (the KC-VKRFR7 1-unconnected-pad tail).
    # So: strip the leaf GND web (its F.Cu traces saturate the signal layer) and
    # pour a B.Cu GND plane + thermal vias up front, then route with the plane
    # PRESENT (clear_zones=False) -- FreeRouting actively keeps every GND pad tied
    # to the plane while it routes signals, which routes a dense array cleanly
    # (0 unconnected). Only if that route fails to produce a board (the plane hung
    # FreeRouting) do we fall back to GND-skip: strip GND, remove it from the DSN,
    # route signals alone, and rebuild GND as a plane after. GND is always poured
    # on both layers post-route to close around the new traces.
    from kicraft.autoplacer.brain.gnd_pour import (
        add_gnd_pour_and_thermal_vias,
        pour_gnd_planes,
    )
    from kicraft.autoplacer.freerouting_runner import strip_net_copper

    gnd_net = cfg.get("gnd_zone_net", "GND")

    def _stamp_shield_ties(pcb_path: str) -> None:
        # Re-stamp connector shield ties: a THT shield leg sits where the B.Cu
        # fill loses its thermal spokes to the slot holes, so without an explicit
        # tie it returns as the 8/8 'unconnected GND at J1' rc7 signature. The
        # stamper's pad/track guards drop any tie that would cross routed copper;
        # it is a no-op on a board with no connector (e.g. an LED array).
        if not cfg.get("shield_tie_enabled", True):
            return
        try:
            from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script

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
                f"board = pcbnew.LoadBoard({pcb_path!r})\n"
                "specs = shield_tie_specs(board, cfg)\n"
                "del board\n"
                f"s = add_breakout_stubs({pcb_path!r}, specs, cfg=cfg)\n"
                "print('parent shield ties:', s['stubs'], 'stamped,',\n"
                "      len(s['skipped']), 'skipped')\n"
            )
        except Exception as exc:
            print(f"warning: parent shield re-tie failed: {exc}", file=sys.stderr)

    # The GND prep and the route shell out to pcbnew/FreeRouting; any can fail
    # (a pcbnew SIGSEGV mid-strip, a routing hang). Guard the whole block so a
    # failure returns a discardable result and the search tries the next round,
    # instead of crashing the compose subprocess and taking the build down.
    used_gnd_skip = False
    power_first_stats: dict[str, Any] | None = None
    try:
        if gnd_net:
            strip_net_copper(str(stamped_pcb), gnd_net)
            add_gnd_pour_and_thermal_vias(str(stamped_pcb), cfg)
            _stamp_shield_ties(str(stamped_pcb))
        # Power-first phase 1: freerouting 1.9.0 has no net priority -- each
        # pass collects incomplete items in board item-list order, so the wide
        # power nets (fattest corridor needed) are structurally last-in-practice
        # and end up walled off by earlier thin-net copper (KC-ZRAUR7: VBUS
        # split across two islands 18 mm apart on a 55%-empty board). Route the
        # power-class nets ALONE first -- every other net's pins are emptied in
        # the DSN while its pads and locked wiring stay obstacles -- then let
        # the main route below run on the result with the power copper locked,
        # exactly like leaf copper. Any phase-1 failure falls through to
        # today's single-phase behavior: this step may improve a board, never
        # fail one.
        power_nets = [
            n for n in (cfg.get("power_nets") or []) if n and n != gnd_net
        ]
        if cfg.get("parent_power_first", True) and power_nets:
            import shutil as _shutil

            p1_cfg = dict(route_cfg)
            p1_cfg["freerouting_clear_zones"] = False
            p1_cfg["freerouting_route_only_nets"] = power_nets
            p1_cfg["freerouting_timeout_s"] = min(
                int(route_cfg.get("freerouting_timeout_s", 60)),
                int(cfg.get("parent_power_first_timeout_s", 120)),
            )
            power_routed = stamped_pcb.parent / "parent_power_routed.kicad_pcb"
            try:
                power_first_stats = route_with_freerouting(
                    kicad_pcb_path=str(stamped_pcb),
                    output_path=str(power_routed),
                    jar_path=jar_path,
                    config=p1_cfg,
                )
                # Adopt the power-routed board as the main route's input; the
                # phase-2 DSN export locks its copper like leaf copper.
                _shutil.copy2(power_routed, stamped_pcb)
                power_first_stats["nets"] = power_nets
                print(f"  parent route: power-first phase routed "
                      f"{', '.join(power_nets)} first")
            except Exception as exc:
                power_first_stats = {"failed": str(exc), "nets": power_nets}
                print(f"  parent route: power-first phase failed ({exc}); "
                      f"continuing single-phase", file=sys.stderr)
        # Attempt 1: route with the GND plane present (clear_zones=False). Cap the
        # timeout so a hang (the large-plane failure mode) is detected promptly
        # and we fall back, rather than burning the full component-scaled budget.
        route_cfg["freerouting_clear_zones"] = False
        probe_cfg = dict(route_cfg)
        probe_cfg["freerouting_timeout_s"] = min(
            int(route_cfg.get("freerouting_timeout_s", 60)),
            int(cfg.get("parent_gnd_plane_probe_timeout_s", 120)),
        )
        try:
            freerouting_stats = route_with_freerouting(
                kicad_pcb_path=str(stamped_pcb),
                output_path=str(routed_pcb),
                jar_path=jar_path,
                config=probe_cfg,
            )
        except Exception as exc:
            # The filled GND plane hung FreeRouting (large board). Fall back to
            # GND-skip: strip every scrap of GND copper -- including the plane +
            # vias we just poured; a stray one makes FreeRouting warn 'net not
            # found' and could be crossed by a signal -- remove GND from the DSN,
            # and rebuild it after routing.
            print(f"  parent route: GND-plane route failed ({exc}); "
                  f"retrying with GND skipped", file=sys.stderr)
            if gnd_net:
                strip_net_copper(str(stamped_pcb), gnd_net)
                route_cfg["freerouting_clear_zones"] = True
                route_cfg["freerouting_skip_nets"] = list(dict.fromkeys(
                    [*route_cfg.get("freerouting_skip_nets", []), gnd_net]))
            used_gnd_skip = True
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
            "freerouting_stats": (
                {"power_first": power_first_stats} if power_first_stats else {}
            ),
        }
    if power_first_stats is not None:
        freerouting_stats = dict(freerouting_stats or {})
        freerouting_stats["power_first"] = power_first_stats

    # GND, post-route. On the skip fallback FreeRouting routed nothing for GND, so
    # rebuild it around the freshly-routed signals: pour the B.Cu plane +
    # collision-guarded thermal vias (runs on the ROUTED board, so _via_blocked
    # drops any via that would cross another net) and re-stamp the shield ties
    # stripped with the GND copper. The plane-route path already carries those.
    if gnd_net:
        if used_gnd_skip:
            try:
                add_gnd_pour_and_thermal_vias(str(routed_pcb), cfg)
            except Exception as exc:
                print(f"warning: post-route GND stitch failed: {exc}",
                      file=sys.stderr)
            _stamp_shield_ties(str(routed_pcb))
        # Pour GND on BOTH layers, closing around the routed interconnects. The
        # F.Cu pour ties F.Cu GND pads on their own layer; the thermal vias bond
        # the two planes (a B.Cu-only plane can't reach an F.Cu SMD pad).
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

    # GND island repair (convergence loop): the single-pass track-based
    # repair (repair_stranded_gnd) ties stranded fill islands with same-net
    # tracks, but the parent re-pour after power-plane addition refills ALL
    # zones (including GND) at higher priority, which can re-fragment the
    # GND plane -- leaving islands that a second pass would tie. The
    # convergence loop also tries via-stitching for cross-layer overlapping
    # fill islands (B.Cu <-> F.Cu), which the track-only pass cannot bridge.
    # pcbnew work -> subprocess. Each repair persists its summary through a
    # sidecar the parent re-prints and stores: the subprocess's own stdout is
    # swallowed by _run_pcbnew_script, which left the 1/655 investigation
    # unable to tell whether these passes ran at all (KC-ZRAUR7 workstream B).
    post_route_repairs: dict[str, Any] = {}
    if gnd_net and cfg.get("gnd_strand_repair_enabled", True):
        sidecar = Path(str(routed_pcb) + ".gnd_island_repair.json")
        try:
            from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script

            _rep_cfg = json.dumps({
                k: cfg[k]
                for k in (
                    "gnd_zone_net",
                    "gnd_strand_repair_enabled",
                    "gnd_strand_repair_max_mm",
                    "gnd_parent_repair_max_iter",
                    "gnd_edge_spine_enabled",
                    "gnd_edge_spine_width_mm",
                    "gnd_edge_spine_max_inset_mm",
                    "component_zones",
                    "freerouting_min_clearance_mm",
                    "freerouting_fine_pitch_track_mm",
                    "via_drill_mm",
                    "via_size_mm",
                    "hole_to_hole_min_mm",
                )
                if k in cfg
            })
            _run_pcbnew_script(
                "import json\n"
                "from kicraft.autoplacer.brain.gnd_pour import repair_parent_gnd_islands\n"
                f"cfg = json.loads({_rep_cfg!r})\n"
                f"s = repair_parent_gnd_islands({str(routed_pcb)!r}, cfg)\n"
                f"open({str(sidecar)!r}, 'w').write(json.dumps(s))\n"
            )
            s = json.loads(sidecar.read_text(encoding="utf-8"))
            post_route_repairs["gnd_islands"] = s
            print(f"  parent gnd island repair: {s.get('stranded')} stranded, "
                  f"{s.get('tied_pads')} tied, {s.get('vias')} vias, "
                  f"{(s.get('edge_spine') or {}).get('stubs', 0)} spine stubs, "
                  f"{s.get('unresolved')} unresolved, "
                  f"iterations: {s.get('iterations')}")
        except Exception as exc:
            post_route_repairs["gnd_islands"] = {"failed": str(exc)}
            print(f"warning: parent gnd island repair failed: {exc}", file=sys.stderr)
        finally:
            sidecar.unlink(missing_ok=True)

    # Power strand repair: the power-rail pour fragments exactly like the GND
    # plane (KC-Z57JEZ: +3V3 split into two F.Cu islands around a fine-pitch
    # LGA, stranding the part's supply pads) -- tie each stranded power
    # cluster back the same way. pcbnew work -> subprocess.
    if cfg.get("power_plane_enabled", True) and cfg.get(
            "power_strand_repair_enabled", True):
        sidecar = Path(str(routed_pcb) + ".power_strand_repair.json")
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
                f"open({str(sidecar)!r}, 'w').write(json.dumps(s))\n"
            )
            s = json.loads(sidecar.read_text(encoding="utf-8"))
            post_route_repairs["power_strand"] = s
            print(f"  power strand repair: {s.get('nets')} -- "
                  f"{s.get('stranded')} stranded, {s.get('tied')} tied, "
                  f"{len(s.get('skipped') or [])} skipped"
                  + (": " + "; ".join(s["skipped"]) if s.get("skipped") else ""))
        except Exception as exc:
            post_route_repairs["power_strand"] = {"failed": str(exc)}
            print(f"warning: power strand repair failed: {exc}", file=sys.stderr)
        finally:
            sidecar.unlink(missing_ok=True)

    # Root parent has no interface anchors -- skip anchor validation.
    # Anchor completeness is a leaf-level gate, not a parent-level gate.
    validation = validate_routed_board(
        str(routed_pcb),
        cfg=cfg,
        expected_anchor_names=[],
        actual_anchor_names=[],
        required_anchor_names=[],
    )
    # Persist the post-route repair summaries (and the power-first phase
    # outcome) into the validation dict -> state.routed_validation ->
    # parent_pipeline.json: their absence must never again be ambiguous
    # (on 1/655 nothing recorded whether any repair ran; KC-ZRAUR7 B3).
    validation["post_route_repairs"] = post_route_repairs
    if power_first_stats is not None:
        validation["power_first"] = {
            k: power_first_stats.get(k) for k in ("nets", "failed")
            if k in power_first_stats
        }

    # Illegal-geometry remediation (self-eval 2026-07-20 N2): freerouting
    # occasionally ships copper the fab gate must reject -- escaped past
    # Edge.Cuts, inside the copper-edge clearance, or in genuine
    # track-clearance conflict. Rip exactly the offending track/via copper,
    # re-close the opens with the existing repair machinery, then
    # accept-or-revert on a full re-validate: the geometry flags must clear
    # and shorts/unconnected must not rise, else the board is byte-restored
    # and the honest rejection below stands.
    if (
        cfg.get("illegal_geometry_repair_enabled", True)
        and (
            validation.get("malformed_board_geometry")
            or validation.get("obviously_illegal_routed_geometry")
        )
    ):
        validation = _attempt_illegal_geometry_repair(
            routed_pcb, cfg, validation
        )

    # C1 signal unconnected repair: freerouting sometimes walls a signal net
    # off (no_clear_path) and never rip-up-recovers. Attempt a constrained
    # local bend/via repair (guarded copper only), then accept-or-revert on a
    # full re-DRC: unconnected must strictly drop and shorts must not rise,
    # else the board is byte-restored and the honest verdict below stands.
    # Design: docs/plans/unconnected-signal-repair-c1-design.md.
    unconnected = int((validation.get("drc") or {}).get("unconnected", 0) or 0)
    if unconnected > 0 and cfg.get("signal_unconnected_repair_enabled", True):
        # Poured POWER nets are normally the strand repair's turf (the signal
        # repair skips every net with a zone), but a power net the strand
        # repair just reported unresolved fell between the two systems on
        # 1/655: the straight-tie-only strand repair skipped no_clear_path
        # while the rich bend/via repair filtered VBUS out as pour-owned.
        # Hand exactly those nets to the signal repair -- its accept-or-revert
        # gate makes trying safe. GND stays excluded (plane machinery owns it).
        pwr = post_route_repairs.get("power_strand") or {}
        extra = sorted({
            str(lbl).split(":", 1)[0]
            for lbl in (pwr.get("skipped") or [])
            if ":" in str(lbl)
        })
        repair_cfg = {**cfg, "signal_repair_extra_nets": extra} if extra else cfg
        validation = _attempt_signal_unconnected_repair(
            routed_pcb, repair_cfg, validation
        )
        unconnected = int(
            (validation.get("drc") or {}).get("unconnected", 0) or 0
        )
    elif unconnected > 0:
        validation["signal_unconnected_repair"] = {
            "ran": False, "reason": "disabled"
        }

    # Import all copper from the routed board (child + new parent traces +
    # any repair ties; must run AFTER the repair so its copper is captured).
    copper = import_routed_copper(str(routed_pcb))

    # A parent must close every net. Unlike a leaf -- whose interface ports are
    # legitimately open, so validate_routed_board waives unconnected -- unrouted
    # nets on the parent mean an unusable board (the final build verify requires
    # 0 unconnected). Reject here so the search keeps trying other rounds for a
    # fully-routed parent instead of promoting one the verify gate would fail.
    if unconnected > 0:
        validation["accepted"] = False
        reasons = validation.setdefault("rejection_reasons", [])
        if "unconnected_nets" not in reasons:
            reasons.append("unconnected_nets")

    # Gross courtyard overlaps are a guaranteed terminal-verify failure, but
    # validate_routed_board's DRC-derived reasons never included them, so a
    # parent whose only defect was courtyards_overlap was accepted here and
    # scored "functional" -- the search converged on boards the fab gate then
    # rejected (replay-confirmed on live 623/628). Mirror the verify gate's
    # severity split exactly (_verify_routed_board): a minor clip (below the
    # warn thresholds) stays a warning there, so it must not reject here
    # either; a gross overlap -- or one whose magnitude cannot be measured --
    # rejects like unconnected. Compose main still promotes the board for
    # inspection via the promotable-defect path.
    courtyard = int((validation.get("drc") or {}).get("courtyard", 0) or 0)
    if courtyard > 0:
        from kicraft.autoplacer.courtyard_overlap import (
            classify_courtyard_overlaps,
            measure_courtyard_overlaps,
        )

        measured = measure_courtyard_overlaps(str(routed_pcb))
        _, gross = classify_courtyard_overlaps(
            measured,
            max_penetration_mm=float(
                cfg.get("courtyard_overlap_warn_penetration_mm", 0.5)
            ),
            max_area_mm2=float(cfg.get("courtyard_overlap_warn_area_mm2", 0.5)),
        )
        if gross or not measured:
            validation["accepted"] = False
            reasons = validation.setdefault("rejection_reasons", [])
            reason = (
                "courtyards_overlap" if measured else "courtyard_unmeasured"
            )
            if reason not in reasons:
                reasons.append(reason)
            validation["courtyard_overlaps"] = [o.to_dict() for o in measured]

    return {
        "failed": False,
        "routed_board_path": str(routed_pcb),
        "_trace_segments": copper.get("traces", []),
        "_via_objects": copper.get("vias", []),
        "validation": validation,
        "freerouting_stats": freerouting_stats,
    }


def _attempt_signal_unconnected_repair(
    routed_pcb, cfg: dict, validation: dict
) -> dict:
    """Run the C1 repair in a pcbnew subprocess; keep it only if re-DRC improves.

    Accept iff unconnected strictly decreased AND shorts did not increase;
    anything else (including a crashed subprocess) restores the pre-repair
    board byte-for-byte and returns the original validation unchanged.
    """
    import shutil

    from kicraft.autoplacer.freerouting_runner import (
        _run_pcbnew_script,
        validate_routed_board,
    )

    drc = validation.get("drc") or {}
    unconnected_before = int(drc.get("unconnected", 0) or 0)
    shorts_before = int(drc.get("shorts", 0) or 0)
    backup = Path(str(routed_pcb) + ".pre_signal_repair")
    shutil.copy2(routed_pcb, backup)
    # Recorded on WHICHEVER validation dict is returned (kept, reverted, or
    # crashed): the pass ran invisibly on 1/655 -- its print went to a
    # swallowed subprocess stdout, its sidecar was unlinked, and neither
    # return path carried a trace (KC-ZRAUR7 workstream B).
    record: dict = {"ran": True, "kept": False}
    try:
        _sig_cfg = json.dumps({
            k: cfg[k]
            for k in (
                "gnd_zone_net",
                "power_plane_nets",
                "signal_repair_extra_nets",
                "signal_repair_max_mm",
                "signal_repair_max_targets",
                "signal_repair_dogleg_offsets_mm",
                "freerouting_min_clearance_mm",
                "freerouting_fine_pitch_track_mm",
                "via_size_mm",
                "via_drill_mm",
            )
            if k in cfg
        })
        _run_pcbnew_script(
            "import json\n"
            "from kicraft.autoplacer.brain.unconnected_repair import "
            "repair_unconnected_signals\n"
            f"cfg = json.loads({_sig_cfg!r})\n"
            f"s = repair_unconnected_signals({str(routed_pcb)!r}, cfg)\n"
            # NB: keys must match repair_unconnected_signals' return contract
            # {edges, tied, skipped, pruned} EXACTLY -- a KeyError here
            # crashes the subprocess AFTER the repair mutated the board, so
            # the except below silently byte-reverts it: the pass ran as a
            # no-op on every rc7 board of the 20260710 batch (N5 sweep).
            "print('signal unconnected repair:', s['edges'], 'edge(s) --',\n"
            "      s['tied'], 'tied,', len(s['skipped']), 'skipped,',\n"
            "      s['pruned'], 'pruned'\n"
            "      + (': ' + '; '.join(s['skipped']) if s['skipped'] else ''))\n"
            # The subprocess's stdout is not echoed into the compose log, so
            # persist the summary for the parent to surface -- the per-edge
            # skip REASONS are what the next repair-geometry iteration needs
            # (the 20260713 batch had to re-run repairs offline to get them).
            f"open({str(routed_pcb) + '.signal_repair.json'!r}, 'w')"
            ".write(json.dumps(s))\n"
        )
        sidecar = Path(str(routed_pcb) + ".signal_repair.json")
        try:
            s = json.loads(sidecar.read_text(encoding="utf-8"))
            record["summary"] = s
            print(
                f"  signal unconnected repair: {s.get('edges')} edge(s) -- "
                f"{s.get('tied')} tied, {len(s.get('skipped') or [])} skipped, "
                f"{s.get('pruned')} pruned"
                + (": " + "; ".join(s["skipped"]) if s.get("skipped") else "")
            )
        except Exception:
            pass
        finally:
            sidecar.unlink(missing_ok=True)
        revalidation = validate_routed_board(
            str(routed_pcb),
            cfg=cfg,
            expected_anchor_names=[],
            actual_anchor_names=[],
            required_anchor_names=[],
        )
        re_drc = revalidation.get("drc") or {}
        unconnected_after = int(re_drc.get("unconnected", 0) or 0)
        shorts_after = int(re_drc.get("shorts", 0) or 0)
        # A tie that closes an open by stamping ILLEGAL copper trades one
        # fab-gate rejection for another (run_10: the USB_DN tie grazed a
        # foreign pad at 0.05 mm and this gate accepted it on the
        # unconnected drop alone) -- the geometry flags must not appear.
        geometry_worse = (
            (revalidation.get("malformed_board_geometry")
             and not validation.get("malformed_board_geometry"))
            or (revalidation.get("obviously_illegal_routed_geometry")
                and not validation.get("obviously_illegal_routed_geometry"))
        )
        record["unconnected"] = [unconnected_before, unconnected_after]
        record["shorts"] = [shorts_before, shorts_after]
        if (unconnected_after < unconnected_before
                and shorts_after <= shorts_before
                and not geometry_worse):
            print(
                f"  signal unconnected repair KEPT: unconnected "
                f"{unconnected_before} -> {unconnected_after}, shorts "
                f"{shorts_before} -> {shorts_after}"
            )
            backup.unlink(missing_ok=True)
            record["kept"] = True
            # The fresh re-validate dict must not drop the earlier
            # annotations (post_route_repairs, power_first, prior repairs).
            for key in ("post_route_repairs", "power_first",
                        "illegal_geometry_repair"):
                if key in validation:
                    revalidation[key] = validation[key]
            revalidation["signal_unconnected_repair"] = record
            return revalidation
        print(
            f"  signal unconnected repair reverted (unconnected "
            f"{unconnected_before} -> {unconnected_after}, shorts "
            f"{shorts_before} -> {shorts_after}"
            + (", ties stamped illegal geometry" if geometry_worse else "")
            + "); board restored"
        )
        shutil.copy2(backup, routed_pcb)
        backup.unlink(missing_ok=True)
        record["reverted"] = True
        validation["signal_unconnected_repair"] = record
        return validation
    except Exception as exc:  # noqa: BLE001 -- a repair may never fail a board
        print(f"warning: signal unconnected repair failed: {exc}",
              file=sys.stderr)
        if backup.exists():
            shutil.copy2(backup, routed_pcb)
            backup.unlink(missing_ok=True)
        record["failed"] = str(exc)
        validation["signal_unconnected_repair"] = record
        return validation


def _attempt_illegal_geometry_repair(
    routed_pcb, cfg: dict, validation: dict
) -> dict:
    """Rip DRC-illegal copper, re-close the opens, keep only a clean result.

    Accept iff BOTH geometry flags cleared AND shorts/unconnected did not
    increase; anything else (including a crashed subprocess) restores the
    pre-repair board byte-for-byte and returns the original validation.
    """
    import shutil

    from kicraft.autoplacer.freerouting_runner import (
        _run_pcbnew_script,
        validate_routed_board,
    )

    drc = validation.get("drc") or {}
    unconnected_before = int(drc.get("unconnected", 0) or 0)
    shorts_before = int(drc.get("shorts", 0) or 0)
    backup = Path(str(routed_pcb) + ".pre_geometry_repair")
    shutil.copy2(routed_pcb, backup)
    sidecar = Path(str(routed_pcb) + ".geometry_repair.json")
    # Recorded on whichever validation dict is returned -- see the signal
    # wrapper's record note (a repair pass must never run traceless again).
    record: dict = {"ran": True, "kept": False}
    try:
        _geo_cfg = json.dumps({
            k: cfg[k] for k in ("geometry_repair_max_rips",) if k in cfg
        })
        _run_pcbnew_script(
            "import json\n"
            "from kicraft.autoplacer.brain.geometry_repair import "
            "rip_illegal_copper\n"
            f"cfg = json.loads({_geo_cfg!r})\n"
            f"s = rip_illegal_copper({str(routed_pcb)!r}, cfg)\n"
            # NB: keys must match rip_illegal_copper's return contract
            # {ripped, over_cap, nets, skipped} EXACTLY (see the signal
            # wrapper's no-op-on-KeyError incident).
            "print('illegal geometry rip:', s['ripped'], 'ripped,',\n"
            "      s['over_cap'], 'over cap,', len(s['skipped']),\n"
            "      'skipped, nets:', ','.join(s['nets']))\n"
            f"open({str(sidecar)!r}, 'w').write(json.dumps(s))\n"
        )
        s = json.loads(sidecar.read_text(encoding="utf-8"))
        record["rip"] = s
        ripped = int(s.get("ripped", 0) or 0)
        print(
            f"  illegal geometry rip: {ripped} item(s) ripped, "
            f"{s.get('over_cap')} over cap, "
            f"{len(s.get('skipped') or [])} skipped"
            + (f", nets: {', '.join(s['nets'])}" if s.get("nets") else "")
        )
        if ripped == 0:
            backup.unlink(missing_ok=True)
            validation["illegal_geometry_repair"] = record
            return validation

        # Close the opens the rip created: pour nets go back through the
        # island/strand machinery, everything else through the C1 stamper.
        pour_nets = {str(cfg.get("gnd_zone_net", "GND"))}
        pour_nets.update(cfg.get("power_plane_nets") or [])
        if pour_nets & set(s.get("nets") or []):
            _rep_cfg = json.dumps({
                k: cfg[k]
                for k in (
                    "gnd_zone_net",
                    "gnd_strand_repair_enabled",
                    "gnd_strand_repair_max_mm",
                    "gnd_parent_repair_max_iter",
                    "gnd_edge_spine_enabled",
                    "gnd_edge_spine_width_mm",
                    "gnd_edge_spine_max_inset_mm",
                    "component_zones",
                    "freerouting_min_clearance_mm",
                    "freerouting_fine_pitch_track_mm",
                    "via_drill_mm",
                    "via_size_mm",
                    "hole_to_hole_min_mm",
                )
                if k in cfg
            })
            _run_pcbnew_script(
                "import json\n"
                "from kicraft.autoplacer.brain.gnd_pour import "
                "repair_parent_gnd_islands\n"
                f"cfg = json.loads({_rep_cfg!r})\n"
                f"s = repair_parent_gnd_islands({str(routed_pcb)!r}, cfg)\n"
                "print('post-rip gnd island repair:', s['stranded'],\n"
                "      'stranded,', s['tied_pads'], 'tied,', s['vias'],\n"
                "      'vias,', s['unresolved'], 'unresolved')\n"
            )
        _sig_cfg = json.dumps({
            k: cfg[k]
            for k in (
                "gnd_zone_net",
                "power_plane_nets",
                "signal_repair_max_mm",
                "freerouting_min_clearance_mm",
                "freerouting_fine_pitch_track_mm",
                "via_size_mm",
                "via_drill_mm",
            )
            if k in cfg
        })
        _run_pcbnew_script(
            "import json\n"
            "from kicraft.autoplacer.brain.unconnected_repair import "
            "repair_unconnected_signals\n"
            f"cfg = json.loads({_sig_cfg!r})\n"
            f"s = repair_unconnected_signals({str(routed_pcb)!r}, cfg)\n"
            "print('post-rip signal repair:', s['edges'], 'edge(s) --',\n"
            "      s['tied'], 'tied,', len(s['skipped']), 'skipped,',\n"
            "      s['pruned'], 'pruned')\n"
        )

        revalidation = validate_routed_board(
            str(routed_pcb),
            cfg=cfg,
            expected_anchor_names=[],
            actual_anchor_names=[],
            required_anchor_names=[],
        )
        re_drc = revalidation.get("drc") or {}
        unconnected_after = int(re_drc.get("unconnected", 0) or 0)
        shorts_after = int(re_drc.get("shorts", 0) or 0)
        geometry_clean = not (
            revalidation.get("malformed_board_geometry")
            or revalidation.get("obviously_illegal_routed_geometry")
        )
        record["unconnected"] = [unconnected_before, unconnected_after]
        record["shorts"] = [shorts_before, shorts_after]
        if (geometry_clean
                and shorts_after <= shorts_before
                and unconnected_after <= unconnected_before):
            print(
                f"  illegal geometry repair KEPT: geometry clean, "
                f"unconnected {unconnected_before} -> {unconnected_after}, "
                f"shorts {shorts_before} -> {shorts_after}"
            )
            backup.unlink(missing_ok=True)
            record["kept"] = True
            for key in ("post_route_repairs", "power_first"):
                if key in validation:
                    revalidation[key] = validation[key]
            revalidation["illegal_geometry_repair"] = record
            return revalidation
        print(
            f"  illegal geometry repair reverted (geometry_clean="
            f"{geometry_clean}, unconnected {unconnected_before} -> "
            f"{unconnected_after}, shorts {shorts_before} -> "
            f"{shorts_after}); board restored"
        )
        shutil.copy2(backup, routed_pcb)
        backup.unlink(missing_ok=True)
        record["reverted"] = True
        validation["illegal_geometry_repair"] = record
        return validation
    except Exception as exc:  # noqa: BLE001 -- a repair may never fail a board
        print(f"warning: illegal geometry repair failed: {exc}",
              file=sys.stderr)
        if backup.exists():
            shutil.copy2(backup, routed_pcb)
            backup.unlink(missing_ok=True)
        record["failed"] = str(exc)
        validation["illegal_geometry_repair"] = record
        return validation
    finally:
        sidecar.unlink(missing_ok=True)
