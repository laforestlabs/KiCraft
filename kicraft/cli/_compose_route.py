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
    try:
        if gnd_net:
            strip_net_copper(str(stamped_pcb), gnd_net)
            add_gnd_pour_and_thermal_vias(str(stamped_pcb), cfg)
            _stamp_shield_ties(str(stamped_pcb))
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
            "freerouting_stats": {},
        }

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
    # pcbnew work -> subprocess.
    if gnd_net and cfg.get("gnd_strand_repair_enabled", True):
        try:
            from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script

            _rep_cfg = json.dumps({
                k: cfg[k]
                for k in (
                    "gnd_zone_net",
                    "gnd_strand_repair_enabled",
                    "gnd_strand_repair_max_mm",
                    "gnd_parent_repair_max_iter",
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
                "print('parent gnd island repair:', s['stranded'], 'stranded,',\n"
                "      s['tied_pads'], 'tied,', s['vias'], 'vias,',\n"
                "      s['unresolved'], 'unresolved, iterations:', s['iterations'])\n"
            )
        except Exception as exc:
            print(f"warning: parent gnd island repair failed: {exc}", file=sys.stderr)

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

    # Root parent has no interface anchors -- skip anchor validation.
    # Anchor completeness is a leaf-level gate, not a parent-level gate.
    validation = validate_routed_board(
        str(routed_pcb),
        cfg=cfg,
        expected_anchor_names=[],
        actual_anchor_names=[],
        required_anchor_names=[],
    )

    # C1 signal unconnected repair: freerouting sometimes walls a signal net
    # off (no_clear_path) and never rip-up-recovers. Attempt a constrained
    # local bend/via repair (guarded copper only), then accept-or-revert on a
    # full re-DRC: unconnected must strictly drop and shorts must not rise,
    # else the board is byte-restored and the honest verdict below stands.
    # Design: docs/plans/unconnected-signal-repair-c1-design.md.
    unconnected = int((validation.get("drc") or {}).get("unconnected", 0) or 0)
    if unconnected > 0 and cfg.get("signal_unconnected_repair_enabled", True):
        validation = _attempt_signal_unconnected_repair(
            routed_pcb, cfg, validation
        )
        unconnected = int(
            (validation.get("drc") or {}).get("unconnected", 0) or 0
        )

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
    try:
        _sig_cfg = json.dumps({
            k: cfg[k]
            for k in (
                "gnd_zone_net",
                "power_plane_nets",
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
            # {edges, tied, skipped} EXACTLY -- a KeyError here crashes the
            # subprocess AFTER the repair mutated the board, so the except
            # below silently byte-reverts it: the pass ran as a no-op on
            # every rc7 board of the 20260710 batch (N5 evidence sweep).
            "print('signal unconnected repair:', s['edges'], 'edge(s) --',\n"
            "      s['tied'], 'tied,', len(s['skipped']), 'skipped'\n"
            "      + (': ' + '; '.join(s['skipped']) if s['skipped'] else ''))\n"
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
        if (unconnected_after < unconnected_before
                and shorts_after <= shorts_before):
            print(
                f"  signal unconnected repair KEPT: unconnected "
                f"{unconnected_before} -> {unconnected_after}, shorts "
                f"{shorts_before} -> {shorts_after}"
            )
            backup.unlink(missing_ok=True)
            return revalidation
        print(
            f"  signal unconnected repair reverted (unconnected "
            f"{unconnected_before} -> {unconnected_after}, shorts "
            f"{shorts_before} -> {shorts_after}); board restored"
        )
        shutil.copy2(backup, routed_pcb)
        backup.unlink(missing_ok=True)
        return validation
    except Exception as exc:  # noqa: BLE001 -- a repair may never fail a board
        print(f"warning: signal unconnected repair failed: {exc}",
              file=sys.stderr)
        if backup.exists():
            shutil.copy2(backup, routed_pcb)
            backup.unlink(missing_ok=True)
        return validation
