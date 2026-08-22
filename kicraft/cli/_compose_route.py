"""Route a stamped parent board via KRT and validate it.

Split out of ``compose_subcircuits.py`` (Lever 2.5); re-exported there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kicraft.cli._compose_state import ParentCompositionState


def _route_parent_board(
    stamped_pcb: Path,
    state: ParentCompositionState,
    project_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Route the stamped parent once with KRT, then finish and validate it."""
    from kicraft.autoplacer.brain.gnd_pour import pour_gnd_planes
    from kicraft.autoplacer.kicad_routing_tools import route_with_kicad_routing_tools
    from kicraft.autoplacer.routing_board import (
        import_routed_copper,
        validate_routed_board,
    )

    if state.composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    routed_pcb = stamped_pcb.parent / "parent_routed.kicad_pcb"
    route_cfg = {**cfg, "pcb_path": str(stamped_pcb)}
    try:
        routing_stats = route_with_kicad_routing_tools(
            str(stamped_pcb), str(routed_pcb), route_cfg
        )
    except Exception as exc:
        return {
            "failed": True,
            "error": str(exc),
            "routed_board_path": str(routed_pcb),
            "_trace_segments": [],
            "_via_objects": [],
            "validation": {},
            "routing_stats": {},
        }

    gnd_net = cfg.get("gnd_zone_net", "GND")
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
            from kicraft.autoplacer.routing_board import run_pcbnew_script

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
                                                            "via_drill_mm",
                    "via_size_mm",
                    "hole_to_hole_min_mm",
                )
                if k in cfg
            })
            run_pcbnew_script(
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
            from kicraft.autoplacer.routing_board import run_pcbnew_script

            _pwr_cfg = json.dumps({
                k: cfg[k]
                for k in (
                    "power_strand_repair_enabled",
                    "power_plane_enabled",
                    "power_plane_nets",
                    "power_plane_max_nets",
                    "gnd_zone_net",
                    "gnd_strand_repair_max_mm",
                                                        )
                if k in cfg
            })
            run_pcbnew_script(
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
    # Persist post-route repair evidence with the validation record.
    validation["post_route_repairs"] = post_route_repairs
    # Illegal-geometry remediation (self-eval 2026-07-20 N2): a router or repair pass can produce copper the fab gate must reject -- escaped past
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

    # C1 signal unconnected repair: routing can leave a signal net open. Attempt a constrained
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
        "backend": "kicad-routing-tools",
        "routing_stats": routing_stats,
    }


def _attempt_signal_unconnected_repair(
    routed_pcb, cfg: dict, validation: dict
) -> dict:
    """Parent-side entry to the shared accept-or-revert C1 repair pass."""
    from kicraft.autoplacer.brain.signal_repair_pass import (
        attempt_signal_unconnected_repair,
    )

    return attempt_signal_unconnected_repair(routed_pcb, cfg, validation)


def _attempt_illegal_geometry_repair(
    routed_pcb, cfg: dict, validation: dict
) -> dict:
    """Rip DRC-illegal copper, re-close the opens, keep only a clean result.

    Accept iff BOTH geometry flags cleared AND shorts/unconnected did not
    increase; anything else (including a crashed subprocess) restores the
    pre-repair board byte-for-byte and returns the original validation.
    """
    import shutil

    from kicraft.autoplacer.routing_board import (
        run_pcbnew_script,
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
        run_pcbnew_script(
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
                    "via_drill_mm",
                    "via_size_mm",
                    "hole_to_hole_min_mm",
                )
                if k in cfg
            })
            run_pcbnew_script(
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
                "via_size_mm",
                "via_drill_mm",
            )
            if k in cfg
        })
        run_pcbnew_script(
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
