from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain import geometry
from kicraft.autoplacer.brain.array_placement import leaf_is_fully_array
from kicraft.autoplacer.brain.leaf_geometry import (
    grow_leaf_outline_to_contain_placement,
    repair_leaf_placement_legality,
)
from kicraft.autoplacer.brain.subcircuit_artifacts import resolve_artifact_paths
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard
from kicraft.autoplacer.brain.subcircuit_render_diagnostics import (
    LeafStageOpts,
    generate_leaf_diagnostic_artifacts,
    generate_stage_diagnostic_artifacts,
    promote_to_round_snapshot,
)
from kicraft.autoplacer.brain.types import AntennaEdgeIntent, BoardState, Component, Layer, Point
from kicraft.autoplacer.routing_board import (
    import_routed_copper,
    run_pcbnew_script,
    validate_routed_board,
)
from kicraft.autoplacer.hardware.adapter import KiCadAdapter
from kicraft.autoplacer.kicad_routing_tools import (
    KICAD_ROUTING_TOOLS_COMMIT,
    KICAD_ROUTING_TOOLS_VERSION,
    route_with_kicad_routing_tools,
)


def _validate_leaf_current_contracts(
    board_path: str | Path, cfg: dict[str, Any], validation: dict[str, Any]
) -> None:
    """Audit final leaf current paths and locally reconstructible returns."""
    specs = cfg.get("local_bypasses", [])
    if not specs:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="kicraft-leaf-current-") as tmp:
            result_path = Path(tmp) / "audit.json"
            script = (
                "import json, pcbnew\n"
                "from kicraft.autoplacer.hardware.local_contract import audit_local_copper\n"
                f"board = pcbnew.LoadBoard({str(board_path)!r})\n"
                "if board is None: raise RuntimeError('Failed to load board')\n"
                f"specs = json.loads({json.dumps(specs)!r})\n"
                "selected, errors = [], []\n"
                "for spec in specs:\n"
                "    owner, member = spec['owner_ref'], spec['bypass_ref']\n"
                "    has_owner = board.FindFootprintByReference(owner) is not None\n"
                "    has_member = board.FindFootprintByReference(member) is not None\n"
                "    if has_owner != has_member:\n"
                "        errors.append(f'local contract {owner}/{member}: owner and member split across leaves')\n"
                "    elif has_owner:\n"
                "        ground = spec.get('ground_return')\n"
                "        if isinstance(ground, dict) and isinstance(ground.get('owner_ref'), str) and ground['owner_ref'].strip() and board.FindFootprintByReference(ground['owner_ref']) is None:\n"
                "            spec = {k: v for k, v in spec.items() if k != 'ground_return'}\n"
                "        selected.append(spec)\n"
                "errors.extend(audit_local_copper(board, selected))\n"
                f"with open({str(result_path)!r}, 'w') as out: json.dump(errors, out)\n"
            )
            run_pcbnew_script(script)
            errors = json.loads(result_path.read_text())
    except Exception as exc:
        errors = [f"current contract native audit failed: {exc}"]
    validation["current_contract_errors"] = errors
    if errors:
        validation["accepted"] = False
        reasons = validation.setdefault("rejection_reasons", [])
        if "current_contract" not in reasons:
            reasons.append("current_contract")


def _validate_leaf_thermal_requirements(
    board_path: str | Path, cfg: dict[str, Any], validation: dict[str, Any]
) -> None:
    """Reject inadequate thermal geometry for owners present on this leaf."""
    requirements = cfg.get("thermal_requirements", [])
    if not requirements:
        return
    try:
        with tempfile.TemporaryDirectory(prefix="kicraft-leaf-thermal-") as tmp:
            result_path = Path(tmp) / "audit.json"
            script = (
                "import json, pcbnew\n"
                "from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground\n"
                f"path = {str(board_path)!r}\n"
                "board = pcbnew.LoadBoard(path)\n"
                "if board is None: raise RuntimeError('Failed to load board')\n"
                f"requirements = json.loads({json.dumps(requirements)!r})\n"
                "selected = [r for r in requirements if board.FindFootprintByReference(r['owner_ref']) is not None]\n"
                "errors = []\n"
                "for requirement in selected:\n"
                "    try:\n"
                f"        audit_thermal_ground(path, [requirement], {cfg.get('gnd_zone_net', 'GND')!r})\n"
                "    except Exception as exc:\n"
                "        errors.append(str(exc))\n"
                f"with open({str(result_path)!r}, 'w') as out: json.dump(errors, out)\n"
            )
            run_pcbnew_script(script)
            errors = json.loads(result_path.read_text())
    except Exception as exc:
        errors = [f"thermal native audit failed: {exc}"]
    validation["thermal_contract_errors"] = errors
    if errors:
        validation["accepted"] = False
        reasons = validation.setdefault("rejection_reasons", [])
        if "thermal_contract" not in reasons:
            reasons.append("thermal_contract")


def _leaf_ground_contracts(board_path: Path, cfg: dict[str, Any]) -> list[dict]:
    """Select ground returns whose two physical terminals are on this leaf."""
    if not any(spec.get("ground_return") for spec in cfg.get("local_bypasses", [])):
        return []
    import pcbnew

    board = pcbnew.LoadBoard(str(board_path))
    if board is None:
        raise ValueError(f"failed to load leaf board: {board_path}")
    return [
        spec for spec in cfg.get("local_bypasses", [])
        if isinstance(spec.get("ground_return"), dict)
        and board.FindFootprintByReference(spec["ground_return"].get("owner_ref", ""))
        and board.FindFootprintByReference(spec.get("bypass_ref", ""))
    ]


def _prepare_leaf_ground_reservations(
    board_path: Path, source_pcb: Path, cfg: dict[str, Any]
) -> tuple[str | None, set[tuple[str, str]]]:
    """Protect ordinary native return witnesses before any signal escapes."""
    import pcbnew
    from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs
    from kicraft.autoplacer.brain.gnd_pour import gnd_escape_specs
    from kicraft.autoplacer.kicad_routing_tools import (
        _project_routing_floors, _propagate_sibling_project_rules,
        _routing_project, _stamp_local_ground_keepouts,
    )

    contracts = _leaf_ground_contracts(board_path, cfg)
    if not contracts:
        return None, set()
    limits: dict[tuple[str, str], float] = {}
    for spec in contracts:
        ground = spec["ground_return"]
        limit = float(ground["max_via_distance_mm"])
        for pin in ((ground["owner_ref"], str(ground["owner_pin"])),
                    (spec["bypass_ref"], str(ground["member_pin"]))):
            limits[pin] = min(limit, limits.get(pin, limit))
    _propagate_sibling_project_rules(str(source_pcb), str(board_path))
    board = pcbnew.LoadBoard(str(board_path))
    escapes = []
    for net in sorted({spec["ground_return"]["net"] for spec in contracts}):
        for escape in gnd_escape_specs(board, {**cfg, "gnd_zone_net": net, "gnd_pre_escape": True}):
            key = (escape.ref, str(escape.pad))
            if key in limits:
                escape.length_mm = min(escape.length_mm, limits[key])
                escapes.append(escape)
    if escapes:
        sidecars = [board_path.with_suffix(suffix) for suffix in (".kicad_pro", ".kicad_dru")]
        rules = [(path, path.read_bytes() if path.is_file() else None) for path in sidecars]
        try:
            add_breakout_stubs(str(board_path), escapes, cfg=cfg)
        finally:
            for path, content in rules:
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
    marker = f"KiCraft leaf ground reservation {uuid.uuid4().hex}"
    project = _routing_project(board_path, {**cfg, "pcb_path": str(source_pcb)})
    clearance = _project_routing_floors(project, cfg)["clearance"]
    _stamp_local_ground_keepouts(
        board_path, {**cfg, "local_bypasses": contracts}, marker,
        router_clearance_mm=clearance, prepare_thermal=False,
    )
    return marker, set(limits)


def _finish_leaf_ground_planes(board_path: Path, cfg: dict[str, Any]) -> None:
    """Refill both faces for leaf-owned ordinary returns before native audit."""
    from kicraft.autoplacer.brain.gnd_pour import pour_gnd_planes

    contracts = _leaf_ground_contracts(board_path, cfg)
    if not contracts:
        return
    sidecars = [board_path.with_suffix(suffix) for suffix in (".kicad_pro", ".kicad_dru")]
    rules = [(path, path.read_bytes() if path.is_file() else None) for path in sidecars]
    try:
        for net in sorted({spec["ground_return"]["net"] for spec in contracts}):
            pour_gnd_planes(str(board_path), {**cfg, "gnd_zone_net": net},
                            layers=("F.Cu", "B.Cu"))
    finally:
        for path, content in rules:
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def _finish_leaf_current_landings(board_path: Path, cfg: dict[str, Any]) -> dict:
    """Close shallow same-net via contacts with guarded, full-width copper."""
    import math
    import pcbnew
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec, add_breakout_stubs
    from kicraft.autoplacer.hardware.local_contract import (
        _position, _project, _track_via_annular_contact, _via_width, audit_local_copper,
    )

    empty = {"stubs": 0, "segments": 0, "vias": 0, "skipped": []}
    if not any("max_route_length_mm" in spec for spec in cfg.get("local_bypasses", [])):
        return empty
    board = pcbnew.LoadBoard(str(board_path))
    contracts = [
        {key: value for key, value in spec.items() if key != "ground_return"}
        for spec in cfg.get("local_bypasses", [])
        if "max_route_length_mm" in spec
        and board.FindFootprintByReference(spec["owner_ref"])
        and board.FindFootprintByReference(spec["bypass_ref"])
    ]
    if not contracts or not audit_local_copper(board, contracts):
        return empty
    nets = set()
    for spec in contracts:
        owner = board.FindFootprintByReference(spec.get("route_anchor_ref", spec["owner_ref"]))
        if owner is None:
            continue
        for pair in spec.get("pin_pairs", [spec.get("power_pins", [])]):
            if not pair:
                continue
            number = str(spec.get("route_anchor_pin", pair[0]))
            nets.update(pad.GetNetname() for pad in owner.Pads()
                        if pad.GetNumber() == number and pad.GetNetname())
    vias: dict[int, list] = {}
    tracks = []
    for item in board.GetTracks():
        if item.GetNetname() not in nets:
            continue
        if item.GetClass() == "PCB_VIA":
            vias.setdefault(item.GetNetCode(), []).append(item)
        elif item.GetClass() == "PCB_TRACK":
            tracks.append(item)
    escapes = []
    seen = set()
    for track in tracks:
        start, end = _position(track.GetStart()), _position(track.GetEnd())
        layer, width = track.GetLayer(), track.GetWidth()
        for via in vias.get(track.GetNetCode(), []):
            if not via.IsOnLayer(layer):
                continue
            center = _position(via.GetPosition())
            closest = _project(start, end, center)
            distance = math.dist(closest, center)
            if distance <= 2 or distance >= (width + _via_width(via, layer)) / 2:
                continue
            if _track_via_annular_contact(
                start, end, width, center, _via_width(via, layer), via.GetDrillValue()
            ) is not None:
                continue
            key = (closest, center, layer, width)
            if key in seen:
                continue
            seen.add(key)
            escapes.append(BreakoutSpec(
                ref="via-landing", pad=str(len(escapes)), net=track.GetNetname(),
                start_xy=tuple(pcbnew.ToMM(value) for value in closest),
                waypoints=[tuple(pcbnew.ToMM(value) for value in center)],
                width_mm=pcbnew.ToMM(width), layer=track.GetLayerName(),
                via_at_end=False,
            ))
    if not escapes:
        return empty
    rules = [(board_path.with_suffix(suffix),
              board_path.with_suffix(suffix).read_bytes()
              if board_path.with_suffix(suffix).is_file() else None)
             for suffix in (".kicad_pro", ".kicad_dru")]
    try:
        result = add_breakout_stubs(str(board_path), escapes, cfg=cfg)
        if result["segments"]:
            board = pcbnew.LoadBoard(str(board_path))
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
            pcbnew.SaveBoard(str(board_path), board)
        return result
    finally:
        for path, content in rules:
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def _validate_leaf_input(board_path: Path, marker: str | None, **kwargs) -> dict:
    """Do not mistake the temporary routing masks for fabricated copper DRC."""
    if marker is None:
        return validate_routed_board(str(board_path), **kwargs)
    from kicraft.autoplacer.kicad_routing_tools import (
        _propagate_sibling_project_rules, _strip_router_keepouts,
    )

    with tempfile.TemporaryDirectory(prefix="kicraft-leaf-ground-drc-") as tmp:
        copy_path = Path(tmp) / board_path.name
        shutil.copy2(board_path, copy_path)
        _propagate_sibling_project_rules(str(board_path), str(copy_path))
        _strip_router_keepouts(copy_path, marker)
        import pcbnew
        board = pcbnew.LoadBoard(str(copy_path))
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        pcbnew.SaveBoard(str(copy_path), board)
        _propagate_sibling_project_rules(str(board_path), str(copy_path))
        return validate_routed_board(str(copy_path), **kwargs)


def _leaf_gnd_pre_escape_specs(
    board: Any, extraction: ExtractedSubcircuitBoard, cfg: dict[str, Any],
    protected_pins: set[tuple[str, str]],
) -> list:
    """Defer optional interface-package bonds until signal routing is complete."""
    from kicraft.autoplacer.brain.gnd_pour import gnd_escape_specs
    from kicraft.design.models import is_power_or_ground_name

    power = set(cfg.get("power_nets", ())) | {cfg.get("gnd_zone_net", "GND")}
    signals = {net for net in extraction.external_net_names
               if net not in power and not is_power_or_ground_name(net)}
    interface_refs = {
        fp.GetReferenceAsString() for fp in board.GetFootprints()
        if any(pad.GetNetname() in signals for pad in fp.Pads())
    }
    # Contracted returns already have immutable copper witnesses. Ordinary
    # interface-package bonds are completed by the existing ground finisher;
    # locking them here can cage the nearby cross-leaf signal landings.
    return [spec for spec in gnd_escape_specs(board, cfg)
            if spec.ref not in interface_refs
            and (spec.ref, str(spec.pad)) not in protected_pins]


def _reserve_signal_escapes(board_path: Path, extraction: ExtractedSubcircuitBoard,
                            cfg: dict[str, Any]) -> dict[str, Any]:
    """Reserve signal exits before other locked copper; report every obstruction."""
    try:
        import pcbnew
        from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs
        from kicraft.autoplacer.brain.signal_escape import signal_escape_specs

        board = pcbnew.LoadBoard(str(board_path))
        if board is None:
            raise ValueError(f"failed to load {board_path}")
        specs = signal_escape_specs(board, cfg, set(extraction.external_net_names))
        del board
        result = add_breakout_stubs(str(board_path), specs, cfg=cfg) if specs else {
            "stubs": 0, "vias": 0, "skipped": [],
        }
        result["candidates"] = len(specs)
        print(
            f"  Signal pre-escapes: {result['stubs']}/{len(specs)} pad(s), "
            f"{result['vias']} fanout via(s), {len(result['skipped'])} blocked"
        )
        if result["skipped"]:
            print("  Signal pre-escapes blocked: " + ", ".join(result["skipped"]))
        return result
    except Exception as exc:
        print(f"  WARNING: signal pre-escape step failed: {exc}")
        return {"stubs": 0, "vias": 0, "skipped": [], "error": str(exc)}


def _finish_interface_landings(
    board_path: Path, extraction: ExtractedSubcircuitBoard, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Retry only unlanded interface pads against the finished routed copper."""
    import pcbnew
    from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs
    from kicraft.autoplacer.brain.signal_escape import (
        signal_escape_specs, unlanded_interface_components,
    )
    from kicraft.design.models import is_power_or_ground_name

    nets = set(extraction.external_net_names) - set(cfg.get("power_nets", ()))
    nets = {name for name in nets if not is_power_or_ground_name(name)}
    board = pcbnew.LoadBoard(str(board_path))
    if board is None:
        raise ValueError(f"failed to load {board_path}")
    missing = unlanded_interface_components(board, nets)
    pads = {label for item in missing for label in item["pads"]}
    specs = [spec for spec in signal_escape_specs(board, cfg, nets)
             if f"{spec.ref}.{spec.pad}" in pads]
    del board
    if not specs:
        return {"stubs": 0, "vias": 0, "skipped": []}
    sidecars = [board_path.with_suffix(suffix) for suffix in (".kicad_pro", ".kicad_dru")]
    rules = [(path, path.read_bytes() if path.is_file() else None) for path in sidecars]
    try:
        return add_breakout_stubs(str(board_path), specs, cfg=cfg)
    finally:
        for path, content in rules:
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def _audit_interface_landings(board_path: Path, extraction: ExtractedSubcircuitBoard,
                              validation: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Require a native landing for every actual cross-leaf signal component."""
    from kicraft.design.models import is_power_or_ground_name

    signal_nets = set(extraction.external_net_names) - set(cfg.get("power_nets", ()))
    signal_nets = {name for name in signal_nets if not is_power_or_ground_name(name)}
    if not signal_nets:
        validation["interface_landing_audit"] = {"unlanded_components": []}
        validation["interface_unlanded_nets"] = []
        return
    try:
        with tempfile.TemporaryDirectory(prefix="kicraft-interface-landings-") as tmp:
            result_path = Path(tmp) / "audit.json"
            script = (
                "import json, pcbnew\n"
                "from kicraft.autoplacer.brain.signal_escape import unlanded_interface_components\n"
                f"board = pcbnew.LoadBoard({str(board_path)!r})\n"
                "if board is None: raise RuntimeError('Failed to load board')\n"
                f"result = unlanded_interface_components(board, set({sorted(signal_nets)!r}))\n"
                f"with open({str(result_path)!r}, 'w') as out: json.dump(result, out)\n"
            )
            run_pcbnew_script(script)
            missing = json.loads(result_path.read_text())
        validation["interface_landing_audit"] = {"unlanded_components": missing}
        validation["interface_unlanded_nets"] = sorted({row["net"] for row in missing})
    except Exception as exc:
        validation["interface_landing_audit"] = {"error": str(exc)}
        validation["interface_unlanded_nets"] = sorted(signal_nets)


def _internal_net_verdict(
    validation: dict[str, Any],
    extraction: ExtractedSubcircuitBoard,
) -> tuple[list[str], list[str]]:
    """``(routed_internal_nets, failed_internal_nets)`` from the DRC that just ran.

    The success path used to HARDCODE "everything routed, nothing failed", so
    per-net truth existed only inside the acceptance DRC and never reached the
    round record -- the retry loop could not react per net, and the round log
    said a net was routed while routing had silently added zero copper on it
    (dense-soc-leaf-unconnected-plan P1.5). When the report has unconnected items
    but no parsable net names (format drift), nothing is claimed either way.
    """
    internal = set(extraction.internal_net_names)
    drc = validation.get("drc", {}) or {}
    nets = {str(n) for n in (drc.get("unconnected_nets", []) or [])}
    if not nets and int(drc.get("unconnected", 0) or 0) > 0:
        return [], []
    return sorted(internal - nets), sorted(internal & nets)


def _resolve_breakout_specs(cfg: dict[str, Any]) -> list:
    """Build :class:`BreakoutSpec` objects from ``cfg['breakout_specs']``.

    Each entry is a dict ``{ref, pad, waypoints?, length_mm?, width_mm?,
    layer?, via_at_end?}``. Returns ``[]`` when none are configured. Specs whose
    footprint isn't on the leaf are harmlessly skipped by ``add_breakout_stubs``.
    """
    raw = cfg.get("breakout_specs")
    if not raw:
        return []
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec

    specs = []
    for d in raw:
        if not d.get("ref") or d.get("pad") is None:
            continue
        specs.append(
            BreakoutSpec(
                ref=str(d["ref"]),
                pad=str(d["pad"]),
                waypoints=[tuple(p) for p in d.get("waypoints", [])],
                length_mm=float(d.get("length_mm", 1.5)),
                width_mm=d.get("width_mm"),
                layer=d.get("layer", "F.Cu"),
                via_at_end=bool(d.get("via_at_end", False)),
            )
        )
    return specs


def _silk_for_leaf(
    extraction: ExtractedSubcircuitBoard,
    components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    traces=None,
    vias=None,
    outline: tuple[Point, Point] | None = None,
) -> list:
    """Compute leaf-local rounded-rect silk + optional label for stamping.

    Lazy-imports the solver helpers so this module stays cheap to import
    in pipelines that don't actually stamp boards. Returns ``[]`` when
    components are empty or no project ``group_labels`` entry matches.

    Pass ``traces`` and ``vias`` post-route so the silk hugs every visible
    piece of copper (courtyards + pad copper + tracks + vias). Pre-route
    callers can omit them; the silk then hugs courtyards + pad copper only.
    """
    if not components:
        return []
    from kicraft.autoplacer.brain.subcircuit_solver import (
        _build_leaf_silkscreen,
        _compute_component_bbox,
    )

    bbox = _compute_component_bbox(components, traces=traces, vias=vias)
    return _build_leaf_silkscreen(components, bbox, extraction, cfg, outline=outline)


def _leaf_mating_edges(
    board_path: str | Path, components: dict[str, Component], cfg: dict[str, Any],
) -> dict[str, float]:
    """Native mating datums, using the parent's marker/barrel/body convention."""
    zones = cfg.get("component_zones") or {}
    refs = {
        ref: zones[ref]["edge"]
        for ref, comp in components.items()
        if comp.mating_axis == "in_plane" and comp.opening_direction is not None
        and isinstance(zones.get(ref), dict)
        and zones[ref].get("edge") in ("left", "right", "top", "bottom")
    }
    if not refs:
        return {}
    overhang = max(0.0, float(cfg.get("connector_edge_overhang_mm", 0.5)))
    pad_clearance = max(0.0, float(cfg.get("connector_edge_pad_clearance_mm", 0.2)))
    with tempfile.TemporaryDirectory(prefix="kicraft-leaf-mating-") as tmp:
        result_path = Path(tmp) / "edges.json"
        script = (
            "import json, pcbnew\n"
            "from kicraft.autoplacer.brain.connector_edge_gap import _mouth_bbox, _edge_marker_point\n"
            "from kicraft.autoplacer.brain.subcircuit_composer import _connector_barrel_edge_anchor\n"
            "from kicraft.autoplacer.brain.types import Point\n"
            f"board = pcbnew.LoadBoard({str(board_path)!r})\n"
            f"refs = json.loads({json.dumps(refs)!r})\n"
            "limits = {}\n"
            "for ref, side in refs.items():\n"
            "    fp = board.FindFootprintByReference(ref)\n"
            "    if fp is None: raise ValueError(f'missing mating connector {ref}')\n"
            "    bb = _mouth_bbox(fp, pcbnew)\n"
            "    lo = Point(pcbnew.ToMM(bb.GetLeft()), pcbnew.ToMM(bb.GetTop()))\n"
            "    hi = Point(pcbnew.ToMM(bb.GetRight()), pcbnew.ToMM(bb.GetBottom()))\n"
            "    marker = _edge_marker_point(fp, pcbnew)\n"
            "    pads = [p.GetBoundingBox() for p in fp.Pads()]\n"
            "    barrel = None\n"
            "    if pads:\n"
            "        p_lo = Point(min(pcbnew.ToMM(p.GetLeft()) for p in pads), min(pcbnew.ToMM(p.GetTop()) for p in pads))\n"
            "        p_hi = Point(max(pcbnew.ToMM(p.GetRight()) for p in pads), max(pcbnew.ToMM(p.GetBottom()) for p in pads))\n"
            "        barrel = _connector_barrel_edge_anchor(lo, hi, p_lo, p_hi, edge_marker=marker)\n"
            "    anchor = Point(*marker) if marker is not None else barrel\n"
            "    if anchor is None: anchor = Point(lo.x if side == 'left' else hi.x, lo.y if side == 'top' else hi.y)\n"
            "    datum = anchor.x if side in ('left', 'right') else anchor.y\n"
            f"    offset = -{pad_clearance!r} if barrel is not None else {overhang!r}\n"
            "    limit = datum + offset if side in ('left', 'top') else datum - offset\n"
            "    if side in limits: limit = max(limits[side], limit) if side in ('left', 'top') else min(limits[side], limit)\n"
            "    limits[side] = limit\n"
            f"with open({str(result_path)!r}, 'w') as out: json.dump(limits, out)\n"
        )
        run_pcbnew_script(script)
        return json.loads(result_path.read_text())


def _clamp_mating_edges(
    outline: tuple[Point, Point], limits: dict[str, float],
) -> tuple[Point, Point]:
    tl, br = outline
    clipped = (
        Point(max(tl.x, limits.get("left", tl.x)), max(tl.y, limits.get("top", tl.y))),
        Point(min(br.x, limits.get("right", br.x)), min(br.y, limits.get("bottom", br.y))),
    )
    if clipped[0].x >= clipped[1].x or clipped[0].y >= clipped[1].y:
        raise ValueError("connector mating boundaries leave no leaf routing area")
    return clipped


def _outline_around_geometry(
    components: dict[str, Component],
    cfg: dict[str, Any],
    antenna_edge_intents: list[AntennaEdgeIntent] | None = None,
    *,
    traces=None,
    vias=None,
    mating_edges: dict[str, float] | None = None,
) -> tuple[Point, Point] | None:
    """Compute a tight Edge.Cuts outline hugging the same bbox as the silk.

    The silk poly is drawn at ``bbox ± silkscreen_margin_mm`` (default 0.5 mm).
    Edge.Cuts defaults to ``edge_margin = silk_margin`` so the yellow board
    outline sits exactly on top of the white silk outline (zero gap).
    Callers can still override via ``leaf_edge_margin_mm`` if they want a
    visible gap. Returns ``None`` for empty leaves so the caller can keep
    the original outline.
    """
    if not components:
        return None
    from kicraft.autoplacer.brain.subcircuit_solver import _compute_component_bbox

    bbox = _compute_component_bbox(components, traces=traces, vias=vias)
    silk_margin = float(cfg.get("silkscreen_margin_mm", 0.5))
    edge_margin = float(cfg.get("leaf_edge_margin_mm", silk_margin))
    tl = Point(bbox["min_x"] - edge_margin, bbox["min_y"] - edge_margin)
    br = Point(bbox["max_x"] + edge_margin, bbox["max_y"] + edge_margin)
    for intent in antenna_edge_intents or []:
        component = components.get(intent.owner_ref)
        if component is None:
            continue
        point = intent.local_anchor_midpoint
        if component.layer == Layer.BACK:
            point = Point(-point.x, point.y)
        anchor = geometry.transform_point(point, component.pos, component.rotation)
        if intent.target_edge == "left":
            tl.x = anchor.x - intent.inset_mm
        elif intent.target_edge == "right":
            br.x = anchor.x + intent.inset_mm
        elif intent.target_edge == "top":
            tl.y = anchor.y - intent.inset_mm
        elif intent.target_edge == "bottom":
            br.y = anchor.y + intent.inset_mm
    return _clamp_mating_edges((tl, br), mating_edges or {})


def _center_on_leaf_page(
    tl: Point, br: Point, cfg: dict[str, Any]
) -> tuple[Point, tuple[Point, Point]]:
    """Translation that centers the ``[tl, br]`` content box on a standard A4 leaf
    page, plus the resulting centered Edge.Cuts outline.

    Mirrors the parent's A4 centering (compose_subcircuits._stamp_parent_board) so a
    standalone leaf opens centered in the title block instead of crammed against the
    top-left origin. Position on the page is free: the parent composer re-bases each
    leaf to its own outline origin on load (subcircuit_instances._layout_from_artifact_payload),
    so a centered leaf composes identically to one anchored at (0, 0). Returns
    ``(delta, (new_tl, new_br))``.
    """
    page_w = float(cfg.get("leaf_page_width_mm", cfg.get("parent_page_width_mm", 297.0)))
    page_h = float(cfg.get("leaf_page_height_mm", cfg.get("parent_page_height_mm", 210.0)))
    w = br.x - tl.x
    h = br.y - tl.y
    dx = (page_w - w) / 2.0 - tl.x
    dy = (page_h - h) / 2.0 - tl.y
    return Point(dx, dy), (Point(tl.x + dx, tl.y + dy), Point(br.x + dx, br.y + dy))


def _deterministic_route_signature(board_state: Any, cfg: dict[str, Any]) -> str:
    """Stable hash of placement and non-timeout KRT routing options."""
    comps = sorted(board_state.components.values(), key=lambda c: c.ref)
    placement = [
        (c.ref, round(c.pos.x, 3), round(c.pos.y, 3), round(float(c.rotation), 1), int(c.layer))
        for c in comps
    ]
    tl, br = board_state.board_outline
    route_keys = {
        k: v for k, v in cfg.items() if k.startswith("kicad_routing_tools_") and "timeout" not in k
    }
    blob = json.dumps(
        [
            placement,
            [round(tl.x, 3), round(tl.y, 3), round(br.x, 3), round(br.y, 3)],
            f"kicad-routing-tools:{KICAD_ROUTING_TOOLS_VERSION}@{KICAD_ROUTING_TOOLS_COMMIT}",
            route_keys,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _array_inrow_stamp_stats(specs: list, skipped_keys: list) -> tuple[int, int]:
    """(in_row_ties, in_row_skipped) for an array leaf's daisy-chain.

    In-row hops are single straight ties (one waypoint); serpentine row-turn hops
    are L/Z routes (multiple waypoints) OR were never emitted at all (no edge
    channel fit). Either way a turn may legitimately fall back to the autorouter, so
    it must NOT count against the stamp gate -- only the in-row hops, which on a
    healthy oriented array essentially always stamp, are measured. ``skipped_keys``
    are the ``"ref.pad:reason"`` strings the stamp guard dropped.
    """
    in_row = [s for s in specs if len(getattr(s, "waypoints", []) or []) == 1]
    keys = {f"{s.ref}.{s.pad}" for s in in_row}
    n_skipped = sum(1 for k in skipped_keys if str(k).split(":", 1)[0] in keys)
    return len(in_row), n_skipped


def array_stamp_gate_tripped(specs: list, skipped_keys: list, cfg: dict[str, Any]) -> bool:
    """True when an array leaf pre-stamped too few of its IN-ROW data hops.

    A healthy array leaf stamps almost every straight in-row DOUT->DIN hop, so
    the autorouter has next to nothing left. A collapsed in-row stamp rate means the
    channel is obstructed (a foreign pad in the data lane) and the autorouter will be
    handed the whole chain -- which it cannot finish on a dense array before the
    build's wall-clock cap kills it. Row-turn hops are excluded (they may
    legitimately go to the autorouter), and the gate is floored at a minimum in-row
    tie count, so a small array -- e.g. a 2x2 whose only un-stamped hop is its row
    turn -- never false-fails.
    """
    if not cfg.get("array_stamp_gate_enabled", True):
        return False
    n_in_row, n_skipped = _array_inrow_stamp_stats(specs, skipped_keys)
    if not n_in_row:
        return False
    min_ties = int(cfg.get("array_min_data_ties_for_gate", 6))
    min_rate = float(cfg.get("array_min_stamp_rate", 0.5))
    if n_in_row < min_ties:
        return False
    return (n_in_row - n_skipped) / n_in_row < min_rate


def route_local_subcircuit(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    generate_diagnostics: bool = True,
    round_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    fast_smoke_mode = bool(cfg.get("subcircuit_fast_smoke_mode", False))
    # Intermediate per-round diagnostics: headless build/worker runs set
    # subcircuit_render_intermediate=False to skip per-round PNG/DRC renders;
    # the winning round's canonical renders are produced downstream and are
    # unaffected by this flag. Defaults True for interactive/debug parity.
    render_intermediate = bool(cfg.get("subcircuit_render_intermediate", True))
    render_pre_route_board_views = bool(
        cfg.get("subcircuit_render_pre_route_board_views", not fast_smoke_mode)
    )
    render_routed_board_views = bool(cfg.get("subcircuit_render_routed_board_views", True))
    render_pre_route_drc_overlay = bool(
        cfg.get("subcircuit_render_pre_route_drc_overlay", not fast_smoke_mode)
    )
    render_routed_drc_overlay = bool(
        cfg.get("subcircuit_render_routed_drc_overlay", not fast_smoke_mode)
    )
    write_pre_route_drc_json = bool(
        cfg.get("subcircuit_write_pre_route_drc_json", not fast_smoke_mode)
    )
    write_routed_drc_json = bool(cfg.get("subcircuit_write_routed_drc_json", True))
    write_pre_route_drc_report = bool(
        cfg.get("subcircuit_write_pre_route_drc_report", not fast_smoke_mode)
    )
    write_routed_drc_report = bool(cfg.get("subcircuit_write_routed_drc_report", True))
    build_comparison_contact_sheet = bool(
        cfg.get("subcircuit_build_comparison_contact_sheet", not fast_smoke_mode)
    )
    # A leaf must fully route the on-leaf span of EVERY net that touches it,
    # including nets that also connect to other sheets -- their inter-sheet hop
    # is completed at the parent stage. extraction.local_state.nets already holds
    # those external nets filtered to their on-leaf pads (see _partition_nets /
    # _filter_net_to_components), so routing the board connects them locally.
    # Only a leaf with no net having >=2 pads on it (e.g. a pass-through
    # connector where every net has a single on-leaf pad) is genuinely trivial.
    routable_on_leaf_nets = [
        name for name, net in extraction.local_state.nets.items() if len(net.pad_refs) >= 2
    ]
    if not routable_on_leaf_nets:
        # Trivial leaf: nothing to route, but we still stamp the placed
        # components onto a real PCB so the leaf flows through the same
        # workflow as every other leaf -- pin_best_leaves can promote it,
        # the GUI snapshot picker shows its rounds, and the composer reads
        # uniformly from leaf_routed.kicad_pcb.
        return _stamp_trivial_leaf(
            extraction=extraction,
            solved_components=solved_components,
            cfg=cfg,
            round_index=round_index,
            generate_diagnostics=generate_diagnostics,
            render_intermediate=render_intermediate,
            fast_smoke_mode=fast_smoke_mode,
        )

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = Path(artifact_paths.artifact_dir) / "leaf_placed.kicad_pcb"
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"
    illegal_board = Path(artifact_paths.artifact_dir) / "leaf_illegal_pre_stamp.kicad_pcb"

    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    legality_start = time.monotonic()
    repaired_components, legality_repair = repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(max(0.0, time.monotonic() - legality_start), 3)

    # Reframe recovery: a placement that is internally legal (no overlapping
    # copper) but whose pads spill past the content-sized canvas is a
    # canvas-too-small problem, not a bad placement -- e.g. a column of stacked
    # THT headers taller than the roughly-square content canvas (KC-99A9M8, an
    # Arduino-shield header sheet). Grow the leaf outline to bound the placement
    # and re-legalize rather than failing the whole build over it. Only kicks in
    # for a *pure* canvas overflow (zero overlaps, pads outside only): a real
    # overlap still fails, unchanged.
    if not legality_repair.get("resolved", False):
        _diag = legality_repair.get("diagnostics", {}) or {}
        _pure_canvas_overflow = (
            int(_diag.get("overlap_count", 0) or 0) == 0
            and int(_diag.get("pad_outside_count", 0) or 0) > 0
        )
        if _pure_canvas_overflow and grow_leaf_outline_to_contain_placement(
            extraction, repaired_components, cfg
        ):
            repaired_components, legality_repair = repair_leaf_placement_legality(
                extraction, solved_components, cfg
            )
            if legality_repair.get("resolved", False):
                print(
                    "  Leaf canvas reframed to contain a legal-but-oversized "
                    "placement -> outline "
                    f"{extraction.local_state.board_width:.1f}x"
                    f"{extraction.local_state.board_height:.1f}mm "
                    f"(was overflowing by {_diag.get('pad_outside_count')} pad(s))"
                )

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = Path(extraction.subcircuit.schematic_path).with_suffix(".kicad_pcb")

    if not source_pcb.exists():
        raise RuntimeError(
            "Leaf the autorouter requires a real source PCB to stamp from; "
            f"could not resolve base board for {extraction.subcircuit.id.instance_path}"
        )

    if not legality_repair.get("resolved", False):
        diagnostics = legality_repair.get("diagnostics", {}) or {}
        overlap_count = int(diagnostics.get("overlap_count", 0) or 0)
        pad_outside_count = int(diagnostics.get("pad_outside_count", 0) or 0)
        overlap_pairs = [
            f"{item.get('a', '?')}:{item.get('b', '?')}" for item in diagnostics.get("overlaps", [])
        ]
        pad_violations = [
            f"{item.get('ref', '?')}:{item.get('pad_id', '?')}:{','.join(item.get('sides', []))}"
            for item in diagnostics.get("pads_outside_board", [])
        ]

        overlap_details = []
        for item in diagnostics.get("overlaps", []):
            overlap_details.append(
                {
                    "a": item.get("a"),
                    "b": item.get("b"),
                    "overlap_x_mm": item.get("overlap_x_mm"),
                    "overlap_y_mm": item.get("overlap_y_mm"),
                    "overlap_area_mm2": item.get("overlap_area_mm2"),
                }
            )

        component_debug = []
        repaired_by_ref = repaired_components or {}
        for ref in sorted(repaired_by_ref.keys()):
            comp = repaired_by_ref[ref]
            component_debug.append(
                {
                    "ref": ref,
                    "kind": comp.kind,
                    "layer": str(comp.layer),
                    "locked": bool(comp.locked),
                    "x_mm": round(comp.pos.x, 4),
                    "y_mm": round(comp.pos.y, 4),
                    "rotation_deg": round(comp.rotation, 4),
                    "width_mm": round(comp.width_mm, 4),
                    "height_mm": round(comp.height_mm, 4),
                    "pad_count": len(comp.pads),
                }
            )

        print(
            "  Leaf legality repair rejected placement: "
            f"overlaps={overlap_count} "
            f"pads_outside={pad_outside_count} "
            f"overlap_pairs={overlap_pairs} "
            f"pad_violations={pad_violations}"
        )
        if overlap_details:
            print(f"  Leaf legality overlap details: {overlap_details}")
        if component_debug:
            print(f"  Leaf legality component states: {component_debug}")

        illegal_input_board = copy.deepcopy(extraction.local_state)
        illegal_input_board.components = copy.deepcopy(repaired_components)
        illegal_input_board.traces = []
        illegal_input_board.vias = []

        illegal_render_diagnostics: dict[str, Any] = {
            "artifact_dir": artifact_paths.artifact_dir,
            "renders_dir": str(Path(artifact_paths.artifact_dir) / "renders"),
            "illegal_pre_stamp": None,
            "errors": [],
        }

        try:
            route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
            route_adapter.stamp_subcircuit_board(
                illegal_input_board,
                output_path=str(illegal_board),
                clear_existing_tracks=True,
                clear_existing_zones=True,
                remove_unmapped_footprints=True,
            )
            illegal_validation = {
                "accepted": False,
                "rejected": True,
                "rejection_stage": "leaf_pre_stamp_legality_repair",
                "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "drc": {
                    "violations": [],
                    "report_text": (
                        "Leaf placement rejected before routing due to placement legality.\n"
                        f"overlap_count={overlap_count}\n"
                        f"pad_outside_count={pad_outside_count}\n"
                        f"overlap_pairs={overlap_pairs}\n"
                        f"pad_violations={pad_violations}\n"
                    ),
                },
            }
            illegal_render_diagnostics["illegal_pre_stamp"] = generate_stage_diagnostic_artifacts(
                pcb_path=str(illegal_board),
                validation=illegal_validation,
                artifact_dir=artifact_paths.artifact_dir,
                stage="illegal_pre_stamp",
                render_board_views=not fast_smoke_mode,
                write_drc_json=not fast_smoke_mode,
                write_drc_report=not fast_smoke_mode,
                render_drc_overlay=not fast_smoke_mode,
            )
        except Exception as exc:
            illegal_render_diagnostics["errors"].append(f"illegal_pre_stamp_render_failed:{exc}")

        # Copy the illegal-pre-stamp PNGs (and the .kicad_pcb) to
        # round_XXXX_pre_route_* names so the GUI's _find_round_renders
        # can show *something* for failed rounds. Without this, failed
        # rounds appear as bare "-inf" cards with a placeholder icon and
        # the user has no way to inspect what placement was rejected.
        if round_index is not None:
            _round_prefix = f"round_{int(round_index):04d}"
            _renders_dir = Path(artifact_paths.artifact_dir) / "renders"
            _renders_dir.mkdir(parents=True, exist_ok=True)
            _ip = illegal_render_diagnostics.get("illegal_pre_stamp") or {}
            _view_paths = (_ip.get("board_views", {}) or {}).get("paths", {}) or {}
            for _view, _suffix in (
                ("front_all", "pre_route_front_all"),
                ("back_all", "pre_route_back_all"),
                ("copper_both", "pre_route_copper_both"),
            ):
                _src = _view_paths.get(_view)
                if _src and Path(_src).exists():
                    try:
                        shutil.copy2(_src, _renders_dir / f"{_round_prefix}_{_suffix}.png")
                    except OSError:
                        pass
            if illegal_board.exists():
                try:
                    shutil.copy2(
                        illegal_board,
                        illegal_board.parent / f"{_round_prefix}_leaf_placed{illegal_board.suffix}",
                    )
                except OSError:
                    pass

        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "illegal_unrepaired_leaf_placement",
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": "leaf_pre_stamp_legality_repair",
                    "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                    "leaf_legality_repair": copy.deepcopy(legality_repair),
                    "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                    if generate_diagnostics
                    else {"skipped": True, "reason": "size_reduction_fast_path"},
                    "illegal_pre_stamp_board_path": str(illegal_board),
                },
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                if generate_diagnostics
                else {"skipped": True, "reason": "size_reduction_fast_path"},
                "illegal_pre_stamp_board_path": str(illegal_board),
                "failed": True,
            },
            route_timing,
        )

    # The final pre-stamp repair may have moved individual footprints after
    # solve(). Re-establish compiler-owned pin locality on the actual input.
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    PlacementSolver(extraction.local_state, cfg)._place_local_bypasses(repaired_components)

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []
    # Silk is stamped post-route (after the autorouter) so the rounded
    # outline can hug the routed copper too -- not just courtyards and
    # pad copper. See the silk re-stamp block after import_routed_copper.
    route_input_board.silkscreen = []
    # Route against the antenna datum that the final board will retain. Applying
    # it only after routing crops legal pre-escapes off the board (including GND
    # corner vias) even though they cleared the original extraction outline.
    if route_input_board.antenna_edge_intents:
        antenna_outline = _outline_around_geometry(
            route_input_board.components,
            cfg,
            antenna_edge_intents=route_input_board.antenna_edge_intents,
        )
        if antenna_outline is not None:
            route_input_board.board_outline = antenna_outline

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    # A component-only extraction canvas may extend in front of a connector.
    # Establish its real mating edge BEFORE escapes or router copper exist.
    mating_edges = _leaf_mating_edges(pre_route_board, route_input_board.components, cfg)
    if mating_edges:
        route_input_board.board_outline = _clamp_mating_edges(
            route_input_board.board_outline, mating_edges,
        )
        route_adapter.stamp_subcircuit_board(
            route_input_board, output_path=str(pre_route_board),
            clear_existing_tracks=True, clear_existing_zones=True,
            remove_unmapped_footprints=True,
        )
    route_timing["stamp_pre_route_board_s"] = round(max(0.0, time.monotonic() - stamp_start), 3)

    routing_start = time.monotonic()
    leaf_routing_cfg = {**cfg, "pcb_path": str(source_pcb)}
    _breakout_specs = _resolve_breakout_specs(cfg)

    # Contracted returns need an actual plane witness before signal copper can
    # fragment it. Ordinary masks remain active through all routing passes.
    _ground_marker, _ground_pins = _prepare_leaf_ground_reservations(
        pre_route_board, source_pcb, cfg
    )
    if _ground_marker:
        leaf_routing_cfg["routing_preserve_existing_copper"] = True
    # Other dense signal exits still precede non-contract GND and power ties.
    _signal_escape = _reserve_signal_escapes(pre_route_board, extraction, cfg)
    if _signal_escape["stubs"] > 0:
        leaf_routing_cfg["routing_preserve_existing_copper"] = True

    # Escape remaining GND pads after reserving dense signal exits.
    if cfg.get("gnd_pre_escape", True):
        try:
            import pcbnew

            _g_board = pcbnew.LoadBoard(str(pre_route_board))
            _breakout_specs = _leaf_gnd_pre_escape_specs(
                _g_board, extraction, cfg, _ground_pins,
            ) + _breakout_specs
            del _g_board
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: gnd pre-escape spec gen failed: {exc}")
    # Auto power-tie (default on): route a tie around any connector whose spread
    # power pads (e.g. USB-C VBUS) would otherwise fragment the power pour.
    if cfg.get("auto_power_tie", True):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.breakout_stubs import auto_power_tie_specs

            _tie_board = pcbnew.LoadBoard(str(pre_route_board))
            _breakout_specs = _breakout_specs + auto_power_tie_specs(_tie_board, cfg)
            del _tie_board
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: auto power-tie spec gen failed: {exc}")
    # Shield tie (default on): a connector's through-hole shield legs sit where
    # neither GND plane can reach them (F.Cu walled out of the pad row by the
    # Power-netclass clearance, B.Cu thermal spokes lost to the slot holes), so
    # the legs facing the pad row survive as unconnected ratlines on an
    # otherwise-routed board. Tie each netted PTH pad to its nearest same-net pad.
    if cfg.get("shield_tie_enabled", True):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.breakout_stubs import shield_tie_specs

            _sh_board = pcbnew.LoadBoard(str(pre_route_board))
            _breakout_specs = _breakout_specs + shield_tie_specs(_sh_board, cfg)
            del _sh_board
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: shield tie spec gen failed: {exc}")
    # Array daisy-chain (default on): for an addressable-LED matrix or similar
    # regular array, deterministically stamp the short data hops (DOUT->DIN) as
    # locked ties + pad escapes, so the autorouter -- which abandons a few of these
    # in the dense inter-component channels every run -- only has to finish from
    # open copper. Power is delivered by the +5V/GND pours, so the channels
    # carry only data. Keyed on the array spec, NOT leaf_is_fully_array (a 3-pin
    # header or other non-passive on the leaf must not disable it).
    _arr_specs: list = []
    _arr_stamp_skipped: list = []
    if cfg.get("array_route_enabled", True) and cfg.get("arrays"):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.array_router import array_daisy_chain_specs

            _arr_board = pcbnew.LoadBoard(str(pre_route_board))
            _arr_specs = array_daisy_chain_specs(_arr_board, cfg)
            del _arr_board
            if _arr_specs:
                _breakout_specs = _breakout_specs + _arr_specs
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: array daisy-chain spec gen failed: {exc}")
    # Ring +5V bus (default on): for a ring array, deterministically stamp a
    # CLOSED loop of member->member (or member->decap->member) power ties at
    # the pad radius, plus a via stub tying each band decap's GND into the
    # B.Cu pour -- the autorouter then has no reason to dip into the ring
    # interior, which shaped-compose nesting needs clear (PR-N5). Kept as a
    # SEPARATE list from _arr_specs: the array stamp gate measures in-row
    # DATA hops only.
    _ring_pwr_specs: list = []
    if cfg.get("array_ring_power_bus", True) and cfg.get("arrays"):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.array_router import array_ring_power_specs

            _rp_board = pcbnew.LoadBoard(str(pre_route_board))
            _ring_pwr_specs = array_ring_power_specs(_rp_board, cfg)
            del _rp_board
            if _ring_pwr_specs:
                _breakout_specs = _breakout_specs + _ring_pwr_specs
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: ring power-bus spec gen failed: {exc}")
    if _breakout_specs:
        try:
            from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs

            _bo = add_breakout_stubs(str(pre_route_board), _breakout_specs, cfg=cfg)
            if _bo["stubs"] > 0:
                leaf_routing_cfg["routing_preserve_existing_copper"] = True
                print(
                    f"  Breakout stubs: {_bo['stubs']} pad(s), "
                    f"{_bo['segments']} segment(s), {_bo['vias']} via(s)"
                )
            # No-silent handoff: a data tie the stamp guards dropped goes to
            # the autorouter -- surface it so an incompletely-stamped chain is
            # visible, not hidden behind a clean-looking "stubs" count.
            if _arr_specs:
                _arr_keys = {f"{s.ref}.{s.pad}" for s in _arr_specs}
                _arr_stamp_skipped = [
                    s for s in _bo.get("skipped", []) if s.split(":", 1)[0] in _arr_keys
                ]
                if _arr_stamp_skipped:
                    print(
                        f"  array-router: {len(_arr_stamp_skipped)}/{len(_arr_specs)} "
                        f"data tie(s) left to the autorouter: {', '.join(_arr_stamp_skipped)}"
                    )
            # Same no-silent-handoff rule for the ring power bus (its ties
            # are excluded from the stamp gate, so this log is the only
            # place a dropped bus segment surfaces).
            if _ring_pwr_specs:
                _rp_keys = {f"{s.ref}.{s.pad}" for s in _ring_pwr_specs}
                _rp_skipped = [s for s in _bo.get("skipped", []) if s.split(":", 1)[0] in _rp_keys]
                if _rp_skipped:
                    print(
                        f"  ring-power: {len(_rp_skipped)}/{len(_ring_pwr_specs)} "
                        f"bus tie(s) left to the autorouter: {', '.join(_rp_skipped)}"
                    )
        except Exception as exc:  # finishing step must never fail the leaf
            print(f"  WARNING: breakout stub step failed: {exc}")

    # Layer-2 guardrail: a healthy array leaf pre-stamps almost its whole data
    # chain so the autorouter has next to nothing left. When the stamp rate collapses
    # (the inter-member channel is obstructed -- e.g. a foreign pad sitting in the
    # data lane, the KC-NZXXEE decaps-on-LEDs signature), the entire chain plus
    # power falls to the autorouter, which cannot finish a dense array before the
    # build's wall-clock cap kills it (30 min -> no board -> raw-component
    # preview). Reject the leaf fast and loud instead, naming the obstruction, so
    # the real cause surfaces in seconds rather than a silent timeout.
    if _arr_specs and array_stamp_gate_tripped(_arr_specs, _arr_stamp_skipped, cfg):
        _n_inrow, _n_inrow_skipped = _array_inrow_stamp_stats(_arr_specs, _arr_stamp_skipped)
        _stamped = _n_inrow - _n_inrow_skipped
        _rate = _stamped / _n_inrow
        _min_rate = float(cfg.get("array_min_stamp_rate", 0.5))
        print(
            f"  ARRAY GATE: only {_stamped}/{_n_inrow} in-row data ties stamped "
            f"({_rate:.0%} < {_min_rate:.0%}) -- the inter-member data channel "
            "is obstructed (a foreign pad in the data lane?). Rejecting the leaf "
            "BEFORE the autorouter rather than handing it a chain it cannot finish "
            "in time. Skipped: "
            + ", ".join(_arr_stamp_skipped[:12])
            + (" ..." if len(_arr_stamp_skipped) > 12 else "")
        )
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        _reason = "array_data_channel_obstructed"
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": _reason,
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": _reason,
                    "rejection_reasons": [_reason],
                    "drc": {
                        "violations": [],
                        "report_text": (
                            "Array leaf rejected before routing: data channel "
                            "obstructed (in-row hops only; row turns excluded).\n"
                            f"in_row_ties_total={_n_inrow}\n"
                            f"in_row_ties_stamped={_stamped}\n"
                            f"in_row_ties_skipped={_n_inrow_skipped}\n"
                            f"all_array_ties={len(_arr_specs)}\n"
                            f"in_row_stamp_rate={_rate:.3f} (min {_min_rate:.3f})\n"
                            "skipped=" + ", ".join(_arr_stamp_skipped) + "\n"
                        ),
                    },
                },
                "failed": True,
            },
            route_timing,
        )
    pre_route_validation_start = time.monotonic()
    pre_route_validation = _validate_leaf_input(
        pre_route_board, _ground_marker,
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[port.name for port in extraction.interface_ports if port.required],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    route_timing["pre_route_validation_s"] = round(
        max(0.0, time.monotonic() - pre_route_validation_start), 3
    )
    # Most pre-route findings remain informational. Keepout damage is different:
    # deterministic locked copper cannot be repaired by the autorouter.
    pre_route_drc = pre_route_validation.get("drc", {})
    if pre_route_drc.get("violations"):
        pre_route_violation_types = {v.get("type") for v in pre_route_drc["violations"]}
        print(
            f"  Pre-route DRC info: {len(pre_route_drc['violations'])} violations ({', '.join(sorted(pre_route_violation_types))})"
        )
    if int(pre_route_drc.get("items_not_allowed", 0) or 0) > 0:
        if generate_diagnostics:
            generate_leaf_diagnostic_artifacts(
                artifact_dir=artifact_paths.artifact_dir,
                pre_route_board=str(pre_route_board),
                routed_board=None,
                pre_route_validation=pre_route_validation,
                pre_route_opts=LeafStageOpts(
                    render_board_views=render_pre_route_board_views,
                    write_drc_json=write_pre_route_drc_json,
                    write_drc_report=write_pre_route_drc_report,
                    render_drc_overlay=render_pre_route_drc_overlay,
                ),
                routed_opts=LeafStageOpts.off(),
                build_contact_sheet=False,
                quiet_render=fast_smoke_mode,
            )
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "illegal_routed_geometry",
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    **pre_route_validation,
                    "rejected": True,
                    "rejection_stage": "leaf_pre_route_drc",
                },
                "failed": True,
            },
            route_timing,
        )
    # Route cache: deterministic array leaves reuse identical KRT output.
    _cache_pcb = None
    _cache_meta = None
    if leaf_is_fully_array(route_input_board.components, cfg.get("arrays", [])):
        _sig = _deterministic_route_signature(route_input_board, leaf_routing_cfg)
        _cache_dir = Path(artifact_paths.artifact_dir) / "route_cache"
        _cache_pcb = _cache_dir / f"{_sig}.kicad_pcb"
        _cache_meta = _cache_dir / f"{_sig}.json"
    if _cache_pcb is not None and _cache_pcb.exists() and _cache_meta.exists():
        shutil.copyfile(_cache_pcb, routed_board)
        routing_stats = json.loads(_cache_meta.read_text())
        routing_stats["route_cache_hit"] = True
        print("  [route-cache] deterministic leaf unchanged -> reused routed board (skipped KRT)")
    else:
        routing_stats = route_with_kicad_routing_tools(
            str(pre_route_board),
            str(routed_board),
            leaf_routing_cfg,
        )
        if _ground_marker:
            from kicraft.autoplacer.kicad_routing_tools import _strip_router_keepouts
            _strip_router_keepouts(routed_board, _ground_marker)
        if _cache_pcb is not None:
            _cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(routed_board, _cache_pcb)
            _cache_meta.write_text(json.dumps(routing_stats))
    if _ground_marker:
        from kicraft.autoplacer.kicad_routing_tools import _strip_router_keepouts
        _strip_router_keepouts(pre_route_board, _ground_marker)
    # Include any recovered signal exits in the uncentred routed layout.
    _signal_finish = _finish_interface_landings(routed_board, extraction, cfg)
    route_timing["routing_s"] = round(max(0.0, time.monotonic() - routing_start), 3)

    if generate_diagnostics and render_intermediate:
        pre_route_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board) if routed_board.exists() else None,
            pre_route_validation=pre_route_validation,
            pre_route_opts=LeafStageOpts(
                render_board_views=render_pre_route_board_views,
                write_drc_json=write_pre_route_drc_json,
                write_drc_report=write_pre_route_drc_report,
                render_drc_overlay=render_pre_route_drc_overlay,
            ),
            routed_opts=LeafStageOpts.off(),
            build_contact_sheet=False,
            quiet_render=fast_smoke_mode,
        )
        route_timing["pre_route_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - pre_route_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["pre_route_render_diagnostics_s"] = 0.0

    round_board_illegal_pre_stamp = ""
    round_board_pre_route = ""
    round_board_routed = ""

    _board_snapshot = promote_to_round_snapshot(pre_route_board, round_index)
    if _board_snapshot is not None:
        round_board_pre_route = str(_board_snapshot)

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        pre_route_section = leaf_diagnostics.get("pre_route", {})
        if isinstance(pre_route_section, dict):
            pre_route_views = pre_route_section.get("board_views", {})
            if isinstance(pre_route_views, dict):
                pre_route_paths = pre_route_views.get("paths", {})
                if isinstance(pre_route_paths, dict):
                    for _view in ("front_all", "back_all", "copper_both"):
                        snap = promote_to_round_snapshot(pre_route_paths.get(_view), round_index)
                        if snap is not None:
                            pre_route_paths[f"round_{_view}"] = str(snap)

    pre_route_validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    pre_route_validation["leaf_legality_repair"] = copy.deepcopy(legality_repair)
    if round_board_pre_route:
        pre_route_validation["round_board_pre_route"] = round_board_pre_route

    import_copper_start = time.monotonic()
    imported_copper = import_routed_copper(str(routed_board))
    route_timing["import_routed_copper_s"] = round(
        max(0.0, time.monotonic() - import_copper_start), 3
    )

    # Re-stamp the routed leaf board with silk computed from the FULL
    # post-route content (courtyards + pad copper + traces + vias).
    # Stamping silk here -- rather than on the pre-route board -- means
    # the rounded outline includes routed copper that extends past the
    # components, so the canvas + round-selector PNGs show one tight
    # outline instead of "silk hugs the chip body but the trace stub
    # pokes out past it."
    silk_stamp_start = time.monotonic()
    silk_adapter = KiCadAdapter(str(routed_board), config=cfg)
    routed_state_for_silk = silk_adapter.load()
    mating_edges = _leaf_mating_edges(routed_board, routed_state_for_silk.components, cfg)
    # Shrink Edge.Cuts to hug the post-route geometry the silk hugs, and center
    # the whole leaf on a standard A4 page so the standalone board opens centered in
    # the title block instead of crammed against the top-left origin. Centering is
    # safe for composition: the parent composer re-bases each leaf to its own outline
    # origin on load (subcircuit_instances._layout_from_artifact_payload) before it
    # rotates around (0, 0) and translates by the placement origin, so leaf page
    # position is free. Without the shrink, the yellow outline stays at whatever
    # size-reduction accepted (or the raw extractor envelope), and the rounded silk
    # sits inside a much larger sharp rect.
    _new_outline = _outline_around_geometry(
        routed_state_for_silk.components,
        cfg,
        traces=routed_state_for_silk.traces,
        vias=routed_state_for_silk.vias,
        antenna_edge_intents=routed_state_for_silk.antenna_edge_intents,
        mating_edges=mating_edges,
    )
    if _new_outline is not None:
        _new_tl, _new_br = _new_outline
        _delta, _centered_outline = _center_on_leaf_page(_new_tl, _new_br, cfg)
        if abs(_delta.x) > 1e-6 or abs(_delta.y) > 1e-6:
            from kicraft.autoplacer.brain.leaf_geometry import (
                copy_components_with_translation,
                copy_traces_with_translation,
                copy_vias_with_translation,
            )

            routed_state_for_silk.components = copy_components_with_translation(
                routed_state_for_silk.components, _delta
            )
            routed_state_for_silk.traces = copy_traces_with_translation(
                routed_state_for_silk.traces, _delta
            )
            routed_state_for_silk.vias = copy_vias_with_translation(
                routed_state_for_silk.vias, _delta
            )
        routed_state_for_silk.board_outline = _centered_outline
    # Silk is computed AFTER the translate so it lands in the centered frame.
    routed_state_for_silk.silkscreen = _silk_for_leaf(
        extraction,
        routed_state_for_silk.components,
        cfg,
        traces=routed_state_for_silk.traces,
        vias=routed_state_for_silk.vias,
        outline=routed_state_for_silk.board_outline if mating_edges else None,
    )
    silk_adapter.stamp_subcircuit_board(
        routed_state_for_silk,
        output_path=str(routed_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=False,
    )
    route_timing["silk_post_route_stamp_s"] = round(
        max(0.0, time.monotonic() - silk_stamp_start), 3
    )

    # Ground-plane finishing (default on): pour a full B.Cu GND plane -- the
    # ZONE_FILLER keeps it clear of rule-area keepouts like the WROOM antenna --
    # and stitch large GND/thermal pads into it with a dense thermal-via array so
    # the plane connects and the boxed-in center pad escapes to ground. Run after
    # the silk re-stamp (the last write to routed_board) and before acceptance
    # validation, so the now-connected center pad is reflected in shorts/unconnected.
    gnd_pour_summary: dict | None = None
    if cfg.get("gnd_plane_enabled", True):
        gnd_pour_start = time.monotonic()
        try:
            from kicraft.autoplacer.brain.gnd_pour import (
                add_gnd_pour_and_thermal_vias,
            )

            _gnd = add_gnd_pour_and_thermal_vias(str(routed_board), cfg)
            # Persisted into the leaf result (-> debug.json): stdout-only
            # summaries made the GND-strand family untriageable from the
            # experiments tree alone.
            gnd_pour_summary = _gnd
            print(
                f"  GND plane: {_gnd.get('thermal_vias_added', 0)} thermal via(s) "
                f"under {_gnd.get('gnd_pads_stitched', 0)} pad(s), "
                f"{_gnd.get('escape_stitched', 0)} escape(s), "
                f"{_gnd.get('thermal_vias_blocked', 0)} blocked; B.Cu pour filled"
            )
        except Exception as exc:  # finishing step must never fail the leaf
            print(f"  WARNING: GND plane step failed: {exc}")
        route_timing["gnd_pour_s"] = round(max(0.0, time.monotonic() - gnd_pour_start), 3)

    # Power-plane finishing (default on): pour the primary power rail on the
    # layer opposite GND so paired connector power pads / regulator input / bulk
    # caps connect through copper. Runs before acceptance so the now-connected
    # power pads are reflected in the unconnected count.
    if cfg.get("power_plane_enabled", True):
        power_pour_start = time.monotonic()
        try:
            from kicraft.autoplacer.brain.gnd_pour import (
                pour_power_planes,
                repair_stranded_power,
            )

            _pwr = pour_power_planes(
                str(routed_board),
                cfg,
                layers=(cfg.get("power_plane_layer", "F.Cu"),),
            )
            if _pwr.get("nets"):
                print(
                    f"  Power plane: poured {_pwr['nets']} on "
                    f"{cfg.get('power_plane_layer', 'F.Cu')}"
                )
                # The pour fragments around foreign copper like the GND plane
                # does and can strand a supply pad on its own island
                # (fine-pitch parts especially); tie stranded clusters back
                # so they count as connected in the acceptance below.
                _rep = repair_stranded_power(str(routed_board), _pwr["nets"], cfg)
                if _rep.get("stranded"):
                    print(
                        f"  Power strand repair: {_rep['stranded']} stranded, "
                        f"{_rep['tied']} tied, {len(_rep['skipped'])} skipped"
                    )
        except Exception as exc:  # finishing step must never fail the leaf
            print(f"  WARNING: power plane step failed: {exc}")
        route_timing["power_pour_s"] = round(max(0.0, time.monotonic() - power_pour_start), 3)
    _finish_leaf_ground_planes(routed_board, cfg)
    from kicraft.autoplacer.brain.gnd_pour import finish_local_thermal_ground
    finish_local_thermal_ground(str(routed_board), cfg)
    _finish_leaf_current_landings(routed_board, cfg)


    routed_validation_start = time.monotonic()
    validation = validate_routed_board(
        str(routed_board),
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[port.name for port in extraction.interface_ports if port.required],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    # Always record the leaf's interface (inter-sheet) net names so the
    # unconnected-acceptance gate can exclude them: an interface net is routed
    # across the parent at compose, not within the leaf, so an unconnected item
    # on it must not count against the leaf (mirrors poured power/GND nets).
    # Previously set only on the reject path, leaving normal-path leaves blind.
    validation["interface_port_names"] = [port.name for port in extraction.interface_ports]
    validation["signal_pre_escape"] = _signal_escape
    validation["signal_finish"] = _signal_finish
    route_timing["routed_validation_s"] = round(
        max(0.0, time.monotonic() - routed_validation_start), 3
    )

    # The same accept-or-revert pass finishes local signals and configured
    # power nets, bounded by the existing leaf repair cap.
    _repair_cap = int(cfg.get("leaf_signal_repair_max_open_nets", 6))
    if (cfg.get("leaf_signal_repair_enabled", True) and _repair_cap > 0
            and (cfg.get("signal_unconnected_repair_enabled", True)
                 or cfg.get("power_strand_repair_enabled", True))):
        from kicraft.autoplacer.brain.leaf_acceptance import (
            acceptance_config_from_dict,
            split_unconnected_nets,
        )

        _local_open, _, _ = split_unconnected_nets(validation, acceptance_config_from_dict(cfg))
        _repair_open = set(_local_open) if cfg.get("signal_unconnected_repair_enabled", True) else set()
        if cfg.get("power_strand_repair_enabled", True):
            _power_nets = {str(n) for n in cfg.get("power_nets") or ()}
            _power_nets.discard(str(cfg.get("gnd_zone_net", "GND")))
            _repair_open.update(_power_nets.intersection(
                (validation.get("drc") or {}).get("unconnected_nets") or ()
            ))
        _repair_open = sorted(_repair_open)
        if _repair_open and len(_repair_open) <= _repair_cap:
            from kicraft.autoplacer.brain.unconnected_repair_pass import (
                attempt_unconnected_repair,
            )

            _repair_start = time.monotonic()
            print(
                f"  leaf unconnected repair: {len(_repair_open)} open net(s) "
                f"({', '.join(_repair_open)}) -- attempting guarded ties"
            )
            validation = attempt_unconnected_repair(
                routed_board,
                cfg,
                validation,
                anchor_names=[port.name for port in extraction.interface_ports],
                required_anchor_names=[
                    port.name for port in extraction.interface_ports if port.required
                ],
                label="leaf unconnected repair",
            )
            route_timing["leaf_signal_repair_s"] = round(
                max(0.0, time.monotonic() - _repair_start), 3
            )
        elif _repair_open:
            validation["unconnected_repair"] = {
                "ran": False,
                "reason": f"{len(_repair_open)}_open_nets_over_cap_{_repair_cap}",
            }
    validation["interface_port_names"] = [port.name for port in extraction.interface_ports]
    validation["signal_pre_escape"] = _signal_escape
    validation["signal_finish"] = _signal_finish
    _audit_interface_landings(routed_board, extraction, validation, cfg)
    if generate_diagnostics and render_intermediate:
        routed_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board),
            pre_route_validation=pre_route_validation,
            routed_validation=validation,
            pre_route_opts=LeafStageOpts(
                render_board_views=render_pre_route_board_views,
                write_drc_json=write_pre_route_drc_json,
                write_drc_report=write_pre_route_drc_report,
                render_drc_overlay=render_pre_route_drc_overlay,
            ),
            routed_opts=LeafStageOpts(
                render_board_views=render_routed_board_views,
                write_drc_json=write_routed_drc_json,
                write_drc_report=write_routed_drc_report,
                render_drc_overlay=render_routed_drc_overlay,
            ),
            build_contact_sheet=build_comparison_contact_sheet,
            quiet_render=fast_smoke_mode,
        )
        route_timing["routed_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - routed_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["routed_render_diagnostics_s"] = 0.0

    _routed_board_snapshot = promote_to_round_snapshot(routed_board, round_index)
    if _routed_board_snapshot is not None:
        round_board_routed = str(_routed_board_snapshot)

    # Round-snapshot paths surfaced to downstream consumers (types.py's
    # round_to_layout, the GUI's per-round scrubber via routing dict).
    # Empty string when this stage didn't produce that view, which the
    # consumers already handle as "no per-round preview" (gracefully fall
    # back to canonical render).
    round_preview_pre_route_front = ""
    round_preview_pre_route_back = ""
    round_preview_pre_route_copper = ""
    round_preview_routed_front = ""
    round_preview_routed_back = ""
    round_preview_routed_copper = ""

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        for _stage_key in ("pre_route", "routed"):
            _section = leaf_diagnostics.get(_stage_key, {})
            if not isinstance(_section, dict):
                continue
            _views = _section.get("board_views", {})
            if not isinstance(_views, dict):
                continue
            _paths = _views.get("paths", {})
            if not isinstance(_paths, dict):
                continue
            for _view in ("front_all", "back_all", "copper_both"):
                snap = promote_to_round_snapshot(_paths.get(_view), round_index)
                if snap is not None:
                    snap_str = str(snap)
                    _paths[f"round_{_view}"] = snap_str
                    if _stage_key == "pre_route":
                        if _view == "front_all":
                            round_preview_pre_route_front = snap_str
                        elif _view == "back_all":
                            round_preview_pre_route_back = snap_str
                        elif _view == "copper_both":
                            round_preview_pre_route_copper = snap_str
                    else:
                        if _view == "front_all":
                            round_preview_routed_front = snap_str
                        elif _view == "back_all":
                            round_preview_routed_back = snap_str
                        elif _view == "copper_both":
                            round_preview_routed_copper = snap_str

    validation["pre_route_validation"] = copy.deepcopy(pre_route_validation)
    validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    if round_board_pre_route:
        validation["round_board_pre_route"] = round_board_pre_route
    if round_board_routed:
        validation["round_board_routed"] = round_board_routed

    drc = validation.get("drc", {})
    drc_stdout = str(drc.get("stdout", ""))
    drc_stderr = str(drc.get("stderr", ""))
    _ = "\n".join(part for part in (drc_stdout, drc_stderr) if part.strip())

    # Post-route ignorable violation types: cosmetic issues and violations
    # that are inherent to the footprint or subcircuit outline, not caused
    # by the routing itself.
    ignorable_warning_types = {
        "silk_overlap",
        "lib_footprint_mismatch",
        "copper_edge_clearance",  # tight subcircuit outlines
        "silk_edge_clearance",  # cosmetic
        "silk_over_copper",  # cosmetic
        "solder_mask_bridge",  # footprint-internal
        "unconnected_items",  # the autorouter may not route all nets
    }
    significant_violations = [
        violation
        for violation in drc.get("violations", [])
        if violation.get("type") not in ignorable_warning_types
    ]

    # --- Generalized DRC exception: config-driven patterns ---
    # If the config provides ignorable_drc_patterns (list of regex strings),
    # check whether ALL significant violations match at least one pattern.
    ignorable_drc_patterns = cfg.get("ignorable_drc_patterns", [])
    _compiled_drc_patterns = [re.compile(p) for p in ignorable_drc_patterns]
    _all_match_config_patterns = (
        significant_violations
        and _compiled_drc_patterns
        and all(
            any(pat.search(v.get("description", "")) for pat in _compiled_drc_patterns)
            for v in significant_violations
        )
        and not drc.get("shorts", 0)
    )

    # --- Generalized DRC exception: footprint-baseline clearance heuristic ---
    # If ALL significant violations are clearance-type violations whose
    # descriptions reference pads from the SAME single footprint, treat them
    # as footprint-internal baseline clearance issues (e.g. dense USB-C,
    # fine-pitch IC pads closer together than the board clearance rule).
    _footprint_ref_re = re.compile(r"\bof\s+(\S+)")
    _clearance_types = {"clearance", "hole_clearance", "solder_mask_bridge"}
    _all_clearance = (
        significant_violations
        and all(v.get("type") in _clearance_types for v in significant_violations)
        and not drc.get("shorts", 0)
    )
    _single_footprint_baseline = False
    _baseline_footprint_ref = None
    if _all_clearance:
        # Collect all footprint references mentioned across violations
        _violation_footprint_refs: set[str] = set()
        for v in significant_violations:
            desc = v.get("description", "")
            for m in _footprint_ref_re.finditer(desc):
                _violation_footprint_refs.add(m.group(1))
        # If every violation references pads from exactly one footprint,
        # this is a footprint-internal clearance issue.
        if len(_violation_footprint_refs) == 1:
            _single_footprint_baseline = True
            _baseline_footprint_ref = next(iter(_violation_footprint_refs))

    if _all_match_config_patterns or _single_footprint_baseline:
        _ignore_reason = (
            "config_ignorable_drc_patterns"
            if _all_match_config_patterns
            else f"footprint_baseline_clearance:{_baseline_footprint_ref}"
        )
        _ignored_types = {v.get("type") for v in significant_violations}
        validation["obviously_illegal_routed_geometry"] = False
        validation["rejection_reasons"] = [
            reason
            for reason in validation.get("rejection_reasons", [])
            if reason != "illegal_routed_geometry"
        ]
        validation["accepted"] = not validation["rejection_reasons"]
        validation["drc"]["ignored_violation_types"] = sorted(
            ignorable_warning_types | _ignored_types
        )
        validation["drc"]["ignored_violation_count"] = len(drc.get("violations", []))
        validation["drc"]["significant_violation_count"] = 0
        validation["drc"]["ignored_clearance_reason"] = _ignore_reason

    _validate_leaf_current_contracts(routed_board, cfg, validation)
    _validate_leaf_thermal_requirements(routed_board, cfg, validation)

    routed_internal, failed_internal = _internal_net_verdict(validation, extraction)

    accepted = bool(validation.get("accepted", False))
    if not accepted:
        validation["accepted"] = False
        validation["rejected"] = True
        validation["rejection_stage"] = "leaf_routed_artifact_validation"
        validation["routed_board_path"] = str(routed_board)
        validation["leaf_placed_board"] = str(pre_route_board)
        validation["router"] = "kicad-routing-tools"
        validation["internal_net_names"] = list(sorted(extraction.internal_net_names))
        validation["interface_port_names"] = [port.name for port in extraction.interface_ports]
        validation["imported_copper_summary"] = {
            "trace_count": int(imported_copper.get("trace_count", 0)),
            "via_count": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
        }
        validation["routing_stats"] = copy.deepcopy(routing_stats)
        validation["rejection_message"] = "Leaf routed artifact rejected: " + ",".join(
            validation.get("rejection_reasons", [])
        )
        print("  Routed DRC rejected placement: " + validation["rejection_message"])
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "routed_drc_rejection",
                "router": "kicad-routing-tools",
                "traces": int(imported_copper.get("trace_count", 0)),
                "vias": int(imported_copper.get("via_count", 0)),
                "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
                "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
                "round_board_pre_route": round_board_pre_route,
                "round_board_routed": round_board_routed,
                "routed_internal_nets": routed_internal,
                "failed_internal_nets": failed_internal,
                "_trace_segments": [],
                "_via_objects": [],
                "validation": copy.deepcopy(validation),
                "routing_stats": copy.deepcopy(routing_stats),
                "render_diagnostics": copy.deepcopy(leaf_diagnostics),
                "routed_board_path": str(routed_board),
                "leaf_placed_board": str(pre_route_board),
                "round_preview_pre_route_front": round_preview_pre_route_front,
                "round_preview_pre_route_back": round_preview_pre_route_back,
                "round_preview_pre_route_copper": round_preview_pre_route_copper,
                "round_preview_routed_front": round_preview_routed_front,
                "round_preview_routed_back": round_preview_routed_back,
                "round_preview_routed_copper": round_preview_routed_copper,
                "failed": True,
            },
            route_timing,
        )

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )
    return (
        {
            "enabled": True,
            "skipped": False,
            "reason": "",
            "router": "kicad-routing-tools",
            "traces": int(imported_copper.get("trace_count", 0)),
            "vias": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
            "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": routed_internal,
            "failed_internal_nets": failed_internal,
            "_trace_segments": [
                copy.deepcopy(trace) for trace in imported_copper.get("traces", [])
            ],
            "_via_objects": [copy.deepcopy(via) for via in imported_copper.get("vias", [])],
            "routing_stats": routing_stats,
            "validation": validation,
            "gnd_pour_summary": copy.deepcopy(gnd_pour_summary),
            "render_diagnostics": copy.deepcopy(leaf_diagnostics),
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "leaf_placed_board": str(pre_route_board),
            "round_preview_pre_route_front": round_preview_pre_route_front,
            "round_preview_pre_route_back": round_preview_pre_route_back,
            "round_preview_pre_route_copper": round_preview_pre_route_copper,
            "round_preview_routed_front": round_preview_routed_front,
            "round_preview_routed_back": round_preview_routed_back,
            "round_preview_routed_copper": round_preview_routed_copper,
            "failed": False,
        },
        route_timing,
    )


def _place_fabricated_edge_interfaces(board_state: BoardState, cfg: dict[str, Any]) -> None:
    """Apply exact edge/pitch datums after trivial-leaf outline reframing."""
    interfaces = cfg.get("edge_interfaces", []) or []
    if not interfaces:
        return
    groups = []
    seen: set[str] = set()
    for interface in interfaces:
        refs = [str(ref) for ref in interface.get("refs", [])]
        members = [board_state.components.get(ref) for ref in refs]
        side = str(interface.get("side", ""))
        pitch = float(interface.get("pitch_mm", 0.0))
        if (
            not refs
            or side not in {"left", "right", "top", "bottom"}
            or pitch <= 0
            or any(member is None or not member.pads for member in members)
        ):
            continue
        for ref in refs:
            if ref in seen:
                raise ValueError(f"edge interface reference declared more than once: {ref}")
            seen.add(ref)
        groups.append((members, side, pitch))
    tl, br = board_state.board_outline
    margin = float(cfg.get("edge_margin_mm", 2.0))
    width, height = br.x - tl.x, br.y - tl.y
    for members, side, pitch in groups:
        span = (len(members) - 1) * pitch
        if side in {"left", "right"}:
            height = max(height, span + 2 * margin)
        else:
            width = max(width, span + 2 * margin)
    center_x, center_y = (tl.x + br.x) / 2, (tl.y + br.y) / 2
    tl = Point(center_x - width / 2, center_y - height / 2)
    br = Point(center_x + width / 2, center_y + height / 2)
    for members, side, pitch in groups:
        span = (len(members) - 1) * pitch
        start = (tl.y + br.y - span) / 2 if side in {"left", "right"} else (tl.x + br.x - span) / 2
        for index, member in enumerate(members):
            pad = member.pads[0]
            target = Point(
                tl.x if side == "left" else br.x if side == "right" else start + index * pitch,
                tl.y if side == "top" else br.y if side == "bottom" else start + index * pitch,
            )
            dx, dy = target.x - pad.pos.x, target.y - pad.pos.y
            member.pos = Point(member.pos.x + dx, member.pos.y + dy)
            if member.body_center is not None:
                member.body_center = Point(member.body_center.x + dx, member.body_center.y + dy)
            for member_pad in member.pads:
                member_pad.pos = Point(member_pad.pos.x + dx, member_pad.pos.y + dy)
            member.locked = True
    board_state.board_outline = (tl, br)


def _stamp_trivial_leaf(
    *,
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    round_index: int | None,
    generate_diagnostics: bool,
    render_intermediate: bool,
    fast_smoke_mode: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Stamp a placed-but-not-routed PCB for a leaf with no internal nets.

    A trivial leaf (e.g. a battery holder with both terminals exposed
    via interface ports) has nothing to route, but we still want a
    real ``leaf_routed.kicad_pcb`` on disk so:

    * ``pin_best_leaves`` can promote a chosen round like every other
      leaf (no more "no-snapshots" status).
    * The GUI snapshot picker can show this leaf's rounds and let the
      user pin keyboard-style.
    * The composer's blocker extraction reads the same way from
      every leaf.

    The autorouter is skipped, but guarded interface escapes may add copper.
    The routed board and returned copper describe those reservations.
    """
    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = Path(artifact_paths.artifact_dir) / "leaf_placed.kicad_pcb"
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"

    legality_start = time.monotonic()
    repaired_components, legality_repair = repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(max(0.0, time.monotonic() - legality_start), 3)

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = Path(extraction.subcircuit.schematic_path).with_suffix(".kicad_pcb")
    if not source_pcb.exists():
        # No internal nets removes the need for routing, not for a real board.
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "missing_source_board",
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": [],
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "reason": "missing_source_board",
                    "board_exists": False,
                },
                "failed": True,
            },
            route_timing,
        )

    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    PlacementSolver(extraction.local_state, cfg)._place_local_bypasses(repaired_components)

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []
    # Trivial leaves skip the autorouter, so the pre-route stamp is also the final
    # stamp. Apply the same shrink-and-center as the main-path silk re-stamp so the
    # rounded silk hugs Edge.Cuts and the standalone leaf opens centered on its A4
    # page (the parent composer re-bases each leaf on load, so this is placement-safe).
    antenna_edge_intents = getattr(route_input_board, "antenna_edge_intents", None)
    if antenna_edge_intents is None:
        _new_outline = _outline_around_geometry(route_input_board.components, cfg)
    else:
        _new_outline = _outline_around_geometry(
            route_input_board.components,
            cfg,
            antenna_edge_intents=antenna_edge_intents,
        )
    if _new_outline is not None:
        _new_tl, _new_br = _new_outline
        _delta, _centered_outline = _center_on_leaf_page(_new_tl, _new_br, cfg)
        if abs(_delta.x) > 1e-6 or abs(_delta.y) > 1e-6:
            from kicraft.autoplacer.brain.leaf_geometry import (
                copy_components_with_translation,
            )

            route_input_board.components = copy_components_with_translation(
                route_input_board.components, _delta
            )
        route_input_board.board_outline = _centered_outline
    _place_fabricated_edge_interfaces(route_input_board, cfg)
    route_input_board.silkscreen = _silk_for_leaf(extraction, route_input_board.components, cfg)

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    route_timing["stamp_pre_route_board_s"] = round(max(0.0, time.monotonic() - stamp_start), 3)

    from kicraft.autoplacer.kicad_routing_tools import (
        _board_has_assembly_pads,
        _propagate_sibling_project_rules,
        _reject_drills_in_assembly_openings,
    )

    if _board_has_assembly_pads(pre_route_board):
        _reject_drills_in_assembly_openings(pre_route_board)

    # No autorouter is needed, but unrouted copper still has to pass real DRC.
    _propagate_sibling_project_rules(str(source_pcb), str(pre_route_board))
    _ground_marker, _ = _prepare_leaf_ground_reservations(pre_route_board, source_pcb, cfg)
    _signal_escape = _reserve_signal_escapes(pre_route_board, extraction, cfg)
    shutil.copy2(pre_route_board, routed_board)
    _propagate_sibling_project_rules(str(pre_route_board), str(routed_board))
    if _ground_marker:
        from kicraft.autoplacer.kicad_routing_tools import _strip_router_keepouts
        _strip_router_keepouts(routed_board, _ground_marker)
        _strip_router_keepouts(pre_route_board, _ground_marker)
    _signal_finish = _finish_interface_landings(routed_board, extraction, cfg)
    if _signal_finish["stubs"]:
        import pcbnew
        _filled_board = pcbnew.LoadBoard(str(routed_board))
        pcbnew.ZONE_FILLER(_filled_board).Fill(_filled_board.Zones())
        _filled_board.Save(str(routed_board))
    _finish_leaf_ground_planes(routed_board, cfg)
    from kicraft.autoplacer.brain.gnd_pour import finish_local_thermal_ground
    finish_local_thermal_ground(str(routed_board), cfg)
    _finish_leaf_current_landings(routed_board, cfg)
    imported_copper = import_routed_copper(str(routed_board))
    validation_start = time.monotonic()
    validation = validate_routed_board(
        str(routed_board),
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[port.name for port in extraction.interface_ports if port.required],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    validation["interface_port_names"] = [port.name for port in extraction.interface_ports]
    validation["signal_pre_escape"] = _signal_escape
    validation["signal_finish"] = _signal_finish
    _audit_interface_landings(routed_board, extraction, validation, cfg)
    _validate_leaf_current_contracts(routed_board, cfg, validation)
    _validate_leaf_thermal_requirements(routed_board, cfg, validation)
    route_timing["routed_validation_s"] = round(
        max(0.0, time.monotonic() - validation_start), 3
    )

    round_board_pre_route = ""
    round_board_routed = ""
    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"
        for src_path, suffix in (
            (pre_route_board, "leaf_placed"),
            (routed_board, "leaf_routed"),
        ):
            if not src_path.exists():
                continue
            dst = src_path.parent / f"{round_prefix}_{suffix}{src_path.suffix}"
            shutil.copy2(src_path, dst)
            _propagate_sibling_project_rules(str(src_path), str(dst))
            if suffix == "leaf_placed":
                round_board_pre_route = str(dst)
            else:
                round_board_routed = str(dst)

    diagnostics_payload: dict[str, Any]
    if generate_diagnostics and render_intermediate and not fast_smoke_mode:
        try:
            diagnostic_opts = LeafStageOpts(
                render_board_views=True,
                write_drc_json=True,
                write_drc_report=True,
                render_drc_overlay=True,
            )
            diagnostics_payload = generate_leaf_diagnostic_artifacts(
                artifact_dir=artifact_paths.artifact_dir,
                pre_route_board=str(pre_route_board),
                routed_board=str(routed_board),
                pre_route_validation=validation,
                routed_validation=validation,
                pre_route_opts=diagnostic_opts,
                routed_opts=diagnostic_opts,
                build_contact_sheet=False,
                quiet_render=fast_smoke_mode,
            )
        except Exception as exc:
            diagnostics_payload = {"skipped": True, "reason": f"diag_failed:{exc}"}
    else:
        diagnostics_payload = {"skipped": True, "reason": "fast_smoke_or_no_diag"}

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )

    return (
        {
            "enabled": True,
            "skipped": True,
            "reason": (
                "current_contract_rejection"
                if validation.get("current_contract_errors")
                else "no_internal_nets"
            ),
            "router": "kicad-routing-tools",
            "traces": int(imported_copper.get("trace_count", 0)),
            "vias": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
            "round_board_illegal_pre_stamp": "",
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": [],
            "failed_internal_nets": [],
            "_trace_segments": [copy.deepcopy(t) for t in imported_copper.get("traces", [])],
            "_via_objects": [copy.deepcopy(v) for v in imported_copper.get("vias", [])],
            "validation": validation,
            "render_diagnostics": diagnostics_payload,
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "leaf_placed_board": str(pre_route_board),
            "failed": not bool(validation.get("accepted", False)),
        },
        route_timing,
    )
