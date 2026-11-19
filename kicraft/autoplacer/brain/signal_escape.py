"""Pre-route escape reservations for dense and cross-leaf signal pads.

The shared stamper remains the authority for pad, copper, hole, mask, edge
and keepout clearance before any of these dog-bones is written.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pcbnew

from kicraft.autoplacer.brain.breakout_stubs import (
    BreakoutSpec,
    STAMP_CLEARANCE_GUARD_MM,
    _own_clearance_mm,
)
from kicraft.autoplacer.fab_profile import fab_floors, fanout_via
from kicraft.design.models import is_power_or_ground_name

_MIN_PAD_FIELD_SIZE = 8
_ESCAPE_LENGTH_MM = 1.0
_NATIVE_GEOMETRY_TOL_MM = 0.002


def _escape_axis(footprint: Any, pad: Any) -> tuple[float, float]:
    """Leave a rectangular pad along its outward long axis, not across its row."""
    centre = footprint.GetPosition()
    pos = pad.GetPosition()
    x_mm, y_mm = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
    dx = x_mm - pcbnew.ToMM(centre.x)
    dy = y_mm - pcbnew.ToMM(centre.y)
    size = pad.GetSize()
    if size.x != size.y:
        angle = math.radians(float(pad.GetOrientationDegrees()))
        ux, uy = (
            (math.cos(angle), -math.sin(angle))
            if size.x > size.y else (math.sin(angle), math.cos(angle))
        )
        if dx * ux + dy * uy < 0:
            ux, uy = -ux, -uy
    else:
        norm = math.hypot(dx, dy)
        ux, uy = (1.0, 0.0) if norm < 1e-6 else (dx / norm, dy / norm)
    return ux, uy


def _staggered_fanout_paths(
    footprint: Any,
    pads: list[Any],
    *,
    via_size_mm: float,
    width_mm: float,
    clearance_mm: float,
    interface_widths: dict[tuple, float],
) -> list[tuple[Any, list[tuple[float, float]], tuple[tuple[float, float], float] | None]]:
    """Fan tracks before staggering vias; never discard an alternate signal.

    At larger netclass clearances a far-row track cannot pass a near-row via
    on the original pad pitch. Spread the tracks only after they leave the
    pad row, using the same resolved clearance as the shared stamper.
    """
    guard = STAMP_CLEARANCE_GUARD_MM
    local_lane_mm = via_size_mm / 2 + width_mm / 2 + clearance_mm + guard
    rows: dict[tuple[float, float], list[tuple[Any, float, float, float]]] = {}
    for pad in pads:
        ux, uy = _escape_axis(footprint, pad)
        position, size = pad.GetPosition(), pad.GetSize()
        x, y = pcbnew.ToMM(position.x), pcbnew.ToMM(position.y)
        key = (round(ux, 9), round(uy, 9))
        rows.setdefault(key, []).append((
            pad, x * ux + y * uy, -x * uy + y * ux,
            max(pcbnew.ToMM(size.x), pcbnew.ToMM(size.y)) / 2,
        ))
    plans: dict[tuple, tuple[list[tuple[float, float]], tuple[tuple[float, float], float] | None]] = {}
    selected_tips: list[tuple[float, float]] = []
    for (ux, uy), row in sorted(rows.items()):
        row.sort(key=lambda item: item[2])
        original = [item[2] for item in row]
        pitches = [b - a for a, b in zip(original, original[1:])]
        launch_width = max((interface_widths.get(_pad_identity(pad), 0.0)
                            for pad, *_ in row), default=0.0)
        # Public launches must admit the parent's normal via envelope, not
        # merely the smaller fabrication-profile escape via. Otherwise the
        # parent's multipoint anchor can be caged between sibling barrels.
        access_via_mm = max(
            (pcbnew.ToMM(pad.GetNet().GetNetClassSlow().GetViaDiameter())
             for pad, *_ in row if _pad_identity(pad) in interface_widths),
            default=via_size_mm,
        )
        via_pitch = max(via_size_mm, access_via_mm) + clearance_mm + guard
        open_row = bool(launch_width and pitches and min(pitches) < via_pitch)
        lane_mm = max(via_pitch, (via_size_mm + launch_width) / 2 + clearance_mm + guard,
                      launch_width + clearance_mm + guard) if open_row else local_lane_mm
        tip_spacing_mm = via_pitch if open_row else 2 * lane_mm
        expanded = [original[0]]
        for tangent in original[1:]:
            expanded.append(max(tangent, expanded[-1] + lane_mm))
        shift = (sum(expanded) - sum(original)) / len(row)
        expanded = [tangent - shift for tangent in expanded]
        max_shift = max(abs(a - b) for a, b in zip(original, expanded, strict=True))
        lead_plane = max(normal + half_pad for _, normal, _, half_pad in row)
        lead_plane += clearance_mm + width_mm / 2 + guard
        # Limit the diagonal slope so its perpendicular track spacing stays
        # legal, not just the separation measured along the pad row.
        track_lane = width_mm + clearance_mm + guard
        slope = min(1.0, math.sqrt(max(0.0, (min(pitches) / track_lane) ** 2 - 1))) if pitches else 1.0
        if max_shift > 1e-9 and slope <= 1e-9:
            # An already illegal row cannot be repaired by fanning tracks.
            # Leave the original pitch for the stamper to reject explicitly.
            expanded, max_shift = original, 0.0
        fan_plane = max(
            max(normal for _, normal, _, _ in row) + _ESCAPE_LENGTH_MM,
            lead_plane + (max_shift / slope if max_shift else 0.0) + guard,
        )
        def point(normal, lateral):
            return ux * normal - uy * lateral, uy * normal + ux * lateral

        row_tips = []
        for (pad, _normal, tangent, _half_pad), target in zip(row, expanded, strict=True):

            lead, fan = point(lead_plane, tangent), point(fan_plane, target)
            # Keep near-row vias beyond the diagonal fanout segments, whose
            # perpendicular distance can be smaller than the final lane pitch.
            end_plane = fan_plane + lane_mm
            while True:
                tip = point(end_plane, target)
                if all(math.dist(tip, other) >= tip_spacing_mm - 1e-8 for other in selected_tips):
                    break
                end_plane += tip_spacing_mm
            waypoints = [lead, fan, tip]
            row_tips.append((pad, target, end_plane, waypoints))
            selected_tips.append(tip)
        # Put sibling launches beyond the whole via row, including vias
        # shifted to avoid a neighboring row on another side of a package.
        launch_plane = max(end for _, _, end, _ in row_tips) + lane_mm
        for pad, target, _, waypoints in row_tips:
            identity = _pad_identity(pad)
            access_width = interface_widths.get(identity, 0.0)
            via_exit = (point(launch_plane, target), access_width) if open_row and access_width else None
            plans[identity] = waypoints, via_exit
    return [(pad, *plans[_pad_identity(pad)]) for pad in sorted(pads, key=_pad_number_key)]


def _pad_identity(pad: Any) -> tuple:
    """Stable physical identity across separate SWIG ``Pads()`` wrappers."""
    try:
        uuid = pad.m_Uuid.AsString()
        if uuid and uuid != "00000000-0000-0000-0000-000000000000":
            return ("uuid", uuid)
    except (AttributeError, TypeError):
        pass
    pos = pad.GetPosition()
    return ("physical", pad.GetNumber(), pos.x, pos.y)


def _pad_number_key(pad: Any) -> tuple[int, int | str]:
    """Natural pad-number ordering, with deterministic connector-label fallback."""
    number = pad.GetNumber()
    return (0, int(number)) if number.isdigit() else (1, number)


def declared_usb_nets(cfg: dict[str, Any]) -> set[str]:
    """Exact pair nets reserved for the differential owner, never name guesses."""
    from kicraft.design.models import USBRoutingContract

    return {
        name
        for raw in cfg.get("usb_routing_contracts", [])
        for segment in USBRoutingContract.model_validate(raw).segments
        for name in (segment.positive_net, segment.negative_net)
    }


def signal_escape_specs(
    board: Any,
    cfg: dict[str, Any] | None = None,
    external_net_names: set[str] | None = None,
) -> list[BreakoutSpec]:
    """Return guarded dog-bones for SMD signal fields and cross-leaf signals.

    The specs use the fabrication-profile fanout via, not the router's normal
    netclass via.  The latter is intentionally 0.6/0.3 mm and does not fit the
    fine-pitch geometry this reservation addresses.  No spec is an exemption:
    the shared breakout stamper drops one unless its complete path, tip via,
    drilled-hole spacing, rule areas and already-reserved copper all clear.
    Each stub uses the source pad's copper face, including bottom-side pads;
    a pad with no outer copper layer has no supported escape.
    """
    cfg = cfg or {}
    via_size_mm, via_drill_mm = fanout_via(cfg)
    floors = fab_floors(cfg)
    width_mm = floors["track_mm"]
    power_nets = set(cfg.get("power_nets", ()))
    external = set(external_net_names or ())
    external_pad_counts = Counter(
        pad.GetNetname() for fp in board.GetFootprints() for pad in fp.Pads()
        if pad.GetNetname() in external
    )
    paired_nets = declared_usb_nets(cfg)
    specs: list[BreakoutSpec] = []
    for footprint in sorted(board.GetFootprints(), key=lambda fp: fp.GetReferenceAsString()):
        # Wider local output pads can be caged by the same earlier power
        # copper as fine-pitch inputs. Reserve the whole signal pad field,
        # not just external nets or an arbitrary pitch/width subset.
        is_pad_field = sum(
            pad.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
            for pad in footprint.Pads()
        ) >= _MIN_PAD_FIELD_SIZE
        candidates = []
        pad_layers = {}
        for pad in footprint.Pads():
            if pad.GetNetname() in power_nets or pad.GetNetname() in paired_nets:
                continue
            is_signal = (
                pad.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
                and bool(pad.GetNetname())
                and not is_power_or_ground_name(pad.GetNetname())
            )
            if not is_signal or (
                not is_pad_field and pad.GetNetname() not in external
            ):
                continue
            if pad.IsOnLayer(pcbnew.F_Cu):
                layer, layer_name = pcbnew.F_Cu, "F.Cu"
            elif pad.IsOnLayer(pcbnew.B_Cu):
                layer, layer_name = pcbnew.B_Cu, "B.Cu"
            else:
                continue
            candidates.append(pad)
            pad_layers[_pad_identity(pad)] = (layer, layer_name)
        clearance_mm = max(
            floors["clearance_mm"],
            *(_own_clearance_mm(pad, pad_layers[_pad_identity(pad)][0], floors["clearance_mm"])
              for pad in candidates),
        ) if candidates else floors["clearance_mm"]
        interface_widths = {
            _pad_identity(pad): max(width_mm, pcbnew.ToMM(pad.GetNet().GetNetClassSlow().GetTrackWidth()))
            for pad in candidates
            if is_pad_field and external_pad_counts[pad.GetNetname()] == 1
        }
        for pad, waypoints, via_exit in _staggered_fanout_paths(
            footprint, candidates, via_size_mm=via_size_mm,
            width_mm=width_mm, clearance_mm=clearance_mm, interface_widths=interface_widths,
        ):
            pos = pad.GetPosition()
            specs.append(
                BreakoutSpec(
                    ref=footprint.GetReferenceAsString(),
                    pad=pad.GetNumber(),
                    near_xy=(pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)),
                    waypoints=waypoints,
                    width_mm=width_mm,
                    layer=pad_layers[_pad_identity(pad)][1],
                    via_at_end=True,
                    strict_same_footprint=True,
                    via_size_mm=via_size_mm,
                    via_drill_mm=via_drill_mm,
                    short_fanout_fallback=via_exit is None,
                    via_exit=via_exit,
                )
            )
    return specs


def _escape_track_record(board, track) -> dict:
    def point(p):
        return pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)

    return {
        "uuid": track.m_Uuid.AsString(), "kind": "track", "net": track.GetNetname(),
        "layer": board.GetLayerName(track.GetLayer()),
        "start_mm": point(track.GetStart()), "end_mm": point(track.GetEnd()),
        "width_mm": pcbnew.ToMM(track.GetWidth()),
    }


def _source_join(board, connectivity, branch, tracks, owned_ids, floor):
    """Plan a canonical replacement for a terminal that only joins one face."""
    from kicraft.autoplacer.hardware.local_contract import _position, _via_width

    before, terminal = branch["items"][-2:]
    if before["kind"] != "track":
        return None
    predecessor, item = (tracks.get(row["uuid"]) for row in (before, terminal))
    if (predecessor is None or item is None or predecessor.GetClass() != "PCB_TRACK"
            or predecessor.GetNetCode() != item.GetNetCode()
            or connectivity.GetConnectedPads(item)):
        return None
    prior = _escape_track_record(board, predecessor)
    if (prior != {**before, "start_mm": tuple(before["start_mm"]),
                  "end_mm": tuple(before["end_mm"])}):
        return None
    neighbors = {t.m_Uuid.AsString(): t for t in connectivity.GetConnectedTracks(item)}
    if len(neighbors) != 2 or predecessor.m_Uuid.AsString() not in neighbors:
        return None
    router_id, router = next((uid, t) for uid, t in neighbors.items() if uid != before["uuid"])
    layer = predecessor.GetLayer()
    if (router_id in owned_ids or router.IsLocked() or router.GetClass() != "PCB_TRACK"
            or router.GetNetCode() != item.GetNetCode() or router.GetLayer() != layer):
        return None
    start = _position(predecessor.GetEnd())
    width = predecessor.GetWidth()
    router_ends = tuple(_position(p) for p in (router.GetStart(), router.GetEnd()))
    if item.GetClass() == "PCB_VIA" and terminal["kind"] == "via":
        center = _position(item.GetPosition())
        via_width = _via_width(item, layer)
        if (not item.IsOnLayer(layer) or center != start or terminal["net"] != item.GetNetname()
                or center != tuple(round(v * 1e6) for v in terminal["position_mm"])
                or via_width != round(terminal["size_mm"] * 1e6)
                or item.GetDrillValue() != round(terminal["drill_mm"] * 1e6)):
            return None
        # The remaining layer has only two contacts. The two track caps
        # cannot touch directly, but the router endpoint is in the annulus
        # expanded by its endcap. Replace that same-face bridge with copper.
        drill = item.GetDrillValue()
        destinations = [
            end for end in router_ends
            if max(width + router.GetWidth(), drill) < 2 * math.dist(start, end)
            < via_width + router.GetWidth()
        ]
    elif item.GetClass() == "PCB_TRACK" and terminal["kind"] == "track":
        original = _escape_track_record(board, item)
        if (original != {**terminal, "start_mm": tuple(terminal["start_mm"]),
                         "end_mm": tuple(terminal["end_mm"])}
                or item.GetLayer() != layer or _position(item.GetStart()) != start):
            return None
        width = item.GetWidth()
        end = _position(item.GetEnd())
        dx, dy = end[0] - start[0], end[1] - start[1]
        span = dx * dx + dy * dy
        destinations = []
        for target in router_ends:
            x, y = target[0] - start[0], target[1] - start[1]
            along, cross = x * dx + y * dy, x * dy - y * dx
            # This landing must be on the interior of the existing copper,
            # not a pad-side tail or an extrapolated repair across a gap.
            if 0 < along < span and 4 * cross * cross <= width * width * span:
                destinations.append(target)
    else:
        return None
    if width < floor or len(destinations) != 1:
        return None
    return item, start, destinations[0], layer, width


def _finish_owned_escape_landings(
    board_path: Path, summaries: tuple[dict, ...], eligible_nets: set[str], cfg: dict,
) -> dict:
    """Keep native removal and replacement lifetimes in an isolated process."""
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    settings = {key: cfg[key] for key in
                ("usb_routing_contracts", "local_bypasses", "fab_capability") if key in cfg}
    settings["power_nets"] = sorted(cfg.get("power_nets", ()))
    with tempfile.TemporaryDirectory(dir=board_path.parent, prefix=".signal-join-proof-") as directory:
        request, result = Path(directory) / "request.json", Path(directory) / "result.json"
        request.write_text(json.dumps({
            "summaries": summaries, "eligible_nets": sorted(eligible_nets), "cfg": settings,
        }))
        run_pcbnew_script(
            "import json\nfrom pathlib import Path\n"
            "from kicraft.autoplacer.brain.signal_escape import _finish_owned_escape_landings_native\n"
            f"request = json.loads(Path({str(request)!r}).read_text())\n"
            f"proof = _finish_owned_escape_landings_native(Path({str(board_path)!r}), "
            "tuple(request['summaries']), set(request['eligible_nets']), request['cfg'])\n"
            f"Path({str(result)!r}).write_text(json.dumps(proof))\n"
        )
        return json.loads(result.read_text())


def _finish_owned_escape_landings_native(
    board_path: Path, summaries: tuple[dict, ...], eligible_nets: set[str], cfg: dict,
) -> dict:
    """Normalize DRC-named joins only on explicitly authorized compiler nets."""
    from kicraft.autoplacer.brain.escape_cleanup import (
        _copy_native_rules, _drc_item_key, _partition, _report_drc,
    )
    from kicraft.autoplacer.hardware.local_contract import _copper_graph, _position, _via_width

    branches = [
        branch for summary in summaries for branch in summary.get("owned_signal_escapes", ())
        if branch["owner"] == "compiler.signal_escape"
    ]
    owned_ids = Counter(row["uuid"] for branch in branches for row in branch["items"])
    eligible = [
        branch for branch in branches if len(branch["items"]) >= 2
        and branch["items"][-1]["net"] in eligible_nets
        and all(owned_ids[row["uuid"]] == 1 for row in branch["items"])
    ]
    if not eligible:
        return {}
    before_sha = hashlib.sha256(board_path.read_bytes()).hexdigest()
    board = pcbnew.LoadBoard(str(board_path))
    if board is None:
        raise ValueError("could not load native source for signal landing correction")
    layers = tuple(board.GetEnabledLayers().CuStack())
    if len(layers) != 2:
        return {}
    protected = {zone.GetNetname() for zone in board.Zones() if not zone.GetIsRuleArea()}
    protected.update(cfg.get("power_nets", ()))
    protected.update(declared_usb_nets(cfg))
    for contract in cfg.get("local_bypasses", ()):
        if contract.get("ground_return"):
            protected.add(contract["ground_return"]["net"])
        if "max_route_length_mm" not in contract:
            continue
        owner = board.FindFootprintByReference(contract.get("route_anchor_ref", contract["owner_ref"]))
        member = board.FindFootprintByReference(contract["bypass_ref"])
        if owner is not None and member is not None:
            for owner_pin, member_pin in contract.get("pin_pairs") or (contract["power_pins"],):
                for footprint, number in ((owner, contract.get("route_anchor_pin", owner_pin)),
                                          (member, member_pin)):
                    protected.update(pad.GetNetname() for pad in footprint.Pads()
                                     if str(pad.GetNumber()) == str(number))

    board.BuildConnectivity()
    connectivity = board.GetConnectivity()
    tracks = {item.m_Uuid.AsString(): item for item in board.GetTracks()}
    floor = round(fab_floors(cfg)["track_mm"] * 1e6)
    plans = {}
    terminals = {}
    for branch in eligible:
        terminal = branch["items"][-1]
        net = terminal["net"]
        if net in protected or is_power_or_ground_name(net):
            continue
        plan = _source_join(board, connectivity, branch, tracks, owned_ids, floor)
        if plan is not None:
            plans[terminal["uuid"]] = plan
            terminals[terminal["uuid"]] = terminal
    if not plans:
        return {}

    def copper_index(native):
        index, anchors = {}, {}
        for item in native.GetTracks():
            uid, kind, net = item.m_Uuid.AsString(), item.GetClass(), item.GetNetname()
            if uid in index:
                raise ValueError("ambiguous native signal escape UUID")
            points = []
            if kind == "PCB_VIA":
                point = _position(item.GetPosition())
                stack = tuple((layer, _via_width(item, layer)) for layer in layers if item.IsOnLayer(layer))
                geometry = (point, stack, item.GetDrillValue(), item.GetViaType())
                points.extend((point, layer) for layer, _ in stack)
            else:
                layer = item.GetLayer()
                ends = tuple(sorted((_position(item.GetStart()), _position(item.GetEnd()))))
                midpoint = _position(item.GetMid()) if kind == "PCB_ARC" else None
                geometry = (ends, layer, item.GetWidth(), midpoint)
                points.extend((point, layer) for point in ends)
            index[uid] = (kind, net, item.IsLocked(), geometry)
            anchors[uid] = (net, points)
        return index, anchors

    with tempfile.TemporaryDirectory(dir=board_path.parent, prefix=".signal-landings-") as directory:
        stage = Path(directory)
        candidate = stage / board_path.name
        shutil.copyfile(board_path, candidate)
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != before_sha:
            raise ValueError("source changed while staging signal landing correction")
        _copy_native_rules(board_path, candidate)
        before_report = _report_drc(candidate, stage / "before.json")
        names = {"PCB_TRACK": "track_dangling", "PCB_VIA": "via_dangling"}
        selected = {}
        selected_drc = set()
        for row in before_report["violations"]:
            if row["severity"] not in ("warning", "error") or len(row["items"]) != 1:
                continue
            uid = row["items"][0]["uuid"]
            if uid in plans and row["type"] == names[plans[uid][0].GetClass()]:
                selected[uid] = plans[uid]
                selected_drc.add(_drc_item_key(row))
        if not selected:
            return {}

        original, _ = copper_index(board)
        affected = {original[uid][1] for uid in selected}
        graphs = {net: _copper_graph(board, net) for net in affected}
        expected = {uid: key for uid, key in original.items() if uid not in selected}
        replacements = {}
        for uid, (old, start, end, layer, width) in selected.items():
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(pcbnew.VECTOR2I(*start))
            track.SetEnd(pcbnew.VECTOR2I(*end))
            track.SetLayer(layer)
            track.SetWidth(width)
            track.SetNetCode(old.GetNetCode())
            track.SetLocked(old.IsLocked())
            row = _escape_track_record(board, track)
            replacements[uid] = row
            expected[row["uuid"]] = (
                "PCB_TRACK", row["net"], old.IsLocked(),
                (tuple(sorted((start, end))), layer, width, None),
            )
            board.Add(track)
        for old, *_ in selected.values():
            board.Remove(old)
        if board.Save(str(candidate)) is False:
            raise RuntimeError("could not save native signal landing correction")

        saved = pcbnew.LoadBoard(str(candidate))
        if saved is None:
            raise ValueError("could not reload native signal landing correction")
        actual, anchors = copper_index(saved)
        if actual != expected:
            raise ValueError("native signal landing correction changed unexpected copper")
        partitions = {}
        for net, (before, pins, pads) in graphs.items():
            after, new_pins, new_pads = _copper_graph(saved, net)
            retained = [*pads, *(p for uid, (name, points) in anchors.items()
                                 if uid in original and name == net for p in points)]
            partition = _partition(before, retained)
            if pins != new_pins or pads != new_pads or partition != _partition(after, retained):
                raise ValueError("native signal landing correction changed retained connectivity")
            partitions[net] = {"pad_nodes": len(pads), "retained_partition": partition}

        after_report = _report_drc(candidate, stage / "after.json")
        expected_violations = Counter(
            key for row in before_report["violations"]
            if (key := _drc_item_key(row)) not in selected_drc
        )
        if (Counter(map(_drc_item_key, after_report["violations"])) != expected_violations
                or Counter(map(_drc_item_key, before_report["unconnected_items"]))
                != Counter(map(_drc_item_key, after_report["unconnected_items"]))):
            raise ValueError("native DRC did not confirm precisely the corrected signal landings")
        if hashlib.sha256(board_path.read_bytes()).hexdigest() != before_sha:
            raise ValueError("source changed before signal landing correction commit")
        after_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        candidate.replace(board_path)
    return {
        "accepted": True, "native_readback": True,
        "warnings_cleared": sorted(selected_drc),
        "connectivity": partitions,
        "board_before_sha256": before_sha, "board_after_sha256": after_sha,
        "replacements": replacements,
        "retired": {uid: terminals[uid] for uid in replacements},
    }


def unlanded_interface_components(
    board: Any, net_names: set[str], cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find interface components without their required native landing.

    KiCad's native connectivity accounts for finite copper shapes, including
    contacts away from pad centres. A dense singleton must also reach its
    reserved opposite-face launch: a connected via alone does not suffice.
    """
    nets = {name for name in net_names if name and not is_power_or_ground_name(name)}
    pads = [pad for fp in board.GetFootprints() for pad in fp.Pads()
            if pad.GetNetname() in nets]
    launches = {
        (spec.ref, spec.pad): spec for spec in signal_escape_specs(board, cfg, nets)
        if spec.via_exit is not None
    }

    def near(position, xy):
        actual = (pcbnew.ToMM(position.x), pcbnew.ToMM(position.y))
        return math.dist(actual, xy) <= _NATIVE_GEOMETRY_TOL_MM

    board.BuildConnectivity()
    connectivity = board.GetConnectivity()
    visited: set[str] = set()
    missing = []
    for pad in pads:
        uid = pad.m_Uuid.AsString()
        if uid in visited:
            continue
        spec = launches.get((pad.GetParentFootprint().GetReferenceAsString(), pad.GetNumber()))
        if spec is not None:
            assert spec.via_exit is not None
            launch_end, launch_width = spec.via_exit
            launch_via = spec.waypoints[-1]
            pad_layer, launch_layer = ((pcbnew.F_Cu, pcbnew.B_Cu) if spec.layer == "F.Cu"
                                       else (pcbnew.B_Cu, pcbnew.F_Cu))
        reached_via = reached_launch = False
        pending = [pad]
        members = []
        landed = False
        while pending:
            item = pending.pop()
            item_id = item.m_Uuid.AsString()
            if item_id in visited or item.GetNetname() != pad.GetNetname():
                continue
            visited.add(item_id)
            if item.Type() == pcbnew.PCB_VIA_T:
                landed = True
                if spec is not None:
                    reached_via |= (item.IsOnLayer(pad_layer) and item.IsOnLayer(launch_layer)
                                    and near(item.GetPosition(), launch_via))
            elif item.Type() == pcbnew.PCB_TRACE_T and spec is not None:
                if (item.GetLayer() == launch_layer
                    and pcbnew.ToMM(item.GetWidth()) + _NATIVE_GEOMETRY_TOL_MM >= launch_width):
                    a, b = item.GetStart(), item.GetEnd()
                    reached_launch |= ((near(a, launch_via) and near(b, launch_end))
                                       or (near(b, launch_via) and near(a, launch_end)))
            if isinstance(item, pcbnew.PAD):
                members.append(f"{item.GetParentFootprint().GetReferenceAsString()}.{item.GetNumber()}")
                if item.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    landed = True
            pending.extend(connectivity.GetConnectedTracks(item))
            pending.extend(connectivity.GetConnectedPads(item))
        if spec is not None and not (reached_via and reached_launch):
            missing.append({"net": pad.GetNetname(), "pads": sorted(members),
                            "reason": "missing_reserved_opposite_face_launch"})
        elif not landed:
            missing.append({"net": pad.GetNetname(), "pads": sorted(members)})
    return missing


