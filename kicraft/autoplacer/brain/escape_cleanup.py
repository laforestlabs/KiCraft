"""Proven escape and router-spur cleanup, plus dangling-copper audit.

Owner-authorized escapes use the stamp receipt. Router-only copper uses exact
input versus output provenance and protects native fill and contracted nets.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _partition(graph: dict, anchors: list) -> list[list[int]]:
    remaining = set(graph)
    components = {}
    component = 0
    while remaining:
        start = remaining.pop()
        stack = [start]
        components[start] = component
        while stack:
            for neighbor, _, _ in graph.get(stack.pop(), ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    components[neighbor] = component
                    stack.append(neighbor)
        component += 1
    grouped: dict[Any, list[int]] = defaultdict(list)
    for index, anchor in enumerate(anchors):
        # A missing physical node is distinct; never infer a connection.
        grouped[("component", components[anchor]) if anchor in components else ("missing", index)].append(index)
    return sorted(grouped.values())


def _verify_saved_copper_removals(original: str, candidate: str, retired: set[str]):
    """Check exact surviving copper and connectivity in a fresh native process."""
    import pcbnew
    from kicraft.autoplacer.hardware.local_contract import _copper_graph, _position, _via_width

    original_board, saved_board = pcbnew.LoadBoard(original), pcbnew.LoadBoard(candidate)
    if original_board is None or saved_board is None:
        raise ValueError("could not reload boards for escape-cleanup custody")
    layers = tuple(original_board.GetEnabledLayers().CuStack())
    if layers != tuple(saved_board.GetEnabledLayers().CuStack()):
        raise ValueError("saved escape cleanup changed copper layers")

    def index(board):
        geometry = {}
        anchors = defaultdict(list)
        for item in board.GetTracks():
            uid, net, kind = item.m_Uuid.AsString(), item.GetNetname(), item.GetClass()
            if uid in geometry:
                raise ValueError("ambiguous native copper UUID in escape cleanup")
            locked = item.IsLocked()
            if isinstance(item, pcbnew.PCB_VIA):
                point = _position(item.GetPosition())
                active_layers = tuple(layer for layer in layers if item.IsOnLayer(layer))
                geometry[uid] = (net, kind, locked, point, item.GetDrillValue(), item.GetViaType(),
                                 tuple((layer, _via_width(item, layer)) for layer in active_layers))
                anchors[net].extend((point, layer) for layer in active_layers)
            else:
                points = tuple(sorted((_position(item.GetStart()), _position(item.GetEnd()))))
                curve = (_position(item.GetMid()),) if kind == "PCB_ARC" else ()
                layer = item.GetLayer()
                geometry[uid] = (net, kind, locked, points, curve, layer, item.GetWidth())
                anchors[net].extend((point, layer) for point in points)
        return geometry, anchors

    before, _ = index(original_board)
    affected_nets = {before[uid][0] for uid in retired if uid in before}

    def zone_geometry(board):
        return {
            (zone.m_Uuid.AsString(), zone.GetNetname()): (
                zone.Outline().Format(),
                tuple((layer, zone.GetFilledPolysList(layer).Format())
                      for layer in layers if zone.IsOnLayer(layer)),
            )
            for zone in board.Zones()
            if not zone.GetIsRuleArea() and zone.GetNetname() in affected_nets
        }

    after, anchors = index(saved_board)
    if not retired <= before.keys() or {uid: key for uid, key in before.items() if uid not in retired} != after:
        raise ValueError("saved escape cleanup changed unexpected native copper")
    if zone_geometry(original_board) != zone_geometry(saved_board):
        raise ValueError("saved escape cleanup changed native zone copper")
    for net in sorted(affected_nets):
        graph_before, _, pads_before = _copper_graph(original_board, net)
        graph_after, _, pads_after = _copper_graph(saved_board, net)
        retained = [*dict.fromkeys((*pads_before, *pads_after)), *anchors[net]]
        if _partition(graph_before, retained) != _partition(graph_after, retained):
            raise ValueError(f"saved escape cleanup changed native {net} connectivity")
    return saved_board


def _readback_signal_cleanup(original: str, candidate: str, proof_path: str) -> None:
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    retired = {row["native_uuid"] for branch in proof["removed"] for row in branch["items"]}
    saved_board = _verify_saved_copper_removals(original, candidate, retired)
    from kicraft.autoplacer.hardware.history_geometry import capture_board, close_native_history

    capture_board(saved_board, phase="signal_escape_cleanup")
    close_native_history()


def _cleanup_native(board_path: str, ownership_path: str, receipt_path: str) -> dict:
    import pcbnew
    from kicraft.autoplacer.hardware.local_contract import (
        _copper_graph, _pad_contains, _pad_shapes, _position,
    )

    board = pcbnew.LoadBoard(board_path)
    if board is None:
        raise ValueError("could not load cleanup board")
    from kicraft.autoplacer.hardware.history_geometry import capture_board
    capture_board(board, phase="signal_escape_cleanup_input")
    ownership = json.loads(Path(ownership_path).read_text())
    if ownership.get("version") != 1:
        raise ValueError("unsupported escape ownership receipt")
    before_sha = _sha(board_path)
    if (ownership.get("routed_board_sha256") != before_sha
            or not ownership.get("stamped_board_sha256")):
        raise ValueError("escape ownership receipt is absent or does not bind this routed board")
    prune_terminal_tracks = ownership.get("prune_terminal_tracks") is True
    enabled_layers = tuple(board.GetEnabledLayers().CuStack())
    # Track remove/add operations can invalidate pcbnew's SWIG container views.
    # Cleanup never modifies zones; record the native zone dependencies once.
    zoned_nets = {zone.GetNetname() for zone in board.Zones() if not zone.GetIsRuleArea()}
    layer_names = {layer: board.GetLayerName(layer) for layer in enabled_layers}

    # The native graph indexes pads independently for each net. Those pads
    # stay fixed throughout this transaction; only their contact edges change.
    pad_nodes_by_net: dict[str, list] = defaultdict(list)
    pad_count_by_net: Counter = Counter()
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            net = pad.GetNetname()
            index = pad_count_by_net[net]
            pad_count_by_net[net] += 1
            if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                pad_nodes_by_net[net].append(("pad", index))
            else:
                pad_nodes_by_net[net].extend(("pad", index, layer)
                                             for layer in enabled_layers if pad.IsOnLayer(layer))

    def pt(x, y):
        return (round(float(x) * 1e6), round(float(y) * 1e6))
    def stamped_key(row, kind):
        if kind == "track":
            ends = sorted((pt(row["start_x"], row["start_y"]), pt(row["end_x"], row["end_y"])))
            return (kind, row["net_name"], row["layer"], *ends, round(float(row["width"]) * 1e6))
        return (kind, row["net_name"], pt(row["x"], row["y"]),
                round(float(row["size"]) * 1e6), round(float(row["drill"]) * 1e6))
    def native_key(item):
        if isinstance(item, pcbnew.PCB_VIA):
            try:
                width = item.GetWidth(pcbnew.F_Cu)
            except TypeError:
                width = item.GetWidth()
            return ("via", item.GetNetname(), _position(item.GetPosition()), int(width), int(item.GetDrillValue()))
        if item.GetClass() != "PCB_TRACK":
            return None
        ends = sorted((_position(item.GetStart()), _position(item.GetEnd())))
        return ("track", item.GetNetname(), board.GetLayerName(item.GetLayer()), *ends, int(item.GetWidth()))

    native: dict[tuple, list] = defaultdict(list)
    by_net = defaultdict(list)
    for item in board.GetTracks():
        key = native_key(item)
        native[key].append(item)
        by_net[item.GetNetname()].append((key, item))
    rows: dict[tuple, list] = defaultdict(list)
    groups: dict[tuple, list] = defaultdict(list)
    for kind, name in (("track", "traces"), ("via", "vias")):
        for row in ownership.get(name, []):
            owner = row.get("escape_ownership") or {}
            if owner.get("owner") != "compiler.signal_escape":
                continue
            key = stamped_key(row, kind)
            rows[key].append(row)
            groups[(owner.get("branch_id"), owner.get("ref"), owner.get("pad"))].append((key, row))
    removed, retained = [], []
    # Even unrelated PCB objects can lose their typed SWIG views after a
    # native removal. Prove the planned subset without mutating the board.
    retired = {}
    board.BuildConnectivity()
    connectivity = board.GetConnectivity()

    def anchored_foreign_endpoint(track, group, branch_ids):
        layer = track.GetLayer()
        shapes = [
            _pad_shapes(pad, layer) for pad in connectivity.GetConnectedPads(track)
            if pad.GetParentFootprint().GetReferenceAsString() == group[1]
            and str(pad.GetNumber()) == str(group[2]) and pad.IsOnLayer(layer)
        ]
        if not shapes:
            return None
        start, end = _position(track.GetStart()), _position(track.GetEnd())
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        for other in connectivity.GetConnectedTracks(track):
            uid = other.m_Uuid.AsString()
            if (uid in branch_ids or native_key(other) in retired
                    or other.GetClass() != "PCB_TRACK" or other.GetLayer() != layer):
                continue
            for endpoint in (other.GetStart(), other.GetEnd()):
                point = _position(endpoint)
                x, y = point[0] - start[0], point[1] - start[1]
                along, cross = x * dx + y * dy, x * dy - y * dx
                if (0 < along < length_sq
                        and 4 * cross * cross <= track.GetWidth() ** 2 * length_sq
                        and not any(_pad_contains(shape, hole, point) for shape, hole in shapes)):
                    return uid
        return None

    for group, entries in groups.items():
        entries.sort(key=lambda pair: int(pair[1]["escape_ownership"].get("branch_item_index", "0")))
        ids = [row["escape_ownership"].get("source_uuid") for _, row in entries]
        expected = {row["escape_ownership"].get("branch_item_count") for _, row in entries}
        if (not all(group) or len(expected) != 1 or not next(iter(expected), "").isdigit()
                or int(next(iter(expected))) != len(entries) or len(set(ids)) != len(ids)
                or any(len(rows[key]) != 1 or len(native.get(key, [])) != 1 for key, _ in entries)):
            retained.append({"branch": list(group), "reason": "incomplete_or_ambiguous_ownership"})
            continue
        items = [native[key][0] for key, _ in entries]
        nets = {item.GetNetname() for item in items}
        if len(nets) != 1:
            retained.append({"branch": list(group), "reason": "mixed_nets"})
            continue
        net = next(iter(nets))
        # The graph intentionally does not model zones. Never use it as proof
        # when a zone on this net could make the removed copper load-bearing.
        if net in zoned_nets:
            retained.append({"branch": list(group), "reason": "same_net_zone_not_in_graph"})
            continue
        try:
            if any(key in retired for key, _ in entries):
                raise ValueError("branch copper already retired")
            all_ids = {item.m_Uuid.AsString() for item in items}
            contacts = {}
            for item in items:
                for other in connectivity.GetConnectedTracks(item):
                    uid = other.m_Uuid.AsString()
                    if native_key(other) in retired:
                        continue
                    if uid not in all_ids:
                        contacts[uid] = "track_or_via"
                for pad in connectivity.GetConnectedPads(item):
                    uid = pad.m_Uuid.AsString()
                    ref = pad.GetParentFootprint().GetReferenceAsString()
                    contacts[uid] = (
                        "owner_pad" if ref == group[1] and str(pad.GetNumber()) == str(group[2])
                        else "other_pad"
                    )
            whole_dead = bool(contacts) and set(contacts.values()) == {"owner_pad"} and len(contacts) == 1
            if whole_dead:
                candidates = [list(zip(items, entries))]
            else:
                # Branch items are emitted from pad to terminal. Retire from
                # terminal to pad: removing an unused opposite-face launch
                # can make its via single-face, then expose redundant tracks.
                candidates = [
                    [(item, entry)] for item, entry in reversed(list(zip(items, entries)))
                    if isinstance(item, pcbnew.PCB_VIA)
                    or (prune_terminal_tracks and item.GetClass() == "PCB_TRACK")
                ]
                if not candidates:
                    retained.append({"branch": list(group), "reason": "not_a_single_pad_dead_branch",
                                     "contacts": contacts})
                    continue
            for candidate in candidates:
                selected = [item for item, _ in candidate]
                selected_ids = {item.m_Uuid.AsString() for item in selected}
                contact_layers = set()
                if not whole_dead and isinstance(selected[0], pcbnew.PCB_VIA):
                    via = selected[0]
                    for other in connectivity.GetConnectedTracks(via):
                        if other.m_Uuid.AsString() in selected_ids or native_key(other) in retired:
                            continue
                        if isinstance(other, pcbnew.PCB_VIA):
                            contact_layers.update(layer for layer in enabled_layers if other.IsOnLayer(layer))
                        elif other.GetClass() == "PCB_TRACK":
                            contact_layers.add(other.GetLayer())
                        else:
                            raise ValueError("unsupported non-linear copper contact")
                    for pad in connectivity.GetConnectedPads(via):
                        contact_layers.update(layer for layer in enabled_layers if pad.IsOnLayer(layer))
                    if len(contact_layers) > 1:
                        retained.append({"branch": list(group), "reason": "via_has_multilayer_contacts",
                                         "native_uuid": via.m_Uuid.AsString(),
                                         "contact_layers": sorted(board.GetLayerName(layer)
                                                                  for layer in contact_layers)})
                        continue
                active = [(key, other) for key, other in by_net[net] if key not in retired]
                graph, _, _ = _copper_graph(board, net, copper=(item for _, item in active))
                if not whole_dead and selected[0].GetClass() == "PCB_TRACK":
                    track = selected[0]
                    # A terminal centerline end has one graph edge and no
                    # finite-width pad, via or neighboring copper contact.
                    ends = [(_position(point), track.GetLayer())
                            for point in (track.GetStart(), track.GetEnd())]
                    if not any(len(graph.get(end, ())) == 1 for end in ends):
                        continue
                    # A foreign route may graze the pad but terminate on this
                    # escape's interior. Finite copper connectivity alone would
                    # permit deleting its proper endpoint landing and leave the
                    # foreign, load-bearing track dangling.
                    anchor = anchored_foreign_endpoint(track, group, all_ids)
                    if anchor is not None:
                        retained.append({"branch": list(group), "reason": "anchors_foreign_pad_landing",
                                         "native_uuid": track.m_Uuid.AsString(), "foreign_uuid": anchor})
                        continue
                candidate_keys = {key for _, (key, _) in candidate}
                pad_nodes = pad_nodes_by_net[net]
                anchors = list(pad_nodes)
                for key, other in active:
                    if key in candidate_keys:
                        continue
                    if isinstance(other, pcbnew.PCB_VIA):
                        pos = _position(other.GetPosition())
                        anchors.extend((pos, layer) for layer in enabled_layers if other.IsOnLayer(layer))
                    elif other.GetClass() == "PCB_TRACK":
                        anchors.extend(((_position(other.GetStart()), other.GetLayer()),
                                        (_position(other.GetEnd()), other.GetLayer())))
                    else:
                        raise ValueError("unsupported non-linear copper")
                before = _partition(graph, anchors)
                graph_after, _, _ = _copper_graph(
                    board, net, copper=(item for key, item in active if key not in candidate_keys),
                )
                after = _partition(graph_after, anchors)
                if after != before:
                    retained.append({"branch": list(group), "reason": "topology_changed"})
                    continue
                removed.append({"branch": list(group), "net": net,
                                "items": [{**entry[1], "native_uuid": item.m_Uuid.AsString()}
                                          for item, entry in candidate],
                                "external_contacts": contacts,
                                "proof": {"pad_nodes": len(pad_nodes), "anchors": len(anchors),
                                          "partition_before": before, "partition_after": after,
                                          "same_net_zone_absent": True,
                                          "single_layer_via_contacts": sorted(
                                              layer_names[layer] for layer in contact_layers
                                          ) if not whole_dead and isinstance(selected[0], pcbnew.PCB_VIA) else None}})
                for item, (key, _) in candidate:
                    retired[key] = item
        except Exception as exc:
            retained.append({"branch": list(group), "reason": "proof_failed", "detail": str(exc)})
    accepted = not any(row["reason"] in {
        "incomplete_or_ambiguous_ownership", "mixed_nets", "proof_failed",
    } for row in retained)
    if removed and accepted:
        for item in retired.values():
            board.Remove(item)
        if board.Save(board_path) is False:
            raise RuntimeError("could not save proven signal escape cleanup candidate")
    receipt = {"version": 1, "ownership_sha256": _sha(ownership_path),
               "board_before_sha256": before_sha, "board_after_sha256": _sha(board_path),
               "stamped_board_sha256": ownership["stamped_board_sha256"],
               "accepted": accepted,
               "removed": removed, "retained": retained}
    Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def cleanup_signal_escapes(board_path: str | Path, ownership_path: str | Path,
                           receipt_path: str | Path) -> dict:
    """Apply proven cleanup in an isolated pcbnew process and return its receipt.

    Call only after strict routing/child custody verification, before final DRC.
    Update any subsequent copper-custody ledger using the removed item records.
    """
    from kicraft.autoplacer.routing_board import _kicad_subprocess_env

    board_path, ownership_path, receipt_path = map(Path, (board_path, ownership_path, receipt_path))
    with tempfile.TemporaryDirectory(dir=board_path.parent, prefix=".signal-escapes-") as directory:
        stage = Path(directory)
        candidate = stage / board_path.name
        shutil.copyfile(board_path, candidate)
        proof_path = stage / "cleanup.json"
        proc = subprocess.run(
            [sys.executable, "-m", __name__, str(candidate), str(ownership_path), str(proof_path)],
            capture_output=True, text=True, env=_kicad_subprocess_env(),
        )
        if proc.returncode:
            raise RuntimeError(f"signal escape cleanup failed: {proc.stderr}")
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if proof["accepted"] and proof["removed"]:
            proc = subprocess.run(
                [sys.executable, "-m", __name__, "--readback-escapes",
                 str(board_path), str(candidate), str(proof_path)],
                capture_output=True, text=True, env=_kicad_subprocess_env(),
            )
            if proc.returncode:
                proof["accepted"] = False
                proof["reason"] = f"native cleanup readback failed: {proc.stderr.strip()}"
            else:
                proof["native_readback"] = True
            if proof["accepted"] and json.loads(ownership_path.read_text(encoding="utf-8")).get("require_native_drc") is True:
                _copy_native_rules(board_path, candidate)
                before = _report_drc(board_path, stage / "before.json")
                after = _report_drc(candidate, stage / "after.json")
                retired = {row["native_uuid"] for branch in proof["removed"] for row in branch["items"]}
                if not _drc_unchanged_except_retired(before, after, retired):
                    proof["accepted"] = False
                    proof["reason"] = "native DRC or opens changed"
            if proof["accepted"] and _sha(board_path) == proof["board_before_sha256"]:
                candidate.replace(board_path)
            else:
                proof["accepted"] = False
        if not proof["accepted"] and proof["removed"]:
            proof["attempted"] = proof["removed"]
            proof["removed"] = []
            proof["board_after_sha256"] = _sha(board_path)
        receipt_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return proof


_DANGLING_TYPES = frozenset({"via_dangling", "track_dangling"})


def _copy_native_rules(original: Path, candidate: Path) -> None:
    for suffix in (".kicad_pro", ".kicad_dru"):
        rules = original.with_suffix(suffix)
        if rules.is_file():
            shutil.copyfile(rules, candidate.with_suffix(suffix))


def _report_drc(board: Path, output: Path) -> dict:
    proc = subprocess.run(
        ["kicad-cli", "pcb", "drc", "--format", "json", "--severity-all",
         "--all-track-errors", "--output", str(output), str(board)],
        capture_output=True, timeout=60,
    )
    if proc.returncode or not output.is_file():
        raise RuntimeError("full native DRC could not prove router spur cleanup")
    report = json.loads(output.read_text(encoding="utf-8"))
    if (set(report.get("included_severities", ())) != {"error", "warning", "exclusion"}
            or not all(isinstance(report.get(k), list)
                       for k in ("violations", "unconnected_items", "schematic_parity"))):
        raise ValueError("incomplete native DRC proof for router spur cleanup")
    return report


def _drc_item_key(row: dict) -> tuple:
    return (row["type"], row.get("severity"), tuple(sorted(item["uuid"] for item in row["items"])))

def _drc_unchanged_except_retired(before: dict, after: dict, retired: set[str]) -> bool:
    retained = {
        _drc_item_key(row) for row in before["violations"]
        if not (row["type"] in _DANGLING_TYPES and len(row["items"]) == 1
                and row["items"][0]["uuid"] in retired)
    }
    return (
        not ({_drc_item_key(row) for row in after["violations"]} - retained)
        and {_drc_item_key(row) for row in before["unconnected_items"]}
        == {_drc_item_key(row) for row in after["unconnected_items"]}
    )


def _touches_router_zone_fill(item, zones, layers):
    """Use the whole native copper envelope, not just its center or endpoints."""
    import pcbnew
    from kicraft.autoplacer.hardware.local_contract import _via_width

    via = isinstance(item, pcbnew.PCB_VIA)
    axis = item.GetPosition() if via else pcbnew.SEG(item.GetStart(), item.GetEnd())
    for zone in zones:
        for layer in layers:
            if not item.IsOnLayer(layer) or not zone.IsOnLayer(layer):
                continue
            if not zone.IsFilled():
                return True  # No reliable fill evidence on this face.
            filled = zone.GetFilledPolysList(layer)
            if not filled.OutlineCount():
                continue
            width = _via_width(item, layer) if via else item.GetWidth()
            # Native Collide uses a strict clearance comparison, so include
            # one extra IU to retain even tangencies. The via disk is a
            # conservative bound for its annulus, including conditional flash.
            if filled.Collide(axis, (width + 1) // 2 + 1):
                return True
    return False


def _router_delta_native(stamped: str, candidate: str, request_path: str) -> None:
    import pcbnew
    from kicraft.autoplacer.hardware.local_contract import _copper_graph, _position, _via_width

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    baseline, board = pcbnew.LoadBoard(stamped), pcbnew.LoadBoard(candidate)
    if baseline is None or board is None:
        raise ValueError("unable to load exact router input/output")
    layers = tuple(board.GetEnabledLayers().CuStack())
    zones_by_net = defaultdict(list)
    for zone in board.Zones():
        if not zone.GetIsRuleArea():
            zones_by_net[zone.GetNetname()].append(zone)
    protected = set(request["protected_nets"])
    if request["thermal"]:
        protected.add(request["ground_net"])
    if request["usb_contracted"]:
        from kicraft.autoplacer.brain.signal_escape import declared_usb_nets

        protected.update(declared_usb_nets({"usb_routing_contracts": request["usb_contracted"]}))

    def local_pins(ref, pins):
        fp = board.FindFootprintByReference(ref)
        if fp is not None:
            protected.update(p.GetNetname() for p in fp.Pads() if str(p.GetNumber()) in pins)

    for spec in request["local_bypasses"]:
        owner, member = spec.get("owner_ref"), spec.get("bypass_ref")
        if member and board.FindFootprintByReference(member) and "max_route_length_mm" in spec:
            for owner_pin, member_pin in spec.get("pin_pairs") or [spec["power_pins"]]:
                local_pins(spec.get("route_anchor_ref", owner), {str(spec.get("route_anchor_pin", owner_pin))})
                local_pins(member, {str(member_pin)})
        if spec.get("ground_return"):
            protected.add(spec["ground_return"]["net"])

    def key(t):
        if isinstance(t, pcbnew.PCB_VIA):
            return ("via", _position(t.GetPosition()), t.GetDrillValue(), t.GetViaType(),
                    tuple((layer, _via_width(t, layer)) for layer in layers if t.IsOnLayer(layer)))
        if t.GetClass() != "PCB_TRACK":
            return None
        return ("track", *sorted((_position(t.GetStart()), _position(t.GetEnd()))), t.GetWidth(), t.GetLayer())

    original = defaultdict(Counter)
    items = {}
    by_net = defaultdict(dict)
    for t in board.GetTracks():
        net, geometry, uuid = t.GetNetname(), key(t), t.m_Uuid.AsString()
        if uuid in items:
            raise ValueError("ambiguous native copper UUID in router output")
        original[net][geometry] += 1
        items[uuid] = t
        by_net[net][uuid] = t
    source = defaultdict(Counter)
    source_uuids = set()
    retired_input = request["retired_input"]
    for t in baseline.GetTracks():
        uuid, net = t.m_Uuid.AsString(), t.GetNetname()
        # Previously authorized compiler cleanup may have retired an input
        # escape. Its identity is still inherited: never authorize deleting
        # a surviving source item, only acknowledge its proven absence.
        source_uuids.add(uuid)
        if uuid in retired_input:
            if uuid in items or retired_input[uuid] != net:
                raise ValueError("retired input escape disagrees with native source/output")
        else:
            source[net][key(t)] += 1
    before_sha = _sha(candidate)
    removed = []
    retired = {}
    conn = None
    for uuid, kind in request["violations"]:
        item = items.get(uuid)
        if item is None or uuid in source_uuids or item.IsLocked():
            continue
        net, geometry = item.GetNetname(), key(item)
        if (not net or net in protected or geometry is None
                or geometry[0] != kind or None in original[net]
                or source[net][geometry] or source[net] - original[net]):
            continue
        zones = zones_by_net.get(net)
        if zones and _touches_router_zone_fill(item, zones, layers):
            continue
        anchors = []
        index = 0
        for fp in board.GetFootprints():
            for pad in fp.Pads():
                if pad.GetNetname() != net:
                    continue
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    anchors.append(("pad", index))
                else:
                    anchors.extend(("pad", index, layer) for layer in layers if pad.IsOnLayer(layer))
                index += 1
        active = by_net[net]
        remaining = []
        for other_uuid, other in active.items():
            if other_uuid == uuid:
                continue
            remaining.append(other)
            if isinstance(other, pcbnew.PCB_VIA):
                point = _position(other.GetPosition())
                anchors.extend((point, layer) for layer in layers if other.IsOnLayer(layer))
            else:
                anchors.extend((_position(end), other.GetLayer()) for end in (other.GetStart(), other.GetEnd()))
        if isinstance(item, pcbnew.PCB_VIA):
            if conn is None:
                board.BuildConnectivity()
                conn = board.GetConnectivity()
            neighbors = [o for o in conn.GetConnectedTracks(item) if o.m_Uuid.AsString() not in retired]
            tracks = {o.GetLayer() for o in neighbors if o.GetClass() == "PCB_TRACK"}
            tracks.update(layer for pad in conn.GetConnectedPads(item) for layer in layers if pad.IsOnLayer(layer))
            if len(tracks) > 1 or any(isinstance(o, pcbnew.PCB_VIA) for o in neighbors):
                continue
        # Native DRC identified the dangling end. The physical copper graph
        # can give both ends degree >1 via incidental overlaps; only losing
        # connectivity among retained copper or pads makes this removal unsafe.
        graph, _, _ = _copper_graph(board, net, copper=active.values())
        before = _partition(graph, anchors)
        after, _, _ = _copper_graph(board, net, copper=remaining)
        if _partition(after, anchors) != before:
            continue
        del active[uuid]
        retired[uuid] = item
        removed.append({"uuid": uuid, "net": net, "kind": kind,
                        "partition": before, "same_net_filled_copper_disjoint": True})
    if retired:
        for item in retired.values():
            board.Remove(item)
        if board.Save(candidate) is False:
            raise RuntimeError("could not save proven router-spur cleanup candidate")
    Path(request_path).write_text(json.dumps({"board_before_sha256": before_sha, "removed": removed}))


def prune_router_spurs(stamped: Path, routed: Path, cfg: dict, *,
                       protected_nets: set[str] | None = None,
                       input_cleanup: dict | None = None) -> dict:
    """Only retire DRC-named copper born in the current routing transaction.

    Never try to prune inherited or engineering-contracted copper, a candidate
    that can contact actual zone fill, or a result changing connectivity or
    DRC beyond the retired UUIDs. Native refill can expose new dependencies.
    A preceding compiler-owned cleanup must bind both this exact input and
    current output before its already-retired escapes can be acknowledged.
    """
    from kicraft.autoplacer.routing_board import _kicad_subprocess_env

    try:
        retired_input = {}
        if input_cleanup is not None:
            if (input_cleanup.get("accepted") is not True
                    or input_cleanup.get("native_readback") is not True
                    or input_cleanup.get("stamped_board_sha256") != _sha(stamped)
                    or input_cleanup.get("board_after_sha256") != _sha(routed)):
                raise ValueError("preceding escape cleanup does not bind this input/output")
            for branch in input_cleanup["removed"]:
                for row in branch["items"]:
                    owner = row["escape_ownership"]
                    uuid = row["native_uuid"]
                    if owner.get("owner") != "compiler.signal_escape" or owner.get("source_uuid") != uuid:
                        raise ValueError("retired input lacks compiler escape identity")
                    if uuid in retired_input:
                        raise ValueError("ambiguous retired input escape")
                    retired_input[uuid] = row["net_name"]
        with tempfile.TemporaryDirectory(dir=routed.parent, prefix=".router-spurs-") as directory:
            stage = Path(directory)
            candidate = stage / routed.name
            shutil.copyfile(routed, candidate)
            _copy_native_rules(routed, candidate)
            before = _report_drc(candidate, stage / "before.json")
            original_sha = _sha(candidate)
            request = stage / "proof.json"
            request_config = {
                "protected_nets": sorted(protected_nets or ()),
                "local_bypasses": cfg.get("local_bypasses") or [], "thermal": bool(cfg.get("thermal_requirements")),
                "ground_net": cfg.get("gnd_zone_net", "GND"), "usb_contracted": cfg.get("usb_routing_contracts") or [],
                "retired_input": retired_input,
            }
            proof = {"board_before_sha256": original_sha, "removed": []}
            if input_cleanup is not None:
                proof["input_cleanup"] = {
                    key: input_cleanup[key] for key in
                    ("ownership_sha256", "stamped_board_sha256", "board_before_sha256", "board_after_sha256")
                }
                proof["retired_input_uuids"] = sorted(retired_input)
            after = before
            while True:
                warnings = {
                    (row["items"][0]["uuid"], row["type"].partition("_")[0])
                    for row in after["violations"]
                    if row["type"] in _DANGLING_TYPES and len(row["items"]) == 1
                }
                if not warnings:
                    break
                request.write_text(json.dumps({**request_config, "violations": sorted(warnings)}))
                proc = subprocess.run(
                    [sys.executable, "-m", __name__, "--router-delta", str(stamped), str(candidate), str(request)],
                    capture_output=True, text=True, timeout=90, env=_kicad_subprocess_env(),
                )
                if proc.returncode:
                    raise RuntimeError(f"router-spur topology proof failed: {proc.stderr}")
                wave = json.loads(request.read_text())
                if not wave["removed"]:
                    break
                # Retiring a terminal track may expose a now-single-face via.
                # Every wave needs fresh native DRC authority and strictly
                # removes copper; no source item can enter this finite cascade.
                proof["removed"].extend(wave["removed"])
                request.write_text(json.dumps(proof))
                proc = subprocess.run(
                    [sys.executable, "-m", __name__, "--readback-router-delta",
                     str(routed), str(candidate), str(request)],
                    capture_output=True, text=True, timeout=90, env=_kicad_subprocess_env(),
                )
                if proc.returncode:
                    return {"accepted": False, "attempted": proof["removed"],
                            "reason": f"native cleanup readback failed: {proc.stderr.strip()}"}
                proof["native_readback"] = True
                after = _report_drc(candidate, stage / f"after-{len(proof['removed'])}.json")
            if not proof["removed"]:
                return {"accepted": True, **proof}
            retired = {row["uuid"] for row in proof["removed"]}
            if not _drc_unchanged_except_retired(before, after, retired):
                return {"accepted": False, "attempted": proof["removed"], "reason": "native DRC or opens changed"}
            if _sha(routed) != proof["board_before_sha256"]:
                return {"accepted": False, "reason": "routed PCB changed before spur cleanup commit"}
            proof["board_after_sha256"] = _sha(candidate)
            candidate.replace(routed)
            return {"accepted": True, **proof}
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"accepted": False, "reason": str(exc)}


def audit_dangling_copper(board_path: str | Path, drc: dict,
                          evidence: list[dict] | None = None) -> dict:
    """Audit *complete native JSON DRC*: unproven dangling items remain failures.

    Evidence must name the board hash, native item UUID, and an engineering
    justification.  An absent/incomplete DRC is not evidence of zero dangling.
    """
    board_hash = _sha(board_path)
    if not isinstance(drc, dict) or not isinstance(drc.get("violations"), list):
        return {"board_sha256": board_hash, "complete": False, "unproven": [], "accepted": False}
    proofs = {row.get("uuid"): row for row in (evidence or [])
              if row.get("board_sha256") == board_hash and row.get("uuid")
              and str(row.get("justification", "")).strip()}
    retained, unproven = [], []
    for violation in drc["violations"]:
        if violation.get("type") not in {"via_dangling", "track_dangling"}:
            continue
        items = violation.get("items") or []
        ids = [item.get("uuid") for item in items if isinstance(item, dict) and item.get("uuid")]
        row = {"violation": violation, "evidence": [proofs[uid] for uid in ids if uid in proofs]}
        (retained if ids and all(uid in proofs for uid in ids) else unproven).append(row)
    return {"board_sha256": board_hash, "complete": True, "retained": retained,
            "unproven": unproven, "accepted": not unproven}


def _router_origin_index(board, translation=None) -> dict[str, tuple]:
    """Exact supported native copper, using the stamper's float-to-IU path."""
    import pcbnew
    from kicraft.autoplacer.hardware.local_contract import _via_width

    layers = tuple(board.GetEnabledLayers().CuStack())
    if layers != (pcbnew.F_Cu, pcbnew.B_Cu):
        raise ValueError("router origin supports only the native two-layer stack")

    def scalar(value):
        return value if translation is None else pcbnew.FromMM(pcbnew.ToMM(value))

    def point(value):
        if translation is None:
            return value.x, value.y
        return tuple(pcbnew.FromMM(pcbnew.ToMM(v) + d)
                     for v, d in zip((value.x, value.y), translation))

    result = {}
    for item in board.GetTracks():
        uid, net = item.m_Uuid.AsString(), item.GetNetname()
        if uid in result:
            raise ValueError("ambiguous native router copper identity")
        if isinstance(item, pcbnew.PCB_VIA):
            widths = tuple(_via_width(item, layer) for layer in layers if item.IsOnLayer(layer))
            if (item.GetViaType() != pcbnew.VIATYPE_THROUGH
                    or len(widths) != 2 or widths[0] != widths[1]):
                continue
            key = ("via", net, *point(item.GetPosition()),
                   scalar(widths[0]), scalar(item.GetDrillValue()))
        elif item.GetClass() == "PCB_TRACK":
            a, b = sorted((point(item.GetStart()), point(item.GetEnd())))
            key = ("track", net, item.GetLayer(), *a, *b, scalar(item.GetWidth()))
        else:
            continue
        result[uid] = key
    return result


def _capture_router_origin(stamped: Path, routed: Path) -> dict:
    """Capture only copper absent by identity AND geometry from router input."""
    import pcbnew

    source, output = pcbnew.LoadBoard(str(stamped)), pcbnew.LoadBoard(str(routed))
    if source is None or output is None:
        raise ValueError("could not load exact router-origin boundary")
    if source.GetCopperLayerCount() != 2 or output.GetCopperLayerCount() != 2:
        return {
            "version": 1, "router_input_sha256": _sha(stamped),
            "router_output_sha256": _sha(routed), "items": {},
            "ineligible": "router-origin frame proof supports two-layer boards only",
        }
    inherited = _router_origin_index(source)
    geometry = set(inherited.values())
    return {
        "version": 1, "router_input_sha256": _sha(stamped),
        "router_output_sha256": _sha(routed),
        "items": {uid: key for uid, key in _router_origin_index(output).items()
                  if uid not in inherited and key not in geometry},
    }


def _prepare_router_origin_frame(origin: dict, routed: Path, translation) -> dict:
    """Carry surviving raw identities through the exact pending restamp."""
    import pcbnew

    board = pcbnew.LoadBoard(str(routed))
    if board is None:
        raise ValueError("could not load source before router-origin transform")
    current = _router_origin_index(board)
    delta = (translation.x, translation.y)
    transformed = _router_origin_index(board, delta)
    return {
        "version": 1, "router_input_sha256": origin["router_input_sha256"],
        "router_output_sha256": origin["router_output_sha256"],
        "raw_origin_sha256": hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest(),
        "precenter_board_sha256": _sha(routed), "translation_mm": delta,
        "items": {uid: transformed[uid] for uid, key in origin["items"].items()
                  if current.get(uid) == tuple(key)},
    }


def _bind_router_origin_frame(prepared: dict, centered: Path) -> dict:
    """Bind unique exact native geometries, never approximate birthplace."""
    import pcbnew

    board = pcbnew.LoadBoard(str(centered))
    if board is None:
        raise ValueError("could not load centered router-origin boundary")
    geometry = defaultdict(list)
    for uid, key in _router_origin_index(board).items():
        geometry[key].append(uid)
    items = {}
    for key in prepared["items"].values():
        matches = geometry[tuple(key)]
        if len(matches) != 1 or matches[0] in items:
            raise ValueError("router origin lost or ambiguous after native restamp")
        items[matches[0]] = tuple(key)
    return {**prepared, "centered_board_sha256": _sha(centered), "items": items}


def _finish_router_origin_copper(origin: dict, routed: Path, cfg: dict) -> dict:
    """Isolate the native protection-fence mutation from long-lived solvers."""
    from kicraft.autoplacer.routing_board import run_pcbnew_script

    settings = {key: cfg[key] for key in
                ("thermal_requirements", "gnd_zone_net", "usb_routing_contracts", "local_bypasses")
                if key in cfg}
    with tempfile.TemporaryDirectory(dir=routed.parent, prefix=".router-origin-proof-") as directory:
        request, result = Path(directory) / "request.json", Path(directory) / "result.json"
        request.write_text(json.dumps({"origin": origin, "cfg": settings}))
        run_pcbnew_script(
            "import json\nfrom pathlib import Path\n"
            "from kicraft.autoplacer.brain.escape_cleanup import _finish_router_origin_copper_native\n"
            f"request = json.loads(Path({str(request)!r}).read_text())\n"
            f"proof = _finish_router_origin_copper_native(request['origin'], Path({str(routed)!r}), "
            "request['cfg'])\n"
            f"Path({str(result)!r}).write_text(json.dumps(proof))\n"
        )
        return json.loads(result.read_text())


def _finish_router_origin_copper_native(origin: dict, routed: Path, cfg: dict) -> dict:
    """Close only raw router copper after fill; all other copper is inherited.

    The temporary board is a protection fence, not a synthetic design result:
    it contains every current non-router item, including new finisher copper.
    Its derivation is bound to the original input/output and exact frame receipt.
    """
    import pcbnew

    before_sha = _sha(routed)
    board = pcbnew.LoadBoard(str(routed))
    if board is None:
        raise ValueError("could not load finished source router copper")
    current = _router_origin_index(board)
    eligible = {uid for uid, key in origin["items"].items() if current.get(uid) == tuple(key)}
    proof_origin = {key: value for key, value in origin.items() if key != "items"}
    proof_origin["eligible_router_items"] = len(eligible)
    if not eligible:
        return {"accepted": True, "removed": [], "router_origin": proof_origin}
    with tempfile.TemporaryDirectory(dir=routed.parent, prefix=".router-origin-") as directory:
        fence = Path(directory) / "protected-copper.kicad_pcb"
        removals = [item for item in board.GetTracks() if item.m_Uuid.AsString() in eligible]
        for item in removals:
            board.Remove(item)
        if board.Save(str(fence)) is False:
            raise RuntimeError("could not save source copper protection fence")
        if _sha(routed) != before_sha:
            raise ValueError("source changed while binding final router copper")
        proof = prune_router_spurs(fence, routed, cfg)
    return {**proof, "router_origin": proof_origin}


def finish_boundary_escape_landings(
    board_path: Path, ownership_path: Path, cleanup: dict,
) -> dict:
    """Canonicalize current compiler-owned public joins, never leaf internals."""
    from kicraft.autoplacer.brain.signal_escape import _finish_owned_escape_landings

    ownership = json.loads(ownership_path.read_text())
    if (cleanup.get("accepted") is not True
            or cleanup["ownership_sha256"] != _sha(ownership_path)
            or cleanup["board_before_sha256"] != ownership["routed_board_sha256"]
            or cleanup["board_after_sha256"] != _sha(board_path)
            or (cleanup["removed"] and cleanup.get("native_readback") is not True)):
        raise ValueError("boundary join authority does not bind this cleanup and board")
    allowed = set(ownership.get("interface_net_names", ()))
    if not allowed:
        return {}
    retired = {row["native_uuid"] for branch in cleanup["removed"] for row in branch["items"]}
    groups = defaultdict(list)
    for kind in ("traces", "vias"):
        for row in ownership[kind]:
            owner = row["escape_ownership"]
            if owner.get("owner") != "compiler.signal_escape" or row["net_name"] not in allowed:
                continue
            uid = row["native_uuid"]
            if uid in retired:
                continue
            if kind == "traces":
                item = {
                    "uuid": uid, "kind": "track", "net": row["net_name"], "layer": row["layer"],
                    "start_mm": (row["start_x"], row["start_y"]),
                    "end_mm": (row["end_x"], row["end_y"]), "width_mm": row["width"],
                }
            else:
                item = {
                    "uuid": uid, "kind": "via", "net": row["net_name"],
                    "position_mm": (row["x"], row["y"]), "size_mm": row["size"], "drill_mm": row["drill"],
                }
            groups[(owner["branch_id"], owner["ref"], str(owner["pad"]))].append(
                (int(owner["branch_item_index"]), item)
            )
    summary = {"owned_signal_escapes": [
        {"owner": "compiler.signal_escape", "branch_id": group[0], "ref": group[1], "pad": group[2],
         "items": [item for _, item in sorted(items)]}
        for group, items in groups.items()
    ]}
    result = _finish_owned_escape_landings(
        board_path, (summary,), allowed, ownership["boundary_landing_constraints"],
    )
    if result:
        result["ownership_sha256"] = cleanup["ownership_sha256"]
        result["interface_net_names"] = sorted(allowed)
    return result


if __name__ == "__main__":
    if sys.argv[1] == "--router-delta":
        _router_delta_native(*sys.argv[2:5])
    elif sys.argv[1] == "--readback-router-delta":
        original, candidate, proof_path = sys.argv[2:5]
        proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        _verify_saved_copper_removals(original, candidate, {row["uuid"] for row in proof["removed"]})
    elif sys.argv[1] == "--readback-escapes":
        _readback_signal_cleanup(*sys.argv[2:5])
    else:
        _cleanup_native(*sys.argv[1:4])
