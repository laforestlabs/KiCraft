"""Owner-authorized signal-escape cleanup and dangling-copper audit.

The ownership document is produced at parent stamping, before routing.  It is
not reconstructed from lock state, net names, or router dead-end reports.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
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


def _cleanup_native(board_path: str, ownership_path: str, receipt_path: str) -> dict:
    import pcbnew
    from kicraft.autoplacer.hardware.local_contract import _copper_graph, _position

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

    def pad_anchors(net):
        # Graph layer nodes depend on touching tracks. Physical pad anchors do
        # not disappear when the last unused stub leaves an SMD land.
        layers = tuple(board.GetEnabledLayers().CuStack())
        result = []
        index = 0
        for footprint in board.GetFootprints():
            for pad in footprint.Pads():
                if pad.GetNetname() != net:
                    continue
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    result.append(("pad", index))
                else:
                    result.extend(("pad", index, layer)
                                  for layer in layers if pad.IsOnLayer(layer))
                index += 1
        return result

    def pt(x, y):
        return (int(pcbnew.FromMM(float(x))), int(pcbnew.FromMM(float(y))))
    def stamped_key(row, kind):
        if kind == "track":
            ends = sorted((pt(row["start_x"], row["start_y"]), pt(row["end_x"], row["end_y"])))
            return (kind, row["net_name"], row["layer"], *ends, int(pcbnew.FromMM(row["width"])))
        return (kind, row["net_name"], pt(row["x"], row["y"]),
                int(pcbnew.FromMM(row["size"])), int(pcbnew.FromMM(row["drill"])))
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
    for item in board.GetTracks():
        native[native_key(item)].append(item)
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
    for group, entries in groups.items():
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
        if any(not z.GetIsRuleArea() and z.GetNetname() == net for z in board.Zones()):
            retained.append({"branch": list(group), "reason": "same_net_zone_not_in_graph"})
            continue
        try:
            board.BuildConnectivity()
            connectivity = board.GetConnectivity()
            all_ids = {item.m_Uuid.AsString() for item in items}
            contacts = {}
            for item in items:
                for other in connectivity.GetConnectedTracks(item):
                    uid = other.m_Uuid.AsString()
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
                # A routed escape track can be essential while its terminal via
                # contacts copper on only one face.  Consider only individually
                # owned vias, after validating the entire branch receipt above.
                candidates = [[(item, entry)] for item, entry in zip(items, entries)
                              if isinstance(item, pcbnew.PCB_VIA)]
                if not candidates:
                    retained.append({"branch": list(group), "reason": "not_a_single_pad_dead_branch",
                                     "contacts": contacts})
                    continue
            for candidate in candidates:
                selected = [item for item, _ in candidate]
                selected_ids = {item.m_Uuid.AsString() for item in selected}
                contact_layers = set()
                if not whole_dead:
                    board.BuildConnectivity()
                    connectivity = board.GetConnectivity()
                    via = selected[0]
                    enabled_layers = tuple(board.GetEnabledLayers().CuStack())
                    for other in connectivity.GetConnectedTracks(via):
                        if other.m_Uuid.AsString() in selected_ids:
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
                graph, _, _ = _copper_graph(board, net)
                pad_nodes = pad_anchors(net)
                anchors = list(pad_nodes)
                for other in board.GetTracks():
                    if other.GetNetname() != net or other.m_Uuid.AsString() in selected_ids:
                        continue
                    if isinstance(other, pcbnew.PCB_VIA):
                        pos = _position(other.GetPosition())
                        anchors.extend((pos, layer) for layer in (pcbnew.F_Cu, pcbnew.B_Cu)
                                       if other.IsOnLayer(layer))
                    elif other.GetClass() == "PCB_TRACK":
                        anchors.extend(((_position(other.GetStart()), other.GetLayer()),
                                        (_position(other.GetEnd()), other.GetLayer())))
                    else:
                        raise ValueError("unsupported non-linear copper")
                before = _partition(graph, anchors)
                for item in selected:
                    board.Remove(item)
                try:
                    graph_after, _, _ = _copper_graph(board, net)
                    after = _partition(graph_after, anchors)
                    valid = pad_anchors(net) == pad_nodes and before == after
                except Exception:
                    for item in selected:
                        board.Add(item)
                    raise
                if not valid:
                    for item in selected:
                        board.Add(item)
                    retained.append({"branch": list(group), "reason": "topology_changed"})
                    continue
                removed.append({"branch": list(group), "net": net,
                                "items": [{"native_uuid": item.m_Uuid.AsString(), **entry[1]}
                                          for item, entry in candidate],
                                "external_contacts": contacts,
                                "proof": {"pad_nodes": len(pad_nodes), "anchors": len(anchors),
                                          "partition_before": before, "partition_after": after,
                                          "same_net_zone_absent": True,
                                          "single_layer_via_contacts": sorted(
                                              board.GetLayerName(layer) for layer in contact_layers
                                          ) if not whole_dead else None}})
                for _, (key, _) in candidate:
                    native[key] = []
        except Exception as exc:
            retained.append({"branch": list(group), "reason": "proof_failed", "detail": str(exc)})
    if removed:
        board.Save(board_path)
        from kicraft.autoplacer.hardware.history_geometry import close_native_history
        capture_board(board, phase="signal_escape_cleanup")
        close_native_history()
    receipt = {"version": 1, "ownership_sha256": _sha(ownership_path),
               "board_before_sha256": before_sha, "board_after_sha256": _sha(board_path),
               "accepted": not any(row["reason"] in {
                   "incomplete_or_ambiguous_ownership", "mixed_nets", "proof_failed"
               } for row in retained),
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
    result = subprocess.run([sys.executable, "-m", __name__, str(board_path),
                             str(ownership_path), str(receipt_path)],
                            capture_output=True, text=True, env=_kicad_subprocess_env())
    if result.returncode:
        raise RuntimeError(f"signal escape cleanup failed: {result.stderr}")
    return json.loads(Path(receipt_path).read_text())


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


if __name__ == "__main__":
    _cleanup_native(*sys.argv[1:4])
