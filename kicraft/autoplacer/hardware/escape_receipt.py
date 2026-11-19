"""Read compiler escape ownership back from the actual stamped PCB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pcbnew


def write_escape_receipt(path: str, board, owners: dict[str, tuple[str, dict]]) -> None:
    """Bind owners by native UUID at stamp time; never infer new ownership."""
    traces, vias = [], []
    remaining = set(owners)
    if remaining:
        for item in board.GetTracks():
            uuid = item.m_Uuid.AsString()
            if uuid not in remaining:
                continue
            net, owner = owners[uuid]
            if item.GetNetname() != net:
                raise RuntimeError(f"escape stamp changed the net for {owner['source_uuid']}")
            remaining.remove(uuid)
            entry = {"net_name": net, "escape_ownership": owner, "native_uuid": uuid}
            if isinstance(item, pcbnew.PCB_VIA):
                pos = item.GetPosition()
                try:
                    width = item.GetWidth(pcbnew.F_Cu)
                except TypeError:
                    width = item.GetWidth()
                vias.append(dict(entry, x=pcbnew.ToMM(pos.x), y=pcbnew.ToMM(pos.y),
                                 size=pcbnew.ToMM(width), drill=pcbnew.ToMM(item.GetDrill())))
            else:
                start, end = item.GetStart(), item.GetEnd()
                traces.append(dict(entry, start_x=pcbnew.ToMM(start.x), start_y=pcbnew.ToMM(start.y),
                                   end_x=pcbnew.ToMM(end.x), end_y=pcbnew.ToMM(end.y),
                                   width=pcbnew.ToMM(item.GetWidth()), layer=item.GetLayerName()))
    if remaining:
        raise RuntimeError(f"stamped board lost compiler escape UUIDs: {sorted(remaining)}")
    pcb = Path(path)
    receipt = {
        "version": 1,
        "stamped_board_sha256": hashlib.sha256(pcb.read_bytes()).hexdigest(),
        "stamped_board_path": str(pcb), "traces": traces, "vias": vias,
    }
    Path(f"{path}.escape_ownership.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
