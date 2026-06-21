"""Replicate a solved leaf's geometry onto its structurally-identical siblings.

When the architecture marks N from-scratch sheets as one ``replication_group``
(e.g. ``STEPPER_AXIS_X/Y/Z``), the layout solves ONLY the representative and
reuses its placement+routing for the rest -- far less compute and an identical,
predictable layout per copy. Each sibling keeps its OWN refs and nets (so ERC
sees N independent circuits with no shorts between them); only the geometry is
shared.

This module is the *pure* remapper: given the representative's ``solved_layout``
dict and the ``(ref_map, net_map)`` correspondence to a sibling, it returns the
sibling's ``solved_layout`` dict -- identical geometry, sibling refs/nets. The
maps are derived from the two leaves' component pad topology:

* ``ref_map`` pairs components by position in each sheet's ``component_refs``
  list (stable schematic-traversal order for identically-generated sheets),
  verified to have matching footprints.
* ``net_map`` matches each representative net to the sibling net that connects
  the *same* pads after ``ref_map`` -- so a shared rail (``+12V``/``GND``, same
  pads either side) maps to itself, while a per-instance signal (``STEP_X``)
  maps to its sibling (``STEP_Y``). Topology, not name suffixes.

A strict structural check (:func:`build_replication_maps`) rejects any pair that
is not truly identical and returns ``None``; the caller then falls back to
solving the sibling independently, so the optimisation can never corrupt a
board -- worst case it just doesn't fire.
"""
from __future__ import annotations

import copy
from typing import Any


def _component_footprint_signature(comp: dict[str, Any]) -> tuple:
    """Placement- and net-invariant identity of one component's footprint.

    Two components are interchangeable for geometry reuse iff this matches:
    same value, body size, through-hole flag, and pad geometry (id + size).
    Deliberately excludes ``pos``/``rotation``/``net`` (those differ/are the
    thing being reused or remapped). ROTATION-INVARIANT: body w/h and each pad's
    x/y are sorted, because the solved layout stores POST-rotation geometry --
    the same footprint placed at 0 vs 90 deg has its width/height swapped, and
    that must still count as identical (the reuse overwrites rotation anyway).
    """
    pads = comp.get("pads", []) or []
    pad_shapes = tuple(sorted(
        (
            str(p.get("pad_id", "")),
            tuple(sorted((
                round(float((p.get("size_mm") or {}).get("x", 0.0)), 3),
                round(float((p.get("size_mm") or {}).get("y", 0.0)), 3),
            ))),
            str(p.get("layer", "")),
        )
        for p in pads
    ))
    wh = tuple(sorted((
        round(float(comp.get("width_mm", 0.0)), 3),
        round(float(comp.get("height_mm", 0.0)), 3),
    )))
    return (
        comp.get("value", ""),
        wh,
        bool(comp.get("is_through_hole")),
        pad_shapes,
    )


def _pads_by_net(components: dict[str, dict]) -> dict[str, frozenset[tuple[str, str]]]:
    """net name -> frozenset of ``(ref, pad_id)`` pads on that net."""
    out: dict[str, set[tuple[str, str]]] = {}
    for ref, comp in components.items():
        for pad in comp.get("pads", []) or []:
            net = pad.get("net")
            if not net:
                continue
            out.setdefault(net, set()).add((ref, str(pad.get("pad_id", ""))))
    return {net: frozenset(pads) for net, pads in out.items()}


def build_replication_maps(
    rep_refs: list[str],
    sib_refs: list[str],
    rep_components: dict[str, dict],
    sib_components: dict[str, dict],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Return ``(ref_map, net_map)`` for reusing rep's geometry on sib, or None.

    ``rep_refs``/``sib_refs`` are each sheet's components in stable schematic
    order. ``rep_components``/``sib_components`` are the per-ref dicts (with
    ``value``/``pads``). Returns None (caller solves the sibling independently)
    when the two leaves are not structurally identical: different component
    count, a footprint mismatch at any position, or a net whose pad topology has
    no exact counterpart.
    """
    if len(rep_refs) != len(sib_refs):
        return None
    ref_map: dict[str, str] = {}
    for r_ref, s_ref in zip(rep_refs, sib_refs):
        r_comp = rep_components.get(r_ref)
        s_comp = sib_components.get(s_ref)
        if r_comp is None or s_comp is None:
            return None
        if _component_footprint_signature(r_comp) != _component_footprint_signature(s_comp):
            return None
        ref_map[r_ref] = s_ref

    # net_map by pad topology: a rep net maps to the sib net connecting the
    # SAME pads after ref_map (shared rails map to themselves; per-instance
    # signals map to their sibling). Must be an exact bijection of pad-sets.
    rep_nets = _pads_by_net(rep_components)
    sib_nets = _pads_by_net(sib_components)
    sib_by_padset = {pads: net for net, pads in sib_nets.items()}
    if len(sib_by_padset) != len(sib_nets):
        return None  # ambiguous: two sib nets share a pad-set
    net_map: dict[str, str] = {}
    for rep_net, rep_pads in rep_nets.items():
        mapped = frozenset((ref_map[r], pid) for (r, pid) in rep_pads if r in ref_map)
        if len(mapped) != len(rep_pads):
            return None  # a pad's ref wasn't in ref_map
        sib_net = sib_by_padset.get(mapped)
        if sib_net is None:
            return None  # no sibling net with this exact topology
        net_map[rep_net] = sib_net
    return ref_map, net_map


def remap_solved_layout(
    rep_layout: dict[str, Any],
    ref_map: dict[str, str],
    net_map: dict[str, str],
    sib_identity: dict[str, Any],
) -> dict[str, Any]:
    """Return the sibling's ``solved_layout`` dict: rep's geometry, sib refs/nets.

    Deep-copies ``rep_layout`` and rewrites every ref- and net-bearing field
    (components + pads, traces, vias, interface anchors, ports) plus the
    identity fields from ``sib_identity`` (instance_path, sheet_name,
    sheet_file, subcircuit_id, parent_instance_path). Geometry (pos, rotation,
    trace/via coordinates, bounding_box) is preserved verbatim.
    """
    out = copy.deepcopy(rep_layout)

    def rn(net: str | None) -> str | None:
        return net_map.get(net, net) if net is not None else None

    def rr(ref: str | None) -> str | None:
        return ref_map.get(ref, ref) if ref is not None else None

    # components: rekey + rewrite ref/pad.ref/pad.net
    new_components: dict[str, dict] = {}
    for ref, comp in (out.get("components") or {}).items():
        new_ref = ref_map.get(ref, ref)
        comp["ref"] = new_ref
        for pad in comp.get("pads", []) or []:
            pad["ref"] = rr(pad.get("ref"))
            pad["net"] = rn(pad.get("net"))
        new_components[new_ref] = comp
    out["components"] = new_components

    for trace in out.get("traces", []) or []:
        trace["net"] = rn(trace.get("net"))
    for via in out.get("vias", []) or []:
        via["net"] = rn(via.get("net"))
    for anchor in out.get("interface_anchors", []) or []:
        pr = anchor.get("pad_ref")
        if isinstance(pr, (list, tuple)) and pr:
            anchor["pad_ref"] = [rr(pr[0]), *pr[1:]]
        anchor["port_name"] = rn(anchor.get("port_name"))
    for port in out.get("ports", []) or []:
        port["name"] = rn(port.get("name"))
        if "net_name" in port:
            port["net_name"] = rn(port.get("net_name"))

    # identity: this artifact IS the sibling now
    for key in (
        "instance_path", "sheet_name", "sheet_file", "parent_instance_path",
    ):
        if key in sib_identity:
            out[key] = sib_identity[key]
    if "subcircuit_id" in sib_identity:
        out["subcircuit_id"] = sib_identity["subcircuit_id"]
    out["replicated_from"] = rep_layout.get("instance_path")
    return out
