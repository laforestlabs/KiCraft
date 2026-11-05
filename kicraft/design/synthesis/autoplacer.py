"""Emit `<PROJECT>_autoplacer.json` from architecture + BOM slots.

Straight serialization of the state into the schema KiCraft already reads
(see `docs/kicraft_schematic_prompt.md` §4 and the existing
`LLUPS_autoplacer.json` for shape).

When the architecture contains library-backed sheets, the synthesis
stage passes in the merged library fragments and a ``library_leaves``
map; both flow into the emitted JSON.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from ..models import Architecture, BOM

# Edge-mount connector footprint families that mate off-board and therefore
# belong at a board edge. Matched (fnmatch, case-insensitive) against a
# BomPart.footprint "Library:Name". When such a connector has no
# component_zone, synthesis injects {edge: DEFAULT_EDGE_CONNECTOR_ZONE} so the
# composer pins it to the board edge even if the BOM stage didn't say so.
# Pure name match; orientation is resolved downstream (detect_opening_direction
# + _best_rotation_for_edge). See .agents/skills/kicraft/stages/bom.md.
DEFAULT_EDGE_CONNECTOR_ZONE = "bottom"
_EDGE_CONNECTOR_FOOTPRINT_PATTERNS = (
    "*usb_c_receptacle*",
    "*usb_c_plug*",
    "*usb_a_*",
    "*usb_b_*",
    "*usb-c*",  # vendored easyeda equivalents, e.g. USB-C_SMD-TYPE-C-31-M-12
    "*type-c*",
    "*barreljack*",
    "*barrel_jack*",
    "*dc-in-th*",  # vendored easyeda barrel jacks, e.g. DC-IN-TH_DC005-5.5-2.1
    # Screw terminals: 90-degree wire-entry blocks that MUST face a board
    # edge (KC-YJ7Q69 shipped with the wire mouths parallel to the edge; the
    # facing gate only covers ZONED refs, so the zone must never depend on
    # the BOM stage remembering to set one).
    "*screw-terminal*",
    "*wj126v*",
    "*terminal_block*",
    "*terminalblock*",
)


def _edge_connector_zone_injections(
    parts: Iterable[tuple[str, str]],
    existing: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Edge zones to inject for edge-mount connectors that lack one.

    Returns ``{ref: {"edge": DEFAULT_EDGE_CONNECTOR_ZONE}}`` for every
    ``(ref, footprint)`` whose footprint matches an edge-mount family and whose
    ref is not already in ``existing``. Defense-in-depth so an off-board
    connector can't float when the LLM forgot to zone it.
    """
    out: dict[str, dict[str, str]] = {}
    for ref, footprint in parts:
        if ref in existing:
            continue
        fp = (footprint or "").lower()
        if any(fnmatch(fp, pat) for pat in _EDGE_CONNECTOR_FOOTPRINT_PATTERNS):
            out[ref] = {"edge": DEFAULT_EDGE_CONNECTOR_ZONE}
    return out

_ALL_CONNECTORS_ONE_EDGE = re.compile(
    r"\ball\s+(?:the\s+)?connectors\b[^.;]{0,80}\b(?:on|along|at)\s+"
    r"(?:the\s+)?(?:one|same|single)\s+(?:board\s+)?edge\b",
    re.IGNORECASE,
)


def _same_edge_connector_constraint(constraints: Iterable[str]) -> bool:
    """Recognize an explicit positive constraint, never model-authored assumptions."""
    for constraint in constraints:
        text = str(constraint)
        for match in _ALL_CONNECTORS_ONE_EDGE.finditer(text):
            prefix = re.split(r"[.;]", text[:match.start()])[-1]
            if not re.search(r"\b(?:not|never|without|avoid)\b", prefix, re.IGNORECASE):
                return True
    return False


def required_same_edge_connector_refs(intent, bom: BOM) -> tuple[str, ...] | None:
    """Return all physical connector refs for the explicit original constraint."""
    if intent is None or not _same_edge_connector_constraint(intent.constraints):
        return None
    from kicraft.design.part_identity import physical_inventory_record

    refs = []
    for part in bom.parts:
        record = physical_inventory_record(mpn=part.mpn, symbol=part.symbol, footprint=part.footprint)
        features = record.physical_features if record is not None else ()
        if (re.fullmatch(r"J[0-9]+", part.ref) or any(
            feature in {"connector", "header", "pin-header", "screw-terminal", "terminal-block"}
            or feature.endswith("-connector") for feature in features
        )):
            refs.append(part.ref)
    return tuple(sorted(refs))


def _same_edge_connector_zones(refs, component_zones, placement):
    """Zone every connector named by the original all-one-edge contract."""
    if not refs:
        raise ValueError("all-connectors-one-edge constraint has no connector references")
    explicit = {
        ref: spec for ref, spec in (getattr(placement, "component_zones", None) or {}).items()
        if ref in refs
    }
    if any("corner" in spec or "zone" in spec for spec in explicit.values()):
        raise ValueError("all-connectors-one-edge conflicts with an explicit connector zone")
    explicit_edges = {spec["edge"] for spec in explicit.values() if "edge" in spec}
    if len(explicit_edges) > 1:
        raise ValueError("all-connectors-one-edge conflicts with explicit connector edges")
    edge = next(iter(explicit_edges), DEFAULT_EDGE_CONNECTOR_ZONE)
    for ref in refs:
        # The hard original brief takes precedence over model-authored hints.
        old = component_zones.get(ref, {})
        component_zones[ref] = {"edge": edge, **({"rotation": old["rotation"]} if "rotation" in old else {})}
    return {"edge": edge, "refs": list(refs)}


def _recipe_local_bypasses(architecture: Architecture, bom: BOM) -> list[dict]:
    """Resolve reviewed recipe pin contracts from selection, manifest and wiring."""
    from kicraft.design.recipes.registry import expand_recipe, get_recipe

    actual = {}
    for connection in bom.connections:
        for endpoint in connection.endpoints:
            actual.setdefault((endpoint.ref, endpoint.pin), []).append(connection.net_name)
    result = []
    for selection in architecture.recipe_selections:
        try:
            definition = get_recipe(selection.recipe)
        except (KeyError, ValueError):
            continue
        rules = [r for r in definition.placement_constraints if r.kind == "owned_bypass"]
        if not rules:
            continue
        manifests = [m for m in bom.recipe_ownership
                     if m.recipe == selection.recipe and m.instance == selection.instance]
        if len(manifests) != 1:
            raise ValueError(f"{selection.instance}: missing canonical bypass recipe ownership")
        manifest = manifests[0]
        if (manifest.port_bindings != selection.port_bindings or
                set(manifest.requirement_ids) != set(selection.requirement_ids)):
            raise ValueError(f"{selection.instance}: bypass recipe ownership bindings mismatch")
        parts = [p for p in bom.parts if p.recipe_id == selection.recipe
                 and p.recipe_instance == selection.instance and p.resolution_source == "recipe"]
        if {p.ref for p in parts} != set(manifest.refs):
            raise ValueError(f"{selection.instance}: bypass recipe component ownership mismatch")
        # Re-expand the selected recipe through the producer, not the saved
        # manifest: internal nets are instance-scoped by that same producer.
        canonical = expand_recipe(selection)
        canonical_roles = {p.recipe_role: p.ref for p in canonical.parts}
        canonical_pins = {(p.ref, p.pin): p.net for p in canonical.ownership.pins}
        ports = {port.name: port for port in definition.ports}
        declared_nets = set(ports) | set(definition.internal_nets)

        def role_part(role):
            groups = [g for g in definition.parts if g.role == role]
            members = [p for p in parts if p.recipe_role == role]
            if len(groups) != 1 or groups[0].quantity != 1 or len(members) != 1:
                raise ValueError(f"{selection.instance}: ambiguous recipe role {role}")
            group, part = groups[0], members[0]
            if (part.value, part.symbol, part.footprint, part.mpn) != (
                    group.value, group.symbol, group.footprint, group.mpn):
                raise ValueError(f"{selection.instance}: altered recipe role {role}")
            return part

        for rule in rules:
            owner = role_part(str(rule.parameters["anchor_role"]))
            cap = role_part(str(rule.role))
            power, ground = str(rule.parameters["power_net"]), str(rule.parameters["ground_net"])

            def pin_numbers(role, logical):
                return [str(p.pin) for p in definition.pins if p.role == role and p.net == logical]

            op = pin_numbers(owner.recipe_role, power)
            og = pin_numbers(owner.recipe_role, ground)
            cp = pin_numbers(cap.recipe_role, power)
            cg = pin_numbers(cap.recipe_role, ground)
            if len(op) != 1 or not og or len(cp) != 1 or len(cg) != 1:
                raise ValueError(f"{selection.instance}: invalid reviewed bypass pins")
            for ref, numbers, logical in ((owner.ref, op, power), (owner.ref, og, ground),
                                           (cap.ref, cp, power), (cap.ref, cg, ground)):
                if logical not in declared_nets:
                    raise ValueError(f"{selection.instance}: unknown bypass net {logical}")
                port = ports.get(logical)
                if (logical not in selection.port_bindings
                        and logical not in definition.internal_nets
                        and (port is None or not port.default_tie
                             or not selection.port_bindings.get(port.default_tie))):
                    raise ValueError(f"{selection.instance}: unbound bypass net {logical}")
                role = owner.recipe_role if ref == owner.ref else cap.recipe_role
                canonical_ref = canonical_roles.get(role)
                for number in numbers:
                    net = canonical_pins.get((canonical_ref, number))
                    rows = [p for p in manifest.pins if (p.ref, p.pin) == (ref, number)]
                    row = rows[0] if len(rows) == 1 else None
                    if (not net or row is None or row.owner != "recipe"
                            or row.owner_id != selection.recipe
                            or (row.requirement_id not in selection.requirement_ids
                                if selection.requirement_ids else row.requirement_id is not None)
                            or row.net != net or actual.get((ref, number)) != [net]):
                        raise ValueError(f"{selection.instance}: unproven bypass pin {ref}.{number}")
            result.append({
                "owner_ref": owner.ref, "bypass_ref": cap.ref,
                "power_pins": [op[0], cp[0]], "ground_pins": [og, cg[0]],
                "max_distance_mm": float(rule.parameters["max_mm"]),
            })
    return result


def write_autoplacer_json(
    project_dir: Path,
    project_stem: str,
    architecture: Architecture,
    bom: BOM,
    *,
    library_fragments: dict[str, Any] | None = None,
    library_leaves: dict[str, dict[str, Any]] | None = None,
    placement=None,
    form_factor=None,
    intent=None,
) -> Path:
    """Write `<project_stem>_autoplacer.json` to project_dir. Returns the path.

    Merge precedence, lowest to highest: library fragments < the BOM's
    LLM-derived hints < the user's ``placement`` section. The generated
    file is a build artifact: hand edits are overwritten on the next
    synthesis, so durable user rules belong in ``state.placement``.

    Args:
        library_fragments: dict already merged from every library leaf's
            ``autoplacer_fragment.json`` (renumbered). Keys like
            ``ic_groups``, ``thermal_refs`` are unioned with the
            project's BOM-derived values.
        library_leaves: ``{sheet_name: InstalledLeaf.to_library_leaves_entry()}``
            written as a top-level audit record.
        placement: the state's ``PlacementSection`` (user rules) or None.
            Rules naming refs the BOM no longer carries are dropped with
            a warning (parts churn across BOM re-runs; a stale rule must
            not fail the §9.6 named-refs check).
    """
    out = project_dir / f"{project_stem}_autoplacer.json"
    body: dict[str, object] = {
        "project_name": project_stem,
        "pcb_file": f"{project_stem}.kicad_pcb",
        "power_nets": sorted(set(architecture.power_nets)),
    }
    # Reconstruct compiler-owned constraints, rather than accepting candidate
    # metadata. The canonical check includes all roles and their wired nets.
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.synthesis.validation import _canonical_lowerer_refs

    local_bypasses = []
    for requirement in architecture.requirements:
        try:
            artifact = lower_requirement(requirement)
        except ValueError:
            # Other requirements may be recipe-owned rather than realizable
            # by a registered lowerer. Do not change their export behavior.
            if any(p.resolution_source == "lowerer" and
                   p.lowering_requirement_id == requirement.id for p in bom.parts):
                raise
            continue
        if artifact is None or not artifact.local_bypasses:
            continue
        refs = _canonical_lowerer_refs(requirement, bom)
        if not refs:
            raise ValueError(f"{requirement.id}: local bypass lacks canonical BOM/wiring ownership")
        by_role = {
            (part.lowering_role, part.lowering_index): part.ref
            for part in bom.parts if part.ref in refs
        }
        for local in artifact.local_bypasses:
            local_bypasses.append({
                "owner_ref": by_role[(local.owner_role, local.owner_index)],
                "bypass_ref": by_role[(local.bypass_role, local.bypass_index)],
                "power_pins": list(local.power_pins),
                "ground_pins": list(local.ground_pins),
                "max_distance_mm": local.max_distance_mm,
            })
    local_bypasses.extend(_recipe_local_bypasses(architecture, bom))
    if local_bypasses:
        body["local_bypasses"] = local_bypasses

    # Start with BOM-derived values; fold in library fragments additively.
    ic_groups: dict[str, list[str]] = {ic: list(members) for ic, members in bom.ic_groups.items()}
    group_labels: dict[str, str] = dict(bom.group_labels)
    thermal_refs: list[str] = list(bom.thermal_refs)
    signal_flow_order: list[str] = list(bom.signal_flow_order)
    component_zones: dict[str, dict[str, str]] = {
        ref: dict(spec) for ref, spec in bom.component_zones.items()
    }

    if library_fragments:
        for k, v in (library_fragments.get("ic_groups") or {}).items():
            if k not in ic_groups:
                ic_groups[k] = list(v)
        for k, v in (library_fragments.get("group_labels") or {}).items():
            group_labels.setdefault(k, v)
        for r in library_fragments.get("thermal_refs", []) or []:
            if r not in thermal_refs:
                thermal_refs.append(r)
        for r in library_fragments.get("signal_flow_order", []) or []:
            if r not in signal_flow_order:
                signal_flow_order.append(r)
        for k, v in (library_fragments.get("component_zones") or {}).items():
            component_zones.setdefault(k, dict(v))

    # Defense-in-depth: auto-derive an edge zone for edge-mount connectors the
    # BOM left unzoned (off-board USB / barrel jacks must sit at a board edge).
    for ref, spec in _edge_connector_zone_injections(
        [(p.ref, p.footprint) for p in bom.parts], component_zones
    ).items():
        component_zones[ref] = spec

    # User placement rules: highest precedence. Stale refs are dropped
    # (with a warning) rather than emitted, because §9.6 fails synthesis
    # for any autoplacer.json ref absent from the schematic.
    backside_leaves: list[str] = []
    if placement is not None:
        known_refs = {p.ref for p in bom.parts}
        dropped: list[str] = []
        for ref, spec in (placement.component_zones or {}).items():
            if ref in known_refs:
                component_zones[ref] = dict(spec)
            else:
                dropped.append(ref)
        for ref in placement.thermal_refs or []:
            if ref not in known_refs:
                dropped.append(ref)
            elif ref not in thermal_refs:
                thermal_refs.append(ref)
        backside_leaves = sorted(placement.backside_through_hole_leaves or [])
        if dropped:
            print(
                "[autoplacer] warning: dropping placement rule(s) for ref(s) "
                f"not in the BOM: {', '.join(sorted(set(dropped)))}"
            )

    required_refs = required_same_edge_connector_refs(intent, bom)
    if required_refs is not None:
        body["required_same_edge_connectors"] = _same_edge_connector_zones(
            required_refs, component_zones, placement
        )

    if ic_groups:
        body["ic_groups"] = ic_groups
    if group_labels:
        body["group_labels"] = group_labels
    if thermal_refs:
        body["thermal_refs"] = thermal_refs
    if signal_flow_order:
        body["signal_flow_order"] = signal_flow_order
    if component_zones:
        body["component_zones"] = component_zones

    # Back-side parts: BomPart.side == "back" -> place on B.Cu. autoplacer.json
    # is the solver config (merged via config.discover_project_config), so the
    # adapter reads this map at board load and sets Component.layer = BACK; the
    # existing stamp path then flips the footprint. Replay-safe (no re-synthesis).
    component_layers = {p.ref: "back" for p in bom.parts if p.side == "back"}
    if component_layers:
        body["component_layers"] = component_layers

    # Matrix/array placement hints. The autoplacer grids these members
    # programmatically (serpentine) instead of running force/SA over them.
    # autoplacer.json is the project config, so this key merges straight into
    # the solver cfg (see autoplacer/config.discover_project_config).
    arrays = []
    for spec in bom.arrays:
        entry: dict[str, Any] = {
            "refs": list(spec.refs),
            "pattern": spec.pattern,
            "pitch_mm": spec.pitch_mm,
            "serpentine": spec.serpentine,
        }
        if spec.pattern == "ring":
            # rows/cols intentionally absent: consumers key ring handling off
            # "pattern" and int(spec.get("rows", 0)) must not see None.
            entry["radius_mm"] = spec.radius_mm
            entry["start_angle_deg"] = spec.start_angle_deg
        else:
            entry["rows"] = spec.rows
            entry["cols"] = spec.cols
        arrays.append(entry)
    if arrays:
        body["arrays"] = arrays

    if bom.edge_interfaces:
        body["edge_interfaces"] = [
            {
                "name": interface.name,
                "refs": list(interface.refs),
                "side": interface.side,
                "pitch_mm": interface.pitch_mm,
                "behavior": interface.behavior,
                "exact_edge_datum": True,
            }
            for interface in bom.edge_interfaces
        ]

    if library_leaves:
        body["library_leaves"] = library_leaves

    if backside_leaves:
        body["parent_placement"] = {"backside_through_hole_leaves": backside_leaves}

    # Fixed board dimensions (user-chosen) pin the solver's board and
    # disable the size search; otherwise the search stays on and derives
    # dimensions from leaf areas.
    board = placement.board if placement is not None else None
    if board is not None and board.width_mm and board.height_mm:
        body["board_width_mm"] = float(board.width_mm)
        body["board_height_mm"] = float(board.height_mm)
        body["enable_board_size_search"] = bool(board.size_search)
    else:
        body["enable_board_size_search"] = True

    # Requested non-rectangular outline shape (captured at the intent stage as
    # ``intent.form_factor``). Flows to the compose pipeline, which resolves it
    # to ``Edge.Cuts`` geometry. "rect" / absent emits nothing, leaving the
    # default rectangular path untouched. A future ``placement.board`` shape
    # override would take precedence over the captured intent here.
    if form_factor is not None and getattr(form_factor, "shape", "rect") not in (
        "",
        "rect",
        None,
    ):
        outline: dict[str, Any] = {"shape": form_factor.shape}
        if form_factor.corner_radius_mm is not None:
            outline["corner_radius_mm"] = float(form_factor.corner_radius_mm)
        if form_factor.chamfer_mm is not None:
            outline["chamfer_mm"] = float(form_factor.chamfer_mm)
        if form_factor.size_mm is not None:
            outline["size_mm"] = float(form_factor.size_mm)
        body["board_outline"] = outline

    # Named standard form factor (Arduino shield, ...): surface the template's
    # fixed geometry (outline + connector positions + holes) so the compose
    # pipeline can honor it. Consumers gate on the embedded ``validated`` flag
    # before laying a board out on the datum. ``form_factor_enforce`` is the
    # master switch that turns the compose placement fork ON -- it is the SAME
    # env gate (``KICRAFT_FORM_FACTOR_ENFORCE``) that drives the synthesis-side
    # replace & rewire reconcile, so both halves activate together: with it off,
    # emitting the template block alone changes no build behavior.
    standard_key = getattr(form_factor, "standard", None) if form_factor else None
    if standard_key:
        from kicraft.form_factors import get_template
        from kicraft.form_factors.reconcile import enforce_enabled

        template = get_template(standard_key)
        if template is not None:
            ff_block = template.to_autoplacer_dict()
            if enforce_enabled():
                body["form_factor_enforce"] = True
                # Map each template connector role to the actual BOM ref the
                # reconcile emitted (sourcing_note "standard form factor: <key>
                # <role>") so the compose scaffold locks the SAME parts the
                # schematic carries rather than guessing J1.. (refs get recycled/
                # renumbered). Only meaningful under enforcement -- the reconcile
                # that creates these parts runs on the same env gate.
                marker = f"standard form factor: {template.key} "
                header_refs = {
                    (p.sourcing_note or "")[len(marker) :].strip(): p.ref
                    for p in bom.parts
                    if (p.sourcing_note or "").startswith(marker)
                }
                if header_refs:
                    ff_block["header_refs"] = header_refs
            body["form_factor_standard"] = ff_block

    out.write_text(json.dumps(body, indent=2) + "\n")
    return out
