"""Plan and materialize identical-leaf reuse for the leaf solve.

Boards with repeated identical functionality (a four-channel relay board's
``OPTO CH1..CH4`` / ``RELAY CH1..CH4``) synthesize each channel as its own sheet
-- byte-identical bar the ref designators and channel-numbered net names. Solving
every channel independently wastes compute and, worse, produces four *different*
placements of the same circuit. This module groups the structurally-identical
leaves so the solve runs ONCE per class and each sibling reuses the
representative's geometry verbatim (its own refs/nets, identical layout).

Two halves:

* :func:`plan_leaf_replication` — group the leaves being solved this invocation
  by structural isomorphism (cheap footprint-multiset bucket, then the strict
  :func:`build_replication_maps` check that also yields the per-sibling
  ``(ref_map, net_map)``). Returns one :class:`LeafGroup` per class; the solve
  drives only ``group.representative``.
* :func:`materialize_sibling` — after the representative is solved+persisted,
  write each sibling's artifact dir (``solved_layout.json`` via
  :func:`remap_solved_layout`, rewritten ``metadata.json``, copied ``debug.json``,
  and a ref-remapped ``layout.kicad_pcb`` blocker board). Compose then discovers
  the sibling dirs on disk exactly as if they had been solved.

Everything is reuse of already-tested primitives (``_replicate_leaves`` and
``leaf_library.sexpr_edit.renumber_pcb_text``); the strict structural check makes
the optimisation fail-safe -- a pair that is not truly identical falls back to an
independent solve, so a board can never be corrupted.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.cli._replicate_leaves import (
    _component_footprint_signature,
    build_replication_maps,
    remap_solved_layout,
)


@dataclass
class LeafGroup:
    """One structural-isomorphism class among the leaves being solved.

    ``representative`` is solved normally; each ``members`` entry is a sibling
    whose geometry is reused from the representative via ``(ref_map, net_map)``.
    An empty ``members`` list is the common case (a leaf with no identical twin)
    and behaves exactly like today.
    """

    representative: Any  # HierarchyNode
    members: list[tuple[Any, dict[str, str], dict[str, str]]] = field(
        default_factory=list
    )


def _leaf_structure(
    node: Any, board_state: Any
) -> tuple[list[str], dict[str, dict]] | None:
    """Own-component refs (schematic order) + serialized component dicts.

    Read straight from the full board state -- the isomorphism is a
    schematic-topology property, so no leaf extraction/solve is needed. Returns
    ``None`` if the leaf has no resolvable components.
    """
    from kicraft.autoplacer.brain.subcircuit_artifacts import serialize_component

    ordered_refs = [
        ref
        for ref in node.definition.component_refs
        if ref in board_state.components
    ]
    if not ordered_refs:
        return None
    components = {
        ref: serialize_component(board_state.components[ref]) for ref in ordered_refs
    }
    return ordered_refs, components


def _bucket_key(components: dict[str, dict]) -> tuple:
    """Cheap, order-independent pre-filter: the multiset of footprint signatures.

    Two leaves can only be isomorphic if they hold the same bag of footprints;
    the strict :func:`build_replication_maps` check confirms topology within a
    bucket. Keeps the pairwise check off unrelated leaves.
    """
    return tuple(sorted(_component_footprint_signature(c) for c in components.values()))


def plan_leaf_replication(
    leaves: list[Any], board_state: Any, cfg: dict[str, Any]
) -> list[LeafGroup]:
    """Group ``leaves`` (this invocation's leaf nodes) by structural isomorphism.

    Representatives appear in first-seen (input) order so seed assignment stays
    stable for boards with no repeats. Kill switch: ``cfg['leaf_replication']``
    (default on). Any leaf whose structure can't be read, or that matches no
    earlier representative, becomes its own single-member group -- identical
    behaviour to today.
    """
    if not cfg.get("leaf_replication", True) or len(leaves) < 2:
        return [LeafGroup(node) for node in leaves]

    # Fail-safe: any unexpected data shape must never break the solve -- degrade
    # to "no replication" (every leaf its own representative) rather than raise.
    try:
        return _plan_leaf_replication(leaves, board_state)
    except Exception as exc:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning(
            "leaf replication planning failed (%s); solving every leaf "
            "independently",
            exc,
        )
        return [LeafGroup(node) for node in leaves]


def _plan_leaf_replication(leaves: list[Any], board_state: Any) -> list[LeafGroup]:
    structs: dict[int, tuple[list[str], dict[str, dict]] | None] = {}
    groups: list[LeafGroup] = []
    by_bucket: dict[tuple, list[int]] = {}

    for node in leaves:
        struct = _leaf_structure(node, board_state)
        if struct is None:
            groups.append(LeafGroup(node))
            continue
        refs, comps = struct
        key = _bucket_key(comps)
        matched = False
        for gi in by_bucket.get(key, []):
            rep_struct = structs[gi]
            if rep_struct is None:
                continue
            rep_refs, rep_comps = rep_struct
            maps = build_replication_maps(rep_refs, refs, rep_comps, comps)
            if maps is not None:
                groups[gi].members.append((node, maps[0], maps[1]))
                matched = True
                break
        if not matched:
            structs[len(groups)] = struct
            by_bucket.setdefault(key, []).append(len(groups))
            groups.append(LeafGroup(node))

    return groups


def _sibling_identity(sib_node: Any) -> dict[str, Any]:
    sid = sib_node.id
    id_dict = {
        "sheet_name": sid.sheet_name,
        "sheet_file": sid.sheet_file,
        "instance_path": sid.instance_path,
        "parent_instance_path": sid.parent_instance_path,
    }
    return {**id_dict, "subcircuit_id": dict(id_dict)}


def materialize_sibling(
    rep_metadata: dict[str, Any],
    sib_node: Any,
    ref_map: dict[str, str],
    net_map: dict[str, str],
) -> dict[str, Any]:
    """Write ``sib_node``'s artifact dir by reusing the representative's geometry.

    ``rep_metadata`` is the representative's persisted ``metadata.json`` dict (as
    returned by ``_persist_solution``); its ``artifact_paths`` locate the rep's
    on-disk ``solved_layout.json`` / ``debug.json`` / ``mini_pcb``. Returns the
    sibling's metadata dict (same shape ``_persist_solution`` returns) so the
    caller can append it to the run summary.
    """
    from kicraft.autoplacer.brain.subcircuit_artifacts import resolve_artifact_paths
    from kicraft.autoplacer.brain.types import SubCircuitId

    rep_paths = rep_metadata.get("artifact_paths", {})
    project_dir = rep_metadata.get("project_dir", "")
    sib_paths = resolve_artifact_paths(
        project_dir,
        SubCircuitId(
            sheet_name=sib_node.id.sheet_name,
            sheet_file=sib_node.id.sheet_file,
            instance_path=sib_node.id.instance_path,
            parent_instance_path=sib_node.id.parent_instance_path,
        ),
    ).to_dict()
    sib_dir = Path(sib_paths["artifact_dir"])
    sib_dir.mkdir(parents=True, exist_ok=True)

    # --- solved_layout.json: the geometry reuse (refs+nets remapped) ---
    rep_solved_layout = json.loads(
        Path(rep_paths["solved_layout_json"]).read_text(encoding="utf-8")
    )
    sib_solved_layout = remap_solved_layout(
        rep_solved_layout, ref_map, net_map, _sibling_identity(sib_node)
    )
    sib_solved_layout["artifact_paths"] = dict(sib_paths)
    Path(sib_paths["solved_layout_json"]).write_text(
        json.dumps(sib_solved_layout, indent=2, default=str), encoding="utf-8"
    )

    # --- layout.kicad_pcb: ref-only remap (blocker board; nets never read) ---
    # The composer reads the mini_pcb ONLY for blocker geometry keyed by refdes
    # (pad rects, connector edge anchors) -- so the sibling's must carry the
    # sibling's refs; net names are irrelevant there and left as-is.
    from kicraft.leaf_library.sexpr_edit import renumber_pcb_text

    sib_mini_pcb = str(sib_dir / "layout.kicad_pcb")
    rep_mini_pcb = rep_paths.get("mini_pcb", "")
    if rep_mini_pcb and Path(rep_mini_pcb).exists():
        rep_pcb_text = Path(rep_mini_pcb).read_text(encoding="utf-8")
        sib_pcb_text, _counts = renumber_pcb_text(rep_pcb_text, ref_map)
        Path(sib_mini_pcb).write_text(sib_pcb_text, encoding="utf-8")
    else:
        sib_mini_pcb = ""

    # --- leaf_routed.kicad_pcb: ref-only remap of the rep's routed board. The
    # manual layout editor keys on this file (discover_leaves skips dirs
    # without it, the canvas PNG renders from it, parse_edge_cuts_aabb reads
    # its Edge.Cuts) and the rescue banner checks its existence -- without it
    # repeated-channel boards lose their siblings in the editor. Like the
    # mini_pcb, nothing electrical is read from it (the composer stamps from
    # solved_layout.json), so net names inside stay the representative's.
    rep_artifact_dir = rep_paths.get("artifact_dir", "")
    rep_leaf_routed = Path(rep_artifact_dir) / "leaf_routed.kicad_pcb"
    if rep_artifact_dir and rep_leaf_routed.is_file():
        routed_text, _counts = renumber_pcb_text(
            rep_leaf_routed.read_text(encoding="utf-8"), ref_map
        )
        (sib_dir / "leaf_routed.kicad_pcb").write_text(
            routed_text, encoding="utf-8"
        )

    # --- metadata.json: sibling identity + paths + the maps (so the post-pin
    # finalize can re-materialize from the representative's PINNED round). ---
    sib_metadata = copy.deepcopy(rep_metadata)
    sib_metadata.update(_sibling_identity(sib_node))
    paths_out = dict(sib_paths)
    paths_out["mini_pcb"] = sib_mini_pcb
    sib_metadata["artifact_paths"] = paths_out
    sib_metadata["replicated_from"] = rep_metadata.get("instance_path")
    sib_metadata["replication_ref_map"] = dict(ref_map)
    sib_metadata["replication_net_map"] = dict(net_map)
    Path(sib_paths["metadata_json"]).write_text(
        json.dumps(sib_metadata, indent=2, default=str), encoding="utf-8"
    )

    # --- debug.json: must EXIST (compose requires the file) but its content is
    # bypassed when solved_layout is present. Write a MINIMAL payload -- crucially
    # WITHOUT the representative's ``all_rounds`` -- so _auto_pin_best_leaves treats
    # the sibling as "no scoreable rounds" and skips it (siblings have no round
    # snapshots to pin anyway; the finalize re-materializes them post-pin). ---
    debug_payload = {**_sibling_identity(sib_node), "replicated_from": rep_metadata.get("instance_path")}
    Path(sib_paths["debug_json"]).write_text(
        json.dumps(debug_payload, indent=2, default=str), encoding="utf-8"
    )

    return sib_metadata


def finalize_leaf_replication(project_dir: str | Path) -> int:
    """Re-materialize every replicated sibling from its representative's PINNED
    artifact. Call AFTER per-leaf pinning (``_auto_pin_best_leaves``).

    Pinning copies a leaf's best round into its canonical ``solved_layout.json``,
    but a sibling has no round snapshots to pin -- it was left at whatever round
    last wrote it, which may not be the representative's finally-selected round.
    This pass reads each sibling's stored ``replication_ref_map`` / ``_net_map``
    and re-derives its ``solved_layout.json`` (and blocker ``mini_pcb``) from the
    representative's now-pinned geometry, so siblings stay byte-identical to the
    representative that actually ships. Disk-only and fully defensive -- any
    per-sibling error is logged and skipped. Returns the number refreshed.
    """
    import logging

    from kicraft.autoplacer.brain.subcircuit_artifacts import artifact_root_dir
    from kicraft.leaf_library.sexpr_edit import renumber_pcb_text

    log = logging.getLogger(__name__)
    root = artifact_root_dir(project_dir)
    if not root.is_dir():
        return 0

    by_instance: dict[str, Path] = {}
    sibling_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        sl_path = child / "solved_layout.json"
        if not (child.is_dir() and sl_path.is_file()):
            continue
        try:
            sl = json.loads(sl_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ip = sl.get("instance_path")
        if ip:
            by_instance[str(ip)] = child
        if sl.get("replicated_from"):
            sibling_dirs.append(child)

    refreshed = 0
    for sib_dir in sibling_dirs:
        try:
            sib_meta = json.loads((sib_dir / "metadata.json").read_text(encoding="utf-8"))
            rep_ip = sib_meta.get("replicated_from")
            ref_map = sib_meta.get("replication_ref_map") or {}
            net_map = sib_meta.get("replication_net_map") or {}
            rep_dir = by_instance.get(str(rep_ip))
            if rep_dir is None or not ref_map:
                continue
            rep_solved = json.loads(
                (rep_dir / "solved_layout.json").read_text(encoding="utf-8")
            )
            sib_identity = {
                k: sib_meta.get(k)
                for k in ("sheet_name", "sheet_file", "instance_path", "parent_instance_path")
            }
            if isinstance(sib_meta.get("subcircuit_id"), dict):
                sib_identity["subcircuit_id"] = sib_meta["subcircuit_id"]
            new_sib = remap_solved_layout(rep_solved, ref_map, net_map, sib_identity)
            new_sib["artifact_paths"] = sib_meta.get("artifact_paths", {})
            (sib_dir / "solved_layout.json").write_text(
                json.dumps(new_sib, indent=2, default=str), encoding="utf-8"
            )

            # Re-derive the blocker mini_pcb from the rep's now-pinned board.
            rep_meta_path = rep_dir / "metadata.json"
            sib_mini = sib_meta.get("artifact_paths", {}).get("mini_pcb", "")
            if rep_meta_path.is_file() and sib_mini:
                rep_meta = json.loads(rep_meta_path.read_text(encoding="utf-8"))
                rep_mini = rep_meta.get("artifact_paths", {}).get("mini_pcb", "")
                if rep_mini and Path(rep_mini).exists():
                    text, _c = renumber_pcb_text(
                        Path(rep_mini).read_text(encoding="utf-8"), ref_map
                    )
                    Path(sib_mini).write_text(text, encoding="utf-8")

            # Re-derive the sibling's leaf_routed.kicad_pcb (the editor-facing
            # artifact: leaf discovery, canvas render, Edge.Cuts AABB) from the
            # rep's now-pinned routed board. Ref-only remap, same contract as
            # the mini_pcb above.
            rep_routed = rep_dir / "leaf_routed.kicad_pcb"
            if rep_routed.is_file():
                text, _c = renumber_pcb_text(
                    rep_routed.read_text(encoding="utf-8"), ref_map
                )
                (sib_dir / "leaf_routed.kicad_pcb").write_text(
                    text, encoding="utf-8"
                )
            refreshed += 1
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(
                "leaf replication finalize failed for %s: %s", sib_dir.name, exc
            )
    return refreshed
