#!/usr/bin/env python3
"""Export solved leaf subcircuit artifacts from a KiCad schematic hierarchy.

This is a Milestone 3 helper CLI for the subcircuits redesign.

It parses the true-sheet schematic hierarchy, finds all leaf subcircuits,
extracts leaf-local board states from the project PCB, builds leaf extraction
records, solves local placement for each leaf, and writes artifact outputs
under `.experiments/subcircuits/`.

Current scope:
- hierarchy parsing
- leaf discovery
- leaf-local board-state extraction
- artifact path resolution
- metadata JSON export
- debug JSON export
- solved layout JSON export
- placement-only mini `.kicad_pcb` export

This command does not yet perform parent-level composition or high-fidelity
DSN/SES-based local autorouting. It prepares solved leaf artifacts so later
pipeline stages can load rigid child layouts directly.

Usage:
    python3 export_subcircuit_artifacts.py LLUPS.kicad_sch
    python3 export_subcircuit_artifacts.py LLUPS.kicad_sch --config config.json
    python3 export_subcircuit_artifacts.py LLUPS.kicad_sch --pcb LLUPS.kicad_pcb
    python3 export_subcircuit_artifacts.py LLUPS.kicad_sch --dry-run
    python3 export_subcircuit_artifacts.py LLUPS.kicad_sch --json
"""

from __future__ import annotations

import site
import sys
from pathlib import Path



def _ensure_kicad_python_path() -> None:
    """Ensure KiCad Python bindings (pcbnew) are importable."""
    try:
        import pcbnew  # noqa: F401

        return
    except Exception:
        pass

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        f"/usr/lib/python{ver}/site-packages",
        f"/usr/lib64/python{ver}/site-packages",
        "/usr/lib/python3/dist-packages",
        "/usr/lib64/python3/dist-packages",
    ]

    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        pcbnew_py = Path(path) / "pcbnew.py"
        pcbnew_pkg = Path(path) / "pcbnew"
        if pcbnew_py.exists() or pcbnew_pkg.exists():
            if path not in sys.path:
                sys.path.append(path)

    try:
        import pcbnew  # noqa: F401
    except Exception as exc:
        raise ModuleNotFoundError(
            "KiCad Python module 'pcbnew' not found. "
            "Install KiCad bindings or set PYTHONPATH to KiCad site-packages."
        ) from exc


_ensure_kicad_python_path()

import argparse
import json
from typing import Any

from kicraft.autoplacer.brain.hierarchy_parser import (
    HierarchyGraph,
    HierarchyNode,
    parse_hierarchy,
)
from kicraft.autoplacer.brain.subcircuit_artifacts import (
    build_artifact_metadata,
    build_leaf_extraction,
    build_solved_layout_artifact,
    extraction_summary,
    save_artifact_metadata,
    save_debug_payload,
    save_solved_layout_artifact,
)
from kicraft.autoplacer.brain.subcircuit_board_export import (
    ExportOptions,
    export_subcircuit_board,
)
from kicraft.autoplacer.brain.subcircuit_extractor import (
    extract_leaf_board_state,
    extraction_debug_dict,
    summarize_extraction,
)
from kicraft.autoplacer.brain.subcircuit_solver import solve_leaf_placement
from kicraft.autoplacer.config import DEFAULT_CONFIG, discover_project_config, load_project_config
from kicraft.autoplacer.hardware.adapter import KiCadAdapter


def _iter_non_root_nodes(graph: HierarchyGraph):
    for child in graph.root.children:
        yield child
        yield from _iter_children(child)


def _iter_children(node: HierarchyNode):
    for child in node.children:
        yield child
        yield from _iter_children(child)


def _load_config(config_path: str | None, project_dir: Path | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(DEFAULT_CONFIG)
    # Auto-discover project config from project directory
    if project_dir is not None:
        proj_cfg = discover_project_config(project_dir)
        if proj_cfg is not None:
            cfg.update(load_project_config(str(proj_cfg)))
    # Explicit --config overrides on top
    if config_path:
        cfg.update(load_project_config(config_path))
    return cfg


def _default_pcb_path(top_schematic: Path) -> Path:
    return top_schematic.with_suffix(".kicad_pcb")


def _extract_leaf_partition(
    node: HierarchyNode, board_state, config: dict[str, Any]
) -> dict[str, Any]:
    extracted = extract_leaf_board_state(
        subcircuit=node.definition,
        full_state=board_state,
        margin_mm=float(config.get("subcircuit_margin_mm", 5.0)),
        include_power_externals=bool(
            config.get("subcircuit_include_power_externals", True)
        ),
        ignored_nets=set(config.get("subcircuit_ignored_nets", [])),
    )

    return {
        "extracted": extracted,
        "internal_nets": extracted.internal_net_names,
        "external_nets": extracted.external_net_names,
        "ignored_nets": extracted.ignored_net_names,
        "internal_pad_count": sum(
            len(net.pad_refs) for net in extracted.net_partition.internal.values()
        ),
        "external_pad_count": sum(
            len(net.pad_refs) for net in extracted.net_partition.external.values()
        ),
        "component_count": len(extracted.component_refs),
        "trace_count": len(extracted.internal_traces),
        "via_count": len(extracted.internal_vias),
        "board_width_mm": extracted.local_state.board_width,
        "board_height_mm": extracted.local_state.board_height,
        "debug": extraction_debug_dict(extracted),
        "summary": summarize_extraction(extracted),
    }


def _load_board_state(pcb_path: Path, config: dict[str, Any]):
    adapter = KiCadAdapter(str(pcb_path), config=config)
    return adapter.load()


def _build_leaf_payloads(
    graph: HierarchyGraph,
    config: dict[str, Any],
    board_state,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    for node in _iter_non_root_nodes(graph):
        if not node.is_leaf:
            continue

        extracted = extract_leaf_board_state(
            subcircuit=node.definition,
            full_state=board_state,
            margin_mm=float(config.get("subcircuit_margin_mm", 5.0)),
            include_power_externals=bool(
                config.get("subcircuit_include_power_externals", True)
            ),
            ignored_nets=set(config.get("subcircuit_ignored_nets", [])),
        )

        partition = {
            "internal_nets": extracted.internal_net_names,
            "external_nets": extracted.external_net_names,
            "ignored_nets": extracted.ignored_net_names,
            "internal_pad_count": sum(
                len(net.pad_refs) for net in extracted.net_partition.internal.values()
            ),
            "external_pad_count": sum(
                len(net.pad_refs) for net in extracted.net_partition.external.values()
            ),
            "component_count": len(extracted.component_refs),
            "trace_count": len(extracted.internal_traces),
            "via_count": len(extracted.internal_vias),
            "board_width_mm": extracted.local_state.board_width,
            "board_height_mm": extracted.local_state.board_height,
        }

        solve_result = solve_leaf_placement(
            extraction=extracted,
            config={
                **config,
                "subcircuit_route_internal_nets": bool(
                    config.get("subcircuit_route_internal_nets", False)
                ),
            },
            seed=int(config.get("subcircuit_seed", 0)),
        )

        extraction = build_leaf_extraction(
            subcircuit=node.definition,
            project_dir=graph.project_dir,
            internal_nets=partition["internal_nets"],
            external_nets=partition["external_nets"],
            local_board_outline={
                "top_left_x": extracted.local_state.board_outline[0].x,
                "top_left_y": extracted.local_state.board_outline[0].y,
                "bottom_right_x": extracted.local_state.board_outline[1].x,
                "bottom_right_y": extracted.local_state.board_outline[1].y,
                "width_mm": extracted.local_state.board_width,
                "height_mm": extracted.local_state.board_height,
            },
            local_translation={
                "x": extracted.translation.x,
                "y": extracted.translation.y,
            },
            internal_trace_count=len(solve_result.layout.traces),
            internal_via_count=len(solve_result.layout.vias),
            notes=[
                "exported by export_subcircuit_artifacts.py",
                f"root_schematic={graph.root_schematic_path}",
                f"component_count={partition['component_count']}",
                f"internal_pad_count={partition['internal_pad_count']}",
                f"external_pad_count={partition['external_pad_count']}",
                f"trace_count={partition['trace_count']}",
                f"via_count={partition['via_count']}",
                f"board_width_mm={partition['board_width_mm']:.3f}",
                f"board_height_mm={partition['board_height_mm']:.3f}",
                f"solve_score={solve_result.score_total:.3f}",
                f"routed_internal_nets={len(solve_result.routed_internal_nets)}",
                f"failed_internal_nets={len(solve_result.failed_internal_nets)}",
            ]
            + list(extracted.notes)
            + list(solve_result.notes),
        )
        metadata = build_artifact_metadata(
            extraction=extraction,
            config=config,
            solver_version="subcircuits-m3-export",
        )
        solved_layout = build_solved_layout_artifact(
            solve_result.layout,
            project_dir=graph.project_dir,
            source_hash=metadata.source_hash,
            config_hash=metadata.config_hash,
            solver_version="subcircuits-m3-export",
            notes=list(solve_result.notes),
        )

        payloads.append(
            {
                "node": node,
                "extracted": extracted,
                "extraction": extraction,
                "metadata": metadata,
                "solved_layout": solved_layout,
                "solve_result": solve_result,
                "partition": partition,
                "summary": extraction_summary(extraction),
            }
        )

    return payloads


def _print_human_summary(payloads: list[dict[str, Any]], dry_run: bool) -> None:
    print("=== Subcircuit Artifact Export ===")
    print(f"leaf_subcircuits : {len(payloads)}")
    print(f"mode             : {'dry-run' if dry_run else 'write'}")
    print()

    for item in payloads:
        metadata = item["metadata"]
        partition = item["partition"]
        solve_result = item["solve_result"]
        print(f"- {item['summary']}")
        print(f"  internal_nets : {len(partition['internal_nets'])}")
        print(f"  external_nets : {len(partition['external_nets'])}")
        print(f"  ignored_nets  : {len(partition['ignored_nets'])}")
        print(f"  internal_pads : {partition['internal_pad_count']}")
        print(f"  external_pads : {partition['external_pad_count']}")
        print(f"  traces        : {partition['trace_count']}")
        print(f"  vias          : {partition['via_count']}")
        print(
            f"  local_size_mm : {partition['board_width_mm']:.1f} x {partition['board_height_mm']:.1f}"
        )
        print(f"  solve_score   : {solve_result.score_total:.2f}")
        print(f"  route_traces  : {solve_result.route_trace_count}")
        print(f"  route_vias    : {solve_result.route_via_count}")
        print(f"  metadata_json : {metadata.artifact_paths['metadata_json']}")
        print(f"  debug_json    : {metadata.artifact_paths['debug_json']}")
        print(f"  mini_pcb      : {metadata.artifact_paths['mini_pcb']}")
        print(f"  solved_layout : {metadata.artifact_paths['solved_layout_json']}")
        print()


def _json_payload(payloads: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    return {
        "leaf_subcircuits": len(payloads),
        "mode": "dry-run" if dry_run else "write",
        "artifacts": [
            {
                "summary": item["summary"],
                "partition": dict(item["partition"]),
                "metadata": item["metadata"].to_dict(),
                "solved_layout": item["solved_layout"],
                "solve_result": {
                    "score_total": item["solve_result"].score_total,
                    "score_breakdown": dict(item["solve_result"].score_breakdown),
                    "routed_internal_nets": list(
                        item["solve_result"].routed_internal_nets
                    ),
                    "failed_internal_nets": list(
                        item["solve_result"].failed_internal_nets
                    ),
                    "route_trace_count": item["solve_result"].route_trace_count,
                    "route_via_count": item["solve_result"].route_via_count,
                    "route_length_mm": item["solve_result"].route_length_mm,
                    "notes": list(item["solve_result"].notes),
                },
            }
            for item in payloads
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export leaf subcircuit artifact metadata from hierarchy"
    )
    parser.add_argument(
        "schematic",
        help="Top-level .kicad_sch file",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file to merge on top of default/project config",
    )
    parser.add_argument(
        "--pcb",
        help="Optional .kicad_pcb file used to derive leaf-local net partitions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/export planned artifact metadata without writing files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of human-readable text",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    top_schematic = Path(args.schematic).resolve()

    if not top_schematic.exists():
        print(f"error: schematic not found: {top_schematic}", file=sys.stderr)
        return 2
    if top_schematic.suffix != ".kicad_sch":
        print(
            f"error: expected a .kicad_sch file, got: {top_schematic}",
            file=sys.stderr,
        )
        return 2

    try:
        config = _load_config(args.config, project_dir=top_schematic.parent)
        pcb_path = (
            Path(args.pcb).resolve() if args.pcb else _default_pcb_path(top_schematic)
        )
        if not pcb_path.exists():
            raise FileNotFoundError(f"PCB not found: {pcb_path}")

        graph = parse_hierarchy(
            project_dir=top_schematic.parent,
            top_schematic=top_schematic,
        )
        board_state = _load_board_state(pcb_path, config)
        payloads = _build_leaf_payloads(graph, config, board_state)
    except Exception as exc:
        print(f"error: failed to export subcircuit artifacts: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        for item in payloads:
            metadata = item["metadata"]
            extraction = item["extraction"]
            solve_result = item["solve_result"]
            save_artifact_metadata(metadata)
            save_solved_layout_artifact(item["solved_layout"])
            export_subcircuit_board(
                solve_result.layout,
                metadata.artifact_paths["mini_pcb"],
                ExportOptions(
                    title="Solved Leaf Subcircuit",
                    comment="Generated by export_subcircuit_artifacts.py",
                ),
            )
            save_debug_payload(
                extraction=extraction,
                metadata=metadata,
                extra={
                    "export_command": "export_subcircuit_artifacts.py",
                    "root_schematic_path": graph.root_schematic_path,
                    "net_partition": dict(item["partition"]),
                    "leaf_board_state": {
                        "component_refs": list(item["extracted"].component_refs),
                        "internal_nets": list(item["extracted"].internal_net_names),
                        "external_nets": list(item["extracted"].external_net_names),
                        "ignored_nets": list(item["extracted"].ignored_net_names),
                        "board_width_mm": item["extracted"].local_state.board_width,
                        "board_height_mm": item["extracted"].local_state.board_height,
                        "translation": {
                            "x": item["extracted"].translation.x,
                            "y": item["extracted"].translation.y,
                        },
                    },
                    "solve_result": {
                        "score_total": solve_result.score_total,
                        "score_breakdown": dict(solve_result.score_breakdown),
                        "routed_internal_nets": list(solve_result.routed_internal_nets),
                        "failed_internal_nets": list(solve_result.failed_internal_nets),
                        "route_trace_count": solve_result.route_trace_count,
                        "route_via_count": solve_result.route_via_count,
                        "route_length_mm": solve_result.route_length_mm,
                        "notes": list(solve_result.notes),
                    },
                    "canonical_solved_layout": item["solved_layout"],
                },
            )

    if args.json:
        print(json.dumps(_json_payload(payloads, args.dry_run), indent=2))
        return 0

    _print_human_summary(payloads, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
