#!/usr/bin/env python3
"""Hierarchical autoexperiment orchestrator for the subcircuit pipeline.

This replaces the legacy whole-board mutate/place/route experiment loop with a
clean bottom-up hierarchical runner.

Design goals:
- no legacy whole-board routing path
- no fallback to the old flat pipeline
- drive only the subcircuit leaf-first flow
- emit machine-readable progress for the Experiment Manager GUI
- preserve visual artifacts for both leaf and top-level progression
- keep the CLI simple and stable

High-level flow per round:
1. solve routed leaf subcircuits with `solve_subcircuits.py`
2. inspect accepted leaf artifacts under `.experiments/subcircuits/`
3. compose and route the parent/top-level board with `compose_subcircuits.py`
4. score the round from accepted artifact quality and hierarchy coverage
5. keep the best round and publish live status / JSONL events / frame metadata

This file intentionally drives only the hierarchical subcircuit
leaf-first composition and routing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from kicraft.autoplacer.config import (
    CONFIG_SEARCH_SPACE,
    DEFAULT_CONFIG,
    discover_project_config,
    enforce_param_constraints,
    load_project_config,
    normalize_bounds,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[4]

def _detect_project_files(project_dir):
    """Auto-detect KiCad project files from directory."""
    pro_files = list(project_dir.glob("*.kicad_pro"))
    if pro_files:
        stem = pro_files[0].stem
        return project_dir / f"{stem}.kicad_sch", project_dir / f"{stem}.kicad_pcb"
    return None, None


DEFAULT_SCHEMATIC, DEFAULT_PCB = _detect_project_files(PROJECT_DIR)


@dataclass
class HierarchyRound:
    round_num: int
    seed: int
    mode: str
    score: float
    kept: bool
    duration_s: float
    leaf_total: int = 0
    leaf_accepted: int = 0
    parent_composed: bool = False
    parent_routed: bool = False
    accepted_trace_count: int = 0
    accepted_via_count: int = 0
    latest_stage: str = ""
    details: str = ""
    artifact_root: str = ""
    composition_json: str = ""
    parent_output_json: str = ""
    leaf_names: list[str] = field(default_factory=list)
    accepted_leaf_names: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    score_notes: list[str] = field(default_factory=list)
    absolute_score: float = 0.0
    improvement_score: float = 0.0
    plateau_escape_score: float = 0.0
    parent_quality_score: float = 0.0
    baseline_score: float = 0.0
    rolling_score: float = 0.0
    improvement_vs_best: float = 0.0
    improvement_vs_baseline: float = 0.0
    improvement_vs_recent: float = 0.0
    plateau_count: int = 0
    timing_breakdown: dict[str, float] = field(default_factory=dict)
    leaf_timing_summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=False)


def _check_stop_request(work_dir: Path) -> bool:
    return (work_dir / "stop.now").exists()


def _request_stop(work_dir: Path) -> None:
    (work_dir / "stop.now").touch()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _format_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _timing_now() -> float:
    return time.monotonic()


def _record_timing(
    timing_breakdown: dict[str, float],
    key: str,
    start_ts: float,
) -> float:
    elapsed_s = round(max(0.0, time.monotonic() - start_ts), 3)
    timing_breakdown[key] = elapsed_s
    return elapsed_s


def _format_timing_breakdown(timing_breakdown: dict[str, float]) -> str:
    if not timing_breakdown:
        return "timing=unavailable"
    ordered_items = sorted(timing_breakdown.items())
    return "timing=" + ", ".join(f"{key}={value:.3f}s" for key, value in ordered_items)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: int | None = None,
) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _discover_artifact_dirs(project_dir: Path) -> list[Path]:
    root = project_dir / ".experiments" / "subcircuits"
    if not root.exists():
        return []
    result: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "metadata.json").exists() and (
            child / "solved_layout.json"
        ).exists():
            result.append(child)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _accepted_leaf_artifacts(project_dir: Path) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for artifact_dir in _discover_artifact_dirs(project_dir):
        solved_layout_path = artifact_dir / "solved_layout.json"
        metadata_path = artifact_dir / "metadata.json"
        try:
            solved_layout = _load_json(solved_layout_path)
            metadata = _load_json(metadata_path)
        except Exception:
            continue

        validation = solved_layout.get("validation", {})
        if validation.get("accepted") is True:
            accepted.append(
                {
                    "artifact_dir": str(artifact_dir),
                    "sheet_name": solved_layout.get(
                        "sheet_name", metadata.get("sheet_name", "")
                    ),
                    "instance_path": solved_layout.get("instance_path", ""),
                    "trace_count": len(solved_layout.get("traces", [])),
                    "via_count": len(solved_layout.get("vias", [])),
                    "validation": validation,
                    "metadata": metadata,
                    "solved_layout": solved_layout,
                }
            )
    return accepted


def _all_leaf_artifacts(project_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for artifact_dir in _discover_artifact_dirs(project_dir):
        solved_layout_path = artifact_dir / "solved_layout.json"
        metadata_path = artifact_dir / "metadata.json"
        try:
            solved_layout = _load_json(solved_layout_path)
            metadata = _load_json(metadata_path)
        except Exception:
            continue
        artifacts.append(
            {
                "artifact_dir": str(artifact_dir),
                "sheet_name": solved_layout.get(
                    "sheet_name", metadata.get("sheet_name", "")
                ),
                "instance_path": solved_layout.get("instance_path", ""),
                "trace_count": len(solved_layout.get("traces", [])),
                "via_count": len(solved_layout.get("vias", [])),
                "validation": solved_layout.get("validation", {}),
                "metadata": metadata,
                "solved_layout": solved_layout,
            }
        )
    return artifacts


def _score_round(
    *,
    accepted_leafs: list[dict[str, Any]],
    all_leafs: list[dict[str, Any]],
    composition_ok: bool,
    parent_routed: bool,
    parent_copper_accounting: dict[str, int] | None,
    baseline_score: float | None,
    recent_scores: list[float],
    plateau_count: int,
    timing_breakdown: dict[str, float] | None = None,
    parent_board_area_mm2: float = 0.0,
    child_total_area_mm2: float = 0.0,
) -> tuple[float, dict[str, float], list[str], dict[str, float]]:
    leaf_total = len(all_leafs)
    leaf_accepted = len(accepted_leafs)
    acceptance_ratio = (leaf_accepted / leaf_total) if leaf_total > 0 else 0.0

    accepted_trace_count = sum(item.get("trace_count", 0) for item in accepted_leafs)
    accepted_via_count = sum(item.get("via_count", 0) for item in accepted_leafs)

    all_trace_count = sum(item.get("trace_count", 0) for item in all_leafs)
    all_via_count = sum(item.get("via_count", 0) for item in all_leafs)

    trace_coverage_ratio = (
        accepted_trace_count / all_trace_count if all_trace_count > 0 else 0.0
    )
    via_coverage_ratio = (
        accepted_via_count / all_via_count if all_via_count > 0 else 0.0
    )

    copper = dict(parent_copper_accounting or {})
    expected_child_traces = int(
        copper.get("expected_preserved_child_trace_count", 0) or 0
    )
    expected_child_vias = int(copper.get("expected_preserved_child_via_count", 0) or 0)
    preserved_child_traces = int(copper.get("preserved_child_trace_count", 0) or 0)
    preserved_child_vias = int(copper.get("preserved_child_via_count", 0) or 0)
    routed_total_traces = int(copper.get("routed_total_trace_count", 0) or 0)
    routed_total_vias = int(copper.get("routed_total_via_count", 0) or 0)
    added_parent_traces = int(copper.get("added_parent_trace_count", 0) or 0)
    added_parent_vias = int(copper.get("added_parent_via_count", 0) or 0)

    preserved_trace_ratio = (
        preserved_child_traces / expected_child_traces
        if expected_child_traces > 0
        else 0.0
    )
    preserved_via_ratio = (
        preserved_child_vias / expected_child_vias if expected_child_vias > 0 else 0.0
    )
    parent_added_copper_present = (
        1.0 if (added_parent_traces + added_parent_vias) > 0 else 0.0
    )
    parent_routed_copper_ratio = (routed_total_traces + routed_total_vias) / max(
        1,
        expected_child_traces
        + expected_child_vias
        + added_parent_traces
        + added_parent_vias,
    )

    leaf_acceptance_score = acceptance_ratio * 30.0
    routed_copper_score = min(
        16.0,
        trace_coverage_ratio * 12.0 + via_coverage_ratio * 4.0,
    )
    parent_composition_score = 8.0 if composition_ok else 0.0
    parent_routed_score = 12.0 if parent_routed else 0.0

    parent_quality_score = min(
        14.0,
        preserved_trace_ratio * 7.0
        + preserved_via_ratio * 3.0
        + parent_added_copper_present * 2.0
        + min(1.0, parent_routed_copper_ratio) * 2.0,
    )

    area_compactness_score = 0.0
    area_utilization_ratio = 0.0
    if parent_board_area_mm2 > 0.0 and child_total_area_mm2 > 0.0:
        area_utilization_ratio = min(
            1.0, child_total_area_mm2 / parent_board_area_mm2
        )
        area_compactness_score = min(9.0, area_utilization_ratio * 9.0)

    absolute_score = round(
        leaf_acceptance_score
        + routed_copper_score
        + parent_composition_score
        + parent_routed_score
        + parent_quality_score
        + area_compactness_score,
        3,
    )

    effective_baseline = absolute_score if baseline_score is None else baseline_score
    rolling_reference = (
        sum(recent_scores) / len(recent_scores) if recent_scores else effective_baseline
    )

    improvement_vs_baseline = absolute_score - effective_baseline
    improvement_vs_recent = absolute_score - rolling_reference

    baseline_improvement_score = max(0.0, min(10.0, improvement_vs_baseline * 0.6))
    recent_improvement_score = max(0.0, min(4.0, improvement_vs_recent * 1.0))

    plateau_escape_trigger = max(2, plateau_count)
    plateau_escape_score = 0.0
    if plateau_count >= plateau_escape_trigger and improvement_vs_recent > 0.5:
        plateau_escape_score = min(4.0, 1.0 + improvement_vs_recent * 0.75)

    improvement_score = round(
        baseline_improvement_score + recent_improvement_score,
        3,
    )
    plateau_escape_score = round(plateau_escape_score, 3)

    score_breakdown = {
        "absolute_leaf_acceptance": round(leaf_acceptance_score, 3),
        "absolute_routed_copper": round(routed_copper_score, 3),
        "absolute_parent_composition": round(parent_composition_score, 3),
        "absolute_parent_routed": round(parent_routed_score, 3),
        "absolute_parent_quality": round(parent_quality_score, 3),
        "absolute_area_compactness": round(area_compactness_score, 3),
        "area_utilization_ratio": round(area_utilization_ratio, 3),
        "improvement_vs_baseline": round(baseline_improvement_score, 3),
        "improvement_vs_recent": round(recent_improvement_score, 3),
        "plateau_escape": plateau_escape_score,
    }
    score = round(
        absolute_score + improvement_score + plateau_escape_score,
        3,
    )

    score_notes = [
        f"leaf_acceptance_ratio={acceptance_ratio:.3f}",
        f"trace_coverage_ratio={trace_coverage_ratio:.3f}",
        f"via_coverage_ratio={via_coverage_ratio:.3f}",
        f"accepted_leafs={leaf_accepted}/{leaf_total}",
        f"accepted_traces={accepted_trace_count}/{all_trace_count}",
        f"accepted_vias={accepted_via_count}/{all_via_count}",
        f"composition_ok={composition_ok}",
        f"parent_routed={parent_routed}",
        f"preserved_child_trace_ratio={preserved_trace_ratio:.3f}",
        f"preserved_child_via_ratio={preserved_via_ratio:.3f}",
        f"parent_added_copper_present={parent_added_copper_present:.3f}",
        f"parent_routed_copper_ratio={parent_routed_copper_ratio:.3f}",
        f"baseline_score={effective_baseline:.3f}",
        f"rolling_score={rolling_reference:.3f}",
        f"improvement_vs_baseline={improvement_vs_baseline:.3f}",
        f"improvement_vs_recent={improvement_vs_recent:.3f}",
        f"plateau_count_in={plateau_count}",
        f"area_utilization_ratio={area_utilization_ratio:.3f}",
        f"parent_board_area_mm2={parent_board_area_mm2:.1f}",
        f"child_total_area_mm2={child_total_area_mm2:.1f}",
        f"area_compactness_score={area_compactness_score:.3f}",
        "score_architecture=absolute_plus_improvement_plus_plateau_escape",
        "score_scale=bounded_components_with_relative_rewards",
    ]
    if timing_breakdown:
        score_notes.append(_format_timing_breakdown(timing_breakdown))
        for key, value in sorted(timing_breakdown.items()):
            score_notes.append(f"timing_{key}={value:.3f}s")
    score_context = {
        "absolute_score": absolute_score,
        "improvement_score": improvement_score,
        "plateau_escape_score": plateau_escape_score,
        "parent_quality_score": round(parent_quality_score, 3),
        "area_compactness_score": round(area_compactness_score, 3),
        "area_utilization_ratio": round(area_utilization_ratio, 3),
        "baseline_score": round(effective_baseline, 3),
        "rolling_score": round(rolling_reference, 3),
        "improvement_vs_baseline": round(improvement_vs_baseline, 3),
        "improvement_vs_recent": round(improvement_vs_recent, 3),
    }
    return score, score_breakdown, score_notes, score_context


def _extract_leaf_timing_summary(solve_payload: dict[str, Any]) -> dict[str, Any]:
    results = solve_payload.get("results", [])
    if not isinstance(results, list):
        results = []

    leaf_rows: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        solved = item.get("solved", {})
        if not isinstance(solved, dict):
            solved = {}
        best_round = solved.get("best_round", {})
        if not isinstance(best_round, dict):
            best_round = {}
        rounds_payload = solved.get("rounds", [])
        if not isinstance(rounds_payload, list):
            rounds_payload = []
        timing = {}
        for round_payload in rounds_payload:
            if not isinstance(round_payload, dict):
                continue
            round_timing = round_payload.get("timing_breakdown", {})
            if not isinstance(round_timing, dict):
                continue
            candidate_leaf_total = float(round_timing.get("leaf_total_s", 0.0) or 0.0)
            current_leaf_total = float(timing.get("leaf_total_s", 0.0) or 0.0)
            if candidate_leaf_total >= current_leaf_total:
                timing = dict(round_timing)
        if not timing:
            timing = best_round.get("timing_breakdown", {})
            if not isinstance(timing, dict):
                timing = {}

        scheduling_metadata = solved.get("scheduling_metadata", {})
        if not isinstance(scheduling_metadata, dict):
            scheduling_metadata = {}
        failure_summary = solved.get("failure_summary", {})
        if not isinstance(failure_summary, dict):
            failure_summary = {}

        sheet_name = str(solved.get("sheet_name", "") or "")
        instance_path = str(solved.get("instance_path", "") or "")
        leaf_total_s = float(timing.get("leaf_total_s", 0.0) or 0.0)
        route_total_s = float(timing.get("route_local_subcircuit_total_s", 0.0) or 0.0)
        freerouting_s = float(timing.get("freerouting_s", 0.0) or 0.0)
        render_total_s = float(
            (timing.get("pre_route_render_diagnostics_s", 0.0) or 0.0)
            + (timing.get("routed_render_diagnostics_s", 0.0) or 0.0)
        )
        placement_total_s = float(
            (timing.get("placement_solve_s", 0.0) or 0.0)
            + (timing.get("passive_ordering_s", 0.0) or 0.0)
            + (timing.get("post_ordering_legality_repair_s", 0.0) or 0.0)
            + (timing.get("placement_scoring_s", 0.0) or 0.0)
        )
        trace_count = int(
            scheduling_metadata.get("trace_count", item.get("trace_count", 0)) or 0
        )
        via_count = int(
            scheduling_metadata.get("via_count", item.get("via_count", 0)) or 0
        )
        internal_net_count = int(scheduling_metadata.get("internal_net_count", 0) or 0)
        failed_round_count = int(failure_summary.get("failed_round_count", 0) or 0)
        accepted_round_count = int(failure_summary.get("accepted_round_count", 0) or 0)
        had_failures = bool(failure_summary.get("had_failures", False))
        historically_trivial_candidate = bool(
            scheduling_metadata.get("historically_trivial_candidate", False)
        )
        unique_failure_reasons = list(failure_summary.get("unique_reasons", []) or [])
        long_pole_candidate = leaf_total_s > 0.0

        leaf_rows.append(
            {
                "sheet_name": sheet_name,
                "instance_path": instance_path,
                "leaf_total_s": round(leaf_total_s, 3),
                "route_total_s": round(route_total_s, 3),
                "freerouting_s": round(freerouting_s, 3),
                "render_total_s": round(render_total_s, 3),
                "placement_total_s": round(placement_total_s, 3),
                "trace_count": trace_count,
                "via_count": via_count,
                "internal_net_count": internal_net_count,
                "failed_round_count": failed_round_count,
                "accepted_round_count": accepted_round_count,
                "had_failures": had_failures,
                "historically_trivial_candidate": historically_trivial_candidate,
                "long_pole_candidate": long_pole_candidate,
                "unique_failure_reasons": unique_failure_reasons,
                "timing_breakdown": dict(timing),
                "scheduling_metadata": dict(scheduling_metadata),
                "failure_summary": dict(failure_summary),
            }
        )

    sorted_by_total = sorted(
        leaf_rows,
        key=lambda row: float(row.get("leaf_total_s", 0.0) or 0.0),
        reverse=True,
    )
    long_poles = sorted_by_total[:3]
    long_pole_paths = {
        str(row.get("instance_path", "") or "")
        for row in long_poles
        if str(row.get("instance_path", "") or "")
    }
    for row in leaf_rows:
        row["long_pole_candidate"] = (
            str(row.get("instance_path", "") or "") in long_pole_paths
        )

    total_leaf_time = round(
        sum(float(row.get("leaf_total_s", 0.0) or 0.0) for row in leaf_rows),
        3,
    )
    max_leaf_time = round(
        max([float(row.get("leaf_total_s", 0.0) or 0.0) for row in leaf_rows] or [0.0]),
        3,
    )
    avg_leaf_time = round(
        total_leaf_time / max(1, len(leaf_rows)),
        3,
    )
    imbalance_ratio = round(
        max_leaf_time / max(avg_leaf_time, 0.001),
        3,
    )

    return {
        "leaf_count": len(leaf_rows),
        "total_leaf_time_s": total_leaf_time,
        "avg_leaf_time_s": avg_leaf_time,
        "max_leaf_time_s": max_leaf_time,
        "imbalance_ratio": imbalance_ratio,
        "long_pole_leafs": long_poles,
        "leafs": leaf_rows,
    }


def _extract_solve_json_payload(stdout_text: str) -> dict[str, Any]:
    text = str(stdout_text or "").strip()
    if not text:
        return {}

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    marker_patterns = [
        r"===SOLVE_SUBCIRCUITS_JSON_START===\s*(\{[\s\S]*?\})\s*===SOLVE_SUBCIRCUITS_JSON_END===",
        r"---SOLVE_SUBCIRCUITS_JSON_START---\s*(\{[\s\S]*?\})\s*---SOLVE_SUBCIRCUITS_JSON_END---",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue

    decoder = json.JSONDecoder()
    last_payload: dict[str, Any] = {}
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            payload, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(end, int) and end > 0:
            if "leaf_subcircuits" in payload or "results" in payload:
                return payload
            last_payload = payload

    if last_payload:
        return last_payload

    match = re.search(r"(\{[\s\S]*\})\s*$", text)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(1))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _leaf_schedule_order(
    leaf_artifacts: list[dict[str, Any]],
    leaf_timing_summary: dict[str, Any] | None,
) -> list[str]:
    timing = dict(leaf_timing_summary or {})
    timed_leafs = timing.get("leafs", [])
    if not isinstance(timed_leafs, list):
        timed_leafs = []

    metrics_by_name: dict[str, dict[str, Any]] = {}
    metrics_by_path: dict[str, dict[str, Any]] = {}

    def _merge_metrics(target: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        merged = dict(target)
        merged["leaf_total_s"] = max(
            float(merged.get("leaf_total_s", 0.0) or 0.0),
            float(row.get("leaf_total_s", 0.0) or 0.0),
        )
        merged["route_total_s"] = max(
            float(merged.get("route_total_s", 0.0) or 0.0),
            float(row.get("route_total_s", 0.0) or 0.0),
        )
        merged["freerouting_s"] = max(
            float(merged.get("freerouting_s", 0.0) or 0.0),
            float(row.get("freerouting_s", 0.0) or 0.0),
        )
        merged["trace_count"] = max(
            int(merged.get("trace_count", 0) or 0),
            int(row.get("trace_count", 0) or 0),
        )
        merged["via_count"] = max(
            int(merged.get("via_count", 0) or 0),
            int(row.get("via_count", 0) or 0),
        )
        merged["internal_net_count"] = max(
            int(merged.get("internal_net_count", 0) or 0),
            int(row.get("internal_net_count", 0) or 0),
        )
        merged["failed_round_count"] = max(
            int(merged.get("failed_round_count", 0) or 0),
            int(row.get("failed_round_count", 0) or 0),
        )
        merged["accepted_round_count"] = max(
            int(merged.get("accepted_round_count", 0) or 0),
            int(row.get("accepted_round_count", 0) or 0),
        )
        merged["had_failures"] = bool(
            merged.get("had_failures", False) or row.get("had_failures", False)
        )
        merged["historically_trivial_candidate"] = bool(
            merged.get("historically_trivial_candidate", False)
            or row.get("historically_trivial_candidate", False)
        )
        merged["long_pole_candidate"] = bool(
            merged.get("long_pole_candidate", False)
            or row.get("long_pole_candidate", False)
        )
        return merged

    for row in timed_leafs:
        if not isinstance(row, dict):
            continue
        sheet_name = str(row.get("sheet_name", "") or "")
        instance_path = str(row.get("instance_path", "") or "")
        if sheet_name:
            metrics_by_name[sheet_name] = _merge_metrics(
                metrics_by_name.get(sheet_name, {}),
                row,
            )
        if instance_path:
            metrics_by_path[instance_path] = _merge_metrics(
                metrics_by_path.get(instance_path, {}),
                row,
            )

    scheduling_rows: list[dict[str, Any]] = []
    for item in leaf_artifacts:
        if not isinstance(item, dict):
            continue
        sheet_name = str(item.get("sheet_name", "") or "")
        instance_path = str(item.get("instance_path", "") or "")
        metrics = {}
        if sheet_name:
            metrics = _merge_metrics(metrics, metrics_by_name.get(sheet_name, {}))
        if instance_path:
            metrics = _merge_metrics(metrics, metrics_by_path.get(instance_path, {}))

        trace_count = max(
            int(item.get("trace_count", 0) or 0),
            int(metrics.get("trace_count", 0) or 0),
        )
        via_count = max(
            int(item.get("via_count", 0) or 0),
            int(metrics.get("via_count", 0) or 0),
        )
        internal_net_count = int(metrics.get("internal_net_count", 0) or 0)
        leaf_total_s = float(metrics.get("leaf_total_s", 0.0) or 0.0)
        route_total_s = float(metrics.get("route_total_s", 0.0) or 0.0)
        freerouting_s = float(metrics.get("freerouting_s", 0.0) or 0.0)
        failed_round_count = int(metrics.get("failed_round_count", 0) or 0)
        accepted_round_count = int(metrics.get("accepted_round_count", 0) or 0)
        had_failures = bool(metrics.get("had_failures", False))
        historically_trivial_candidate = bool(
            metrics.get("historically_trivial_candidate", False)
        )
        long_pole_candidate = bool(metrics.get("long_pole_candidate", False))

        failure_pressure = 0.0
        if had_failures:
            failure_pressure += 40.0
        failure_pressure += min(25.0, failed_round_count * 8.0)
        if accepted_round_count <= 0 and had_failures:
            failure_pressure += 15.0

        routing_pressure = (
            leaf_total_s * 1.0 + route_total_s * 1.35 + freerouting_s * 1.75
        )
        topology_pressure = (
            trace_count * 0.35 + via_count * 0.8 + internal_net_count * 1.5
        )
        long_pole_bonus = 18.0 if long_pole_candidate else 0.0
        trivial_penalty = (
            30.0 if historically_trivial_candidate or internal_net_count <= 0 else 0.0
        )

        scheduling_score = round(
            routing_pressure
            + topology_pressure
            + failure_pressure
            + long_pole_bonus
            - trivial_penalty,
            3,
        )

        scheduling_rows.append(
            {
                "sheet_name": sheet_name,
                "instance_path": instance_path,
                "trace_count": trace_count,
                "via_count": via_count,
                "internal_net_count": internal_net_count,
                "leaf_total_s": round(leaf_total_s, 3),
                "route_total_s": round(route_total_s, 3),
                "freerouting_s": round(freerouting_s, 3),
                "failed_round_count": failed_round_count,
                "accepted_round_count": accepted_round_count,
                "had_failures": had_failures,
                "historically_trivial_candidate": historically_trivial_candidate,
                "long_pole_candidate": long_pole_candidate,
                "scheduling_score": scheduling_score,
            }
        )

    ordered = sorted(
        scheduling_rows,
        key=lambda item: (
            -float(item.get("scheduling_score", 0.0) or 0.0),
            -float(item.get("freerouting_s", 0.0) or 0.0),
            -float(item.get("route_total_s", 0.0) or 0.0),
            -float(item.get("leaf_total_s", 0.0) or 0.0),
            -int(item.get("trace_count", 0) or 0),
            -int(item.get("via_count", 0) or 0),
            str(item.get("sheet_name", "") or ""),
        ),
    )
    return [str(item.get("sheet_name", "") or "") for item in ordered]


def _extract_parent_copper_accounting(project_dir: Path) -> dict[str, int]:
    def _normalize_copper_accounting(payload: dict[str, Any]) -> dict[str, int]:
        return {
            "expected_preserved_child_trace_count": int(
                payload.get("expected_preserved_child_trace_count", 0) or 0
            ),
            "expected_preserved_child_via_count": int(
                payload.get("expected_preserved_child_via_count", 0) or 0
            ),
            "preserved_child_trace_count": int(
                payload.get("preserved_child_trace_count", 0) or 0
            ),
            "preserved_child_via_count": int(
                payload.get("preserved_child_via_count", 0) or 0
            ),
            "routed_total_trace_count": int(
                payload.get("routed_total_trace_count", 0) or 0
            ),
            "routed_total_via_count": int(
                payload.get("routed_total_via_count", 0) or 0
            ),
            "added_parent_trace_count": int(
                payload.get("added_parent_trace_count", 0) or 0
            ),
            "added_parent_via_count": int(
                payload.get("added_parent_via_count", 0) or 0
            ),
        }

    candidate_paths = [
        project_dir
        / ".experiments"
        / "subcircuits"
        / "subcircuit__8a5edab282"
        / "debug.json",
        project_dir
        / ".experiments"
        / "subcircuits"
        / "subcircuit__8a5edab282"
        / "metadata.json",
        project_dir / ".experiments" / "hierarchical_pipeline" / "parent_pipeline.json",
    ]

    for candidate in candidate_paths:
        try:
            payload = _load_json(candidate)
        except Exception:
            continue

        if candidate.name == "debug.json":
            routing_result = payload.get("routing_result", {})
            if isinstance(routing_result, dict):
                copper = routing_result.get("copper_accounting", {})
                if isinstance(copper, dict) and any(
                    int(copper.get(k, 0) or 0) for k in copper
                ):
                    return _normalize_copper_accounting(copper)

        elif candidate.name == "metadata.json":
            normalized = _normalize_copper_accounting(payload)
            if any(normalized.values()):
                return normalized

        elif candidate.name == "parent_pipeline.json":
            state = payload.get("state", {})
            if isinstance(state, dict):
                normalized = _normalize_copper_accounting(state)
                if any(normalized.values()):
                    return normalized

    return {}


def _extract_parent_board_dimensions(
    parent_output_json: Path,
) -> tuple[float, float]:
    """Return (width_mm, height_mm) of the parent board from pipeline JSON.

    Falls back to (0.0, 0.0) if the file is missing or malformed.
    """
    try:
        payload = _load_json(parent_output_json)
    except Exception:
        return (0.0, 0.0)
    state = payload.get("state", {})
    if not isinstance(state, dict):
        return (0.0, 0.0)
    bbox = state.get("bounding_box", {})
    if not isinstance(bbox, dict):
        return (0.0, 0.0)
    try:
        w = float(bbox.get("width_mm", 0.0) or 0.0)
        h = float(bbox.get("height_mm", 0.0) or 0.0)
    except (ValueError, TypeError):
        return (0.0, 0.0)
    return (max(0.0, w), max(0.0, h))


def _discover_latest_parent_artifact_dir(project_dir: Path) -> Path | None:
    root = project_dir / ".experiments" / "subcircuits"
    if not root.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        debug_path = child / "debug.json"
        metadata_path = child / "metadata.json"
        if not debug_path.exists() and not metadata_path.exists():
            continue

        payload = {}
        try:
            if debug_path.exists():
                payload = _load_json(debug_path)
            elif metadata_path.exists():
                payload = _load_json(metadata_path)
        except Exception:
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        if not (
            payload.get("parent_composition") is True
            or payload.get("schema_version") == "parent-compose-v1"
        ):
            continue

        try:
            mtime = max(
                debug_path.stat().st_mtime if debug_path.exists() else 0.0,
                metadata_path.stat().st_mtime if metadata_path.exists() else 0.0,
            )
        except OSError:
            mtime = 0.0
        candidates.append((mtime, child))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _discover_live_preview_paths(project_dir: Path) -> dict[str, str]:
    previews: dict[str, str] = {}

    latest_parent_dir = _discover_latest_parent_artifact_dir(project_dir)
    if latest_parent_dir is not None:
        renders_dir = latest_parent_dir / "renders"
        stamped_candidates = [
            renders_dir / "parent_stamped.png",
            renders_dir / "front_all.png",
            renders_dir / "board.png",
        ]
        routed_candidates = [
            renders_dir / "parent_routed.png",
            renders_dir / "routed.png",
            renders_dir / "board_routed.png",
        ]
        stamped_board_candidates = [
            latest_parent_dir / "parent_pre_freerouting.kicad_pcb",
            latest_parent_dir / "parent_stamped.kicad_pcb",
        ]
        routed_board_candidates = [
            latest_parent_dir / "parent_routed.kicad_pcb",
        ]
        for candidate in stamped_candidates:
            if candidate.exists():
                previews["parent_stamped_preview"] = str(candidate)
                break
        for candidate in routed_candidates:
            if candidate.exists():
                previews["parent_routed_preview"] = str(candidate)
                break
        for candidate in stamped_board_candidates:
            if candidate.exists():
                previews["parent_stamped_board"] = str(candidate)
                break
        for candidate in routed_board_candidates:
            if candidate.exists():
                previews["parent_routed_board"] = str(candidate)
                break
        previews["parent_artifact_dir"] = str(latest_parent_dir)

    sub_root = project_dir / ".experiments" / "subcircuits"
    latest_leaf: tuple[float, Path] | None = None
    latest_leaf_payload: dict[str, Any] = {}
    if sub_root.exists():
        for child in sub_root.iterdir():
            if not child.is_dir():
                continue
            solved_path = child / "solved_layout.json"
            metadata_path = child / "metadata.json"
            debug_path = child / "debug.json"
            if not solved_path.exists():
                continue
            try:
                payload = _load_json(metadata_path) if metadata_path.exists() else {}
            except Exception:
                payload = {}
            if isinstance(payload, dict) and payload.get("parent_composition") is True:
                continue
            try:
                debug_payload = _load_json(debug_path) if debug_path.exists() else {}
            except Exception:
                debug_payload = {}
            try:
                mtime = solved_path.stat().st_mtime
            except OSError:
                continue
            if latest_leaf is None or mtime > latest_leaf[0]:
                latest_leaf = (mtime, child)
                latest_leaf_payload = (
                    debug_payload if isinstance(debug_payload, dict) else {}
                )

    if latest_leaf is not None:
        leaf_dir = latest_leaf[1]
        renders_dir = leaf_dir / "renders"
        leaf_candidates = [
            renders_dir / "routed_front_all.png",
            renders_dir / "pre_route_front_all.png",
            renders_dir / "routed_copper_both.png",
        ]
        for candidate in leaf_candidates:
            if candidate.exists():
                previews["leaf_preview"] = str(candidate)
                break

        latest_round = {}
        extra = latest_leaf_payload.get("extra", {})
        if isinstance(extra, dict):
            all_rounds = extra.get("all_rounds", [])
            if isinstance(all_rounds, list) and all_rounds:
                latest_round = (
                    all_rounds[-1] if isinstance(all_rounds[-1], dict) else {}
                )

        preview_paths = latest_round.get("preview_paths", {})
        if isinstance(preview_paths, dict):
            for source_key, target_key in [
                ("pre_route_front", "leaf_round_pre_route_preview"),
                ("routed_front", "leaf_round_routed_preview"),
                ("routed_copper", "leaf_round_routed_copper_preview"),
            ]:
                value = preview_paths.get(source_key)
                if value:
                    previews[target_key] = str(value)

        board_paths = latest_round.get("board_paths", {})
        if isinstance(board_paths, dict):
            for source_key, target_key in [
                ("illegal_pre_stamp", "leaf_round_illegal_board"),
                ("pre_route", "leaf_round_pre_route_board"),
                ("routed", "leaf_round_routed_board"),
            ]:
                value = board_paths.get(source_key)
                if value:
                    previews[target_key] = str(value)

        previews["leaf_artifact_dir"] = str(leaf_dir)

    return previews


def _write_live_status(
    status_json_path: Path,
    status_txt_path: Path,
    *,
    phase: str,
    rounds_total: int,
    round_num: int,
    best_score: float,
    kept_count: int,
    latest_score: float | None,
    latest_marker: str | None,
    start_ts: float,
    current_stage: str,
    leaf_total: int,
    leaf_accepted: int,
    parent_routed: bool,
    leaf_worker_count: int = 1,
    leaf_workers_active: int = 0,
    leaf_workers_queued: int = 0,
    leaf_workers_completed: int | None = None,
    current_node: str | None = None,
    current_leaf: str | None = None,
    current_parent: str | None = None,
    top_level_status: str | None = None,
    composition_status: str | None = None,
    copper_accounting: dict[str, int] | None = None,
    current_action: str | None = None,
    current_command: str | None = None,
    preview_paths: dict[str, str] | None = None,
    leaf_timing_summary: dict[str, Any] | None = None,
) -> None:
    now = time.monotonic()
    elapsed_s = now - start_ts
    normalized_leaf_worker_count = max(1, leaf_worker_count)
    normalized_leaf_workers_active = max(
        0, min(normalized_leaf_worker_count, int(leaf_workers_active or 0))
    )
    normalized_leaf_workers_queued = max(0, int(leaf_workers_queued or 0))
    normalized_leaf_workers_completed = max(
        0,
        int(
            leaf_accepted if leaf_workers_completed is None else leaf_workers_completed
        ),
    )
    progress_pct = (round_num / rounds_total * 100.0) if rounds_total > 0 else 100.0
    copper = dict(copper_accounting or {})
    hierarchy_top_level_status = (
        top_level_status
        if top_level_status is not None
        else ("ready" if parent_routed else "not_ready")
    )
    hierarchy_composition_status = (
        composition_status if composition_status is not None else current_stage
    )

    payload = {
        "phase": phase,
        "pipeline": "hierarchical_subcircuits",
        "round": round_num,
        "total_rounds": rounds_total,
        "progress_percent": round(progress_pct, 2),
        "best_score": round(best_score, 3),
        "latest_score": None if latest_score is None else round(latest_score, 3),
        "latest_marker": latest_marker,
        "kept_count": kept_count,
        "elapsed_s": round(elapsed_s, 1),
        "eta_s": 0.0,
        "current_node": current_node,
        "current_leaf": current_leaf,
        "current_parent": current_parent,
        "current_action": current_action,
        "current_command": current_command,
        "preview_paths": dict(preview_paths or {}),
        "top_level_status": hierarchy_top_level_status,
        "composition_status": hierarchy_composition_status,
        "workers": {
            "total": 1,
            "in_flight": 0 if phase != "running" else 1,
            "idle": 0 if phase == "running" else 1,
            "leaf_workers": normalized_leaf_worker_count,
        },
        "maybe_stuck": False,
        "hierarchy": {
            "current_stage": current_stage,
            "leaf_total": leaf_total,
            "leaf_accepted": leaf_accepted,
            "parent_routed": parent_routed,
            "leaf_parallelism_enabled": normalized_leaf_worker_count > 1,
            "leaf_worker_count": normalized_leaf_worker_count,
            "leaf_workers": {
                "total": normalized_leaf_worker_count,
                "active": normalized_leaf_workers_active,
                "idle": max(
                    0, normalized_leaf_worker_count - normalized_leaf_workers_active
                ),
                "queued": normalized_leaf_workers_queued,
                "completed": normalized_leaf_workers_completed,
            },
            "current_node": current_node,
            "current_leaf": current_leaf,
            "current_parent": current_parent,
            "current_action": current_action,
            "current_command": current_command,
            "preview_paths": dict(preview_paths or {}),
            "top_level_status": hierarchy_top_level_status,
            "composition_status": hierarchy_composition_status,
            "copper_accounting": copper,
            "leaf_timing_summary": dict(leaf_timing_summary or {}),
        },
        "timestamp_epoch_s": time.time(),
    }
    _write_json(status_json_path, payload)

    lines = [
        "=== Hierarchical Autoexperiment Live Status ===",
        f"phase: {phase}",
        "pipeline: hierarchical_subcircuits",
        f"progress: {round_num}/{rounds_total} ({progress_pct:.1f}%)",
        f"best_score: {best_score:.2f}",
        f"latest_score: {'n/a' if latest_score is None else f'{latest_score:.2f}'}",
        f"latest_event: {latest_marker or 'n/a'}",
        f"kept_count: {kept_count}",
        f"elapsed: {_format_mmss(elapsed_s)}",
        f"current_stage: {current_stage}",
        f"current_action: {current_action or 'n/a'}",
        f"current_node: {current_node or 'n/a'}",
        f"current_leaf: {current_leaf or 'n/a'}",
        f"current_parent: {current_parent or 'n/a'}",
        f"current_command: {current_command or 'n/a'}",
        f"leafs: accepted={leaf_accepted} total={leaf_total}",
        (
            "leaf_workers: "
            f"active={normalized_leaf_workers_active}/"
            f"{normalized_leaf_worker_count} "
            f"queued={normalized_leaf_workers_queued} "
            f"completed={normalized_leaf_workers_completed}"
        ),
        f"composition_status: {hierarchy_composition_status}",
        f"top_level_status: {hierarchy_top_level_status}",
        f"parent_routed: {parent_routed}",
    ]
    if copper:
        lines.extend(
            [
                "copper_accounting:",
                (
                    "  preserved_child_traces: "
                    f"{copper.get('preserved_child_trace_count', 0)}/"
                    f"{copper.get('expected_preserved_child_trace_count', 0)}"
                ),
                (
                    "  preserved_child_vias: "
                    f"{copper.get('preserved_child_via_count', 0)}/"
                    f"{copper.get('expected_preserved_child_via_count', 0)}"
                ),
                (f"  routed_total_traces: {copper.get('routed_total_trace_count', 0)}"),
                (f"  routed_total_vias: {copper.get('routed_total_via_count', 0)}"),
                (f"  added_parent_traces: {copper.get('added_parent_trace_count', 0)}"),
                (f"  added_parent_vias: {copper.get('added_parent_via_count', 0)}"),
            ]
        )
    if preview_paths:
        lines.append("preview_paths:")
        for key, value in sorted(preview_paths.items()):
            lines.append(f"  {key}: {value}")

        board_path_keys = [
            "leaf_round_illegal_board",
            "leaf_round_pre_route_board",
            "leaf_round_routed_board",
            "parent_stamped_board",
            "parent_routed_board",
        ]
        board_path_items = [
            (key, preview_paths.get(key, ""))
            for key in board_path_keys
            if preview_paths.get(key)
        ]
        if board_path_items:
            lines.append("kicad_board_paths:")
            for key, value in board_path_items:
                lines.append(f"  {key}: {value}")
    status_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_round_detail(
    rounds_dir: Path,
    round_result: HierarchyRound,
    *,
    accepted_leafs: list[dict[str, Any]],
    all_leafs: list[dict[str, Any]],
    solve_exit_code: int,
    compose_exit_code: int,
    parent_route_exit_code: int | None,
    solve_stdout: str,
    solve_stderr: str,
    compose_stdout: str,
    compose_stderr: str,
    parent_route_stdout: str,
    parent_route_stderr: str,
    parent_copper_accounting: dict[str, int] | None = None,
    preview_paths: dict[str, str] | None = None,
) -> None:
    payload = {
        "round": round_result.round_num,
        "round_num": round_result.round_num,
        "seed": round_result.seed,
        "mode": round_result.mode,
        "score": round_result.score,
        "kept": round_result.kept,
        "duration_s": round_result.duration_s,
        "latest_stage": round_result.latest_stage,
        "timing_breakdown": dict(round_result.timing_breakdown),
        "leaf_timing_summary": dict(round_result.leaf_timing_summary),
        "hierarchy": {
            "leaf_total": round_result.leaf_total,
            "leaf_accepted": round_result.leaf_accepted,
            "parent_composed": round_result.parent_composed,
            "parent_routed": round_result.parent_routed,
            "accepted_trace_count": round_result.accepted_trace_count,
            "accepted_via_count": round_result.accepted_via_count,
            "leaf_names": round_result.leaf_names,
            "accepted_leaf_names": round_result.accepted_leaf_names,
            "score_breakdown": dict(round_result.score_breakdown),
            "score_notes": list(round_result.score_notes),
            "absolute_score": round_result.absolute_score,
            "improvement_score": round_result.improvement_score,
            "plateau_escape_score": round_result.plateau_escape_score,
            "parent_quality_score": round_result.parent_quality_score,
            "baseline_score": round_result.baseline_score,
            "rolling_score": round_result.rolling_score,
            "improvement_vs_best": round_result.improvement_vs_best,
            "improvement_vs_baseline": round_result.improvement_vs_baseline,
            "improvement_vs_recent": round_result.improvement_vs_recent,
            "plateau_count": round_result.plateau_count,
            "timing_breakdown": dict(round_result.timing_breakdown),
            "leaf_timing_summary": dict(round_result.leaf_timing_summary),
        },
        "artifacts": {
            "artifact_root": round_result.artifact_root,
            "composition_json": round_result.composition_json,
            "parent_output_json": round_result.parent_output_json,
            "parent_copper_accounting": dict(parent_copper_accounting or {}),
            "parent_board_paths": {
                "parent_stamped_board": str(
                    (preview_paths or {}).get("parent_stamped_board", "") or ""
                ),
                "parent_routed_board": str(
                    (preview_paths or {}).get("parent_routed_board", "") or ""
                ),
            },
            "preview_paths": dict(preview_paths or {}),
        },
        "commands": {
            "solve_exit_code": solve_exit_code,
            "compose_exit_code": compose_exit_code,
            "parent_route_exit_code": parent_route_exit_code,
        },
        "accepted_leaf_artifacts": accepted_leafs,
        "all_leaf_artifacts": all_leafs,
        "logs": {
            "solve_stdout_tail": solve_stdout[-12000:],
            "solve_stderr_tail": solve_stderr[-12000:],
            "compose_stdout_tail": compose_stdout[-12000:],
            "compose_stderr_tail": compose_stderr[-12000:],
            "parent_route_stdout_tail": parent_route_stdout[-12000:],
            "parent_route_stderr_tail": parent_route_stderr[-12000:],
        },
    }
    _write_json(rounds_dir / f"round_{round_result.round_num:04d}.json", payload)


def _write_frame_metadata(
    frames_dir: Path,
    round_result: HierarchyRound,
) -> None:
    payload = {
        "round_num": round_result.round_num,
        "kept": round_result.kept,
        "accepted": round_result.kept,
        "score": round_result.score,
        "mode": round_result.mode,
        "stage": round_result.latest_stage,
        "latest_stage": round_result.latest_stage,
        "leaf_total": round_result.leaf_total,
        "leaf_accepted": round_result.leaf_accepted,
        "parent_routed": round_result.parent_routed,
        "parent_composed": round_result.parent_composed,
        "accepted_trace_count": round_result.accepted_trace_count,
        "accepted_via_count": round_result.accepted_via_count,
        "score_breakdown": dict(round_result.score_breakdown),
        "score_notes": list(round_result.score_notes),
        "absolute_score": round_result.absolute_score,
        "improvement_score": round_result.improvement_score,
        "plateau_escape_score": round_result.plateau_escape_score,
        "parent_quality_score": round_result.parent_quality_score,
        "baseline_score": round_result.baseline_score,
        "rolling_score": round_result.rolling_score,
        "improvement_vs_best": round_result.improvement_vs_best,
        "improvement_vs_baseline": round_result.improvement_vs_baseline,
        "improvement_vs_recent": round_result.improvement_vs_recent,
        "plateau_count": round_result.plateau_count,
        "timing_breakdown": dict(round_result.timing_breakdown),
        "leaf_timing_summary": dict(round_result.leaf_timing_summary),
        "sheet_name": ", ".join(round_result.accepted_leaf_names[:3])
        if round_result.accepted_leaf_names
        else "",
        "instance_path": round_result.parent_output_json
        or round_result.composition_json,
        "artifact_root": round_result.artifact_root,
        "composition_json": round_result.composition_json,
        "parent_output_json": round_result.parent_output_json,
        "details": round_result.details,
    }
    _write_json(frames_dir / f"frame_{round_result.round_num:04d}.json", payload)


def _select_preview_image(parent_output_dir: Path) -> Path | None:
    candidates = [
        parent_output_dir / "parent_routed.png",
        parent_output_dir / "routed.png",
        parent_output_dir / "parent_stamped.png",
        parent_output_dir / "board_routed.png",
        parent_output_dir / "board.png",
        parent_output_dir / "snapshot.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pngs = sorted(
        parent_output_dir.glob("*.png"),
        key=lambda path: (
            0
            if "routed" in path.name.lower()
            else 1
            if "stamped" in path.name.lower()
            else 2,
            path.name.lower(),
        ),
    )
    return pngs[0] if pngs else None


def _mutate_config(
    base_config: dict[str, Any],
    search_space: Mapping[str, Mapping[str, Any]],
    rng: random.Random,
    mutation_rate: float = 0.3,
    enable_board_size: bool = False,
) -> dict[str, int | float]:
    """Mutate numeric config parameters using Gaussian perturbation.

    Each parameter in search_space has a mutation_rate chance of being
    perturbed. Perturbation is Gaussian with the configured sigma, clamped
    to [min, max]. Integer params are rounded after perturbation.

    Returns a new dict with only the mutated keys (sparse overlay suitable
    for writing as a JSON config file that merges with DEFAULT_CONFIG).
    """
    mutated = {}
    for key, spec in search_space.items():
        spec_min = float(spec["min"])
        spec_max = float(spec["max"])
        spec_sigma = float(spec["sigma"])
        spec_type = str(spec["type"])
        if not enable_board_size and key in {"board_width_mm", "board_height_mm"}:
            continue
        if rng.random() > mutation_rate:
            continue
        current = base_config.get(
            key,
            spec.get("default", (spec_min + spec_max) / 2),
        )
        if isinstance(current, set):
            continue
        current = float(current)
        delta = rng.gauss(0.0, spec_sigma)
        new_val = current + delta
        new_val = max(spec_min, min(spec_max, new_val))
        if spec_type == "int":
            new_val = int(round(new_val))
        else:
            new_val = round(new_val, 4)
        mutated[key] = new_val
    return mutated


def _jsonable_config_value(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _jsonable_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_config_value(item) for item in value]
    return value


def _config_overlay_from_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    for key, value in config.items():
        if key not in DEFAULT_CONFIG or DEFAULT_CONFIG[key] != value:
            overlay[key] = value
    return overlay


def _build_solve_cmd(
    *,
    schematic: Path,
    pcb: Path,
    rounds: int,
    seed: int,
    config: str | None,
    only: list[str],
    workers: int,
    fast_smoke: bool = False,
    leaf_order: list[str] | None = None,
    experiment_round: int = 0,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "solve_subcircuits.py"),
        str(schematic),
        "--pcb",
        str(pcb),
        "--rounds",
        str(max(1, rounds)),
        "--seed",
        str(seed),
        "--route",
        "--json",
        "--workers",
        str(max(1, workers)),
        "--experiment-round",
        str(int(experiment_round)),
    ]
    if fast_smoke:
        cmd.append("--fast-smoke")
    if config:
        cmd.extend(["--config", config])
    for selector in only:
        cmd.extend(["--only", selector])
    for selector in leaf_order or []:
        cmd.extend(["--leaf-order", selector])
    return cmd


def _build_compose_cmd(
    *,
    project_dir: Path,
    parent: str,
    output_json: Path,
    only: list[str],
    config: str | None = None,
    spacing_mm: float = 2.0,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compose_subcircuits.py"),
        "--project",
        str(project_dir),
        "--parent",
        parent,
        "--mode",
        "packed",
        "--spacing-mm",
        str(spacing_mm),
        "--output",
        str(output_json),
        "--json",
    ]
    if config:
        cmd.extend(["--config", config])
    for selector in only:
        cmd.extend(["--only", selector])
    return cmd


def _build_visible_cmd(
    *,
    project_dir: Path,
    schematic: Path,
    pcb: Path,
    parent: str,
    output_dir: Path,
    config: str | None,
    rounds: int,
    seed: int,
    only: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "solve_hierarchy.py"),
        str(schematic),
        "--pcb",
        str(pcb),
        "--rounds",
        str(max(1, rounds)),
        "--route",
    ]
    for selector in only:
        cmd.extend(["--only", selector])
    return cmd


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical subcircuit autoexperiment orchestrator"
    )
    parser.add_argument(
        "pcb",
        nargs="?",
        default=str(DEFAULT_PCB) if DEFAULT_PCB else None,
        help="Top-level PCB path (auto-detected from *.kicad_pro)",
    )
    parser.add_argument(
        "--schematic",
        default=str(DEFAULT_SCHEMATIC) if DEFAULT_SCHEMATIC else None,
        help="Top-level schematic path (auto-detected from *.kicad_pro)",
    )
    parser.add_argument(
        "--rounds",
        "-n",
        type=int,
        default=10,
        help="Number of hierarchical experiment rounds",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=0,
        help="Leaf solve worker count (0 = auto-select based on leaf count and CPU availability)",
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Pass through fast smoke mode to leaf solving for faster routed verification rounds",
    )
    parser.add_argument(
        "--plateau",
        type=int,
        default=2,
        help="Number of non-improving rounds tolerated before plateau markers become explicit",
    )
    parser.add_argument(
        "--jar",
        help="Optional FreeRouting jar path passed through to parent routing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master RNG seed",
    )
    parser.add_argument(
        "--config",
        help="Optional project config path passed through to child commands",
    )
    parser.add_argument(
        "--param-ranges",
        help="JSON file with parameter min/max bounds to constrain mutation search space",
    )
    parser.add_argument(
        "--parent",
        default="/",
        help="Parent selector for hierarchical parent composition and routing",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to specific leaf selectors",
    )
    parser.add_argument(
        "--leaf-rounds",
        type=int,
        default=1,
        help="Leaf solve rounds per experiment round",
    )

    parser.add_argument(
        "--status-file",
        default="run_status.json",
        help="Status JSON filename inside .experiments/",
    )
    parser.add_argument(
        "--log",
        default="experiments.jsonl",
        help="JSONL log filename inside .experiments/",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output best top-level artifact directory marker file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    project_dir = Path(args.pcb).resolve().parent
    pcb = Path(args.pcb).resolve()
    schematic = Path(args.schematic).resolve()

    if not pcb.exists():
        print(f"error: pcb not found: {pcb}", file=sys.stderr)
        return 2
    if not schematic.exists():
        print(f"error: schematic not found: {schematic}", file=sys.stderr)
        return 2

    work_dir = project_dir / ".experiments"
    work_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir = work_dir / "rounds"
    frames_dir = work_dir / "frames"
    best_dir = work_dir / "best"
    hierarchy_dir = work_dir / "hierarchical_autoexperiment"

    for path in (rounds_dir, frames_dir, best_dir, hierarchy_dir):
        path.mkdir(parents=True, exist_ok=True)

    log_path = work_dir / args.log
    status_json_path = work_dir / args.status_file
    status_txt_path = work_dir / "run_status.txt"

    _safe_unlink(log_path)
    _safe_unlink(status_json_path)
    _safe_unlink(status_txt_path)
    _safe_unlink(work_dir / "stop.now")

    def _sigterm_handler(signum: int, frame: Any) -> None:
        _request_stop(work_dir)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    master_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(master_seed)

    requested_workers = int(args.workers or 0)
    available_cpus = max(1, int(os.cpu_count() or 1))
    if requested_workers > 0:
        effective_workers = max(1, requested_workers)
    else:
        effective_workers = min(available_cpus, 6)
    effective_workers = max(1, effective_workers)

    print("=== Hierarchical Autoexperiment Complete ===")
    print(f"Project:      {project_dir}")
    print(f"Schematic:    {schematic}")
    print(f"PCB:          {pcb}")
    print(f"Rounds:       {args.rounds}")
    print(f"Leaf rounds:  {args.leaf_rounds}")
    print(f"Parent:       {args.parent}")
    print(f"Master seed:  {master_seed}")
    print(f"Workers:      {effective_workers} (requested={requested_workers})")
    print(f"Fast smoke:   {args.fast_smoke}")
    print("Mode:         subcircuit leaf bottom-up only")
    print()

    start_ts = time.monotonic()
    best_score = -1.0
    kept_count = 0
    plateau_count = 0
    baseline_score: float | None = None
    recent_scores: list[float] = []
    best_round: HierarchyRound | None = None
    resolved_config_path = args.config
    if not resolved_config_path:
        discovered_config_path = discover_project_config(project_dir)
        if discovered_config_path is not None:
            resolved_config_path = str(discovered_config_path)
    _base_project_config = load_project_config(resolved_config_path) if resolved_config_path else {}
    _initial_config = {**DEFAULT_CONFIG, **_base_project_config}
    _best_config = dict(_initial_config)

    effective_search_space = dict(CONFIG_SEARCH_SPACE)
    param_ranges_path = args.param_ranges
    if param_ranges_path:
        try:
            with open(param_ranges_path, "r", encoding="utf-8") as f:
                user_ranges = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"  [param-ranges] ERROR: failed to load {param_ranges_path}: {exc}",
                file=sys.stderr,
            )
            user_ranges = {}
        if not isinstance(user_ranges, dict):
            print(
                f"  [param-ranges] ERROR: expected JSON object at root, got {type(user_ranges).__name__}",
                file=sys.stderr,
            )
            user_ranges = {}
        applied_count = 0
        skipped_keys: list[str] = []
        for key, bounds in user_ranges.items():
            if not isinstance(bounds, list) or len(bounds) < 2:
                continue
            if key not in effective_search_space:
                skipped_keys.append(key)
                continue
            try:
                lo, hi = float(bounds[0]), float(bounds[1])
            except (TypeError, ValueError):
                continue
            result = normalize_bounds(key, lo, hi, effective_search_space[key])
            if result is None:
                print(
                    f"  [param-ranges] WARNING: {key} range is empty after "
                    f"normalization, skipping",
                    file=sys.stderr,
                )
                continue
            norm_lo, norm_hi = result
            effective_search_space[key] = {
                **effective_search_space[key],
                "min": norm_lo,
                "max": norm_hi,
            }
            applied_count += 1
        if applied_count:
            print(f"  [param-ranges] {applied_count} search-space bounds applied from {param_ranges_path}")
        if skipped_keys:
            print(
                f"  [param-ranges] NOTE: {len(skipped_keys)} key(s) not in search space: "
                f"{', '.join(skipped_keys[:5])}",
                file=sys.stderr,
            )
        if not applied_count and not skipped_keys:
            print(
                f"  [param-ranges] WARNING: 0 bounds applied from {param_ranges_path} "
                f"(keys may not match CONFIG_SEARCH_SPACE)",
                file=sys.stderr,
            )

    _write_live_status(
        status_json_path,
        status_txt_path,
        phase="running",
        rounds_total=args.rounds,
        round_num=0,
        best_score=0.0,
        kept_count=0,
        latest_score=None,
        latest_marker="starting hierarchical run",
        start_ts=start_ts,
        current_stage="startup",
        leaf_total=0,
        leaf_accepted=0,
        parent_routed=False,
        leaf_worker_count=effective_workers,
        leaf_workers_active=0,
        leaf_workers_queued=0,
        leaf_workers_completed=0,
        current_node=args.parent,
        current_parent=args.parent,
        top_level_status="pending",
        composition_status="startup",
        copper_accounting={},
    )

    for round_num in range(1, args.rounds + 1):
        if _check_stop_request(work_dir):
            break

        round_seed = rng.randint(0, 2**31 - 1)
        round_mutated: dict[str, int | float] = {}
        round_dir = hierarchy_dir / f"round_{round_num:04d}"
        parent_output_json = round_dir / "parent_pipeline.json"
        composition_json = round_dir / "parent_composition.json"
        round_dir.mkdir(parents=True, exist_ok=True)

        round_candidate_config = dict(_best_config)
        current_round_config = resolved_config_path
        round_mutated = _mutate_config(
            _best_config,
            effective_search_space,
            rng,
            enable_board_size=bool(_best_config.get("enable_board_size_search", False)),
        )
        round_candidate_config.update(round_mutated)
        enforce_param_constraints(round_candidate_config)
        if round_candidate_config != _initial_config:
            round_config_file = round_dir / "round_config.json"
            round_config_overlay = _config_overlay_from_defaults(round_candidate_config)
            serializable = _jsonable_config_value(round_config_overlay)
            with open(round_config_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
                f.write("\n")
            current_round_config = str(round_config_file)
        if round_mutated:
            print(
                f"  [mutate] {len(round_mutated)} params: "
                f"{', '.join(f'{k}={v}' for k, v in sorted(round_mutated.items())[:5])}"
            )

        _write_live_status(
            status_json_path,
            status_txt_path,
            phase="running",
            rounds_total=args.rounds,
            round_num=round_num,
            best_score=max(best_score, 0.0),
            kept_count=kept_count,
            latest_score=None,
            latest_marker=f"round {round_num} started",
            start_ts=start_ts,
            current_stage="solve_leafs",
            leaf_total=0,
            leaf_accepted=0,
            parent_routed=False,
            leaf_worker_count=effective_workers,
            leaf_workers_active=min(effective_workers, 1),
            leaf_workers_queued=0,
            leaf_workers_completed=0,
            current_node=args.parent,
            current_parent=args.parent,
            top_level_status="pending",
            composition_status="solving_leafs",
            copper_accounting={},
            current_action="starting leaf solve round",
            current_command=" ".join(
                _build_solve_cmd(
                    schematic=schematic,
                    pcb=pcb,
                    rounds=args.leaf_rounds,
                    seed=round_seed,
                    config=current_round_config,
                    only=args.only,
                    workers=effective_workers,
                    fast_smoke=args.fast_smoke,
                    leaf_order=[],
                    experiment_round=round_num,
                )
            ),
            preview_paths=_discover_live_preview_paths(project_dir),
            leaf_timing_summary={},
        )

        t0 = time.monotonic()

        previous_round_path = rounds_dir / f"round_{round_num - 1:04d}.json"
        previous_leaf_order: list[str] = []
        if previous_round_path.exists():
            try:
                previous_payload = _load_json(previous_round_path)
            except Exception:
                previous_payload = {}
            if isinstance(previous_payload, dict):
                previous_leaf_timing = previous_payload.get("leaf_timing_summary", {})
                if not isinstance(previous_leaf_timing, dict):
                    previous_leaf_timing = {}
                previous_all_leafs = previous_payload.get("all_leaf_artifacts", [])
                if not isinstance(previous_all_leafs, list):
                    previous_all_leafs = []
                previous_leaf_order = _leaf_schedule_order(
                    previous_all_leafs,
                    previous_leaf_timing,
                )

        solve_cmd = _build_solve_cmd(
            schematic=schematic,
            pcb=pcb,
            rounds=args.leaf_rounds,
            seed=round_seed,
            config=current_round_config,
            only=args.only,
            workers=effective_workers,
            fast_smoke=args.fast_smoke,
            leaf_order=previous_leaf_order,
            experiment_round=round_num,
        )
        leaf_timing_summary: dict[str, Any] = {}
        _write_live_status(
            status_json_path,
            status_txt_path,
            phase="running",
            rounds_total=args.rounds,
            round_num=round_num,
            best_score=max(best_score, 0.0),
            kept_count=kept_count,
            latest_score=None,
            latest_marker=f"round {round_num} leaf solve launched",
            start_ts=start_ts,
            current_stage="solve_leafs",
            leaf_total=0,
            leaf_accepted=0,
            parent_routed=False,
            leaf_worker_count=effective_workers,
            leaf_workers_active=min(effective_workers, 1),
            leaf_workers_queued=0,
            leaf_workers_completed=0,
            current_node=args.parent,
            current_parent=args.parent,
            top_level_status="pending",
            composition_status="solving_leafs",
            copper_accounting={},
            current_action="running solve_subcircuits",
            current_command=" ".join(solve_cmd),
            preview_paths=_discover_live_preview_paths(project_dir),
            leaf_timing_summary=leaf_timing_summary,
        )
        round_timing_breakdown: dict[str, float] = {}
        solve_start_ts = _timing_now()
        solve_rc, solve_stdout, solve_stderr = _run_command(
            solve_cmd,
            cwd=project_dir,
        )
        solve_elapsed_s = _record_timing(
            round_timing_breakdown,
            "solve_subcircuits_total",
            solve_start_ts,
        )
        print(
            f"[timing] round {round_num} solve_subcircuits_total={solve_elapsed_s:.3f}s"
        )

        solve_payload = _extract_solve_json_payload(solve_stdout)
        leaf_timing_summary = _extract_leaf_timing_summary(solve_payload)

        all_leafs = _all_leaf_artifacts(project_dir)
        accepted_leafs = _accepted_leaf_artifacts(project_dir)

        leaf_names = [item.get("sheet_name", "") for item in all_leafs]
        accepted_leaf_names = [item.get("sheet_name", "") for item in accepted_leafs]
        scheduled_leaf_names = _leaf_schedule_order(all_leafs, leaf_timing_summary)

        scheduled_leaf_rows: list[dict[str, Any]] = []
        scheduled_leaf_lookup: dict[str, dict[str, Any]] = {}
        timed_leaf_rows = leaf_timing_summary.get("leafs", [])
        if not isinstance(timed_leaf_rows, list):
            timed_leaf_rows = []
        for row in timed_leaf_rows:
            if not isinstance(row, dict):
                continue
            instance_path = str(row.get("instance_path", "") or "")
            sheet_name = str(row.get("sheet_name", "") or "")
            if instance_path:
                scheduled_leaf_lookup[instance_path] = dict(row)
            if sheet_name and sheet_name not in scheduled_leaf_lookup:
                scheduled_leaf_lookup[sheet_name] = dict(row)

        for position, selector in enumerate(scheduled_leaf_names, start=1):
            row = dict(scheduled_leaf_lookup.get(selector, {}))
            if not row:
                for candidate in timed_leaf_rows:
                    if not isinstance(candidate, dict):
                        continue
                    if str(candidate.get("sheet_name", "") or "") == selector:
                        row = dict(candidate)
                        break
            row["scheduled_position"] = position
            row["scheduled_selector"] = selector
            scheduled_leaf_rows.append(row)

        leaf_timing_summary["scheduled_leafs"] = scheduled_leaf_rows
        leaf_timing_summary["schedule_recommendation"] = list(scheduled_leaf_names)

        _write_live_status(
            status_json_path,
            status_txt_path,
            phase="running",
            rounds_total=args.rounds,
            round_num=round_num,
            best_score=max(best_score, 0.0),
            kept_count=kept_count,
            latest_score=None,
            latest_marker=f"round {round_num} leaf solve complete",
            start_ts=start_ts,
            current_stage="compose_parent",
            leaf_total=len(all_leafs),
            leaf_accepted=len(accepted_leafs),
            parent_routed=False,
            leaf_worker_count=effective_workers,
            leaf_workers_active=0,
            leaf_workers_queued=0,
            leaf_workers_completed=len(accepted_leafs),
            current_node=args.parent,
            current_parent=args.parent,
            current_leaf=", ".join(accepted_leaf_names[:3])
            if accepted_leaf_names
            else None,
            top_level_status="pending",
            composition_status="composing_parent",
            copper_accounting={},
            current_action="leaf solve complete; composing parent snapshot",
            current_command=" ".join(
                _build_compose_cmd(
                    project_dir=project_dir,
                    parent=args.parent,
                    output_json=composition_json,
                    only=args.only,
                    config=current_round_config,
                    spacing_mm=round_candidate_config.get("parent_spacing_mm", 2.0),
                )
            ),
            preview_paths=_discover_live_preview_paths(project_dir),
            leaf_timing_summary=leaf_timing_summary,
        )

        compose_cmd = _build_compose_cmd(
            project_dir=project_dir,
            parent=args.parent,
            output_json=composition_json,
            only=args.only,
            config=current_round_config,
            spacing_mm=round_candidate_config.get("parent_spacing_mm", 2.0),
        )
        _write_live_status(
            status_json_path,
            status_txt_path,
            phase="running",
            rounds_total=args.rounds,
            round_num=round_num,
            best_score=max(best_score, 0.0),
            kept_count=kept_count,
            latest_score=None,
            latest_marker=f"round {round_num} parent composition launched",
            start_ts=start_ts,
            current_stage="compose_parent",
            leaf_total=len(all_leafs),
            leaf_accepted=len(accepted_leafs),
            parent_routed=False,
            leaf_worker_count=effective_workers,
            leaf_workers_active=0,
            leaf_workers_queued=0,
            leaf_workers_completed=len(accepted_leafs),
            current_node=args.parent,
            current_parent=args.parent,
            current_leaf=", ".join(accepted_leaf_names[:3])
            if accepted_leaf_names
            else None,
            top_level_status="pending",
            composition_status="composing_parent",
            copper_accounting={},
            current_action="running compose_subcircuits snapshot stage",
            current_command=" ".join(compose_cmd),
            preview_paths=_discover_live_preview_paths(project_dir),
            leaf_timing_summary=leaf_timing_summary,
        )
        compose_start_ts = _timing_now()
        compose_rc, compose_stdout, compose_stderr = _run_command(
            compose_cmd,
            cwd=project_dir,
        )
        compose_elapsed_s = _record_timing(
            round_timing_breakdown,
            "compose_subcircuits_total",
            compose_start_ts,
        )
        print(
            f"[timing] round {round_num} compose_subcircuits_total={compose_elapsed_s:.3f}s"
        )

        parent_route_rc: int | None = None
        parent_route_stdout = ""
        parent_route_stderr = ""
        parent_routed = False
        parent_copper_accounting: dict[str, int] = {}

        if True:
            _write_live_status(
                status_json_path,
                status_txt_path,
                phase="running",
                rounds_total=args.rounds,
                round_num=round_num,
                best_score=max(best_score, 0.0),
                kept_count=kept_count,
                latest_score=None,
                latest_marker=f"round {round_num} parent composed",
                start_ts=start_ts,
                current_stage="route_parent",
                leaf_total=len(all_leafs),
                leaf_accepted=len(accepted_leafs),
                parent_routed=False,
                leaf_worker_count=effective_workers,
                leaf_workers_active=0,
                leaf_workers_queued=0,
                leaf_workers_completed=len(accepted_leafs),
                current_node=args.parent,
                current_parent=args.parent,
                current_leaf=", ".join(accepted_leaf_names[:3])
                if accepted_leaf_names
                else None,
                top_level_status="routing_top_level",
                composition_status="routing_parent",
                copper_accounting={},
                current_action="parent snapshot complete; preparing parent routing run",
                preview_paths=_discover_live_preview_paths(project_dir),
                leaf_timing_summary=leaf_timing_summary,
            )

            parent_route_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "compose_subcircuits.py"),
                "--project",
                str(project_dir),
                "--parent",
                args.parent,
                "--mode",
                "packed",
                "--spacing-mm",
                str(round_candidate_config.get("parent_spacing_mm", 2.0)),
                "--pcb",
                str(pcb),
                "--route",
                "--output",
                str(parent_output_json),
            ]
            if args.jar:
                parent_route_cmd.extend(["--jar", args.jar])
            if current_round_config:
                parent_route_cmd.extend(["--config", current_round_config])
            for selector in args.only:
                parent_route_cmd.extend(["--only", selector])

            _write_live_status(
                status_json_path,
                status_txt_path,
                phase="running",
                rounds_total=args.rounds,
                round_num=round_num,
                best_score=max(best_score, 0.0),
                kept_count=kept_count,
                latest_score=None,
                latest_marker=f"round {round_num} parent routing launched",
                start_ts=start_ts,
                current_stage="route_parent",
                leaf_total=len(all_leafs),
                leaf_accepted=len(accepted_leafs),
                parent_routed=False,
                leaf_worker_count=effective_workers,
                leaf_workers_active=0,
                leaf_workers_queued=0,
                leaf_workers_completed=len(accepted_leafs),
                current_node=args.parent,
                current_parent=args.parent,
                current_leaf=", ".join(accepted_leaf_names[:3])
                if accepted_leaf_names
                else None,
                top_level_status="routing_top_level",
                composition_status="routing_parent",
                copper_accounting={},
                current_action="running unified parent stamping/routing pipeline",
                current_command=" ".join(parent_route_cmd),
                preview_paths=_discover_live_preview_paths(project_dir),
                leaf_timing_summary=leaf_timing_summary,
            )
            parent_route_start_ts = _timing_now()
            parent_route_rc, parent_route_stdout, parent_route_stderr = _run_command(
                parent_route_cmd,
                cwd=project_dir,
            )
            parent_route_elapsed_s = _record_timing(
                round_timing_breakdown,
                "parent_route_total",
                parent_route_start_ts,
            )
            print(
                f"[timing] round {round_num} parent_route_total={parent_route_elapsed_s:.3f}s"
            )
            parent_routed = parent_route_rc == 0
            parent_copper_accounting = _extract_parent_copper_accounting(project_dir)
            _write_live_status(
                status_json_path,
                status_txt_path,
                phase="running",
                rounds_total=args.rounds,
                round_num=round_num,
                best_score=max(best_score, 0.0),
                kept_count=kept_count,
                latest_score=None,
                latest_marker=f"round {round_num} parent routing complete",
                start_ts=start_ts,
                current_stage="score_round",
                leaf_total=len(all_leafs),
                leaf_accepted=len(accepted_leafs),
                parent_routed=parent_routed,
                leaf_worker_count=effective_workers,
                leaf_workers_active=0,
                leaf_workers_queued=0,
                leaf_workers_completed=len(accepted_leafs),
                current_node=args.parent,
                current_parent=args.parent,
                current_leaf=", ".join(accepted_leaf_names[:3])
                if accepted_leaf_names
                else None,
                top_level_status="ready" if parent_routed else "not_ready",
                composition_status="parent_routed"
                if parent_routed
                else "parent_failed",
                copper_accounting=parent_copper_accounting,
                current_action="parent routing complete; scoring round",
                preview_paths=_discover_live_preview_paths(project_dir),
                leaf_timing_summary=leaf_timing_summary,
            )

        composition_ok = compose_rc == 0
        score_round_start_ts = _timing_now()

        # Extract parent board dimensions for area compactness scoring
        parent_w, parent_h = _extract_parent_board_dimensions(parent_output_json)
        parent_board_area_mm2 = parent_w * parent_h

        # Sum child leaf areas from solved_layout bounding boxes
        child_total_area_mm2 = 0.0
        for leaf in accepted_leafs:
            sl = leaf.get("solved_layout", {})
            bb = sl.get("bounding_box", {})
            if isinstance(bb, dict):
                lw = float(bb.get("width_mm", 0.0) or 0.0)
                lh = float(bb.get("height_mm", 0.0) or 0.0)
                child_total_area_mm2 += lw * lh

        score, score_breakdown, score_notes, score_context = _score_round(
            accepted_leafs=accepted_leafs,
            all_leafs=all_leafs,
            composition_ok=composition_ok,
            parent_routed=parent_routed,
            parent_copper_accounting=parent_copper_accounting,
            baseline_score=baseline_score,
            recent_scores=recent_scores,
            plateau_count=plateau_count,
            timing_breakdown=dict(round_timing_breakdown),
            parent_board_area_mm2=parent_board_area_mm2,
            child_total_area_mm2=child_total_area_mm2,
        )
        score_round_elapsed_s = _record_timing(
            round_timing_breakdown,
            "score_round_total",
            score_round_start_ts,
        )
        print(
            f"[timing] round {round_num} score_round_total={score_round_elapsed_s:.3f}s"
        )
        duration_s = round(time.monotonic() - t0, 2)
        round_timing_breakdown["round_total"] = round(duration_s, 3)

        if baseline_score is None:
            baseline_score = score_context["absolute_score"]

        improvement_vs_best = (
            score if best_score < 0.0 else round(score - best_score, 3)
        )
        keep_threshold = 0.5
        is_meaningful_improvement = (
            best_score < 0.0 or improvement_vs_best >= keep_threshold
        )

        if is_meaningful_improvement:
            plateau_count = 0
        else:
            plateau_count += 1

        long_pole_leafs = leaf_timing_summary.get("long_pole_leafs", [])
        if not isinstance(long_pole_leafs, list):
            long_pole_leafs = []
        long_pole_notes = [
            (
                f"{str(item.get('sheet_name', '') or '')}:"
                f"{float(item.get('leaf_total_s', 0.0) or 0.0):.3f}s"
            )
            for item in long_pole_leafs[:3]
            if isinstance(item, dict)
        ]
        scheduled_leafs = leaf_timing_summary.get("scheduled_leafs", [])
        if not isinstance(scheduled_leafs, list):
            scheduled_leafs = []
        scheduled_leaf_notes = [
            (
                f"{int(item.get('scheduled_position', 0) or 0)}:"
                f"{str(item.get('sheet_name', item.get('scheduled_selector', '')) or '')}"
                f"@{float(item.get('scheduling_score', 0.0) or 0.0):.3f}"
            )
            for item in scheduled_leafs[:5]
            if isinstance(item, dict)
        ]

        round_result = HierarchyRound(
            round_num=round_num,
            seed=round_seed,
            mode="hierarchical",
            score=score,
            kept=is_meaningful_improvement,
            duration_s=duration_s,
            leaf_total=len(all_leafs),
            leaf_accepted=len(accepted_leafs),
            parent_composed=composition_ok,
            parent_routed=parent_routed,
            accepted_trace_count=sum(
                item.get("trace_count", 0) for item in accepted_leafs
            ),
            accepted_via_count=sum(item.get("via_count", 0) for item in accepted_leafs),
            latest_stage="done",
            details=(
                f"leafs {len(accepted_leafs)}/{len(all_leafs)} accepted; "
                f"compose={'ok' if composition_ok else 'fail'}; "
                f"parent_route={'ok' if parent_routed else 'fail'}; "
                f"absolute={score_context['absolute_score']:.2f}; "
                f"parent_quality={score_context['parent_quality_score']:.2f}; "
                f"improvement={score_context['improvement_score']:.2f}; "
                f"escape={score_context['plateau_escape_score']:.2f}; "
                f"improvement_vs_best={improvement_vs_best:+.2f}; "
                f"plateau={plateau_count}; "
                f"leaf_imbalance={float(leaf_timing_summary.get('imbalance_ratio', 0.0) or 0.0):.2f}; "
                f"{_format_timing_breakdown(round_timing_breakdown)}"
            ),
            artifact_root=str(project_dir / ".experiments" / "subcircuits"),
            composition_json=str(composition_json),
            parent_output_json=str(parent_output_json),
            leaf_names=leaf_names,
            accepted_leaf_names=accepted_leaf_names,
            score_breakdown=score_breakdown,
            score_notes=score_notes
            + [
                f"absolute_score={score_context['absolute_score']:.3f}",
                f"parent_quality_score={score_context['parent_quality_score']:.3f}",
                f"improvement_score={score_context['improvement_score']:.3f}",
                f"plateau_escape_score={score_context['plateau_escape_score']:.3f}",
                "leaf_size_reduction_loop_plan=after_best_leaf_layout_found_shrink_local_outline_in_small_steps_until_validation_fails_then_keep_last_passing_size",
                "parent_size_reduction_loop_plan=after_best_parent_layout_found_shrink_parent_outline_in_small_steps_preserving_child_copper_then_reroute_and_keep_last_passing_size",
                "size_reduction_acceptance=must_preserve_required_anchors_and_avoid_new_illegal_geometry_or_drc_regression",
                f"keep_threshold={keep_threshold:.2f}",
                f"plateau_threshold={max(1, args.plateau)}",
                f"meaningful_improvement={is_meaningful_improvement}",
                f"leaf_timing_total_s={float(leaf_timing_summary.get('total_leaf_time_s', 0.0) or 0.0):.3f}",
                f"leaf_timing_avg_s={float(leaf_timing_summary.get('avg_leaf_time_s', 0.0) or 0.0):.3f}",
                f"leaf_timing_max_s={float(leaf_timing_summary.get('max_leaf_time_s', 0.0) or 0.0):.3f}",
                f"leaf_timing_imbalance_ratio={float(leaf_timing_summary.get('imbalance_ratio', 0.0) or 0.0):.3f}",
                f"leaf_schedule_recommendation={','.join(scheduled_leaf_names)}",
                f"leaf_schedule_top5={','.join(scheduled_leaf_notes) if scheduled_leaf_notes else 'none'}",
                f"leaf_long_poles={','.join(long_pole_notes) if long_pole_notes else 'none'}",
            ],
            absolute_score=score_context["absolute_score"],
            improvement_score=score_context["improvement_score"],
            plateau_escape_score=score_context["plateau_escape_score"],
            parent_quality_score=score_context["parent_quality_score"],
            baseline_score=score_context["baseline_score"],
            rolling_score=score_context["rolling_score"],
            improvement_vs_best=improvement_vs_best,
            improvement_vs_baseline=score_context["improvement_vs_baseline"],
            improvement_vs_recent=score_context["improvement_vs_recent"],
            plateau_count=plateau_count,
            leaf_timing_summary=leaf_timing_summary,
        )

        if round_result.kept:
            best_score = score
            kept_count += 1
            best_round = round_result
            _best_config = dict(round_candidate_config)

            best_summary = {
                "round_num": round_num,
                "seed": round_seed,
                "score": score,
                "absolute_score": round_result.absolute_score,
                "improvement_score": round_result.improvement_score,
                "plateau_escape_score": round_result.plateau_escape_score,
                "parent_quality_score": round_result.parent_quality_score,
                "baseline_score": round_result.baseline_score,
                "rolling_score": round_result.rolling_score,
                "details": round_result.details,
                "composition_json": str(composition_json),
                "parent_output_json": str(parent_output_json),
            }
            _write_json(best_dir / "best_hierarchical_round.json", best_summary)

            if composition_json.exists():
                _copy_if_exists(
                    composition_json, best_dir / "best_parent_composition.json"
                )

            live_previews = _discover_live_preview_paths(project_dir)
            preview = None
            parent_routed_preview = live_previews.get("parent_routed_preview")
            parent_stamped_preview = live_previews.get("parent_stamped_preview")
            leaf_preview = live_previews.get("leaf_preview")
            if parent_routed_preview:
                preview = Path(parent_routed_preview)
            elif parent_stamped_preview:
                preview = Path(parent_stamped_preview)
            elif leaf_preview:
                preview = Path(leaf_preview)
            if preview is not None:
                _copy_if_exists(preview, work_dir / "best_preview.png")
                _copy_if_exists(preview, frames_dir / f"frame_{round_num:04d}.png")
                _copy_if_exists(preview, frames_dir / "frame_latest.png")

        recent_scores.append(round_result.absolute_score)
        if len(recent_scores) > 5:
            recent_scores = recent_scores[-5:]

        _append_jsonl(log_path, asdict(round_result))
        _write_round_detail(
            rounds_dir,
            round_result,
            accepted_leafs=accepted_leafs,
            all_leafs=all_leafs,
            solve_exit_code=solve_rc,
            compose_exit_code=compose_rc,
            parent_route_exit_code=parent_route_rc,
            solve_stdout=solve_stdout,
            solve_stderr=solve_stderr,
            compose_stdout=compose_stdout,
            compose_stderr=compose_stderr,
            parent_route_stdout=parent_route_stdout,
            parent_route_stderr=parent_route_stderr,
            parent_copper_accounting=parent_copper_accounting,
            preview_paths=_discover_live_preview_paths(project_dir),
        )
        _write_frame_metadata(frames_dir, round_result)

        _write_live_status(
            status_json_path,
            status_txt_path,
            phase="running",
            rounds_total=args.rounds,
            round_num=round_num,
            best_score=max(best_score, score),
            kept_count=kept_count,
            latest_score=score,
            latest_marker=f"round {round_num} {'kept' if round_result.kept else 'discarded'}",
            start_ts=start_ts,
            current_stage="done",
            leaf_total=len(all_leafs),
            leaf_accepted=len(accepted_leafs),
            parent_routed=parent_routed,
            leaf_worker_count=effective_workers,
            leaf_workers_active=0,
            leaf_workers_queued=0,
            leaf_workers_completed=len(accepted_leafs),
            current_node=args.parent,
            current_parent=args.parent,
            current_leaf=", ".join(accepted_leaf_names[:3])
            if accepted_leaf_names
            else None,
            top_level_status="ready" if parent_routed else "not_ready",
            composition_status="complete" if parent_routed else "parent_failed",
            copper_accounting=parent_copper_accounting,
            current_action="round complete",
            preview_paths=_discover_live_preview_paths(project_dir),
            leaf_timing_summary=leaf_timing_summary,
        )

        print(
            f"Round {round_num:3d}/{args.rounds} "
            f"score={score:6.2f} "
            f"leafs={len(accepted_leafs)}/{len(all_leafs)} "
            f"compose={'ok' if composition_ok else 'fail'} "
            f"parent_route={'ok' if parent_routed else 'fail'} "
            f"[{'KEPT' if round_result.kept else 'discard'}]"
        )

    phase = "done"
    if _check_stop_request(work_dir):
        phase = "done"
        _safe_unlink(work_dir / "stop.now")

    final_payload = {
        "pipeline": "hierarchical_subcircuits",
        "status": phase,
        "master_seed": master_seed,
        "rounds_requested": args.rounds,
        "best_score": max(best_score, 0.0),
        "baseline_score": baseline_score,
        "recent_scores": recent_scores,
        "kept_count": kept_count,
        "plateau_count": plateau_count,
        "best_round": asdict(best_round) if best_round else None,
    }
    _write_json(work_dir / "hierarchical_summary.json", final_payload)

    final_parent_copper_accounting = _extract_parent_copper_accounting(project_dir)

    _write_live_status(
        status_json_path,
        status_txt_path,
        phase=phase,
        rounds_total=args.rounds,
        round_num=best_round.round_num if best_round else 0,
        best_score=max(best_score, 0.0),
        kept_count=kept_count,
        latest_score=max(best_score, 0.0) if best_round else None,
        latest_marker="run complete",
        start_ts=start_ts,
        current_stage="complete",
        leaf_total=best_round.leaf_total if best_round else 0,
        leaf_accepted=best_round.leaf_accepted if best_round else 0,
        parent_routed=best_round.parent_routed if best_round else False,
        leaf_worker_count=effective_workers,
        leaf_workers_active=0,
        leaf_workers_queued=0,
        leaf_workers_completed=best_round.leaf_accepted if best_round else 0,
        current_node=args.parent,
        current_parent=args.parent,
        current_leaf=", ".join(best_round.accepted_leaf_names[:3])
        if best_round
        else None,
        top_level_status=(
            "ready" if best_round and best_round.parent_routed else "not_ready"
        ),
        composition_status="complete",
        copper_accounting=final_parent_copper_accounting,
        current_action="run complete",
        preview_paths=_discover_live_preview_paths(project_dir),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)
            f.write("\n")

    print()
    print("=== Hierarchical Autoexperiment Complete ===")
    print(f"Best score: {max(best_score, 0.0):.2f}")
    if best_round:
        print(f"Best round: {best_round.round_num}")
        print(
            f"Leafs:      {best_round.leaf_accepted}/{best_round.leaf_total} accepted"
        )
        print(f"Parent:     {'routed' if best_round.parent_routed else 'not routed'}")
    print(f"Log:        {log_path}")
    print(f"Status:     {status_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
