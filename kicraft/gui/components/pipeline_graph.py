"""Pipeline flowchart component -- graphical hierarchy visualization.

Renders the subcircuit hierarchy as a horizontal flowchart:
leaves on the left flowing into the parent/root on the right.
Each node is a clickable card showing status badge and PCB thumbnail.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nicegui import ui

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for pipeline node status
# ---------------------------------------------------------------------------

@dataclass
class RoundInfo:
    """Summary of one solve round for a leaf node."""

    index: int
    score: float
    routed: bool
    thumbnail: str | None = None  # path to round render PNG
    pre_route_thumbnail: str | None = None
    experiment_round: int = 0  # parent round this solve belongs to (0 = unknown)
    # Human-readable reason the round failed to route / was rejected, pulled
    # from routing.reason, rejection_stage, or the first rejection_reasons
    # entry when available. None when the round routed successfully.
    rejection_reason: str | None = None


@dataclass
class NodeStatus:
    """Status of a single hierarchy node (leaf or parent)."""

    name: str
    node_id: str
    is_leaf: bool
    status: str = "pending"  # pending | solving | routing | accepted | failed
    score: float | None = None
    best_render: str | None = None  # path to best/final render PNG
    traces: int = 0
    vias: int = 0
    component_count: int = 0
    rounds: list[RoundInfo] = field(default_factory=list)
    total_rounds_run: int = 0
    artifact_dir: str | None = None


@dataclass
class PipelineState:
    """Full pipeline state for the monitor view."""

    root_name: str = "Project"
    # pending | composing | routing | done | failed | routing_failed
    root_status: str = "pending"
    root_render: str | None = None
    # True if the currently-displayed round's parent was successfully routed,
    # False if it composed but routing failed, None when unknown/not yet run.
    root_routed: bool | None = None
    leaves: list[NodeStatus] = field(default_factory=list)
    phase: str = "idle"
    current_node: str | None = None
    elapsed_s: float = 0.0
    eta_s: float = 0.0
    round_num: int = 0
    total_rounds: int = 0

    def graph_fingerprint(self) -> str:
        """Hash of fields that affect the pipeline graph rendering.

        Used to skip UI rebuilds when only timing/progress changed but
        the graph layout and images are identical.
        """
        parts = [
            self.root_name,
            self.root_status,
            _path_with_mtime(self.root_render),
        ]
        for leaf in self.leaves:
            parts.append(
                f"{leaf.node_id}|{leaf.status}|"
                f"{_path_with_mtime(leaf.best_render)}|{leaf.score}"
            )
        return "|".join(parts)


def _path_with_mtime(path: str | None) -> str:
    """Return 'path@mtime' for fingerprint comparison, or '' if no path."""
    if not path:
        return ""
    try:
        mtime = os.path.getmtime(path)
        return f"{path}@{mtime}"
    except OSError:
        return path


# ---------------------------------------------------------------------------
# Data gathering -- scan artifacts + run_status to build PipelineState
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file safely, returning None on any error."""
    try:
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _find_best_render(renders_dir: Path) -> str | None:
    """Find the best available render for a leaf node."""
    # Priority: routed front > pre-route front > copper both
    candidates = [
        renders_dir / "routed_front_all.png",
        renders_dir / "pre_route_front_all.png",
        renders_dir / "routed_copper_both.png",
        renders_dir / "pre_route_copper_both.png",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _find_round_renders(renders_dir: Path, round_index: int) -> tuple[str | None, str | None]:
    """Find routed and pre-route renders for a specific round."""
    routed = renders_dir / f"round_{round_index:04d}_routed_front_all.png"
    pre_route = renders_dir / f"round_{round_index:04d}_pre_route_front_all.png"
    return (
        str(routed) if routed.exists() else None,
        str(pre_route) if pre_route.exists() else None,
    )


def _load_round_statuses(experiments_dir: Path) -> dict[int, dict[str, Any]]:
    """Read experiments.jsonl and return per-round parent routing status."""
    log = experiments_dir / "experiments.jsonl"
    result: dict[int, dict[str, Any]] = {}
    if not log.exists():
        return result
    try:
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rn = rec.get("round_num")
                if isinstance(rn, int) and rn > 0:
                    result[rn] = {
                        "parent_composed": bool(rec.get("parent_composed", False)),
                        "parent_routed": bool(rec.get("parent_routed", False)),
                        "score": rec.get("score"),
                        "leaf_accepted": rec.get("leaf_accepted"),
                        "leaf_total": rec.get("leaf_total"),
                    }
    except OSError:
        pass
    return result


def _leaf_is_pinned(experiments_dir: Path, leaf_key: str) -> bool:
    """Return True if pins.json claims this leaf is pinned.

    Used by ``gather_pipeline_state`` to decide whether to fall back
    to canonical state when the selected_round filter yields no
    rounds. A pin means "use this exact state regardless of which
    parent round the chart is on" -- so don't blow away score /
    render / status just because the parent round produced no per-leaf
    rounds (the parents-only-after-leaves-only path).
    """
    try:
        from kicraft.autoplacer.brain import pins as pins_module

        return pins_module.is_pinned(experiments_dir, leaf_key) is not None
    except Exception:
        return False


def _determine_leaf_status(artifact_dir: Path, *, run_in_progress: bool = False) -> str:
    """Determine leaf status from artifact presence.

    ``run_in_progress`` is True while a fresh autoexperiment run is
    still solving. In that mode, ``debug.json`` is the authoritative
    "this leaf has been processed in the current run" signal -- the
    run-start cleanup deletes it, so its absence means "queued, not
    yet visited". We avoid trusting stale canonical files
    (``solved_layout.json`` validation, ``leaf_routed.kicad_pcb``)
    while running, since they describe the previous run's output.
    """
    debug_path = artifact_dir / "debug.json"
    debug_present = debug_path.is_file()

    if run_in_progress and not debug_present:
        # Fresh-run state: cleanup removed debug.json, the leaf has
        # not been processed yet. Don't read stale canonical.
        return "queued"

    solved = artifact_dir / "solved_layout.json"
    if solved.exists():
        data = _safe_read_json(solved)
        if isinstance(data, dict):
            validation = data.get("validation", {})
            if isinstance(validation, dict):
                if validation.get("accepted"):
                    return "accepted"
                if validation.get("failed") or validation.get("rejected"):
                    return "failed"

    if debug_present:
        debug = _safe_read_json(debug_path)
        if isinstance(debug, dict):
            if debug.get("error") or debug.get("failed"):
                return "failed"

    if (artifact_dir / "leaf_routed.kicad_pcb").exists():
        return "routing"
    if (artifact_dir / "leaf_pre_freerouting.kicad_pcb").exists():
        return "solving"
    if (artifact_dir / "metadata.json").exists():
        return "solving"
    return "pending"


def _build_rounds_from_debug(artifact_dir: Path, renders_dir: Path) -> list[RoundInfo]:
    """Extract per-round info from debug.json."""
    debug = _safe_read_json(artifact_dir / "debug.json")
    if not isinstance(debug, dict):
        return []

    all_rounds = debug.get("extra", {})
    if not isinstance(all_rounds, dict):
        return []
    all_rounds = all_rounds.get("all_rounds", [])
    if not isinstance(all_rounds, list):
        return []

    rounds: list[RoundInfo] = []
    for r in all_rounds:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("round_index", len(rounds)))
        except (TypeError, ValueError):
            idx = len(rounds)
        try:
            score = float(r.get("score", 0.0)) if r.get("score") is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        routed = bool(r.get("routed", False))

        try:
            exp_round = int(r.get("experiment_round", 0) or 0)
        except (TypeError, ValueError):
            exp_round = 0

        rejection_reason: str | None = None
        if not routed:
            routing = r.get("routing") or {}
            if isinstance(routing, dict):
                candidate = (
                    routing.get("reason")
                    or routing.get("rejection_stage")
                )
                if not candidate:
                    reasons = routing.get("rejection_reasons") or []
                    if isinstance(reasons, list) and reasons:
                        candidate = str(reasons[0])
                    else:
                        validation = routing.get("validation") or {}
                        if isinstance(validation, dict):
                            candidate = (
                                validation.get("rejection_stage")
                                or validation.get("rejection_message")
                            )
                if candidate:
                    rejection_reason = str(candidate)

        routed_thumb, pre_route_thumb = _find_round_renders(renders_dir, idx)
        rounds.append(RoundInfo(
            index=idx,
            score=score,
            routed=routed,
            thumbnail=routed_thumb,
            pre_route_thumbnail=pre_route_thumb,
            experiment_round=exp_round,
            rejection_reason=rejection_reason,
        ))
    return rounds


def gather_pipeline_state(
    experiments_dir: Path,
    run_status: dict[str, Any],
    project_dir: Path | None = None,
    project_name: str | None = None,
    selected_round: int | None = None,
) -> PipelineState:
    """Build full pipeline state from artifacts and run_status.

    Args:
        experiments_dir: Path to .experiments/ directory
        run_status: Live run_status.json contents (from runner.read_status())
        project_dir: Project root for hierarchy parsing (populates pending nodes)
        project_name: Fallback project name if hierarchy unavailable

    Returns:
        PipelineState with all leaves populated from disk artifacts + hierarchy
    """
    phase = run_status.get("phase", "idle")
    hierarchy = run_status.get("hierarchy", {})
    if not isinstance(hierarchy, dict):
        hierarchy = {}

    current_stage = (
        hierarchy.get("current_stage")
        or run_status.get("stage")
        or run_status.get("pipeline_phase")
        or phase
    )

    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            return int(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    root_name = project_name or "Project"
    hierarchy_leaves: list[tuple[str, str]] = []

    if project_dir:
        try:
            from ...autoplacer.brain.hierarchy_parser import parse_hierarchy

            hgraph = parse_hierarchy(project_dir)
            root_name = hgraph.root.definition.id.sheet_name or root_name
            for leaf_node in hgraph.leaf_nodes():
                hierarchy_leaves.append((
                    leaf_node.definition.id.sheet_name,
                    leaf_node.definition.id.instance_path,
                ))
        except Exception:
            logger.debug("Could not parse hierarchy for pending nodes", exc_info=True)

    state = PipelineState(
        root_name=root_name,
        phase=phase,
        current_node=run_status.get("current_leaf") or hierarchy.get("current_leaf"),
        elapsed_s=_safe_float(run_status.get("elapsed_s", 0)),
        eta_s=_safe_float(run_status.get("eta_s", 0)),
        round_num=_safe_int(run_status.get("round", 0)),
        total_rounds=_safe_int(run_status.get("total_rounds", 0)),
    )

    # "Run in progress" gates leaf status decisions below: while
    # running we trust per-leaf debug.json over stale canonical files
    # (the cleanup wiped debug.json at run start, so its absence
    # genuinely means "leaf not yet processed in this run").
    run_in_progress = phase in ("running", "stopping", "starting") or current_stage in (
        "solve_leafs",
        "compose_parent",
        "route_parent",
        "score_round",
    )

    if phase == "done":
        state.root_status = "done"
    elif current_stage in ("done", "complete", "score_round"):
        state.root_status = "done"
    # We may downgrade root_status to "routing_failed" below once we know
    # the most-recent round's parent_routed flag. Doing it here would be
    # premature -- round_parent_routed isn't populated until line ~370.
    elif current_stage == "route_parent":
        state.root_status = "routing"
    elif current_stage == "compose_parent":
        state.root_status = "composing"
    elif current_stage == "solve_leafs" or phase == "running":
        state.root_status = "pending"
    elif phase == "error":
        state.root_status = "failed"

    preview_paths = run_status.get("preview_paths", {})
    if not isinstance(preview_paths, dict):
        preview_paths = {}

    round_statuses = _load_round_statuses(experiments_dir)

    # Determine which round drives the parent render + status. For a user
    # selection we use that round; otherwise we use the most recent completed
    # round so the Monitor reflects reality, not an older best.
    status_round: int | None = None
    if selected_round is not None and selected_round > 0:
        status_round = selected_round
    elif round_statuses:
        status_round = max(round_statuses.keys())

    round_parent_routed: bool | None = None
    round_parent_composed: bool | None = None
    if status_round is not None and status_round in round_statuses:
        round_parent_routed = round_statuses[status_round]["parent_routed"]
        round_parent_composed = round_statuses[status_round]["parent_composed"]

    # If the run is "done" but the displayed round's parent failed to
    # route, downgrade root_status so the parent node card shows its
    # red "FAILED TO ROUTE" badge and the detail panel triggers the
    # rejection-reason banner. Without this, a leaves-only or pin-bad
    # run displays "done" with no indication anything went wrong.
    if (
        state.root_status == "done"
        and round_parent_routed is False
    ):
        state.root_status = "routing_failed"

    # Per-round parent render. When the round failed to route, prefer the
    # pre-route (stamped) snapshot over the routed one -- the routed PNG may
    # not exist, or may be the reject-candidate that misled the user.
    if status_round is not None and status_round > 0:
        round_dir = (
            experiments_dir
            / "hierarchical_autoexperiment"
            / f"round_{status_round:04d}"
        )
        if round_parent_routed is False:
            preferred_names = ("parent_stamped.png", "parent_routed.png")
        else:
            preferred_names = ("parent_routed.png", "parent_stamped.png")
        for name in preferred_names:
            p = round_dir / name
            if p.exists():
                state.root_render = str(p)
                break

    if state.root_render is None:
        parent_routed = preview_paths.get("parent_routed_preview")
        parent_stamped = preview_paths.get("parent_stamped_preview")
        # When the round we're viewing failed to route, prefer the stamped
        # preview path from run_status as well.
        if round_parent_routed is False:
            if parent_stamped and Path(str(parent_stamped)).exists():
                state.root_render = str(parent_stamped)
            elif parent_routed and Path(str(parent_routed)).exists():
                state.root_render = str(parent_routed)
        else:
            if parent_routed and Path(str(parent_routed)).exists():
                state.root_render = str(parent_routed)
            elif parent_stamped and Path(str(parent_stamped)).exists():
                state.root_render = str(parent_stamped)
        if state.root_render is None:
            hp = experiments_dir / "hierarchical_pipeline"
            for name in ("parent_routed.png", "parent_stamped.png"):
                p = hp / name
                if p.exists():
                    state.root_render = str(p)
                    break

    # Last resort: when the parent composition's metadata/debug JSON is
    # missing (acceptance gate rejection, truncated run, etc.) the discovery
    # helpers return nothing. Probe subcircuits/*/renders/ directly so we
    # still surface whatever parent render was produced.
    if state.root_render is None:
        sub_root = experiments_dir / "subcircuits"
        if sub_root.exists():
            best_mtime = -1.0
            best_path: str | None = None
            probe_names = (
                ("parent_stamped.png", "parent_routed.png")
                if round_parent_routed is False
                else ("parent_routed.png", "parent_stamped.png")
            )
            for child in sub_root.iterdir():
                if not child.is_dir():
                    continue
                for name in probe_names:
                    candidate = child / "renders" / name
                    if candidate.exists():
                        try:
                            mt = candidate.stat().st_mtime
                        except OSError:
                            mt = 0.0
                        if mt > best_mtime:
                            best_mtime = mt
                            best_path = str(candidate)
            if best_path:
                state.root_render = best_path

    # Override root_status and root_routed for the round being viewed so the
    # UI shows "FAILED TO ROUTE" instead of a misleading "DONE" badge.
    state.root_routed = round_parent_routed
    if status_round is not None and status_round in round_statuses:
        if round_parent_routed is False:
            state.root_status = (
                "routing_failed" if round_parent_composed else "failed"
            )
        elif round_parent_routed is True:
            state.root_status = "done"

    sub_root = experiments_dir / "subcircuits"
    if sub_root.exists():
        for artifact_dir in sorted(sub_root.iterdir()):
            if not artifact_dir.is_dir():
                continue
            meta = _safe_read_json(artifact_dir / "metadata.json")
            if not isinstance(meta, dict):
                continue

            # Skip parent composition artifacts -- they are not leaves
            if meta.get("parent_composition"):
                continue

            sheet_name = meta.get("sheet_name", artifact_dir.name)
            instance_path = meta.get("instance_path", "")
            component_refs = meta.get("component_refs", [])

            renders_dir = artifact_dir / "renders"
            leaf_status = _determine_leaf_status(
                artifact_dir, run_in_progress=run_in_progress
            )
            best_render = _find_best_render(renders_dir) if renders_dir.exists() else None

            score = None
            traces = 0
            vias = 0
            solved = _safe_read_json(artifact_dir / "solved_layout.json")
            if isinstance(solved, dict):
                score = solved.get("score")
                if isinstance(score, (int, float)):
                    score = float(score)
                else:
                    score = None
                traces_val = solved.get("traces", [])
                traces = len(traces_val) if isinstance(traces_val, list) else 0
                vias_val = solved.get("vias", [])
                vias = len(vias_val) if isinstance(vias_val, list) else 0
                if not sheet_name or sheet_name == artifact_dir.name:
                    sheet_name = solved.get("sheet_name", sheet_name)

            rounds = _build_rounds_from_debug(artifact_dir, renders_dir)

            # If the user has selected a specific parent round, narrow rounds
            # and best_render to that round's data.
            if selected_round is not None:
                filtered = [
                    r for r in rounds
                    if r.experiment_round == selected_round
                ]
                if filtered:
                    rounds = filtered
                    # Prefer the best *routed* round with a thumbnail. If no
                    # routed round produced a render at all and any round
                    # failed routing outright, mark the leaf as routing_failed
                    # and show the pre-route preview. Trivial routes (leaves
                    # with no internal nets) still count as successful even
                    # though they produce no PNG.
                    routed_w_render = [r for r in filtered if r.routed and r.thumbnail]
                    any_failed = any(not r.routed for r in filtered)
                    if routed_w_render:
                        best_of_round = max(
                            routed_w_render,
                            key=lambda r: (r.score if r.score is not None else float("-inf")),
                        )
                        best_render = best_of_round.thumbnail or best_render
                    elif any_failed:
                        pre_candidates = [
                            r for r in filtered if r.pre_route_thumbnail
                        ]
                        if pre_candidates:
                            best_of_round = max(
                                pre_candidates,
                                key=lambda r: (r.score if r.score is not None else float("-inf")),
                            )
                            best_render = best_of_round.pre_route_thumbnail
                        leaf_status = "routing_failed"
                else:
                    # No rounds for this leaf in the selected parent round.
                    # Three cases:
                    #   (a) the leaf is PINNED -- the user explicitly froze
                    #       its state from a prior run. selected_round is
                    #       a parent-round filter that's irrelevant to a
                    #       pinned leaf; fall back to canonical state.
                    #       This is the parents-only-after-leaves-only
                    #       workflow where parent rounds don't produce
                    #       per-leaf rounds at all.
                    #   (b) the run is still in flight and this leaf
                    #       hasn't been solved yet -- "queued" / WAITING.
                    #   (c) the run finished and this leaf actually
                    #       missed the round -- that's a real failure.
                    leaf_key = artifact_dir.name
                    is_pinned = _leaf_is_pinned(experiments_dir, leaf_key)
                    if is_pinned:
                        # Keep canonical-derived leaf_status / score /
                        # traces / vias / best_render -- the pin already
                        # represents the user's chosen state for THIS
                        # leaf. We just clear the per-round timeline
                        # because the parent round in question didn't
                        # produce any.
                        rounds = []
                    else:
                        rounds = []
                        best_render = None
                        score = None
                        traces = 0
                        vias = 0
                        leaf_status = "queued" if run_in_progress else "failed"

            node = NodeStatus(
                name=sheet_name,
                node_id=instance_path or artifact_dir.name,
                is_leaf=True,
                status=leaf_status,
                score=score,
                best_render=best_render,
                traces=traces,
                vias=vias,
                component_count=len(component_refs) if isinstance(component_refs, list) else 0,
                rounds=rounds,
                total_rounds_run=len(rounds),
                artifact_dir=str(artifact_dir),
            )
            state.leaves.append(node)

    if hierarchy_leaves:
        existing_ids = {leaf.node_id for leaf in state.leaves}
        for sheet_name, instance_path in hierarchy_leaves:
            node_id = instance_path or sheet_name
            if node_id in existing_ids:
                continue
            state.leaves.append(NodeStatus(
                name=sheet_name,
                node_id=node_id,
                is_leaf=True,
                status="pending",
            ))

    if state.current_node:
        for leaf in state.leaves:
            if state.current_node in (leaf.name, leaf.node_id):
                # The leaf the runner is currently working on becomes
                # "solving" regardless of whether it was previously
                # "pending" (never started) or "queued" (run in progress
                # but not yet visited).
                if leaf.status in ("pending", "queued"):
                    leaf.status = "solving"

    return state


# ---------------------------------------------------------------------------
# UI component -- pipeline flowchart
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "pending": "grey",
    "queued": "grey",
    "solving": "blue",
    "routing": "orange",
    "accepted": "green",
    "failed": "red",
    "routing_failed": "red",
    "composing": "amber",
    "done": "green",
}

_STATUS_ICONS = {
    "pending": "hourglass_empty",
    "queued": "schedule",
    "solving": "build",
    "routing": "route",
    "accepted": "check_circle",
    "failed": "error",
    "routing_failed": "wrong_location",
    "composing": "construction",
    "done": "check_circle",
}

_STATUS_LABELS = {
    "queued": "WAITING",
    "routing_failed": "FAILED TO ROUTE",
}


def pipeline_graph(
    state: PipelineState,
    on_node_select: Callable[[NodeStatus | None], None] | None = None,
    selected_node_id: str | None = None,
    pin_status: dict[str, str] | None = None,
) -> None:
    """Render the pipeline flowchart.

    Horizontal layout: leaf cards on left -> connecting lines -> root card on right.

    ``pin_status`` maps each leaf's ``node_id`` to one of:

    * ``"pinned"`` -- a round is pinned for this leaf; show normal styling
    * ``"unpinned"`` -- snapshots exist but no pin set; highlight (amber dashed border + PIN? badge)
    * ``"no-snapshots"`` -- nothing to pin (e.g. trivial leaves with 0 internal nets); normal styling

    When ``pin_status`` is None, every leaf renders normally (no highlighting).
    """
    pin_status = pin_status or {}
    with ui.row().classes("w-full items-center justify-center gap-0 min-h-[300px]"):
        with ui.column().classes("gap-2 items-end"):
            for leaf in state.leaves:
                _leaf_card(
                    leaf,
                    on_node_select,
                    selected_node_id,
                    pin_state=pin_status.get(leaf.node_id, "no-snapshots"),
                )

        with ui.column().classes("items-center justify-center px-4"):
            n_leaves = len(state.leaves)
            svg_h = max(200, n_leaves * 36 + 20)
            mid_y = svg_h // 2
            ui.html(
                f'<svg width="60" height="{svg_h}" class="text-gray-500">'
                '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
                'refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" '
                'fill="currentColor"/></marker></defs>'
                + "".join(
                    f'<line x1="0" y1="{20 + i * (svg_h - 40) // max(n_leaves - 1, 1)}" '
                    f'x2="50" y2="{mid_y}" '
                    f'stroke="currentColor" stroke-width="1.5" marker-end="url(#arrow)" '
                    f'opacity="0.5"/>'
                    for i in range(n_leaves)
                )
                + "</svg>"
            )

        _root_card(state, on_node_select, selected_node_id)


def _leaf_card(
    node: NodeStatus,
    on_select: Callable[[NodeStatus | None], None] | None,
    selected_id: str | None,
    pin_state: str = "no-snapshots",
) -> None:
    """Render a single leaf node card.

    ``pin_state`` (one of ``"pinned"`` / ``"unpinned"`` / ``"no-snapshots"``)
    controls the unpinned-leaf highlight: leaves with snapshots that the
    user hasn't picked yet get a dashed amber border and a "PIN?" badge so
    the leaves-first workflow's progress is visible at a glance.
    """
    is_selected = selected_id == node.node_id
    if is_selected:
        border = "border-2 border-blue-400"
    elif pin_state == "unpinned":
        border = "border-2 border-dashed border-amber-400"
    else:
        border = "border border-slate-600"
    color = _STATUS_COLORS.get(node.status, "grey")

    def _handle_click(n: NodeStatus = node) -> None:
        if on_select:
            on_select(n)

    with ui.card().classes(
        f"p-2 w-[180px] cursor-pointer hover:border-blue-300 {border} bg-slate-800/80"
    ).on("click", lambda _e, n=node: _handle_click(n)):
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon(_STATUS_ICONS.get(node.status, "circle")).classes(f"text-{color}-400")
            ui.label(node.name).classes("font-medium text-sm truncate flex-1")
            if pin_state == "unpinned":
                ui.badge("PIN?", color="amber").classes("text-[10px]")
            elif pin_state == "pinned":
                ui.badge("PINNED", color="amber").classes("text-[10px]")
            badge_text = _STATUS_LABELS.get(node.status, node.status.upper())
            ui.badge(badge_text, color=color).classes("text-[10px]")

        if node.best_render:
            ui.image(node.best_render).classes(
                "w-full h-[80px] object-contain rounded mt-1 bg-slate-950"
            )
        else:
            with ui.row().classes("w-full h-[80px] items-center justify-center bg-slate-950 rounded mt-1"):
                ui.icon("image_not_supported", size="sm").classes("text-gray-600")

        with ui.row().classes("w-full items-center gap-1 mt-1"):
            if node.score is not None:
                ui.label(f"{node.score:.1f}").classes("text-[11px] text-green-400 font-mono")
            if node.traces:
                ui.label(f"T{node.traces}").classes("text-[11px] text-cyan-300 font-mono")
            if node.vias:
                ui.label(f"V{node.vias}").classes("text-[11px] text-amber-300 font-mono")
            ui.space()
            if node.total_rounds_run:
                ui.label(f"R{node.total_rounds_run}").classes("text-[11px] text-gray-400")


def _root_card(
    state: PipelineState,
    on_select: Callable[[NodeStatus | None], None] | None,
    selected_id: str | None,
) -> None:
    """Render the root/parent node card."""
    is_selected = selected_id == "__root__"
    border = "border-2 border-blue-400" if is_selected else "border border-slate-600"
    color = _STATUS_COLORS.get(state.root_status, "grey")

    root_node = NodeStatus(
        name=state.root_name,
        node_id="__root__",
        is_leaf=False,
        status=state.root_status,
        best_render=state.root_render,
    )

    with ui.card().classes(
        f"p-3 w-[200px] cursor-pointer hover:border-blue-300 {border} bg-slate-800/80"
    ).on("click", lambda _e, n=root_node: on_select(n) if on_select else None):
        with ui.row().classes("items-center gap-2 w-full"):
            ui.icon(_STATUS_ICONS.get(state.root_status, "circle")).classes(
                f"text-{color}-400 text-lg"
            )
            ui.label(state.root_name).classes("font-bold text-base")
            root_badge = _STATUS_LABELS.get(state.root_status, state.root_status.upper())
            ui.badge(root_badge, color=color).classes("text-[10px]")

        if state.root_status == "routing_failed":
            ui.label("Parent Assembly (pre-route shown)").classes(
                "text-xs text-red-400"
            )
        else:
            ui.label("Parent Assembly").classes("text-xs text-gray-400")

        if state.root_render:
            ui.image(state.root_render).classes(
                "w-full h-[120px] object-contain rounded mt-2 bg-slate-950"
            )
        else:
            with ui.row().classes(
                "w-full h-[120px] items-center justify-center bg-slate-950 rounded mt-2"
            ):
                ui.icon("account_tree", size="lg").classes("text-gray-600")

        accepted = sum(1 for l in state.leaves if l.status == "accepted")
        total = len(state.leaves)
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            ui.label(f"Leaves: {accepted}/{total}").classes("text-xs text-gray-300")
            if total > 0:
                ui.linear_progress(
                    value=accepted / total, color="green"
                ).classes("flex-1").props("size=6px")
