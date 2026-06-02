"""Whole-pipeline state reader for the GUI tracker.

Merges the upstream CircuitChat design stages (``.kicraft/state.json``, written
in the user's project CWD) with the downstream place/route/fab signals
(``.experiments/run_status.json`` and ``fab/``, written into the synthesized
project tree). It is deliberately tolerant of where those live: a CircuitChat
project keeps ``state.json`` in the CWD and synthesizes into
``generated/<STEM>/``, but the synthesized dir may also be the project root
itself. This module finds whichever layout is present, so the tracker works
whether the GUI is launched from the user's CWD (observing from intent onward)
or from a synthesized project dir.

Pure Python (no NiceGUI) so it is cheap to unit-test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The full pipeline, in order. Upstream keys map to ConversationState slots;
# downstream keys are derived from route/fab artifacts.
_UPSTREAM: list[tuple[str, str]] = [
    ("intent", "Intent"),
    ("functional_spec", "Spec"),
    ("architecture", "Architecture"),
    ("bom", "BOM"),
    ("wiring", "Wiring"),
]


@dataclass
class StageStatus:
    key: str
    label: str
    state: str  # "done" | "active" | "pending"
    detail: str = ""


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def find_synth_project(project_root: Path) -> Path | None:
    """Locate the synthesized project dir (the one holding ``*.kicad_pro``).

    Checks the root itself first, then ``generated/<STEM>/``. Returns None
    before synthesis has run.
    """
    if list(project_root.glob("*.kicad_pro")):
        return project_root
    hits = sorted(project_root.glob("generated/*/*.kicad_pro"))
    return hits[0].parent if hits else None


def pipeline_progress(project_root: Path) -> list[StageStatus]:
    """Return the ordered pipeline stages with done/active/pending status.

    ``project_root`` is the user's project dir (where ``.kicraft/state.json``
    lives). Downstream stages are read from the synthesized project under it.
    """
    project_root = Path(project_root)
    state = _load_json(project_root / ".kicraft" / "state.json") or {}
    bom = state.get("bom") or {}

    stages: list[StageStatus] = []
    for key, label in _UPSTREAM:
        if key == "bom":
            present = bool(bom.get("parts"))
        elif key == "wiring":
            present = bool(bom.get("connections"))
        else:
            present = bool(state.get(key))
        stages.append(StageStatus(key, label, "done" if present else "pending"))

    synth_dir = find_synth_project(project_root)
    stages.append(
        StageStatus(
            "synthesize",
            "Synthesize",
            "done" if synth_dir is not None else "pending",
        )
    )

    # Downstream signals live under the synthesized project dir.
    run_status = (
        _load_json(synth_dir / ".experiments" / "run_status.json")
        if synth_dir
        else None
    ) or {}
    routed = bool(synth_dir and list(synth_dir.glob("**/parent_routed.kicad_pcb")))
    fab_dir = (synth_dir / "fab") if synth_dir else None
    gerbers = (
        [p for p in fab_dir.glob("*.g*") if p.is_file()]
        if (fab_dir and fab_dir.exists())
        else []
    )

    route_state, route_detail = "pending", ""
    phase = run_status.get("phase")
    if phase == "running":
        pct = run_status.get("progress_percent")
        route_state = "active"
        route_detail = f"{pct}%" if pct is not None else "routing"
    elif phase in ("done", "stopping") or routed:
        route_state = "done"
    stages.append(StageStatus("route", "Place & Route", route_state, route_detail))

    stages.append(
        StageStatus(
            "fab",
            "Fab (Gerbers)",
            "done" if gerbers else "pending",
            f"{len(gerbers)} layers" if gerbers else "",
        )
    )

    # Highlight the current stage: the first non-done one, unless a stage is
    # already actively running (the router).
    if not any(s.state == "active" for s in stages):
        for s in stages:
            if s.state == "pending":
                s.state = "active"
                break

    return stages
