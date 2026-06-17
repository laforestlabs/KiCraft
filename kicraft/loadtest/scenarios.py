"""Declarative defaults + fixture discovery for the load scenarios."""
from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parents[1]

# The committed mock transcript (reconstructed from a frozen usb-pd-trigger run).
DEFAULT_TRANSCRIPT = _PKG / "fixtures" / "transcript_usb_pd_trigger.json"

# A generic brief; the mock ignores brief text (it replays the transcript slots),
# so this is just the label the pipeline load drives.
DEFAULT_BRIEF = "a usb-c pd trigger board (load-test mock)"

SCENARIOS = ("build-storm", "pipeline", "web")


def find_synth_workspace() -> Path | None:
    """Best-effort: newest synthesized workspace under logs/self_eval (a finished
    run dir with .kicraft/state.json + generated/<stem>). Used as the default
    build-storm source so a run needs no --source on the box."""
    roots = [_REPO / "logs" / "self_eval", Path.home() / ".kicraft" / "self_eval"]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for state in root.glob("*/*/.kicraft/state.json"):
            ws = state.parent.parent
            if (ws / "generated").is_dir() and any((ws / "generated").iterdir()):
                candidates.append(ws)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
