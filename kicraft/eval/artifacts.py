"""Artifact discovery and parsing shared by every eval front-end.

These read the deterministic files a KiCraft run leaves behind (state.json, the
synthesis check, ERC reports, the generated KiCad tree) and reduce them to small
dicts the dimension scorers consume. They are intentionally source-agnostic: the
same parsers serve the offline harness's harvested run records and the web app's
per-project artifact tree. Every reader degrades to a "not present" shape rather
than raising, so a missing or malformed file never crashes a score.
"""
from __future__ import annotations

import json
from pathlib import Path


def _find_one(run_dir: Path, name: str) -> Path | None:
    """First match of an exact filename anywhere under run_dir (shallowest wins)."""
    hits = sorted(run_dir.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _find_glob(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.rglob(pattern), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _load_json(path: Path | None):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def parse_erc(path: Path | None) -> dict:
    """Count error/warning violations from a KiCad erc.v1.json report."""
    if not path or not path.exists():
        return {"present": False, "errors": None, "warnings": None}
    data = _load_json(path)
    if not isinstance(data, dict):
        return {"present": True, "errors": None, "warnings": None, "note": "unparseable"}
    errors = warnings = 0
    for sheet in data.get("sheets", []):
        for v in sheet.get("violations", []):
            sev = v.get("severity")
            if sev == "error":
                errors += 1
            elif sev == "warning":
                warnings += 1
    return {"present": True, "errors": errors, "warnings": warnings}


def parse_synthesis_check(path: Path | None) -> dict:
    data = _load_json(path)
    if not isinstance(data, dict):
        return {"present": False, "status": None, "failed_checks": None, "checked_at": None}
    checks = data.get("checks", []) or []
    failed = data.get("failed_checks")
    if failed is None:
        failed = [c.get("name") for c in checks if c.get("ok") is False]
    return {
        "present": True,
        "status": data.get("status"),
        "failed_checks": failed,
        "failed_count": len(failed),
        "checked_at": data.get("checked_at"),
        "checks": checks,
    }


def analyze_state(path: Path | None) -> dict:
    s = _load_json(path)
    if not isinstance(s, dict):
        return {"present": False}
    bom = s.get("bom") or {}
    connections = bom.get("connections") or []
    history = s.get("history") or []
    slots = {k: s.get(k) is not None for k in ("intent", "functional_spec", "architecture", "bom")}
    return {
        "present": True,
        "slots": slots,
        "all_slots": all(slots.values()),
        "wiring_done": bool(connections),
        "history_len": len(history),
        "history_first_ts": (history[0].get("timestamp") if history else None),
        "open_questions": len(s.get("open_questions") or []),
        "bom_parts": len(bom.get("parts") or []),
        "project_stem": s.get("project_stem"),
    }


def count_generated(run_dir: Path) -> dict:
    pcb = list(run_dir.rglob("*.kicad_pcb"))
    sch = list(run_dir.rglob("*.kicad_sch"))
    pro = list(run_dir.rglob("*.kicad_pro"))
    return {"pcb": len(pcb), "sch": len(sch), "pro": len(pro), "synthesized": bool(pcb or sch)}
