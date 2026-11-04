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

from kicraft.design.advisories import recorded_advisory_codes


def _find_one(run_dir: Path, name: str) -> Path | None:
    """First match of an exact filename anywhere under run_dir (shallowest wins)."""
    hits = sorted(run_dir.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _find_glob(run_dir: Path, pattern: str) -> Path | None:
    hits = sorted(run_dir.rglob(pattern), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def find_erc_report(run_dir: Path, synth: dict) -> Path | None:
    """Use the current ERC receipt when present; only legacy checks use discovery."""
    checks = synth.get("checks") or []
    rows = [
        row for row in checks
        if isinstance(row, dict) and row.get("name") == "9.12 ERC"
    ]
    explicit = [row for row in rows if "report_path" in row]
    if not explicit:
        return _find_glob(run_dir, "*_erc.rpt")
    if len(rows) != 1:
        return None
    name = explicit[0]["report_path"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    root = Path(run_dir).resolve()
    matches = []
    for path in root.rglob("*_erc.rpt"):
        if path.name != name or not path.is_file():
            continue
        try:
            path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        matches.append(path)
    return matches[0] if len(matches) == 1 else None


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
        # RECORD-class advisories the committed design shipped with (design-yield-recovery plan
        # §4.2): the rubric's recording gate and the self-eval record both read them here.
        "advisory_codes": recorded_advisory_codes(s.get("architecture"), bom),
    }


def count_generated(run_dir: Path) -> dict:
    state = _load_json(Path(run_dir) / ".kicraft" / "state.json")
    artifacts = state.get("artifacts") if isinstance(state, dict) else None
    directory = artifacts.get("project_dir") if isinstance(artifacts, dict) else None
    if directory is not None:
        root = Path(run_dir).resolve()
        if not isinstance(directory, str) or not directory:
            return {"pcb": 0, "sch": 0, "pro": 0, "synthesized": False}
        project = Path(directory)
        if not project.is_absolute():
            project = root / project
        try:
            project.resolve().relative_to((root / "generated").resolve())
        except (OSError, ValueError):
            return {"pcb": 0, "sch": 0, "pro": 0, "synthesized": False}
        pcb = list(project.glob("*.kicad_pcb"))
        sch = list(project.glob("*.kicad_sch"))
        pro = list(project.glob("*.kicad_pro"))
    else:
        pcb = list(run_dir.rglob("*.kicad_pcb"))
        sch = list(run_dir.rglob("*.kicad_sch"))
        pro = list(run_dir.rglob("*.kicad_pro"))
    return {"pcb": len(pcb), "sch": len(sch), "pro": len(pro), "synthesized": bool(pcb or sch)}
