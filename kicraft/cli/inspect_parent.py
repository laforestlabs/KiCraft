#!/usr/bin/env python3
"""inspect_parent — generic AI-agent-friendly diagnostic for a stamped/routed parent PCB.

Reads any KiCad parent .kicad_pcb (no project-specific paths baked in) and produces:

  - Structured JSON report (machine-readable; all geometry in mm)
  - Markdown summary (human-readable; AI agents can scan it quickly)
  - Annotated top-view PNG: overlays constraint anchors, "PCB Edge" markers,
    leaf labels, and edge-overhang status
  - Stacking heatmap PNG: colored cells showing front-only / back-only /
    stacked / empty regions on a 5 mm grid

Project config is auto-discovered via discover_project_config(pcb_dir).
Constraint info comes from the project's component_zones section. The
inspector itself is project-agnostic -- a leaf-board inspector can be
built on the same base by pointing it at a different PCB.

Usage:
    python -m kicraft.cli.inspect_parent <pcb_path> [--output-dir DIR]

Exit code 0 always (this is a diagnostic, not a gate).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pcbnew
from PIL import Image, ImageDraw, ImageFont

from kicraft.autoplacer.config import discover_project_config, load_project_config


GRID_MM = 5.0
ANNOTATION_PAD_MM = 4.0  # how far outside the board to draw labels
KICAD_CLI = shutil.which("kicad-cli") or "kicad-cli"


@dataclass
class Bbox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "width": self.width,
            "height": self.height,
        }

    def overlaps(self, other: "Bbox") -> bool:
        return (
            self.min_x < other.max_x
            and self.max_x > other.min_x
            and self.min_y < other.max_y
            and self.max_y > other.min_y
        )


def _to_mm(nm: int) -> float:
    return nm / 1e6


def _bbox_from_kibbox(kb) -> Bbox:
    return Bbox(
        min_x=_to_mm(kb.GetLeft()),
        min_y=_to_mm(kb.GetTop()),
        max_x=_to_mm(kb.GetRight()),
        max_y=_to_mm(kb.GetBottom()),
    )


def _board_outline_bbox(board) -> Bbox:
    return _bbox_from_kibbox(board.GetBoardEdgesBoundingBox())


def _footprint_courtyard(footprint) -> Bbox:
    layer = pcbnew.F_CrtYd if footprint.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
    try:
        c = footprint.GetCourtyard(layer)
        bb = c.BBox()
        if bb.GetWidth() > 0 and bb.GetHeight() > 0:
            return _bbox_from_kibbox(bb)
    except Exception:
        pass
    return _bbox_from_kibbox(footprint.GetBoundingBox(False, False))


def _edge_marker(footprint) -> tuple[float, float] | None:
    """Return ('PCB Edge' marker world position, mm) or None.

    The marker convention: a Dwgs.User layer text or line in the footprint
    indicates where the actual board edge should align relative to the
    footprint -- used by edge-attached connectors that physically overhang.
    """
    for item in footprint.GraphicalItems():
        try:
            if item.GetLayer() != pcbnew.Dwgs_User:
                continue
        except Exception:
            continue
        try:
            text = item.GetText() if hasattr(item, "GetText") else ""
        except Exception:
            text = ""
        if "edge" in text.lower():
            pos = item.GetPosition()
            return _to_mm(pos.x), _to_mm(pos.y)
        if hasattr(item, "GetStart") and hasattr(item, "GetEnd"):
            s, e = item.GetStart(), item.GetEnd()
            return (
                _to_mm((s.x + e.x) / 2),
                _to_mm((s.y + e.y) / 2),
            )
    return None


@dataclass
class FootprintInfo:
    ref: str
    layer: str  # "front" or "back"
    pos: tuple[float, float]
    courtyard: Bbox
    edge_marker: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "layer": self.layer,
            "pos": list(self.pos),
            "courtyard": self.courtyard.to_dict(),
            "edge_marker": list(self.edge_marker) if self.edge_marker else None,
        }


@dataclass
class EdgeFinding:
    ref: str
    edge: str
    marker_world: tuple[float, float]
    marker_distance_from_edge_mm: float
    courtyard_overhangs: bool
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "edge": self.edge,
            "marker_world": list(self.marker_world),
            "marker_distance_from_edge_mm": self.marker_distance_from_edge_mm,
            "courtyard_overhangs": self.courtyard_overhangs,
            "interpretation": self.interpretation,
        }


@dataclass
class DRCViolation:
    type: str
    severity: str
    description: str
    pos: tuple[float, float] | None
    refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "description": self.description,
            "pos": list(self.pos) if self.pos else None,
            "refs": list(self.refs),
        }


@dataclass
class DRCSummary:
    ran: bool = False
    error_count: int = 0
    warning_count: int = 0
    unconnected: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    violations: list[DRCViolation] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ran": self.ran,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "unconnected": self.unconnected,
            "by_type": dict(self.by_type),
            "violations": [v.to_dict() for v in self.violations],
            "note": self.note,
        }


@dataclass
class Issue:
    severity: str  # "error" / "warning" / "info"
    kind: str
    message: str
    ref: str | None = None
    location: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "message": self.message,
            "ref": self.ref,
            "location": list(self.location) if self.location else None,
        }


_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Report:
    pcb_path: str
    board_outline: Bbox
    footprints: list[FootprintInfo] = field(default_factory=list)
    edge_findings: list[EdgeFinding] = field(default_factory=list)
    drc: DRCSummary = field(default_factory=DRCSummary)
    issues: list[Issue] = field(default_factory=list)
    grid_mm: float = GRID_MM
    cells_total: int = 0
    cells_empty: int = 0
    cells_front_only: int = 0
    cells_back_only: int = 0
    cells_stacked: int = 0

    @property
    def board_area_mm2(self) -> float:
        return self.board_outline.area

    @property
    def wasted_area_mm2(self) -> float:
        return self.cells_empty * self.grid_mm * self.grid_mm

    @property
    def stacked_area_mm2(self) -> float:
        return self.cells_stacked * self.grid_mm * self.grid_mm

    @property
    def wasted_fraction(self) -> float:
        if self.board_area_mm2 <= 0:
            return 0.0
        return self.wasted_area_mm2 / self.board_area_mm2

    @property
    def stacked_fraction(self) -> float:
        if self.board_area_mm2 <= 0:
            return 0.0
        return self.stacked_area_mm2 / self.board_area_mm2

    @property
    def stacking_efficiency(self) -> float:
        """Of the back-side footprint area, what fraction has a front-side
        stack on it. 1.0 = every back cell has a front stack; 0.0 = none.
        High values mean dual-layer real estate is being used well.
        """
        denom = self.cells_stacked + self.cells_back_only
        if denom <= 0:
            return 0.0
        return self.cells_stacked / denom

    @property
    def packing_density(self) -> float:
        """Fraction of the board occupied by at least one footprint
        (front or back). 1.0 = full board; 0.0 = empty board."""
        if self.cells_total <= 0:
            return 0.0
        return (
            self.cells_stacked
            + self.cells_front_only
            + self.cells_back_only
        ) / self.cells_total

    def to_dict(self) -> dict:
        return {
            "pcb_path": self.pcb_path,
            "board_outline": self.board_outline.to_dict(),
            "board_area_mm2": self.board_area_mm2,
            "footprints": [fp.to_dict() for fp in self.footprints],
            "edge_findings": [ef.to_dict() for ef in self.edge_findings],
            "drc": self.drc.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "area_grid_mm": self.grid_mm,
            "cells_total": self.cells_total,
            "cells_empty": self.cells_empty,
            "cells_front_only": self.cells_front_only,
            "cells_back_only": self.cells_back_only,
            "cells_stacked": self.cells_stacked,
            "wasted_area_mm2": self.wasted_area_mm2,
            "stacked_area_mm2": self.stacked_area_mm2,
            "wasted_fraction": self.wasted_fraction,
            "stacked_fraction": self.stacked_fraction,
            "stacking_efficiency": self.stacking_efficiency,
            "packing_density": self.packing_density,
        }


def _run_drc(pcb_path: Path) -> DRCSummary:
    if not shutil.which(KICAD_CLI):
        return DRCSummary(ran=False, note="kicad-cli not found in PATH")
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                KICAD_CLI,
                "pcb",
                "drc",
                "--format",
                "json",
                "--severity-error",
                "--severity-warning",
                "-o",
                str(tmp_path),
                str(pcb_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            return DRCSummary(
                ran=False,
                note=f"kicad-cli drc produced no output (rc={result.returncode}): {result.stderr[:200]}",
            )
        data = json.loads(tmp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return DRCSummary(ran=False, note=f"drc failed: {exc}")
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    violations: list[DRCViolation] = []
    by_type: dict[str, int] = {}
    error_count = 0
    warning_count = 0
    for v in data.get("violations", []):
        sev = v.get("severity", "error")
        if sev == "error":
            error_count += 1
        elif sev == "warning":
            warning_count += 1
        items = v.get("items", []) or []
        refs: list[str] = []
        pos = None
        for it in items:
            desc = it.get("description", "")
            # Try to extract ref like "of J1" or "[GND] of J1"
            m_idx = desc.rfind(" of ")
            if m_idx >= 0:
                tail = desc[m_idx + 4 :].strip().split()[0].rstrip(",")
                if tail and tail not in refs:
                    refs.append(tail)
            if pos is None and "pos" in it:
                p = it["pos"]
                pos = (p.get("x", 0.0), p.get("y", 0.0))
        vtype = v.get("type", "unknown")
        by_type[vtype] = by_type.get(vtype, 0) + 1
        violations.append(
            DRCViolation(
                type=vtype,
                severity=sev,
                description=v.get("description", ""),
                pos=pos,
                refs=refs,
            )
        )
    unconnected = len(data.get("unconnected_items") or [])
    return DRCSummary(
        ran=True,
        error_count=error_count,
        warning_count=warning_count,
        unconnected=unconnected,
        by_type=by_type,
        violations=violations,
    )


def _build_issues(report: "Report") -> list[Issue]:
    """Aggregate findings into a single severity-ordered issues list.

    AI agents should be able to skim this list to know what's broken
    without parsing the rest of the report. Nearby same-type DRC
    violations on the same ref are clustered (e.g., 14 clearance
    violations on J1 collapse to one issue) so the list stays
    actionable rather than noisy.
    """
    issues: list[Issue] = []

    # DRC errors -- cluster by (type, ref) so 14 J1-clearance entries
    # collapse to one "J1 has 14 clearance violations" issue.
    drc_clusters: dict[tuple[str, str | None], list[DRCViolation]] = {}
    for v in report.drc.violations:
        if v.severity != "error":
            continue
        ref = v.refs[0] if v.refs else None
        drc_clusters.setdefault((v.type, ref), []).append(v)
    for (vtype, ref), vlist in drc_clusters.items():
        if len(vlist) == 1:
            v = vlist[0]
            issues.append(
                Issue(
                    severity="error",
                    kind=f"drc.{vtype}",
                    message=v.description,
                    ref=ref,
                    location=v.pos,
                )
            )
        else:
            sample = vlist[0]
            issues.append(
                Issue(
                    severity="error",
                    kind=f"drc.{vtype}",
                    message=(
                        f"{len(vlist)} {vtype} violations clustered on "
                        f"{ref or 'multiple refs'} "
                        f"(e.g., {sample.description})"
                    ),
                    ref=ref,
                    location=sample.pos,
                )
            )

    # Edge findings flagged BUG
    for f in report.edge_findings:
        if f.interpretation.startswith("BUG"):
            issues.append(
                Issue(
                    severity="error",
                    kind="edge.marker_inside_board",
                    message=f.interpretation,
                    ref=f.ref,
                    location=f.marker_world,
                )
            )
        elif f.interpretation.startswith("WARN"):
            issues.append(
                Issue(
                    severity="warning",
                    kind="edge.marker_anomaly",
                    message=f.interpretation,
                    ref=f.ref,
                    location=f.marker_world,
                )
            )

    # DRC warnings -- same clustering treatment.
    warn_clusters: dict[tuple[str, str | None], list[DRCViolation]] = {}
    for v in report.drc.violations:
        if v.severity != "warning":
            continue
        ref = v.refs[0] if v.refs else None
        warn_clusters.setdefault((v.type, ref), []).append(v)
    for (vtype, ref), vlist in warn_clusters.items():
        if len(vlist) == 1:
            v = vlist[0]
            issues.append(
                Issue(
                    severity="warning",
                    kind=f"drc.{vtype}",
                    message=v.description,
                    ref=ref,
                    location=v.pos,
                )
            )
        else:
            sample = vlist[0]
            issues.append(
                Issue(
                    severity="warning",
                    kind=f"drc.{vtype}",
                    message=(
                        f"{len(vlist)} {vtype} warnings clustered on "
                        f"{ref or 'multiple refs'} "
                        f"(e.g., {sample.description})"
                    ),
                    ref=ref,
                    location=sample.pos,
                )
            )

    # Wasted-area heuristic (info severity unless very high)
    if report.wasted_fraction > 0.45:
        sev = "warning" if report.wasted_fraction > 0.55 else "info"
        issues.append(
            Issue(
                severity=sev,
                kind="utilization.wasted_area",
                message=(
                    f"{report.wasted_fraction * 100:.1f}% of the board "
                    f"({report.wasted_area_mm2:.0f} mm^2) is empty. Stacking "
                    f"more SMT on top of back-side blocks could shrink the "
                    f"board substantially."
                ),
            )
        )

    # Low stacked-area heuristic (info)
    cells_back_only = report.cells_back_only
    if cells_back_only > 0 and report.cells_stacked < cells_back_only * 0.4:
        issues.append(
            Issue(
                severity="info",
                kind="utilization.poor_stacking",
                message=(
                    f"Only {report.cells_stacked} of "
                    f"{cells_back_only + report.cells_stacked} back-side cells "
                    f"have a front-side stack on them. The opposite-side "
                    f"front area is largely unused."
                ),
            )
        )

    if report.drc.unconnected:
        issues.append(
            Issue(
                severity="warning",
                kind="route.unconnected",
                message=f"{report.drc.unconnected} unconnected ratlines remain",
            )
        )

    issues.sort(key=lambda i: (_SEVERITY_ORDER.get(i.severity, 9), i.kind))
    return issues


def _find_zones(pcb_path: Path) -> dict[str, Any]:
    """Walk up from the PCB's directory looking for a project config.

    Stamped/routed PCBs land in `.experiments/subcircuits/<hash>/` -- the
    project config lives at the project root, several levels up. Cap the
    walk at 6 levels to stay bounded.
    """
    parent = pcb_path.parent
    for _ in range(6):
        cfg_file = discover_project_config(parent)
        if cfg_file is not None:
            try:
                cfg = load_project_config(str(cfg_file))
                return cfg.get("component_zones", {})
            except Exception:
                return {}
        if parent.parent == parent:
            break
        parent = parent.parent
    return {}


def collect(pcb_path: Path) -> Report:
    board = pcbnew.LoadBoard(str(pcb_path))
    if board is None:
        raise RuntimeError(f"Could not load {pcb_path}")
    outline = _board_outline_bbox(board)
    zones = _find_zones(pcb_path)

    footprints: list[FootprintInfo] = []
    front_courtyards: list[Bbox] = []
    back_courtyards: list[Bbox] = []
    edge_findings: list[EdgeFinding] = []

    for fp in board.GetFootprints():
        ref = fp.GetReferenceAsString()
        layer = "back" if fp.GetLayer() == pcbnew.B_Cu else "front"
        cb = _footprint_courtyard(fp)
        marker = _edge_marker(fp)
        info = FootprintInfo(
            ref=ref,
            layer=layer,
            pos=(_to_mm(fp.GetPosition().x), _to_mm(fp.GetPosition().y)),
            courtyard=cb,
            edge_marker=marker,
        )
        footprints.append(info)
        (front_courtyards if layer == "front" else back_courtyards).append(cb)

        zone_cfg = zones.get(ref)
        if not zone_cfg:
            continue
        edge = zone_cfg.get("edge")
        if not edge or marker is None:
            continue
        if edge == "left":
            distance = marker[0] - outline.min_x
            outboard = cb.min_x < outline.min_x - 0.05
        elif edge == "right":
            distance = outline.max_x - marker[0]
            outboard = cb.max_x > outline.max_x + 0.05
        elif edge == "top":
            distance = marker[1] - outline.min_y
            outboard = cb.min_y < outline.min_y - 0.05
        else:
            distance = outline.max_y - marker[1]
            outboard = cb.max_y > outline.max_y + 0.05

        if abs(distance) < 0.5 and outboard:
            interpretation = "OK: marker at edge, body overhangs"
        elif abs(distance) < 0.5:
            interpretation = (
                "OK-ish: marker at edge, body fully inside (no overhang needed)"
            )
        elif abs(distance) >= 0.5 and not outboard:
            interpretation = (
                f"BUG: marker is {distance:.2f} mm inside the board edge "
                f"(connector body fully inside; will not be usable if it's "
                f"meant to physically overhang)"
            )
        else:
            interpretation = (
                f"WARN: marker at {distance:.2f} mm from edge but courtyard overhangs"
            )
        edge_findings.append(
            EdgeFinding(
                ref=ref,
                edge=edge,
                marker_world=marker,
                marker_distance_from_edge_mm=distance,
                courtyard_overhangs=outboard,
                interpretation=interpretation,
            )
        )

    # Wasted-area / stacking grid analysis.
    nx = max(1, int(round(outline.width / GRID_MM)))
    ny = max(1, int(round(outline.height / GRID_MM)))
    cells_front_only = 0
    cells_back_only = 0
    cells_stacked = 0
    cells_empty = 0
    for ix in range(nx):
        for iy in range(ny):
            cx = outline.min_x + (ix + 0.5) * GRID_MM
            cy = outline.min_y + (iy + 0.5) * GRID_MM
            cell = Bbox(
                cx - GRID_MM / 2,
                cy - GRID_MM / 2,
                cx + GRID_MM / 2,
                cy + GRID_MM / 2,
            )
            on_front = any(cell.overlaps(c) for c in front_courtyards)
            on_back = any(cell.overlaps(c) for c in back_courtyards)
            if on_front and on_back:
                cells_stacked += 1
            elif on_front:
                cells_front_only += 1
            elif on_back:
                cells_back_only += 1
            else:
                cells_empty += 1

    drc = _run_drc(pcb_path)
    report = Report(
        pcb_path=str(pcb_path),
        board_outline=outline,
        footprints=footprints,
        edge_findings=edge_findings,
        drc=drc,
        grid_mm=GRID_MM,
        cells_total=nx * ny,
        cells_empty=cells_empty,
        cells_front_only=cells_front_only,
        cells_back_only=cells_back_only,
        cells_stacked=cells_stacked,
    )
    report.issues = _build_issues(report)
    return report


# --- Markdown summary -----------------------------------------------------


def to_markdown(report: Report, *, png_paths: dict[str, Path] | None = None) -> str:
    """Render an AI-agent-friendly markdown summary.

    The summary leads with the most actionable signals (edge bugs,
    wasted-area %) and pushes the long footprint table to the end.
    """
    lines: list[str] = []
    bo = report.board_outline
    lines.append(f"# Parent PCB inspection: `{Path(report.pcb_path).name}`")
    lines.append("")

    # Top-line health: the first thing an AI agent should see.
    error_count = sum(1 for i in report.issues if i.severity == "error")
    warn_count = sum(1 for i in report.issues if i.severity == "warning")
    info_count = sum(1 for i in report.issues if i.severity == "info")
    if error_count:
        verdict = f"BROKEN ({error_count} error{'s' if error_count != 1 else ''})"
    elif warn_count:
        verdict = f"WORKS WITH WARNINGS ({warn_count})"
    elif info_count:
        verdict = "WORKS (with utilization notes)"
    else:
        verdict = "OK"
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")

    lines.append(f"- **Board** : ({bo.min_x:.2f}, {bo.min_y:.2f}) to "
                 f"({bo.max_x:.2f}, {bo.max_y:.2f}) "
                 f"= **{bo.width:.2f} x {bo.height:.2f} mm** "
                 f"({report.board_area_mm2:.0f} mm^2)")
    lines.append(f"- **Wasted area**     : "
                 f"{report.wasted_area_mm2:.0f} mm^2 "
                 f"({report.wasted_fraction * 100:.1f}% of board)")
    lines.append(f"- **Dual-layer stacked** : "
                 f"{report.stacked_area_mm2:.0f} mm^2 "
                 f"({report.stacked_fraction * 100:.1f}% of board)")
    lines.append(f"- **Stacking efficiency** : "
                 f"{report.stacking_efficiency * 100:.1f}% "
                 f"(fraction of back-side area with a front-side stack)")
    lines.append(f"- **Packing density**     : "
                 f"{report.packing_density * 100:.1f}% "
                 f"(fraction of board occupied by any footprint)")
    lines.append(f"- **Footprints**     : "
                 f"{sum(1 for f in report.footprints if f.layer == 'front')} front, "
                 f"{sum(1 for f in report.footprints if f.layer == 'back')} back")
    drc = report.drc
    if drc.ran:
        type_summary = ", ".join(f"{k}={v}" for k, v in sorted(drc.by_type.items()))
        lines.append(
            f"- **DRC**            : {drc.error_count} errors, {drc.warning_count} warnings, "
            f"{drc.unconnected} unconnected"
            + (f"  ({type_summary})" if type_summary else "")
        )
    else:
        lines.append(f"- **DRC**            : not run ({drc.note})")
    lines.append("")

    # Structured issues list -- this is the part AI agents care about most.
    if report.issues:
        lines.append("## Issues (sorted by severity)")
        lines.append("")
        for i in report.issues:
            tag = {"error": "ERR", "warning": "WARN", "info": "INFO"}.get(i.severity, i.severity.upper())
            ref_part = f" `{i.ref}`" if i.ref else ""
            loc_part = (
                f" @ ({i.location[0]:.2f}, {i.location[1]:.2f})"
                if i.location
                else ""
            )
            lines.append(f"- **{tag}** [{i.kind}]{ref_part}{loc_part}: {i.message}")
        lines.append("")
    else:
        lines.append("## Issues")
        lines.append("")
        lines.append("(none)")
        lines.append("")

    # Edge findings: lead with anything flagged BUG.
    if report.edge_findings:
        lines.append("## Edge-attachment findings")
        bugs = [f for f in report.edge_findings if f.interpretation.startswith("BUG")]
        warns = [f for f in report.edge_findings if f.interpretation.startswith("WARN")]
        oks = [f for f in report.edge_findings if f.interpretation.startswith("OK")]
        for group, label in ((bugs, "BUG"), (warns, "WARN"), (oks, "OK")):
            for f in group:
                lines.append(
                    f"- **{label}** `{f.ref}` (edge={f.edge}): "
                    f"marker @ ({f.marker_world[0]:.2f}, {f.marker_world[1]:.2f}), "
                    f"distance from board edge = {f.marker_distance_from_edge_mm:.2f} mm, "
                    f"courtyard_overhangs={f.courtyard_overhangs}. "
                    f"{f.interpretation}"
                )
        lines.append("")

    # Visual artifacts -- with brief descriptions so AI agents know
    # which one to load for which question.
    PNG_DESCRIPTIONS = {
        "annotated_top": (
            "top-view with leaf courtyards (front=green, back=red), "
            "edge-marker arrows, and DRC violation markers (red=err, "
            "yellow=warn). Read this to confirm marker alignment, "
            "stacking layout, and DRC location clusters."
        ),
        "stacking_heatmap": (
            "5 mm grid colored by occupancy: yellow=stacked (front+back), "
            "green=front-only, red=back-only, black=empty. Read this to "
            "see at a glance how much board area is unused and where "
            "back-only opportunity zones for more stacking exist."
        ),
    }
    if png_paths:
        lines.append("## Visual artifacts")
        lines.append("")
        for label, path in png_paths.items():
            desc = PNG_DESCRIPTIONS.get(label, "")
            lines.append(f"- **{label}** ({desc})")
            lines.append(f"  - path: `{path}`")
        lines.append("")

    # Stacking analysis.
    lines.append("## Area utilization (5 mm grid)")
    lines.append("")
    lines.append("| Region              |       Area | Fraction |")
    lines.append("| ------------------- | ---------- | -------- |")
    cell_area = report.grid_mm * report.grid_mm
    for label, cells in (
        ("stacked (front+back)", report.cells_stacked),
        ("front-only",           report.cells_front_only),
        ("back-only",            report.cells_back_only),
        ("empty (wasted)",       report.cells_empty),
    ):
        area = cells * cell_area
        frac = (area / report.board_area_mm2) * 100 if report.board_area_mm2 > 0 else 0
        lines.append(f"| {label:<19s} | {area:>7.0f} mm^2 | {frac:>6.1f}% |")
    lines.append("")

    # Next-actions hints derived from issues. These are heuristics, not
    # guarantees -- but they save the AI agent from reasoning through
    # the issue list manually for the common cases.
    next_actions = _suggest_next_actions(report)
    if next_actions:
        lines.append("## Suggested next actions")
        lines.append("")
        for action in next_actions:
            lines.append(f"- {action}")
        lines.append("")

    # Footprint table -- only the largest/refs-with-issues to keep the
    # report skimmable. Full list is always in report.json.
    interesting_refs: set[str] = set()
    for f in report.edge_findings:
        interesting_refs.add(f.ref)
    for v in report.drc.violations:
        for r in v.refs:
            interesting_refs.add(r)
    sorted_by_area = sorted(
        report.footprints,
        key=lambda f: f.courtyard.area,
        reverse=True,
    )
    pick: list[FootprintInfo] = []
    seen: set[str] = set()
    for fp in sorted_by_area[:8]:
        pick.append(fp)
        seen.add(fp.ref)
    for fp in sorted(report.footprints, key=lambda f: f.ref):
        if fp.ref in interesting_refs and fp.ref not in seen:
            pick.append(fp)
            seen.add(fp.ref)

    if pick:
        lines.append("## Notable footprints")
        lines.append("")
        lines.append(f"(largest 8 + refs with edge/DRC findings; full list in report.json)")
        lines.append("")
        lines.append("| Ref     | Layer | Position (mm)        | Courtyard (mm)                                |")
        lines.append("| ------- | ----- | -------------------- | --------------------------------------------- |")
        for fp in sorted(pick, key=lambda f: f.ref):
            c = fp.courtyard
            lines.append(
                f"| `{fp.ref:<5}` | {fp.layer:<5} "
                f"| ({fp.pos[0]:>6.2f}, {fp.pos[1]:>6.2f}) "
                f"| ({c.min_x:>6.2f}, {c.min_y:>6.2f}) -> "
                f"({c.max_x:>6.2f}, {c.max_y:>6.2f}) |"
            )
        lines.append("")

    return "\n".join(lines)


def _suggest_next_actions(report: Report) -> list[str]:
    """Translate aggregated findings into concrete suggestions."""
    actions: list[str] = []
    drc_errors_by_type: dict[str, int] = {}
    for issue in report.issues:
        if issue.severity != "error" or not issue.kind.startswith("drc."):
            continue
        kind = issue.kind.split(".", 1)[1]
        drc_errors_by_type[kind] = drc_errors_by_type.get(kind, 0) + 1
    if drc_errors_by_type.get("clearance"):
        actions.append(
            "Investigate clearance violations: probably a leaf-internal "
            "issue (the parent compose only places leaves, doesn't move "
            "their internal pads). Check the smallest leaf bbox + "
            "design rule clearance."
        )
    if drc_errors_by_type.get("items_not_allowed"):
        actions.append(
            "Items-not-allowed (keepout) errors usually mean a mounting "
            "hole or pad is inside a configured keepout zone. Either move "
            "the offending ref out of the keepout or relax the keepout."
        )
    if any(i.kind == "edge.marker_inside_board" for i in report.issues):
        actions.append(
            "Edge marker is inside the board: the connector body is sitting "
            "fully on the PCB instead of overhanging. Check that "
            "_compute_final_outline does NOT widen the constrained side "
            "to enclose the connector overhang, and that "
            "connector_edge_inset_mm in the project config is 0 (or "
            "negative for explicit overhang)."
        )
    if report.stacking_efficiency < 0.3 and report.cells_back_only > 5:
        actions.append(
            f"Low stacking efficiency ({report.stacking_efficiency * 100:.0f}%): "
            f"{report.cells_back_only * report.grid_mm * report.grid_mm:.0f} mm^2 "
            f"of back-only board area has no front-side block on top. "
            f"Consider strengthening the opposite-side stacking pass / "
            f"attraction (cfg.opposite_side_attraction_k) or seeding more "
            f"SMT blocks inside large back-side block bboxes during "
            f"_place_clusters."
        )
    if report.wasted_fraction > 0.5 and report.packing_density < 0.4:
        actions.append(
            f"Board is mostly empty ({report.wasted_fraction * 100:.0f}% wasted, "
            f"only {report.packing_density * 100:.0f}% packed). Corner-pinned "
            f"mounting holes anchor the outline to the seed corners; consider "
            f"re-snapping H4/H86-style refs to the actual leaf-geometry "
            f"corners after the stacking pass to shrink the board."
        )
    if report.drc.unconnected:
        actions.append(
            f"{report.drc.unconnected} unconnected ratlines: re-run the "
            f"router with more passes, widen the route channel, or check "
            f"that the parent net inference picked up all interconnect nets."
        )
    return actions


# --- Annotated rendering --------------------------------------------------


def _world_to_image(
    point: tuple[float, float],
    outline: Bbox,
    img_size: tuple[int, int],
    padding_px: int,
) -> tuple[int, int]:
    """World mm -> image pixel, with `padding_px` around the board."""
    width_px, height_px = img_size
    inner_w = max(1, width_px - 2 * padding_px)
    inner_h = max(1, height_px - 2 * padding_px)
    sx = inner_w / max(0.001, outline.width)
    sy = inner_h / max(0.001, outline.height)
    s = min(sx, sy)
    px = padding_px + (point[0] - outline.min_x) * s
    py = padding_px + (point[1] - outline.min_y) * s
    return int(px), int(py)


def _world_scale(outline: Bbox, img_size: tuple[int, int], padding_px: int) -> float:
    width_px, height_px = img_size
    inner_w = max(1, width_px - 2 * padding_px)
    inner_h = max(1, height_px - 2 * padding_px)
    sx = inner_w / max(0.001, outline.width)
    sy = inner_h / max(0.001, outline.height)
    return min(sx, sy)


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_annotated_top(report: Report, output: Path) -> Path:
    """Render an annotated top-view PNG.

    Drawn from scratch (does not require pcbnew render). Shows:
    - board outline (cyan)
    - front-side courtyards (green)
    - back-side courtyards (red, dashed via cross-hatch)
    - edge-marker arrows pointing from marker to expected board edge
    - constraint anchor labels
    - footprint refs

    The colors and labels are chosen for AI-agent legibility, not
    pretty-print. An agent reading this PNG should be able to see
    instantly: "is the connector marker at the board edge?", "how
    much front-side area is unused?".
    """
    bo = report.board_outline
    pad = 60
    aspect = bo.height / max(0.01, bo.width)
    width = 1200
    height = int(width * aspect) + 2 * pad
    width += 2 * pad
    img = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(img, "RGBA")
    font_label = _load_font(16)
    font_small = _load_font(11)
    font_title = _load_font(20)
    scale = _world_scale(bo, (width, height), pad)

    # Title.
    draw.text(
        (pad, 10),
        f"{Path(report.pcb_path).name}  board {bo.width:.1f} x {bo.height:.1f} mm",
        fill="#e5e7eb",
        font=font_title,
    )

    # Board outline.
    tl = _world_to_image((bo.min_x, bo.min_y), bo, (width, height), pad)
    br = _world_to_image((bo.max_x, bo.max_y), bo, (width, height), pad)
    draw.rectangle([tl, br], outline="#22d3ee", width=3)

    # Back courtyards first (so front overlays them).
    for fp in report.footprints:
        if fp.layer != "back":
            continue
        c = fp.courtyard
        a = _world_to_image((c.min_x, c.min_y), bo, (width, height), pad)
        b = _world_to_image((c.max_x, c.max_y), bo, (width, height), pad)
        draw.rectangle([a, b], outline="#f87171", width=2, fill=(248, 113, 113, 60))
        # Label at top-left of bbox.
        draw.text((a[0] + 3, a[1] + 3), fp.ref, fill="#fecaca", font=font_small)

    # Front courtyards on top.
    for fp in report.footprints:
        if fp.layer != "front":
            continue
        c = fp.courtyard
        a = _world_to_image((c.min_x, c.min_y), bo, (width, height), pad)
        b = _world_to_image((c.max_x, c.max_y), bo, (width, height), pad)
        draw.rectangle([a, b], outline="#34d399", width=1)
        # Show ref only for largish footprints to avoid clutter.
        if c.area > 5:
            draw.text((a[0] + 3, a[1] + 3), fp.ref, fill="#a7f3d0", font=font_small)

    # Edge findings: draw arrow from marker to nearest board edge.
    for f in report.edge_findings:
        marker_world = f.marker_world
        if f.edge == "left":
            target_world = (bo.min_x, marker_world[1])
        elif f.edge == "right":
            target_world = (bo.max_x, marker_world[1])
        elif f.edge == "top":
            target_world = (marker_world[0], bo.min_y)
        else:
            target_world = (marker_world[0], bo.max_y)
        m = _world_to_image(marker_world, bo, (width, height), pad)
        t = _world_to_image(target_world, bo, (width, height), pad)
        color = (
            "#fb923c" if f.interpretation.startswith("BUG")
            else "#facc15" if f.interpretation.startswith("WARN")
            else "#22d3ee"
        )
        draw.line([m, t], fill=color, width=3)
        draw.ellipse([m[0] - 4, m[1] - 4, m[0] + 4, m[1] + 4], fill=color)
        label = f"{f.ref} {f.marker_distance_from_edge_mm:+.2f} mm"
        # Place label slightly offset toward the board interior.
        ox, oy = (8, -6)
        if f.edge == "right":
            ox = -8 - len(label) * 7
        if f.edge == "bottom":
            oy = 6
        draw.text((m[0] + ox, m[1] + oy), label, fill=color, font=font_label)

    # DRC violations -- only show errors prominently; cluster nearby
    # violations of the same (type, ref) so an AI agent doesn't get a
    # cloud of overlapping markers on a single cluster of pins.
    drc_pin_clusters: dict[tuple[str, str | None], tuple[float, float, int]] = {}
    for v in report.drc.violations:
        if v.pos is None:
            continue
        ref = v.refs[0] if v.refs else None
        key = (v.type, ref)
        if key in drc_pin_clusters:
            cx, cy, count = drc_pin_clusters[key]
            drc_pin_clusters[key] = (
                (cx * count + v.pos[0]) / (count + 1),
                (cy * count + v.pos[1]) / (count + 1),
                count + 1,
            )
        else:
            drc_pin_clusters[key] = (v.pos[0], v.pos[1], 1)

    # Draw errors with bigger emphasis than warnings.
    sev_by_key = {
        (v.type, v.refs[0] if v.refs else None): v.severity
        for v in report.drc.violations
    }
    for (vtype, ref), (cx, cy, count) in drc_pin_clusters.items():
        sev = sev_by_key.get((vtype, ref), "warning")
        color = "#f87171" if sev == "error" else "#facc15"
        radius = 9 if sev == "error" else 6
        p = _world_to_image((cx, cy), bo, (width, height), pad)
        draw.ellipse(
            [p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius],
            outline=color,
            width=2 if sev == "error" else 1,
        )
        draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)
        label = f"{vtype}" + (f" x{count}" if count > 1 else "")
        draw.text(
            (p[0] + radius + 3, p[1] - 7),
            label,
            fill=color,
            font=font_small,
        )

    # Legend.
    legend_y = height - pad + 8
    draw.text((pad, legend_y), "front", fill="#34d399", font=font_label)
    draw.text((pad + 70, legend_y), "back", fill="#f87171", font=font_label)
    draw.text(
        (pad + 140, legend_y),
        "marker->edge (cyan=ok, yellow=warn, orange=bug)  DRC: red=err  yellow=warn",
        fill="#e5e7eb",
        font=font_label,
    )

    img.save(output)
    return output


def render_stacking_heatmap(report: Report, output: Path) -> Path:
    """Render a 5 mm-grid heatmap of layer occupancy.

    Cells are color-coded:
      - empty  : black
      - front  : green
      - back   : red
      - stacked: yellow (front + back; the goal for dual-layer parents)
    """
    bo = report.board_outline
    pad = 60
    aspect = bo.height / max(0.01, bo.width)
    width = 1200
    height = int(width * aspect) + 2 * pad
    width += 2 * pad
    img = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(img, "RGBA")
    font_title = _load_font(20)
    font_legend = _load_font(14)

    front_courts = [fp.courtyard for fp in report.footprints if fp.layer == "front"]
    back_courts = [fp.courtyard for fp in report.footprints if fp.layer == "back"]
    nx = max(1, int(round(bo.width / report.grid_mm)))
    ny = max(1, int(round(bo.height / report.grid_mm)))
    for ix in range(nx):
        for iy in range(ny):
            cx = bo.min_x + (ix + 0.5) * report.grid_mm
            cy = bo.min_y + (iy + 0.5) * report.grid_mm
            cell = Bbox(
                cx - report.grid_mm / 2,
                cy - report.grid_mm / 2,
                cx + report.grid_mm / 2,
                cy + report.grid_mm / 2,
            )
            on_front = any(cell.overlaps(c) for c in front_courts)
            on_back = any(cell.overlaps(c) for c in back_courts)
            if on_front and on_back:
                color = (250, 204, 21, 200)  # stacked - yellow
            elif on_front:
                color = (52, 211, 153, 130)  # front - green
            elif on_back:
                color = (248, 113, 113, 130)  # back - red
            else:
                color = (24, 24, 27, 0)  # empty - leave board bg
            a = _world_to_image((cell.min_x, cell.min_y), bo, (width, height), pad)
            b = _world_to_image((cell.max_x, cell.max_y), bo, (width, height), pad)
            draw.rectangle([a, b], fill=color, outline=(255, 255, 255, 30))

    # Board outline.
    tl = _world_to_image((bo.min_x, bo.min_y), bo, (width, height), pad)
    br = _world_to_image((bo.max_x, bo.max_y), bo, (width, height), pad)
    draw.rectangle([tl, br], outline="#22d3ee", width=3)

    draw.text(
        (pad, 10),
        f"Stacking heatmap  stacked {report.stacked_fraction * 100:.1f}% / "
        f"wasted {report.wasted_fraction * 100:.1f}%",
        fill="#e5e7eb",
        font=font_title,
    )
    legend_y = height - pad + 8
    draw.rectangle([pad, legend_y, pad + 18, legend_y + 14], fill="#facc15")
    draw.text((pad + 24, legend_y - 1), "stacked", fill="#e5e7eb", font=font_legend)
    draw.rectangle([pad + 110, legend_y, pad + 128, legend_y + 14], fill="#34d399")
    draw.text((pad + 134, legend_y - 1), "front", fill="#e5e7eb", font=font_legend)
    draw.rectangle([pad + 200, legend_y, pad + 218, legend_y + 14], fill="#f87171")
    draw.text((pad + 224, legend_y - 1), "back", fill="#e5e7eb", font=font_legend)
    draw.rectangle([pad + 290, legend_y, pad + 308, legend_y + 14], fill="#18181b", outline="#fff")
    draw.text((pad + 314, legend_y - 1), "empty", fill="#e5e7eb", font=font_legend)

    img.save(output)
    return output


# --- CLI -----------------------------------------------------------------


def diff_reports(prev: dict, cur: dict) -> list[str]:
    """Diff two inspector report.json dicts and return human-readable
    deltas suitable for an AI agent's 'did this change help?' check."""
    out: list[str] = []

    def _f(path: list[str], default=0):
        d = cur
        for p in path:
            d = d.get(p, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) else default

    def _p(path: list[str], default=0):
        d = prev
        for p in path:
            d = d.get(p, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) else default

    metrics = [
        ("board_area_mm2", "board area mm^2", "lower"),
        ("wasted_fraction", "wasted fraction", "lower"),
        ("stacked_fraction", "stacked fraction", "higher"),
        ("stacking_efficiency", "stacking efficiency", "higher"),
        ("packing_density", "packing density", "higher"),
    ]
    for key, label, direction in metrics:
        p_val = _p([key])
        c_val = _f([key])
        if p_val == c_val:
            continue
        delta = c_val - p_val
        sign = "+" if delta >= 0 else ""
        verdict = ""
        if direction == "lower":
            verdict = " (BETTER)" if delta < 0 else " (WORSE)"
        elif direction == "higher":
            verdict = " (BETTER)" if delta > 0 else " (WORSE)"
        if isinstance(c_val, float) and abs(c_val) < 1.0:
            out.append(
                f"- {label}: {p_val * 100:.1f}% -> {c_val * 100:.1f}% "
                f"({sign}{delta * 100:.1f} pp){verdict}"
            )
        else:
            out.append(f"- {label}: {p_val:.1f} -> {c_val:.1f} ({sign}{delta:.1f}){verdict}")

    p_drc = prev.get("drc", {})
    c_drc = cur.get("drc", {})
    p_err = p_drc.get("error_count", 0)
    c_err = c_drc.get("error_count", 0)
    if p_err != c_err:
        verdict = " (BETTER)" if c_err < p_err else " (WORSE)"
        out.append(f"- DRC errors: {p_err} -> {c_err}{verdict}")

    p_kinds = {i["kind"] for i in prev.get("issues", [])}
    c_kinds = {i["kind"] for i in cur.get("issues", [])}
    new_kinds = c_kinds - p_kinds
    fixed_kinds = p_kinds - c_kinds
    if new_kinds:
        out.append(f"- New issue kinds: {', '.join(sorted(new_kinds))}")
    if fixed_kinds:
        out.append(f"- Fixed issue kinds: {', '.join(sorted(fixed_kinds))}")

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pcb", help="Stamped or routed parent .kicad_pcb")
    parser.add_argument(
        "--output-dir",
        help="Directory to write reports + PNGs (default: <pcb_dir>/inspect/)",
    )
    parser.add_argument("--json-only", action="store_true", help="Skip PNGs, only emit JSON")
    parser.add_argument(
        "--baseline",
        help="Path to a previous report.json to diff against; deltas are "
        "appended to summary.md and printed on stdout.",
    )
    args = parser.parse_args(argv)

    pcb_path = Path(args.pcb).resolve()
    if not pcb_path.is_file():
        print(f"error: {pcb_path} not found", file=sys.stderr)
        return 2
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else pcb_path.parent / "inspect"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = collect(pcb_path)

    json_path = out_dir / "report.json"
    # Atomic write: tmp + replace. inspect_parent runs after every parent
    # route in the autoexperiment loop; a downstream tool reading
    # report.json mid-write would otherwise see truncated JSON.
    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    tmp_json.replace(json_path)

    pngs: dict[str, Path] = {}
    if not args.json_only:
        try:
            pngs["annotated_top"] = render_annotated_top(
                report, out_dir / "annotated_top.png"
            )
            pngs["stacking_heatmap"] = render_stacking_heatmap(
                report, out_dir / "stacking_heatmap.png"
            )
        except Exception as exc:
            print(f"warning: render failed: {exc}", file=sys.stderr)

    md_text = to_markdown(report, png_paths=pngs)

    diff_lines: list[str] = []
    if args.baseline:
        try:
            prev = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            cur = report.to_dict()
            diff_lines = diff_reports(prev, cur)
        except Exception as exc:
            diff_lines = [f"(diff failed: {exc})"]
    if diff_lines:
        md_text += "\n## Diff vs baseline\n\n" + "\n".join(diff_lines) + "\n"

    md_path = out_dir / "summary.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"summary  : {md_path}")
    print(f"json     : {json_path}")
    for label, p in pngs.items():
        print(f"{label:<8} : {p}")
    if diff_lines:
        print("\n--- diff vs baseline ---")
        for line in diff_lines:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
