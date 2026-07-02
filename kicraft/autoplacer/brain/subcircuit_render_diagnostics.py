"""Leaf/subcircuit render diagnostics helpers.

This module builds a small, standardized visual diagnostic bundle for routed
leaf artifacts under a leaf artifact directory, typically:

    .experiments/subcircuits/<slug>/renders/

It is intentionally orchestration-focused and reuses the existing rendering
helpers in the scripts directory:

- `render_pcb.py` for board snapshots
- `render_drc_overlay.py` for DRC overlays

Primary outputs:
- pre-route / routed board snapshots
- pre-route / routed DRC JSON sidecars
- pre-route / routed DRC overlays
- a simple contact sheet comparing pre-route vs routed artifacts

The helpers are designed to degrade gracefully when optional external tools
(e.g. `kicad-cli`, ImageMagick) are unavailable.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from kicraft.cli.render_drc_overlay import render_overlay
except Exception:  # pragma: no cover - best-effort import
    render_overlay = None

try:
    from kicraft.render import render_views as _render_views
except Exception:  # pragma: no cover - best-effort import
    _render_views = None


DEFAULT_VIEWS = ("front_all", "back_all", "copper_both")


@dataclass(frozen=True)
class LeafStageOpts:
    """Per-stage diagnostic toggles. Replaces the four boolean kwargs
    (``render_X_board_views``, ``write_X_drc_json``, ``write_X_drc_report``,
    ``render_X_drc_overlay``) that the legacy 14-param API repeated for
    each stage. Defaults render everything; ``LeafStageOpts.off()`` skips
    every artifact for that stage."""

    render_board_views: bool = True
    write_drc_json: bool = True
    write_drc_report: bool = True
    render_drc_overlay: bool = True

    @classmethod
    def off(cls) -> "LeafStageOpts":
        return cls(False, False, False, False)


_NOISY_STDERR_PATTERNS = (
    "Adding duplicate image handler",
    "swig/python detected a memory leak",
)


@contextlib.contextmanager
def _suppress_noisy_stderr() -> Any:
    """Suppress known non-actionable KiCad/wx stderr noise during rendering."""
    original_stderr = sys.stderr
    buffer = io.StringIO()
    try:
        sys.stderr = buffer
        yield
    finally:
        sys.stderr = original_stderr
        for line in buffer.getvalue().splitlines():
            if not any(pattern in line for pattern in _NOISY_STDERR_PATTERNS):
                print(line, file=original_stderr)


def ensure_renders_dir(artifact_dir: str | Path) -> Path:
    """Create and return the standard renders directory for one leaf artifact."""
    renders_dir = Path(artifact_dir) / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    return renders_dir


def promote_to_round_snapshot(
    canonical: str | Path | None, round_index: int | None
) -> Path | None:
    """Make ``round_NNNN_<canonical_basename>`` next to ``canonical``
    as a byte-level copy. Returns the snapshot path, or ``None`` when
    there is nothing to promote.

    Earlier this function hardlinked the snapshot to canonical for
    disk savings + byte-equivalence-by-construction. That works for
    PNGs (``render_pcb`` writes via tmpfile + os.replace so canonical
    gets a fresh inode on every render, leaving prior hardlinks
    pointing at the now-frozen old inode). It does NOT work for
    ``.kicad_pcb`` files: ``pcbnew.Save()`` overwrites the canonical
    inode in place each round, so a hardlinked ``round_NNNN_*.kicad_pcb``
    would silently see whatever the LATEST round of routing wrote, not
    the bytes that were on disk when THIS round produced them. Auto-pin
    + manual layout then read back the wrong geometry for every
    non-latest round. Copy is the only safe primitive for files we do
    not control the writer of.
    """
    if canonical is None or round_index is None:
        return None
    src = Path(canonical)
    if not src.is_file():
        return None
    dst = src.parent / f"round_{int(round_index):04d}_{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    shutil.copy2(src, dst)
    return dst


# Suffixes (after the ``round_NNNN_`` prefix) of artifacts that MUST be
# kept for every round -- the manual-pin path (pins.py) reads the
# .kicad_pcb files and the pin picker / inspect tooling reads the small
# metadata JSON. These are never trimmed.
_KEEP_ROUND_SUFFIXES = frozenset({
    "leaf_routed.kicad_pcb",
    "leaf_pre_freerouting.kicad_pcb",
    "solved_layout.json",
    "debug.json",
    "metadata.json",
})

# Diagnostic file suffixes safe to trim from losing rounds.
_TRIM_DIAGNOSTIC_SUFFIXES = (".png", "_drc.json", "_drc_report.txt")


def trim_losing_round_diagnostics(
    artifact_dir: str | Path,
    winner_round_index: int | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Delete heavy diagnostic artifacts from non-winning leaf rounds.

    After the winner is chosen, losing rounds' render PNGs and DRC
    JSON/report files are pure overhead in headless builds. This trims
    them while keeping every round's ``.kicad_pcb`` (the manual-pin path
    in ``pins.py`` reads losing rounds' boards) and small metadata JSON
    (``solved_layout`` / ``debug`` / ``metadata``).

    Gated on ``cfg['keep_all_round_artifacts']`` (default False) -- when
    True, no files are deleted (restores today's keep-everything behavior).

    Returns a summary dict with counts of trimmed/kept files for logging.
    """
    cfg = cfg or {}
    if bool(cfg.get("keep_all_round_artifacts", False)):
        return {"trimmed": 0, "kept": 0, "skipped": "keep_all_round_artifacts"}

    base = Path(artifact_dir)
    renders_dir = base / "renders"
    search_dirs = [d for d in (base, renders_dir) if d.is_dir()]

    winner_key: str | None = None
    if winner_round_index is not None:
        winner_key = f"round_{int(winner_round_index):04d}"

    trimmed = 0
    kept = 0
    for search_dir in search_dirs:
        for entry in sorted(search_dir.iterdir()):
            name = entry.name
            if not name.startswith("round_") or not entry.is_file():
                continue
            # Parse ``round_NNNN_<suffix>``.
            parts = name.split("_", 2)
            if len(parts) < 3:
                continue
            round_key = f"{parts[0]}_{parts[1]}"
            suffix = parts[2]
            # Keep the winner's full diagnostics.
            if winner_key is not None and round_key == winner_key:
                kept += 1
                continue
            # Keep small metadata / board files for every round.
            if suffix in _KEEP_ROUND_SUFFIXES:
                kept += 1
                continue
            # Trim diagnostic artifacts (PNGs, DRC JSON, DRC reports).
            if suffix.endswith(_TRIM_DIAGNOSTIC_SUFFIXES):
                try:
                    entry.unlink()
                    trimmed += 1
                except OSError:
                    kept += 1
                continue
            # Unknown round_NNNN_ file -- keep it (safe default).
            kept += 1

    return {"trimmed": trimmed, "kept": kept}


def write_leaf_drc_json(drc_dict: dict[str, Any], output_path: str | Path) -> str:
    """Persist a DRC payload as pretty JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drc_dict, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)


def render_leaf_board_views(
    pcb_path: str | Path,
    output_dir: str | Path,
    prefix: str,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Render a small set of board snapshots for one PCB.

    Returns a dict with:
    - `requested_views`
    - `rendered_views`
    - `paths`
    - `errors`
    """
    result: dict[str, Any] = {
        "pcb_path": str(pcb_path),
        "requested_views": list(views),
        "rendered_views": [],
        "paths": {},
        "errors": [],
    }

    if _render_views is None:
        result["errors"].append("render_pcb_import_failed")
        return result

    pcb = Path(pcb_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pcb.exists():
        result["errors"].append("pcb_missing")
        return result

    try:
        with (
            _suppress_noisy_stderr(),
            contextlib.redirect_stdout(io.StringIO() if quiet else sys.stdout),
        ):
            rendered = _render_views(
                pcb,
                out_dir,
                views=list(views),
                name_template=f"{prefix}_{{view}}.png",
            )
        for view_name, path in rendered.items():
            result["rendered_views"].append(view_name)
            result["paths"][view_name] = str(path)
    except Exception as exc:  # pragma: no cover - external tool path
        result["errors"].append(f"render_failed:{exc}")

    return result


def render_leaf_drc_overlay(
    pcb_path: str | Path,
    drc_dict: dict[str, Any],
    output_png: str | Path,
) -> dict[str, Any]:
    """Render a DRC overlay image when coordinate-bearing violations exist."""
    result: dict[str, Any] = {
        "pcb_path": str(pcb_path),
        "output_png": str(output_png),
        "rendered": False,
        "violation_count": 0,
        "located_violation_count": 0,
        "errors": [],
    }

    if render_overlay is None:
        result["errors"].append("render_drc_overlay_import_failed")
        return result

    violations = list(drc_dict.get("violations", []) or [])
    result["violation_count"] = len(violations)
    located = [
        item
        for item in violations
        if isinstance(item, dict)
        and item.get("x_mm") is not None
        and item.get("y_mm") is not None
    ]
    result["located_violation_count"] = len(located)

    if not located:
        result["errors"].append("no_located_violations")
        return result

    try:
        ok = render_overlay(
            str(pcb_path),
            located,
            str(output_png),
        )
        result["rendered"] = bool(ok)
        if not ok:
            result["errors"].append("overlay_render_failed")
    except Exception as exc:  # pragma: no cover - external tool path
        result["errors"].append(f"overlay_exception:{exc}")

    return result


def build_leaf_contact_sheet(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    tile: str = "2x2",
    background: str = "white",
) -> dict[str, Any]:
    """Build a simple contact sheet from existing PNGs using ImageMagick."""
    result: dict[str, Any] = {
        "output_path": str(output_path),
        "input_paths": [str(p) for p in image_paths],
        "created": False,
        "errors": [],
    }

    existing = [str(Path(p)) for p in image_paths if Path(p).exists()]
    if not existing:
        result["errors"].append("no_input_images")
        return result

    magick = shutil.which("magick")
    if magick is None:
        result["errors"].append("imagemagick_unavailable")
        return result

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        magick,
        "montage",
        *existing,
        "-background",
        background,
        "-tile",
        tile,
        "-geometry",
        "+8+8",
        str(out),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            result["errors"].append(
                f"montage_failed:{stderr or f'rc={completed.returncode}'}"
            )
            return result
        result["created"] = out.exists()
        if not result["created"]:
            result["errors"].append("montage_missing_output")
    except Exception as exc:  # pragma: no cover - external tool path
        result["errors"].append(f"montage_exception:{exc}")

    return result


def _stage_prefix(stage: str) -> str:
    normalized = stage.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"pre", "pre_route", "pre_freerouting"}:
        return "pre_route"
    if normalized in {"routed", "post_route", "post"}:
        return "routed"
    return normalized or "stage"


def generate_stage_diagnostic_artifacts(
    *,
    pcb_path: str | Path,
    validation: dict[str, Any] | None,
    artifact_dir: str | Path,
    stage: str,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    render_board_views: bool = True,
    write_drc_json: bool = True,
    write_drc_report: bool = True,
    render_drc_overlay: bool = True,
    quiet_board_render: bool = False,
) -> dict[str, Any]:
    """Generate render diagnostics for one board stage.

    Typical stages:
    - `pre_route`
    - `routed`
    """
    prefix = _stage_prefix(stage)
    renders_dir = ensure_renders_dir(artifact_dir)
    stage_result: dict[str, Any] = {
        "stage": prefix,
        "renders_dir": str(renders_dir),
        "pcb_path": str(pcb_path),
        "board_views": {},
        "drc_json_path": None,
        "drc_overlay": {},
        "errors": [],
    }

    pcb = Path(pcb_path)
    if not pcb.exists():
        stage_result["errors"].append("pcb_missing")
        return stage_result

    if render_board_views:
        stage_result["board_views"] = render_leaf_board_views(
            pcb_path=pcb,
            output_dir=renders_dir,
            prefix=prefix,
            views=views,
            quiet=quiet_board_render,
        )
    else:
        stage_result["board_views"] = {
            "pcb_path": str(pcb),
            "requested_views": list(views),
            "rendered_views": [],
            "paths": {},
            "errors": ["board_views_skipped"],
        }

    drc = dict((validation or {}).get("drc", {}) or {})
    drc_json_path = renders_dir / f"{prefix}_drc.json"
    if write_drc_json:
        try:
            stage_result["drc_json_path"] = write_leaf_drc_json(drc, drc_json_path)
        except Exception as exc:
            stage_result["errors"].append(f"drc_json_write_failed:{exc}")
    else:
        stage_result["errors"].append("drc_json_skipped")

    report_text_path = renders_dir / f"{prefix}_drc_report.txt"
    if write_drc_report:
        try:
            report_text = str(drc.get("report_text", "") or "")
            tmp_report = report_text_path.with_suffix(report_text_path.suffix + ".tmp")
            tmp_report.write_text(report_text, encoding="utf-8")
            tmp_report.replace(report_text_path)
            stage_result["drc_report_text_path"] = str(report_text_path)
        except Exception as exc:
            stage_result["drc_report_text_path"] = None
            stage_result["errors"].append(f"drc_report_write_failed:{exc}")
    else:
        stage_result["drc_report_text_path"] = None
        stage_result["errors"].append("drc_report_skipped")

    overlay_path = renders_dir / f"{prefix}_drc_overlay.png"
    if render_drc_overlay:
        stage_result["drc_overlay"] = render_leaf_drc_overlay(
            pcb_path=pcb,
            drc_dict=drc,
            output_png=overlay_path,
        )
    else:
        stage_result["drc_overlay"] = {
            "pcb_path": str(pcb),
            "output_png": str(overlay_path),
            "rendered": False,
            "violation_count": len(list(drc.get("violations", []) or [])),
            "located_violation_count": 0,
            "errors": ["drc_overlay_skipped"],
        }

    return stage_result


def generate_leaf_diagnostic_artifacts(
    *,
    artifact_dir: str | Path,
    pre_route_board: str | Path | None = None,
    routed_board: str | Path | None = None,
    pre_route_validation: dict[str, Any] | None = None,
    routed_validation: dict[str, Any] | None = None,
    pre_route_opts: LeafStageOpts = LeafStageOpts(),
    routed_opts: LeafStageOpts = LeafStageOpts(),
    views: tuple[str, ...] = DEFAULT_VIEWS,
    build_contact_sheet: bool = True,
    quiet_render: bool = False,
) -> dict[str, Any]:
    """Generate the full leaf diagnostic bundle.

    Returns a JSON-serializable dict describing all generated artifacts.
    Pass ``LeafStageOpts.off()`` for ``pre_route_opts`` or ``routed_opts``
    to skip a stage entirely (used by the pre-route-only first pass
    inside ``route_local_subcircuit``).
    """
    renders_dir = ensure_renders_dir(artifact_dir)
    result: dict[str, Any] = {
        "artifact_dir": str(Path(artifact_dir)),
        "renders_dir": str(renders_dir),
        "pre_route": None,
        "routed": None,
        "comparison": {
            "contact_sheet_path": None,
            "created": False,
            "errors": [],
        },
    }

    for stage_key, board, validation, opts in (
        ("pre_route", pre_route_board, pre_route_validation, pre_route_opts),
        ("routed", routed_board, routed_validation, routed_opts),
    ):
        if not board:
            continue
        result[stage_key] = generate_stage_diagnostic_artifacts(
            pcb_path=board,
            validation=validation,
            artifact_dir=artifact_dir,
            stage=stage_key,
            views=views,
            render_board_views=opts.render_board_views,
            write_drc_json=opts.write_drc_json,
            write_drc_report=opts.write_drc_report,
            render_drc_overlay=opts.render_drc_overlay,
            quiet_board_render=quiet_render,
        )

    if build_contact_sheet:
        contact_inputs: list[str] = []
        for stage_key in ("pre_route", "routed"):
            stage_payload = result.get(stage_key) or {}
            board_views = stage_payload.get("board_views", {})
            paths = board_views.get("paths", {})
            overlay = stage_payload.get("drc_overlay", {})
            if paths.get("front_all"):
                contact_inputs.append(paths["front_all"])
            if paths.get("back_all"):
                contact_inputs.append(paths["back_all"])
            if paths.get("copper_both"):
                contact_inputs.append(paths["copper_both"])
            if overlay.get("rendered") and overlay.get("output_png"):
                contact_inputs.append(overlay["output_png"])

        contact_sheet_path = renders_dir / "pre_vs_routed_contact_sheet.png"
        comparison = build_leaf_contact_sheet(contact_inputs, contact_sheet_path)
        result["comparison"]["contact_sheet_path"] = str(contact_sheet_path)
        result["comparison"]["created"] = bool(comparison.get("created", False))
        result["comparison"]["errors"] = list(comparison.get("errors", []))
    else:
        result["comparison"]["errors"] = ["contact_sheet_skipped"]

    summary_path = renders_dir / "diagnostics_summary.json"
    try:
        tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        tmp_summary.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_summary.replace(summary_path)
        result["summary_json_path"] = str(summary_path)
    except Exception as exc:
        result["summary_json_path"] = None
        result.setdefault("errors", []).append(f"summary_write_failed:{exc}")

    return result
