"""Library helpers shared by the skill-driven workflow.

Two libraries the BOM/architecture stages consult:

- Leaf library — pinned sheet-level subcircuits (this module's original
  responsibility; lifted from the old ``stages/architecture.py``).
- Parts library — atomic symbol+footprint bundles, four-tier search
  (project / home / vendored / extras).

Each library has a loader + an "available …" markdown formatter the
relevant stage pastes into its sub-agent prompt.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .models import Architecture

logger = logging.getLogger(__name__)


class ArchitectureLibraryError(ValueError):
    """Raised when a library-backed sheet selection fails validation."""


def _load_library_leaves() -> list:
    """Best-effort load of the leaf library; logs but never raises."""
    try:
        from kicraft.leaf_library import LeafLibrary
        loaded, broken = LeafLibrary.from_env().load_all()
        if broken:
            logger.warning(
                "skipping %d broken library leaves: %s",
                len(broken),
                [b.dir.name for b in broken],
            )
        return loaded
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load leaf library: %s", exc)
        return []


def _format_available_leaves_block(loaded_leaves: list) -> str | None:
    """Render the "Available leaves" block the architecture stage sees.

    ``None`` if the library is empty -- callers should suppress the
    heading entirely in that case.
    """
    if not loaded_leaves:
        return None
    lines: list[str] = ["## Available leaves\n"]
    lines.append(
        "If one of these is a good match for a sheet you would otherwise "
        "design from scratch, set `Sheet.from_library = '<name>@<version>'` "
        "and `Sheet.library_instance = N` (1 for the first instance, 2 for "
        "the second, etc., if you reuse the same leaf multiple times). For "
        "each instance, choose a DISTINCT `Sheet.name` and `Sheet.stem` "
        "(e.g. `CHARGER` and `CHARGER_2`). The leaf's hierarchical-label "
        "interface MUST match the architecture's `inter_sheet_nets` for "
        "that sheet exactly (same names + directions). Use the leaf's "
        "label names verbatim. If no leaf is a good match, design the "
        "sheet from scratch.\n"
    )
    for leaf in loaded_leaves:
        m = leaf.manifest
        lines.append(f"### {m.name}@{m.version}\n")
        lines.append(f"**What it does**: {m.description}\n")
        lines.append("**Interface (hierarchical labels)**:")
        for lbl in m.interface.hierarchical_labels:
            lines.append(f"- {lbl.name} ({lbl.direction})")
        lines.append("")
        if m.watch_out_for:
            lines.append(f"**Watch out for**: {m.watch_out_for}\n")
    return "\n".join(lines)


def _validate_library_picks(
    architecture: Architecture, loaded_leaves: list
) -> None:
    """Check every library-backed sheet in the proposed architecture.

    Raises ``ArchitectureLibraryError`` on the first failure so the
    caller can re-prompt with the specific diagnostic.
    """
    by_slug = {f"{l.manifest.name}@{l.manifest.version}": l for l in loaded_leaves}

    instances_by_slug: dict[str, list[tuple[int, str]]] = {}
    for sheet in architecture.sheets:
        if sheet.from_library is None:
            continue
        instances_by_slug.setdefault(sheet.from_library, []).append(
            (sheet.library_instance or 0, sheet.name)
        )

    for slug, picks in instances_by_slug.items():
        if slug not in by_slug:
            raise ArchitectureLibraryError(
                f"Sheet references unknown library leaf {slug!r}. "
                f"Available: {sorted(by_slug)}"
            )
        nums = sorted(n for n, _ in picks)
        if nums != list(range(1, len(nums) + 1)):
            raise ArchitectureLibraryError(
                f"library_instance values for {slug} must be sequential "
                f"1..N with no gaps, got {nums}"
            )

    for sheet in architecture.sheets:
        if sheet.from_library is None:
            continue
        leaf = by_slug[sheet.from_library]
        leaf_iface = {
            (lbl.name, lbl.direction)
            for lbl in leaf.manifest.interface.hierarchical_labels
        }
        arch_iface = {
            (net.name, ep.direction)
            for net in architecture.inter_sheet_nets
            for ep in net.endpoints
            if ep.sheet == sheet.name
        }
        if leaf_iface != arch_iface:
            missing = leaf_iface - arch_iface
            extra = arch_iface - leaf_iface
            raise ArchitectureLibraryError(
                f"Library sheet {sheet.name!r} ({sheet.from_library}) "
                f"interface mismatch with inter_sheet_nets: "
                f"missing from architecture: {sorted(missing)}; "
                f"extra in architecture: {sorted(extra)}"
            )


# ---------- parts library ----------


def _load_library_parts(project_root: Path | None = None) -> tuple[list, list]:
    """Best-effort load of the parts library across all tiers.

    Returns ``(active, broken)`` — never raises. ``shadowed`` entries
    are discarded here; the BOM stage only needs the active set.
    """
    try:
        from kicraft.parts_library import load_all_with_overrides
        active, _shadowed, broken = load_all_with_overrides(project_root)
        if broken:
            logger.warning(
                "skipping %d broken parts: %s",
                len(broken),
                [f"{b.tier.value}:{b.dir.name}" for b in broken],
            )
        return active, broken
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load parts library: %s", exc)
        return [], []


def _format_available_parts_block(loaded_parts: list) -> str | None:
    """Render the "Available parts" block the BOM stage sees.

    ``None`` if the library is empty (the BOM stage falls back to
    KiCad stock libraries and ad-hoc questions). Each row carries
    enough context for the LLM to pick a part by tag, MPN, or
    sourcing vendor and to know which symbol/footprint id to use in
    the BOM (``<name>:<symbol_name>`` and ``<name>:<footprint_name>``).
    """
    if not loaded_parts:
        return None
    lines: list[str] = ["## Available parts\n"]
    lines.append(
        "These are pre-curated symbol+footprint bundles outside the stock "
        "KiCad libraries. Use them by reference (no substitutions) when "
        "they match a needed part. In the BOM, the `symbol` field is "
        "`<name>:<symbol_name>` and the `footprint` field is "
        "`<name>:<footprint_name>` — both verbatim from the rows below.\n"
    )
    lines.append(
        "| name | mpn | sourcing | tags | symbol | footprint | tier |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for part in loaded_parts:
        m = part.manifest
        sourcing = (
            ", ".join(f"{k}:{v}" for k, v in sorted(m.sourcing.items()))
            if m.sourcing
            else "—"
        )
        tags = ", ".join(m.tags) if m.tags else "—"
        lines.append(
            f"| `{m.name}` | {m.mpn} | {sourcing} | {tags} | "
            f"`{m.name}:{m.symbol_name}` | "
            f"`{m.name}:{m.footprint_name}` | {part.tier.value} |"
        )
    # Watch-out notes for parts that have them — easy to miss in the table.
    flagged = [p for p in loaded_parts if p.manifest.watch_out_for]
    if flagged:
        lines.append("\n### Watch out for")
        for part in flagged:
            lines.append(f"- **{part.manifest.name}**: {part.manifest.watch_out_for}")
    return "\n".join(lines)
