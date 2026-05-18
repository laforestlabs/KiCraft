"""Leaf-library helpers shared by the skill-driven workflow.

Lifted out of the old ``stages/architecture.py``. The Python pipeline no
longer drives the architecture LLM call; instead the Claude Code skill
asks the CLI to format an "Available leaves" block (so the model can see
what's reusable) and to validate library picks after the architecture
slot is written. Both responsibilities live here.
"""
from __future__ import annotations

import logging

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
