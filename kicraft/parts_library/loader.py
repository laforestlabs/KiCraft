"""Resolve and iterate the parts library across all four tiers.

Search order, highest precedence first:

1. ``<project_root>/.kicraft/parts/<name>/``  — project-local override
2. ``~/.kicraft/parts/<name>/``               — user-wide accumulator
3. ``<kicraft_install>/parts_library/<name>/`` — curated, vendored
4. ``$KICRAFT_EXTRA_PARTS_DIRS``              — escape hatch (colon-separated)

A part loaded from a higher tier wins; the loader returns the first
match by ``name`` and reports which tier supplied it. Callers that need
the full picture (e.g. ``list-parts``) ask for ``load_all_with_overrides``
to see both the active entry and shadowed-but-discoverable ones.

KiCad's stock libraries at ``/usr/share/kicad/{symbols,footprints}`` are
*not* parts in this sense — they're resolved separately by the
synthesis lookup helper as tier 5 (last resort).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .manifest import (
    PartManifest,
    compute_content_hash,
    load_manifest,
    manifest_path,
    required_files_present,
)

log = logging.getLogger(__name__)

PARTS_SUBDIR = "parts"  # within `.kicraft/` (project) or `~/.kicraft/` (home)
ENV_VAR = "KICRAFT_EXTRA_PARTS_DIRS"


class Tier(str, Enum):
    PROJECT = "project"
    HOME = "home"
    VENDORED = "vendored"
    EXTRA = "extra"


@dataclass(frozen=True, slots=True)
class LoadedPart:
    dir: Path
    tier: Tier
    manifest: PartManifest

    @property
    def slug(self) -> str:
        return f"{self.manifest.name}@{self.manifest.version}"


@dataclass(frozen=True, slots=True)
class BrokenPart:
    dir: Path
    tier: Tier
    reason: str


def project_parts_dir(project_root: Path) -> Path:
    return Path(project_root) / ".kicraft" / PARTS_SUBDIR


def home_parts_dir() -> Path:
    return Path.home() / ".kicraft" / PARTS_SUBDIR


def vendored_parts_dir() -> Path:
    """Directory shipped with the KiCraft package: ``<install>/parts_library/``."""
    # this file lives at kicraft/parts_library/loader.py
    return Path(__file__).resolve().parent


def extra_parts_dirs() -> list[Path]:
    raw = os.environ.get(ENV_VAR, "")
    out: list[Path] = []
    if not raw:
        return out
    for entry in raw.split(os.pathsep):
        if entry:
            out.append(Path(entry).expanduser())
    return out


def resolve_tier_dirs(project_root: Path | None) -> list[tuple[Tier, Path]]:
    """Return the ordered list of (tier, base_dir) to search.

    Tiers whose base dirs do not exist are still returned — callers
    handle missing directories as empty tiers, which lets ``list-parts``
    report the search path even when nothing is installed yet.
    """
    out: list[tuple[Tier, Path]] = []
    if project_root is not None:
        out.append((Tier.PROJECT, project_parts_dir(project_root)))
    out.append((Tier.HOME, home_parts_dir()))
    out.append((Tier.VENDORED, vendored_parts_dir()))
    for d in extra_parts_dirs():
        out.append((Tier.EXTRA, d))
    return out


def _iter_part_dirs(base: Path) -> Iterator[Path]:
    if not base.is_dir():
        return
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        # Skip directories without a manifest — the vendored tier sits
        # inside the kicraft package and contains its own .py files.
        if not (child / "manifest.json").is_file():
            continue
        yield child


def _load_one(part_dir: Path, tier: Tier) -> LoadedPart | BrokenPart:
    """Validate a single part directory; never raises."""
    if not manifest_path(part_dir).is_file():
        return BrokenPart(dir=part_dir, tier=tier, reason="missing manifest.json")
    try:
        manifest = load_manifest(part_dir)
    except ValidationError as exc:
        return BrokenPart(dir=part_dir, tier=tier, reason=f"manifest schema: {exc}")
    except Exception as exc:  # noqa: BLE001 — JSON decode, encoding, etc.
        return BrokenPart(dir=part_dir, tier=tier, reason=f"manifest read: {exc}")

    if part_dir.name != manifest.name:
        return BrokenPart(
            dir=part_dir,
            tier=tier,
            reason=(
                f"directory name {part_dir.name!r} does not match "
                f"manifest name {manifest.name!r}"
            ),
        )

    missing = required_files_present(part_dir, manifest)
    if missing:
        return BrokenPart(
            dir=part_dir,
            tier=tier,
            reason=f"missing required file(s): {', '.join(missing)}",
        )

    if compute_content_hash(part_dir) != manifest.content_hash:
        return BrokenPart(
            dir=part_dir,
            tier=tier,
            reason=(
                "content_hash mismatch -- files were edited after the manifest "
                "was written; re-run `validate-part` to recompute it"
            ),
        )

    return LoadedPart(dir=part_dir, tier=tier, manifest=manifest)


def find_part(name: str, project_root: Path | None) -> LoadedPart | None:
    """Return the highest-priority loaded part with this name, or None.

    Walks tiers in order; the first tier whose ``<base>/<name>/`` loads
    cleanly wins. Broken parts in a tier are logged at warning level but
    don't prevent lower tiers from being checked.
    """
    for tier, base in resolve_tier_dirs(project_root):
        candidate = base / name
        if not candidate.is_dir():
            continue
        result = _load_one(candidate, tier)
        if isinstance(result, LoadedPart):
            return result
        log.warning("part %s in %s tier: %s", name, tier.value, result.reason)
    return None


def load_all_with_overrides(
    project_root: Path | None,
) -> tuple[list[LoadedPart], list[LoadedPart], list[BrokenPart]]:
    """Return (active, shadowed, broken) across all tiers.

    - ``active`` — one entry per name, from the highest-priority tier
      where it loads cleanly. Sorted by name.
    - ``shadowed`` — lower-tier copies of names that are also active in
      a higher tier; useful for ``list-parts`` to report what's hidden.
    - ``broken`` — directories that failed validation, in tier order.
    """
    seen_active: dict[str, LoadedPart] = {}
    shadowed: list[LoadedPart] = []
    broken: list[BrokenPart] = []
    for tier, base in resolve_tier_dirs(project_root):
        for part_dir in _iter_part_dirs(base):
            result = _load_one(part_dir, tier)
            if isinstance(result, BrokenPart):
                broken.append(result)
                continue
            name = result.manifest.name
            if name in seen_active:
                shadowed.append(result)
            else:
                seen_active[name] = result
    active = sorted(seen_active.values(), key=lambda p: p.manifest.name)
    return active, shadowed, broken


__all__ = [
    "BrokenPart",
    "ENV_VAR",
    "LoadedPart",
    "PARTS_SUBDIR",
    "Tier",
    "extra_parts_dirs",
    "find_part",
    "home_parts_dir",
    "load_all_with_overrides",
    "project_parts_dir",
    "resolve_tier_dirs",
    "vendored_parts_dir",
]
