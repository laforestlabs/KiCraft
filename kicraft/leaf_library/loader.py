"""Resolve and iterate the on-disk leaf library.

The library lives at ``$KICRAFT_LEAF_LIB`` (default
``~/.kicraft/leaves/``). It is created lazily; a missing directory is
treated as an empty library, never as an error.

The loader hash-verifies each leaf using an mtime-keyed in-memory cache
so re-running the architecture stage on the same process doesn't pay
for a fresh sha256 of every leaf on every turn. Synthesis re-verifies
unconditionally at install time -- the one moment a stale hash matters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .manifest import (
    LEAF_REQUIRED_FILES,
    Manifest,
    compute_content_hash,
    load_manifest,
    manifest_path,
    required_files_present,
)

log = logging.getLogger(__name__)

DEFAULT_LIBRARY_DIR = Path.home() / ".kicraft" / "leaves"
LIBRARY_ENV_VAR = "KICRAFT_LEAF_LIB"


@dataclass(frozen=True, slots=True)
class LoadedLeaf:
    """A library leaf that passed all validation, ready for reuse."""

    dir: Path
    manifest: Manifest

    @property
    def slug(self) -> str:
        return f"{self.manifest.name}@{self.manifest.version}"


@dataclass(frozen=True, slots=True)
class BrokenLeaf:
    """A library directory that failed validation, with the reason."""

    dir: Path
    reason: str


def resolve_library_dir() -> Path:
    """Return the active library directory, honoring ``$KICRAFT_LEAF_LIB``.

    The result may not exist on disk; callers handle that case.
    """
    override = os.environ.get(LIBRARY_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_LIBRARY_DIR


def _max_mtime(leaf_dir: Path) -> float:
    """Return the largest mtime under ``leaf_dir`` (used as a cache key)."""
    latest = 0.0
    for p in leaf_dir.rglob("*"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > latest:
            latest = mtime
    return latest


class LeafLibrary:
    """A view over the on-disk leaf library.

    Construct with a custom ``base_dir`` for tests / GUI; otherwise call
    :meth:`from_env` to resolve from ``$KICRAFT_LEAF_LIB`` / the default.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        # leaf_dir -> (mtime_at_verify, recomputed_hash)
        self._hash_cache: dict[Path, tuple[float, str]] = {}

    @classmethod
    def from_env(cls) -> "LeafLibrary":
        return cls(resolve_library_dir())

    def iter_dirs(self) -> Iterator[Path]:
        """Yield every direct child directory of the library base.

        Returns an empty iterator if the base dir does not exist.
        """
        if not self.base_dir.is_dir():
            return
        for child in sorted(self.base_dir.iterdir()):
            if child.is_dir():
                yield child

    def _verify_hash(self, leaf_dir: Path, manifest: Manifest) -> bool:
        """Hash-verify with mtime cache."""
        mtime = _max_mtime(leaf_dir)
        cached = self._hash_cache.get(leaf_dir)
        if cached and cached[0] == mtime:
            return cached[1] == manifest.content_hash
        actual = compute_content_hash(leaf_dir)
        self._hash_cache[leaf_dir] = (mtime, actual)
        return actual == manifest.content_hash

    def load_one(self, leaf_dir: Path) -> LoadedLeaf | BrokenLeaf:
        """Load and validate a single leaf directory."""
        missing = required_files_present(leaf_dir)
        if missing:
            return BrokenLeaf(
                dir=leaf_dir,
                reason=f"missing required file(s): {', '.join(missing)}",
            )
        if not manifest_path(leaf_dir).exists():
            return BrokenLeaf(dir=leaf_dir, reason="missing manifest.json")
        try:
            manifest = load_manifest(leaf_dir)
        except ValidationError as exc:
            return BrokenLeaf(dir=leaf_dir, reason=f"manifest schema: {exc}")
        except Exception as exc:  # JSON decode, encoding, etc.
            return BrokenLeaf(dir=leaf_dir, reason=f"manifest read: {exc}")

        # Directory name must match manifest name -- catches rename mistakes.
        if leaf_dir.name != manifest.name:
            return BrokenLeaf(
                dir=leaf_dir,
                reason=(
                    f"directory name {leaf_dir.name!r} does not match "
                    f"manifest name {manifest.name!r}"
                ),
            )

        if not self._verify_hash(leaf_dir, manifest):
            return BrokenLeaf(
                dir=leaf_dir,
                reason=(
                    "content_hash mismatch -- files were edited after "
                    "promotion (re-promote with a bumped patch version)"
                ),
            )

        return LoadedLeaf(dir=leaf_dir, manifest=manifest)

    def load_all(self) -> tuple[list[LoadedLeaf], list[BrokenLeaf]]:
        """Load every leaf. Returns (loaded, broken).

        Broken leaves are logged at warning level but never raise.
        """
        loaded: list[LoadedLeaf] = []
        broken: list[BrokenLeaf] = []
        for leaf_dir in self.iter_dirs():
            result = self.load_one(leaf_dir)
            if isinstance(result, LoadedLeaf):
                loaded.append(result)
            else:
                log.warning(
                    "leaf %s excluded: %s", leaf_dir.name, result.reason
                )
                broken.append(result)
        return loaded, broken

    def find(self, slug: str) -> LoadedLeaf | None:
        """Resolve a ``"<name>@<version>"`` slug to a loaded leaf.

        Returns None if the leaf is missing or broken.
        """
        if "@" not in slug:
            return None
        name, version = slug.split("@", 1)
        leaf_dir = self.base_dir / name
        if not leaf_dir.is_dir():
            return None
        result = self.load_one(leaf_dir)
        if not isinstance(result, LoadedLeaf):
            return None
        if result.manifest.version != version:
            return None
        return result


__all__ = [
    "BrokenLeaf",
    "DEFAULT_LIBRARY_DIR",
    "LIBRARY_ENV_VAR",
    "LeafLibrary",
    "LoadedLeaf",
    "resolve_library_dir",
]
