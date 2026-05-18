"""Leaf-library manifest model + content-hash + on-disk read/write.

A library leaf lives in ``$KICRAFT_LEAF_LIB/<name>/`` and consists of:

- ``manifest.json``           -- this model, serialized
- ``leaf_routed.kicad_pcb``   -- pinned routed PCB fragment (canonical name)
- ``metadata.json``           -- subcircuit metadata (canonical name)
- ``solved_layout.json``      -- canonical solved layout (composer input)
- ``schematic.kicad_sch``     -- the leaf sheet
- ``autoplacer_fragment.json``-- leaf-scoped slice of project autoplacer JSON
- ``bom.csv``                 -- leaf-scoped BOM rows
- ``renders/``                -- ``front_all.png``, ``back_copper.png``,
                                  ``copper_both.png``, ``thumbnail.png``

The canonical triad (``leaf_routed.kicad_pcb`` + ``metadata.json`` +
``solved_layout.json``) is what the existing pin manager and parent
composer expect; the manifest declares it explicitly so a library leaf
is drop-in compatible with ``.experiments/subcircuits/<leaf_key>/``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Re-defined here rather than imported from circuitchat.models so the
# leaf_library package stays self-contained. A test asserts the two
# definitions stay in sync.
PinDirection = Literal["input", "output", "bidirectional", "passive"]

LEAF_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")
REF_RE = re.compile(r"^[A-Z]+[0-9]+$")
HIER_LABEL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

CANONICAL_TRIAD = (
    "leaf_routed.kicad_pcb",
    "metadata.json",
    "solved_layout.json",
)
LEAF_REQUIRED_FILES = CANONICAL_TRIAD + (
    "schematic.kicad_sch",
    "autoplacer_fragment.json",
    "bom.csv",
)


class HierarchicalLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    direction: PinDirection

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not HIER_LABEL_NAME_RE.match(v):
            raise ValueError(
                f"hierarchical label name {v!r} must match {HIER_LABEL_NAME_RE.pattern}"
            )
        return v


class Interface(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hierarchical_labels: list[HierarchicalLabel]


class Dependencies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kicad_symbol_libs: list[str] = Field(default_factory=list)
    kicad_footprint_libs: list[str] = Field(default_factory=list)
    kicad_version_min: str


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_project_stem: str
    source_sheet_name: str
    source_experiment_round: int
    promoted_at: str  # ISO 8601 UTC, e.g. "2026-05-17T14:23:00Z"
    kicad_version: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    name: str
    version: str
    content_hash: str
    description: str
    tags: list[str] = Field(default_factory=list)
    watch_out_for: str | None = None
    interface: Interface
    refs: list[str]
    dependencies: Dependencies
    provenance: Provenance

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not LEAF_NAME_RE.match(v):
            raise ValueError(
                f"leaf name {v!r} must match {LEAF_NAME_RE.pattern}"
            )
        return v

    @field_validator("version")
    @classmethod
    def _semver_format(cls, v: str) -> str:
        if not SEMVER_RE.match(v):
            raise ValueError(
                f"version {v!r} must be a valid semver triple"
            )
        return v

    @field_validator("content_hash")
    @classmethod
    def _content_hash_format(cls, v: str) -> str:
        if not v.startswith("sha256:") or len(v) != len("sha256:") + 64:
            raise ValueError(
                "content_hash must be sha256:<64-hex-chars>"
            )
        try:
            int(v[len("sha256:"):], 16)
        except ValueError as exc:
            raise ValueError("content_hash hex segment is not valid hex") from exc
        return v

    @field_validator("refs")
    @classmethod
    def _refs_format(cls, v: list[str]) -> list[str]:
        for r in v:
            if not REF_RE.match(r):
                raise ValueError(
                    f"ref {r!r} must match {REF_RE.pattern} (no suffix forms in v1)"
                )
        if len(set(v)) != len(v):
            raise ValueError("refs must be unique")
        return v


def manifest_path(leaf_dir: Path) -> Path:
    return Path(leaf_dir) / "manifest.json"


def load_manifest(leaf_dir: Path) -> Manifest:
    """Load and validate the manifest in ``leaf_dir``.

    Raises ``FileNotFoundError`` if the manifest is missing; pydantic
    ``ValidationError`` on schema failure. Content-hash verification
    happens separately via :func:`verify_content_hash`.
    """
    path = manifest_path(leaf_dir)
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {leaf_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(payload)


def dump_manifest(manifest: Manifest, leaf_dir: Path) -> None:
    """Serialize ``manifest`` to ``leaf_dir/manifest.json`` atomically."""
    path = manifest_path(leaf_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def compute_content_hash(leaf_dir: Path) -> str:
    """Return ``sha256:<hex>`` over every file in ``leaf_dir`` *except*
    ``manifest.json`` itself.

    Files are visited in sorted relative-path order; each file's
    contribution is its UTF-8 relative path, a NUL byte, its size as a
    decimal string, a NUL byte, and its raw bytes. The encoding is
    independent of the filesystem's tar layout so the result is stable
    across machines.
    """
    h = hashlib.sha256()
    root = Path(leaf_dir)
    entries: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.name == "manifest.json" and rel.parent == Path("."):
            continue
        entries.append(p)
    for p in entries:
        rel = p.relative_to(root).as_posix().encode("utf-8")
        size = p.stat().st_size
        h.update(rel)
        h.update(b"\x00")
        h.update(str(size).encode("ascii"))
        h.update(b"\x00")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def verify_content_hash(leaf_dir: Path, manifest: Manifest) -> bool:
    """Return True if the recomputed hash matches the stored value."""
    return compute_content_hash(leaf_dir) == manifest.content_hash


def required_files_present(leaf_dir: Path) -> list[str]:
    """Return the names of required files that are missing from ``leaf_dir``.

    Empty list = nothing missing. Used by the loader to reject leaves
    that lack the canonical triad or core artifacts before the manifest
    is even parsed.
    """
    missing: list[str] = []
    for name in LEAF_REQUIRED_FILES:
        if not (Path(leaf_dir) / name).is_file():
            missing.append(name)
    return missing


__all__ = [
    "CANONICAL_TRIAD",
    "Dependencies",
    "HierarchicalLabel",
    "Interface",
    "LEAF_REQUIRED_FILES",
    "Manifest",
    "PinDirection",
    "Provenance",
    "compute_content_hash",
    "dump_manifest",
    "load_manifest",
    "manifest_path",
    "required_files_present",
    "verify_content_hash",
]
