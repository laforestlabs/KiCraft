"""Parts-library manifest model + content-hash + on-disk read/write.

A library part lives in ``<root>/<name>/`` and consists of:

- ``manifest.json``                       -- this model, serialized
- ``<name>.kicad_sym``                    -- a KiCad symbol library file whose
                                             library prefix equals ``<name>``
- ``<name>.pretty/<footprint>.kicad_mod`` -- the footprint, inside a KiCad
                                             ``.pretty`` directory whose prefix
                                             also equals ``<name>``
- ``3d/*.{step,wrl}``                     -- optional 3D models
- ``datasheet.pdf``                       -- optional cached PDF

A part is referenced by the BOM as ``<name>:<symbol_name>`` (for the symbol)
and ``<name>:<footprint_name>`` (for the footprint). The directory name,
the KiCad library prefix, and the manifest's ``name`` field are locked
together so a single regex validates all three.

Compared to ``leaf_library``: no Interface, no refs, no symbol-libs /
footprint-libs in Dependencies. A part is a leaf with no sub-structure.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PART_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")
SOURCING_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")

# Quality/trust tier of a bundle, orthogonal to the storage tier it lives in
# (project/home/vendored/extra). Lets an auto-fetched part be reused for the cost
# win while staying visibly flagged until a human vets it.
#   prototype  = auto-fetched (e.g. via add-part --from-lcsc); validated but unreviewed.
#   reviewed   = a human checked it; the curated vendored bundles sit here.
#   production = polished and verified (e.g. a real 3D model present).
Maturity = Literal["prototype", "reviewed", "production"]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # free-form: "manual", "easyeda2kicad", "snapeda", "vendored", etc.
    source_project_stem: str | None = None
    added_at: str  # ISO 8601 UTC, e.g. "2026-05-21T14:23:00Z"
    kicad_version: str


class PartManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    name: str  # lowercase kebab; equals directory name and KiCad library prefix
    version: str  # semver
    content_hash: str  # sha256 over every file except manifest.json
    description: str
    mpn: str  # canonical manufacturer part number
    sourcing: dict[str, str] = Field(default_factory=dict)
    datasheet_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    watch_out_for: str | None = None
    # Defaults to the most conservative tier so an unmarked or freshly fetched
    # bundle is treated as experimental until explicitly promoted. Editing this
    # never changes content_hash (which excludes manifest.json).
    maturity: Maturity = "prototype"
    symbol_name: str  # symbol name inside <name>.kicad_sym
    footprint_name: str  # footprint name inside <name>.pretty/ (without .kicad_mod)
    kicad_version_min: str
    provenance: Provenance

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not PART_NAME_RE.match(v):
            raise ValueError(
                f"part name {v!r} must match {PART_NAME_RE.pattern}"
            )
        return v

    @field_validator("version")
    @classmethod
    def _semver_format(cls, v: str) -> str:
        if not SEMVER_RE.match(v):
            raise ValueError(f"version {v!r} must be a valid semver triple")
        return v

    @field_validator("content_hash")
    @classmethod
    def _content_hash_format(cls, v: str) -> str:
        if not v.startswith("sha256:") or len(v) != len("sha256:") + 64:
            raise ValueError("content_hash must be sha256:<64-hex-chars>")
        try:
            int(v[len("sha256:"):], 16)
        except ValueError as exc:
            raise ValueError("content_hash hex segment is not valid hex") from exc
        return v

    @field_validator("sourcing")
    @classmethod
    def _sourcing_keys_format(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if not SOURCING_KEY_RE.match(key):
                raise ValueError(
                    f"sourcing key {key!r} must match {SOURCING_KEY_RE.pattern}"
                )
        return v


# ---------- on-disk helpers ----------


def manifest_path(part_dir: Path) -> Path:
    return Path(part_dir) / "manifest.json"


def symbol_file_path(part_dir: Path) -> Path:
    """Return the expected symbol-library file path inside ``part_dir``.

    The filename is locked to ``<dirname>.kicad_sym`` so KiCad's
    library-prefix resolution lines up with the directory layout.
    """
    return Path(part_dir) / f"{Path(part_dir).name}.kicad_sym"


def footprint_dir_path(part_dir: Path) -> Path:
    """Return the expected ``.pretty`` footprint directory inside ``part_dir``."""
    return Path(part_dir) / f"{Path(part_dir).name}.pretty"


def footprint_file_path(part_dir: Path, footprint_name: str) -> Path:
    return footprint_dir_path(part_dir) / f"{footprint_name}.kicad_mod"


def required_files_present(part_dir: Path, manifest: PartManifest | None = None) -> list[str]:
    """Return the names of required files that are missing from ``part_dir``.

    Empty list = nothing missing. If ``manifest`` is None, only checks
    for ``manifest.json`` and the symbol-library file. With a manifest,
    also checks for the specific footprint file declared in it.
    """
    missing: list[str] = []
    if not manifest_path(part_dir).is_file():
        missing.append("manifest.json")
    if not symbol_file_path(part_dir).is_file():
        missing.append(symbol_file_path(part_dir).name)
    if manifest is not None:
        fp = footprint_file_path(part_dir, manifest.footprint_name)
        if not fp.is_file():
            missing.append(str(fp.relative_to(part_dir)))
    return missing


def load_manifest(part_dir: Path) -> PartManifest:
    path = manifest_path(part_dir)
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {part_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PartManifest.model_validate(payload)


def dump_manifest(manifest: PartManifest, part_dir: Path) -> None:
    """Serialize ``manifest`` to ``part_dir/manifest.json`` atomically."""
    path = manifest_path(part_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def compute_content_hash(part_dir: Path) -> str:
    """Return ``sha256:<hex>`` over every file in ``part_dir`` *except*
    ``manifest.json`` itself.

    Same encoding as ``leaf_library.manifest.compute_content_hash`` so
    the hashes are filesystem-independent and stable across machines.
    """
    h = hashlib.sha256()
    root = Path(part_dir)
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


def verify_content_hash(part_dir: Path, manifest: PartManifest) -> bool:
    return compute_content_hash(part_dir) == manifest.content_hash


__all__ = [
    "PART_NAME_RE",
    "SEMVER_RE",
    "SOURCING_KEY_RE",
    "Maturity",
    "PartManifest",
    "Provenance",
    "compute_content_hash",
    "dump_manifest",
    "footprint_dir_path",
    "footprint_file_path",
    "load_manifest",
    "manifest_path",
    "required_files_present",
    "symbol_file_path",
    "verify_content_hash",
]
