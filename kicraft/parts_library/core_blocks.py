"""Canonical core-blocks catalog: one curated default part per functional block.

This module (with ``core_blocks.json`` beside it) is the single source of
truth for the core-components registry. The accounts DB only mirrors it:
``AccountStore._sync_core_components_from_catalog()`` re-syncs the
``core_components`` table from this catalog on every store init, so block
and part edits happen here (via git) while the DB owns nothing but runtime
state (the ``enabled`` flag and jlcparts price/stock snapshots).

Every block is exactly one of three kinds:

- ``bundle``: the default part is a vendored parts-library bundle
  (``kicraft/parts_library/<bundle>/``). Its MPN and LCSC id are DERIVED
  from the bundle manifest at sync time and never duplicated here, so the
  bundle and the registry cannot disagree.
- ``stock``: a stock-KiCad-backed passive series (``Device:R``/``Device:C``
  symbols); there is no bundle and no single LCSC id.
- transitional ``default_mpn``/``default_lcsc``: a curated default that is
  not vendored yet. Allowed only while the vendoring batches are in flight;
  the catalog guard test counts these down to zero, after which the fields
  are removed from the schema.

``package`` stays authored prose ("SOT-23-5", "0402"): manifests only carry
the raw footprint name, and the short package label is curation, not
duplication. Two blocks may share one bundle (gyroscope and imu-6axis both
resolve to the same 6-axis IMU).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .manifest import PART_NAME_RE, load_manifest

CORE_COMPONENT_CATEGORIES = ("power", "sensors", "drivers", "interface", "passives")
FUNCTION_KEY_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,62}[a-z0-9]")
_LCSC_RE = re.compile(r"C\d{1,12}")

CORE_BLOCKS_PATH = Path(__file__).resolve().parent / "core_blocks.json"


class StockSeries(BaseModel):
    """A stock-KiCad-backed passive series ("UNI-ROYAL 0402WGF series").

    The series text becomes the row's ``default_mpn`` at sync time; the BOM
    stage maps these rows to stock ``Device:R``/``Device:C`` symbols.
    """

    model_config = ConfigDict(extra="forbid")

    series: str


class CoreBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_key: str
    display_name: str
    category: Literal["power", "sensors", "drivers", "interface", "passives"]
    qualifier: str | None = None
    package: str | None = None  # authored prose, not a KiCad footprint id
    selection_notes: str | None = None
    sort_order: int = 0
    bundle: str | None = None
    stock: StockSeries | None = None
    # Transitional kind: a default not vendored yet. Forbidden once the
    # vendoring batches complete (the catalog guard counts these to zero).
    default_mpn: str | None = None
    default_lcsc: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CoreBlock":
        if not FUNCTION_KEY_RE.fullmatch(self.function_key):
            raise ValueError(
                f"function_key {self.function_key!r} must be 3-64 chars of "
                f"a-z, 0-9, '-' or '_', starting and ending alphanumeric"
            )
        if not self.display_name.strip():
            raise ValueError(f"{self.function_key}: display_name must not be empty")
        kinds = sum(
            x is not None for x in (self.bundle, self.stock, self.default_lcsc)
        )
        if kinds != 1:
            raise ValueError(
                f"{self.function_key}: exactly one of bundle / stock / "
                f"default_lcsc must be set (got {kinds})"
            )
        if (self.default_lcsc is None) != (self.default_mpn is None):
            raise ValueError(
                f"{self.function_key}: default_mpn and default_lcsc go together"
            )
        if self.default_lcsc is not None and not _LCSC_RE.fullmatch(self.default_lcsc):
            raise ValueError(
                f"{self.function_key}: default_lcsc must be an LCSC id like C14259"
            )
        if self.bundle is not None and not PART_NAME_RE.match(self.bundle):
            raise ValueError(
                f"{self.function_key}: bundle {self.bundle!r} is not a valid "
                f"parts-library name"
            )
        if self.stock is not None and self.category != "passives":
            raise ValueError(
                f"{self.function_key}: stock-series rows must be category "
                f"'passives', not {self.category!r}"
            )
        return self


class CoreBlockCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    blocks: list[CoreBlock]

    @model_validator(mode="after")
    def _unique_keys(self) -> "CoreBlockCatalog":
        seen: set[str] = set()
        dupes = sorted(
            b.function_key for b in self.blocks
            if b.function_key in seen or seen.add(b.function_key)
        )
        if dupes:
            raise ValueError(f"duplicate function_key(s): {', '.join(dupes)}")
        return self


def load_core_catalog(path: Path | None = None) -> CoreBlockCatalog:
    p = path or CORE_BLOCKS_PATH
    return CoreBlockCatalog.model_validate(json.loads(p.read_text(encoding="utf-8")))


def resolve_block(block: CoreBlock, *, parts_dir: Path | None = None) -> dict:
    """Flatten a catalog block to the ``core_components`` DB row shape.

    Bundle-backed blocks read the vendored bundle's manifest for MPN and
    LCSC id. Deliberately ``load_manifest`` only (no content-hash
    verification): this runs on every AccountStore init and must not hash
    the bundles' 3D payloads; the parts-library CI guards own integrity.
    ``parts_dir`` overrides the vendored dir for tests.
    """
    out: dict = {
        "function_key": block.function_key,
        "display_name": block.display_name,
        "category": block.category,
        "qualifier": block.qualifier,
        "package": block.package,
        "selection_notes": block.selection_notes,
        "sort_order": block.sort_order,
        "bundle": block.bundle,
    }
    if block.bundle is not None:
        from .loader import vendored_parts_dir

        base = parts_dir if parts_dir is not None else vendored_parts_dir()
        manifest = load_manifest(base / block.bundle)
        out["default_mpn"] = manifest.mpn
        out["default_lcsc"] = (manifest.sourcing or {}).get("lcsc")
    elif block.stock is not None:
        out["default_mpn"] = block.stock.series
        out["default_lcsc"] = None
    else:
        out["default_mpn"] = block.default_mpn
        out["default_lcsc"] = block.default_lcsc
    return out


__all__ = [
    "CORE_BLOCKS_PATH",
    "CORE_COMPONENT_CATEGORIES",
    "CoreBlock",
    "CoreBlockCatalog",
    "FUNCTION_KEY_RE",
    "StockSeries",
    "load_core_catalog",
    "resolve_block",
]
