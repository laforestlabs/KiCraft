"""View-model + renderer for the web app's part-library browser.

The catalog is the parts library seen through the four-tier loader
(:func:`kicraft.parts_library.load_all_with_overrides`): the curated vendored
bundles are the "standard library"; anything in the home / project tiers is a
part the user added. This module turns each :class:`LoadedPart` into what the
``/parts`` pages render:

* a "how to use" markdown doc built from the manifest (no per-bundle files
  required, so it works for every part and never disturbs ``content_hash``),
* an LCSC product link derived from the sourcing code, and
* SVG previews of the symbol and footprint, produced by ``kicad-cli`` and
  cached on disk keyed by the bundle's ``content_hash`` so an edited part
  re-renders automatically.

KiCanvas (used elsewhere for whole schematics/boards) cannot parse a bare
``.kicad_sym`` or ``.kicad_mod``, so previews go through ``kicad-cli`` instead,
the same already-required tool the synthesis/fab/render paths use. A missing or
broken ``kicad-cli`` degrades to "no preview", never an exception.

No web-framework imports live here on purpose: the page wiring is in ``web.py``
and these helpers stay unit-testable.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..parts_library import (
    LoadedPart,
    PartManifest,
    Tier,
    find_part,
    footprint_dir_path,
    load_all_with_overrides,
    symbol_file_path,
)

log = logging.getLogger(__name__)

_KICAD_CLI = "kicad-cli"

# Front-side layers that read well in a catalog thumbnail: copper, paste, silk,
# mask, the fab outline, the courtyard, and the board edge.
_FP_LAYERS = "F.Cu,F.Paste,F.SilkS,F.Mask,F.Fab,F.CrtYd,Edge.Cuts"

# An LCSC part code, e.g. "C6186". Anything else (empty, a non-LCSC sourcing key)
# yields no link rather than a malformed URL.
_LCSC_CODE_RE = re.compile(r"^C\d+$")

# Friendly names for the storage tier a part loaded from.
_TIER_LABELS = {
    Tier.VENDORED: "Standard",
    Tier.HOME: "Yours",
    Tier.PROJECT: "Project",
    Tier.EXTRA: "Extra",
}


# ---------- listing ----------


def catalog(project_root: Path | None = None) -> list[LoadedPart]:
    """Every active part across the four tiers, sorted by part number.

    One entry per name (a higher tier shadows a lower one); broken bundles are
    dropped. ``project_root=None`` (the web app's case) searches home + vendored
    + extras, which is exactly "standard library plus what the user added".
    """
    active, _shadowed, _broken = load_all_with_overrides(project_root)
    return sorted(
        active, key=lambda p: ((p.manifest.mpn or p.manifest.name).lower(), p.manifest.name)
    )


def get_part(name: str, project_root: Path | None = None) -> LoadedPart | None:
    """The single active part with this library ``name``, or None."""
    return find_part(name, project_root)


def tier_label(tier: Tier) -> str:
    """A short, user-facing label for a storage tier (e.g. ``Standard``)."""
    return _TIER_LABELS.get(tier, tier.value)


# ---------- links + docs ----------


def lcsc_url(manifest: PartManifest) -> str | None:
    """The LCSC product page for this part, or None if it has no LCSC code."""
    code = (manifest.sourcing or {}).get("lcsc", "").strip()
    if _LCSC_CODE_RE.match(code):
        return f"https://www.lcsc.com/product-detail/{code}.html"
    return None


def usage_markdown(part: LoadedPart) -> str:
    """A "how to use it" markdown doc assembled from the manifest.

    The manifest's ``description`` and ``watch_out_for`` already carry the prose
    a user needs; this lays them out plus the ids needed to actually reference
    the part in a BOM. If the bundle ships its own ``usage.md`` it is appended
    under "Notes" (vendored bundles do not, since adding a file would change
    their content_hash; this is the path for richer per-part docs later).
    """
    m = part.manifest
    lines: list[str] = [f"# {m.mpn}", "", m.description.strip(), ""]
    if m.watch_out_for:
        lines += ["## Watch out for", "", m.watch_out_for.strip(), ""]
    lines += [
        "## At a glance",
        "",
        f"- Library name: `{m.name}`",
        f"- BOM symbol id: `{m.name}:{m.symbol_name}`",
        f"- BOM footprint id: `{m.name}:{m.footprint_name}`",
        f"- Package: {m.footprint_name}",
    ]
    if m.tags:
        lines.append(f"- Tags: {', '.join(m.tags)}")
    lines += [
        f"- Maturity: {m.maturity}",
        f"- Library tier: {tier_label(part.tier)}",
        "",
    ]
    extra = part.dir / "usage.md"
    if extra.is_file():
        try:
            txt = extra.read_text(encoding="utf-8").strip()
        except OSError:
            txt = ""
        if txt:
            lines += ["## Notes", "", txt, ""]
    return "\n".join(lines)


# ---------- SVG previews ----------


def kicad_cli_available() -> bool:
    """True when ``kicad-cli`` is on PATH (previews are unavailable without it)."""
    return shutil.which(_KICAD_CLI) is not None


def _content_hash_key(manifest: PartManifest) -> str:
    """A short, filesystem-safe slice of the bundle's content hash.

    The hash covers every bundle file except manifest.json, so it changes when
    the symbol or footprint changes, which is exactly when a cached preview must
    be invalidated. Embedding it in the cache-dir name makes that automatic.
    """
    return manifest.content_hash.removeprefix("sha256:")[:12]


def _cache_dir(part: LoadedPart) -> Path:
    base = Path(tempfile.gettempdir()) / "kicraft-part-previews"
    return base / f"{part.manifest.name}-{_content_hash_key(part.manifest)}"


def _run_ok(cmd: list[str]) -> bool:
    """Run ``cmd``, returning True on success; log and return False otherwise.

    Tolerant by design: a bundle that one ``kicad-cli`` build can't render must
    not 500 the catalog, just show "preview unavailable".
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("part-preview: %s failed: %s", " ".join(cmd[:4]), exc)
        return False
    if r.returncode != 0:
        log.warning(
            "part-preview: %s rc=%s: %s",
            " ".join(cmd[:4]),
            r.returncode,
            (r.stderr or r.stdout).strip()[:300],
        )
        return False
    return True


def symbol_svgs(part: LoadedPart) -> list[Path]:
    """Cached SVG(s) for the part's symbol: one per unit (usually a single file).

    Empty list if ``kicad-cli`` is missing or the export fails.
    """
    out_dir = _cache_dir(part) / "symbol"
    if not (out_dir / ".ok").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        sym_file = symbol_file_path(part.dir)
        ok = _run_ok([
            _KICAD_CLI, "sym", "export", "svg",
            "-o", str(out_dir),
            "--symbol", part.manifest.symbol_name,
            str(sym_file),
        ])
        if ok:
            (out_dir / ".ok").touch()
    return sorted(out_dir.glob("*.svg"))


def footprint_svg(part: LoadedPart) -> Path | None:
    """Cached SVG for the part's footprint (front side), or None on failure."""
    out_dir = _cache_dir(part) / "footprint"
    if not (out_dir / ".ok").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = _run_ok([
            _KICAD_CLI, "fp", "export", "svg",
            "-o", str(out_dir),
            "--footprint", part.manifest.footprint_name,
            "--layers", _FP_LAYERS,
            str(footprint_dir_path(part.dir)),
        ])
        if ok:
            (out_dir / ".ok").touch()
    svgs = sorted(out_dir.glob("*.svg"))
    return svgs[0] if svgs else None


__all__ = [
    "catalog",
    "footprint_svg",
    "get_part",
    "kicad_cli_available",
    "lcsc_url",
    "symbol_svgs",
    "tier_label",
    "usage_markdown",
]
