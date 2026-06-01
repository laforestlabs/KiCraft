"""Extract (symbol "Library:Name" ...) blocks from KiCad symbol libraries.

A leaf ``.kicad_sch`` file must contain a ``(lib_symbols ...)`` block
listing every symbol it references. KiCad expects those blocks to be
the exact text from the corresponding ``<Library>.kicad_sym`` file,
qualified with the library prefix (``Library:Name`` instead of bare
``Name``) and with any ``(extends ...)`` references resolved into a
self-contained block.

This module owns that extraction. Library discovery is delegated to
:mod:`kicraft.design.synthesis.parts_lookup`, which walks the
four-tier parts library before falling back to KiCad stock. Missing
libraries or symbols raise, not warn.
"""
from __future__ import annotations

import re
from pathlib import Path

from .parts_lookup import (
    DEFAULT_KICAD_SYMBOL_DIR,
    LibraryNotFoundError,
    resolve_symbol_library_path,
)


class SymbolNotFoundError(LookupError):
    """Raised when a KiCad symbol library is missing or the symbol isn't in it."""


# ---------- helpers ----------


def _match_block(text: str, start: int) -> str:
    """Return the parenthesized block beginning at the '(' at `start`."""
    if text[start] != "(":
        raise ValueError(f"expected '(' at position {start}, got {text[start]!r}")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unmatched parenthesis while extracting symbol block")


def _find_symbol_start(text: str, name: str) -> int | None:
    """Locate `(symbol "<name>" ...)` in `text`. Returns the index of `(` or None."""
    needle = f'(symbol "{name}"'
    pos = 0
    while True:
        idx = text.find(needle, pos)
        if idx == -1:
            return None
        # Verify the character after the name+quote is whitespace or another paren —
        # otherwise we matched a prefix (e.g. "C" matching "CP1").
        after = idx + len(needle)
        if after < len(text) and text[after] in " \t\n\r(":
            return idx
        pos = idx + 1


def _extract_properties(symbol_text: str) -> dict[str, str]:
    """Extract every top-level (property "Name" ...) block from a symbol's text."""
    props: dict[str, str] = {}
    i = 0
    while True:
        idx = symbol_text.find('(property "', i)
        if idx == -1:
            break
        name_start = idx + len('(property "')
        name_end = symbol_text.find('"', name_start)
        name = symbol_text[name_start:name_end]
        block = _match_block(symbol_text, idx)
        props[name] = block
        i = idx + len(block)
    return props


def _get_extends_base(symbol_text: str) -> str | None:
    """Return the base symbol name if `symbol_text` is an `(extends ...)` derivative."""
    m = re.search(r'\(extends "([^"]+)"\)', symbol_text)
    return m.group(1) if m else None


def _resolve_extends_chain(lib_text: str, symbol_name: str) -> str:
    """Resolve (extends ...) by inlining the base's graphics with the derived's properties.

    KiCad's stock libraries usually have a single level of extends; chains
    deeper than that are rare but supported here by recursion.
    """
    start = _find_symbol_start(lib_text, symbol_name)
    if start is None:
        raise SymbolNotFoundError(f"symbol {symbol_name!r} not found in library")
    derived = _match_block(lib_text, start)
    base_name = _get_extends_base(derived)
    if base_name is None:
        return derived

    base_resolved = _resolve_extends_chain(lib_text, base_name)
    # KiCad names a symbol's unit/body-style sub-symbols `<name>_<unit>_<body>`
    # and rejects the whole symbol (the embedded `(lib_symbols ...)` fails to
    # load, leaving the sheet empty) if a sub-symbol's prefix doesn't match the
    # parent. The inherited graphics carry the BASE name's sub-symbols
    # (e.g. `USBLC6-2P6_0_1`), so rename their prefix to the derived name.
    # Do this BEFORE renaming the header: while the header is still the exact
    # `(symbol "<base>"`, the `(symbol "<base>_` pattern can't match it — which
    # matters when the derived name itself begins with `<base>_` (e.g. the base
    # `C` and the derivative `C_Small`).
    merged = base_resolved.replace(
        f'(symbol "{base_name}_', f'(symbol "{symbol_name}_'
    )
    # Now rename the base symbol header to the derived name. Use the exact bytes
    # `(symbol "<base_name>"` so we don't substring-match inside other symbols
    # in the chain.
    merged = merged.replace(
        f'(symbol "{base_name}"', f'(symbol "{symbol_name}"', 1
    )
    derived_props = _extract_properties(derived)
    merged_props = _extract_properties(merged)
    for prop_name, prop_block in derived_props.items():
        if prop_name in merged_props:
            merged = merged.replace(merged_props[prop_name], prop_block, 1)
    return merged


def _qualify_with_prefix(symbol_text: str, symbol_name: str, library: str) -> str:
    """Rewrite (symbol "Name" ...) to (symbol "Library:Name" ...) once."""
    return symbol_text.replace(
        f'(symbol "{symbol_name}"', f'(symbol "{library}:{symbol_name}"', 1
    )


# ---------- public API ----------


def extract_symbol_block(
    library: str,
    symbol_name: str,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_SYMBOL_DIR,
) -> str:
    """Return the fully-resolved, qualified `(symbol "Library:Name" ...)` text.

    The library prefix is resolved through the parts-library 4-tier
    search (project / home / vendored / extras) before falling back to
    ``stock_dir``. See :mod:`parts_lookup` for tier ordering.

    Args:
        library: KiCad library prefix (e.g. ``Device``, ``ip2368``).
        symbol_name: Symbol within that library (e.g. ``C``, ``IP2368``).
        project_root: Defaults to ``Path.cwd()``. Pass explicitly when
            invoking from a directory other than the KiCraft project.
        stock_dir: KiCad stock-library directory used as tier 5.

    Raises:
        SymbolNotFoundError: library file missing, or symbol not in it.
    """
    try:
        lib_path = resolve_symbol_library_path(
            library, project_root=project_root, stock_dir=stock_dir
        )
    except LibraryNotFoundError as exc:
        raise SymbolNotFoundError(str(exc)) from exc
    lib_text = lib_path.read_text()
    resolved = _resolve_extends_chain(lib_text, symbol_name)
    return _qualify_with_prefix(resolved, symbol_name, library)


def build_lib_symbols_block(
    pairs: list[tuple[str, str]],
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_SYMBOL_DIR,
    indent: str = "\t",
) -> str:
    """Build a complete `(lib_symbols ...)` block from a list of (library, name) pairs.

    Pairs are deduplicated; missing symbols raise SymbolNotFoundError before any
    output is produced (no partial blocks).
    """
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)

    if not unique:
        return f"{indent}(lib_symbols)"

    blocks = [
        extract_symbol_block(lib, name, project_root=project_root, stock_dir=stock_dir)
        for lib, name in unique
    ]
    body = "\n".join(f"{indent}\t{b}" for b in blocks)
    return f"{indent}(lib_symbols\n{body}\n{indent})"
