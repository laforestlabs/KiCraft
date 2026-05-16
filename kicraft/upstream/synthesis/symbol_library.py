"""Extract (symbol "Library:Name" ...) blocks from stock KiCad symbol libraries.

A leaf `.kicad_sch` file must contain a `(lib_symbols ...)` block listing
every symbol it references. KiCad expects those blocks to be the exact text
from the corresponding `<Library>.kicad_sym` file, qualified with the
library prefix (`Library:Name` instead of bare `Name`) and with any
`(extends ...)` references resolved into a self-contained block.

This module owns that extraction. Ported from `generate_project.py` with a
strict API: missing libraries or symbols raise, not warn.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

DEFAULT_KICAD_SYMBOL_DIR: Final = Path("/usr/share/kicad/symbols")


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
    # Rename the base symbol header to the derived name. Use the exact bytes
    # `(symbol "<base_name>"` so we don't substring-match inside other symbols
    # in the chain.
    merged = base_resolved.replace(
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
    symbol_dir: Path = DEFAULT_KICAD_SYMBOL_DIR,
) -> str:
    """Return the fully-resolved, qualified `(symbol "Library:Name" ...)` text.

    Args:
        library: KiCad library name (e.g. `Device`, `Regulator_Linear`).
        symbol_name: Symbol within the library (e.g. `C`, `AP2112K-3.3`).
        symbol_dir: Directory containing `<Library>.kicad_sym` files.

    Raises:
        SymbolNotFoundError: library file missing or symbol not in library.
    """
    lib_path = symbol_dir / f"{library}.kicad_sym"
    if not lib_path.is_file():
        raise SymbolNotFoundError(f"library {library!r} not found at {lib_path}")
    lib_text = lib_path.read_text()
    resolved = _resolve_extends_chain(lib_text, symbol_name)
    return _qualify_with_prefix(resolved, symbol_name, library)


def build_lib_symbols_block(
    pairs: list[tuple[str, str]],
    symbol_dir: Path = DEFAULT_KICAD_SYMBOL_DIR,
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

    blocks = [extract_symbol_block(lib, name, symbol_dir) for lib, name in unique]
    body = "\n".join(f"{indent}\t{b}" for b in blocks)
    return f"{indent}(lib_symbols\n{body}\n{indent})"
