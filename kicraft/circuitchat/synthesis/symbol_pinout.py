"""Pin-level inventory for KiCad 9 stock symbols.

Reuses ``symbol_library._resolve_extends_chain`` so the pin list of an
``(extends …)`` derivative is the resolved base's pin list. v1 callers
read unit 1 only.

The output schema matches what the wiring LLM stage needs: every pin's
number, name, electrical type, position, orientation, length.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .parts_lookup import (
    DEFAULT_KICAD_SYMBOL_DIR,
    LibraryNotFoundError,
    resolve_symbol_library_path,
)
from .symbol_library import (
    SymbolNotFoundError,
    _match_block,
    _resolve_extends_chain,
)


_SUB_SYMBOL_RE = re.compile(r'\(symbol\s+"([^"]+)_(\d+)_1"')
_PIN_HEADER_RE = re.compile(r"\(pin\s+(\w+)\s+\w+")
_AT_RE = re.compile(r"\(at\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\)")
_LENGTH_RE = re.compile(r"\(length\s+(-?\d+\.?\d*)\)")
_NAME_RE = re.compile(r'\(name\s+"([^"]*)"')
_NUMBER_RE = re.compile(r'\(number\s+"([^"]+)"')


def lookup_pins(
    lib_id: str,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_SYMBOL_DIR,
) -> dict:
    """Resolve `Library:Name` to its pin inventory.

    The library prefix is resolved through the parts-library four-tier
    search before falling back to KiCad stock at ``stock_dir``.

    Returns a dict with keys ``symbol``, ``unit_count``, and ``pins``
    (a list of pin dicts; see the README for the schema).
    """
    library, _, name = lib_id.partition(":")
    if not library or not name:
        raise SymbolNotFoundError(f"bad lib_id {lib_id!r} (expected 'Library:Name')")
    root_str = str(project_root) if project_root is not None else ""
    return _lookup_cached(library, name, root_str, str(stock_dir))


@lru_cache(maxsize=256)
def _lookup_cached(
    library: str, name: str, project_root_str: str, stock_dir_str: str
) -> dict:
    project_root = Path(project_root_str) if project_root_str else None
    stock_dir = Path(stock_dir_str)
    try:
        lib_path = resolve_symbol_library_path(
            library, project_root=project_root, stock_dir=stock_dir
        )
    except LibraryNotFoundError as exc:
        raise SymbolNotFoundError(str(exc)) from exc
    lib_text = lib_path.read_text()
    resolved = _resolve_extends_chain(lib_text, name)

    # Find every (symbol "<base>_<unit>_1" ...) sub-block. Unit 0 is the
    # graphics-only sub-symbol (rectangles, polylines); units 1+ carry
    # the pin definitions.
    pins_by_unit: dict[int, list[dict]] = {}
    pos = 0
    while True:
        m = _SUB_SYMBOL_RE.search(resolved, pos)
        if not m:
            break
        unit_num = int(m.group(2))
        sub_start = m.start()
        sub_block = _match_block(resolved, sub_start)
        if unit_num >= 1:
            sub_pins = _parse_pins(sub_block)
            if sub_pins:
                pins_by_unit.setdefault(unit_num, []).extend(sub_pins)
        pos = sub_start + len(sub_block)

    if not pins_by_unit:
        # Symbol without sub-symbol structure: parse pins from the outer
        # block directly. Rare in KiCad 9 stock libraries but possible
        # for hand-written symbols.
        outer_pins = _parse_pins(resolved)
        if outer_pins:
            pins_by_unit[1] = outer_pins

    unit_count = max(pins_by_unit) if pins_by_unit else 1
    unit_1_pins = [
        {**p, "unit": 1} for p in pins_by_unit.get(1, [])
    ]

    return {
        "symbol": f"{library}:{name}",
        "unit_count": unit_count,
        "pins": unit_1_pins,
    }


def _parse_pins(text: str) -> list[dict]:
    """Yield every parsed `(pin ...)` block found in `text` (single level)."""
    pins: list[dict] = []
    i = 0
    while True:
        idx = text.find("(pin ", i)
        if idx == -1:
            break
        block = _match_block(text, idx)
        info = _parse_one_pin(block)
        if info is not None:
            pins.append(info)
        i = idx + len(block)
    return pins


def _parse_one_pin(block: str) -> dict | None:
    m = _PIN_HEADER_RE.match(block)
    if not m:
        return None
    electrical_type = m.group(1)
    at = _AT_RE.search(block)
    if not at:
        return None
    name_m = _NAME_RE.search(block)
    num_m = _NUMBER_RE.search(block)
    if not num_m:
        return None
    length_m = _LENGTH_RE.search(block)
    return {
        "number": num_m.group(1),
        "name": name_m.group(1) if name_m else "",
        "electrical_type": electrical_type,
        "position": {"x": float(at.group(1)), "y": float(at.group(2))},
        "orientation": int(float(at.group(3))),
        "length": float(length_m.group(1)) if length_m else 2.54,
        "unit": 1,
    }
