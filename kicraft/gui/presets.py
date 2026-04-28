"""Configuration-preset store backed by a flat JSON file.

Replaces the SQL ``presets`` table with a single human-editable JSON file at
``~/.config/kicraft/presets.json`` (or ``$XDG_CONFIG_HOME/kicraft/presets.json``
when set). Presets are user-level state, not project-level, so they live in
the user config dir rather than ``.experiments/``.

Built-in presets (e.g. ``DEFAULT``) are computed at runtime from the current
source-of-truth defaults (``DEFAULT_CONFIG`` / ``CONFIG_SEARCH_SPACE`` /
``PLACEMENT_PARAMS``). They are read-only -- the GUI exposes them in the
preset list but refuses to overwrite or delete them. This guarantees that
"DEFAULT" always means "the values in the code", regardless of any user
edits to the on-disk preset store.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Names of presets that are computed from source-of-truth defaults instead of
# read from the on-disk store. These cannot be saved over or deleted.
BUILTIN_PRESET_NAMES: frozenset[str] = frozenset({"DEFAULT"})


@dataclass
class Preset:
    name: str
    config: dict[str, Any]
    notes: str = ""
    created_at: str = ""
    builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": dict(self.config),
            "notes": self.notes,
            "created_at": self.created_at,
            "builtin": self.builtin,
        }


def _build_default_preset() -> Preset:
    """Build the read-only DEFAULT preset from current source-of-truth values.

    The preset config carries:
      - ``_placement_config``: a snapshot of PLACEMENT_PARAMS default values
        for every numeric/bool/text/list parameter, so loading the preset
        resets every GUI slider/toggle to its source-coded default.
      - ``_mutation_bounds``: the CONFIG_SEARCH_SPACE [min, max] for every
        searchable knob, so loading the preset resets the search ranges too.

    Computed lazily so the preset always reflects the current source code,
    not whatever was on disk when the GUI process started.
    """
    from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE
    from kicraft.gui.state import PLACEMENT_PARAMS

    placement_defaults: dict[str, Any] = {}
    for param in PLACEMENT_PARAMS:
        key = param.get("key")
        if not isinstance(key, str):
            continue
        if "default" in param:
            placement_defaults[key] = param["default"]

    mutation_bounds: dict[str, list[float | int]] = {
        key: [spec["min"], spec["max"]]
        for key, spec in CONFIG_SEARCH_SPACE.items()
    }

    config = {
        "_placement_config": placement_defaults,
        "_mutation_bounds": mutation_bounds,
    }
    notes = (
        "Read-only built-in preset. Resets every placement param to its "
        "source-coded default and every search-space knob to its current "
        "min/max. Loading this preset is the safe way to undo experiment "
        "drift; you cannot overwrite or delete it from the GUI."
    )
    return Preset(
        name="DEFAULT",
        config=config,
        notes=notes,
        created_at="",
        builtin=True,
    )


def _builtin_presets() -> list[Preset]:
    """Return all read-only built-in presets, in display order."""
    return [_build_default_preset()]


def is_builtin(name: str) -> bool:
    return name in BUILTIN_PRESET_NAMES


def _store_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "kicraft" / "presets.json"
    return Path.home() / ".config" / "kicraft" / "presets.json"


def _load_all() -> list[dict[str, Any]]:
    path = _store_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _write_all(items: list[dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, sort_keys=False), encoding="utf-8")


def list_presets() -> list[Preset]:
    """Return built-in presets first, then on-disk user presets.

    User entries whose name collides with a built-in are filtered out so
    a corrupted or hand-edited presets.json can't shadow the read-only
    defaults.
    """
    out: list[Preset] = list(_builtin_presets())
    for item in _load_all():
        name = str(item.get("name", ""))
        if name in BUILTIN_PRESET_NAMES:
            continue
        out.append(
            Preset(
                name=name,
                config=item.get("config", {}) or {},
                notes=str(item.get("notes", "") or ""),
                created_at=str(item.get("created_at", "") or ""),
                builtin=False,
            )
        )
    # Built-ins always sort first; user presets sorted alphabetically below.
    builtins = [p for p in out if p.builtin]
    user = sorted([p for p in out if not p.builtin], key=lambda p: p.name.lower())
    return builtins + user


def save_preset(name: str, config: dict[str, Any], notes: str = "") -> Preset:
    """Save a user preset.

    Raises ``ValueError`` if ``name`` matches a built-in preset
    (the read-only defaults must not be overwritten from the GUI or
    anywhere else in code).
    """
    if name in BUILTIN_PRESET_NAMES:
        raise ValueError(
            f"'{name}' is a built-in read-only preset and cannot be overwritten. "
            f"Save under a different name."
        )
    items = _load_all()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    found = False
    for i, item in enumerate(items):
        if item.get("name") == name:
            items[i] = {
                "name": name,
                "config": dict(config),
                "notes": notes,
                "created_at": item.get("created_at", now),
            }
            found = True
            break
    if not found:
        items.append(
            {
                "name": name,
                "config": dict(config),
                "notes": notes,
                "created_at": now,
            }
        )
    _write_all(items)
    return Preset(name=name, config=dict(config), notes=notes, created_at=now)


def load_preset(name: str) -> dict[str, Any] | None:
    """Return a preset's config dict, or None if not found.

    Built-ins are computed on every call from current source defaults.
    """
    if name in BUILTIN_PRESET_NAMES:
        for preset in _builtin_presets():
            if preset.name == name:
                return dict(preset.config)
        return None
    for item in _load_all():
        if item.get("name") == name:
            cfg = item.get("config", {})
            return dict(cfg) if isinstance(cfg, dict) else None
    return None


def delete_preset(name: str) -> bool:
    """Delete a user preset. Returns False (no-op) for built-ins."""
    if name in BUILTIN_PRESET_NAMES:
        return False
    items = _load_all()
    new_items = [item for item in items if item.get("name") != name]
    if len(new_items) == len(items):
        return False
    _write_all(new_items)
    return True
