"""Configuration-preset store backed by a flat JSON file.

Replaces the SQL ``presets`` table with a single human-editable JSON file at
``~/.config/kicraft/presets.json`` (or ``$XDG_CONFIG_HOME/kicraft/presets.json``
when set). Presets are user-level state, not project-level, so they live in
the user config dir rather than ``.experiments/``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Preset:
    name: str
    config: dict[str, Any]
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": dict(self.config),
            "notes": self.notes,
            "created_at": self.created_at,
        }


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
    out: list[Preset] = []
    for item in _load_all():
        out.append(
            Preset(
                name=str(item.get("name", "")),
                config=item.get("config", {}) or {},
                notes=str(item.get("notes", "") or ""),
                created_at=str(item.get("created_at", "") or ""),
            )
        )
    out.sort(key=lambda p: p.name.lower())
    return out


def save_preset(name: str, config: dict[str, Any], notes: str = "") -> Preset:
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
    for item in _load_all():
        if item.get("name") == name:
            cfg = item.get("config", {})
            return dict(cfg) if isinstance(cfg, dict) else None
    return None


def delete_preset(name: str) -> bool:
    items = _load_all()
    new_items = [item for item in items if item.get("name") != name]
    if len(new_items) == len(items):
        return False
    _write_all(new_items)
    return True
