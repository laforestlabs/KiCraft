"""Unit tests for the R2 architecture pre-resolve gate.

Covers ``kicraft.design.cli_app._unresolved_architecture_parts``, which scans
an architecture's assumptions/topologies text for core_defaults bundle names
and verifies each matched bundle's symbol resolves and its LCSC exists.

These tests touch the real vendored parts library and the real core catalog
(``kicraft.parts_library.core_blocks.load_core_catalog``); they do NOT require
KiCad stock symbols to be installed because the resolve-hit path only asserts
the function returns an empty list (the real library bundles resolve via the
project-vendored symbol files). The resolve-miss path monkeypatches
``lookup_pins`` so it is hermetic regardless.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kicraft.design.cli_app as cli_app
from kicraft.design.cli_app import _unresolved_architecture_parts
from kicraft.design.models import (
    Architecture,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesis.symbol_pinout import SymbolNotFoundError

# A core_defaults bundle name that is present in both the core catalog and the
# vendored parts library (verified against ``load_core_catalog`` + the live
# parts library). Referenced in the architecture assumptions text below.
_CORE_BUNDLE = "bmp388"


def _architecture_with_assumption(text: str) -> Architecture:
    """Minimal valid Architecture whose assumptions carry ``text``."""
    return Architecture(
        sheets=[Sheet(name="MCU", stem="MCU", function="controller")],
        power_nets=["GND", "+3V3"],
        inter_sheet_nets=[
            InterSheetNet(
                name="SDA",
                endpoints=[
                    SheetPin(sheet="MCU", direction="bidirectional"),
                    SheetPin(sheet="MCU", direction="bidirectional"),
                ],
            )
        ],
        assumptions=[text],
    )


def test_resolve_hit() -> None:
    """A real core_defaults bundle referenced in assumptions resolves cleanly.

    The function scans assumptions text for core_defaults bundle names; a
    reference to a bundle whose symbol resolves (and whose LCSC, if any, is in
    the offline catalog) yields an empty error list.
    """
    arch = _architecture_with_assumption(
        f"We use the {_CORE_BUNDLE} sensor for pressure measurement."
    )
    errors = _unresolved_architecture_parts(arch, Path.cwd())
    assert errors == [], f"expected no errors, got: {errors}"


def test_resolve_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A core_defaults bundle whose symbol fails to resolve is reported.

    ``_unresolved_architecture_parts`` calls the module-level ``lookup_pins``
    binding inside ``cli_app``; patching that name to raise
    ``SymbolNotFoundError`` simulates an unresolvable symbol while keeping the
    bundle's manifest present in the parts library (so the symbol-resolution
    branch, not the missing-manifest branch, is exercised).
    """

    def _boom(lib_id: str, *args, **kwargs):
        raise SymbolNotFoundError(f"forced miss for {lib_id}")

    monkeypatch.setattr(cli_app, "lookup_pins", _boom)

    arch = _architecture_with_assumption(
        f"We use the {_CORE_BUNDLE} sensor for pressure measurement."
    )
    errors = _unresolved_architecture_parts(arch, Path.cwd())
    assert errors, "expected at least one error for an unresolvable symbol"
    assert any(_CORE_BUNDLE in e for e in errors), (
        f"expected error to mention bundle {_CORE_BUNDLE!r}, got: {errors}"
    )
