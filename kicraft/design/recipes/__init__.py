"""Versioned deterministic circuit recipes."""

from .registry import (
    expand_recipe,
    expand_selections,
    get_recipe,
    locked_no_connect_pins,
    locked_pin_assignments,
    recipe_summaries,
    register_recipe,
)
from .rp2040_minimal import RP2040_MINIMAL

register_recipe(RP2040_MINIMAL)

__all__ = [
    "expand_recipe",
    "expand_selections",
    "get_recipe",
    "locked_no_connect_pins",
    "locked_pin_assignments",
    "recipe_summaries",
]
