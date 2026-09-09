"""Versioned deterministic circuit recipes."""

from .registry import (
    expand_recipe,
    expand_selections,
    get_recipe,
    locked_no_connect_pins,
    locked_pin_assignments,
    protected_identities,
    protected_identity_matches,
    recipe_summaries,
    register_recipe,
    registered_recipes,
)
from .resolver import (
    RecipeResolutionError,
    ResolutionResult,
    apply_architecture_recipe_resolution,
    resolve_architecture_recipes,
)
from .esp32_s3_mini_1_minimal import (
    ESP32_S3_MINI_1_MINIMAL,
    expand_esp32_s3_mini_1,
)
from .rp2040_minimal import RP2040_MINIMAL
from .rp2040_minimal_v2 import RP2040_MINIMAL_V2

register_recipe(RP2040_MINIMAL)
register_recipe(RP2040_MINIMAL_V2)
register_recipe(ESP32_S3_MINI_1_MINIMAL, expand_esp32_s3_mini_1)

__all__ = [
    "RecipeResolutionError",
    "ResolutionResult",
    "apply_architecture_recipe_resolution",
    "expand_recipe",
    "expand_selections",
    "get_recipe",
    "locked_no_connect_pins",
    "locked_pin_assignments",
    "protected_identities",
    "protected_identity_matches",
    "recipe_summaries",
    "registered_recipes",
    "resolve_architecture_recipes",
]
