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
from .wave_a_mcus import WAVE_A_MCU_RECIPES, expand_wave_a_mcu
from .wave_b_power import (
    USB_C_USB2_DEVICE,
    WAVE_B_POWER_RECIPES,
    expand_usb_c_usb2_device,
)
from .wave_c_interfaces import (
    CH340C_USB_UART,
    WAVE_C_INTERFACE_RECIPES,
    WS2812_OUTPUT,
    expand_ch340c_usb_uart,
    expand_ws2812_output,
)

register_recipe(RP2040_MINIMAL)
register_recipe(RP2040_MINIMAL_V2)
register_recipe(ESP32_S3_MINI_1_MINIMAL, expand_esp32_s3_mini_1)
for definition in WAVE_A_MCU_RECIPES:
    register_recipe(definition, expand_wave_a_mcu)
for definition in WAVE_B_POWER_RECIPES:
    register_recipe(
        definition,
        expand_usb_c_usb2_device
        if definition.recipe == USB_C_USB2_DEVICE.recipe
        else None,
    )
for definition in WAVE_C_INTERFACE_RECIPES:
    register_recipe(
        definition,
        expand_ch340c_usb_uart
        if definition.recipe == CH340C_USB_UART.recipe
        else expand_ws2812_output
        if definition.recipe == WS2812_OUTPUT.recipe
        else None,
    )

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
