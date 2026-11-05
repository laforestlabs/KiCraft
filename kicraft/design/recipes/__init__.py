"""Versioned deterministic circuit recipes."""

from .registry import (
    complete_unused_published_ports,
    expand_recipe,
    expand_selections,
    get_recipe,
    locked_no_connect_pins,
    locked_pin_assignments,
    migrate_recipe_selection,
    migrate_recipe_selections,
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
from .rp2040_minimal_v2 import RP2040_MINIMAL_V3, expand_rp2040_minimal
from .wave_a_mcus import WAVE_A_MCU_RECIPES, expand_wave_a_mcu
from .wave_b_power import (
    USB_C_USB2_DEVICE,
    WAVE_B_POWER_RECIPES,
    expand_lm2577t_adj_12v_boost,
    expand_tps54331_buck,
    expand_tps61088_boost,
    expand_usb_c_usb2_device,
)
from .wave_c_interfaces import (
    CH340C_USB_UART,
    WAVE_C_INTERFACE_RECIPES,
    WS2812_OUTPUT,
    expand_ch340c_usb_uart,
    expand_ws2812_output,
)
from .wave_d_display import WAVE_D_DISPLAY_RECIPES, expand_max7219_four_digit
from .wave_e_usb_bridge import WAVE_E_USB_BRIDGE_RECIPES, expand_cp2102n_usb_uart
from .wave_f_sensing import WAVE_F_SENSING_RECIPES, expand_ah49entr_g1_analog_hall
from .actuator_interfaces import WAVE_F_ACTUATOR_RECIPES
from .data_logging_storage import DATA_LOGGING_STORAGE_RECIPES
from .dual_source_battery_power import WAVE_G_DUAL_SOURCE_POWER_RECIPES, expand_dual_source_battery_power



register_recipe(RP2040_MINIMAL_V3, expand_rp2040_minimal)
register_recipe(ESP32_S3_MINI_1_MINIMAL, expand_esp32_s3_mini_1)
for definition in WAVE_A_MCU_RECIPES:
    register_recipe(definition, expand_wave_a_mcu)
_POWER_EXPANDERS = {
    USB_C_USB2_DEVICE.recipe: expand_usb_c_usb2_device,
    "tps54331-adjustable@2": expand_tps54331_buck,
    "tps54331-12v-1a@2": expand_tps54331_buck,
    "lm2577t-adj-12v-boost@1": expand_lm2577t_adj_12v_boost,
    "lm2577t-adj-2s-12v-boost@1": expand_lm2577t_adj_12v_boost,
    "tps61088-2s-12v-1a@1": expand_tps61088_boost,
    "tps61088-host5-12v-1a@1": expand_tps61088_boost,
}
for definition in WAVE_B_POWER_RECIPES:
    register_recipe(
        definition,
        _POWER_EXPANDERS.get(definition.recipe),
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
for definition in WAVE_D_DISPLAY_RECIPES:
    register_recipe(
        definition,
        expand_max7219_four_digit
        if definition.recipe == "max7219-four-digit-seven-segment@1"
        else None,
    )
for definition in WAVE_E_USB_BRIDGE_RECIPES:
    register_recipe(definition, expand_cp2102n_usb_uart)
for definition in WAVE_F_SENSING_RECIPES:
    register_recipe(definition, expand_ah49entr_g1_analog_hall)
for definition in WAVE_F_ACTUATOR_RECIPES:
    register_recipe(definition)
for definition in DATA_LOGGING_STORAGE_RECIPES:
    register_recipe(definition)
for definition in WAVE_G_DUAL_SOURCE_POWER_RECIPES:
    register_recipe(definition, expand_dual_source_battery_power)



__all__ = [
    "RecipeResolutionError",
    "ResolutionResult",
    "apply_architecture_recipe_resolution",
    "complete_unused_published_ports",
    "expand_recipe",
    "expand_selections",
    "get_recipe",
    "locked_no_connect_pins",
    "locked_pin_assignments",
    "migrate_recipe_selection",
    "migrate_recipe_selections",
    "protected_identities",
    "protected_identity_matches",
    "recipe_summaries",
    "registered_recipes",
    "resolve_architecture_recipes",
]
