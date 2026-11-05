"""Legacy recipe identities whose derived artifacts require regeneration."""

_LEGACY_RECIPE_REPLACEMENTS = {
    "rp2040-minimal@1": "rp2040-minimal@3",
    "rp2040-minimal@2": "rp2040-minimal@3",
    "drv8833-dual-motor@1": "drv8833-dual-motor@5",
    "drv8833-dual-motor@2": "drv8833-dual-motor@5",
    "drv8833-dual-motor@3": "drv8833-dual-motor@5",
    "drv8833-dual-motor@4": "drv8833-dual-motor@5",
    "esp32-c3-mini-1-minimal@1": "esp32-c3-mini-1-minimal@2",
    "esp32-c3-mini-1-h4x-minimal@1": "esp32-c3-mini-1-h4x-minimal@2",
    "stm32g030k6t6-minimal@1": "stm32g030k6t6-minimal@2",
    "tlv62569-3v3@1": "tlv62569-3v3@3",
    "tlv62569-3v3@2": "tlv62569-3v3@3",
    "tps54331-adjustable@1": "tps54331-adjustable@2",
    "tps54331-12v-1a@1": "tps54331-12v-1a@2",
}


def legacy_recipe_replacement(value: object) -> str | None:
    """Return a replacement identity, without interpreting the old graph."""
    return _LEGACY_RECIPE_REPLACEMENTS.get(value) if isinstance(value, str) else None


def is_legacy_recipe_id(value: object) -> bool:
    """Identify old ownership IDs in raw state without loading design models."""
    return legacy_recipe_replacement(value) is not None
