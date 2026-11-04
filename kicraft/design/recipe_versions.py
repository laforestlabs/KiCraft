"""Legacy recipe identities whose derived artifacts require regeneration."""

_LEGACY_RECIPE_REPLACEMENTS = {
    "rp2040-minimal@1": "rp2040-minimal@3",
    "rp2040-minimal@2": "rp2040-minimal@3",
    "drv8833-dual-motor@1": "drv8833-dual-motor@2",
    "tps54331-adjustable@1": "tps54331-adjustable@2",
    "tps54331-12v-1a@1": "tps54331-12v-1a@2",
}


def legacy_recipe_replacement(value: object) -> str | None:
    """Return a replacement identity, without interpreting the old graph."""
    return _LEGACY_RECIPE_REPLACEMENTS.get(value) if isinstance(value, str) else None


def is_legacy_recipe_id(value: object) -> bool:
    """Identify old ownership IDs in raw state without loading design models."""
    return legacy_recipe_replacement(value) is not None
