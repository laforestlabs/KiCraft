"""The reference data a design stage is shown, built from the committed state.

Two callers need these blocks and must never disagree: the stage runtime hands them to the
provider, and `kicraft stage-prep` prints them so a reviewer can see what the stage was offered.
Both read this module, which is why the builders live in the design layer rather than beside the
provider call: a checkpoint that under-reports the reference data asks the owner to judge a draft
against inputs it never had.

Every builder takes the committed intent (or an empty dict, during tests) and returns text or
plain rows. Nothing here reads the database or the network.
"""

from __future__ import annotations

import re

from kicraft.design.lowering import lowerer_summaries

def standard_form_factor_block(intent: dict) -> str | None:
    """The pin/net contract of the design's standard mechanical template.

    A brief that resolved to a standard form factor (an Arduino Uno shield, say) owns that
    standard's *fixed* connectors, and the architecture stage must emit exactly those
    connectors with the template's own pin map. The template lives in
    ``kicraft.form_factors``, so without this block the stage is asked to reproduce role
    names and a 32-pin map it was never shown: the proto-shield brief answered with ONE
    composite requirement whose stacking role was the template key, and three correction
    rounds could not recover the four roles or their maps. Same shape as the recipe and
    lowerer summaries: deterministic reference data, rendered from the committed intent.

    Returns None when no validated standard is in play, so every other design's prompt is
    unchanged.
    """
    from kicraft.form_factors import get_template

    form_factor = intent.get("form_factor") or {}
    standard = form_factor.get("standard") if isinstance(form_factor, dict) else None
    template = get_template(standard)
    if template is None or not template.validated:
        return None
    lines = [
        f"STANDARD FORM FACTOR: {template.key} ({template.display_name}), "
        f"{template.board_width_mm} x {template.board_height_mm} mm, "
        f"{len(template.mounting_holes)} fixed mounting holes.",
        "This board's host interface is the standard's own FIXED CONNECTORS. Emit exactly "
        "one requirement per role below — never a composite, renamed, or arbitrary header — "
        "each with:",
        '  role: "connector" | family: "pin-header" | exact_part: null | '
        'parameters: {"rows": 1, "gender": "female"}',
        "  standard_stacking_role: the role name below",
        "  functional_blocks: the committed functional block that owns the host interface",
        "  ties: leave empty to accept this template's pin map (state it only if a pin must "
        "tie to a net this map does not already name)",
    ]
    for connector in template.fixed_connectors:
        pins = ", ".join(
            f"pin{index}={net}" for index, net in enumerate(connector.net_by_pin, start=1)
        )
        lines.append(
            f"  {connector.role} ({len(connector.net_by_pin)} pins): {pins}"
        )
    lines.append(
        "The template owns these pin positions and nets; do not re-plan them, and do not add "
        "headers for them after wiring."
    )
    return "\n".join(lines)




def reviewed_option_detail(part, ratings: list[str]) -> list[str]:
    """What a stage needs to pick between reviewed carriers of one class, count first.

    A class can be carried by several reviewed parts that differ in exactly one property, and the
    model is choosing blind without it. The terminal class is carried by a 2-, a 3- and a
    4-position block that differ in nothing else, and the architecture stage that named the
    4-position part for a 2-contact requirement was refused five times out of six on the frozen
    live inputs *after* the refusal text had been taught to name the mismatch -- the choice has to
    be informed where it is made, which is what the ratings in this block are for too.
    """
    detail: list[str] = []
    if part.contacts:
        count = len(part.contacts)
        detail.append(f"{count} contact" + ("" if count == 1 else "s"))
    detail.extend(ratings)
    return detail




def reviewed_class_options_block(intent: dict) -> str | None:
    """The reviewed parts that can satisfy each physical class this design demands.

    A demanded class is realized only by a reviewed part (``E_PHYSICAL_REALIZATION``), and
    the stage that picks the part never sees which parts those are: one proto-shield run
    answered the demanded ``voltage-regulator`` with the familiar but unreviewed
    ``AMS1117-3.3``, which resolves as a bundle and therefore passes every earlier gate,
    then refuses at the BOM with "found 0 ... exact MPN/symbol/footprint evidence" and no
    correction round left. Listing the reviewed options with their ratings lets the stage
    choose a realizable part where it is still choosing. Classes with no reviewed coverage
    are omitted: their honest route is the exact part the user named.
    """
    from kicraft.design.part_identity import (
        canonical_physical_features,
        reviewed_parts_for_feature,
    )

    demanded = sorted(
        {
            str(row.get("component_class") or "").strip().casefold()
            for row in (intent.get("obligations") or [])
            if isinstance(row, dict) and row.get("kind") == "physical"
        }
        - {""}
    )
    lines: list[str] = []
    for component_class in demanded:
        options: dict[str, str] = {}
        for feature in sorted(canonical_physical_features(component_class)):
            for part in reviewed_parts_for_feature(feature):
                limits = dict(part.operating_limits or {})
                ratings = [
                    f"{key.removesuffix('_v').removesuffix('_a').replace('_', ' ')}="
                    f"{limits[key]}"
                    for key in ("output_voltage_v", "output_current_a", "input_voltage_max_v")
                    if limits.get(key) is not None
                ]
                options.setdefault(
                    part.identity,
                    f"{part.identity}"
                    + (
                        f" ({part.family}: {', '.join(detail)})"
                        if (detail := reviewed_option_detail(part, ratings))
                        else f" ({part.family})"
                    ),
                )
        if options:
            lines.append(
                f"  {component_class}: " + "; ".join(sorted(options.values())[:6])
            )
    if not lines:
        return None
    return "\n".join(
        [
            "REVIEWED PARTS FOR THE DEMANDED CLASSES: a demanded physical class is realized "
            "only by one of these reviewed identities (with its own symbol/footprint pair), "
            "so prefer them, and name `exact_part` only from this list. Where a part's contact "
            "count is given, it must match the contacts the requirement declares:",
            *lines,
        ]
    )




def architecture_recipe_summaries(intent: dict) -> list[dict]:
    """Do not offer unrelated MCU alternatives to an explicitly named design.

    Identity comes from the reviewed ``part_identity`` relations as well as the
    name shape: a brief naming an order code of a registered family (the
    ESP32-S3-WROOM-1 ``-N16R8`` against the registered ``-N8R8``) must be offered
    that family's recipe, or the stage is asked to bind a circuit it was never
    shown and blocks on a variant it cannot discover. When the named parts
    identify no MCU recipe, every MCU recipe stays on offer rather than starving
    the stage.
    """
    from kicraft.design.part_identity import matches_part_identity
    from kicraft.design.recipes import get_recipe, recipe_summaries

    summaries = recipe_summaries()
    raw_named = [str(value) for value in intent.get("named_parts") or []]
    named = {
        re.sub(r"[^a-z0-9]", "", str(value).lower()) for value in intent.get("named_parts") or []
    }
    named = {value for value in named if len(value) >= 5}
    if not named:
        return summaries
    selected = set()
    for summary in summaries:
        if "mcu" not in summary["required_sheet_roles"]:
            continue
        definition = get_recipe(summary["recipe"])
        selectors = {
            re.sub(r"[^a-z0-9]", "", str(value).lower())
            for value in (definition.exact_part, definition.family, *definition.identity_aliases)
            if value
        }
        if any(
            selector.startswith(part) or part.startswith(selector)
            for selector in selectors
            if len(selector) >= 5
            for part in named
        ) or any(
            # Reviewed order-code membership preserves punctuation, so it gets
            # the raw name, not the separator-stripped token above.
            definition.exact_part and matches_part_identity(part, definition.exact_part)
            for part in raw_named
        ):
            selected.add(summary["recipe"])
    if not selected:
        return summaries
    return [
        summary
        for summary in summaries
        if "mcu" not in summary["required_sheet_roles"] or summary["recipe"] in selected
    ]


def architecture_reference_extras(intent: dict) -> dict:
    """The architecture stage's reference blocks, in the order the prompt shows them.

    The admin-curated default parts are deliberately absent: they come from the store, not from
    the state, so the runtime adds that one block itself.
    """
    extras: dict = {}
    extras["circuit_recipes"] = architecture_recipe_summaries(intent)
    extras["circuit_lowerers"] = lowerer_summaries()
    standard = standard_form_factor_block(intent)
    if standard:
        extras["standard_form_factor"] = standard
    classes = reviewed_class_options_block(intent)
    if classes:
        extras["reviewed_class_options"] = classes
    return extras
