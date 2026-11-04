"""The reference data a design stage is shown, built from the committed state.

Two callers need these blocks and must never disagree: the stage runtime hands them to the
provider, and `kicraft stage-prep` prints them so a reviewer can see what the stage was offered.
Both read this module, which is why the builders live in the design layer rather than beside the
provider call: a checkpoint that under-reports the reference data asks the owner to judge a draft
against inputs it never had.

The recipe reference may read the local assembly catalog and vendored manifest
for a snapshot of a recipe's exact principal part. It never accesses the network.
"""

from __future__ import annotations

import json
from pathlib import Path
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
    features = part.physical_features
    if "pin-header" in features:
        detail.extend(gender for gender in ("male", "female") if gender in features)
        pitch = (part.operating_limits or {}).get("pitch_mm")
        if pitch is not None:
            detail.append(f"{pitch:g} mm pitch")
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
        quantity_subject_binds,
    )
    from kicraft.parts_library import jlcparts
    from kicraft.design.recipes.registry import registered_recipes
    from kicraft.design.cli_app import _bom_stock_floor

    demanded = sorted(
        {
            str(row.get("component_class") or "").strip().casefold()
            for row in (intent.get("obligations") or [])
            if isinstance(row, dict) and row.get("kind") == "physical"
        }
        - {""}
    )
    # Conversion facts already require this hardware; a second physical-part
    # obligation is not necessary to expose its qualified constructors.
    if any(
        isinstance(row, dict) and row.get("kind") == "conversion"
        and all(re.search(r"\d+(?:\.\d+)?\s*m?V\b", str(row.get(key) or ""), re.I)
                for key in ("input_kind", "output_kind"))
        for row in intent.get("obligations") or []
    ):
        demanded = sorted({*demanded, "voltage-regulator"})
    lines: list[str] = []
    terminal_demanded = False
    for component_class in demanded:
        options: dict[str, str] = {}
        wanted_widths = {
            row.get("minimum") for row in intent.get("obligations") or []
            if isinstance(row, dict) and row.get("kind") == "quantity"
            and "pin-header" in canonical_physical_features(component_class)
            and quantity_subject_binds(str(row.get("subject") or ""), component_class)
        }
        if "pin-header" in canonical_physical_features(component_class):
            wanted_widths.update(
                row.get("value") for row in intent.get("obligations") or []
                if isinstance(row, dict) and row.get("kind") == "quantitative"
                and row.get("relation") == "equal"
                and str(row.get("unit") or "").casefold() in {"pins", "contacts", "positions"}
                and "header" in str(row.get("quantity") or "").casefold()
            )
        width_matches: set[str] = set()
        terminal_demanded |= "screw-terminal" in canonical_physical_features(component_class)
        for feature in sorted(canonical_physical_features(component_class)):
            for part in reviewed_parts_for_feature(feature):
                if part.identity.startswith("kicad-"):
                    continue  # Reserved library-pattern namespace, not manufacturer MPNs.
                if (part.family.endswith("header") or part.family == "screw-terminal"
                        or "voltage-regulator" in part.physical_features):
                    # Choices must clear the same assembly-stock policy as the
                    # sourcing gate, not merely exist with one unit available.
                    row = jlcparts.lookup(part.lcsc) if part.lcsc else None
                    if (not row or (row.get("model") or "").casefold() != part.identity.casefold()
                            or (row.get("stock") or 0) < max(1, _bom_stock_floor())):
                        continue
                limits = dict(part.operating_limits or {})
                ratings = [
                    f"{key.removesuffix('_v').removesuffix('_a').replace('_', ' ')}="
                    f"{limits[key]}"
                    for key in ("output_voltage_v", "output_current_a", "input_voltage_min_v",
                                "input_voltage_max_v", "dissipated_power_limit_w")
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
                if len(part.contacts) in wanted_widths:
                    width_matches.add(part.identity)
        if options:
            choices = sorted(options, key=lambda identity: (identity not in width_matches, identity))
            lines.append(
                f"  {component_class}: " + "; ".join(options[identity] for identity in choices[:6])
            )
    external = [
        row for row in intent.get("obligations") or []
        if isinstance(row, dict) and row.get("kind") == "external_interface"
    ]
    for obligation in external:
        protocol = str(obligation.get("protocol") or "").casefold()
        for registered in registered_recipes():
            definition = registered.definition
            if definition.maturity != "production" or protocol not in definition.external_interface_protocols:
                continue
            ports = ", ".join(port.name for port in definition.ports if port.required)
            lines.append(
                f"  external interface {protocol}: assign obligation "
                f"{obligation.get('original_obligation_id')} to family {definition.family} "
                f"({definition.recipe}); wire required ports {ports}. Preserve the brief's "
                "current and power-source assumptions. An external load is not an onboard "
                "physical component; satisfy any separate onboard physical obligation only "
                "with a construction that actually emits that component."
            )
    if any(
        isinstance(row, dict) and row.get("kind") == "adjustability"
        and str(row.get("parameter") or "").casefold() == "enable"
        and str(row.get("mechanism") or "").casefold() == "jumper"
        for row in intent.get("obligations") or []
    ):
        lines.append(
            "  enable-jumper construction: use the published switch-input circuit, "
            "pull_policy=external, active_level=high; its signal controls exactly one "
            "recipe-published enable port, and its supply must be a sourced rail within "
            "that recipe's reviewed EN limit. A bare pin-header does not supply this "
            "bias/control contract. Preserve the physical jumper and adjustability IDs; "
            "the typed construction includes the separately procured removable shunt."
        )
    if terminal_demanded:
        lines.append(
            "  screw-terminal construction: leave exact_part null unless the brief requires "
            "an exact ordering code. A generic contact count can be realized by multiple "
            "reviewed housings; bind every intended net to its own contiguous pin, and "
            "undeclared positions remain unconnected. If the brief requires a single "
            "housing, preserve that requirement and choose a suitable exact part; do not "
            "substitute a bank. Counts of contacts are not counts of housings."
        )
    if not lines:
        return None
    return "\n".join(
        [
            "REVIEWED MANUFACTURER PARTS FOR THE DEMANDED CLASSES: these identities "
            "have reviewed symbol/footprint pairs; listed headers, terminals and regulators "
            "also have an exact matching offline catalog row above the sourcing stock floor. "
            "Leave exact_part null for "
            "an unspecified generic component; retain any exact part required by the brief.",
            *lines,
        ]
    )


def _principal_part_assembly_snapshot(definition) -> dict | None:
    """Report only the vendored exact part's pinned C#; this is not a whole-BOM verdict."""
    from kicraft.parts_library import jlcparts

    if not definition.exact_part or not jlcparts.available():
        return None
    for part in definition.parts:
        if not part.mpn or part.mpn.casefold() != definition.exact_part.casefold():
            continue
        bundle, separator, _ = part.symbol.partition(":")
        if not separator or not bundle or "/" in bundle or "\\" in bundle:
            continue
        path = Path(__file__).resolve().parents[1] / "parts_library" / bundle / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(manifest.get("mpn") or "").casefold() != part.mpn.casefold():
            continue
        lcsc = str((manifest.get("sourcing") or {}).get("lcsc") or "").strip()
        if not lcsc:
            continue
        row = jlcparts.lookup(lcsc)
        return {
            "mpn": part.mpn,
            "lcsc": lcsc,
            "assembly_catalog": "present" if row is not None else "absent",
            "assembly_stock": row.get("stock") if row is not None else None,
        }
    return None




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
    for summary in summaries:
        snapshot = _principal_part_assembly_snapshot(get_recipe(summary["recipe"]))
        if snapshot is not None:
            summary["principal_part_assembly_snapshot"] = snapshot
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
