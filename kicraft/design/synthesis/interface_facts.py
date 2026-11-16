"""Provider-independent interface facts tied to recipe ownership and wiring.

Only facts proved by the selected reviewed recipe and the committed endpoints are
returned.  In particular, unrelated design prose is not a source of ratings.
"""
from __future__ import annotations

from kicraft.design.recipes.registry import get_recipe, expand_recipe


def reviewed_interface_facts(state, project_root=None) -> list[dict]:
    bom = getattr(state, "bom", None)
    arch = getattr(state, "architecture", None)
    if not bom or not arch:
        return []
    parts = {p.ref: p for p in bom.parts}
    selections = {(s.recipe, s.instance): s for s in arch.recipe_selections}
    actual = {}
    for connection in bom.connections:
        for ep in connection.endpoints:
            actual.setdefault((ep.ref, str(ep.pin)), set()).add(connection.net_name)
    facts = []
    owners = {(o.recipe, o.instance): o for o in bom.recipe_ownership}
    expansions = {}
    no_connects = {(pin.ref, str(pin.pin)) for pin in bom.no_connect_pins}

    def verified(owner, selection, part, definition):
        key = (selection.recipe, selection.instance)
        if key not in expansions:
            expansions[key] = expand_recipe(selection)
        canonical = expansions[key]
        candidates = [p for p in canonical.parts if p.recipe_role == part.recipe_role]
        if len(candidates) != 1 or part.ref not in owner.refs:
            return False
        expected_part = candidates[0]
        if any(getattr(part, field) != getattr(expected_part, field)
               for field in ("symbol", "footprint", "value", "mpn")):
            return False
        expected = [p for p in canonical.ownership.pins if p.ref == expected_part.ref]
        if not expected:
            return False
        owned = {(p.ref, str(p.pin)): p for p in owner.pins}
        for pin in expected:
            identity = (part.ref, str(pin.pin))
            record = owned.get(identity)
            nets = actual.get(identity, set())
            if (not record or record.owner != pin.owner or record.owner_id != pin.owner_id
                    or record.net != pin.net):
                return False
            if pin.net is None:
                if identity not in no_connects or nets:
                    return False
            elif nets != {pin.net}:
                return False
        return True

    def verified_roles(owner, selection, definition, roles):
        for role in roles:
            matches = [parts[r] for r in owner.refs if r in parts
                       and parts[r].recipe_role == role
                       and parts[r].recipe_id == selection.recipe
                       and parts[r].recipe_instance == selection.instance]
            if len(matches) != 1 or not verified(owner, selection, matches[0], definition):
                return False
        return True
    def audio_estimate_proven(owner, selection, definition):
        roles = ("amplifier", "speaker_header", "pwm_series", "pwm_shunt",
                 "pwm_filter", "input_coupling", "reference_coupling",
                 "bead_positive", "bead_negative", "output_filter_positive",
                 "output_filter_negative")
        if not verified_roles(owner, selection, definition, roles):
            return False
        pwm = selection.port_bindings.get("pwm")
        for source in arch.recipe_selections:
            source_owner = owners.get((source.recipe, source.instance))
            if not source_owner or not source.recipe.startswith("esp32-s3-wroom-1-"):
                continue
            source_parts = [parts[r] for r in source_owner.refs if r in parts
                            and parts[r].recipe_role == "mcu"]
            if len(source_parts) != 1:
                continue
            try:
                source_definition = get_recipe(source.recipe)
            except ValueError:
                continue
            if not verified(source_owner, source, source_parts[0], source_definition):
                continue
            if not any(a.net == pwm and a.capability in {"output", "pwm"}
                       and actual.get((source_parts[0].ref, a.pin)) == {pwm}
                       for a in source_owner.pin_allocations):
                continue
            rail = source.port_bindings.get("vdd")
            for regulator in arch.recipe_selections:
                reg_owner = owners.get((regulator.recipe, regulator.instance))
                if (regulator.recipe == "ap63203-3v3@1" and reg_owner
                        and regulator.port_bindings.get("output") == rail
                        and regulator.parameters.get("output_voltage", get_recipe(regulator.recipe).parameter_defaults.get("output_voltage")) == 3.3
                        and reg_owner.port_bindings == regulator.port_bindings
                        and verified_roles(reg_owner, regulator, get_recipe(regulator.recipe),
                                           ("converter", "inductor"))):
                    return True
        return False


    def pin_map(owner, ref, names):
        owned = {str(p.pin): p.net for p in owner.pins if p.ref == ref and p.net is not None}
        return [{"pin": pin, "text": text, "net": owned[pin]}
                for pin, text in names if pin in owned]

    for part in bom.parts:
        key = (part.recipe_id, part.recipe_instance)
        owner, selection = owners.get(key), selections.get(key)
        if not owner or not selection or owner.port_bindings != selection.port_bindings:
            continue
        try:
            definition = get_recipe(selection.recipe)
        except ValueError:
            continue
        if not verified(owner, selection, part, definition):
            continue
        recipe, role = key[0], part.recipe_role
        kind = None
        lines, names, notes, extra = [], [], [], {}
        if recipe == "ch224k-pd-trigger@2" and role == "connector":
            if not verified_roles(owner, selection, definition, ("controller", "config", "vdd_feed", "vbus_sense")):
                continue
            voltage = selection.parameters.get("output_voltage", definition.parameter_defaults.get("output_voltage"))
            if voltage not in definition.allowed_parameters.get("output_voltage", ()):
                continue
            kind, lines = "pd_power", [f"USB PD {voltage:g}V IN"]
            extra["voltage_v"] = voltage
        elif recipe == "usb-c-usb2-device@1" and role == "connector":
            from .autoplacer import usb_routing_contracts
            try:
                contracts = usb_routing_contracts(arch, bom)
            except ValueError:
                continue
            if not any(contract.id.startswith(f"{selection.instance}:") for contract in contracts):
                continue
            vbus = selection.port_bindings.get("vbus")
            power_ports = {o.port_bindings.get(port) for o in bom.recipe_ownership
                           if o.recipe != recipe for port in ("input", "vdd", "vdd_5v", "vbus")}
            if not vbus or vbus in power_ports:
                continue
            kind, lines = "usb_programming", ["USB PROGRAM", "BOARD POWER SEPARATE"]
            notes = ["Programming USB is data only; power the board separately."]
        elif recipe in {"hub75-sn74hct245-interface@2", "led-matrix-hub75-interface@1"} and role == "connector":
            kind, lines, names = "hub75", ["HUB75"], [("1", "1 R0")]
            notes = ["Panel power is external; connect the common signal ground."]
        elif recipe in {"ws2812-external-strip-interface@1", "ws2812-external-strip-header-interface@1"} and role == "connector":
            if not verified_roles(owner, selection, definition, ("level_shifter", "series_resistor")):
                continue
            kind, lines, names = "led_output", ["LED"], [("1", "+5V"), ("2", "DATA"), ("3", "GND")]
            supply = selection.port_bindings.get("vdd_5v")
            for s in arch.recipe_selections:
                reg_owner = owners.get((s.recipe, s.instance))
                if (s.recipe == "ap63205-5v@2" and s.port_bindings.get("output") == supply
                        and s.parameters.get("output_voltage", get_recipe(s.recipe).parameter_defaults.get("output_voltage")) == 5 and reg_owner
                        and reg_owner.port_bindings == s.port_bindings
                        and verified_roles(reg_owner, s, get_recipe(s.recipe), ("converter", "inductor"))):
                    extra["voltage_v"] = 5
                    break
            if "voltage_v" not in extra:
                continue
            current = selection.parameters.get("external_load_current_a", definition.parameter_defaults.get("external_load_current_a"))
            if current not in definition.allowed_parameters.get("external_load_current_a", ()):
                continue
            extra["current_a"] = current
            lines.append(f"{current:g}A ASSUMED")
            notes = ["External LED current is a design allocation, not a measured capability or a verified copper rating."]
        elif recipe == "pam8302a-pwm-speaker@1" and role == "speaker_header":
            if not verified_roles(owner, selection, definition, ("amplifier", "bead_positive", "bead_negative")):
                continue
            impedance = selection.parameters.get("speaker_impedance_ohm", definition.parameter_defaults.get("speaker_impedance_ohm"))
            if impedance not in definition.allowed_parameters.get("speaker_impedance_ohm", ()):
                continue
            kind, lines, names = "speaker", [f"SPKR {impedance:g}ohm FLOAT"], [("1", "SPK+"), ("2", "SPK-")]
            extra["impedance_ohm"] = impedance
            notes = ["Bridge-tied speaker output: neither terminal may be grounded.", "Delivered audio power is not measured."]
            estimate = selection.parameters.get("estimated_low_frequency_output_w", definition.parameter_defaults.get("estimated_low_frequency_output_w"))
            if (estimate in definition.allowed_parameters.get("estimated_low_frequency_output_w", ())
                    and audio_estimate_proven(owner, selection, definition)):
                extra["estimated_low_frequency_output_w"] = estimate
                notes.append(f"Estimated low-frequency output {estimate:g} W into {impedance:g} ohm; unmeasured.")
        elif role in {"boot_button", "reset_button"} and recipe.startswith(("esp32-s3-wroom-1-", "esp32-s3-mini-1-")):
            kind = "boot" if role == "boot_button" else "reset"
            lines = [kind.upper()]
        if not kind:
            continue
        pins = pin_map(owner, part.ref, names)
        if len(pins) != len(names):
            continue
        label_id = f"required:{part.ref}:{kind}"
        ids = [label_id] + [f"{label_id}:pins:{p['pin']}" for p in pins]
        facts.append({"ref": part.ref, "kind": kind, "title": lines[0], "lines": lines,
                      "pins": pins, "required_label_ids": ids, "notes": notes,
                      "source": {"recipe": recipe, "instance": selection.instance,
                                 "requirement_ids": list(owner.requirement_ids)}, **extra})
    return facts
