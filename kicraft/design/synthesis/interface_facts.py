"""Provider-independent interface facts tied to deterministic ownership and wiring.

Only facts proved by the selected recipes/lowerers and the committed endpoints
are returned. Unrelated design prose is not a source of ratings.
"""
from __future__ import annotations

import re

from kicraft.design.lowering import requires_power_output_header, verified_lowering
from kicraft.design.recipe_versions import RP2040_RECIPE_IDS
from kicraft.design.recipes.registry import expand_recipe, get_recipe
from kicraft.design.recipes.wave_b_power import TPS54331_3V3_SHUNT_CONTROLS


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
        if part.ref not in owner.refs:
            return False
        owned = {(p.ref, str(p.pin)): p for p in owner.pins}
        for expected_part in canonical.parts:
            if expected_part.recipe_role != part.recipe_role or any(
                getattr(part, field) != getattr(expected_part, field)
                for field in ("symbol", "footprint", "value", "mpn")
            ):
                continue
            expected = [p for p in canonical.ownership.pins if p.ref == expected_part.ref]
            if expected and all(
                (record := owned.get(identity := (part.ref, str(pin.pin)))) is not None
                and (record.owner, record.owner_id, record.net) == (pin.owner, pin.owner_id, pin.net)
                and actual.get(identity, set()) == (set() if pin.net is None else {pin.net})
                and (identity in no_connects) == (pin.net is None)
                for pin in expected
            ):
                return True
        return False

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

    def add_fact(ref, kind, lines, pins, source, *, notes=(), **extra):
        label_id = f"required:{ref}:{kind}"
        facts.append({
            "ref": ref, "kind": kind, "title": lines[0], "lines": lines,
            "pins": pins, "notes": list(notes), "source": source,
            "required_label_ids": [label_id, *(f"{label_id}:pins:{p['pin']}" for p in pins)],
            **extra,
        })

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
        elif recipe == "usb-c-5v-sink@1" and role == "connector":
            returns = selection.port_bindings.get("gnd")
            cc = {net for pin in ("A5", "B5") for net in actual.get((part.ref, pin), ())}
            termination = {
                net for ref in owner.refs if (rd := parts.get(ref)) is not None
                and rd.recipe_role == "cc_pulldown" and rd.recipe_id == recipe
                and rd.recipe_instance == selection.instance
                and verified(owner, selection, rd, definition)
                and actual.get((ref, "2")) == {returns}
                for net in actual.get((ref, "1"), ())
            }
            rail = selection.port_bindings.get("vbus")
            if len(cc) != 2 or termination != cc or arch.rail_voltages.get(rail) != 5.0:
                continue
            kind, lines = "usb_power", ["USB 5V IN"]
            extra["voltage_v"] = 5.0
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
        elif recipe in {"ch32v003f4p6-minimal@1", "ch32v003f4u6-minimal@1"}:
            if not verified_roles(owner, selection, definition,
                                  ("mcu", "program_header", "reset_button", "reset_pullup")):
                continue
            if role == "program_header":
                kind, lines = "programming", ["WCH-LINK"]
                names = [("1", "VDD"), ("2", "GND"), ("3", "SWIO"), ("4", "NRST")]
            elif role == "reset_button":
                kind, lines = "reset", ["RESET"]
        elif recipe in RP2040_RECIPE_IDS:
            if role == "bootsel":
                if not verified_roles(owner, selection, definition, ("mcu", "flash", "bootsel_resistor")):
                    continue
                kind, lines = "boot", ["BOOTSEL"]
            elif role in {"reset_button", "swd_header"}:
                if not verified_roles(owner, selection, definition, ("mcu", "reset_button", "reset_pullup")):
                    continue
                if role == "reset_button":
                    kind, lines = "reset", ["RESET"]
                elif selection.parameters.get("programming") == "swd" and (
                    selection.parameters.get("native_usb") is False
                ):
                    kind, lines = "programming", ["SWD"]
                    names = [("1", "VTREF"), ("2", "GND"), ("3", "SWDIO"), ("4", "SWCLK"), ("5", "RUN")]
                    notes = ["Contact 6 is intentionally not connected."]
        elif recipe == TPS54331_3V3_SHUNT_CONTROLS.recipe and role in {"reset_button", "enable_jumper"}:
            from .validation import verified_regulator_shunt_controls

            reset = role == "reset_button"
            matches = [req for req in arch.requirements if req.id in owner.requirement_ids
                       and verified_regulator_shunt_controls(req, bom, reset=reset)]
            if len(matches) != 1:
                continue
            kind, lines = ("reset", ["RESET"]) if reset else ("enable", ["FIT TO DISABLE"])
        elif role == "enable_jumper" and recipe.startswith(("tps54331-", "tps61088-")):
            target = "output_limiter" if recipe.startswith("tps61088-") else "converter"
            if not verified_roles(owner, selection, definition, (target,)):
                continue
            kind, lines = "enable", ["FIT TO DISABLE"]
        if not kind:
            continue
        pins = pin_map(owner, part.ref, names)
        if len(pins) != len(names):
            continue
        add_fact(part.ref, kind, lines, pins,
                 {"recipe": recipe, "instance": selection.instance,
                  "requirement_ids": list(owner.requirement_ids)}, notes=notes, **extra)

    def reviewed_gpio_contacts():
        from .electrical_review import _pin_names

        result: dict[str, set[tuple[str, str, str]]] = {}
        for selection in arch.recipe_selections:
            owner = owners.get((selection.recipe, selection.instance))
            if (
                owner is None or owner.port_bindings != selection.port_bindings
                or owner.pin_allocations != selection.pin_allocations
                or not (ground := selection.port_bindings.get("gnd"))
                or not selection.pin_allocations
            ):
                continue
            try:
                definition = get_recipe(selection.recipe)
            except ValueError:
                continue
            if not verified_roles(owner, selection, definition, ("mcu",)):
                continue
            mcu = next(parts[ref] for ref in owner.refs if ref in parts
                       and parts[ref].recipe_role == "mcu"
                       and parts[ref].recipe_id == selection.recipe
                       and parts[ref].recipe_instance == selection.instance)
            pin_names = _pin_names(mcu.symbol, project_root)
            for allocation in selection.pin_allocations:
                if (
                    allocation.capability not in {"gpio", "input", "output"}
                    or actual.get((mcu.ref, allocation.pin)) != {allocation.net}
                ):
                    continue
                label = pin_names.get(allocation.pin, "")
                # Only a simple native GPIO designator is legible as a pin label.
                if not re.fullmatch(r"(?:GPIO\d{1,3}|P[A-Z]\d{1,2})", label):
                    label = ""
                result.setdefault(allocation.net, set()).add(
                    (f"{mcu.ref}.{allocation.pin}", label, ground)
                )
        return result

    gpio_contacts = None

    connector_titles = {
        "pin-header@1": ("header", "SIGNAL HEADER"),
        "reviewed-jst-xh-connector@1": ("jst", "JST XH"),
        "screw-terminal@1": ("terminal", "TERMINAL"),
    }
    for requirement in arch.requirements:
        proof = verified_lowering(requirement, bom)
        if proof is None:
            continue
        artifact, refs = proof
        source = {"lowerer": artifact.lowerer_id, "instance": requirement.id,
                  "requirement_ids": [requirement.id]}
        for (role, index), ref in refs.items():
            owned = {pin.pin: pin.net for pin in artifact.pins
                     if (pin.role, pin.index) == (role, index)}
            names = []
            if requires_power_output_header(requirement):
                from .validation import verified_power_output_header

                if not verified_power_output_header(arch, bom, requirement):
                    continue
                kind, title = "power_output", "5V OUT"
                names = [(pin, "5V" if arch.rail_voltages[net] == 5.0 else "GND")
                         for pin, net in owned.items()]
            elif artifact.lowerer_id in connector_titles and role == "connector" and owned:
                kind, title = connector_titles[artifact.lowerer_id]
                if requirement.role == "power_input":
                    kind, title = "power_input", "POWER IN"
                if kind == "jst":
                    if gpio_contacts is None:
                        gpio_contacts = reviewed_gpio_contacts()
                    grounds = {ground for net in owned.values()
                               for _, _, ground in gpio_contacts.get(net, ())}
                    if grounds and (len(grounds) != 1 or not grounds <= set(owned.values())):
                        continue  # a legacy GPIO carrier without its return is not an interface
                for pin, net in owned.items():
                    if net == "GND":
                        names.append((pin, "GND"))
                    elif net in arch.rail_voltages and arch.rail_voltages[net] > 0:
                        names.append((pin, f"{arch.rail_voltages[net]:g}V"))
                    elif (kind == "jst" and gpio_contacts is not None
                          and len(rows := gpio_contacts.get(net, ())) == 1
                          and (label := next(iter(rows))[1])):
                        names.append((pin, label))
                    elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]{0,7}", net):
                        names.append((pin, net.upper()))
                    elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", net):
                        # Keep the exact connection name in the off-board pin
                        # facts; the PCB can identify an otherwise long net by
                        # its unambiguous physical contact number.
                        names.append((pin, f"PIN {pin}"))
            elif artifact.lowerer_id == "qwiic-i2c-connector@1" and role == "connector":
                kind, title = "qwiic", "I2C / QWIIC"
                if arch.rail_voltages.get(requirement.ports.get("vcc")) != 3.3:
                    continue
                names = [("1", "GND"), ("2", "3V3"), ("3", "SDA"), ("4", "SCL")]
            elif artifact.lowerer_id == "switch-input@1" and role == "jumper":
                kind, title = "enable", "FIT TO ENABLE"
            elif artifact.lowerer_id == "switch-input@1" and role == "switch":
                kind, title = "button", "BUTTON"
            else:
                continue
            if (role == "connector" and not names) or any(pin not in owned for pin, _ in names):
                continue
            add_fact(ref, kind, [title],
                     [{"pin": pin, "text": name, "net": owned[pin]} for pin, name in names], source)
    return facts
